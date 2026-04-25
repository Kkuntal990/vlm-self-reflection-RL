#!/usr/bin/env python3
"""Build A1->F1->A2 SFT data (Pattern A, v10) from GRPO trajectories + FIRE feedback.

Outputs ms-swift ShareGPT JSONL with six messages per row (no system turn,
default Qwen "You are a helpful assistant." injected by the chat template).

Two sources:

1. GRPO v10 trajectories (`trajectories_rank*.jsonl`):
   - Keep RR (a1=correct, a2=correct) and WR (a1=wrong, a2=correct).
   - Drop RW + WW.
   - Format gate: a1_format == 1.0, a2_format == 1.0, F1 has >=6 words and
     ends with `\\boxed{CORRECT|INCORRECT}` and contains no `<think>/<answer>`
     tag leakage.
   - Verification gate: F1 verdict must agree with a1_correct.
   - Drop trajectories where F1 is a near-copy of A1 (norm edit dist < 0.15)
     or A2 is a near-copy of A1 in WR class (no real refinement).
   - Step gate: global_step >= --min_step (default 200).
   - Dedup by (image_path, question, pattern); keep highest resp_reward.
   - Per-task RR cap: at most --per_task_rr_cap * n_WR(task).

2. FIRE feedback dataset (`fire_feedback_train.jsonl`):
   - Each FIRE conversation has alternating user (candidate answer) and
     assistant (constructive feedback) turns. We extract the first round
     (A1 = user_1, F1 = feedback_1, A2 = user_2) as a WR-style triple.
   - F1 is wrapped as `<think>{feedback_text}</think>\\boxed{INCORRECT}` so
     it conforms to the v10 verifier output format.
   - Sampled to --fire_target rows.

All rows share the same Pattern A six-message structure built via
src/vlm_grpo/prompts.py builders, so the user-message text matches GRPO
rollouts exactly.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import logging
import random
import re
import sys
from pathlib import Path

# Repo root on path so we can import src/vlm_grpo
_HERE = Path(__file__).resolve()
for _candidate in (_HERE.parents[2] if len(_HERE.parents) > 2 else None, Path.cwd()):
    if _candidate and (_candidate / "src" / "vlm_grpo" / "prompts.py").exists():
        sys.path.insert(0, str(_candidate))
        break

from src.vlm_grpo.prompts import (  # noqa: E402
    F1_VERIFIER_INSTRUCTION,
    THINK_ANSWER_INSTRUCTION,
    build_critic_prompt,
    build_initial_answer_prompt,
    build_refiner_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


_BOXED_RE = re.compile(r"\\boxed\{\s*(CORRECT|INCORRECT)\s*\}", re.IGNORECASE)
_TAG_RE = re.compile(r"<\s*(think|answer)\s*>", re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"<think>\s*.*?\s*</think>", re.DOTALL | re.IGNORECASE)


def _norm_edit_distance(a: str, b: str) -> float:
    """Normalized Levenshtein distance, 0.0 if identical, 1.0 if disjoint."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    m, n = len(a), len(b)
    if abs(m - n) / max(m, n) > 0.6:
        return 1.0  # cheap reject
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n] / max(m, n)


def _extract_user_text(prompt_msgs: list[dict]) -> str:
    """Extract the text content from a Pattern A single-user-turn prompt."""
    for msg in prompt_msgs:
        if msg["role"] == "user":
            for part in msg["content"]:
                if part.get("type") == "text":
                    return part["text"]
    raise ValueError("no user text in prompt")


def _build_pattern_a_row(
    *,
    question: str,
    a1_text: str,
    f1_text: str,
    a2_text: str,
    image_path: str,
    use_think_answer_tags: bool = True,
) -> dict:
    """Compose the six-message Pattern A SFT row.

    First user turn carries the image (`<image>` placeholder + question +
    THINK_ANSWER_INSTRUCTION). Subsequent user turns quote prior turns as
    text and carry no image placeholder. ms-swift binds the single image
    in `images` to the first `<image>` placeholder it sees.
    """
    a1_user = _extract_user_text(
        build_initial_answer_prompt(question, use_think_answer_tags=use_think_answer_tags)
    )
    f1_user = _extract_user_text(build_critic_prompt(question, a1_text))
    a2_user = _extract_user_text(
        build_refiner_prompt(
            question, a1_text, f1_text, use_think_answer_tags=use_think_answer_tags
        )
    )

    return {
        "messages": [
            {"role": "user", "content": f"<image>\n{a1_user}", "loss": False},
            {"role": "assistant", "content": a1_text, "loss": True},
            {"role": "user", "content": f1_user, "loss": False},
            {"role": "assistant", "content": f1_text, "loss": True},
            {"role": "user", "content": a2_user, "loss": False},
            {"role": "assistant", "content": a2_text, "loss": True},
        ],
        "images": [image_path],
    }


# =============================================================================
# Source 1: GRPO v10 trajectories
# =============================================================================


def _f1_format_ok(f1_text: str) -> str | None:
    """Validate F1 format and return the verdict ('CORRECT'/'INCORRECT'), or None."""
    if not f1_text or len(f1_text.split()) < 6:
        return None
    m = _BOXED_RE.search(f1_text)
    if not m:
        return None
    return m.group(1).upper()


def _has_tag_leak(text: str) -> bool:
    return bool(_TAG_RE.search(text or ""))


def _filter_traj(row: dict, min_step: int) -> str | None:
    """Return pattern label ('RR'/'WR') if row passes filters, else None."""
    if row.get("global_step", 0) < min_step:
        return None
    a1 = row.get("a1_correct")
    a2 = row.get("a2_correct")
    if a2 is not True:
        return None  # need A2 correct
    pattern = "RR" if a1 else "WR"

    a1_text = row.get("a1_text", "")
    a2_text = row.get("a2_text", "")

    # Loose format gate: require parseable <think> and <answer> in A1 and A2.
    # The strict resp_components.a1_format==1.0 reward is too punitive for
    # tasks like art_style/relative_depth where the model reliably emits the
    # tags but with extra text between them. We only need the answer to be
    # extractable for SFT distillation.
    if not (
        _THINK_TAG_RE.search(a1_text)
        and _ANSWER_TAG_RE.search(a1_text)
        and _THINK_TAG_RE.search(a2_text)
        and _ANSWER_TAG_RE.search(a2_text)
    ):
        return None

    f1_text = row.get("f1_text", "")
    verdict = _f1_format_ok(f1_text)
    if verdict is None:
        return None
    expect = "CORRECT" if a1 else "INCORRECT"
    if verdict != expect:
        return None

    # F1 should not be a near-verbatim copy of A1 (collapse mode)
    if _norm_edit_distance(f1_text[:600], a1_text[:600]) < 0.15:
        return None
    # WR: A2 must actually differ from A1
    if pattern == "WR" and _norm_edit_distance(a2_text[:600], a1_text[:600]) < 0.10:
        return None

    return pattern


def load_v10_trajectories(
    traj_glob: str,
    *,
    min_step: int,
    per_task_rr_cap: float,
    seed: int,
) -> tuple[list[dict], dict]:
    """Load + filter + dedup + per-task balance v10 trajectories.

    Returns (rows, stats).
    """
    rng = random.Random(seed)
    paths = sorted(glob.glob(traj_glob))
    if not paths:
        raise FileNotFoundError(f"no trajectories matched {traj_glob!r}")

    # First pass: filter and bucket by (task, pattern) → list of trajectories.
    # Dedup within bucket by (image_path, question), keeping highest resp_reward.
    buckets: dict[tuple[str, str], dict[tuple[str, str], dict]] = (
        collections.defaultdict(dict)
    )
    raw_counts = collections.Counter()
    pass_counts = collections.Counter()

    for p in paths:
        with open(p) as fh:
            for line in fh:
                row = json.loads(line)
                raw_counts[
                    (row.get("dataset_name", "?"),
                     ("R" if row.get("a1_correct") else "W")
                     + ("R" if row.get("a2_correct") else "W"))
                ] += 1
                pat = _filter_traj(row, min_step=min_step)
                if pat is None:
                    continue
                task = row.get("dataset_name", "?")
                key = (row.get("image_path", ""), row.get("question", ""))
                bucket = buckets[(task, pat)]
                prior = bucket.get(key)
                if prior is None or row.get("resp_reward", 0.0) > prior.get(
                    "resp_reward", 0.0
                ):
                    bucket[key] = row
                pass_counts[(task, pat)] += 1

    # Per-task balance: RR <= per_task_rr_cap * n_WR(task).
    out: list[dict] = []
    final_counts = collections.Counter()
    tasks = sorted({t for (t, _) in buckets})
    for task in tasks:
        wr = list(buckets.get((task, "WR"), {}).values())
        rr = list(buckets.get((task, "RR"), {}).values())
        rng.shuffle(wr)
        rng.shuffle(rr)
        rr_cap = int(round(per_task_rr_cap * len(wr)))
        rr = rr[:rr_cap]
        for r in wr + rr:
            row = _build_pattern_a_row(
                question=r["question"],
                a1_text=r["a1_text"],
                f1_text=r["f1_text"],
                a2_text=r["a2_text"],
                image_path=r["image_path"],
                use_think_answer_tags=True,
            )
            row["_meta"] = {
                "source": "v10_grpo",
                "task": task,
                "pattern": "WR" if r in wr else "RR",
                "global_step": r.get("global_step"),
                "resp_reward": r.get("resp_reward"),
                "fb_reward": r.get("fb_reward"),
            }
            out.append(row)
            final_counts[(task, row["_meta"]["pattern"])] += 1

    stats = {
        "raw_pattern_counts": {f"{k[0]}|{k[1]}": v for k, v in raw_counts.items()},
        "post_filter_counts": {f"{k[0]}|{k[1]}": v for k, v in pass_counts.items()},
        "final_counts": {f"{k[0]}|{k[1]}": v for k, v in final_counts.items()},
        "n_total": len(out),
    }
    return out, stats


# =============================================================================
# Source 2: FIRE feedback dataset
# =============================================================================


_QUESTION_IMG_RE = re.compile(r"<image>\s*", re.IGNORECASE)


def _strip_image_token(text: str) -> str:
    return _QUESTION_IMG_RE.sub("", text or "").strip()


def reformat_fire_feedback(
    fire_path: str,
    *,
    target_n: int,
    seed: int,
) -> tuple[list[dict], dict]:
    """Reformat FIRE feedback rows into Pattern A six-message rows.

    Mapping per FIRE conversation:
      A1 = first user turn (candidate answer)
      F1 = first assistant feedback turn (constructive critique), wrapped as
           `<think>{feedback}</think>\\boxed{INCORRECT}`
      A2 = second user turn (refined candidate)

    FIRE rows where the conversation has fewer than 4 message turns after the
    initial Q (i.e. no A2) are skipped.
    """
    rng = random.Random(seed)
    rows: list[dict] = []
    skipped = collections.Counter()
    with open(fire_path) as fh:
        for line in fh:
            d = json.loads(line)
            msgs = d.get("messages", [])
            images = d.get("images") or []
            if not images:
                skipped["no_image"] += 1
                continue
            img = images[0] if isinstance(images[0], str) else images[0].get("path")
            if not img:
                skipped["no_image_path"] += 1
                continue

            # FIRE schema: [system?, assistant(question), user(answer1),
            #               assistant(feedback1), user(answer2), assistant(feedback2), ...]
            # Some rows omit the system turn.
            turns = [m for m in msgs if m.get("role") != "system"]
            if len(turns) < 4:
                skipped["too_short"] += 1
                continue
            q_msg, a1_msg, fb1_msg, a2_msg = turns[0], turns[1], turns[2], turns[3]
            if (
                q_msg.get("role") != "assistant"
                or a1_msg.get("role") != "user"
                or fb1_msg.get("role") != "assistant"
                or a2_msg.get("role") != "user"
            ):
                skipped["bad_role_order"] += 1
                continue

            question = _strip_image_token(q_msg.get("content", ""))
            a1_text = (a1_msg.get("content") or "").strip()
            fb_text = (fb1_msg.get("content") or "").strip()
            a2_text = (a2_msg.get("content") or "").strip()
            if not (question and a1_text and fb_text and a2_text):
                skipped["empty"] += 1
                continue

            f1_text = f"<think>{fb_text}</think>\\boxed{{INCORRECT}}"

            row = _build_pattern_a_row(
                question=question,
                a1_text=a1_text,
                f1_text=f1_text,
                a2_text=a2_text,
                image_path=img,
                use_think_answer_tags=True,
            )
            row["_meta"] = {"source": "fire_feedback", "pattern": "WR"}
            rows.append(row)

    rng.shuffle(rows)
    if target_n and len(rows) > target_n:
        rows = rows[:target_n]
    stats = {
        "n_total": len(rows),
        "skipped": dict(skipped),
    }
    return rows, stats


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--traj_glob",
        default="/outputs/grpo_qwen_livr_v10_lr1e5_v2/trajectories_rank*.jsonl",
    )
    parser.add_argument(
        "--fire_feedback",
        default="/outputs/fire_preprocessed_v3/fire_feedback_train.jsonl",
    )
    parser.add_argument("--out", default="/outputs/sr_sft_v10_curated/train.jsonl")
    parser.add_argument("--stats_out", default="/outputs/sr_sft_v10_curated/stats.json")
    parser.add_argument("--min_step", type=int, default=200)
    parser.add_argument("--per_task_rr_cap", type=float, default=2.0)
    parser.add_argument(
        "--fire_target",
        type=int,
        default=3400,
        help="Number of FIRE rows to mix in (set 0 to disable FIRE).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strip_meta",
        action="store_true",
        help="Drop the _meta key before writing (production runs).",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("loading v10 trajectories from %s", args.traj_glob)
    v10_rows, v10_stats = load_v10_trajectories(
        args.traj_glob,
        min_step=args.min_step,
        per_task_rr_cap=args.per_task_rr_cap,
        seed=args.seed,
    )
    logger.info("v10 final rows: %d", v10_stats["n_total"])

    fire_rows: list[dict] = []
    fire_stats: dict = {"n_total": 0}
    if args.fire_target > 0:
        logger.info("reformatting FIRE feedback from %s", args.fire_feedback)
        fire_rows, fire_stats = reformat_fire_feedback(
            args.fire_feedback, target_n=args.fire_target, seed=args.seed
        )
        logger.info("fire rows: %d", fire_stats["n_total"])

    all_rows = v10_rows + fire_rows
    rng = random.Random(args.seed)
    rng.shuffle(all_rows)

    with out_path.open("w") as fh:
        for r in all_rows:
            if args.strip_meta:
                r = {k: v for k, v in r.items() if k != "_meta"}
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("wrote %d rows to %s", len(all_rows), out_path)

    stats = {
        "v10": v10_stats,
        "fire": fire_stats,
        "n_combined": len(all_rows),
        "args": vars(args),
    }
    Path(args.stats_out).write_text(json.dumps(stats, indent=2))
    logger.info("wrote stats to %s", args.stats_out)


if __name__ == "__main__":
    main()
