#!/usr/bin/env python3
"""Bucket prompts by A1 pass rate and emit a per-task histogram report.

Reads the JSONL produced by `profile_difficulty_a1.py`, classifies each
prompt into one of four buckets, and writes:
  1. An enriched JSONL with a `difficulty_bucket` field on each row
  2. A markdown report with per-task histograms and bucket counts

Usage:
    uv run python scripts/analysis/difficulty_buckets.py \\
        --input /outputs/livr_data/livr_difficulty_a1.jsonl \\
        --output_jsonl /outputs/livr_data/livr_difficulty_a1_bucketed.jsonl \\
        --output_report /outputs/livr_data/livr_difficulty_report.md

Bucket predicates (assuming K=8):
    trivial      : a1_pass_rate >= 7/8     (model already knows)
    easy         : 4/8 <= a1_pass_rate < 7/8
    medium       : 1/8 <= a1_pass_rate < 4/8
    brick_wall   : a1_pass_rate == 0       (zero K rollouts correct)
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

BUCKETS = ("trivial", "easy", "medium", "brick_wall")


def _bucket_for(pass_rate: float, k: int) -> str:
    """Classify a prompt by its A1 pass rate.

    Thresholds expressed as fractions of K so the bucketing stays correct
    if K differs from 8.
    """
    if k <= 0:
        return "brick_wall"
    correct = round(pass_rate * k)
    if correct == 0:
        return "brick_wall"
    if correct >= max(1, int(k * 7 / 8)):  # >= 7 of 8
        return "trivial"
    if correct >= max(1, int(k * 4 / 8)):  # >= 4 of 8
        return "easy"
    return "medium"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="profile_difficulty_a1.py output JSONL")
    parser.add_argument("--output_jsonl", required=True, help="Enriched JSONL with bucket field")
    parser.add_argument("--output_report", required=True, help="Markdown report path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    rows: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    logger.info("Loaded %d profiled prompts", len(rows))

    # Per-task bucket counts (and global)
    task_buckets: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(BUCKETS, 0))
    task_correct_hist: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    global_buckets: dict[str, int] = dict.fromkeys(BUCKETS, 0)

    # Enrich and write
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as out_f:
        for row in rows:
            k = int(row.get("k", 0))
            pr = float(row.get("a1_pass_rate", 0.0))
            bucket = _bucket_for(pr, k)
            row["difficulty_bucket"] = bucket
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            task = row.get("dataset_name", "unknown")
            task_buckets[task][bucket] += 1
            global_buckets[bucket] += 1
            correct = int(row.get("a1_correct_count", 0))
            task_correct_hist[task][correct] += 1

    # Build markdown report
    lines: list[str] = []
    lines.append("# LIVR A1 Difficulty Profile\n")
    lines.append(f"- Total prompts: **{len(rows)}**\n")
    lines.append(f"- Source: `{args.input}`\n\n")

    # Global bucket summary
    lines.append("## Global bucket counts\n")
    lines.append("| Bucket | Count | Share |")
    lines.append("|--------|------:|------:|")
    for b in BUCKETS:
        n = global_buckets[b]
        share = 100.0 * n / len(rows) if rows else 0.0
        lines.append(f"| {b} | {n} | {share:.1f}% |")
    lines.append("")

    # Per-task buckets
    lines.append("## Per-task bucket counts\n")
    header = "| Task | " + " | ".join(BUCKETS) + " | total |"
    sep = "|" + "------|" * (len(BUCKETS) + 2)
    lines.append(header)
    lines.append(sep)
    for task in sorted(task_buckets.keys()):
        b = task_buckets[task]
        total = sum(b.values())
        row = f"| {task} | " + " | ".join(str(b[x]) for x in BUCKETS) + f" | {total} |"
        lines.append(row)
    lines.append("")

    # Per-task correct-count histogram (0..K)
    if rows:
        sample_k = int(rows[0].get("k", 8))
    else:
        sample_k = 8
    lines.append(f"## Per-task A1 correct-count histogram (K={sample_k})\n")
    head_cells = " | ".join(str(i) for i in range(sample_k + 1))
    lines.append(f"| Task | {head_cells} |")
    lines.append("|------|" + "-----|" * (sample_k + 1))
    for task in sorted(task_correct_hist.keys()):
        h = task_correct_hist[task]
        cells = " | ".join(str(h.get(i, 0)) for i in range(sample_k + 1))
        lines.append(f"| {task} | {cells} |")
    lines.append("")

    # Filter recommendation
    drop = global_buckets["trivial"] + global_buckets["brick_wall"]
    keep = global_buckets["easy"] + global_buckets["medium"]
    lines.append("## Recommended filter (drop trivial + brick_wall)\n")
    lines.append(f"- Drop: **{drop}** prompts ({100.0 * drop / len(rows):.1f}% of total)")
    lines.append(f"- Keep: **{keep}** prompts ({100.0 * keep / len(rows):.1f}% of total)")
    lines.append("")

    os.makedirs(os.path.dirname(args.output_report) or ".", exist_ok=True)
    with open(args.output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Wrote enriched JSONL: %s", args.output_jsonl)
    logger.info("Wrote report: %s", args.output_report)


if __name__ == "__main__":
    main()
