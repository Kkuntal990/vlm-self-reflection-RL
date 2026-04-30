#!/usr/bin/env python3
"""livr-v2 Visual Similarity builder.

Per Appendix A:
  * Source: NIGHTS dataset (DreamSim) — triplets (reference, A, B, vote)
  * Construction: triplet -> three-image example (reference + 2 candidates,
    candidate order randomized)
  * 2-way MCQ
  * Cross-dataset CLIP dedup: drop any triad whose reference or candidate
    is a near-duplicate of a BLINK Visual_Similarity image
  * Splits: 1000 / 250 / 135 (BLINK)

NIGHTS layout (from data.csail.mit.edu/nights/nights.zip):
    nights/
      train/
        ref/<id>.png
        distort_0/<id>.png
        distort_1/<id>.png
      train.csv (id, ref_path, distort_0_path, distort_1_path, vote)

Usage:
    python scripts/data/livr_v2/build_visual_similarity.py \\
        --nights-dir /outputs/livr_v2_sources/nights \\
        --blink-val-dir /outputs/livr_v2_sources/blink_val \\
        --output-dir /outputs/livr_v2/image_base/visual_similarity \\
        --output-jsonl-prefix /outputs/livr_v2/data/visual_similarity
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from dedup import full_dedup  # noqa: E402
from livr_common import (  # noqa: E402
    OPTION_LETTERS,
    get_font,
    log_task_banner,
    logger,
    make_livr_record,
    save_image,
    shuffle_choices_with_index,
    write_jsonl,
)

TASK_NAME = "livr_visual_similarity"
N_TRAIN = 1000
N_VAL = 250
N_TEST = 135


def _find_nights_root(root: Path) -> Path:
    candidates = list(root.rglob("nights"))
    candidates = [c for c in candidates if c.is_dir()]
    if candidates:
        return candidates[0]
    return root


def _load_triplets(nights_root: Path) -> list[dict]:
    """Load (ref_path, candidate_a_path, candidate_b_path, vote) triplets."""
    csv_candidates = list(nights_root.rglob("train.csv")) + list(nights_root.rglob("data.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No NIGHTS CSV found under {nights_root}")
    csv_path = csv_candidates[0]
    triplets = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ref = (
                    nights_root / row["ref_path"]
                    if "ref_path" in row
                    else nights_root / row.get("ref", "")
                )
                a = nights_root / (row.get("distort_0_path") or row.get("left", ""))
                b = nights_root / (row.get("distort_1_path") or row.get("right", ""))
                vote_raw = row.get("vote") or row.get("right_vote") or "0"
                # NIGHTS convention: vote=0 means LEFT (distort_0) is more similar to ref;
                # vote=1 means RIGHT (distort_1). We map "more similar" -> "correct" candidate.
                if str(vote_raw).strip() in {"0", "0.0", "left"}:
                    correct_idx = 0
                else:
                    correct_idx = 1
            except Exception:
                continue
            if ref.exists() and a.exists() and b.exists():
                triplets.append({"ref": ref, "a": a, "b": b, "correct_idx": correct_idx})
    return triplets


def _stack3(reference: Image.Image, a: Image.Image, b: Image.Image) -> Image.Image:
    cell = 256
    pad = 12
    label_h = 36
    width = 2 * cell + 3 * pad
    height = cell + 2 * label_h + 3 * pad + cell
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    font = get_font(28)
    rref = reference.copy()
    rref.thumbnail((cell, cell), Image.LANCZOS)
    rx = (width - rref.width) // 2
    ry = pad + label_h
    canvas.paste(rref, (rx, ry))
    ImageDraw.Draw(canvas).text((rx, pad), "Reference", fill=(0, 0, 0), font=font)
    cy_label = cell + 2 * pad + label_h
    cy = cy_label + label_h
    for k, img in enumerate([a, b]):
        c = img.copy()
        c.thumbnail((cell, cell), Image.LANCZOS)
        cx = pad + k * (cell + pad)
        canvas.paste(c, (cx + (cell - c.width) // 2, cy))
        ImageDraw.Draw(canvas).text(
            (cx + cell // 2 - 12, cy_label),
            f"({OPTION_LETTERS[k]})",
            fill=(0, 0, 0),
            font=font,
        )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nights-dir", default="/outputs/livr_v2_sources/nights")
    parser.add_argument("--blink-val-dir", default="/outputs/livr_v2_sources/blink_val")
    parser.add_argument("--output-dir", default="/outputs/livr_v2/image_base/visual_similarity")
    parser.add_argument("--output-jsonl-prefix", default="/outputs/livr_v2/data/visual_similarity")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-dedup", action="store_true")
    parser.add_argument("--clip-device", default="cuda")
    args = parser.parse_args()

    log_task_banner(TASK_NAME)
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nights_root = _find_nights_root(Path(args.nights_dir))
    logger.info("NIGHTS root: %s", nights_root)
    triplets = _load_triplets(nights_root)
    logger.info("Loaded %d triplets", len(triplets))
    rng.shuffle(triplets)

    needed = N_TRAIN + N_VAL + N_TEST + 200
    # (out_composite_path, record, ref_path, candidate_a_path, candidate_b_path)
    # — all three NIGHTS images are tracked so dedup can check each
    # individually against BLINK Visual_Similarity images. Appendix A:
    # "drop any triad whose reference OR candidate image is a near-
    # duplicate of a BLINK Visual Similarity image".
    saved: list[tuple[Path, dict, Path, Path, Path]] = []
    for tr in triplets:
        if len(saved) >= needed:
            break
        try:
            ref_img = Image.open(tr["ref"]).convert("RGB")
            a_img = Image.open(tr["a"]).convert("RGB")
            b_img = Image.open(tr["b"]).convert("RGB")
        except Exception:
            continue
        # Map the NIGHTS correct_idx through randomized A/B order.
        cands = [a_img, b_img]
        correct_pos, ordering = shuffle_choices_with_index(tr["correct_idx"], 2, rng)
        ordered = [cands[i] for i in ordering]
        composite = _stack3(ref_img, ordered[0], ordered[1])
        out_filename = f"vissim_{len(saved):05d}.png"
        out_path = out_dir / out_filename
        save_image(composite, str(out_path), format="PNG")

        question = "Which image is more visually similar to the reference image (top)? (A) (B)"
        formatted_choices = ["(A)", "(B)"]
        rec = make_livr_record(
            question=question,
            ground_truth=OPTION_LETTERS[correct_pos],
            choices=formatted_choices,
            image_path=str(out_path),
            dataset_name=TASK_NAME,
        )
        saved.append((out_path, rec, tr["ref"], tr["a"], tr["b"]))
        if len(saved) % 200 == 0:
            logger.info("  built %d/%d", len(saved), needed)

    keep_set: set[int] | None = None
    if not args.skip_dedup:
        blink_val = Path(args.blink_val_dir)
        images_dir = blink_val / "Visual_Similarity" / "images"
        if images_dir.exists():
            blink_imgs = sorted(images_dir.glob("*.png"))
        else:
            candidates = list(blink_val.rglob("*Visual_Similarity*/*"))
            candidates += list(blink_val.rglob("*visual_similarity*/*"))
            blink_imgs = [
                p
                for p in candidates
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        logger.info("BLINK Visual_Similarity val images: %d", len(blink_imgs))
        if blink_imgs:
            # Dedup over reference AND both candidates per Appendix A.
            unique_imgs: list[Path] = []
            seen: set[Path] = set()
            for _, _, ref, ap, bp in saved:
                for p in (ref, ap, bp):
                    if p not in seen:
                        seen.add(p)
                        unique_imgs.append(p)
            keep_imgs = full_dedup(
                candidates=unique_imgs,
                exclude=blink_imgs,
                clip_sim_thresh=0.95,
                phash_thresh=8,
                ssim_thresh=0.95,
                device=args.clip_device,
            )
            # A triad is kept only if ref AND both candidates are kept.
            keep_set = {
                i
                for i, (_, _, ref, ap, bp) in enumerate(saved)
                if ref in keep_imgs and ap in keep_imgs and bp in keep_imgs
            }

    final = [(p, r) for i, (p, r, _, _, _) in enumerate(saved) if keep_set is None or i in keep_set]
    logger.info("After dedup: %d", len(final))

    train = [r for _, r in final[:N_TRAIN]]
    val = [r for _, r in final[N_TRAIN : N_TRAIN + N_VAL]]
    test = [r for _, r in final[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]]
    for r in train:
        r["split"] = "train"
    for r in val:
        r["split"] = "val"
    for r in test:
        r["split"] = "test"
    write_jsonl(train, f"{args.output_jsonl_prefix}_train.jsonl")
    write_jsonl(val, f"{args.output_jsonl_prefix}_val.jsonl")
    write_jsonl(test, f"{args.output_jsonl_prefix}_test.jsonl")
    logger.info("Visual_Similarity done: %d/%d/%d", len(train), len(val), len(test))


if __name__ == "__main__":
    main()
