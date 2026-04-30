#!/usr/bin/env python3
"""livr-v2 Semantic Correspondence builder.

Per Appendix A:
  * Source: SPair-71k (training split)
  * Filter: pairs with at least 4 valid keypoint correspondences
  * Construction: REF marker on source at one annotated keypoint;
    on target image place 4 candidate circles (the matching point +
    3 distractor keypoints sampled from the remaining annotations)
  * Pair-aware CLIP dedup vs BLINK Semantic_Correspondence val
  * 4-way MCQ
  * Splits: 1000 / 250 / 139

SPair-71k layout (after extraction):
    SPair-71k/
      JPEGImages/<category>/<image>.jpg
      ImageAnnotation/<category>/<image>.json
      Layout/large/trn.txt   (training pairs)
      PairAnnotation/trn/<pair_id>.json   (keypoint correspondences per pair)

Usage:
    python scripts/data/livr_v2/build_semantic_correspondence.py \\
        --spair-dir /outputs/livr_v2_sources/spair_71k \\
        --blink-val-dir /outputs/livr_v2_sources/blink_val \\
        --output-dir /outputs/livr_v2/image_base/semantic_correspondence \\
        --output-jsonl-prefix /outputs/livr_v2/data/semantic_correspondence
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dedup import clip_dedup  # noqa: E402
from livr_common import (  # noqa: E402
    OPTION_COLORS,
    OPTION_LETTERS,
    create_side_by_side,
    draw_keypoint,
    draw_ref_marker,
    log_task_banner,
    logger,
    make_livr_record,
    save_image,
    shuffle_choices_with_index,
    write_jsonl,
)

TASK_NAME = "livr_semantic_correspondence"
N_TRAIN = 1000
N_VAL = 250
N_TEST = 139
MIN_VALID_KEYPOINTS = 4

MIN_DISTRACTOR_DIST = 60  # px in target image, between GT and any distractor
MARGIN = 20  # keep keypoints away from image edges


def _find_spair_root(root: Path) -> Path:
    candidates = list(root.rglob("Layout/large/trn.txt"))
    if not candidates:
        candidates = list(root.rglob("trn.txt"))
    if not candidates:
        raise FileNotFoundError(f"Could not find SPair-71k Layout/large/trn.txt under {root}")
    return candidates[0].parents[2]


def _load_train_pairs(spair_root: Path) -> list[dict]:
    """Load all training pair annotations (with keypoint correspondences)."""
    layout = spair_root / "Layout" / "large" / "trn.txt"
    with open(layout) as f:
        pair_ids = [line.strip() for line in f if line.strip()]
    pairs = []
    for pid in pair_ids:
        ann = spair_root / "PairAnnotation" / "trn" / f"{pid}.json"
        if not ann.exists():
            continue
        with open(ann) as f:
            d = json.load(f)
        pairs.append(d)
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spair-dir", default="/outputs/livr_v2_sources/spair_71k")
    parser.add_argument("--blink-val-dir", default="/outputs/livr_v2_sources/blink_val")
    parser.add_argument(
        "--output-dir", default="/outputs/livr_v2/image_base/semantic_correspondence"
    )
    parser.add_argument(
        "--output-jsonl-prefix", default="/outputs/livr_v2/data/semantic_correspondence"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-dedup", action="store_true")
    parser.add_argument("--clip-device", default="cuda")
    args = parser.parse_args()

    log_task_banner(TASK_NAME)
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spair_root = _find_spair_root(Path(args.spair_dir))
    logger.info("SPair-71k root: %s", spair_root)
    raw_pairs = _load_train_pairs(spair_root)
    logger.info("Loaded %d training pair annotations", len(raw_pairs))

    # Filter for >=4 valid keypoint correspondences.
    valid = []
    for p in raw_pairs:
        # SPair pair JSON contains src_kps and trg_kps as lists of [x, y]
        # with -1 for invalid; filter aligned indices.
        src_kps = p.get("src_kps", [])
        trg_kps = p.get("trg_kps", [])
        if not src_kps or not trg_kps or len(src_kps) != len(trg_kps):
            continue
        kp_idx = [
            i
            for i in range(len(src_kps))
            if src_kps[i][0] >= 0
            and src_kps[i][1] >= 0
            and trg_kps[i][0] >= 0
            and trg_kps[i][1] >= 0
        ]
        if len(kp_idx) < MIN_VALID_KEYPOINTS:
            continue
        valid.append((p, kp_idx))
    logger.info("Pairs with >=%d valid keypoints: %d", MIN_VALID_KEYPOINTS, len(valid))

    rng.shuffle(valid)
    needed = N_TRAIN + N_VAL + N_TEST + 200

    saved: list[tuple[Path, dict]] = []
    for pair_idx, (pdata, kp_idx) in enumerate(valid):
        if len(saved) >= needed:
            break
        try:
            src_path = spair_root / "JPEGImages" / pdata["category"] / pdata["src_imname"]
            tgt_path = spair_root / "JPEGImages" / pdata["category"] / pdata["trg_imname"]
            src_img = Image.open(src_path).convert("RGB")
            tgt_img = Image.open(tgt_path).convert("RGB")
        except Exception as e:
            logger.warning("Failed to load pair %s: %s", pdata.get("filename"), e)
            continue

        # Pick reference keypoint and 3 distractors from remaining annotations.
        rng.shuffle(kp_idx)
        ref_idx = kp_idx[0]
        distractor_idxs = kp_idx[1:4]
        if len(distractor_idxs) < 3:
            continue

        sx, sy = pdata["src_kps"][ref_idx]
        tx, ty = pdata["trg_kps"][ref_idx]
        sw, sh = src_img.size
        tw, th = tgt_img.size

        # Validate margins.
        if not (MARGIN <= sx <= sw - MARGIN and MARGIN <= sy <= sh - MARGIN):
            continue
        if not (MARGIN <= tx <= tw - MARGIN and MARGIN <= ty <= th - MARGIN):
            continue

        gt_tgt = (tx, ty)
        distractors = []
        for di in distractor_idxs:
            dx, dy = pdata["trg_kps"][di]
            if (
                MARGIN <= dx <= tw - MARGIN
                and MARGIN <= dy <= th - MARGIN
                and ((dx - tx) ** 2 + (dy - ty) ** 2) ** 0.5 >= MIN_DISTRACTOR_DIST
            ):
                distractors.append((dx, dy))
        if len(distractors) < 3:
            continue

        points = [gt_tgt] + distractors[:3]
        correct_pos, ordering = shuffle_choices_with_index(0, 4, rng)
        ordered = [points[k] for k in ordering]

        src_an = draw_ref_marker(src_img, int(sx), int(sy))
        tgt_an = tgt_img.copy()
        for k, pt in enumerate(ordered):
            tgt_an = draw_keypoint(
                tgt_an,
                int(pt[0]),
                int(pt[1]),
                color=OPTION_COLORS[k],
                radius=10,
                label=f"({OPTION_LETTERS[k]})",
            )

        composite = create_side_by_side(
            src_an,
            tgt_an,
            left_label="Source (REF on object)",
            right_label="Target (semantically matching point)",
        )
        out_filename = f"semcorr_{len(saved):05d}.png"
        out_path = out_dir / out_filename
        save_image(composite, str(out_path), format="PNG")

        category = pdata.get("category", "object")
        question = (
            f"A semantic keypoint is marked with REF on the {category} in the source image (left). "
            f"Which point in the target image (right) corresponds to the same semantic location? "
            f"(A) (B) (C) (D)"
        )
        formatted_choices = [f"({OPTION_LETTERS[k]})" for k in range(4)]
        rec = make_livr_record(
            question=question,
            ground_truth=OPTION_LETTERS[correct_pos],
            choices=formatted_choices,
            image_path=str(out_path),
            dataset_name=TASK_NAME,
        )
        saved.append((out_path, rec))
        if (pair_idx + 1) % 200 == 0:
            logger.info("  built %d/%d", len(saved), needed)
    logger.info("Saved %d semcorr composites", len(saved))

    # Pair-aware dedup vs BLINK Semantic_Correspondence val.
    keep_set: set[Path] | None = None
    if not args.skip_dedup:
        blink_val = Path(args.blink_val_dir)
        candidates = list(blink_val.rglob("*Semantic_Correspondence*/*"))
        candidates += list(blink_val.rglob("*semantic_correspondence*/*"))
        blink_imgs = [
            p for p in candidates if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        logger.info("BLINK Sem_Corr val images: %d", len(blink_imgs))
        if blink_imgs:
            keep_set = clip_dedup(
                candidates=[p for p, _ in saved],
                exclude=blink_imgs,
                sim_thresh=0.95,
                device=args.clip_device,
            )

    final = [(p, r) for p, r in saved if keep_set is None or p in keep_set]
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
    logger.info("Semantic_Correspondence done: %d/%d/%d", len(train), len(val), len(test))


if __name__ == "__main__":
    main()
