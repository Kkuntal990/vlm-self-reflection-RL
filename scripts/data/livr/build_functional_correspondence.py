#!/usr/bin/env python3
"""Build LIVR Functional Correspondence MCQ dataset from FunKPoint.

Pairs images of different objects performing the same function, with
keypoint annotations marking functionally equivalent points.

Falls back to SPair-71k cross-category pairs if FunKPoint unavailable.

Usage:
    python scripts/livr/build_functional_correspondence.py
"""

import argparse
import json
import logging
import os
import random
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from livr_common import (
    OPTION_COLORS,
    OPTION_LETTERS,
    create_side_by_side,
    draw_keypoint,
    logger,
    make_livr_record,
    save_image,
    write_jsonl,
)

TASK_NAME = "livr_functional_correspondence"
N_SAMPLES = 1200


def _load_funkpoint_annotations(funk_dir: str) -> list[dict]:
    """Load FunKPoint annotations.

    Args:
        funk_dir: Root directory of FunKPoint.

    Returns:
        List of annotation dicts.
    """
    anns = []
    repo_dir = os.path.join(funk_dir, "FunKPoint")

    # Look for annotation files
    ann_dir = os.path.join(repo_dir, "annotations")
    if not os.path.exists(ann_dir):
        ann_dir = os.path.join(repo_dir, "data", "annotations")
    if not os.path.exists(ann_dir):
        # Search for JSON files recursively
        for root, dirs, files in os.walk(repo_dir):
            for f in files:
                if f.endswith(".json") and "pair" in f.lower():
                    with open(os.path.join(root, f)) as fp:
                        data = json.load(fp)
                    if isinstance(data, list):
                        anns.extend(data)
                    elif isinstance(data, dict):
                        anns.append(data)
        return anns

    for fname in os.listdir(ann_dir):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(ann_dir, fname)) as f:
            data = json.load(f)
        if isinstance(data, list):
            anns.extend(data)
        elif isinstance(data, dict):
            anns.append(data)

    return anns


def _build_from_spair_fallback(args: argparse.Namespace, rng: random.Random) -> list[dict]:
    """Build functional correspondence MCQs using SPair-71k as fallback.

    Uses cross-category pairs where similar body parts (e.g., legs of
    different animals) serve as functional correspondences.

    Args:
        args: Command-line arguments.
        rng: Random number generator.

    Returns:
        List of MCQ records.
    """
    spair_dir = "/outputs/livr_sources/spair71k"
    pair_dir = os.path.join(spair_dir, "PairAnnotation")
    if not os.path.exists(pair_dir):
        pair_dir = os.path.join(spair_dir, "SPair-71k", "PairAnnotation")

    jpg_dir = os.path.join(spair_dir, "JPEGImages")
    if not os.path.exists(jpg_dir):
        jpg_dir = os.path.join(spair_dir, "SPair-71k", "JPEGImages")

    # Load all pairs
    all_pairs = []
    for split in ["trn", "val"]:
        split_dir = os.path.join(pair_dir, split)
        if not os.path.exists(split_dir):
            continue
        for fname in os.listdir(split_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(split_dir, fname)) as f:
                ann = json.load(f)
            all_pairs.append(ann)

    rng.shuffle(all_pairs)
    records = []

    for ann in all_pairs:
        if len(records) >= args.n_samples:
            break

        src_name = ann.get("src_imname", "")
        tgt_name = ann.get("trg_imname", "")
        category = ann.get("category", "")

        src_path = os.path.join(jpg_dir, category, src_name)
        tgt_path = os.path.join(jpg_dir, category, tgt_name)

        if not os.path.exists(src_path) or not os.path.exists(tgt_path):
            continue

        src_kps = ann.get("src_kps", [])
        tgt_kps = ann.get("trg_kps", [])
        src_vis = ann.get("src_kps_visible", [1] * len(src_kps))
        tgt_vis = ann.get("trg_kps_visible", [1] * len(tgt_kps))

        valid_pairs = []
        n_kps = min(len(src_kps), len(tgt_kps))
        for k in range(n_kps):
            sv = src_vis[k] if k < len(src_vis) else 0
            tv = tgt_vis[k] if k < len(tgt_vis) else 0
            if sv and tv:
                valid_pairs.append(k)

        if len(valid_pairs) < 4:
            continue

        try:
            src_img = Image.open(src_path).convert("RGB")
            tgt_img = Image.open(tgt_path).convert("RGB")
        except Exception:
            continue

        query_kp_idx = rng.choice(valid_pairs)
        src_pt = src_kps[query_kp_idx]
        tgt_correct = tgt_kps[query_kp_idx]

        distractor_pool = [k for k in valid_pairs if k != query_kp_idx]
        if len(distractor_pool) < 3:
            continue

        distractor_kp_indices = rng.sample(distractor_pool, 3)
        distractor_points = [tgt_kps[k] for k in distractor_kp_indices]

        all_points = [tgt_correct] + distractor_points
        indices = list(range(4))
        rng.shuffle(indices)
        correct_pos = indices.index(0)
        correct_letter = OPTION_LETTERS[correct_pos]
        shuffled_points = [all_points[i] for i in indices]

        sx, sy = int(src_pt[0]), int(src_pt[1])
        src_annotated = draw_keypoint(src_img, sx, sy, color=(255, 0, 0), radius=10, label="Q")

        tgt_annotated = tgt_img.copy()
        for j, pt in enumerate(shuffled_points):
            color = OPTION_COLORS[j]
            label = f"({OPTION_LETTERS[j]})"
            tgt_annotated = draw_keypoint(
                tgt_annotated,
                int(pt[0]),
                int(pt[1]),
                color=color,
                radius=10,
                label=label,
            )

        composite = create_side_by_side(
            src_annotated,
            tgt_annotated,
            left_label=f"Object 1 (red dot = functional point)",
            right_label=f"Object 2 (find same function)",
            target_height=350,
        )

        out_filename = f"funcorr_{len(records):04d}.jpg"
        out_path = os.path.join(args.output_dir, out_filename)
        save_image(composite, out_path)

        formatted_choices = [f"({OPTION_LETTERS[j]})" for j in range(4)]
        choices_str = " ".join(formatted_choices)
        question = (
            f"A functional keypoint is marked with a red dot on the object in "
            f"the left image. Which point in the right image has the same "
            f"functional role? {choices_str}"
        )

        record = make_livr_record(
            question=question,
            ground_truth=correct_letter,
            choices=formatted_choices,
            image_path=out_path,
            dataset_name=TASK_NAME,
        )
        records.append(record)

        if len(records) % 200 == 0:
            logger.info(
                f"Processed {len(records)}/{args.n_samples} functional correspondence samples"
            )

    return records


def main() -> None:
    """Build functional correspondence MCQ dataset."""
    parser = argparse.ArgumentParser(description="Build LIVR functional correspondence MCQs.")
    parser.add_argument(
        "--source-dir",
        default="/outputs/livr_sources/funkpoint",
        help="Path to FunKPoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="/outputs/image_base/livr/functional_correspondence",
        help="Output directory for composite images.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/outputs/livr_data/livr_functional_correspondence.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Try FunKPoint first
    anns = _load_funkpoint_annotations(args.source_dir)
    if len(anns) < 100:
        logger.warning(
            f"FunKPoint has only {len(anns)} annotations. "
            f"Falling back to SPair-71k for functional correspondence."
        )
        records = _build_from_spair_fallback(args, rng)
    else:
        logger.info(f"Loaded {len(anns)} FunKPoint annotations")
        # TODO: Implement FunKPoint-specific MCQ generation if annotations available
        logger.warning("FunKPoint annotation format not yet implemented. Using SPair fallback.")
        records = _build_from_spair_fallback(args, rng)

    write_jsonl(records, args.output_jsonl)
    logger.info(f"Built {len(records)} functional correspondence MCQ samples.")


if __name__ == "__main__":
    main()
