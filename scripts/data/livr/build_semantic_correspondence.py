#!/usr/bin/env python3
"""Build LIVR Semantic Correspondence MCQ dataset from SPair-71k.

Uses annotated keypoint correspondences between images of the same
object category to create 4-way keypoint matching MCQs.

Usage:
    python scripts/livr/build_semantic_correspondence.py
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

TASK_NAME = "livr_semantic_correspondence"
N_SAMPLES = 1200


def _load_spair_pairs(spair_dir: str) -> list[dict]:
    """Load SPair-71k annotation pairs.

    Args:
        spair_dir: Root directory of SPair-71k.

    Returns:
        List of annotation dicts with keypoint pairs.
    """
    pairs = []
    # SPair-71k structure: PairAnnotation/{trn,val,test}/*.json
    pair_dir = os.path.join(spair_dir, "PairAnnotation")
    if not os.path.exists(pair_dir):
        # Try alternative structure
        pair_dir = os.path.join(spair_dir, "SPair-71k", "PairAnnotation")

    for split in ["trn", "val"]:
        split_dir = os.path.join(pair_dir, split)
        if not os.path.exists(split_dir):
            continue
        for fname in os.listdir(split_dir):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(split_dir, fname)) as f:
                ann = json.load(f)
            pairs.append(ann)

    return pairs


def main() -> None:
    """Build semantic correspondence MCQ dataset from SPair-71k."""
    parser = argparse.ArgumentParser(description="Build LIVR semantic correspondence MCQs.")
    parser.add_argument(
        "--source-dir",
        default="/outputs/livr_sources/spair71k",
        help="Path to SPair-71k directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="/outputs/image_base/livr/semantic_correspondence",
        help="Output directory for composite images.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/outputs/livr_data/livr_semantic_correspondence.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    logger.info("Loading SPair-71k annotations...")
    pairs = _load_spair_pairs(args.source_dir)
    logger.info(f"Loaded {len(pairs)} annotation pairs")

    # Find image base directory
    jpg_dir = os.path.join(args.source_dir, "JPEGImages")
    if not os.path.exists(jpg_dir):
        jpg_dir = os.path.join(args.source_dir, "SPair-71k", "JPEGImages")

    rng.shuffle(pairs)
    os.makedirs(args.output_dir, exist_ok=True)
    records = []

    for idx, ann in enumerate(pairs):
        if len(records) >= args.n_samples:
            break

        # Get image paths
        src_name = ann.get("src_imname", "")
        tgt_name = ann.get("trg_imname", "")
        category = ann.get("category", "")

        src_path = os.path.join(jpg_dir, category, src_name)
        tgt_path = os.path.join(jpg_dir, category, tgt_name)

        if not os.path.exists(src_path) or not os.path.exists(tgt_path):
            continue

        # Get keypoints
        src_kps = ann.get("src_kps", [])
        tgt_kps = ann.get("trg_kps", [])

        # Filter for visible keypoints (visibility flag)
        src_vis = ann.get("src_kps_visible", [1] * len(src_kps))
        tgt_vis = ann.get("trg_kps_visible", [1] * len(tgt_kps))

        # Build list of valid keypoint pairs (both visible)
        valid_pairs = []
        n_kps = min(len(src_kps), len(tgt_kps))
        for k in range(n_kps):
            sv = src_vis[k] if k < len(src_vis) else 0
            tv = tgt_vis[k] if k < len(tgt_vis) else 0
            if sv and tv:
                valid_pairs.append(k)

        if len(valid_pairs) < 4:
            continue  # Need at least 4 visible keypoints for 4-way MCQ

        try:
            src_img = Image.open(src_path).convert("RGB")
            tgt_img = Image.open(tgt_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load images: {e}")
            continue

        # Pick a query keypoint
        query_kp_idx = rng.choice(valid_pairs)
        src_pt = src_kps[query_kp_idx]
        tgt_correct = tgt_kps[query_kp_idx]

        # Pick 3 distractor keypoints from other visible target keypoints
        distractor_pool = [k for k in valid_pairs if k != query_kp_idx]
        if len(distractor_pool) < 3:
            continue

        distractor_kp_indices = rng.sample(distractor_pool, 3)
        distractor_points = [tgt_kps[k] for k in distractor_kp_indices]

        # Shuffle correct among distractors
        all_points = [tgt_correct] + distractor_points
        indices = list(range(4))
        rng.shuffle(indices)
        correct_pos = indices.index(0)
        correct_letter = OPTION_LETTERS[correct_pos]
        shuffled_points = [all_points[i] for i in indices]

        # Draw source keypoint
        sx, sy = int(src_pt[0]), int(src_pt[1])
        src_annotated = draw_keypoint(src_img, sx, sy, color=(255, 0, 0), radius=10, label="Q")

        # Draw target keypoints
        tgt_annotated = tgt_img.copy()
        for j, pt in enumerate(shuffled_points):
            color = OPTION_COLORS[j]
            label = f"({OPTION_LETTERS[j]})"
            tgt_annotated = draw_keypoint(
                tgt_annotated, int(pt[0]), int(pt[1]),
                color=color, radius=10, label=label,
            )

        composite = create_side_by_side(
            src_annotated, tgt_annotated,
            left_label=f"Source {category} (red dot = query)",
            right_label=f"Target {category} (find match)",
            target_height=350,
        )

        out_filename = f"semcorr_{len(records):04d}.jpg"
        out_path = os.path.join(args.output_dir, out_filename)
        save_image(composite, out_path)

        formatted_choices = [f"({OPTION_LETTERS[j]})" for j in range(4)]
        choices_str = " ".join(formatted_choices)
        question = (
            f"A keypoint is marked with a red dot on the {category} in the source image (left). "
            f"Which point in the target image (right) corresponds to the same "
            f"semantic location? {choices_str}"
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
            logger.info(f"Processed {len(records)}/{args.n_samples} semantic correspondence samples")

    write_jsonl(records, args.output_jsonl)
    logger.info(f"Built {len(records)} semantic correspondence MCQ samples.")


if __name__ == "__main__":
    main()
