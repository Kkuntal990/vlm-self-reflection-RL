#!/usr/bin/env python3
"""Build LIVR Object Localization MCQ dataset from COCO.

Filters for objects with 15-50% image area, generates IoU 0.2-0.5
distractor bounding boxes, draws all on the image as a 4-way MCQ.

Usage:
    python scripts/livr/build_object_localization.py
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
    compute_iou,
    draw_bbox,
    get_font,
    logger,
    make_livr_record,
    save_image,
    write_jsonl,
)

TASK_NAME = "livr_object_localization"
N_SAMPLES = 1200
MIN_AREA_FRAC = 0.15
MAX_AREA_FRAC = 0.50
MIN_DISTRACTOR_IOU = 0.2
MAX_DISTRACTOR_IOU = 0.5


def _generate_distractor_bbox(
    gt_bbox: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    rng: random.Random,
) -> tuple[float, float, float, float]:
    """Generate a single distractor bbox with IoU ~0.2-0.5 from ground truth.

    Args:
        gt_bbox: Ground truth (x, y, w, h).
        img_w: Image width.
        img_h: Image height.
        rng: Random number generator.

    Returns:
        Distractor bbox as (x, y, w, h).
    """
    x, y, w, h = gt_bbox
    for _ in range(100):
        # Random shift: 20-60% of bbox dimensions
        dx = rng.uniform(-0.6, 0.6) * w
        dy = rng.uniform(-0.6, 0.6) * h
        # Small scale variation
        sw = rng.uniform(0.8, 1.2)
        sh = rng.uniform(0.8, 1.2)

        nx = max(0, x + dx)
        ny = max(0, y + dy)
        nw = min(w * sw, img_w - nx)
        nh = min(h * sh, img_h - ny)

        if nw <= 0 or nh <= 0:
            continue

        iou = compute_iou(gt_bbox, (nx, ny, nw, nh))
        if MIN_DISTRACTOR_IOU <= iou <= MAX_DISTRACTOR_IOU:
            return (nx, ny, nw, nh)

    # Fallback: just shift by 30%
    dx = 0.3 * w * rng.choice([-1, 1])
    dy = 0.3 * h * rng.choice([-1, 1])
    return (max(0, x + dx), max(0, y + dy), w, h)


def main() -> None:
    """Build object localization MCQ dataset from COCO."""
    parser = argparse.ArgumentParser(description="Build LIVR object localization MCQs.")
    parser.add_argument(
        "--annotations-path",
        default="/outputs/livr_sources/coco_annotations/annotations/instances_train2017.json",
        help="Path to COCO instances_train2017.json.",
    )
    parser.add_argument(
        "--images-dir",
        default="/outputs/image_base/coco/train2017",
        help="Path to COCO train2017 images.",
    )
    parser.add_argument(
        "--output-dir",
        default="/outputs/image_base/livr/object_localization",
        help="Output directory for annotated images.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/outputs/livr_data/livr_object_localization.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    logger.info("Loading COCO annotations...")
    with open(args.annotations_path) as f:
        coco = json.load(f)

    # Build lookups
    images_info = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    # Filter annotations: area 15-50% of image
    valid_anns = []
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        img_info = images_info.get(ann["image_id"])
        if not img_info:
            continue
        img_area = img_info["width"] * img_info["height"]
        ann_area = ann["bbox"][2] * ann["bbox"][3]
        area_frac = ann_area / img_area if img_area > 0 else 0

        if MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC:
            valid_anns.append((ann, img_info))

    logger.info(f"Found {len(valid_anns)} annotations with area in [{MIN_AREA_FRAC}, {MAX_AREA_FRAC}]")

    rng.shuffle(valid_anns)
    selected = valid_anns[: args.n_samples]

    os.makedirs(args.output_dir, exist_ok=True)
    records = []

    for idx, (ann, img_info) in enumerate(selected):
        img_filename = img_info["file_name"]
        img_path = os.path.join(args.images_dir, img_filename)

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            continue

        gt_bbox = tuple(ann["bbox"])  # (x, y, w, h)
        cat_name = categories.get(ann["category_id"], "object")
        img_w, img_h = img.size

        # Generate 3 distractor bboxes
        distractor_bboxes = []
        for _ in range(3):
            d_bbox = _generate_distractor_bbox(gt_bbox, img_w, img_h, rng)
            distractor_bboxes.append(d_bbox)

        # Shuffle correct among distractors
        all_bboxes = [gt_bbox] + distractor_bboxes
        indices = list(range(4))
        rng.shuffle(indices)
        correct_pos = indices.index(0)
        correct_letter = OPTION_LETTERS[correct_pos]
        shuffled_bboxes = [all_bboxes[i] for i in indices]

        # Draw all bboxes on image
        annotated = img.copy()
        for j, bbox in enumerate(shuffled_bboxes):
            color = OPTION_COLORS[j]
            label = f"({OPTION_LETTERS[j]})"
            annotated = draw_bbox(annotated, bbox, color=color, label=label, width=3)

        out_filename = f"objloc_{idx:04d}.jpg"
        out_path = os.path.join(args.output_dir, out_filename)
        save_image(annotated, out_path)

        # Build MCQ
        formatted_choices = [f"({OPTION_LETTERS[j]})" for j in range(4)]
        choices_str = " ".join(formatted_choices)
        question = (
            f"Which bounding box most accurately localizes the {cat_name} in the image? "
            f"{choices_str}"
        )

        record = make_livr_record(
            question=question,
            ground_truth=correct_letter,
            choices=formatted_choices,
            image_path=out_path,
            dataset_name=TASK_NAME,
        )
        records.append(record)

        if (idx + 1) % 200 == 0:
            logger.info(f"Processed {idx + 1}/{len(selected)} localization samples")

    write_jsonl(records, args.output_jsonl)
    logger.info(f"Built {len(records)} object localization MCQ samples.")


if __name__ == "__main__":
    main()
