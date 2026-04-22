#!/usr/bin/env python3
"""Build LIVR Relative Reflectance MCQ dataset from Multi-Illumination Dataset.

Compares surface reflectance at two marked points in an image.
3-way MCQ: "Point 1 brighter", "Point 2 brighter", "About the same".

If MID albedo images are unavailable, falls back to generating
luminance comparison questions from COCO images.

Usage:
    python scripts/livr/build_relative_reflectance.py
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from livr_common import (
    OPTION_COLORS,
    OPTION_LETTERS,
    draw_keypoint,
    logger,
    make_livr_record,
    save_image,
    shuffle_choices,
    write_jsonl,
)

TASK_NAME = "livr_relative_reflectance"
N_SAMPLES = 1200
PATCH_RADIUS = 15  # Radius for averaging pixel values around a point
SAME_THRESHOLD = 0.08  # Relative difference below this = "about the same"


def _get_patch_luminance(
    img_array: np.ndarray,
    x: int,
    y: int,
    radius: int = PATCH_RADIUS,
) -> float:
    """Get average luminance of a patch centered at (x, y).

    Args:
        img_array: Image as numpy array (H, W, 3), float [0, 1].
        x: X coordinate.
        y: Y coordinate.
        radius: Patch radius.

    Returns:
        Average luminance in [0, 1].
    """
    h, w = img_array.shape[:2]
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)

    patch = img_array[y1:y2, x1:x2]
    # Luminance: 0.2126*R + 0.7152*G + 0.0722*B
    luminance = 0.2126 * patch[:, :, 0] + 0.7152 * patch[:, :, 1] + 0.0722 * patch[:, :, 2]
    return float(np.mean(luminance))


def _build_from_images(
    images_dir: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict]:
    """Build reflectance comparison MCQs from images.

    Works with any image directory (MID albedo or COCO as fallback).

    Args:
        images_dir: Directory containing source images.
        args: Command-line arguments.
        rng: Random number generator.

    Returns:
        List of MCQ records.
    """
    # Collect image files
    img_files = []
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".ppm", ".tif")):
                img_files.append(os.path.join(root, f))

    logger.info(f"Found {len(img_files)} images in {images_dir}")
    rng.shuffle(img_files)

    records = []
    target_per_class = args.n_samples // 3  # Balance the 3 classes

    class_counts = {"brighter_1": 0, "brighter_2": 0, "same": 0}

    for img_path in img_files:
        if len(records) >= args.n_samples:
            break

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        w, h = img.size
        if w < 100 or h < 100:
            continue

        img_array = np.array(img, dtype=np.float32) / 255.0
        margin = 40

        # Try to find a pair of points with desired relationship
        for _ in range(20):
            x1 = rng.randint(margin, w - margin)
            y1 = rng.randint(margin, h - margin)
            x2 = rng.randint(margin, w - margin)
            y2 = rng.randint(margin, h - margin)

            # Ensure points are far enough apart
            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if dist < 50:
                continue

            lum1 = _get_patch_luminance(img_array, x1, y1)
            lum2 = _get_patch_luminance(img_array, x2, y2)

            avg_lum = (lum1 + lum2) / 2.0
            if avg_lum < 0.01:
                continue

            rel_diff = abs(lum1 - lum2) / max(avg_lum, 0.01)

            if rel_diff < SAME_THRESHOLD:
                answer_class = "same"
                correct_text = "Both points have similar reflectance"
            elif lum1 > lum2:
                answer_class = "brighter_1"
                correct_text = "Point 1 (red) has higher reflectance"
            else:
                answer_class = "brighter_2"
                correct_text = "Point 2 (blue) has higher reflectance"

            # Balance classes
            if class_counts[answer_class] >= target_per_class + 50:
                continue

            class_counts[answer_class] += 1

            # Draw points on image
            annotated = draw_keypoint(img, x1, y1, color=(255, 0, 0), radius=12, label="1")
            annotated = draw_keypoint(annotated, x2, y2, color=(0, 0, 255), radius=12, label="2")

            out_filename = f"reflectance_{len(records):04d}.jpg"
            out_path = os.path.join(args.output_dir, out_filename)
            save_image(annotated, out_path)

            # 3-way MCQ
            distractors = [
                t
                for t in [
                    "Point 1 (red) has higher reflectance",
                    "Point 2 (blue) has higher reflectance",
                    "Both points have similar reflectance",
                ]
                if t != correct_text
            ]

            correct_letter, formatted_choices, _ = shuffle_choices(correct_text, distractors, rng)

            choices_str = " ".join(formatted_choices)
            question = (
                f"Two points are marked in the image: Point 1 (red) and Point 2 (blue). "
                f"Which statement about their relative surface reflectance is correct? "
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

            if len(records) % 200 == 0:
                logger.info(f"Processed {len(records)}/{args.n_samples} reflectance samples")
            break

    logger.info(f"Class distribution: {class_counts}")
    return records


def main() -> None:
    """Build relative reflectance MCQ dataset."""
    parser = argparse.ArgumentParser(description="Build LIVR relative reflectance MCQs.")
    parser.add_argument(
        "--source-dir",
        default="/outputs/livr_sources/mid",
        help="Path to MID albedo images.",
    )
    parser.add_argument(
        "--fallback-images-dir",
        default="/outputs/image_base/coco/train2017",
        help="Fallback image directory if MID unavailable.",
    )
    parser.add_argument(
        "--output-dir",
        default="/outputs/image_base/livr/relative_reflectance",
        help="Output directory for annotated images.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/outputs/livr_data/livr_relative_reflectance.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if MID albedo images exist
    mid_img_count = 0
    if os.path.exists(args.source_dir):
        for root, _, files in os.walk(args.source_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".png", ".tif")):
                    mid_img_count += 1

    if mid_img_count >= 50:
        logger.info(f"Using MID dataset ({mid_img_count} images)")
        records = _build_from_images(args.source_dir, args, rng)
    else:
        logger.warning(
            f"MID has only {mid_img_count} images. "
            f"Falling back to COCO images for reflectance comparison."
        )
        records = _build_from_images(args.fallback_images_dir, args, rng)

    write_jsonl(records, args.output_jsonl)
    logger.info(f"Built {len(records)} relative reflectance MCQ samples.")


if __name__ == "__main__":
    main()
