#!/usr/bin/env python3
"""Build LIVR Relative Depth MCQ dataset from NYUv2 depth maps.

Compares depth (distance from camera) at two marked points in an image.
3-way MCQ: "Point 1 is closer", "Point 2 is closer", "About the same".

Ground truth comes directly from per-pixel depth values in NYUv2.
Lower depth value = closer to camera.

Usage:
    python scripts/livr/build_relative_depth.py
    python scripts/livr/build_relative_depth.py --source-dir /outputs/livr_sources/nyuv2
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
    draw_keypoint,
    logger,
    make_livr_record,
    save_image,
    shuffle_choices,
    write_jsonl,
)

TASK_NAME = "livr_relative_depth"
N_SAMPLES = 1200
PATCH_RADIUS = 10  # Radius for averaging depth values around a point
# Relative difference threshold: below this, depths are "about the same"
SAME_THRESHOLD = 0.10
# Minimum relative difference for clear closer/farther pairs
MIN_DIFF_THRESHOLD = 0.15


def _get_patch_depth(
    depth_array: np.ndarray,
    x: int,
    y: int,
    radius: int = PATCH_RADIUS,
) -> float:
    """Get average depth of a patch centered at (x, y).

    Args:
        depth_array: Depth map as numpy array (H, W), float values in meters.
        x: X coordinate.
        y: Y coordinate.
        radius: Patch radius.

    Returns:
        Average depth value (meters). Lower = closer to camera.
    """
    h, w = depth_array.shape[:2]
    y1 = max(0, y - radius)
    y2 = min(h, y + radius + 1)
    x1 = max(0, x - radius)
    x2 = min(w, x + radius + 1)

    patch = depth_array[y1:y2, x1:x2]
    # Filter out zero/invalid depth values
    valid = patch[patch > 0]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(valid))


def _build_from_huggingface(
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict]:
    """Build depth comparison MCQs from NYUv2 HuggingFace dataset.

    Uses streaming mode to avoid downloading the full 47K-sample dataset
    (~34GB). We only need ~1500 images for 1200 MCQ samples.

    Args:
        args: Command-line arguments.
        rng: Random number generator.

    Returns:
        List of MCQ records.
    """
    from datasets import load_dataset

    logger.info("Loading NYUv2 from HuggingFace (streaming mode)...")
    ds = load_dataset(
        "sayakpaul/nyu_depth_v2",
        split="train",
        trust_remote_code=True,
        streaming=True,
    )

    # Collect samples into memory (only keep what we need)
    samples = []
    max_to_load = 2000  # Buffer above n_samples for class balancing
    for i, sample in enumerate(ds):
        samples.append(sample)
        if len(samples) >= max_to_load:
            break
        if (i + 1) % 500 == 0:
            logger.info(f"  Streamed {i + 1} samples...")

    logger.info(f"Loaded {len(samples)} NYUv2 samples via streaming")

    indices = list(range(len(samples)))
    rng.shuffle(indices)

    return _build_records(
        indices=indices,
        get_rgb_fn=lambda idx: samples[idx]["image"].convert("RGB"),
        get_depth_fn=lambda idx: np.array(samples[idx]["depth_map"], dtype=np.float32),
        args=args,
        rng=rng,
    )


def _build_from_disk(
    source_dir: str,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict]:
    """Build depth comparison MCQs from NYUv2 saved to disk.

    Expects the HuggingFace dataset saved via save_to_disk().

    Args:
        source_dir: Path to saved dataset directory.
        args: Command-line arguments.
        rng: Random number generator.

    Returns:
        List of MCQ records.
    """
    from datasets import load_from_disk

    dataset_path = os.path.join(source_dir, "dataset")
    if os.path.exists(dataset_path):
        ds = load_from_disk(dataset_path)
    else:
        ds = load_from_disk(source_dir)
    logger.info(f"Loaded {len(ds)} NYUv2 samples from disk")

    indices = list(range(len(ds)))
    rng.shuffle(indices)

    return _build_records(
        indices=indices,
        get_rgb_fn=lambda idx: ds[idx]["image"].convert("RGB"),
        get_depth_fn=lambda idx: np.array(ds[idx]["depth_map"], dtype=np.float32),
        args=args,
        rng=rng,
    )


def _build_records(
    indices: list[int],
    get_rgb_fn: callable,
    get_depth_fn: callable,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict]:
    """Core MCQ construction from RGB + depth pairs.

    For each image, samples multiple point pairs to maximize yield.
    Balances the 3 answer classes (closer_1, closer_2, same).

    Args:
        indices: Shuffled dataset indices.
        get_rgb_fn: Function(idx) -> PIL Image (RGB).
        get_depth_fn: Function(idx) -> np.ndarray (H, W) depth map.
        args: Command-line arguments.
        rng: Random number generator.

    Returns:
        List of MCQ records.
    """
    records = []
    target_per_class = args.n_samples // 3
    class_counts = {"closer_1": 0, "closer_2": 0, "same": 0}
    # Allow multiple pairs per image (NYUv2 has only ~1449 images)
    max_pairs_per_image = 3

    for idx in indices:
        if len(records) >= args.n_samples:
            break

        try:
            img = get_rgb_fn(idx)
            depth = get_depth_fn(idx)
        except Exception as e:
            logger.debug(f"Skipping index {idx}: {e}")
            continue

        w, h = img.size
        if w < 100 or h < 100:
            continue

        # Normalize depth if needed (NYUv2 HF depth_map may be uint16 or float)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0  # mm to meters
        elif depth.max() > 100:
            # Likely in mm or some large scale
            depth = depth / depth.max() * 10.0

        pairs_from_this = 0
        margin = 30

        for _ in range(30):
            if pairs_from_this >= max_pairs_per_image:
                break
            if len(records) >= args.n_samples:
                break

            x1 = rng.randint(margin, w - margin)
            y1 = rng.randint(margin, h - margin)
            x2 = rng.randint(margin, w - margin)
            y2 = rng.randint(margin, h - margin)

            # Ensure points are far enough apart spatially
            dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if dist < 60:
                continue

            d1 = _get_patch_depth(depth, x1, y1)
            d2 = _get_patch_depth(depth, x2, y2)

            # Skip invalid depth
            if d1 <= 0 or d2 <= 0:
                continue

            avg_d = (d1 + d2) / 2.0
            rel_diff = abs(d1 - d2) / max(avg_d, 0.01)

            if rel_diff < SAME_THRESHOLD:
                answer_class = "same"
                correct_text = "Both points are at about the same distance"
            elif rel_diff < MIN_DIFF_THRESHOLD:
                # Ambiguous zone — skip to avoid noisy labels
                continue
            elif d1 < d2:
                answer_class = "closer_1"
                correct_text = "Point 1 (red) is closer to the camera"
            else:
                answer_class = "closer_2"
                correct_text = "Point 2 (blue) is closer to the camera"

            # Balance classes
            if class_counts[answer_class] >= target_per_class + 50:
                continue

            class_counts[answer_class] += 1

            # Draw points on image
            annotated = draw_keypoint(img, x1, y1, color=(255, 0, 0), radius=12, label="1")
            annotated = draw_keypoint(annotated, x2, y2, color=(0, 0, 255), radius=12, label="2")

            out_filename = f"depth_{len(records):04d}.jpg"
            out_path = os.path.join(args.output_dir, out_filename)
            save_image(annotated, out_path)

            # 3-way MCQ
            distractors = [
                t
                for t in [
                    "Point 1 (red) is closer to the camera",
                    "Point 2 (blue) is closer to the camera",
                    "Both points are at about the same distance",
                ]
                if t != correct_text
            ]

            correct_letter, formatted_choices, _ = shuffle_choices(correct_text, distractors, rng)

            choices_str = " ".join(formatted_choices)
            question = (
                f"Two points are marked in the image: Point 1 (red) and Point 2 (blue). "
                f"Which point is closer to the camera? "
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
            pairs_from_this += 1

            if len(records) % 200 == 0:
                logger.info(f"Processed {len(records)}/{args.n_samples} depth samples")

    logger.info(f"Class distribution: {class_counts}")
    return records


def main() -> None:
    """Build relative depth MCQ dataset."""
    parser = argparse.ArgumentParser(description="Build LIVR relative depth MCQs.")
    parser.add_argument(
        "--source-dir",
        default="/outputs/livr_sources/nyuv2",
        help="Path to NYUv2 dataset (saved HF dataset or directory).",
    )
    parser.add_argument(
        "--output-dir",
        default="/outputs/image_base/livr/relative_depth",
        help="Output directory for annotated images.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/outputs/livr_data/livr_relative_depth.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if NYUv2 is already on disk
    dataset_path = os.path.join(args.source_dir, "dataset")
    if os.path.exists(dataset_path) or os.path.exists(
        os.path.join(args.source_dir, "dataset_info.json")
    ):
        logger.info(f"Loading NYUv2 from disk: {args.source_dir}")
        records = _build_from_disk(args.source_dir, args, rng)
    else:
        logger.info("NYUv2 not found on disk, loading from HuggingFace...")
        records = _build_from_huggingface(args, rng)

    write_jsonl(records, args.output_jsonl)
    logger.info(f"Built {len(records)} relative depth MCQ samples.")


if __name__ == "__main__":
    main()
