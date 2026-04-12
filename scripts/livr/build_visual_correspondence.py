#!/usr/bin/env python3
"""Build LIVR Visual Correspondence MCQ dataset from HPatches.

Uses homography ground truth to create keypoint correspondence questions.
Side-by-side composite with source keypoint and 4 candidate target points.

Usage:
    python scripts/livr/build_visual_correspondence.py
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
    create_side_by_side,
    draw_keypoint,
    logger,
    make_livr_record,
    save_image,
    write_jsonl,
)

TASK_NAME = "livr_visual_correspondence"
N_SAMPLES = 1200


def _random_homography(rng: np.random.RandomState, strength: float = 0.15) -> np.ndarray:
    """Generate a random perspective homography matrix.

    Args:
        rng: Numpy random state.
        strength: Controls how aggressive the warp is.

    Returns:
        3x3 homography matrix.
    """
    # Random perspective transform: identity + small perturbations
    H = np.eye(3) + rng.uniform(-strength, strength, (3, 3))
    H[2, 2] = 1.0  # Keep homogeneous scale
    H[2, 0] *= 0.3  # Reduce perspective distortion
    H[2, 1] *= 0.3
    return H


def _warp_point(pt: tuple[float, float], H: np.ndarray) -> tuple[float, float]:
    """Warp a 2D point using a homography matrix.

    Args:
        pt: Source point (x, y).
        H: 3x3 homography matrix.

    Returns:
        Warped point (x, y).
    """
    p = np.array([pt[0], pt[1], 1.0])
    wp = H @ p
    wp /= wp[2]
    return (float(wp[0]), float(wp[1]))


def _warp_image(img: Image.Image, H: np.ndarray) -> Image.Image:
    """Warp an image using a homography (perspective transform).

    Args:
        img: Source PIL image.
        H: 3x3 homography matrix.

    Returns:
        Warped PIL image.
    """
    # PIL uses inverse transform coefficients
    H_inv = np.linalg.inv(H)
    coeffs = H_inv.flatten()[:8] / H_inv[2, 2]
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def main() -> None:
    """Build visual correspondence MCQ dataset using synthetic homographies on COCO.

    Since HPatches is unavailable, we apply random perspective transforms
    to COCO images and use the known homography to create ground-truth
    point correspondences.
    """
    parser = argparse.ArgumentParser(description="Build LIVR visual correspondence MCQs.")
    parser.add_argument(
        "--images-dir",
        default="/outputs/image_base/coco/train2017",
        help="Path to COCO train2017 images (fallback source).",
    )
    parser.add_argument(
        "--output-dir",
        default="/outputs/image_base/livr/visual_correspondence",
        help="Output directory for composite images.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/outputs/livr_data/livr_visual_correspondence.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)

    # List source images
    all_images = sorted([
        f for f in os.listdir(args.images_dir)
        if f.endswith((".jpg", ".png"))
    ])
    logger.info(f"Found {len(all_images)} source images")

    rng.shuffle(all_images)
    os.makedirs(args.output_dir, exist_ok=True)
    records = []

    for img_idx, img_filename in enumerate(all_images):
        if len(records) >= args.n_samples:
            break

        img_path = os.path.join(args.images_dir, img_filename)
        try:
            src_img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        sw, sh = src_img.size
        if sw < 200 or sh < 200:
            continue

        # Generate random homography and warp
        H = _random_homography(np_rng, strength=0.12)
        try:
            tgt_img = _warp_image(src_img, H)
        except Exception:
            continue

        margin = 50
        src_x = rng.randint(margin, max(margin + 1, sw - margin))
        src_y = rng.randint(margin, max(margin + 1, sh - margin))

        tgt_correct = _warp_point((src_x, src_y), H)

        if not (margin <= tgt_correct[0] <= sw - margin and
                margin <= tgt_correct[1] <= sh - margin):
            continue

        # Generate 3 distractor points
        distractors = []
        for _ in range(3):
            for _ in range(50):
                dx = tgt_correct[0] + np_rng.uniform(-80, 80)
                dy = tgt_correct[1] + np_rng.uniform(-80, 80)
                dist = ((dx - tgt_correct[0])**2 + (dy - tgt_correct[1])**2) ** 0.5
                if dist > 25 and margin <= dx <= sw - margin and margin <= dy <= sh - margin:
                    distractors.append((dx, dy))
                    break
            else:
                distractors.append((
                    max(margin, min(sw - margin, tgt_correct[0] + rng.choice([-50, 50]))),
                    max(margin, min(sh - margin, tgt_correct[1] + rng.choice([-50, 50]))),
                ))

        # Shuffle
        all_points = [tgt_correct] + distractors
        indices = list(range(4))
        rng.shuffle(indices)
        correct_pos = indices.index(0)
        correct_letter = OPTION_LETTERS[correct_pos]
        shuffled_points = [all_points[i] for i in indices]

        src_annotated = draw_keypoint(src_img, src_x, src_y, color=(255, 0, 0), radius=10, label="Q")

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
            left_label="Source (red dot = query point)",
            right_label="Target (find corresponding point)",
            target_height=350,
        )

        out_filename = f"viscorr_{len(records):04d}.jpg"
        out_path = os.path.join(args.output_dir, out_filename)
        save_image(composite, out_path)

        formatted_choices = [f"({OPTION_LETTERS[j]})" for j in range(4)]
        choices_str = " ".join(formatted_choices)
        question = (
            f"A red query point is marked in the source image (left). "
            f"Which point in the target image (right) corresponds to the "
            f"same location? {choices_str}"
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
            logger.info(f"Processed {len(records)}/{args.n_samples} visual correspondence samples")

    write_jsonl(records, args.output_jsonl)
    logger.info(f"Built {len(records)} visual correspondence MCQ samples.")


if __name__ == "__main__":
    main()
