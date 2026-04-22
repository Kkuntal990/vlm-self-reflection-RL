#!/usr/bin/env python3
"""Build LIVR Visual Similarity MCQ dataset.

Since NIGHTS/DreamSim is unavailable, constructs visual similarity
questions from COCO by pairing images from the same vs different scenes.
Uses color histogram distance as a proxy for visual similarity.

Usage:
    python scripts/livr/build_visual_similarity.py
"""

import argparse
import json
import logging
import os
import random
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from livr_common import (
    OPTION_LETTERS,
    create_reference_and_options,
    logger,
    make_livr_record,
    save_image,
    write_jsonl,
)

TASK_NAME = "livr_visual_similarity"
N_SAMPLES = 1200


def _color_histogram(img: Image.Image, bins: int = 32) -> np.ndarray:
    """Compute a normalized color histogram for an image.

    Args:
        img: PIL Image (RGB).
        bins: Number of bins per channel.

    Returns:
        Normalized histogram vector.
    """
    arr = np.array(img.resize((128, 128), Image.LANCZOS))
    hists = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=bins, range=(0, 256))
        hists.append(h)
    hist = np.concatenate(hists).astype(np.float32)
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist


def _hist_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Compute L2 distance between two histograms.

    Args:
        h1: First histogram.
        h2: Second histogram.

    Returns:
        L2 distance.
    """
    return float(np.linalg.norm(h1 - h2))


def main() -> None:
    """Build visual similarity MCQ dataset from COCO images."""
    parser = argparse.ArgumentParser(description="Build LIVR visual similarity MCQs.")
    parser.add_argument(
        "--images-dir",
        default="/outputs/image_base/coco/train2017",
        help="Path to source images.",
    )
    parser.add_argument(
        "--annotations-path",
        default="/outputs/livr_sources/coco_annotations/annotations/instances_train2017.json",
        help="COCO annotations for category-based grouping.",
    )
    parser.add_argument(
        "--output-dir",
        default="/outputs/image_base/livr/visual_similarity",
        help="Output directory for composite images.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/outputs/livr_data/livr_visual_similarity.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load COCO annotations to group images by dominant category
    logger.info("Loading COCO annotations for category grouping...")
    with open(args.annotations_path) as f:
        coco = json.load(f)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}
    images_info = {img["id"]: img for img in coco["images"]}

    # Group images by their dominant (largest) annotation category
    img_to_cat: dict[int, int] = {}
    img_to_area: dict[int, float] = {}
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        img_id = ann["image_id"]
        area = ann["bbox"][2] * ann["bbox"][3]
        if img_id not in img_to_cat or area > img_to_area.get(img_id, 0):
            img_to_cat[img_id] = ann["category_id"]
            img_to_area[img_id] = area

    cat_to_imgs: dict[int, list[int]] = {}
    for img_id, cat_id in img_to_cat.items():
        cat_to_imgs.setdefault(cat_id, []).append(img_id)

    # Filter categories with enough images
    valid_cats = [c for c, imgs in cat_to_imgs.items() if len(imgs) >= 10]
    logger.info(f"Found {len(valid_cats)} categories with >=10 images")

    os.makedirs(args.output_dir, exist_ok=True)
    records = []

    for idx in range(args.n_samples * 2):  # Extra iterations for failures
        if len(records) >= args.n_samples:
            break

        # Pick a reference category and a different category
        ref_cat = rng.choice(valid_cats)
        diff_cat = rng.choice([c for c in valid_cats if c != ref_cat])

        ref_imgs = cat_to_imgs[ref_cat]
        diff_imgs = cat_to_imgs[diff_cat]

        if len(ref_imgs) < 3:
            continue

        # Pick reference, similar (same category), and dissimilar (different category)
        ref_id, sim_id = rng.sample(ref_imgs, 2)
        diff_id = rng.choice(diff_imgs)

        ref_info = images_info.get(ref_id)
        sim_info = images_info.get(sim_id)
        diff_info = images_info.get(diff_id)

        if not all([ref_info, sim_info, diff_info]):
            continue

        ref_path = os.path.join(args.images_dir, ref_info["file_name"])
        sim_path = os.path.join(args.images_dir, sim_info["file_name"])
        diff_path = os.path.join(args.images_dir, diff_info["file_name"])

        try:
            ref_img = Image.open(ref_path).convert("RGB")
            sim_img = Image.open(sim_path).convert("RGB")
            diff_img = Image.open(diff_path).convert("RGB")
        except Exception:
            continue

        # Verify similarity ordering using color histograms
        ref_hist = _color_histogram(ref_img)
        sim_hist = _color_histogram(sim_img)
        diff_hist = _color_histogram(diff_img)

        sim_dist = _hist_distance(ref_hist, sim_hist)
        diff_dist = _hist_distance(ref_hist, diff_hist)

        # Only keep if the same-category image is actually closer
        if sim_dist >= diff_dist:
            continue

        # 2-way MCQ: which is more similar?
        options = [(sim_img, 0), (diff_img, 1)]  # 0 = correct (similar)
        indices = [0, 1]
        rng.shuffle(indices)
        correct_pos = indices.index(0)
        correct_letter = OPTION_LETTERS[correct_pos]
        shuffled_imgs = [options[i][0] for i in indices]

        composite = create_reference_and_options(
            reference=ref_img,
            options=shuffled_imgs,
            ref_label="Reference Image",
            cell_size=300,
        )

        out_filename = f"vissim_{len(records):04d}.jpg"
        out_path = os.path.join(args.output_dir, out_filename)
        save_image(composite, out_path)

        formatted_choices = [f"({OPTION_LETTERS[j]})" for j in range(2)]
        choices_str = " ".join(formatted_choices)
        question = (
            f"Which image is more visually similar to the reference image shown above? "
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
            logger.info(f"Processed {len(records)}/{args.n_samples} visual similarity samples")

    write_jsonl(records, args.output_jsonl)
    logger.info(f"Built {len(records)} visual similarity MCQ samples.")


if __name__ == "__main__":
    main()
