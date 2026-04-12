#!/usr/bin/env python3
"""Build LIVR Art Style MCQ dataset from ArtBench-10.

Binary style classification: given a reference painting, identify which
of two candidate paintings shares its art style.

Usage:
    python scripts/livr/build_art_style.py
"""

import argparse
import logging
import os
import random
import sys

from datasets import load_from_disk
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from livr_common import (
    create_reference_and_options,
    logger,
    make_livr_record,
    save_image,
    shuffle_choices,
    write_jsonl,
)

TASK_NAME = "livr_art_style"
N_SAMPLES = 1200


def main() -> None:
    """Build art style binary MCQ dataset."""
    parser = argparse.ArgumentParser(description="Build LIVR art style MCQs.")
    parser.add_argument(
        "--source-dir",
        default="/outputs/livr_sources/artbench/dataset",
        help="Path to saved ArtBench-10 HF dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="/outputs/image_base/livr/art_style",
        help="Output directory for composite images.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/outputs/livr_data/livr_art_style.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    logger.info(f"Loading ArtBench-10 from {args.source_dir}")
    ds = load_from_disk(args.source_dir)
    logger.info(f"Loaded {len(ds)} total samples")

    # Group by style — ArtBench-10 uses 'prompt' field like "a post-impressionism painting"
    style_to_indices: dict[str, list[int]] = {}
    for i, sample in enumerate(ds):
        style = sample.get("style", sample.get("label", None))
        if style is None:
            # Extract style from prompt field (e.g., "a post-impressionism painting" -> "post-impressionism")
            prompt = sample.get("prompt", "")
            if prompt:
                # Remove "a " prefix and " painting" suffix
                style = prompt.replace("a ", "").replace(" painting", "").strip()
            else:
                continue
        if isinstance(style, int):
            style = str(style)
        if not style:
            continue
        style_to_indices.setdefault(style, []).append(i)

    styles = list(style_to_indices.keys())
    logger.info(f"Found {len(styles)} styles: {styles}")

    if len(styles) < 2:
        logger.error("Need at least 2 styles for binary classification.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    records = []

    for idx in range(args.n_samples):
        # Pick reference style and a different style
        ref_style = rng.choice(styles)
        neg_style = rng.choice([s for s in styles if s != ref_style])

        ref_indices = style_to_indices[ref_style]
        neg_indices = style_to_indices[neg_style]

        if len(ref_indices) < 2 or len(neg_indices) < 1:
            continue

        # Pick reference, positive (same style), negative (different style)
        ref_idx, pos_idx = rng.sample(ref_indices, 2)
        neg_idx = rng.choice(neg_indices)

        ref_img = ds[ref_idx].get("image")
        pos_img = ds[pos_idx].get("image")
        neg_img = ds[neg_idx].get("image")

        if not all(isinstance(im, Image.Image) for im in [ref_img, pos_img, neg_img]):
            continue

        # Shuffle positive and negative
        correct_letter, formatted_choices, ordered = shuffle_choices(
            f"Same style as reference ({ref_style})",
            [f"Different style ({neg_style})"],
            rng,
        )

        # Create composite: reference on top, two options below
        if ordered[0].startswith("Same"):
            option_images = [pos_img, neg_img]
        else:
            option_images = [neg_img, pos_img]

        composite = create_reference_and_options(
            reference=ref_img,
            options=option_images,
            ref_label="Reference Painting",
            cell_size=300,
        )

        img_filename = f"art_style_{idx:04d}.jpg"
        img_path = os.path.join(args.output_dir, img_filename)
        save_image(composite, img_path)

        choices_str = " ".join(formatted_choices)
        question = (
            f"Which painting shares the same art style as the reference painting? "
            f"{choices_str}"
        )

        record = make_livr_record(
            question=question,
            ground_truth=correct_letter,
            choices=formatted_choices,
            image_path=img_path,
            dataset_name=TASK_NAME,
        )
        records.append(record)

        if (idx + 1) % 200 == 0:
            logger.info(f"Processed {idx + 1}/{args.n_samples} art style samples")

    write_jsonl(records, args.output_jsonl)
    logger.info(f"Built {len(records)} art style MCQ samples.")


if __name__ == "__main__":
    main()
