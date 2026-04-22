#!/usr/bin/env python3
"""Build LIVR Counting MCQ dataset from PixMo-Count.

Filters for count range [2, 10], generates 4-way numeric MCQs.
Single-image task (no composite needed).

Usage:
    python scripts/livr/build_counting.py
"""

import argparse
import io
import logging
import os
import random
import sys

import requests
from datasets import load_from_disk
from PIL import Image

# Add parent to path for livr_common
sys.path.insert(0, os.path.dirname(__file__))
from livr_common import (
    generate_numeric_distractors,
    logger,
    make_livr_record,
    save_image,
    shuffle_choices,
    write_jsonl,
)

TASK_NAME = "livr_counting"
N_SAMPLES = 1200  # Generate extra buffer for decontamination
MIN_COUNT = 2
MAX_COUNT = 10


def main() -> None:
    """Build counting MCQ dataset from PixMo-Count."""
    parser = argparse.ArgumentParser(description="Build LIVR counting MCQs.")
    parser.add_argument(
        "--source-dir",
        default="/outputs/livr_sources/pixmo_count/dataset",
        help="Path to saved PixMo-Count HF dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="/outputs/image_base/livr/counting",
        help="Output directory for images.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/outputs/livr_data/livr_counting.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    logger.info(f"Loading PixMo-Count from {args.source_dir}")
    ds = load_from_disk(args.source_dir)
    logger.info(f"Loaded {len(ds)} total samples")

    # Filter for count range
    filtered = []
    for i, sample in enumerate(ds):
        count = sample.get("count", sample.get("number", None))
        if count is None:
            # Try to parse from label/answer fields
            for key in ["label", "answer", "target"]:
                if key in sample:
                    try:
                        count = int(sample[key])
                        break
                    except (ValueError, TypeError):
                        continue
        if count is not None and MIN_COUNT <= count <= MAX_COUNT:
            filtered.append((i, sample, count))

    logger.info(f"Filtered to {len(filtered)} samples with count in [{MIN_COUNT}, {MAX_COUNT}]")

    if len(filtered) < args.n_samples:
        logger.warning(
            f"Only {len(filtered)} samples available, requested {args.n_samples}"
        )

    rng.shuffle(filtered)
    selected = filtered[: args.n_samples]

    os.makedirs(args.output_dir, exist_ok=True)
    records = []

    for idx, (orig_idx, sample, count) in enumerate(selected):
        # Get image — PixMo-Count uses image_url (images not embedded)
        img = sample.get("image")
        if img is None or not isinstance(img, Image.Image):
            image_url = sample.get("image_url", "")
            if not image_url:
                logger.warning(f"Sample {orig_idx} has no image or URL, skipping.")
                continue
            try:
                resp = requests.get(image_url, timeout=10)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content))
            except Exception as e:
                logger.warning(f"Sample {orig_idx} URL download failed: {e}")
                continue

        if img.mode != "RGB":
            img = img.convert("RGB")

        # Save image
        img_filename = f"counting_{idx:04d}.jpg"
        img_path = os.path.join(args.output_dir, img_filename)
        save_image(img, img_path)

        # Generate MCQ
        distractors = generate_numeric_distractors(
            count, n_distractors=3, rng=rng, min_val=1, max_val=15
        )

        correct_letter, formatted_choices, _ = shuffle_choices(
            str(count), [str(d) for d in distractors], rng
        )

        # Use the label (object type) for a more specific question
        obj_label = sample.get("label", "objects")
        choices_str = " ".join(formatted_choices)
        question = f"How many {obj_label} are in the image? {choices_str}"

        record = make_livr_record(
            question=question,
            ground_truth=correct_letter,
            choices=formatted_choices,
            image_path=img_path,
            dataset_name=TASK_NAME,
        )
        records.append(record)

        if (idx + 1) % 200 == 0:
            logger.info(f"Processed {idx + 1}/{len(selected)} counting samples")

    write_jsonl(records, args.output_jsonl)
    logger.info(f"Built {len(records)} counting MCQ samples.")


if __name__ == "__main__":
    main()
