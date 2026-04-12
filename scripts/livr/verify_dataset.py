#!/usr/bin/env python3
"""Verify LIVR perception MCQ dataset integrity.

Checks field presence, image existence, answer validity, and
answer distribution uniformity.

Usage:
    python scripts/livr/verify_dataset.py
    python scripts/livr/verify_dataset.py --check-images
"""

import argparse
import logging
import os
import sys
from collections import Counter

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from livr_common import OPTION_LETTERS, load_jsonl, logger

REQUIRED_FIELDS = ["question", "ground_truth", "answer_type", "choices", "images", "dataset_name"]


def main() -> None:
    """Verify LIVR dataset integrity."""
    parser = argparse.ArgumentParser(description="Verify LIVR dataset.")
    parser.add_argument(
        "--input",
        default="/outputs/livr_data/livr_perception_mcq.jsonl",
        help="Path to merged JSONL.",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Actually open and verify each image (slower).",
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    logger.info(f"Loaded {len(records)} records from {args.input}")

    errors = []
    warnings = []

    # Per-task stats
    task_counts: Counter = Counter()
    task_answer_dist: dict[str, Counter] = {}
    broken_images = []

    for i, rec in enumerate(records):
        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in rec or not rec[field]:
                errors.append(f"Record {i}: missing field '{field}'")

        # Check answer_type
        if rec.get("answer_type") != "mcq":
            errors.append(f"Record {i}: answer_type is '{rec.get('answer_type')}', expected 'mcq'")

        # Check ground_truth is a valid letter or (letter) format
        gt = rec.get("ground_truth", "")
        # Accept both "A" and "(A)" formats
        gt_letter = gt.strip("()")
        if gt_letter not in OPTION_LETTERS[:6]:
            errors.append(f"Record {i}: ground_truth '{gt}' is not a valid option letter")

        # Check ground_truth letter appears in choices
        choices = rec.get("choices", "")
        if gt_letter and f"({gt_letter})" not in choices:
            errors.append(f"Record {i}: ground_truth '({gt})' not found in choices '{choices}'")

        # Check images
        images = rec.get("images", [])
        if not images:
            errors.append(f"Record {i}: empty images list")
        elif args.check_images:
            img_path = images[0]
            if not os.path.exists(img_path):
                broken_images.append(img_path)
                errors.append(f"Record {i}: image not found: {img_path}")
            else:
                try:
                    img = Image.open(img_path)
                    img.verify()
                except Exception as e:
                    broken_images.append(img_path)
                    errors.append(f"Record {i}: image corrupt: {img_path}: {e}")

        # Track stats
        dataset_name = rec.get("dataset_name", "unknown")
        task_counts[dataset_name] += 1
        if dataset_name not in task_answer_dist:
            task_answer_dist[dataset_name] = Counter()
        task_answer_dist[dataset_name][gt] += 1

    # Check answer distribution uniformity
    for task, dist in task_answer_dist.items():
        total = sum(dist.values())
        n_options = len(dist)
        if n_options < 2:
            warnings.append(f"{task}: only {n_options} distinct answers")
            continue
        expected = total / n_options
        for letter, count in dist.items():
            deviation = abs(count - expected) / expected
            if deviation > 0.3:
                warnings.append(
                    f"{task}: answer '{letter}' has {count}/{total} "
                    f"({deviation:.0%} deviation from expected {expected:.0f})"
                )

    # Print report
    print("\n" + "=" * 60)
    print("LIVR DATASET VERIFICATION REPORT")
    print("=" * 60)

    print(f"\nTotal records: {len(records)}")
    print(f"\nPer-task counts:")
    for task, count in sorted(task_counts.items()):
        status = "OK" if count >= 1000 else "LOW"
        print(f"  {task}: {count} [{status}]")

    print(f"\nAnswer distribution per task:")
    for task, dist in sorted(task_answer_dist.items()):
        total = sum(dist.values())
        dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
        print(f"  {task} (n={total}): {dist_str}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
    else:
        print(f"\nNo errors found.")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    if broken_images:
        print(f"\nBroken images ({len(broken_images)}):")
        for p in broken_images[:10]:
            print(f"  - {p}")

    print("\n" + "=" * 60)

    # Exit with error code if critical issues found
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
