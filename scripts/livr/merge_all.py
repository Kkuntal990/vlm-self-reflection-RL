#!/usr/bin/env python3
"""Merge all 10 LIVR task JSONLs into a single dataset file.

Concatenates per-task JSONLs, shuffles, and writes the merged output.
Optionally trims each task to exactly N samples.

Usage:
    python scripts/livr/merge_all.py
    python scripts/livr/merge_all.py --max-per-task 1000
"""

import argparse
import logging
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
from livr_common import load_jsonl, logger, write_jsonl

TASK_FILES = [
    "livr_counting.jsonl",
    "livr_jigsaw.jsonl",
    "livr_object_localization.jsonl",
    "livr_visual_correspondence.jsonl",
    "livr_art_style.jsonl",
    "livr_semantic_correspondence.jsonl",
    "livr_functional_correspondence.jsonl",
    "livr_relative_reflectance.jsonl",
    "livr_relative_depth.jsonl",
    "livr_visual_similarity.jsonl",
]


def main() -> None:
    """Merge all LIVR task JSONLs into one file."""
    parser = argparse.ArgumentParser(description="Merge LIVR task JSONLs.")
    parser.add_argument(
        "--input-dir",
        default="/outputs/livr_data",
        help="Directory containing per-task JSONL files.",
    )
    parser.add_argument(
        "--output",
        default="/outputs/livr_data/livr_perception_mcq.jsonl",
        help="Output merged JSONL path.",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=1000,
        help="Maximum samples per task (0 = no limit).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    all_records = []

    for task_file in TASK_FILES:
        path = os.path.join(args.input_dir, task_file)
        if not os.path.exists(path):
            logger.warning(f"Missing: {path} — skipping.")
            continue

        records = load_jsonl(path)
        task_name = records[0]["dataset_name"] if records else task_file

        if args.max_per_task > 0 and len(records) > args.max_per_task:
            rng.shuffle(records)
            records = records[: args.max_per_task]
            logger.info(f"{task_name}: trimmed to {len(records)} samples")
        else:
            logger.info(f"{task_name}: {len(records)} samples")

        all_records.extend(records)

    rng.shuffle(all_records)
    write_jsonl(all_records, args.output)

    # Print summary
    task_counts: dict[str, int] = {}
    for rec in all_records:
        dn = rec.get("dataset_name", "unknown")
        task_counts[dn] = task_counts.get(dn, 0) + 1

    logger.info(f"\n=== Merge Summary ===")
    logger.info(f"Total samples: {len(all_records)}")
    for task, count in sorted(task_counts.items()):
        logger.info(f"  {task}: {count}")


if __name__ == "__main__":
    main()
