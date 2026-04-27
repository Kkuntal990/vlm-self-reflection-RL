#!/usr/bin/env python3
"""Filter the LIVR training JSONL using a precomputed difficulty bucket file.

Joins the original training JSONL with the bucketed JSONL from
`difficulty_buckets.py` (matched by `sample_index`) and drops rows whose
bucket is in the --drop list.

Usage:
    uv run python scripts/data/filter_by_difficulty.py \\
        --dataset /outputs/livr_data/livr_perception_mcq.jsonl \\
        --difficulty /outputs/livr_data/livr_difficulty_a1_bucketed.jsonl \\
        --output /outputs/livr_data/livr_perception_mcq_filtered.jsonl \\
        --drop trivial brick_wall
"""

import argparse
import json
import logging
import os
import sys
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Original training JSONL")
    parser.add_argument(
        "--difficulty", required=True, help="Bucketed JSONL from difficulty_buckets.py"
    )
    parser.add_argument("--output", required=True, help="Filtered training JSONL output")
    parser.add_argument(
        "--drop",
        nargs="+",
        default=["trivial", "brick_wall"],
        help="Bucket names to drop (default: trivial brick_wall)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Build sample_index -> bucket map
    drop_set: set[int] = set()
    bucket_counts: Counter[str] = Counter()
    drop_buckets = set(args.drop)
    with open(args.difficulty, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = row.get("sample_index")
            bucket = row.get("difficulty_bucket", "unknown")
            bucket_counts[bucket] += 1
            if isinstance(idx, int) and bucket in drop_buckets:
                drop_set.add(idx)

    logger.info("Bucket counts in difficulty file: %s", dict(bucket_counts))
    logger.info("Dropping %d sample_indices", len(drop_set))

    # Stream-filter the original dataset
    n_in = 0
    n_out = 0
    n_dropped = 0
    n_unmatched = 0
    drop_by_bucket: Counter[str] = Counter()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with (
        open(args.dataset, encoding="utf-8") as in_f,
        open(args.output, "w", encoding="utf-8") as out_f,
    ):
        for i, line in enumerate(in_f):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            # Match by 0-based index — `load_self_reflection_dataset` assigns
            # `sample_index = i` from the same enumeration order.
            if i in drop_set:
                n_dropped += 1
                # Track per-task drop for reporting (best effort)
                try:
                    sample = json.loads(line)
                    drop_by_bucket[sample.get("dataset_name", "unknown")] += 1
                except json.JSONDecodeError:
                    pass
                continue
            out_f.write(line + "\n")
            n_out += 1

    if n_in == 0:
        logger.warning("Input dataset is empty: %s", args.dataset)
        n_unmatched = 0
    else:
        n_unmatched = n_in - n_dropped - n_out

    logger.info("Input rows:    %d", n_in)
    logger.info("Dropped:       %d (%.1f%%)", n_dropped, 100.0 * n_dropped / max(n_in, 1))
    logger.info("Kept (output): %d (%.1f%%)", n_out, 100.0 * n_out / max(n_in, 1))
    if n_unmatched:
        logger.warning("Unmatched rows: %d", n_unmatched)
    logger.info("Drop counts by task: %s", dict(drop_by_bucket))
    logger.info("Wrote filtered dataset: %s", args.output)


if __name__ == "__main__":
    main()
