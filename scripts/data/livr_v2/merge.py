#!/usr/bin/env python3
"""Merge per-task LIVR-v2 JSONLs into the multi-task corpus.

Per Appendix A of arXiv:2512.21218:

  Multi-task corpus = 6 tasks x 1000 train each = 6000 total
  Tasks: Counting, Object_Localization, Visual_Correspondence,
         Semantic_Correspondence, Functional_Correspondence,
         Relative_Reflectance

We omit Functional_Correspondence (FunKPoint not publicly available),
yielding a 5-task / 5000-example multi-task corpus until FunKPoint
arrives via direct author request.

Usage:
    python scripts/data/livr_v2/merge.py \\
        --data-dir /outputs/livr_v2/data \\
        --output-prefix /outputs/livr_v2/data/livr_v2_5task
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Paper's multi-task setup: 6 tasks. We have 5 (FunKPoint unavailable).
MULTI_TASK_TASKS = [
    "counting",
    "object_localization",
    "visual_correspondence",
    "semantic_correspondence",
    "relative_reflectance",
]

# Single-task ablation (tasks paper omits because base is already strong,
# but useful as side experiments).
SINGLE_TASK_ALSO = [
    "jigsaw",
    "art_style",
    "visual_similarity",
]


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records -> %s", len(records), path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/outputs/livr_v2/data")
    parser.add_argument("--output-prefix", default="/outputs/livr_v2/data/livr_v2_5task")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)

    # Build multi-task train (5 tasks x 1000) and val (5 tasks x 250).
    multi_train: list[dict] = []
    multi_val: list[dict] = []
    manifest = {"multi_task_tasks": MULTI_TASK_TASKS, "single_task_also": SINGLE_TASK_ALSO}

    for task in MULTI_TASK_TASKS:
        train_path = data_dir / f"{task}_train.jsonl"
        val_path = data_dir / f"{task}_val.jsonl"
        if not train_path.exists():
            logger.warning("Missing %s — skipping", train_path)
            continue
        train_recs = _read_jsonl(train_path)
        val_recs = _read_jsonl(val_path) if val_path.exists() else []
        manifest.setdefault("counts", {})[task] = {
            "train": len(train_recs),
            "val": len(val_recs),
        }
        logger.info("  %s: train=%d val=%d", task, len(train_recs), len(val_recs))
        multi_train.extend(train_recs)
        multi_val.extend(val_recs)

    rng.shuffle(multi_train)
    rng.shuffle(multi_val)
    _write_jsonl(multi_train, Path(f"{args.output_prefix}_train.jsonl"))
    _write_jsonl(multi_val, Path(f"{args.output_prefix}_val.jsonl"))

    # Also build a 9-task superset (for ablations) — INCLUDING the
    # single-task-only tasks (jigsaw, art_style, visual_similarity).
    full_train: list[dict] = list(multi_train)
    full_val: list[dict] = list(multi_val)
    for task in SINGLE_TASK_ALSO:
        train_path = data_dir / f"{task}_train.jsonl"
        val_path = data_dir / f"{task}_val.jsonl"
        if not train_path.exists():
            logger.warning("Missing %s (single-task additional) — skipping", train_path)
            continue
        full_train.extend(_read_jsonl(train_path))
        if val_path.exists():
            full_val.extend(_read_jsonl(val_path))
    rng.shuffle(full_train)
    rng.shuffle(full_val)
    full_prefix = args.output_prefix.replace("_5task", "_8task")
    if full_prefix == args.output_prefix:
        full_prefix = args.output_prefix + "_full"
    _write_jsonl(full_train, Path(f"{full_prefix}_train.jsonl"))
    _write_jsonl(full_val, Path(f"{full_prefix}_val.jsonl"))

    manifest["multi_task_train"] = len(multi_train)
    manifest["multi_task_val"] = len(multi_val)
    manifest["full_train"] = len(full_train)
    manifest["full_val"] = len(full_val)
    manifest_path = Path(f"{args.output_prefix}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest -> %s", manifest_path)


if __name__ == "__main__":
    main()
