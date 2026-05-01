#!/usr/bin/env python3
"""Merge per-task LIVR-v2 JSONLs into the multi-task corpus.

Per Appendix A of arXiv:2512.21218:

  Full corpus     = 9 tasks x 1000 train each = 9000 total
  Multi-task abl. = 6 tasks x 1000 train each = 6000 total
                    (Counting, Object_Localization, Visual_Correspondence,
                     Semantic_Correspondence, Functional_Correspondence,
                     Relative_Reflectance — omits Jigsaw, Art_Style,
                     Visual_Similarity since base accuracy is already
                     high on those.)

Outputs (under --output-prefix):
  *_9task_train.jsonl  — all 9 tasks merged (8000 if FunKPoint missing)
  *_6task_train.jsonl  — paper's multi-task ablation (5000 if FunKPoint
                         missing)
  *_manifest.json      — per-task counts.

Usage:
    python scripts/data/livr_v2/merge.py \\
        --data-dir /outputs/livr_v2/data \\
        --output-prefix /outputs/livr_v2/data/livr_v2
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

# Paper's full single-task pool: 9 tasks x 1000 = 9000 train examples
# (Counting, Jigsaw, Object_Localization, Visual_Correspondence,
#  Semantic_Correspondence, Functional_Correspondence, Relative_Reflectance,
#  Art_Style, Visual_Similarity).
# We omit Functional_Correspondence (FunKPoint not publicly available),
# yielding 8 tasks x 1000 = 8000 total.
ALL_TASKS = [
    "counting",
    "jigsaw",
    "object_localization",
    "visual_correspondence",
    "semantic_correspondence",
    "functional_correspondence",
    "relative_reflectance",
    "art_style",
    "visual_similarity",
]

# Paper's specific 6-task multi-task ablation (Appendix A "Multi-Task Setup"
# omits Jigsaw, Art_Style, Visual_Similarity since base is already strong).
ABLATION_6TASK = [
    "counting",
    "object_localization",
    "visual_correspondence",
    "semantic_correspondence",
    "functional_correspondence",
    "relative_reflectance",
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


def _merge_tasks(task_list, data_dir, rng):
    train: list[dict] = []
    val: list[dict] = []
    counts: dict[str, dict[str, int]] = {}
    for task in task_list:
        train_path = data_dir / f"{task}_train.jsonl"
        val_path = data_dir / f"{task}_val.jsonl"
        if not train_path.exists():
            logger.warning("Missing %s - skipping", train_path)
            continue
        train_recs = _read_jsonl(train_path)
        val_recs = _read_jsonl(val_path) if val_path.exists() else []
        counts[task] = {"train": len(train_recs), "val": len(val_recs)}
        logger.info("  %s: train=%d val=%d", task, len(train_recs), len(val_recs))
        train.extend(train_recs)
        val.extend(val_recs)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="/outputs/livr_v2/data")
    parser.add_argument("--output-prefix", default="/outputs/livr_v2/data/livr_v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_dir = Path(args.data_dir)
    manifest = {}

    # Primary corpus: all 9 tasks (Counting, Jigsaw, Object_Localization,
    # Visual_Correspondence, Semantic_Correspondence, Functional_Correspondence,
    # Relative_Reflectance, Art_Style, Visual_Similarity).
    logger.info("=== Merging all-tasks corpus (9 tasks) ===")
    all_train, all_val, all_counts = _merge_tasks(ALL_TASKS, data_dir, rng)
    _write_jsonl(all_train, Path(f"{args.output_prefix}_9task_train.jsonl"))
    _write_jsonl(all_val, Path(f"{args.output_prefix}_9task_val.jsonl"))
    manifest["9task"] = {
        "tasks": ALL_TASKS,
        "train": len(all_train),
        "val": len(all_val),
        "per_task": all_counts,
    }

    # Ablation corpus: paper's 6-task multi-task ablation.
    logger.info("=== Merging ablation 6-task corpus ===")
    abl_train, abl_val, abl_counts = _merge_tasks(ABLATION_6TASK, data_dir, rng)
    _write_jsonl(abl_train, Path(f"{args.output_prefix}_6task_train.jsonl"))
    _write_jsonl(abl_val, Path(f"{args.output_prefix}_6task_val.jsonl"))
    manifest["6task"] = {
        "tasks": ABLATION_6TASK,
        "train": len(abl_train),
        "val": len(abl_val),
        "per_task": abl_counts,
    }

    manifest_path = Path(f"{args.output_prefix}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote manifest -> %s", manifest_path)
    logger.info(
        "Done. Corpora: 8task=%d train, 5task=%d train",
        len(all_train),
        len(abl_train),
    )


if __name__ == "__main__":
    main()
