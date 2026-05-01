#!/usr/bin/env python3
"""One-shot repair for livr-v2 counting JSONLs whose object labels were
misaligned with their images.

Background: the original `build_counting.py` skipped failed URL fetches
inside `_materialize` but only appended successful saves; a separate
post-hoc loop in `main()` then walked the source list and the saved
list in lockstep, so every dropped fetch shifted every later object
label by one position. The result was, e.g., a wallet image asking
"How many people are in the image?".

The on-disk PNG filenames `counting_<split>_<idx>.png` already encode
the source-row index inside the deterministic `train_pool` / shuffled
`val_filt` / shuffled `test_filt`. With the same seed (42) we can
reproduce those source lists exactly and look up the correct label
from `train_pool[idx]["label"]`.

Usage on the helper pod:
    python scripts/data/livr_v2/repair_counting_labels.py \\
        --pixmo-dir /outputs/livr_v2_sources/pixmo_count \\
        --image-base-dir /outputs/livr_v2/image_base/counting \\
        --jsonl-prefix /outputs/livr_v2/data/counting
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_from_disk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

MIN_COUNT = 2
MAX_COUNT = 10
N_TRAIN = 1000
PER_BUCKET_TRAIN = N_TRAIN // (MAX_COUNT - MIN_COUNT + 1)
FILENAME_RE = re.compile(r"counting_(train|val|test)_(\d{5})\.png$")


def _get_count(sample: dict) -> int | None:
    count = sample.get("count")
    if count is None:
        for k in ("number", "label", "answer", "target"):
            if k in sample:
                try:
                    return int(sample[k])
                except (ValueError, TypeError):
                    continue
        return None
    try:
        return int(count)
    except (ValueError, TypeError):
        return None


def _filter_count_range(split):
    out = []
    for i, sample in enumerate(split):
        c = _get_count(sample)
        if c is not None and MIN_COUNT <= c <= MAX_COUNT:
            out.append((i, sample, c))
    return out


def _uniform_sample_train(filtered, rng):
    by_count: dict[int, list] = defaultdict(list)
    for r in filtered:
        by_count[r[2]].append(r)
    for c in by_count:
        rng.shuffle(by_count[c])
    target_per_bucket = int(PER_BUCKET_TRAIN * 1.5)
    pool: list = []
    for c in range(MIN_COUNT, MAX_COUNT + 1):
        bucket = by_count.get(c, [])
        pool.extend(bucket[:target_per_bucket])
    rng.shuffle(pool)
    return pool


def _load_pixmo_splits(pixmo_dir: Path):
    """Reproduce the same train/val/test source lists the builder used.

    The builder does:
      train_pool = _uniform_sample_train(_filter_count_range(train_split), rng)
      rng.shuffle(val_filt); rng.shuffle(test_filt)
    All three with the same `rng = random.Random(42)`.
    """
    ds = load_from_disk(str(pixmo_dir))
    if hasattr(ds, "keys"):
        train_split = ds["train"] if "train" in ds.keys() else ds
        val_split = ds.get("validation") if "validation" in ds.keys() else None
        test_split = ds.get("test") if "test" in ds.keys() else None
    else:
        train_split = ds
        val_split = None
        test_split = None
    if val_split is None or test_split is None:
        from datasets import load_dataset

        if val_split is None:
            val_split = load_dataset("allenai/pixmo-count", split="validation")
        if test_split is None:
            test_split = load_dataset("allenai/pixmo-count", split="test")

    train_filt = _filter_count_range(train_split)
    val_filt = _filter_count_range(val_split)
    test_filt = _filter_count_range(test_split)

    rng = random.Random(42)
    train_pool = _uniform_sample_train(train_filt, rng)
    rng.shuffle(val_filt)
    rng.shuffle(test_filt)
    return train_pool, val_filt, test_filt


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records -> %s", len(records), path)


def _repair_split(
    jsonl_path: Path,
    source_list: list,
    split_name: str,
) -> int:
    """Rewrite the `question` field on every record using the correct label
    recovered via the filename index. Returns number of records repaired.
    """
    if not jsonl_path.exists():
        logger.warning("Missing %s — skipping", jsonl_path)
        return 0
    records = _read_jsonl(jsonl_path)
    repaired = 0
    misses = 0
    for rec in records:
        img = rec.get("images", [None])[0] or rec.get("image_path")
        if not img:
            misses += 1
            continue
        m = FILENAME_RE.search(img)
        if not m or m.group(1) != split_name:
            misses += 1
            continue
        idx = int(m.group(2))
        if idx >= len(source_list):
            misses += 1
            continue
        _, sample, _count = source_list[idx]
        obj_label = sample.get("label", "objects") if sample else "objects"
        new_q = f"How many {obj_label} are in the image? Provide a single integer answer."
        if rec.get("question") != new_q:
            rec["question"] = new_q
            repaired += 1
    if misses:
        logger.warning("[%s] %d records had no recoverable filename idx", split_name, misses)
    _write_jsonl(records, jsonl_path)
    return repaired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixmo-dir", default="/outputs/livr_v2_sources/pixmo_count")
    parser.add_argument(
        "--jsonl-prefix",
        default="/outputs/livr_v2/data/counting",
        help="Prefix of the counting JSONLs to repair (without _train/_val/_test suffix).",
    )
    args = parser.parse_args()

    logger.info("Reproducing PixMo train/val/test source lists with seed=42...")
    train_pool, val_filt, test_filt = _load_pixmo_splits(Path(args.pixmo_dir))
    logger.info(
        "Source list sizes: train_pool=%d  val_filt=%d  test_filt=%d",
        len(train_pool),
        len(val_filt),
        len(test_filt),
    )

    for split, source in [
        ("train", train_pool),
        ("val", val_filt),
        ("test", test_filt),
    ]:
        path = Path(f"{args.jsonl_prefix}_{split}.jsonl")
        n = _repair_split(path, source, split)
        logger.info("[%s] questions repaired: %d", split, n)


if __name__ == "__main__":
    main()
