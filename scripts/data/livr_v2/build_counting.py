#!/usr/bin/env python3
"""livr-v2 Counting builder — strict Appendix A replication.

Per Appendix A of the LIVR paper (arXiv:2512.21218):

  Source     : PixMo-Count (`allenai/pixmo-count` on HuggingFace)
  Format     : OPEN-ENDED (model generates an integer count, no MCQ)
  Filter     : ground-truth counts c in {2,3,...,10}
  Train      : 1,000-example subset sampled from the PixMo-Count
               TRAIN split, "approximately uniformly represented" across
               c in {2..10}, after URL-validity filtering.
  Validation : the OFFICIAL PixMo-Count validation split, after URL
               validity filtering (paper reports 534 surviving examples).
  Test       : the OFFICIAL PixMo-Count test split, after URL validity
               filtering (paper reports 528 surviving examples).
  Dedup      : "CLIP embeddings together with perceptual hashing and
               SSIM-based image similarity" between train+val images
               and the test images — drop near-duplicates.

Output schema is open-ended:
    {
      "question": "How many <object> are in the image? Provide a single integer answer.",
      "ground_truth": "<integer>",
      "answer_type": "counting",
      "choices": "",
      "images": ["<png path>"],
      "dataset_name": "livr_counting",
      "split": "train"|"val"|"test"
    }
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import requests
from datasets import load_from_disk
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dedup import full_dedup  # noqa: E402
from livr_common import (  # noqa: E402
    log_task_banner,
    logger,
    save_image,
    write_jsonl,
)

TASK_NAME = "livr_counting"
MIN_COUNT = 2
MAX_COUNT = 10
N_TRAIN = 1000  # paper Appendix A
N_VAL = 534  # paper Appendix A — official PixMo val after URL filter
N_TEST = 528  # paper Appendix A — official PixMo test after URL filter

# Appendix A: "counts are approximately uniformly represented".
# 9 buckets c=2..10 -> ~111 per bucket for N_TRAIN=1000 (rounding):
PER_BUCKET_TRAIN = N_TRAIN // (MAX_COUNT - MIN_COUNT + 1)  # = 111


def _get_count(sample: dict) -> int | None:
    """Extract integer count from a PixMo-Count sample row."""
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


def _open_image(sample: dict) -> Image.Image | None:
    """Get the PIL image for a PixMo sample (embedded or via URL).

    PixMo-Count samples carry images by URL; transient fetch failures
    or 404s are common, hence the URL-validity filter mentioned in
    Appendix A.
    """
    img = sample.get("image")
    if isinstance(img, Image.Image):
        return img
    url = sample.get("image_url", "")
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))
    except Exception as e:
        logger.warning("URL fetch failed: %s (%s)", url, e)
        return None


def _materialize(samples_with_count, out_dir: Path, prefix: str):
    """Download + save images for a list of (orig_idx, sample, count) tuples.

    Returns a list of (out_path, count, sample) for the successfully saved
    images. Carrying `sample` through means `_build_records` can read the
    object label off the original PixMo row without trying to re-align two
    list iterators (which silently misaligns whenever any URL fetch fails).
    Skips samples whose URL is invalid (this is the "removing bad URLs"
    step from the paper — drives the 534/528 numbers).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[tuple[Path, int, dict]] = []
    for idx, (_, sample, count) in enumerate(samples_with_count):
        img = _open_image(sample)
        if img is None:
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")
        out_path = out_dir / f"{prefix}_{idx:05d}.png"
        save_image(img, str(out_path), format="PNG")
        saved.append((out_path, count, sample))
        if (idx + 1) % 200 == 0:
            logger.info("  [%s] saved %d/%d", prefix, len(saved), idx + 1)
    return saved


def _filter_count_range(split):
    """Yield (orig_idx, sample, count) for rows with c in {2..10}."""
    out = []
    for i, sample in enumerate(split):
        c = _get_count(sample)
        if c is not None and MIN_COUNT <= c <= MAX_COUNT:
            out.append((i, sample, c))
    return out


def _uniform_sample_train(filtered, rng) -> list[tuple[int, dict, int]]:
    """Sample ~uniformly across c=2..10. We sample more than 1000 so
    after image-fetch failures we still hit 1000 saved images."""
    by_count: dict[int, list] = defaultdict(list)
    for r in filtered:
        by_count[r[2]].append(r)
    for c in by_count:
        rng.shuffle(by_count[c])
    # Oversample by 50% to absorb URL-failure attrition.
    target_per_bucket = int(PER_BUCKET_TRAIN * 1.5)
    pool: list = []
    for c in range(MIN_COUNT, MAX_COUNT + 1):
        bucket = by_count.get(c, [])
        pool.extend(bucket[:target_per_bucket])
        logger.info(
            "  count=%d -> pool size=%d (had %d)",
            c,
            min(len(bucket), target_per_bucket),
            len(bucket),
        )
    rng.shuffle(pool)
    return pool


def _build_records(saved: list[tuple[Path, int, dict]], split_name: str) -> list[dict]:
    """Build open-ended counting records from (path, count, sample) triples."""
    recs = []
    for img_path, count, sample in saved:
        obj_label = sample.get("label", "objects") if sample else "objects"
        question = f"How many {obj_label} are in the image? Provide a single integer answer."
        recs.append(
            {
                "question": question,
                "ground_truth": str(count),
                "answer_type": "counting",  # open-ended numeric — fuzzy partial credit
                "choices": "",
                "images": [str(img_path)],
                "dataset_name": TASK_NAME,
                "split": split_name,
            }
        )
    return recs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pixmo-dir",
        default="/outputs/livr_v2_sources/pixmo_count",
        help="Path to PixMo-Count HF dataset (load_from_disk-able). Must "
        "contain train + validation + test splits.",
    )
    parser.add_argument("--output-dir", default="/outputs/livr_v2/image_base/counting")
    parser.add_argument("--output-jsonl-prefix", default="/outputs/livr_v2/data/counting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-dedup",
        action="store_true",
        help="Skip the CLIP+pHash+SSIM dedup vs official test images.",
    )
    parser.add_argument("--clip-device", default="cuda")
    args = parser.parse_args()

    log_task_banner(TASK_NAME)
    rng = random.Random(args.seed)

    ds = load_from_disk(args.pixmo_dir)
    # `load_from_disk` returns a `Dataset` for a single-split snapshot or a
    # `DatasetDict` for a multi-split one. Older snapshots saved on the PVC
    # may be a single split (just `train`), in which case we pull `validation`
    # and `test` directly from HuggingFace.
    if hasattr(ds, "keys"):  # DatasetDict
        train_split = ds["train"] if "train" in ds.keys() else ds
        val_split = ds.get("validation") if "validation" in ds.keys() else None
        test_split = ds.get("test") if "test" in ds.keys() else None
    else:  # plain Dataset (single split)
        train_split = ds
        val_split = None
        test_split = None
    # Fallback: download missing splits from HF directly.
    if val_split is None or test_split is None:
        from datasets import load_dataset

        logger.info("PixMo-Count snapshot lacks validation/test splits; loading from HF.")
        if val_split is None:
            val_split = load_dataset("allenai/pixmo-count", split="validation")
        if test_split is None:
            test_split = load_dataset("allenai/pixmo-count", split="test")
    logger.info(
        "PixMo-Count: train=%d  validation=%d  test=%d",
        len(train_split),
        len(val_split),
        len(test_split),
    )

    # 1) filter to c in {2..10}
    train_filt = _filter_count_range(train_split)
    val_filt = _filter_count_range(val_split)
    test_filt = _filter_count_range(test_split)
    logger.info(
        "After count-range filter: train=%d  val=%d  test=%d",
        len(train_filt),
        len(val_filt),
        len(test_filt),
    )

    out_root = Path(args.output_dir)

    # 2) uniform sample for TRAIN, materialize all three splits
    train_pool = _uniform_sample_train(train_filt, rng)
    rng.shuffle(val_filt)
    rng.shuffle(test_filt)

    logger.info("Materializing test split...")
    test_saved = _materialize(test_filt[: N_TEST + 100], out_root / "test", "counting_test")
    logger.info("Materializing val split...")
    val_saved = _materialize(val_filt[: N_VAL + 100], out_root / "val", "counting_val")
    logger.info("Materializing train pool...")
    train_saved = _materialize(train_pool, out_root / "train", "counting_train")

    # 3) Dedup train+val vs test (per Appendix A: CLIP + pHash + SSIM).
    if not args.skip_dedup and test_saved:
        test_paths = [p for p, _, _ in test_saved]
        candidates = [p for p, _, _ in train_saved] + [p for p, _, _ in val_saved]
        # Appendix A: "CLIP embeddings together with perceptual hashing
        # and SSIM-based image similarity" — full_dedup ANDs all three
        # so a candidate is dropped only if all three checks agree it's
        # a near-duplicate.
        keep_set = full_dedup(
            candidates=candidates,
            exclude=test_paths,
            clip_sim_thresh=0.95,
            phash_thresh=8,
            ssim_thresh=0.95,
            device=args.clip_device,
        )
        train_saved = [(p, c, s) for p, c, s in train_saved if p in keep_set]
        val_saved = [(p, c, s) for p, c, s in val_saved if p in keep_set]
        logger.info(
            "After dedup: train=%d  val=%d",
            len(train_saved),
            len(val_saved),
        )

    # 4) Trim to exactly the paper's split sizes (1000 / 534 / 528)
    if len(train_saved) >= N_TRAIN:
        # Re-balance to keep counts roughly uniform after dedup attrition.
        by_count: dict[int, list] = defaultdict(list)
        for p, c, s in train_saved:
            by_count[c].append((p, c, s))
        for c in by_count:
            rng.shuffle(by_count[c])
        per_bucket = N_TRAIN // (MAX_COUNT - MIN_COUNT + 1)
        balanced: list[tuple[Path, int, dict]] = []
        for c in range(MIN_COUNT, MAX_COUNT + 1):
            balanced.extend(by_count.get(c, [])[:per_bucket])
        # Top up if a bucket was short.
        if len(balanced) < N_TRAIN:
            extras = [item for c in by_count for item in by_count[c][per_bucket:]]
            rng.shuffle(extras)
            balanced += extras[: N_TRAIN - len(balanced)]
        train_saved = balanced[:N_TRAIN]
    val_saved = val_saved[:N_VAL]
    test_saved = test_saved[:N_TEST]

    # 5) Build + write records.
    train_records = _build_records(train_saved, "train")
    val_records = _build_records(val_saved, "val")
    test_records = _build_records(test_saved, "test")

    write_jsonl(train_records, f"{args.output_jsonl_prefix}_train.jsonl")
    write_jsonl(val_records, f"{args.output_jsonl_prefix}_val.jsonl")
    write_jsonl(test_records, f"{args.output_jsonl_prefix}_test.jsonl")
    logger.info(
        "Counting build done: train=%d  val=%d  test=%d",
        len(train_records),
        len(val_records),
        len(test_records),
    )

    # Per-bucket distribution sanity log.
    train_dist: dict[int, int] = defaultdict(int)
    for r in train_records:
        train_dist[int(r["ground_truth"])] += 1
    logger.info("Train count distribution: %s", dict(sorted(train_dist.items())))


if __name__ == "__main__":
    main()
