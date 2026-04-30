#!/usr/bin/env python3
"""livr-v2 Counting builder.

Per Appendix A of the LIVR paper (arXiv:2512.21218):
  * Source: PixMo-Count (allenai/pixmo-count)
  * Filter: counts c in {2..10}
  * Format: open-ended (model generates integer count)
  * Dedup: CLIP+pHash+SSIM vs official PixMo-Count test split
  * Splits: 1000 train / 534 val / 528 test

Usage:
    python scripts/data/livr_v2/build_counting.py \\
        --pixmo-dir /outputs/livr_v2_sources/pixmo_count \\
        --output-dir /outputs/livr_v2/image_base/counting \\
        --output-jsonl-prefix /outputs/livr_v2/data/counting
"""

from __future__ import annotations

import argparse
import io
import os
import random
import sys
from pathlib import Path

import requests
from datasets import load_from_disk
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from dedup import perceptual_dedup  # noqa: E402
from livr_common import (  # noqa: E402
    log_task_banner,
    logger,
    save_image,
    write_jsonl,
)

TASK_NAME = "livr_counting"
MIN_COUNT = 2
MAX_COUNT = 10
N_TRAIN = 1000
N_VAL = 534
N_TEST = 528


def _open_image(sample: dict) -> Image.Image | None:
    """Get the PIL image for a PixMo sample (embedded or via URL)."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pixmo-dir",
        default="/outputs/livr_v2_sources/pixmo_count",
        help="Path to PixMo-Count HF dataset (load_from_disk-able).",
    )
    parser.add_argument("--output-dir", default="/outputs/livr_v2/image_base/counting")
    parser.add_argument("--output-jsonl-prefix", default="/outputs/livr_v2/data/counting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-dedup",
        action="store_true",
        help="Skip the perceptual+SSIM dedup vs PixMo test (useful for debugging).",
    )
    args = parser.parse_args()

    log_task_banner(TASK_NAME)
    rng = random.Random(args.seed)

    ds = load_from_disk(args.pixmo_dir)
    # PixMo-Count has 'train' and 'test' splits.
    if "train" in ds:
        train_split = ds["train"]
        test_split = ds.get("test", None)
    else:
        train_split = ds
        test_split = None
    logger.info(
        "Loaded PixMo-Count (train=%d, test=%s)",
        len(train_split),
        len(test_split) if test_split else "n/a",
    )

    # Step 1: filter for count range on the source (will sample from this).
    def filt(split):
        out = []
        for i, sample in enumerate(split):
            count = sample.get("count")
            if count is None:
                for k in ("number", "label", "answer", "target"):
                    if k in sample:
                        try:
                            count = int(sample[k])
                            break
                        except (ValueError, TypeError):
                            continue
            if count is not None and MIN_COUNT <= count <= MAX_COUNT:
                out.append((i, sample, count))
        return out

    filtered = filt(train_split)
    logger.info("Filtered train: %d in count range [%d, %d]", len(filtered), MIN_COUNT, MAX_COUNT)

    # Step 2: produce material in excess so we can dedup + still hit the
    # split sizes. Need at least N_TRAIN+N_VAL = 1534 surviving the dedup.
    needed = N_TRAIN + N_VAL + N_TEST + 200
    rng.shuffle(filtered)
    material = filtered[:needed]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[tuple[Path, dict, int]] = []
    for idx, (orig_idx, sample, count) in enumerate(material):
        img = _open_image(sample)
        if img is None:
            continue
        if img.mode != "RGB":
            img = img.convert("RGB")
        out_path = out_dir / f"counting_{idx:05d}.png"
        save_image(img, str(out_path), format="PNG")
        saved.append((out_path, sample, count))
        if (idx + 1) % 200 == 0:
            logger.info("  saved %d/%d images", idx + 1, len(material))
    logger.info("Saved %d images for counting", len(saved))

    # Step 3: dedup against PixMo test split if available.
    keep_set: set[Path] | None = None
    if test_split is not None and not args.skip_dedup:
        test_filtered = filt(test_split)
        # Save test images temporarily for dedup comparison.
        test_dir = out_dir / "_pixmo_test_for_dedup"
        test_dir.mkdir(parents=True, exist_ok=True)
        test_paths = []
        for j, (_, ts, _) in enumerate(test_filtered[:1000]):
            ti = _open_image(ts)
            if ti is None:
                continue
            if ti.mode != "RGB":
                ti = ti.convert("RGB")
            tp = test_dir / f"pixmo_test_{j:05d}.png"
            save_image(ti, str(tp), format="PNG")
            test_paths.append(tp)
        keep_set = perceptual_dedup(
            candidates=[p for (p, _, _) in saved],
            exclude=test_paths,
            phash_thresh=8,
            ssim_thresh=0.95,
        )
        # Cleanup temp dir.
        for tp in test_paths:
            tp.unlink(missing_ok=True)
        try:
            test_dir.rmdir()
        except OSError:
            pass

    final = [(p, s, c) for (p, s, c) in saved if keep_set is None or p in keep_set]
    logger.info("After dedup: %d remain (need %d)", len(final), N_TRAIN + N_VAL + N_TEST)

    if len(final) < N_TRAIN + N_VAL + N_TEST:
        logger.warning(
            "Not enough samples after dedup (have %d, need %d). Continuing with what we have.",
            len(final),
            N_TRAIN + N_VAL + N_TEST,
        )

    # Step 4: build records. Open-ended counting matches the existing
    # numeric/counting answer_type in vlm_grpo (fuzzy partial credit
    # already implemented in the verifier).
    def build_records(slice_, split_name):
        recs = []
        for img_path, sample, count in slice_:
            obj_label = sample.get("label", "objects")
            question = f"How many {obj_label} are in the image? Provide a single integer answer."
            recs.append(
                {
                    "question": question,
                    "ground_truth": str(count),
                    "answer_type": "counting",
                    "choices": "",
                    "images": [str(img_path)],
                    "dataset_name": TASK_NAME,
                    "split": split_name,
                }
            )
        return recs

    train_records = build_records(final[:N_TRAIN], "train")
    val_records = build_records(final[N_TRAIN : N_TRAIN + N_VAL], "val")
    test_records = build_records(final[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST], "test")

    write_jsonl(train_records, f"{args.output_jsonl_prefix}_train.jsonl")
    write_jsonl(val_records, f"{args.output_jsonl_prefix}_val.jsonl")
    write_jsonl(test_records, f"{args.output_jsonl_prefix}_test.jsonl")
    logger.info(
        "Counting build done: %d/%d/%d (train/val/test)",
        len(train_records),
        len(val_records),
        len(test_records),
    )


if __name__ == "__main__":
    main()
