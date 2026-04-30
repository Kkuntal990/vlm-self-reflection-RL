#!/usr/bin/env python3
"""livr-v2 Jigsaw builder.

Per Appendix A:
  * Source: COCO 2017 (train2017 + val2017)
  * Crop a 400 x h canvas with random h in [170, 230] from each image
  * Treat the bottom-right quadrant as the GT patch; black it out in
    the displayed image
  * Distractor patch: same size, sampled from elsewhere in the SAME
    COCO image, must intersect the canvas and not overlap the GT
  * 2-way MCQ
  * Splits: 1000 train / 250 val / 150 test (BLINK)

Usage:
    python scripts/data/livr_v2/build_jigsaw.py \\
        --coco-dir /outputs/livr_v2_sources/coco \\
        --output-dir /outputs/livr_v2/image_base/jigsaw \\
        --output-jsonl-prefix /outputs/livr_v2/data/jigsaw
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from livr_common import (  # noqa: E402
    OPTION_LETTERS,
    log_task_banner,
    logger,
    make_livr_record,
    save_image,
    shuffle_choices_with_index,
    write_jsonl,
)

TASK_NAME = "livr_jigsaw"
N_TRAIN = 1000
N_VAL = 250
N_TEST = 150
CANVAS_W = 400
H_RANGE = (170, 230)


def _pick_canvas(img: Image.Image, rng: random.Random):
    """Pick a 400 x h crop region from `img`. Returns (x, y, w, h)."""
    iw, ih = img.size
    h = rng.randint(*H_RANGE)
    w = CANVAS_W
    if iw < w + 20 or ih < h + 20:
        return None
    x = rng.randint(0, iw - w)
    y = rng.randint(0, ih - h)
    return x, y, w, h


def _quadrants(x: int, y: int, w: int, h: int):
    """Return TL, TR, BL, BR quadrant boxes."""
    hw, hh = w // 2, h // 2
    return {
        "TL": (x, y, x + hw, y + hh),
        "TR": (x + hw, y, x + w, y + hh),
        "BL": (x, y + hh, x + hw, y + h),
        "BR": (x + hw, y + hh, x + w, y + h),
    }


def _rect_intersects(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _rect_overlaps(a, b):
    return _rect_intersects(a, b)


def _try_distractor(img_size, gt_box, canvas_box, rng, max_tries=50):
    """Sample a same-size patch elsewhere in the image that intersects
    the canvas and does not overlap the GT patch."""
    iw, ih = img_size
    pw = gt_box[2] - gt_box[0]
    ph = gt_box[3] - gt_box[1]
    for _ in range(max_tries):
        dx = rng.randint(0, iw - pw)
        dy = rng.randint(0, ih - ph)
        d = (dx, dy, dx + pw, dy + ph)
        if not _rect_intersects(d, canvas_box):
            continue
        if _rect_overlaps(d, gt_box):
            continue
        return d
    return None


def _list_coco_images(coco_dir: Path) -> list[Path]:
    """Return both train2017 and val2017 image paths (Appendix A says
    they use both COCO 2017 detection splits)."""
    images = []
    for sub in ("train2017", "val2017"):
        d = coco_dir / sub
        if not d.exists():
            continue
        images += sorted(d.glob("*.jpg"))
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-dir", default="/outputs/livr_v2_sources/coco")
    parser.add_argument("--output-dir", default="/outputs/livr_v2/image_base/jigsaw")
    parser.add_argument("--output-jsonl-prefix", default="/outputs/livr_v2/data/jigsaw")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    log_task_banner(TASK_NAME)
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coco_imgs = _list_coco_images(Path(args.coco_dir))
    logger.info("Found %d COCO images", len(coco_imgs))
    rng.shuffle(coco_imgs)

    needed = N_TRAIN + N_VAL + N_TEST + 200
    records: list[tuple[str, dict]] = []

    for img_idx, img_path in enumerate(coco_imgs):
        if len(records) >= needed:
            break
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        canvas = _pick_canvas(img, rng)
        if canvas is None:
            continue
        cx, cy, cw, ch = canvas
        canvas_box = (cx, cy, cx + cw, cy + ch)
        quads = _quadrants(cx, cy, cw, ch)
        gt_box = quads["BR"]
        distractor = _try_distractor(img.size, gt_box, canvas_box, rng)
        if distractor is None:
            continue

        # Build the displayed canvas: crop, then black out the BR quadrant.
        canvas_img = img.crop(canvas_box).copy()
        bw_x = cw // 2
        bw_y = ch // 2
        ImageDraw.Draw(canvas_img).rectangle((bw_x, bw_y, cw, ch), fill=(0, 0, 0))

        # Crop the two patches.
        gt_patch = img.crop(gt_box)
        d_patch = img.crop(distractor)

        # 2-way MCQ — randomize order.
        correct_pos, ordering = shuffle_choices_with_index(0, 2, rng)
        patches = [gt_patch, d_patch]
        ordered_patches = [patches[i] for i in ordering]

        # Compose: canvas on top, two patches side-by-side below labeled (A) (B).
        from livr_common import get_font  # local import; cheap

        font = get_font(28)
        patch_w = max(p.width for p in ordered_patches)
        patch_h = max(p.height for p in ordered_patches)
        label_h = 36
        gap = 12
        out_w = max(canvas_img.width, 2 * patch_w + 3 * gap)
        out_h = canvas_img.height + label_h + patch_h + 3 * gap
        composite = Image.new("RGB", (out_w, out_h), (255, 255, 255))
        cx0 = (out_w - canvas_img.width) // 2
        composite.paste(canvas_img, (cx0, gap))
        # Patches.
        x0 = (out_w - 2 * patch_w - gap) // 2
        y0 = canvas_img.height + 2 * gap + label_h
        for k, patch in enumerate(ordered_patches):
            xk = x0 + k * (patch_w + gap)
            composite.paste(patch, (xk, y0))
            ImageDraw.Draw(composite).text(
                (xk + patch_w // 2 - 12, y0 - label_h + 4),
                f"({OPTION_LETTERS[k]})",
                fill=(0, 0, 0),
                font=font,
            )

        out_filename = f"jigsaw_{len(records):05d}.png"
        out_path = out_dir / out_filename
        save_image(composite, str(out_path), format="PNG")

        question = (
            "The top image has its bottom-right quadrant blacked out. "
            "Which of the two patches below is the correct missing patch? "
            "(A) (B)"
        )
        formatted_choices = [f"({OPTION_LETTERS[i]})" for i in range(2)]
        rec = make_livr_record(
            question=question,
            ground_truth=OPTION_LETTERS[correct_pos],
            choices=formatted_choices,
            image_path=str(out_path),
            dataset_name=TASK_NAME,
        )
        records.append((str(out_path), rec))
        if (len(records)) % 200 == 0:
            logger.info("  built %d/%d", len(records), needed)

    logger.info("Built %d jigsaw composites (target %d)", len(records), needed)

    train = [r for _, r in records[:N_TRAIN]]
    val = [r for _, r in records[N_TRAIN : N_TRAIN + N_VAL]]
    test = [r for _, r in records[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]]
    for r in train:
        r["split"] = "train"
    for r in val:
        r["split"] = "val"
    for r in test:
        r["split"] = "test"
    write_jsonl(train, f"{args.output_jsonl_prefix}_train.jsonl")
    write_jsonl(val, f"{args.output_jsonl_prefix}_val.jsonl")
    write_jsonl(test, f"{args.output_jsonl_prefix}_test.jsonl")
    logger.info("Jigsaw build done: %d/%d/%d (train/val/test)", len(train), len(val), len(test))


if __name__ == "__main__":
    main()
