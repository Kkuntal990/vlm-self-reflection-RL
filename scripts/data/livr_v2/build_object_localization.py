#!/usr/bin/env python3
"""livr-v2 Object Localization builder.

Per Appendix A:
  * Source: COCO 2017 detection splits (train + val annotations)
  * Filter: bbox covers 15-50% of image area AND mask fills >=60% of bbox
  * Distractor: jitter the four bbox corners until IoU in [0.2, 0.5]
  * 2-way MCQ
  * Dedup: CLIP cosine vs BLINK Object_Localization val
  * Splits: 1000 / 250 / 122 (BLINK)

Usage:
    python scripts/data/livr_v2/build_object_localization.py \\
        --coco-dir /outputs/livr_v2_sources/coco \\
        --blink-val-dir /outputs/livr_v2_sources/blink_val \\
        --output-dir /outputs/livr_v2/image_base/object_localization \\
        --output-jsonl-prefix /outputs/livr_v2/data/object_localization
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from dedup import full_dedup  # noqa: E402
from livr_common import (  # noqa: E402
    OPTION_COLORS,
    OPTION_LETTERS,
    get_font,
    log_task_banner,
    logger,
    make_livr_record,
    save_image,
    shuffle_choices_with_index,
    write_jsonl,
)

TASK_NAME = "livr_object_localization"
N_TRAIN = 1000
N_VAL = 250
N_TEST = 122

BBOX_AREA_FRAC_MIN = 0.15
BBOX_AREA_FRAC_MAX = 0.50
MASK_FILL_MIN = 0.60
DISTRACTOR_IOU_LO = 0.20
DISTRACTOR_IOU_HI = 0.50
JITTER_PX_MAX = 200
JITTER_TRIES = 100


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0)
    ih = max(0, iy1 - iy0)
    inter = iw * ih
    a_area = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    b_area = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def _jitter_bbox(box, img_w, img_h, rng):
    """Jitter each corner until IoU lies in [0.2, 0.5]."""
    x0, y0, x1, y1 = box
    for _ in range(JITTER_TRIES):
        dx0 = rng.randint(-JITTER_PX_MAX, JITTER_PX_MAX)
        dy0 = rng.randint(-JITTER_PX_MAX, JITTER_PX_MAX)
        dx1 = rng.randint(-JITTER_PX_MAX, JITTER_PX_MAX)
        dy1 = rng.randint(-JITTER_PX_MAX, JITTER_PX_MAX)
        nx0 = max(0, min(img_w - 1, x0 + dx0))
        ny0 = max(0, min(img_h - 1, y0 + dy0))
        nx1 = max(nx0 + 5, min(img_w, x1 + dx1))
        ny1 = max(ny0 + 5, min(img_h, y1 + dy1))
        new = (nx0, ny0, nx1, ny1)
        if (nx1 - nx0) <= 5 or (ny1 - ny0) <= 5:
            continue
        v = _iou(box, new)
        if DISTRACTOR_IOU_LO <= v <= DISTRACTOR_IOU_HI:
            return new
    return None


def _draw_box(image, box, color, label, label_font_size=24):
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle(box, outline=color, width=4)
    font = get_font(label_font_size)
    tx = box[0] + 4
    ty = box[1] + 4
    # Label background.
    bg_w = label_font_size * (len(label) + 2)
    draw.rectangle((tx - 2, ty - 2, tx + bg_w, ty + label_font_size + 4), fill=color)
    draw.text((tx, ty), label, fill=(255, 255, 255), font=font)
    return out


def _load_blink_val_images(blink_dir: Path) -> list[Path]:
    """Locate BLINK Object_Localization val images for dedup target.

    BLINK ships as parquet files; `download_sources._extract_blink_images`
    writes per-row images under `<task>/images/`. We look there first.
    """
    images_dir = blink_dir / "Object_Localization" / "images"
    if images_dir.exists():
        return sorted(images_dir.glob("*.png"))
    # Fallback: any png/jpg under any *Object_Localization* dir.
    candidates = list(blink_dir.rglob("*Object_Localization*/*"))
    candidates += list(blink_dir.rglob("*object_localization*/*"))
    return [p for p in candidates if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]


def _load_coco_annotations(coco_dir: Path) -> list[tuple[Path, dict]]:
    """Load COCO 2017 instance annotations (both train + val)."""
    pairs = []
    for ann_file, img_subdir in [
        ("annotations/instances_train2017.json", "train2017"),
        ("annotations/instances_val2017.json", "val2017"),
    ]:
        ann_path = coco_dir / ann_file
        if not ann_path.exists():
            logger.warning("Missing annotation file: %s", ann_path)
            continue
        with open(ann_path) as f:
            data = json.load(f)
        id_to_file = {im["id"]: im["file_name"] for im in data["images"]}
        id_to_size = {im["id"]: (im["width"], im["height"]) for im in data["images"]}
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            img_id = ann["image_id"]
            if img_id not in id_to_file:
                continue
            iw, ih = id_to_size[img_id]
            x, y, w, h = ann["bbox"]
            box = (int(x), int(y), int(x + w), int(y + h))
            bbox_area = max(1, w * h)
            img_area = max(1, iw * ih)
            mask_area = ann.get("area", bbox_area)
            if not (BBOX_AREA_FRAC_MIN <= bbox_area / img_area <= BBOX_AREA_FRAC_MAX):
                continue
            if mask_area / bbox_area < MASK_FILL_MIN:
                continue
            full_path = coco_dir / img_subdir / id_to_file[img_id]
            pairs.append(
                (
                    full_path,
                    {
                        "box": box,
                        "img_size": (iw, ih),
                        "category": ann.get("category_id"),
                    },
                )
            )
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-dir", default="/outputs/livr_v2_sources/coco")
    parser.add_argument("--blink-val-dir", default="/outputs/livr_v2_sources/blink_val")
    parser.add_argument("--output-dir", default="/outputs/livr_v2/image_base/object_localization")
    parser.add_argument(
        "--output-jsonl-prefix", default="/outputs/livr_v2/data/object_localization"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-dedup", action="store_true")
    parser.add_argument("--clip-device", default="cuda")
    args = parser.parse_args()

    log_task_banner(TASK_NAME)
    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _load_coco_annotations(Path(args.coco_dir))
    logger.info("Loaded %d candidate (image, bbox) pairs after filter", len(pairs))
    rng.shuffle(pairs)

    needed = N_TRAIN + N_VAL + N_TEST + 300
    pairs = pairs[:needed]

    cat_id_to_name = {}
    cat_path = Path(args.coco_dir) / "annotations" / "instances_train2017.json"
    if cat_path.exists():
        with open(cat_path) as f:
            cats = json.load(f)["categories"]
        cat_id_to_name = {c["id"]: c["name"] for c in cats}

    saved: list[tuple[Path, dict]] = []
    for idx, (img_path, ann) in enumerate(pairs):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        iw, ih = img.size
        gt_box = ann["box"]
        distractor = _jitter_bbox(gt_box, iw, ih, rng)
        if distractor is None:
            continue
        # Draw both boxes; assign letters via shuffle.
        correct_pos, ordering = shuffle_choices_with_index(0, 2, rng)
        boxes = [gt_box, distractor]
        ordered = [boxes[i] for i in ordering]
        annotated = img.copy()
        for k, box in enumerate(ordered):
            color = OPTION_COLORS[k]
            label = f"({OPTION_LETTERS[k]})"
            annotated = _draw_box(annotated, box, color, label)
        out_filename = f"objloc_{len(saved):05d}.png"
        out_path = out_dir / out_filename
        save_image(annotated, str(out_path), format="PNG")

        category = cat_id_to_name.get(ann.get("category_id"), "object")
        question = (
            f"Which bounding box most accurately localizes the {category} in the image? (A) (B)"
        )
        formatted_choices = [f"({OPTION_LETTERS[i]})" for i in range(2)]
        rec = make_livr_record(
            question=question,
            ground_truth=OPTION_LETTERS[correct_pos],
            choices=formatted_choices,
            image_path=str(out_path),
            dataset_name=TASK_NAME,
        )
        saved.append((out_path, rec))
        if (idx + 1) % 200 == 0:
            logger.info("  built %d/%d", len(saved), needed)
    logger.info("Saved %d candidates", len(saved))

    # Dedup vs BLINK Object_Localization val.
    # Appendix A: "CLIP embeddings together with perceptual hashing and
    # SSIM-based image similarity (on both raw and blurred grayscale
    # images)" — three confirming checks PLUS a blurred-grayscale pass.
    keep_set: set[Path] | None = None
    if not args.skip_dedup:
        blink_imgs = _load_blink_val_images(Path(args.blink_val_dir))
        logger.info("BLINK Object_Localization val: %d images", len(blink_imgs))
        if blink_imgs:
            keep_set = full_dedup(
                candidates=[p for p, _ in saved],
                exclude=blink_imgs,
                clip_sim_thresh=0.95,
                phash_thresh=8,
                ssim_thresh=0.95,
                device=args.clip_device,
                blurred_grayscale=True,
            )

    final = [(p, r) for p, r in saved if keep_set is None or p in keep_set]
    logger.info("After dedup: %d", len(final))

    train = [r for _, r in final[:N_TRAIN]]
    val = [r for _, r in final[N_TRAIN : N_TRAIN + N_VAL]]
    test = [r for _, r in final[N_TRAIN + N_VAL : N_TRAIN + N_VAL + N_TEST]]
    for r in train:
        r["split"] = "train"
    for r in val:
        r["split"] = "val"
    for r in test:
        r["split"] = "test"
    write_jsonl(train, f"{args.output_jsonl_prefix}_train.jsonl")
    write_jsonl(val, f"{args.output_jsonl_prefix}_val.jsonl")
    write_jsonl(test, f"{args.output_jsonl_prefix}_test.jsonl")
    logger.info("Object_Localization done: %d/%d/%d", len(train), len(val), len(test))


if __name__ == "__main__":
    main()
