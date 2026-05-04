"""Repack BLINK per-task TSVs into single-composite-image format.

Multi-image BLINK tasks (Visual_Similarity, Multi-view_Reasoning, Forensic_Detection,
Art_Style, Visual_Correspondence, Semantic_Correspondence, Functional_Correspondence,
Jigsaw) are composited into a single labeled image matching the LIVR training
distribution. Single-image tasks pass through unchanged.

Output: BLINK TSVs with one base64 image per row, plus an `image_path` column,
matching the upstream BLINK.tsv schema vlmevalkit's `dump_image` already supports.

Run on the helper pod with shared PVC access.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
from typing import Callable

import pandas as pd
from PIL import Image

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from livr.livr_common import (
    create_grid_image,
    create_reference_and_options,
    create_side_by_side,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def b64_to_image(b64_str: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def image_to_b64(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_LABEL_FONT_SIZE = 14


def composite_visual_similarity(images: list[Image.Image]) -> Image.Image:
    # image=reference, image_1=(A) "the second image", image_2=(B) "the third image"
    return create_reference_and_options(
        reference=images[0],
        options=images[1:],
        ref_label="Reference (the first image)",
        labels=["(A) the second image", "(B) the third image"],
        cell_size=320,
        label_font_size=_LABEL_FONT_SIZE,
    )


def composite_art_style(images: list[Image.Image]) -> Image.Image:
    return create_reference_and_options(
        reference=images[0],
        options=images[1:],
        ref_label="Reference (the first image)",
        labels=["(A) the second image", "(B) the third image"],
        cell_size=320,
        label_font_size=_LABEL_FONT_SIZE,
    )


def composite_jigsaw(images: list[Image.Image]) -> Image.Image:
    return create_reference_and_options(
        reference=images[0],
        options=images[1:],
        ref_label="The first image (lower-right corner missing)",
        labels=["(A) the second image", "(B) the third image"],
        cell_size=320,
        label_font_size=_LABEL_FONT_SIZE,
    )


def composite_multi_view(images: list[Image.Image]) -> Image.Image:
    return create_side_by_side(
        left=images[0],
        right=images[1],
        left_label="First image",
        right_label="Second image",
        target_height=400,
        label_font_size=_LABEL_FONT_SIZE,
    )


def composite_correspondence(images: list[Image.Image]) -> Image.Image:
    # Visual / Semantic / Functional correspondence. The REF marker is drawn
    # inside the source image and A/B/C/D markers are inside the target —
    # labels just need to identify which is which (question text gives roles).
    return create_side_by_side(
        left=images[0],
        right=images[1],
        left_label="First image",
        right_label="Second image",
        target_height=400,
        label_font_size=_LABEL_FONT_SIZE,
    )


def composite_forensic(images: list[Image.Image]) -> Image.Image:
    return create_grid_image(
        images=images,
        labels=["(A) the first image", "(B) the second image",
                "(C) the third image", "(D) the fourth image"],
        grid_cols=2,
        cell_size=320,
        label_font_size=_LABEL_FONT_SIZE,
    )


# task_name -> compositor (None means single-image, copy through)
COMPOSITORS: dict[str, Callable[[list[Image.Image]], Image.Image] | None] = {
    "Visual_Similarity": composite_visual_similarity,
    "Art_Style": composite_art_style,
    "Jigsaw": composite_jigsaw,
    "Multi-view_Reasoning": composite_multi_view,
    "Visual_Correspondence": composite_correspondence,
    "Semantic_Correspondence": composite_correspondence,
    "Functional_Correspondence": composite_correspondence,
    "Forensic_Detection": composite_forensic,
    # single-image (no composite needed):
    "Counting": None,
    "IQ_Test": None,
    "Object_Localization": None,
    "Relative_Depth": None,
    "Relative_Reflectance": None,
    "Spatial_Relation": None,
}


def process_task(in_tsv: str, out_tsv: str, task_name: str) -> tuple[int, int]:
    df = pd.read_csv(in_tsv, sep="\t")
    img_cols = sorted(
        [c for c in df.columns if c == "image" or c.startswith("image_")],
        key=lambda c: 0 if c == "image" else int(c.split("_")[1]),
    )
    n_imgs = len(img_cols)
    compositor = COMPOSITORS.get(task_name)

    new_rows = []
    for ridx, row in df.iterrows():
        try:
            if n_imgs == 1 or compositor is None:
                composite_b64 = row["image"]
                file_name = f"{row['index']}.jpg"
            else:
                images = [b64_to_image(row[c]) for c in img_cols]
                composite = compositor(images)
                composite_b64 = image_to_b64(composite, quality=90)
                file_name = f"{row['index']}_composite.jpg"
        except Exception as e:
            logger.warning(f"  row {ridx}: composite failed ({e}); skipping")
            continue

        new_row = {c: row[c] for c in df.columns if not c.startswith("image")}
        new_row["image"] = composite_b64
        new_row["image_path"] = file_name
        new_rows.append(new_row)

    out_df = pd.DataFrame(new_rows)
    cols = ["index", "image", "image_path", "question", "A", "B", "C", "D", "answer"]
    out_df = out_df[[c for c in cols if c in out_df.columns]]

    os.makedirs(os.path.dirname(out_tsv), exist_ok=True)
    out_df.to_csv(out_tsv, sep="\t", index=False)
    return len(out_df), n_imgs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src_dir", default="/outputs/benchmark_data/blink/tsv")
    parser.add_argument("--dst_dir", default="/outputs/benchmark_data/blink_v2/tsv")
    args = parser.parse_args()

    tasks = sorted(COMPOSITORS.keys())
    logger.info(f"Repacking {len(tasks)} BLINK tasks: {args.src_dir} -> {args.dst_dir}")

    for task in tasks:
        in_p = os.path.join(args.src_dir, f"BLINK_{task}.tsv")
        out_p = os.path.join(args.dst_dir, f"BLINK_{task}.tsv")
        if not os.path.exists(in_p):
            logger.warning(f"missing source TSV: {in_p}; skipping")
            continue
        n_rows, n_imgs = process_task(in_p, out_p, task)
        kind = "single" if (n_imgs == 1 or COMPOSITORS[task] is None) else "composite"
        logger.info(f"  {task:30s}  {n_rows:4d} rows  src_imgs={n_imgs}  ->  {kind}")

    logger.info("done")


if __name__ == "__main__":
    main()
