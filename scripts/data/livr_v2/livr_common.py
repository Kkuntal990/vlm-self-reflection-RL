#!/usr/bin/env python3
"""Shared utilities for LIVR-v2 dataset construction.

Differences from the v1 livr_common:
  * No aggressive resize. Composites preserve native resolution by
    default (Qwen2.5-VL has dynamic resolution up to ~640x640 effective;
    we don't need to pre-shrink).
  * Default save format is PNG (lossless), not JPEG q=85.
  * `draw_ref_marker()` for the "REF" annotation used by viscorr,
    semcorr, funccorr per Appendix A.
  * `clip_dedup()` and `perceptual_dedup()` deduplication helpers
    (used by counting, object_localization, art_style, semantic_corr,
    visual_similarity).

Usage:
    from scripts.data.livr_v2.livr_common import (
        OPTION_LETTERS,
        OPTION_COLORS,
        get_font,
        make_livr_record,
        save_image,
        write_jsonl,
        draw_keypoint,
        draw_ref_marker,
        create_side_by_side,
        ...
    )
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Letters for MCQ options.
OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]

# Colors for annotations (RGB), kept consistent with v1 so eval prompts
# look familiar across versions.
OPTION_COLORS = [
    (255, 0, 0),  # Red    — A
    (0, 0, 255),  # Blue   — B
    (0, 180, 0),  # Green  — C
    (255, 165, 0),  # Orange — D
    (148, 0, 211),  # Violet — E
    (0, 206, 209),  # Cyan   — F
]


def get_font(size: int = 24) -> ImageFont.FreeTypeFont:
    """Get a font for drawing labels, with fallback to PIL default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


# =============================================================================
# Record I/O
# =============================================================================


def make_livr_record(
    question: str,
    ground_truth: str,
    choices: list[str],
    image_path: str,
    dataset_name: str,
    split: str = "train",
) -> dict:
    """Format a single LIVR-v2 MCQ record for the GRPO data loader.

    Ground truth is stored as "(A)" format (with parentheses) so it
    matches BLINK eval and the existing v1 records.

    Args:
        question: Full question text including choice labels.
        ground_truth: Correct answer letter (e.g. "A"). Will be wrapped
            as "(A)" in the output record.
        choices: Formatted choices, e.g. ["(A) 3", "(B) 5", ...].
        image_path: Absolute path to the (composite) image.
        dataset_name: Task identifier, e.g. "livr_counting".
        split: One of "train", "val", "test".

    Returns:
        Dict matching the flat JSONL schema in vlm_grpo.data.
    """
    if len(ground_truth) == 1 and ground_truth.isalpha():
        ground_truth = f"({ground_truth})"

    return {
        "question": question,
        "ground_truth": ground_truth,
        "answer_type": "mcq",
        "choices": ", ".join(choices),
        "images": [image_path],
        "dataset_name": dataset_name,
        "split": split,
    }


def write_jsonl(records: list[dict], path: str) -> None:
    """Write a list of dicts as JSONL."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records -> %s", len(records), path)


def save_image(img: Image.Image, path: str, format: str = "PNG") -> None:
    """Save image at native resolution. Default PNG (lossless)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if format == "PNG":
        img.save(path, format="PNG", optimize=True)
    elif format in ("JPEG", "JPG"):
        # Allowed for tasks where lossless is overkill (e.g. ArtBench
        # paintings already JPEG); keep quality high.
        img.save(path, format="JPEG", quality=95, subsampling=0)
    else:
        img.save(path, format=format)


# =============================================================================
# Drawing helpers
# =============================================================================


def draw_keypoint(
    image: Image.Image,
    x: int,
    y: int,
    color: tuple[int, int, int] = (255, 0, 0),
    radius: int = 10,
    label: Optional[str] = None,
    font_size: int = 20,
) -> Image.Image:
    """Draw a colored circle at (x, y) with optional label.

    Returns a copy; original `image` unchanged.
    """
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=color,
        outline=(255, 255, 255),
        width=2,
    )
    if label is not None:
        font = get_font(font_size)
        # Place label slightly above-right of the circle.
        tx = x + radius + 4
        ty = max(0, y - radius - font_size - 2)
        # Outlined text for legibility on any background.
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((tx + dx, ty + dy), label, fill=(255, 255, 255), font=font)
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)
    return out


def draw_ref_marker(
    image: Image.Image,
    x: int,
    y: int,
    radius: int = 10,
    label: str = "REF",
    color: tuple[int, int, int] = (255, 0, 0),
) -> Image.Image:
    """Draw the 'REF' (reference keypoint) marker used by viscorr,
    semcorr, funccorr.

    Per Appendix A of the LIVR paper: a single red circle on the source
    image at the reference keypoint, labeled 'REF'.
    """
    return draw_keypoint(image, x, y, color=color, radius=radius, label=label)


def create_side_by_side(
    left: Image.Image,
    right: Image.Image,
    left_label: str = "Source",
    right_label: str = "Target",
    target_height: Optional[int] = None,
    padding: int = 10,
    label_font_size: int = 24,
) -> Image.Image:
    """Horizontal side-by-side composite.

    `target_height=None` (default) preserves native resolution by
    matching to the larger of the two image heights — i.e. only the
    smaller image is upscaled to match. This is the v2 default and is
    the right choice for Qwen2.5-VL's dynamic resolution.

    If you pass an explicit `target_height` (e.g. for tasks that need
    a fixed canvas), both images are resized to that height with
    aspect-preserving scaling.
    """
    if target_height is None:
        target_height = max(left.height, right.height)

    font = get_font(label_font_size)
    label_h = label_font_size + 6

    l_ratio = target_height / left.height
    r_ratio = target_height / right.height
    l_resized = left.resize((max(1, int(left.width * l_ratio)), target_height), Image.LANCZOS)
    r_resized = right.resize((max(1, int(right.width * r_ratio)), target_height), Image.LANCZOS)

    total_w = l_resized.width + r_resized.width + 3 * padding
    total_h = target_height + label_h + 2 * padding

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Left
    draw.text((padding, padding), left_label, fill=(0, 0, 0), font=font)
    canvas.paste(l_resized, (padding, padding + label_h))

    # Right
    rx = l_resized.width + 2 * padding
    draw.text((rx, padding), right_label, fill=(0, 0, 0), font=font)
    canvas.paste(r_resized, (rx, padding + label_h))

    return canvas


# =============================================================================
# Distractor / shuffle helpers
# =============================================================================


def shuffle_choices_with_index(
    correct_idx: int,
    n_total: int,
    rng,
) -> tuple[int, list[int]]:
    """Shuffle n_total option indices and return the new position of
    the correct one.

    Returns:
        (new_correct_pos, ordering)
        new_correct_pos: position 0..n_total-1 of the correct option.
        ordering: list of length n_total mapping new_pos -> original_idx.
    """
    indices = list(range(n_total))
    rng.shuffle(indices)
    new_correct_pos = indices.index(correct_idx)
    return new_correct_pos, indices


# =============================================================================
# Logging banner
# =============================================================================


def log_task_banner(task: str, version: str = "livr-v2") -> None:
    bar = "=" * 60
    logger.info(bar)
    logger.info("  %s :: %s", version, task)
    logger.info(bar)
