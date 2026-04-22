#!/usr/bin/env python3
"""Build LIVR Jigsaw MCQ dataset from COCO images.

Cuts images into 4 quadrants, creates a scrambled puzzle, and generates
4-way MCQs where the model must identify the correct quadrant arrangement.

Usage:
    python scripts/livr/build_jigsaw.py
"""

import argparse
import itertools
import logging
import os
import random
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(__file__))
from livr_common import (
    OPTION_COLORS,
    OPTION_LETTERS,
    get_font,
    logger,
    make_livr_record,
    save_image,
    write_jsonl,
)

TASK_NAME = "livr_jigsaw"
N_SAMPLES = 1200
PUZZLE_SIZE = 400  # Resize source image to this before cutting


def _cut_quadrants(img: Image.Image) -> list[Image.Image]:
    """Cut an image into 4 quadrants (TL, TR, BL, BR).

    Args:
        img: Source image resized to PUZZLE_SIZE x PUZZLE_SIZE.

    Returns:
        List of 4 PIL Images [TL, TR, BL, BR].
    """
    w, h = img.size
    hw, hh = w // 2, h // 2
    return [
        img.crop((0, 0, hw, hh)),  # TL
        img.crop((hw, 0, w, hh)),  # TR
        img.crop((0, hh, hw, h)),  # BL
        img.crop((hw, hh, w, h)),  # BR
    ]


def _assemble_quadrants(
    quads: list[Image.Image],
    order: tuple[int, ...],
    size: int,
) -> Image.Image:
    """Assemble 4 quadrants in a given order into a 2x2 grid.

    Args:
        quads: List of 4 quadrant images.
        order: Tuple of 4 indices specifying arrangement.
        size: Total output size (width = height = size).

    Returns:
        Assembled 2x2 image.
    """
    hs = size // 2
    canvas = Image.new("RGB", (size, size), (200, 200, 200))
    positions = [(0, 0), (hs, 0), (0, hs), (hs, hs)]
    for pos, idx in zip(positions, order):
        q = quads[idx].resize((hs, hs), Image.LANCZOS)
        canvas.paste(q, pos)
    return canvas


def _get_distinct_permutations(
    correct: tuple[int, ...],
    n: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    """Get n random permutations of (0,1,2,3) distinct from correct and each other.

    Args:
        correct: The correct permutation.
        n: Number of distractors needed.
        rng: Random number generator.

    Returns:
        List of n distinct permutations.
    """
    all_perms = list(itertools.permutations(range(4)))
    all_perms.remove(correct)
    rng.shuffle(all_perms)
    return all_perms[:n]


def main() -> None:
    """Build jigsaw puzzle MCQ dataset from COCO images."""
    parser = argparse.ArgumentParser(description="Build LIVR jigsaw MCQs.")
    parser.add_argument(
        "--images-dir",
        default="/outputs/image_base/coco/train2017",
        help="Path to COCO train2017 images.",
    )
    parser.add_argument(
        "--output-dir",
        default="/outputs/image_base/livr/jigsaw",
        help="Output directory for composite images.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="/outputs/livr_data/livr_jigsaw.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # List COCO images
    all_images = sorted([f for f in os.listdir(args.images_dir) if f.endswith((".jpg", ".png"))])
    logger.info(f"Found {len(all_images)} COCO images")

    rng.shuffle(all_images)
    selected = all_images[: args.n_samples]

    os.makedirs(args.output_dir, exist_ok=True)
    records = []

    correct_order = (0, 1, 2, 3)  # TL, TR, BL, BR

    for idx, img_filename in enumerate(selected):
        img_path = os.path.join(args.images_dir, img_filename)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            continue

        # Resize to square
        img = img.resize((PUZZLE_SIZE, PUZZLE_SIZE), Image.LANCZOS)
        quads = _cut_quadrants(img)

        # Generate a scrambled version (for display)
        scramble = list(correct_order)
        while tuple(scramble) == correct_order:
            rng.shuffle(scramble)
        scrambled_img = _assemble_quadrants(quads, tuple(scramble), PUZZLE_SIZE)

        # Generate 3 distractor arrangements (different from correct and scramble)
        distractor_perms = _get_distinct_permutations(correct_order, 3, rng)

        # All candidate arrangements: correct + 3 distractors
        all_options = [correct_order] + distractor_perms
        indices = list(range(4))
        rng.shuffle(indices)
        correct_pos = indices.index(0)
        correct_letter = OPTION_LETTERS[correct_pos]
        shuffled_options = [all_options[i] for i in indices]

        # Build composite: scrambled puzzle on top, 4 candidate solutions below
        option_size = 180
        font = get_font(20)
        label_h = 24

        comp_w = 4 * option_size + 5 * 6  # 4 options side by side
        comp_h = PUZZLE_SIZE + label_h + option_size + label_h + 20

        canvas = Image.new("RGB", (comp_w, comp_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Draw scrambled puzzle centered
        scramble_x = (comp_w - PUZZLE_SIZE) // 2
        draw.text((scramble_x, 2), "Scrambled Puzzle", fill=(0, 0, 0), font=font)
        canvas.paste(scrambled_img, (scramble_x, label_h))

        # Draw separator
        sep_y = PUZZLE_SIZE + label_h + 4
        draw.line([(6, sep_y), (comp_w - 6, sep_y)], fill=(180, 180, 180), width=2)

        # Draw 4 option arrangements
        for j, perm in enumerate(shuffled_options):
            opt_img = _assemble_quadrants(quads, perm, option_size)
            ox = 6 + j * (option_size + 6)
            oy = sep_y + 6
            letter = OPTION_LETTERS[j]
            color = OPTION_COLORS[j]
            draw.text((ox + 2, oy), f"({letter})", fill=color, font=font)
            canvas.paste(opt_img, (ox, oy + label_h))

        out_filename = f"jigsaw_{idx:04d}.jpg"
        out_path = os.path.join(args.output_dir, out_filename)
        save_image(canvas, out_path)

        # Build MCQ
        formatted_choices = [f"({OPTION_LETTERS[j]})" for j in range(4)]
        choices_str = " ".join(formatted_choices)
        question = (
            f"The top image shows a scrambled jigsaw puzzle. "
            f"Which option below shows the correct arrangement of the pieces "
            f"to form the original image? {choices_str}"
        )

        record = make_livr_record(
            question=question,
            ground_truth=correct_letter,
            choices=formatted_choices,
            image_path=out_path,
            dataset_name=TASK_NAME,
        )
        records.append(record)

        if (idx + 1) % 200 == 0:
            logger.info(f"Processed {idx + 1}/{len(selected)} jigsaw samples")

    write_jsonl(records, args.output_jsonl)
    logger.info(f"Built {len(records)} jigsaw MCQ samples.")


if __name__ == "__main__":
    main()
