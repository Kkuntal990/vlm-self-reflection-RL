#!/usr/bin/env python3
"""Shared utilities for LIVR perception MCQ dataset construction.

Provides composite image creation, MCQ formatting, JSONL I/O,
and annotation drawing helpers used by all per-task build scripts.
"""

import json
import logging
import os
import random
import sys
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Letters for MCQ options
OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]

# Colors for annotations (RGB)
OPTION_COLORS = [
    (255, 0, 0),  # Red
    (0, 0, 255),  # Blue
    (0, 180, 0),  # Green
    (255, 165, 0),  # Orange
    (148, 0, 211),  # Violet
    (0, 206, 209),  # Cyan
]

# Default JPEG quality for composite images
JPEG_QUALITY = 85


def get_font(size: int = 24) -> ImageFont.FreeTypeFont:
    """Get a font for drawing labels, with fallback to default.

    Args:
        size: Font size in pixels.

    Returns:
        PIL font object.
    """
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


def make_livr_record(
    question: str,
    ground_truth: str,
    choices: list[str],
    image_path: str,
    dataset_name: str,
) -> dict:
    """Format a single LIVR MCQ record for the GRPO data loader.

    Ground truth is stored as "(A)" format (with parentheses) to match
    BLINK benchmark evaluation format, which compares predictions against
    "(A)"/"(B)"/etc. The reward pipeline's extract_answer_from_text()
    handles stripping parentheses for internal correctness checking.

    Args:
        question: Full question text including choice labels.
        ground_truth: Correct answer letter (e.g. "A"). Will be
            wrapped as "(A)" in the output record.
        choices: List of formatted choices, e.g. ["(A) 3", "(B) 5", ...].
        image_path: Absolute path to the (composite) image.
        dataset_name: Task identifier, e.g. "livr_counting".

    Returns:
        Dict matching the flat JSONL schema in data.py.
    """
    # Wrap bare letter in parens to match BLINK eval format: "A" -> "(A)"
    if len(ground_truth) == 1 and ground_truth.isalpha():
        ground_truth = f"({ground_truth})"

    return {
        "question": question,
        "ground_truth": ground_truth,
        "answer_type": "mcq",
        "choices": ", ".join(choices),
        "images": [image_path],
        "dataset_name": dataset_name,
    }


def shuffle_choices(
    correct_text: str,
    distractor_texts: list[str],
    rng: random.Random,
) -> tuple[str, list[str], list[str]]:
    """Shuffle correct answer among distractors, assign option letters.

    Args:
        correct_text: Text of the correct answer.
        distractor_texts: Texts of distractor answers.
        rng: Random number generator for reproducibility.

    Returns:
        Tuple of (correct_letter, formatted_choices, ordered_texts).
        formatted_choices: e.g. ["(A) 3", "(B) 7", "(C) 5", "(D) 9"].
        ordered_texts: texts in shuffled order (for composite image ordering).
    """
    all_texts = [correct_text] + list(distractor_texts)
    indices = list(range(len(all_texts)))
    rng.shuffle(indices)

    correct_pos = indices.index(0)
    correct_letter = OPTION_LETTERS[correct_pos]

    ordered_texts = [all_texts[i] for i in indices]
    formatted = [f"({OPTION_LETTERS[j]}) {ordered_texts[j]}" for j in range(len(ordered_texts))]
    return correct_letter, formatted, ordered_texts


def generate_numeric_distractors(
    correct: int,
    n_distractors: int,
    rng: random.Random,
    min_val: int = 1,
    max_val: int = 20,
) -> list[int]:
    """Generate plausible numeric distractors for counting MCQs.

    Distractors are within ±5 of correct, distinct from correct
    and from each other.

    Args:
        correct: The correct count.
        n_distractors: Number of distractors to generate.
        rng: Random number generator.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.

    Returns:
        List of distractor integers.
    """
    candidates = [
        v for v in range(max(min_val, correct - 5), min(max_val, correct + 5) + 1) if v != correct
    ]
    if len(candidates) < n_distractors:
        candidates = [v for v in range(min_val, max_val + 1) if v != correct]
    return rng.sample(candidates, min(n_distractors, len(candidates)))


def create_grid_image(
    images: list[Image.Image],
    labels: Optional[list[str]] = None,
    grid_cols: int = 2,
    cell_size: int = 300,
    padding: int = 8,
    label_font_size: int = 28,
    bg_color: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """Tile images into a labeled grid composite.

    Args:
        images: List of PIL images to tile.
        labels: Optional labels for each cell (e.g. ["(A)", "(B)", ...]).
        grid_cols: Number of columns in the grid.
        cell_size: Width and height of each cell in pixels.
        padding: Pixels between cells.
        label_font_size: Font size for cell labels.
        bg_color: Background color.

    Returns:
        Composite PIL Image.
    """
    n = len(images)
    grid_rows = (n + grid_cols - 1) // grid_cols
    if labels is None:
        labels = [f"({OPTION_LETTERS[i]})" for i in range(n)]

    font = get_font(label_font_size)
    label_height = label_font_size + 4

    total_w = grid_cols * cell_size + (grid_cols + 1) * padding
    total_h = grid_rows * (cell_size + label_height) + (grid_rows + 1) * padding

    canvas = Image.new("RGB", (total_w, total_h), bg_color)
    draw = ImageDraw.Draw(canvas)

    for idx, (img, label) in enumerate(zip(images, labels)):
        row, col = divmod(idx, grid_cols)
        x = padding + col * (cell_size + padding)
        y = padding + row * (cell_size + label_height + padding)

        # Draw label
        draw.text((x + 4, y), label, fill=(0, 0, 0), font=font)

        # Resize and paste image
        resized = img.copy()
        resized.thumbnail((cell_size, cell_size), Image.LANCZOS)
        # Center in cell
        offset_x = x + (cell_size - resized.width) // 2
        offset_y = y + label_height + (cell_size - resized.height) // 2
        canvas.paste(resized, (offset_x, offset_y))

    return canvas


def create_reference_and_options(
    reference: Image.Image,
    options: list[Image.Image],
    labels: Optional[list[str]] = None,
    ref_label: str = "Reference",
    cell_size: int = 280,
    padding: int = 8,
    label_font_size: int = 24,
) -> Image.Image:
    """Create composite with a reference image on top and option grid below.

    Layout:
        [   Reference Image   ]
        [(A) opt1] [(B) opt2]
        [(C) opt3] [(D) opt4]

    Args:
        reference: The reference image.
        options: List of option images.
        labels: Labels for options. Defaults to (A), (B), ...
        ref_label: Label for the reference image.
        cell_size: Size of each cell.
        padding: Padding between elements.
        label_font_size: Font size for labels.

    Returns:
        Composite PIL Image.
    """
    if labels is None:
        labels = [f"({OPTION_LETTERS[i]})" for i in range(len(options))]

    font = get_font(label_font_size)
    label_h = label_font_size + 4
    grid_cols = 2 if len(options) > 2 else len(options)
    grid_rows = (len(options) + grid_cols - 1) // grid_cols

    ref_width = min(grid_cols * cell_size + (grid_cols - 1) * padding, cell_size * 2)
    ref_height = cell_size

    total_w = grid_cols * cell_size + (grid_cols + 1) * padding
    ref_section_h = label_h + ref_height + padding
    opt_section_h = grid_rows * (cell_size + label_h) + (grid_rows + 1) * padding
    total_h = ref_section_h + opt_section_h + padding

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Draw reference
    draw.text((padding, padding), ref_label, fill=(0, 0, 0), font=font)
    ref_resized = reference.copy()
    ref_resized.thumbnail((ref_width, ref_height), Image.LANCZOS)
    ref_x = (total_w - ref_resized.width) // 2
    ref_y = padding + label_h
    canvas.paste(ref_resized, (ref_x, ref_y))

    # Draw separator line
    sep_y = ref_section_h + padding // 2
    draw.line([(padding, sep_y), (total_w - padding, sep_y)], fill=(180, 180, 180), width=2)

    # Draw options grid
    for idx, (img, label) in enumerate(zip(options, labels)):
        row, col = divmod(idx, grid_cols)
        x = padding + col * (cell_size + padding)
        y = ref_section_h + padding + row * (cell_size + label_h + padding)

        draw.text((x + 4, y), label, fill=OPTION_COLORS[idx % len(OPTION_COLORS)], font=font)
        resized = img.copy()
        resized.thumbnail((cell_size, cell_size), Image.LANCZOS)
        offset_x = x + (cell_size - resized.width) // 2
        offset_y = y + label_h + (cell_size - resized.height) // 2
        canvas.paste(resized, (offset_x, offset_y))

    return canvas


def create_side_by_side(
    left: Image.Image,
    right: Image.Image,
    left_label: str = "Source",
    right_label: str = "Target",
    target_height: int = 400,
    padding: int = 10,
    label_font_size: int = 24,
) -> Image.Image:
    """Create a horizontal side-by-side composite of two images.

    Args:
        left: Left image.
        right: Right image.
        left_label: Label for left image.
        right_label: Label for right image.
        target_height: Target height for each image.
        padding: Padding between images.
        label_font_size: Font size for labels.

    Returns:
        Composite PIL Image.
    """
    font = get_font(label_font_size)
    label_h = label_font_size + 6

    # Resize both to same height
    l_ratio = target_height / left.height
    r_ratio = target_height / right.height
    l_resized = left.resize((int(left.width * l_ratio), target_height), Image.LANCZOS)
    r_resized = right.resize((int(right.width * r_ratio), target_height), Image.LANCZOS)

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


def draw_keypoint(
    image: Image.Image,
    x: int,
    y: int,
    color: tuple[int, int, int] = (255, 0, 0),
    radius: int = 10,
    label: Optional[str] = None,
    font_size: int = 20,
) -> Image.Image:
    """Draw a keypoint marker (filled circle + optional label) on an image.

    Args:
        image: PIL image (modified in-place copy).
        x: X coordinate.
        y: Y coordinate.
        color: RGB color of the marker.
        radius: Radius of the circle.
        label: Optional text label next to the marker.
        font_size: Font size for the label.

    Returns:
        Image with keypoint drawn.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=color,
        outline=(255, 255, 255),
        width=2,
    )
    if label:
        font = get_font(font_size)
        draw.text((x + radius + 4, y - radius), label, fill=color, font=font)
    return img


def draw_bbox(
    image: Image.Image,
    bbox: tuple[float, float, float, float],
    color: tuple[int, int, int] = (255, 0, 0),
    width: int = 3,
    label: Optional[str] = None,
    font_size: int = 20,
) -> Image.Image:
    """Draw a bounding box on an image.

    Args:
        image: PIL image (modified in-place copy).
        bbox: (x, y, w, h) in pixels.
        color: RGB color.
        width: Line width.
        label: Optional label at top-left of box.
        font_size: Font size for label.

    Returns:
        Image with bounding box drawn.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline=color, width=width)
    if label:
        font = get_font(font_size)
        # Background for label readability
        draw.rectangle([x, y - font_size - 4, x + len(label) * font_size, y], fill=color)
        draw.text((x + 2, y - font_size - 2), label, fill=(255, 255, 255), font=font)
    return img


def compute_iou(
    box1: tuple[float, float, float, float],
    box2: tuple[float, float, float, float],
) -> float:
    """Compute IoU between two (x, y, w, h) bounding boxes.

    Args:
        box1: First box as (x, y, w, h).
        box2: Second box as (x, y, w, h).

    Returns:
        Intersection over Union value.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter

    return inter / union if union > 0 else 0.0


def write_jsonl(records: list[dict], path: str) -> None:
    """Write a list of dicts to a JSONL file.

    Args:
        records: List of dictionaries to serialize.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(records)} records to {path}")


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts.

    Args:
        path: Path to JSONL file.

    Returns:
        List of parsed dictionaries.
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_image(image: Image.Image, path: str, quality: int = JPEG_QUALITY) -> None:
    """Save a PIL image as JPEG.

    Args:
        image: PIL Image to save.
        path: Output file path.
        quality: JPEG quality (1-100).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path, "JPEG", quality=quality)
