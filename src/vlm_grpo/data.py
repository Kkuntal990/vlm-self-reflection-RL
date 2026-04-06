#!/usr/bin/env python3
"""
Dataset loading and preprocessing for self-reflection GRPO training.

Provides dataset loaders for the multi-turn self-reflection flow
(A1 → F1 → A2) and supporting utilities for image loading and
answer type detection.

Usage:
    from vlm_grpo.data import load_self_reflection_dataset

    samples = load_self_reflection_dataset(
        dataset_path="/outputs/fire_preprocessed_v3/dataset.jsonl",
        image_base_dir="/outputs/image_base",
    )
"""

import json
import logging
import os
import re
import sys
from typing import Optional

from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Maximum image dimension (resize larger images to save memory)
MAX_IMAGE_DIM = 1024


def load_image_safe(
    image_path: str,
    max_pixels: Optional[int] = None,
) -> Optional[Image.Image]:
    """Load image from path with error handling and RGB conversion.

    Resizes images to save GPU memory using one of two strategies:
    - max_pixels: resize based on total pixel count (for Qwen2.5-VL dynamic resolution)
    - MAX_IMAGE_DIM: resize based on max dimension (legacy LLaVA behavior)

    Args:
        image_path: Absolute path to image file
        max_pixels: Maximum total pixels (width * height). If provided,
            uses pixel-count-based resizing. If None, falls back to
            MAX_IMAGE_DIM dimension-based resizing.

    Returns:
        PIL Image in RGB mode, or None if loading failed
    """
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        w, h = image.size

        if max_pixels is not None:
            # Pixel-count-based resizing (Qwen2.5-VL style)
            total_pixels = w * h
            if total_pixels > max_pixels:
                scale = (max_pixels / total_pixels) ** 0.5
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = image.resize((new_w, new_h), Image.LANCZOS)
        else:
            # Dimension-based resizing (LLaVA legacy)
            if max(w, h) > MAX_IMAGE_DIM:
                scale = MAX_IMAGE_DIM / max(w, h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                image = image.resize((new_w, new_h), Image.LANCZOS)

        return image
    except Exception as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return None


def detect_answer_type(
    question: str,
    ground_truth: str,
    choices: str = "",
) -> str:
    """Detect answer type from question and ground truth characteristics.

    Heuristic classification:
    - MCQ: question contains choice patterns (A)/(B) or A./B.
    - YesNo: ground truth is "Yes" or "No"
    - Numeric: ground truth is a number
    - Open: everything else

    Args:
        question: Question text
        ground_truth: Ground truth answer string
        choices: Explicitly provided choices (if any)

    Returns:
        Answer type string: "mcq", "yesno", "numeric", or "open"
    """
    gt_lower = ground_truth.strip().lower()

    # Check for explicit choices
    if choices:
        return "mcq"

    # Check for MCQ pattern in question: "(A) text" or "A. text"
    mcq_paren_pattern = re.compile(r"\([A-F]\)\s*\w+", re.IGNORECASE)
    mcq_dot_pattern = re.compile(r"^[A-F]\.\s*.+", re.MULTILINE | re.IGNORECASE)
    if mcq_paren_pattern.search(question) or mcq_dot_pattern.search(question):
        return "mcq"

    # Check for yes/no ground truth
    if gt_lower in ("yes", "no", "y", "n", "true", "false"):
        return "yesno"

    # Check for single letter answer (likely MCQ)
    if len(gt_lower) == 1 and gt_lower in "abcdef":
        return "mcq"

    # Check for numeric answer
    try:
        float(ground_truth.strip().replace(",", ""))
        return "numeric"
    except ValueError:
        pass

    return "open"


# Category → answer_type mapping for datasets with explicit category field
_CATEGORY_TO_ANSWER_TYPE: dict[str, str] = {
    "mcq": "mcq",
    "yes_no": "yesno",
    "counting": "counting",
    "chart_reasoning": "open",
    "descriptive_vqa": "open",
}


def _category_to_answer_type(category: str, fallback: str) -> str:
    """Map dataset category to answer_type, falling back to heuristic.

    Args:
        category: Dataset category field (e.g. "mcq", "yes_no", "counting")
        fallback: Fallback answer_type from heuristic detection

    Returns:
        Answer type string
    """
    if category and category in _CATEGORY_TO_ANSWER_TYPE:
        return _CATEGORY_TO_ANSWER_TYPE[category]
    return fallback


def _resolve_image_path(sample: dict, image_base_dir: str) -> str:
    """Resolve image path from sample, checking image_path and images fields.

    Args:
        sample: JSONL record
        image_base_dir: Base directory for relative paths

    Returns:
        Absolute image path string
    """
    image_path = sample.get("image_path", "")
    if not image_path:
        images_list = sample.get("images", [])
        if images_list:
            image_path = images_list[0]
    if image_path and not os.path.isabs(image_path):
        image_path = os.path.join(image_base_dir, image_path)
    return image_path


def _parse_messages_format(sample: dict) -> dict:
    """Extract flat fields from messages-format JSONL.

    Parses conversation messages to extract question, answer1, and
    ground_truth. Uses the last assistant response as ground_truth proxy.

    Args:
        sample: JSONL record with 'messages' key

    Returns:
        Dict with question, answer1, ground_truth, answer_type, choices,
        dataset_name
    """
    messages = sample.get("messages", [])

    user_msgs = [m for m in messages if m["role"] == "user"]
    assistant_msgs = [m for m in messages if m["role"] == "assistant"]

    question = user_msgs[0]["content"] if user_msgs else ""
    answer1 = assistant_msgs[0]["content"] if assistant_msgs else ""
    # Last assistant response as ground truth proxy
    ground_truth = assistant_msgs[-1]["content"] if assistant_msgs else ""

    # Detect answer type from question and ground truth
    answer_type = detect_answer_type(question, ground_truth)

    # Extract choices from question if MCQ
    choices = ""
    if answer_type == "mcq":
        choices = _extract_choices_from_question(question)

    # Infer dataset name from image path
    images = sample.get("images", [])
    dataset_name = "unknown"
    if images:
        parts = images[0].replace("/outputs/image_base/", "").split("/")
        if parts:
            dataset_name = parts[0]

    return {
        "question": question,
        "answer1": answer1,
        "ground_truth": ground_truth,
        "answer_type": answer_type,
        "choices": choices,
        "dataset_name": dataset_name,
    }


def _extract_choices_from_question(question: str) -> str:
    """Extract MCQ choices from question text.

    Args:
        question: Question text potentially containing (A)...(B)... patterns

    Returns:
        Comma-separated choices string, e.g. "(A) Yes, (B) No"
    """
    matches = re.findall(r"(\([A-F]\)\s*[^()\n]+)", question)
    if matches:
        return ", ".join(m.strip() for m in matches)
    return ""


def load_self_reflection_dataset(
    dataset_path: str,
    image_base_dir: str = "/outputs/image_base",
    max_samples: int = 0,
    max_pixels: Optional[int] = None,
) -> list[dict]:
    """Load dataset for full self-reflection GRPO training.

    Returns raw dicts (not HF Dataset) since the custom training loop
    handles batching. Each dict has the fields needed for rollout.

    Supports both flat JSONL and messages-format JSONL.

    Args:
        dataset_path: Path to JSONL dataset
        image_base_dir: Base directory for resolving relative image paths
        max_samples: Maximum samples to load (0 = all)
        max_pixels: Maximum total pixels per image for resizing. If None,
            uses legacy MAX_IMAGE_DIM-based resizing.

    Returns:
        List of dicts with keys: question, image_path, ground_truth,
        answer_type, choices, dataset_name, sample_index
    """
    logger.info(f"Loading self-reflection dataset from {dataset_path}")

    raw_samples = _load_jsonl(dataset_path, max_samples)
    logger.info(f"Loaded {len(raw_samples)} raw samples")

    processed = []
    skipped = 0

    for i, sample in enumerate(raw_samples):
        # Extract fields (handle both flat and messages format)
        if "messages" in sample and "question" not in sample:
            fields = _parse_messages_format(sample)
        else:
            fields = {
                "question": sample.get("question", ""),
                "ground_truth": sample.get("ground_truth", ""),
                "answer_type": sample.get("answer_type", "open"),
                "choices": sample.get("choices", ""),
                "dataset_name": sample.get("dataset_name", "unknown"),
            }

        # Rule 1: Use category field from dataset when available
        answer_type = _category_to_answer_type(sample.get("category", ""), fields["answer_type"])

        # Resolve image path
        image_path = _resolve_image_path(sample, image_base_dir)

        # Skip isfile() check — it's extremely slow on network filesystems
        # (CephFS/NFS). 70K stat calls × 4 accelerate processes can take 30+ min.
        # The dataset was validated at preparation time; trust the paths.
        if not image_path:
            skipped += 1
            continue

        question = fields["question"].replace("<image>", "").strip()

        processed.append(
            {
                "question": question,
                "image_path": image_path,
                "ground_truth": fields["ground_truth"],
                "answer_type": answer_type,
                "choices": fields["choices"],
                "dataset_name": fields["dataset_name"],
                "sample_index": i,
            }
        )

    if skipped > 0:
        logger.warning(f"Skipped {skipped} samples due to missing images")

    logger.info(f"Prepared {len(processed)} samples for self-reflection training")
    return processed


def _load_jsonl(path: str, max_samples: int = 0) -> list[dict]:
    """Load JSONL file with error handling.

    Args:
        path: Path to JSONL file
        max_samples: Maximum samples to load (0 = all)

    Returns:
        List of parsed dictionaries
    """
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples > 0 and i >= max_samples:
                break
            try:
                sample = json.loads(line.strip())
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {i}: {e}")

    return samples
