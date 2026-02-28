#!/usr/bin/env python3
"""
Dataset loading and preprocessing for GRPO RW training.

Loads precomputed Answer1-correct JSONL files and converts them to
TRL's GRPOTrainer-compatible HuggingFace Dataset format with vision support.

Dataset columns required by TRL:
- prompt: list[dict] (conversational messages for VLM)
- images: list[PIL.Image] (image inputs)
Additional columns passed as kwargs to reward functions:
- ground_truth, answer1, answer_type, choices, dataset_name

Usage:
    from vlm_grpo.data import load_grpo_dataset

    dataset = load_grpo_dataset(
        dataset_path="/outputs/grpo_data/answer1_correct_train.jsonl",
        image_base_dir="/outputs/image_base",
        max_samples=100,
    )
"""

import json
import logging
import os
import re
import sys
from typing import Optional

from datasets import Dataset
from PIL import Image

from vlm_grpo.prompts import build_reflection_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Maximum image dimension (resize larger images to save memory)
MAX_IMAGE_DIM = 1024


def load_grpo_dataset(
    dataset_path: str,
    image_base_dir: str = "/outputs/image_base",
    max_samples: int = 0,
) -> Dataset:
    """Load precomputed Answer1-correct JSONL and convert to HF Dataset.

    The JSONL file should have rows with:
    - image_path: str (absolute or relative to image_base_dir)
    - question: str
    - ground_truth: str
    - answer1: str
    - answer_type: str ("mcq", "yesno", "numeric", "open")
    - choices: str (comma-separated for MCQ, empty otherwise)
    - dataset_name: str

    Args:
        dataset_path: Path to precomputed Answer1-correct JSONL
        image_base_dir: Base directory for resolving relative image paths
        max_samples: Maximum samples to load (0 = all)

    Returns:
        HuggingFace Dataset with columns: prompt, images, ground_truth,
        answer1, answer_type, choices, dataset_name
    """
    logger.info(f"Loading dataset from {dataset_path}")

    raw_samples = _load_jsonl(dataset_path, max_samples)
    logger.info(f"Loaded {len(raw_samples)} raw samples")

    # Convert to TRL format
    records = {
        "prompt": [],
        "images": [],
        "ground_truth": [],
        "answer1": [],
        "answer_type": [],
        "choices": [],
        "dataset_name": [],
    }

    skipped = 0
    for i, sample in enumerate(raw_samples):
        image_path = sample.get("image_path", "")
        question = sample.get("question", "")
        ground_truth = sample.get("ground_truth", "")
        answer1 = sample.get("answer1", "")
        answer_type = sample.get("answer_type", "open")
        choices = sample.get("choices", "")
        dataset_name = sample.get("dataset_name", "unknown")

        # Resolve image path
        if not os.path.isabs(image_path):
            image_path = os.path.join(image_base_dir, image_path)

        # Load image
        image = load_image_safe(image_path)
        if image is None:
            skipped += 1
            continue

        # Clean question (remove <image> placeholder)
        question = question.replace("<image>", "").strip()

        # Build prompt messages
        prompt = build_reflection_prompt(question, answer1, answer_type, choices)

        records["prompt"].append(prompt)
        records["images"].append([image])
        records["ground_truth"].append(ground_truth)
        records["answer1"].append(answer1)
        records["answer_type"].append(answer_type)
        records["choices"].append(choices)
        records["dataset_name"].append(dataset_name)

    if skipped > 0:
        logger.warning(f"Skipped {skipped} samples due to missing/invalid images")

    dataset = Dataset.from_dict(records)
    logger.info(f"Created dataset with {len(dataset)} samples")

    return dataset


def load_image_safe(image_path: str) -> Optional[Image.Image]:
    """Load image from path with error handling and RGB conversion.

    Resizes images larger than MAX_IMAGE_DIM to save GPU memory.

    Args:
        image_path: Absolute path to image file

    Returns:
        PIL Image in RGB mode, or None if loading failed
    """
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize if too large
        w, h = image.size
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
    - MCQ: question contains choice patterns (A), (B), (C)...
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

    # Check for MCQ pattern in question
    mcq_pattern = re.compile(r"\([A-F]\)\s*\w+", re.IGNORECASE)
    if mcq_pattern.search(question):
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


# =============================================================================
# Private Helpers
# =============================================================================


def load_critic_dataset(
    dataset_path: str,
    image_base_dir: str = "/outputs/image_base",
    max_samples: int = 0,
) -> Dataset:
    """Load dataset for critic GRPO training.

    Builds critic prompts (role-flipped) where the model generates
    feedback given an image, question, and initial answer.

    The JSONL file should have rows with:
    - image_path, question, ground_truth, answer1, answer_type, choices, dataset_name
    - Optionally: a1_is_correct (defaults to True for rw_first phase)

    Args:
        dataset_path: Path to JSONL dataset
        image_base_dir: Base directory for resolving relative image paths
        max_samples: Maximum samples to load (0 = all)

    Returns:
        HuggingFace Dataset with columns: prompt, images, ground_truth,
        answer1, answer_type, choices, dataset_name, a1_is_correct
    """
    from vlm_grpo.prompts import build_critic_prompt

    logger.info(f"Loading critic dataset from {dataset_path}")

    raw_samples = _load_jsonl(dataset_path, max_samples)
    logger.info(f"Loaded {len(raw_samples)} raw samples")

    records: dict[str, list] = {
        "prompt": [],
        "images": [],
        "ground_truth": [],
        "answer1": [],
        "answer_type": [],
        "choices": [],
        "dataset_name": [],
        "a1_is_correct": [],
    }

    skipped = 0
    for sample in raw_samples:
        image_path = sample.get("image_path", "")
        question = sample.get("question", "")
        ground_truth = sample.get("ground_truth", "")
        answer1 = sample.get("answer1", "")
        answer_type = sample.get("answer_type", "open")
        choices_str = sample.get("choices", "")
        dataset_name = sample.get("dataset_name", "unknown")
        a1_is_correct = sample.get("a1_is_correct", True)

        if not os.path.isabs(image_path):
            image_path = os.path.join(image_base_dir, image_path)

        image = load_image_safe(image_path)
        if image is None:
            skipped += 1
            continue

        question = question.replace("<image>", "").strip()
        prompt = build_critic_prompt(question, answer1, answer_type, choices_str)

        records["prompt"].append(prompt)
        records["images"].append([image])
        records["ground_truth"].append(ground_truth)
        records["answer1"].append(answer1)
        records["answer_type"].append(answer_type)
        records["choices"].append(choices_str)
        records["dataset_name"].append(dataset_name)
        records["a1_is_correct"].append(a1_is_correct)

    if skipped > 0:
        logger.warning(f"Skipped {skipped} samples due to missing/invalid images")

    dataset = Dataset.from_dict(records)
    logger.info(f"Created critic dataset with {len(dataset)} samples")

    return dataset


def load_refiner_dataset(
    dataset_path: str,
    image_base_dir: str = "/outputs/image_base",
    feedback_path: str = "",
    max_samples: int = 0,
) -> Dataset:
    """Load dataset for refiner GRPO training.

    Builds refiner prompts where the model generates a refined answer (A2)
    given an image, question, initial answer (A1), and feedback (F1).

    Feedback can come from:
    1. A separate JSONL file (feedback_path) with sample_index → feedback mapping
    2. A "feedback1" field in the main dataset JSONL

    Args:
        dataset_path: Path to JSONL dataset
        image_base_dir: Base directory for resolving relative image paths
        feedback_path: Path to JSONL with pre-computed feedbacks
        max_samples: Maximum samples to load (0 = all)

    Returns:
        HuggingFace Dataset with columns: prompt, images, ground_truth,
        answer1, feedback1, answer_type, choices, dataset_name, a1_is_correct
    """
    from vlm_grpo.prompts import build_refiner_prompt

    logger.info(f"Loading refiner dataset from {dataset_path}")

    raw_samples = _load_jsonl(dataset_path, max_samples)
    logger.info(f"Loaded {len(raw_samples)} raw samples")

    # Load pre-computed feedbacks if provided
    feedbacks: dict[int, str] = {}
    if feedback_path and os.path.exists(feedback_path):
        logger.info(f"Loading pre-computed feedbacks from {feedback_path}")
        feedback_samples = _load_jsonl(feedback_path)
        for fb_sample in feedback_samples:
            idx = fb_sample.get("sample_index", -1)
            text = fb_sample.get("feedback", "")
            if idx >= 0 and text:
                feedbacks[idx] = text
        logger.info(f"Loaded {len(feedbacks)} feedbacks")

    records: dict[str, list] = {
        "prompt": [],
        "images": [],
        "ground_truth": [],
        "answer1": [],
        "feedback1": [],
        "answer_type": [],
        "choices": [],
        "dataset_name": [],
        "a1_is_correct": [],
    }

    skipped = 0
    for i, sample in enumerate(raw_samples):
        image_path = sample.get("image_path", "")
        question = sample.get("question", "")
        ground_truth = sample.get("ground_truth", "")
        answer1 = sample.get("answer1", "")
        answer_type = sample.get("answer_type", "open")
        choices_str = sample.get("choices", "")
        dataset_name = sample.get("dataset_name", "unknown")
        a1_is_correct = sample.get("a1_is_correct", True)

        # Get feedback: from feedback_path, or from sample itself
        feedback1 = feedbacks.get(i, sample.get("feedback1", ""))
        if not feedback1:
            skipped += 1
            continue

        if not os.path.isabs(image_path):
            image_path = os.path.join(image_base_dir, image_path)

        image = load_image_safe(image_path)
        if image is None:
            skipped += 1
            continue

        question = question.replace("<image>", "").strip()
        prompt = build_refiner_prompt(
            question, answer1, feedback1, answer_type, choices_str
        )

        records["prompt"].append(prompt)
        records["images"].append([image])
        records["ground_truth"].append(ground_truth)
        records["answer1"].append(answer1)
        records["feedback1"].append(feedback1)
        records["answer_type"].append(answer_type)
        records["choices"].append(choices_str)
        records["dataset_name"].append(dataset_name)
        records["a1_is_correct"].append(a1_is_correct)

    if skipped > 0:
        logger.warning(f"Skipped {skipped} samples due to missing images/feedback")

    dataset = Dataset.from_dict(records)
    logger.info(f"Created refiner dataset with {len(dataset)} samples")

    return dataset


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
