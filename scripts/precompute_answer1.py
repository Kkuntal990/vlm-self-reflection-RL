#!/usr/bin/env python3
"""
Precompute Answer1 and filter to correct samples for GRPO training.

This script:
1. Loads the FIRE dataset (inference results or raw dataset)
2. Generates Answer1 for each sample using the target VLM (if not precomputed)
3. Evaluates correctness using deterministic matching
4. Outputs a filtered JSONL of Answer1-correct samples

The output JSONL is the input to train_grpo_rw.py.

Usage:
    # From existing inference results (recommended - no GPU needed)
    python scripts/precompute_answer1.py \
        --inference_results /path/to/inference_results.jsonl \
        --output_path /outputs/grpo_data/answer1_correct_train.jsonl \
        --image_base_dir /outputs/image_base

    # From raw FIRE dataset (requires GPU for generation)
    accelerate launch --num_processes=4 scripts/precompute_answer1.py \
        --model_path /outputs/llava-1.5-sft-checkpoint \
        --dataset_path /outputs/fire_preprocessed/fire_messages_train.jsonl \
        --output_path /outputs/grpo_data/answer1_correct_train.jsonl \
        --image_base_dir /outputs/image_base
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vlm_grpo.data import detect_answer_type
from vlm_grpo.rewards.deterministic import match_answer
from vlm_grpo.trajectory import extract_answer_from_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Precompute Answer1 and filter to correct samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input sources (one of these required)
    parser.add_argument(
        "--inference_results",
        type=str,
        default="",
        help="Path to existing inference results JSONL (has answer1 precomputed)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="",
        help="Path to raw FIRE dataset JSONL (requires model for generation)",
    )

    # Model for generation (only needed if --dataset_path is used)
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Model path for generating Answer1 (only with --dataset_path)",
    )

    # Output
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output JSONL path for Answer1-correct samples",
    )

    # Configuration
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="/outputs/image_base",
        help="Base directory for resolving relative image paths",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum samples to process (0 = all)",
    )

    return parser.parse_args()


def process_inference_results(
    results_path: str,
    image_base_dir: str,
    max_samples: int = 0,
) -> list[dict]:
    """Process existing inference results to extract Answer1-correct samples.

    Expected input format (from self_reflective_inference_v2.py):
    {
        "sample_index": 0,
        "image_path": "/outputs/image_base/mathvista/images/660.jpg",
        "question": "...",
        "generated_turns": [{"answer": "(A) Yes", "feedback": "..."}],
        "final_answer": "...",
        "gt_final_answer": "No",
    }

    Args:
        results_path: Path to inference results JSONL
        image_base_dir: Base directory for images
        max_samples: Max samples to process (0 = all)

    Returns:
        List of Answer1-correct sample dicts
    """
    logger.info(f"Processing inference results from {results_path}")

    correct_samples = []
    total = 0
    correct = 0
    skipped = 0

    with open(results_path) as f:
        for i, line in enumerate(tqdm(f, desc="Processing")):
            if max_samples > 0 and i >= max_samples:
                break

            try:
                sample = json.loads(line.strip())
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {i}: {e}")
                continue

            total += 1

            # Extract fields
            image_path = sample.get("image_path", "")
            question = sample.get("question", "")
            gt = sample.get("gt_final_answer", "")
            turns = sample.get("generated_turns", [])

            if not turns:
                skipped += 1
                continue

            answer1 = turns[0].get("answer", "")
            if not answer1:
                skipped += 1
                continue

            # Detect answer type
            choices = _extract_choices_from_question(question)
            answer_type = detect_answer_type(question, gt, choices)

            # Detect dataset name from image path
            dataset_name = _detect_dataset_name(image_path)

            # Check Answer1 correctness
            a1_extracted = extract_answer_from_text(answer1, answer_type, choices)
            if not a1_extracted:
                # Fallback: use raw answer1
                a1_extracted = answer1.strip()

            is_correct = match_answer(a1_extracted, gt, answer_type)

            if is_correct is True:
                correct += 1
                correct_samples.append({
                    "sample_index": sample.get("sample_index", i),
                    "image_path": image_path,
                    "question": question,
                    "ground_truth": gt,
                    "answer1": answer1,
                    "answer_type": answer_type,
                    "choices": choices,
                    "dataset_name": dataset_name,
                })

    logger.info(
        f"Processed {total} samples: {correct} correct ({correct / max(total, 1):.1%}), "
        f"{skipped} skipped"
    )

    return correct_samples


def _extract_choices_from_question(question: str) -> str:
    """Extract MCQ choices from question text.

    Looks for patterns like "(A) Yes (B) No" or "Choices:\\n(A) ..."

    Args:
        question: Question text

    Returns:
        Comma-separated choices string, or empty string
    """
    # Pattern: (A) text (B) text ...
    pattern = re.compile(r"\(([A-F])\)\s*([^(]+?)(?=\s*\([A-F]\)|$)", re.DOTALL)
    matches = pattern.findall(question)

    if len(matches) >= 2:
        choices = [f"({letter}) {text.strip()}" for letter, text in matches]
        return ", ".join(choices)

    return ""


def _detect_dataset_name(image_path: str) -> str:
    """Detect dataset name from image path.

    Args:
        image_path: Image file path

    Returns:
        Dataset name string
    """
    path_lower = image_path.lower()

    known_datasets = [
        "mathvista",
        "scienceqa",
        "seed_bench",
        "ai2d",
        "chartqa",
        "docvqa",
        "infovqa",
        "ocrvqa",
        "textvqa",
        "gqa",
        "vqav2",
        "okvqa",
        "vizwiz",
        "realworldqa",
    ]

    for ds in known_datasets:
        if ds in path_lower:
            return ds

    return "unknown"


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not args.inference_results and not args.dataset_path:
        logger.error("Must provide either --inference_results or --dataset_path")
        sys.exit(1)

    if args.inference_results:
        samples = process_inference_results(
            args.inference_results,
            args.image_base_dir,
            args.max_samples,
        )
    elif args.dataset_path:
        if not args.model_path:
            logger.error("--model_path required when using --dataset_path")
            sys.exit(1)
        logger.error(
            "Online Answer1 generation not yet implemented. "
            "Use --inference_results with precomputed results instead."
        )
        sys.exit(1)

    # Save output
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Saved {len(samples)} Answer1-correct samples to {args.output_path}")

    # Print dataset statistics
    _print_statistics(samples)


def _print_statistics(samples: list[dict]) -> None:
    """Print dataset statistics.

    Args:
        samples: List of Answer1-correct sample dicts
    """
    logger.info("=" * 50)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 50)

    # By answer type
    type_counts: dict[str, int] = {}
    for s in samples:
        a_type = s.get("answer_type", "unknown")
        type_counts[a_type] = type_counts.get(a_type, 0) + 1

    logger.info("By answer type:")
    for a_type, count in sorted(type_counts.items()):
        logger.info(f"  {a_type}: {count} ({count / len(samples):.1%})")

    # By dataset
    ds_counts: dict[str, int] = {}
    for s in samples:
        ds = s.get("dataset_name", "unknown")
        ds_counts[ds] = ds_counts.get(ds, 0) + 1

    logger.info("By dataset:")
    for ds, count in sorted(ds_counts.items()):
        logger.info(f"  {ds}: {count} ({count / len(samples):.1%})")

    logger.info("=" * 50)


if __name__ == "__main__":
    main()
