#!/usr/bin/env python3
"""
Label each sample in a training JSONL with a correct answer_type tag.

Reads any JSONL file in the training data format (flat or messages-format)
and assigns one of 5 answer_type tags based on the ground truth:

    mcq       — single letter A-F, or question contains choice patterns
    yesno     — ground truth is yes/no/true/false
    numeric   — ground truth parses as a number (int, float, percentage)
    counting  — question asks "how many" / "number of" AND GT is numeric
    open      — everything else (freeform text)

This script should be run as the LAST preprocessing step before training,
so the answer_type field is authoritative and the data loader can trust it.

Usage:
    # Label a dataset in-place (overwrites answer_type field)
    python scripts/label_answer_types.py \
        --input /outputs/grpo_data/answer1_correct_train.jsonl \
        --output /outputs/grpo_data/answer1_correct_train_labeled.jsonl

    # Overwrite the original file
    python scripts/label_answer_types.py \
        --input /outputs/grpo_data/train.jsonl \
        --output /outputs/grpo_data/train.jsonl

    # Dry-run: show reclassification statistics without writing
    python scripts/label_answer_types.py \
        --input /outputs/grpo_data/train.jsonl \
        --dry_run
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# =============================================================================
# Answer Type Detection (self-contained, no imports from src/)
# =============================================================================

# MCQ choice patterns in questions
_MCQ_PAREN_RE = re.compile(r"\([A-F]\)\s*\w+", re.IGNORECASE)
_MCQ_DOT_RE = re.compile(r"^[A-F]\.\s*.+", re.MULTILINE | re.IGNORECASE)

# Counting question patterns
_COUNTING_RE = re.compile(r"\bhow\s+many\b|\bnumber\s+of\b", re.IGNORECASE)


def detect_answer_type(
    question: str,
    ground_truth: str,
    choices: str = "",
) -> str:
    """Detect answer type from question and ground truth characteristics.

    Priority order:
    1. Explicit choices field → mcq
    2. MCQ patterns in question → mcq
    3. GT is yes/no/true/false → yesno
    4. GT is single letter A-F → mcq
    5. GT is numeric AND question is counting → counting
    6. GT is numeric → numeric
    7. Everything else → open

    Args:
        question: Question text
        ground_truth: Ground truth answer string
        choices: Explicitly provided choices (if any)

    Returns:
        One of: "mcq", "yesno", "numeric", "counting", "open"
    """
    gt_stripped = ground_truth.strip()
    gt_lower = gt_stripped.lower()

    # 1. Explicit choices
    if choices:
        return "mcq"

    # 2. MCQ patterns in question
    if _MCQ_PAREN_RE.search(question) or _MCQ_DOT_RE.search(question):
        return "mcq"

    # 3. Yes/No ground truth
    if gt_lower in ("yes", "no", "y", "n", "true", "false"):
        return "yesno"

    # 4. Single letter (likely MCQ without explicit choices)
    if len(gt_lower) == 1 and gt_lower in "abcdef":
        return "mcq"

    # 5-6. Numeric ground truth (with counting sub-type)
    if _is_numeric(gt_stripped):
        if _COUNTING_RE.search(question):
            return "counting"
        return "numeric"

    return "open"


def _is_numeric(text: str) -> bool:
    """Check if text represents a number (int, float, percentage, fraction).

    Args:
        text: Text to check

    Returns:
        True if text parses as a numeric value
    """
    cleaned = text.replace(",", "").strip()

    # Handle percentage: "35%" → "35"
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1].strip()

    # Handle fraction: "3/4"
    if "/" in cleaned:
        parts = cleaned.split("/")
        if len(parts) == 2:
            try:
                float(parts[0].strip())
                float(parts[1].strip())
                return True
            except ValueError:
                return False

    try:
        float(cleaned)
        return True
    except ValueError:
        return False


# =============================================================================
# JSONL Processing
# =============================================================================


def _extract_question(sample: dict) -> str:
    """Extract question text from flat or messages-format sample.

    Args:
        sample: JSONL record

    Returns:
        Question text string
    """
    # Flat format
    if "question" in sample:
        return sample["question"]

    # Messages format
    messages = sample.get("messages", [])
    for m in messages:
        if m.get("role") == "user":
            return m.get("content", "")

    return ""


def _extract_ground_truth(sample: dict) -> str:
    """Extract ground truth from flat or messages-format sample.

    Args:
        sample: JSONL record

    Returns:
        Ground truth text string
    """
    # Flat format
    if "ground_truth" in sample:
        return sample["ground_truth"]

    # Inference results format
    if "gt_final_answer" in sample:
        return sample["gt_final_answer"]

    # Messages format: last assistant message
    messages = sample.get("messages", [])
    for m in reversed(messages):
        if m.get("role") == "assistant":
            return m.get("content", "")

    return ""


def process_jsonl(
    input_path: str,
    output_path: str,
    dry_run: bool = False,
) -> None:
    """Read JSONL, label each sample with answer_type, write output.

    Args:
        input_path: Input JSONL path
        output_path: Output JSONL path (can be same as input)
        dry_run: If True, only print statistics without writing
    """
    logger.info(f"Reading {input_path}")

    samples = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    logger.info(f"Loaded {len(samples)} samples")

    # Track reclassifications
    old_type_counter: Counter = Counter()
    new_type_counter: Counter = Counter()
    changed = 0
    transition_counter: Counter = Counter()

    for sample in samples:
        question = _extract_question(sample)
        ground_truth = _extract_ground_truth(sample)
        choices = sample.get("choices", "")

        old_type = sample.get("answer_type", "unknown")
        # Also check category-based type for samples from filter script
        if old_type == "unknown" and "category" in sample:
            old_type = f"cat:{sample['category']}"

        new_type = detect_answer_type(question, ground_truth, choices)

        old_type_counter[old_type] += 1
        new_type_counter[new_type] += 1

        if old_type != new_type:
            changed += 1
            transition_counter[f"{old_type} -> {new_type}"] += 1

        sample["answer_type"] = new_type

    # Print statistics
    _print_statistics(
        len(samples),
        old_type_counter,
        new_type_counter,
        changed,
        transition_counter,
    )

    if dry_run:
        logger.info("Dry run — no output written")
        return

    # Write output (read all into memory first to support in-place overwrite)
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Wrote {len(samples)} labeled samples to {output_path}")


def _print_statistics(
    total: int,
    old_counts: Counter,
    new_counts: Counter,
    changed: int,
    transitions: Counter,
) -> None:
    """Print labeling statistics.

    Args:
        total: Total number of samples
        old_counts: Counter of old answer_type values
        new_counts: Counter of new answer_type values
        changed: Number of samples whose type changed
        transitions: Counter of "old -> new" transition strings
    """
    logger.info("=" * 60)
    logger.info("ANSWER TYPE LABELING STATISTICS")
    logger.info("=" * 60)

    logger.info(f"Total samples: {total}")
    logger.info(f"Reclassified:  {changed} ({changed / max(total, 1):.1%})")

    logger.info("")
    logger.info("Distribution BEFORE:")
    for t, c in sorted(old_counts.items()):
        logger.info(f"  {t:20s}: {c:6d} ({c / max(total, 1):6.1%})")

    logger.info("")
    logger.info("Distribution AFTER:")
    for t, c in sorted(new_counts.items()):
        logger.info(f"  {t:20s}: {c:6d} ({c / max(total, 1):6.1%})")

    if transitions:
        logger.info("")
        logger.info("Reclassifications:")
        for trans, c in sorted(transitions.items(), key=lambda x: -x[1]):
            logger.info(f"  {trans:35s}: {c:6d}")

    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Label training JSONL samples with correct answer_type tags",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSONL file path (defaults to --input if not specified)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print statistics, do not write output",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    output = args.output if args.output else args.input
    process_jsonl(args.input, output, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
