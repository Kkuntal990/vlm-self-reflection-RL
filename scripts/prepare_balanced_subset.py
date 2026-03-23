#!/usr/bin/env python3
"""
Prepare a balanced subset of the training data for GRPO experiments.

Reads multiple JSONL dataset files, classifies each sample by answer_type,
and creates a balanced subset with configurable target counts per type.

Default strategy (70K total):
  - MCQ: 13,000 (random sample)
  - Yes/No: 7,000 (random sample)
  - Counting: all (~4,500, take all)
  - Open-ended: ~22,750 (split remaining evenly)
  - Short answer: ~22,750 (split remaining evenly)

Usage:
    python scripts/prepare_balanced_subset.py \
        --input_files /outputs/mixed_training_v1/fire_messages_filtered.jsonl \
                      /outputs/mixed_training_v1/fire_messages_single_turn.jsonl \
                      /outputs/mixed_training_v1/lvlm_nlf_multiturn.jsonl \
                      /outputs/mixed_training_v1/lvlm_nlf_single_turn.jsonl \
                      /outputs/mixed_training_v1/vqa_aokvqa.jsonl \
                      /outputs/mixed_training_v1/vqa_scienceqa.jsonl \
                      /outputs/mixed_training_v1/vqa_tallyqa.jsonl \
        --output /outputs/grpo_data/balanced_70k.jsonl \
        --total 70000 \
        --mcq 13000 \
        --yesno 7000

    # Dry run (show statistics only)
    python scripts/prepare_balanced_subset.py \
        --input_files /outputs/mixed_training_v1/*.jsonl \
        --dry_run
"""

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def detect_answer_type(
    question: str,
    ground_truth: str,
    choices: str = "",
    category: str = "",
) -> str:
    """Classify a sample into an answer type.

    Args:
        question: Question text
        ground_truth: Ground truth answer string
        choices: Explicit choices string (if any)
        category: Dataset category field (if any)

    Returns:
        One of: "mcq", "yesno", "counting", "open", "short"
    """
    gt_lower = ground_truth.strip().lower()
    q_lower = question.lower()

    # Explicit category mapping
    category_map = {
        "mcq": "mcq",
        "yes_no": "yesno",
        "counting": "counting",
    }
    if category in category_map:
        return category_map[category]

    # Check for explicit choices
    if choices:
        return "mcq"

    # MCQ pattern in question: "(A) text" or "A. text"
    mcq_paren = re.compile(r"\([A-F]\)\s*\w+", re.IGNORECASE)
    mcq_dot = re.compile(r"^[A-F]\.\s*.+", re.MULTILINE | re.IGNORECASE)
    if mcq_paren.search(question) or mcq_dot.search(question):
        return "mcq"

    # Single letter answer (likely MCQ)
    if len(gt_lower) == 1 and gt_lower in "abcdef":
        return "mcq"

    # Yes/No
    if gt_lower in ("yes", "no", "y", "n", "true", "false"):
        return "yesno"

    # Counting: numeric GT + counting question pattern
    counting_patterns = [
        r"how many",
        r"number of",
        r"count",
        r"total number",
        r"how much",
        r"what is the number",
    ]
    is_counting_q = any(re.search(p, q_lower) for p in counting_patterns)
    try:
        float(ground_truth.strip().replace(",", ""))
        is_numeric = True
    except ValueError:
        is_numeric = False

    if is_numeric and is_counting_q:
        return "counting"

    # Numeric (non-counting)
    if is_numeric:
        return "short"

    # Short vs open: heuristic based on GT length
    gt_words = ground_truth.strip().split()
    if len(gt_words) <= 5:
        return "short"

    return "open"


def load_samples(input_files: list[str]) -> list[dict]:
    """Load and concatenate samples from multiple JSONL files.

    Args:
        input_files: List of paths to JSONL files

    Returns:
        List of sample dicts with source file info
    """
    all_samples = []
    for filepath in input_files:
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"File not found, skipping: {filepath}")
            continue

        count = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    sample["_source_file"] = path.name
                    all_samples.append(sample)
                    count += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in {path.name}: {e}")

        logger.info(f"Loaded {count} samples from {path.name}")

    logger.info(f"Total samples loaded: {len(all_samples)}")
    return all_samples


def extract_fields(sample: dict) -> dict:
    """Extract question, ground_truth, and other fields from a sample.

    Handles both flat JSONL and messages-format JSONL.

    Args:
        sample: Raw JSONL record

    Returns:
        Dict with question, ground_truth, answer_type, choices, etc.
    """
    if "messages" in sample and "question" not in sample:
        messages = sample.get("messages", [])
        user_msgs = [m for m in messages if m["role"] == "user"]
        assistant_msgs = [m for m in messages if m["role"] == "assistant"]
        question = user_msgs[0]["content"] if user_msgs else ""
        ground_truth = assistant_msgs[-1]["content"] if assistant_msgs else ""
    else:
        question = sample.get("question", "")
        ground_truth = sample.get("ground_truth", "")

    return {
        "question": question,
        "ground_truth": ground_truth,
        "choices": sample.get("choices", ""),
        "category": sample.get("category", ""),
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Prepare balanced dataset subset for GRPO training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input_files",
        nargs="+",
        required=True,
        help="Input JSONL files to read from",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSONL file path",
    )
    parser.add_argument("--total", type=int, default=70000, help="Target total samples")
    parser.add_argument("--mcq", type=int, default=13000, help="Target MCQ count")
    parser.add_argument("--yesno", type=int, default=7000, help="Target Yes/No count")
    parser.add_argument(
        "--counting", type=int, default=0, help="Target counting count (0 = take all)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry_run", action="store_true", help="Show statistics without writing")

    return parser.parse_args()


def main() -> None:
    """Main function for balanced dataset preparation."""
    args = parse_args()
    random.seed(args.seed)

    # Load all samples
    all_samples = load_samples(args.input_files)
    if not all_samples:
        logger.error("No samples loaded. Check input file paths.")
        return

    # Classify each sample by answer type
    type_buckets: dict[str, list[dict]] = defaultdict(list)
    for sample in all_samples:
        fields = extract_fields(sample)
        answer_type = detect_answer_type(
            fields["question"],
            fields["ground_truth"],
            fields["choices"],
            fields["category"],
        )
        sample["answer_type"] = answer_type
        type_buckets[answer_type].append(sample)

    # Print distribution
    logger.info("\n=== Original Distribution ===")
    total_orig = len(all_samples)
    for atype in ["mcq", "yesno", "counting", "short", "open"]:
        count = len(type_buckets[atype])
        pct = 100 * count / total_orig if total_orig > 0 else 0
        logger.info(f"  {atype:>10s}: {count:>7d} ({pct:>5.1f}%)")
    logger.info(f"  {'TOTAL':>10s}: {total_orig:>7d}")

    if args.dry_run:
        logger.info("\nDry run complete. Use --output to write balanced subset.")
        return

    if not args.output:
        logger.error("--output is required (or use --dry_run)")
        return

    # Determine target counts
    # Fixed targets
    mcq_target = min(args.mcq, len(type_buckets["mcq"]))
    yesno_target = min(args.yesno, len(type_buckets["yesno"]))
    counting_target = (
        len(type_buckets["counting"])
        if args.counting == 0
        else min(args.counting, len(type_buckets["counting"]))
    )

    fixed_total = mcq_target + yesno_target + counting_target
    remaining = args.total - fixed_total

    # Split remaining evenly between open and short
    open_target = remaining // 2
    short_target = remaining - open_target  # handle odd number

    # Cap at available
    open_target = min(open_target, len(type_buckets["open"]))
    short_target = min(short_target, len(type_buckets["short"]))

    targets = {
        "mcq": mcq_target,
        "yesno": yesno_target,
        "counting": counting_target,
        "open": open_target,
        "short": short_target,
    }

    logger.info("\n=== Target Distribution ===")
    actual_total = 0
    for atype in ["mcq", "yesno", "counting", "short", "open"]:
        available = len(type_buckets[atype])
        target = targets[atype]
        logger.info(f"  {atype:>10s}: {target:>7d} / {available:>7d} available")
        actual_total += target
    logger.info(f"  {'TOTAL':>10s}: {actual_total:>7d}")

    # Sample from each bucket
    selected: list[dict] = []
    for atype, target in targets.items():
        bucket = type_buckets[atype]
        if target >= len(bucket):
            selected.extend(bucket)
        else:
            selected.extend(random.sample(bucket, target))

    # Shuffle the final dataset
    random.shuffle(selected)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for sample in selected:
            # Remove internal metadata
            sample.pop("_source_file", None)
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    logger.info(f"\nWrote {len(selected)} samples to {output_path}")

    # Verify final distribution
    final_counts: Counter[str] = Counter()
    for s in selected:
        final_counts[s["answer_type"]] += 1

    logger.info("\n=== Final Distribution ===")
    for atype in ["mcq", "yesno", "counting", "short", "open"]:
        count = final_counts[atype]
        pct = 100 * count / len(selected) if selected else 0
        logger.info(f"  {atype:>10s}: {count:>7d} ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
