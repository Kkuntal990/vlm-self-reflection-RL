#!/usr/bin/env python3
"""
Prepare a balanced subset of the training data for GRPO experiments.

Two-step classification pipeline:
1. Extract the actual answer from GT (strip "Thought: ... Answer: ..." wrapper,
   strip "Rationale: ..." suffix)
2. Classify using BOTH the question text and extracted answer

Default strategy (70K total):
  - MCQ: 13,000 (random sample)
  - Yes/No: 7,000 (random sample)
  - Counting: all (~5,700, take all)
  - Numeric: merge into short
  - Open-ended + Short: split remaining evenly

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
        --total 70000 --mcq 13000 --yesno 7000

    # Dry run (show statistics only)
    python scripts/prepare_balanced_subset.py \
        --input_files /outputs/mixed_training_v1/*.jsonl --dry_run
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


# =============================================================================
# Answer Extraction
# =============================================================================


def extract_answer_from_gt(raw_gt: str) -> str:
    """Extract the actual answer from a GT that may have Thought/Answer format.

    Handles:
    - "Thought: ... Answer: actual answer" → "actual answer"
    - "Answer: office.\nRationale: ..." → "office."
    - Multiple "Answer:" tags → uses the LAST one
    - Plain text (no tags) → returns as-is

    Args:
        raw_gt: Raw ground truth text from the dataset

    Returns:
        Extracted answer string, stripped of thought/rationale wrappers
    """
    # Find all "Answer:" occurrences, use the last one
    matches = list(re.finditer(r"Answer:\s*(.*?)(?=\nThought:|\Z)", raw_gt, re.DOTALL))
    if matches:
        answer = matches[-1].group(1).strip()
        # Strip rationale suffix (aokvqa format: "Answer: X\nRationale: ...")
        if "\nRationale:" in answer:
            answer = answer.split("\nRationale:")[0].strip()
        return answer
    return raw_gt.strip()


# =============================================================================
# Answer Type Classification
# =============================================================================


def classify_sample(question: str, extracted_answer: str) -> str:
    """Classify a sample using BOTH the question and extracted answer.

    Uses question patterns to detect MCQ (options/choices) and Yes/No
    (interrogative form), combined with answer format analysis.

    Fixes from audit (v2):
    - Filter refusal/safety responses ("I cannot", "I'm sorry")
    - Exclude proper names from MCQ detection ("A. Singh" is NOT MCQ)
    - Detect inline options ("based on the options") for MCQ
    - Detect lowercase (a)/(b)/(c)/(d) MCQ patterns
    - Exclude years (1900-2099) and large IDs (>5 digits) from counting

    Args:
        question: Question text
        extracted_answer: Answer text after extraction from GT

    Returns:
        One of: "mcq", "yesno", "counting", "numeric", "short", "open",
        or "_refusal" for filtered samples
    """
    a = extracted_answer.strip()
    a_lower = a.lower().rstrip(".")
    q_lower = question.lower()

    # --- Filter: refusal/safety responses are not useful for training ---
    if re.search(
        r"\b(i cannot|i'm sorry|i apologize|i'm unable|as an ai)\b",
        a_lower,
    ):
        return "_refusal"

    # --- MCQ detection ---

    # 1. Single letter A-F → MCQ
    if len(a_lower) == 1 and a_lower in "abcdef":
        return "mcq"

    # 2. Answer starts with MCQ pattern: '(A)', 'A.', 'A)' + text
    # Exclude proper names: "A. Singh" (uppercase after space) is NOT MCQ
    # Allow: "A. 50 degrees" (digit after space) IS MCQ
    mcq_prefix = re.match(r"^(?:\([A-F]\)|[A-F][).])\s+(\S)", a)
    if mcq_prefix:
        next_char = mcq_prefix.group(1)
        if not next_char.isupper() or next_char.isdigit():
            return "mcq"

    # 3. Answer starts with lowercase (a)/(b)/(c)/(d) → MCQ
    if re.match(r"^\([a-d]\)\s", a):
        return "mcq"

    # 4. Question has explicit options AND answer is short (<=5 words)
    has_options = bool(
        re.search(r"\([A-F]\)\s*\w", question, re.IGNORECASE)
        or re.search(r"^[A-F]\.\s*.+", question, re.MULTILINE | re.IGNORECASE)
        or re.search(r"\bOptions?:\s", question, re.IGNORECASE)
        or re.search(r"\bchoices?\s*(?:given|are|:)", question, re.IGNORECASE)
        or re.search(r"\bselect\b.*\b(?:answer|option|choice)", question, re.IGNORECASE)
        or re.search(r"\bbased on the options\b", question, re.IGNORECASE)
        or re.search(
            r"\bfrom the (?:following|given|above) (?:options|choices)\b",
            question,
            re.IGNORECASE,
        )
    )
    if has_options and len(a.split()) <= 5:
        return "mcq"

    # --- Yes/No detection ---

    # 5. Bare yes/no answer
    if a_lower in ("yes", "no"):
        return "yesno"

    # 6. Question is yes/no form AND answer starts with yes/no
    yesno_q = bool(
        re.match(
            r"^(?:is |are |was |were |do |does |did |can |could |will |would "
            r"|has |have |had |should )",
            q_lower,
        )
    )
    if yesno_q and re.match(r"^(yes|no)\b", a_lower):
        return "yesno"

    # --- Numeric / Counting detection ---

    # Strip trailing %, °, and common suffixes for numeric check
    num_text = a_lower.replace(",", "").rstrip("%°")
    try:
        float(num_text)
        is_numeric = True
    except ValueError:
        is_numeric = False

    if is_numeric:
        counting_pats = [
            r"\bhow many\b",
            r"\bnumber of\b",
            r"\bcount\b",
            r"\btotal number\b",
            r"\bhow much\b",
        ]
        is_counting_q = any(re.search(p, q_lower) for p in counting_pats)

        # Exclude years (4-digit 1900-2099) and large IDs (>5 digits)
        is_year = bool(re.match(r"^(19|20)\d{2}$", num_text))
        is_large_id = len(num_text.replace(".", "").replace("-", "")) > 5

        if is_counting_q and not is_year and not is_large_id:
            return "counting"
        return "numeric"

    # --- Short vs Open ---
    if len(a.split()) <= 5:
        return "short"
    return "open"


# =============================================================================
# Data Loading
# =============================================================================


def load_and_process_samples(input_files: list[str]) -> list[dict]:
    """Load JSONL files, extract answers, classify, and return processed samples.

    Each output sample has:
    - Original fields (messages, images, etc.)
    - answer_type: classified type
    - ground_truth: extracted answer (stripped of Thought/Rationale)
    - question: extracted question text
    - _source_file: source filename (removed before writing)

    Args:
        input_files: Paths to JSONL files

    Returns:
        List of processed sample dicts
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
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line in {path.name}: {e}")
                    continue

                # Extract question and raw GT from messages format
                if "messages" in sample:
                    msgs = sample["messages"]
                    user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
                    asst_msgs = [m["content"] for m in msgs if m["role"] == "assistant"]
                    question = user_msgs[0] if user_msgs else ""
                    raw_gt = asst_msgs[-1] if asst_msgs else ""
                else:
                    question = sample.get("question", "")
                    raw_gt = sample.get("ground_truth", "")

                # Step 1: Extract answer from GT
                extracted = extract_answer_from_gt(raw_gt)

                # Step 2: Classify using question + extracted answer
                answer_type = classify_sample(question, extracted)

                # Store processed fields
                sample["question"] = question.replace("<image>", "").strip()
                sample["ground_truth"] = extracted
                sample["answer_type"] = answer_type
                sample["_source_file"] = path.name

                all_samples.append(sample)
                count += 1

        logger.info(f"Loaded {count} samples from {path.name}")

    logger.info(f"Total samples loaded: {len(all_samples)}")
    return all_samples


# =============================================================================
# CLI
# =============================================================================


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
        "--counting",
        type=int,
        default=0,
        help="Target counting count (0 = take all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry_run", action="store_true", help="Show statistics without writing")

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Main function for balanced dataset preparation."""
    args = parse_args()
    random.seed(args.seed)

    # Load, extract answers, and classify
    all_samples = load_and_process_samples(args.input_files)
    if not all_samples:
        logger.error("No samples loaded. Check input file paths.")
        return

    # Bucket by answer type (merge numeric into short, filter refusals)
    type_buckets: dict[str, list[dict]] = defaultdict(list)
    refusal_count = 0
    for sample in all_samples:
        at = sample["answer_type"]
        if at == "_refusal":
            refusal_count += 1
            continue
        if at == "numeric":
            at = "short"
            sample["answer_type"] = "short"
        type_buckets[at].append(sample)

    if refusal_count > 0:
        logger.info(f"Filtered {refusal_count} refusal/safety samples")

    # Print distribution
    logger.info("\n=== Original Distribution ===")
    total_orig = len(all_samples)
    for atype in ["mcq", "yesno", "counting", "short", "open"]:
        count = len(type_buckets[atype])
        pct = 100 * count / total_orig if total_orig > 0 else 0
        logger.info(f"  {atype:>10s}: {count:>7d} ({pct:>5.1f}%)")
    logger.info(f"  {'TOTAL':>10s}: {total_orig:>7d}")

    if args.dry_run:
        # Show per-source breakdown
        logger.info("\n=== Per-Source Breakdown ===")
        source_type_counts: dict[str, Counter] = defaultdict(Counter)
        for s in all_samples:
            source_type_counts[s["_source_file"]][s["answer_type"]] += 1
        for src, counts in sorted(source_type_counts.items()):
            total_src = sum(counts.values())
            parts = [f"{at}={c}" for at, c in sorted(counts.items()) if c > 0]
            logger.info(f"  {src}: {total_src} total — {', '.join(parts)}")

        logger.info("\nDry run complete. Use --output to write balanced subset.")
        return

    if not args.output:
        logger.error("--output is required (or use --dry_run)")
        return

    # Determine target counts
    mcq_target = min(args.mcq, len(type_buckets["mcq"]))
    yesno_target = min(args.yesno, len(type_buckets["yesno"]))
    counting_target = (
        len(type_buckets["counting"])
        if args.counting == 0
        else min(args.counting, len(type_buckets["counting"]))
    )

    fixed_total = mcq_target + yesno_target + counting_target
    remaining = args.total - fixed_total

    # Split remaining evenly between open and short, overflow to the other
    short_available = len(type_buckets["short"])
    open_available = len(type_buckets["open"])
    short_target = min(remaining // 2, short_available)
    open_target = min(remaining - short_target, open_available)

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
