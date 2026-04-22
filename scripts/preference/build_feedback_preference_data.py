#!/usr/bin/env python3
"""Build pairwise FEEDBACK preference data for critic GRPO warm-up.

===========================================================================
DESIGN OVERVIEW — Feedback Preference GRPO for Critic Warm-Up
===========================================================================

Problem:
    The critic (F1) in our A1→F1→A2 pipeline is sycophantic — it says
    "correct" 81% of the time even when A1 is wrong. We need to train
    the model to produce calibrated critiques that correctly identify
    errors and provide useful corrections.

Key Insight:
    We don't need answer preference data ("which answer is better?").
    We need FEEDBACK preference data ("which critique is better?").
    Training the model to judge answers ≠ training it to critique them.

Approach:
    1. Generate A1 (initial answer) for each LIVR question (greedy)
    2. We know ground truth → know if A1 is right or wrong
    3. Generate K=16 diverse F1 critiques per A1
    4. For each F1, generate A2 (greedy) to measure downstream impact
    5. Score each F1 by:
       - Calibration: did F1 correctly identify A1 as right/wrong?
       - Downstream: did F1 lead to a correct A2?
    6. Pair high-scoring F1 vs low-scoring F1 → feedback preference data

Pipeline:
    Step 1 — `generate`: For each LIVR question:
             a) Generate A1 (greedy, deterministic baseline)
             b) Generate K=16 F1 critiques (temp=1.0, diverse)
             c) For each F1, generate A2 (greedy)
             d) Score everything
    Step 2 — `build`: Construct preference pairs:
             For each question with mixed-quality F1s:
             Pair a "good" F1 vs a "bad" F1
             Good = correctly calibrated AND/OR led to better A2
             Bad = miscalibrated AND/OR led to worse A2
    Step 3 — `eval`: Measure baseline sycophancy rate before warm-up

Expected Yield:
    With 9K LIVR MCQ questions and K=16 critiques per A1:
    - ~50% of A1s are wrong → F1 quality varies widely for wrong A1s
    - Among wrong A1s: some F1s correctly identify error, some don't
    - Among right A1s: some F1s correctly confirm, some sow doubt
    - ~60-70% of questions will have mixed-quality F1s
    - ~5.4K-6.3K usable questions × 3 pairs = 16K-19K feedback pairs

Training Recipe (Feedback GRPO warm-up):
    Uses our EXISTING A1→F1→A2 pipeline but on constructed data where
    we KNOW which F1s are good. The F1 reward is:
      - Calibration: +1 if F1 correctly assesses A1, -1 if wrong
      - Downstream: +1 if A2 correct, -1 if wrong
    This is the same reward structure as our main training, but the
    data is curated so that K-groups always have non-zero variance
    (we only include questions with mixed F1 quality).

    Alternative: DPO on (good_F1, bad_F1) pairs in the critic prompt
    format. Simpler than GRPO, doesn't need rollouts during training.

Evaluation:
    After warm-up, measure on held-out LIVR questions:
    1. Sycophancy rate: F1 says "correct" when A1 wrong (target: <50%)
    2. False negative rate: F1 says "wrong" when A1 right (target: <30%)
    3. Downstream WR rate in full A1→F1→A2 (target: >10%)

===========================================================================

Usage:
    # Step 1: Generate A1 + K critiques + A2 per question (GPU)
    uv run python scripts/build_feedback_preference_data.py generate \\
        --dataset_path /outputs/livr_data/livr_perception_mcq.jsonl \\
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \\
        --output_dir /outputs/feedback_preference/raw \\
        --k_critiques 16

    # Step 2: Build feedback preference pairs (CPU only)
    uv run python scripts/build_feedback_preference_data.py build \\
        --raw_dir /outputs/feedback_preference/raw \\
        --output_path /outputs/feedback_preference/feedback_pairs.jsonl \\
        --max_pairs_per_question 3

    # Step 3: Evaluate sycophancy rate
    uv run python scripts/build_feedback_preference_data.py eval \\
        --raw_dir /outputs/feedback_preference/raw \\
        --n_samples 500
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Data Classes
# ===========================================================================


@dataclass
class CritiqueCandidate:
    """A single F1 critique and its downstream A2 result.

    Attributes:
        f1_text: Generated feedback/critique text
        a2_text: A2 generated after this F1 (greedy)
        a2_extracted: Extracted MCQ letter from A2
        a2_correct: Whether A2 matches ground truth
        calibration_correct: Whether F1 correctly assesses A1
        is_good: Overall quality (calibration correct AND a2 not regressed)
        score: Composite quality score for ranking
    """

    f1_text: str
    a2_text: str
    a2_extracted: str
    a2_correct: bool
    calibration_correct: bool
    is_good: bool
    score: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class QuestionCritiques:
    """All generated critiques for one (question, A1) pair.

    Attributes:
        sample_index: Index in the original dataset
        question: Question text
        image_path: Path to the image
        ground_truth: Ground truth answer
        answer_type: Always "mcq" for LIVR
        dataset_name: LIVR task name
        a1_text: Initial answer (greedy)
        a1_extracted: Extracted MCQ letter from A1
        a1_correct: Whether A1 matches ground truth
        critiques: List of K critique candidates
    """

    sample_index: int
    question: str
    image_path: str
    ground_truth: str
    answer_type: str
    dataset_name: str
    a1_text: str
    a1_extracted: str
    a1_correct: bool
    critiques: list[CritiqueCandidate] = field(default_factory=list)

    @property
    def n_good(self) -> int:
        """Number of good critiques."""
        return sum(1 for c in self.critiques if c.is_good)

    @property
    def n_bad(self) -> int:
        """Number of bad critiques."""
        return sum(1 for c in self.critiques if not c.is_good)

    @property
    def has_mixed(self) -> bool:
        """Whether there are both good and bad critiques."""
        return self.n_good > 0 and self.n_bad > 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class FeedbackPreferenceSample:
    """A single feedback preference training sample.

    Attributes:
        question: Original question text
        image_path: Path to the image
        ground_truth: Ground truth answer
        a1_text: Initial answer being critiqued
        a1_correct: Whether A1 was correct
        chosen_f1: The better critique
        rejected_f1: The worse critique
        chosen_score: Quality score of chosen F1
        rejected_score: Quality score of rejected F1
        dataset_name: LIVR task name
        source_sample_index: Index in original LIVR dataset
    """

    question: str
    image_path: str
    ground_truth: str
    a1_text: str
    a1_correct: bool
    chosen_f1: str
    rejected_f1: str
    chosen_score: float
    rejected_score: float
    dataset_name: str
    source_sample_index: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# ===========================================================================
# Step 1: Generate A1, K critiques, and A2s
# ===========================================================================


def _assess_calibration(f1_text: str, a1_correct: bool) -> bool:
    """Check if F1 correctly identifies A1's correctness.

    Uses simple keyword matching: if F1 says "correct"/"right"
    when A1 is correct, or "incorrect"/"wrong" when A1 is wrong,
    calibration is correct.

    Args:
        f1_text: Critique text
        a1_correct: Whether A1 was correct

    Returns:
        True if F1's assessment matches A1's actual correctness
    """
    text_lower = f1_text.lower()

    # Hedged positives count as negative
    hedged = any(p in text_lower for p in [
        "partially correct", "mostly correct", "on the right track",
    ])
    has_positive = any(p in text_lower for p in [
        "correct", "accurate", "well done", "good answer",
        "no change needed",
    ])
    has_negative = any(p in text_lower for p in [
        "incorrect", "wrong", "error", "mistake", "should be",
        "not correct", "not accurate", "reconsider",
    ])

    if hedged:
        # Hedged positive → treating as negative
        f1_says_correct = False
    elif has_negative and not has_positive:
        f1_says_correct = False
    elif has_positive and not has_negative:
        f1_says_correct = True
    elif has_positive and has_negative:
        # Mixed → treat as negative (erring on side of correction)
        f1_says_correct = False
    else:
        # Neutral → uninformative, mark as wrong
        return False

    return f1_says_correct == a1_correct


def _score_critique(
    a1_correct: bool,
    a2_correct: bool,
    calibration_correct: bool,
) -> float:
    """Compute composite quality score for a critique.

    Components:
    - Calibration: +1.0 if correct, -1.0 if wrong
    - Downstream transition:
        WR (wrong→right): +2.0
        RR (right→right): +0.5
        RW (right→wrong): -2.0
        WW (wrong→wrong): -0.5

    Args:
        a1_correct: Whether A1 was correct
        a2_correct: Whether A2 was correct
        calibration_correct: Whether F1 correctly assessed A1

    Returns:
        Composite score (higher = better critique)
    """
    cal_score = 1.0 if calibration_correct else -1.0

    if not a1_correct and a2_correct:
        downstream = 2.0   # WR: critique fixed the error
    elif a1_correct and a2_correct:
        downstream = 0.5   # RR: critique preserved correctness
    elif a1_correct and not a2_correct:
        downstream = -2.0  # RW: critique caused regression
    else:
        downstream = -0.5  # WW: critique failed to help

    return cal_score + downstream


def generate_critiques(
    dataset_path: str,
    model_id: str,
    output_dir: str,
    k_critiques: int = 16,
    critique_temperature: float = 1.0,
    max_tokens: int = 256,
    max_questions: int = 0,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = 4,
    image_base_dir: str = "/outputs/image_base",
    seed: int = 42,
) -> str:
    """Generate A1, K diverse F1 critiques, and A2 per question.

    For each LIVR question:
    1. Generate A1 (greedy)
    2. Generate K F1 critiques of A1 (temperature sampling)
    3. For each F1, generate A2 (greedy)
    4. Score each critique by calibration + downstream

    Args:
        dataset_path: Path to LIVR MCQ JSONL
        model_id: HuggingFace model ID or local checkpoint
        output_dir: Directory to save critique data
        k_critiques: Number of F1 critiques per A1
        critique_temperature: Sampling temperature for F1 diversity
        max_tokens: Maximum tokens per generation
        max_questions: Maximum questions to process (0 = all)
        gpu_memory_utilization: vLLM GPU memory fraction
        tensor_parallel_size: Number of GPUs for tensor parallel
        image_base_dir: Base directory for resolving image paths
        seed: Random seed

    Returns:
        Path to the output JSONL file
    """
    from vllm import LLM, SamplingParams

    from vlm_grpo.data import load_self_reflection_dataset
    from vlm_grpo.prompts import (
        build_critic_prompt,
        build_initial_answer_prompt,
        build_refiner_prompt,
    )
    from vlm_grpo.rewards.verifier import verify_answer
    from vlm_grpo.trajectory import extract_answer_from_text

    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    samples = load_self_reflection_dataset(
        dataset_path, image_base_dir=image_base_dir, max_samples=max_questions,
    )
    logger.info(f"Loaded {len(samples)} LIVR questions")

    # Init vLLM
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=4096,
        seed=seed,
    )
    processor = llm.get_tokenizer()

    greedy_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    diverse_params = SamplingParams(
        temperature=critique_temperature, max_tokens=max_tokens,
        top_p=0.9, n=k_critiques,
    )

    output_path = os.path.join(output_dir, "critique_candidates.jsonl")
    stats = {"total": 0, "mixed": 0, "all_good": 0, "all_bad": 0,
             "a1_correct": 0, "a1_wrong": 0}

    with open(output_path, "w") as fout:
        for i, sample in enumerate(samples):
            if i % 100 == 0:
                logger.info(f"Processing question {i}/{len(samples)}")

            question = sample["question"]
            gt = sample["ground_truth"]
            answer_type = sample["answer_type"]
            image_path = sample["image_path"]

            # Step 1a: Generate A1 (greedy)
            a1_prompt = build_initial_answer_prompt(question)
            a1_text = _generate_single(
                llm, processor, a1_prompt, image_path, greedy_params,
            )
            a1_extracted = extract_answer_from_text(a1_text, answer_type)
            a1_result = verify_answer(a1_text, gt, answer_type)
            a1_correct = a1_result.is_correct

            if a1_correct:
                stats["a1_correct"] += 1
            else:
                stats["a1_wrong"] += 1

            # Step 1b: Generate K F1 critiques (diverse)
            f1_prompt = build_critic_prompt(
                question, a1_text, model_type="qwen2vl",
            )
            f1_texts = _generate_multiple(
                llm, processor, f1_prompt, image_path, diverse_params,
                k_critiques,
            )

            # Step 1c: For each F1, generate A2 (greedy)
            critiques = []
            for f1_text in f1_texts:
                a2_prompt = build_refiner_prompt(question, a1_text, f1_text)
                a2_text = _generate_single(
                    llm, processor, a2_prompt, image_path, greedy_params,
                )
                a2_extracted = extract_answer_from_text(a2_text, answer_type)
                a2_result = verify_answer(a2_text, gt, answer_type)
                a2_correct = a2_result.is_correct

                cal_correct = _assess_calibration(f1_text, a1_correct)
                score = _score_critique(a1_correct, a2_correct, cal_correct)

                # "Good" = calibration correct AND no regression
                is_good = cal_correct and not (a1_correct and not a2_correct)

                critiques.append(CritiqueCandidate(
                    f1_text=f1_text,
                    a2_text=a2_text,
                    a2_extracted=a2_extracted,
                    a2_correct=a2_correct,
                    calibration_correct=cal_correct,
                    is_good=is_good,
                    score=score,
                ))

            qc = QuestionCritiques(
                sample_index=sample.get("sample_index", i),
                question=question,
                image_path=image_path,
                ground_truth=gt,
                answer_type=answer_type,
                dataset_name=sample.get("dataset_name", "livr"),
                a1_text=a1_text,
                a1_extracted=a1_extracted,
                a1_correct=a1_correct,
                critiques=critiques,
            )

            stats["total"] += 1
            if qc.has_mixed:
                stats["mixed"] += 1
            elif qc.n_good == len(critiques):
                stats["all_good"] += 1
            else:
                stats["all_bad"] += 1

            fout.write(json.dumps(qc.to_dict()) + "\n")

    logger.info(f"Stats: {json.dumps(stats, indent=2)}")
    logger.info(f"Saved to {output_path}")
    return output_path


def _generate_single(
    llm: Any,
    processor: Any,
    messages: list[dict],
    image_path: str,
    params: Any,
) -> str:
    """Generate a single completion via vLLM.

    Args:
        llm: vLLM model instance
        processor: Tokenizer/processor
        messages: Chat messages
        image_path: Path to image
        params: Sampling parameters

    Returns:
        Generated text
    """
    from PIL import Image

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image = Image.open(image_path).convert("RGB")

    outputs = llm.generate(
        [{"prompt": prompt_text, "multi_modal_data": {"image": image}}],
        sampling_params=params,
    )
    return outputs[0].outputs[0].text.strip()


def _generate_multiple(
    llm: Any,
    processor: Any,
    messages: list[dict],
    image_path: str,
    params: Any,
    k: int,
) -> list[str]:
    """Generate K diverse completions via vLLM.

    Args:
        llm: vLLM model instance
        processor: Tokenizer/processor
        messages: Chat messages
        image_path: Path to image
        params: Sampling parameters (n=K)
        k: Expected number of completions

    Returns:
        List of K generated texts
    """
    from PIL import Image

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image = Image.open(image_path).convert("RGB")

    outputs = llm.generate(
        [{"prompt": prompt_text, "multi_modal_data": {"image": image}}],
        sampling_params=params,
    )
    return [o.text.strip() for o in outputs[0].outputs[:k]]


# ===========================================================================
# Step 2: Build feedback preference pairs
# ===========================================================================


def build_feedback_pairs(
    raw_dir: str,
    output_path: str,
    max_pairs_per_question: int = 3,
    seed: int = 42,
) -> int:
    """Build pairwise feedback preference data from scored critiques.

    For each question with mixed good/bad critiques:
    - Sample up to N pairs of (good_F1, bad_F1)
    - Store as chosen/rejected for DPO or pairwise GRPO

    Args:
        raw_dir: Directory containing critique_candidates.jsonl
        output_path: Output JSONL path
        max_pairs_per_question: Max pairs per question
        seed: Random seed

    Returns:
        Number of pairs generated
    """
    random.seed(seed)
    input_path = os.path.join(raw_dir, "critique_candidates.jsonl")

    questions = []
    with open(input_path) as f:
        for line in f:
            questions.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(questions)} questions with critiques")

    pairs = []
    skipped_all_good = 0
    skipped_all_bad = 0

    for q in questions:
        critiques = q["critiques"]
        good = [c for c in critiques if c["is_good"]]
        bad = [c for c in critiques if not c["is_good"]]

        if not good or not bad:
            if not bad:
                skipped_all_good += 1
            else:
                skipped_all_bad += 1
            continue

        # Sort by score: pick best good, worst bad for max contrast
        good.sort(key=lambda c: c["score"], reverse=True)
        bad.sort(key=lambda c: c["score"])

        n_pairs = min(max_pairs_per_question, len(good), len(bad))
        for pi in range(n_pairs):
            pairs.append(FeedbackPreferenceSample(
                question=q["question"],
                image_path=q["image_path"],
                ground_truth=q["ground_truth"],
                a1_text=q["a1_text"],
                a1_correct=q["a1_correct"],
                chosen_f1=good[pi % len(good)]["f1_text"],
                rejected_f1=bad[pi % len(bad)]["f1_text"],
                chosen_score=good[pi % len(good)]["score"],
                rejected_score=bad[pi % len(bad)]["score"],
                dataset_name=q.get("dataset_name", "livr"),
                source_sample_index=q.get("sample_index", 0),
            ))

    # Shuffle pairs
    random.shuffle(pairs)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p.to_dict()) + "\n")

    logger.info(
        f"Generated {len(pairs)} feedback preference pairs "
        f"(skipped {skipped_all_good} all-good, {skipped_all_bad} all-bad)"
    )
    return len(pairs)


# ===========================================================================
# Step 3: Evaluate sycophancy rate
# ===========================================================================


def evaluate_sycophancy(
    raw_dir: str,
    n_samples: int = 0,
) -> dict[str, float]:
    """Evaluate baseline sycophancy rate from generated critiques.

    Args:
        raw_dir: Directory containing critique_candidates.jsonl
        n_samples: Max samples to evaluate (0 = all)

    Returns:
        Dict of calibration metrics
    """
    input_path = os.path.join(raw_dir, "critique_candidates.jsonl")

    questions = []
    with open(input_path) as f:
        for line in f:
            questions.append(json.loads(line.strip()))

    if n_samples > 0:
        questions = questions[:n_samples]

    # Aggregate
    total_critiques = 0
    cal_correct = 0
    syc_count = 0  # says "correct" when A1 wrong
    false_neg = 0  # says "wrong" when A1 right
    wr_count = 0
    rw_count = 0
    rr_count = 0
    ww_count = 0

    for q in questions:
        a1_correct = q["a1_correct"]
        for c in q["critiques"]:
            total_critiques += 1
            if c["calibration_correct"]:
                cal_correct += 1
            elif a1_correct:
                false_neg += 1  # F1 says wrong, A1 is right
            else:
                syc_count += 1  # F1 says correct, A1 is wrong

            # Transition
            a2_correct = c["a2_correct"]
            if a1_correct and a2_correct:
                rr_count += 1
            elif a1_correct and not a2_correct:
                rw_count += 1
            elif not a1_correct and a2_correct:
                wr_count += 1
            else:
                ww_count += 1

    n = max(total_critiques, 1)
    a1_wrong_total = sum(
        len(q["critiques"]) for q in questions if not q["a1_correct"]
    )
    a1_right_total = sum(
        len(q["critiques"]) for q in questions if q["a1_correct"]
    )

    metrics = {
        "total_questions": len(questions),
        "total_critiques": total_critiques,
        "calibration_accuracy": cal_correct / n,
        "sycophancy_rate": syc_count / max(a1_wrong_total, 1),
        "false_negative_rate": false_neg / max(a1_right_total, 1),
        "wr_rate": wr_count / n,
        "rw_rate": rw_count / n,
        "rr_rate": rr_count / n,
        "ww_rate": ww_count / n,
    }

    logger.info("=== Sycophancy Evaluation ===")
    logger.info(f"  Total questions: {len(questions)}")
    logger.info(f"  Total critiques: {total_critiques}")
    logger.info(f"  Calibration accuracy: {metrics['calibration_accuracy']:.1%}")
    logger.info(f"  Sycophancy rate (says correct when A1 wrong): "
                f"{metrics['sycophancy_rate']:.1%}")
    logger.info(f"  False negative rate (says wrong when A1 right): "
                f"{metrics['false_negative_rate']:.1%}")
    logger.info(f"  WR: {metrics['wr_rate']:.1%}  RW: {metrics['rw_rate']:.1%}  "
                f"RR: {metrics['rr_rate']:.1%}  WW: {metrics['ww_rate']:.1%}")

    return metrics


# ===========================================================================
# CLI
# ===========================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Build feedback preference data for critic GRPO warm-up",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    gen = sub.add_parser("generate", help="Generate A1 + K critiques + A2")
    gen.add_argument("--dataset_path", required=True)
    gen.add_argument("--model_id", required=True)
    gen.add_argument("--output_dir", required=True)
    gen.add_argument("--k_critiques", type=int, default=16)
    gen.add_argument("--critique_temperature", type=float, default=1.0)
    gen.add_argument("--max_tokens", type=int, default=256)
    gen.add_argument("--max_questions", type=int, default=0)
    gen.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    gen.add_argument("--tensor_parallel_size", type=int, default=4)
    gen.add_argument("--image_base_dir", default="/outputs/image_base")
    gen.add_argument("--seed", type=int, default=42)

    # build
    bld = sub.add_parser("build", help="Build feedback preference pairs")
    bld.add_argument("--raw_dir", required=True)
    bld.add_argument("--output_path", required=True)
    bld.add_argument("--max_pairs_per_question", type=int, default=3)
    bld.add_argument("--seed", type=int, default=42)

    # eval
    ev = sub.add_parser("eval", help="Evaluate sycophancy rate")
    ev.add_argument("--raw_dir", required=True)
    ev.add_argument("--n_samples", type=int, default=0)

    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()

    if args.command == "generate":
        generate_critiques(
            dataset_path=args.dataset_path,
            model_id=args.model_id,
            output_dir=args.output_dir,
            k_critiques=args.k_critiques,
            critique_temperature=args.critique_temperature,
            max_tokens=args.max_tokens,
            max_questions=args.max_questions,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            image_base_dir=args.image_base_dir,
            seed=args.seed,
        )
    elif args.command == "build":
        build_feedback_pairs(
            raw_dir=args.raw_dir,
            output_path=args.output_path,
            max_pairs_per_question=args.max_pairs_per_question,
            seed=args.seed,
        )
    elif args.command == "eval":
        evaluate_sycophancy(
            raw_dir=args.raw_dir,
            n_samples=args.n_samples,
        )


if __name__ == "__main__":
    main()
