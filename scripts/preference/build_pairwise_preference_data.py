#!/usr/bin/env python3
"""Build pairwise preference data for critic GRPO warm-up training.

===========================================================================
DESIGN OVERVIEW — Pairwise Preference GRPO for Critic Warm-Up
===========================================================================

Problem:
    The critic (F1) in our A1→F1→A2 pipeline is sycophantic — it says
    "correct" 81% of the time even when A1 is wrong. We need to teach
    the model to distinguish correct from incorrect VQA answers BEFORE
    running the full self-reflection GRPO.

Approach (LLaVA-Critic-R1 style):
    1. Generate diverse answer candidates per question (K=16, temp=1.0)
    2. Score each deterministically (MCQ matching)
    3. Pair correct+incorrect answers → pairwise preference task
    4. Train with GRPO: model must identify which response is better

Pipeline:
    Step 1 — Generate K=16 answers per LIVR question using vLLM
    Step 2 — Score answers deterministically (MCQ letter match)
    Step 3 — For each question with mixed correct/incorrect answers:
             Sample up to 3 pairs (correct, incorrect)
             Randomize which is "Response A" vs "Response B"
    Step 4 — Write JSONL for pairwise GRPO training

Expected Yield:
    With 9K LIVR MCQ questions and K=16:
    - Base Qwen2.5-VL-7B typically gets ~40-60% correct on perception MCQ
    - ~70-80% of questions will have mixed correct/incorrect (mixed pool)
    - ~6.3K-7.2K mixed questions x 3 pairs = 19K-22K preference pairs
    - Remaining ~20-30% all-correct or all-wrong are skipped
    - If yield < 15K, increase K to 24 or lower temperature to 0.8

Training Recipe (GRPO warm-up):
    - Prompt: "Given image+question, which response is better?"
    - Model output: reasoning + \\boxed{A} or \\boxed{B}
    - Reward: +1 correct preference, 0 wrong
    - K=2 for GRPO (per LLaVA-Critic-R1, small K is sufficient)
    - Steps: ~1500 (1 epoch over ~20K pairs, batch=16, 4 GPUs)
    - LR: 5e-7 (gentler than main training 1e-6)
    - LoRA: same as main (r=64, alpha=128)
    - Clip range: 0.2 (same as main)
    - Loss: dr_grpo (same as main)
    - Max completion: 256 tokens
    - No KL penalty (fresh warm-up, not anchored to base)

Evaluation:
    After warm-up, test critique calibration on held-out LIVR questions:
    1. Generate A1 at temp=0 (greedy)
    2. Generate F1 (critic feedback)
    3. Measure: "says correct when A1 wrong" rate (target: < 50%, from 81%)
    4. Measure: "says incorrect when A1 right" rate (target: < 30%)
    5. Compare WR rate (wrong→right) in full A1→F1→A2 pipeline

===========================================================================

Usage:
    # Step 1: Generate answers (requires GPU + vLLM)
    uv run python scripts/build_pairwise_preference_data.py generate \\
        --dataset_path /outputs/livr_data/livr_perception_mcq.jsonl \\
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \\
        --output_dir /outputs/pairwise_preference/raw_answers \\
        --k_samples 16 --temperature 1.0 --batch_size 32

    # Step 2: Build preference pairs (CPU only)
    uv run python scripts/build_pairwise_preference_data.py build \\
        --answers_dir /outputs/pairwise_preference/raw_answers \\
        --output_path /outputs/pairwise_preference/pairwise_grpo.jsonl \\
        --max_pairs_per_question 3

    # Step 3: Run warm-up training (4 GPUs)
    accelerate launch --config_file k8s/multi_gpu.yaml --num_processes=4 \\
        scripts/train_pairwise_preference.py \\
        --model_id Qwen/Qwen2.5-VL-7B-Instruct \\
        --dataset_path /outputs/pairwise_preference/pairwise_grpo.jsonl \\
        --output_dir /outputs/grpo_qwen_critic_warmup_v1

    # Step 4: Evaluate calibration (see eval section below)
    uv run python scripts/build_pairwise_preference_data.py eval \\
        --dataset_path /outputs/livr_data/livr_perception_mcq.jsonl \\
        --model_id /outputs/grpo_qwen_critic_warmup_v1/checkpoint-1500 \\
        --output_path /outputs/pairwise_preference/eval_calibration.json \\
        --n_samples 500
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import asdict, dataclass, field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ===========================================================================
# Constants
# ===========================================================================

# Pairwise preference prompt template.
# Matches LLaVA-Critic-R1 format: present two responses, ask which is better.
# The model should output reasoning followed by \boxed{A} or \boxed{B}.
PAIRWISE_SYSTEM_PROMPT = (
    "You are a visual question answering judge. Given an image, a question, "
    "and two candidate responses, determine which response better answers the "
    "question based on what is visible in the image. Think step by step, then "
    "give your final verdict."
)

PAIRWISE_USER_TEMPLATE = (
    "{question}\n\n"
    "Response A: {response_a}\n"
    "Response B: {response_b}\n\n"
    "Which response better answers the question? Reason step by step about "
    "what the image shows, then conclude with \\boxed{{A}} or \\boxed{{B}}."
)

# For answer extraction from pairwise judgments
BOXED_PATTERN_STR = r"\\boxed\{([AB])\}"


# ===========================================================================
# Data Classes
# ===========================================================================


@dataclass
class AnswerCandidate:
    """A single generated answer for one question.

    Attributes:
        text: Raw answer text from the model
        extracted_letter: Extracted MCQ letter (A-F) or empty
        is_correct: Whether this answer matches ground truth
    """

    text: str
    extracted_letter: str
    is_correct: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class QuestionAnswers:
    """All generated answers for a single question.

    Attributes:
        sample_index: Index in the original dataset
        question: Question text
        image_path: Path to the image
        ground_truth: Ground truth answer (e.g., "(A)")
        answer_type: Always "mcq" for LIVR
        choices: MCQ choices string
        dataset_name: LIVR task name
        candidates: List of K answer candidates
    """

    sample_index: int
    question: str
    image_path: str
    ground_truth: str
    answer_type: str
    choices: str
    dataset_name: str
    candidates: list[AnswerCandidate] = field(default_factory=list)

    @property
    def n_correct(self) -> int:
        """Number of correct candidates."""
        return sum(1 for c in self.candidates if c.is_correct)

    @property
    def n_wrong(self) -> int:
        """Number of incorrect candidates."""
        return sum(1 for c in self.candidates if not c.is_correct)

    @property
    def has_mixed(self) -> bool:
        """Whether there are both correct and incorrect candidates."""
        return self.n_correct > 0 and self.n_wrong > 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        return d


@dataclass
class PairwisePreferenceSample:
    """A single pairwise preference training sample.

    Attributes:
        question: Question text
        image_path: Path to the image
        ground_truth: Ground truth answer
        response_a: Text of Response A
        response_b: Text of Response B
        correct_choice: "A" or "B" — which response is correct
        a_is_correct: Whether Response A is correct
        b_is_correct: Whether Response B is correct
        dataset_name: LIVR task name
        source_sample_index: Index in original LIVR dataset
    """

    question: str
    image_path: str
    ground_truth: str
    response_a: str
    response_b: str
    correct_choice: str
    a_is_correct: bool
    b_is_correct: bool
    dataset_name: str
    source_sample_index: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_grpo_record(self) -> dict:
        """Convert to GRPO training record format.

        Returns:
            Dict with keys matching the GRPO data loader expectations:
            question, images, ground_truth, answer_type, dataset_name,
            plus pairwise-specific fields.
        """
        user_text = PAIRWISE_USER_TEMPLATE.format(
            question=self.question,
            response_a=self.response_a,
            response_b=self.response_b,
        )
        return {
            "question": user_text,
            "images": [self.image_path],
            "ground_truth": self.correct_choice,
            "answer_type": "pairwise_preference",
            "dataset_name": self.dataset_name,
            "source_sample_index": self.source_sample_index,
            # Metadata for analysis
            "original_question": self.question,
            "original_ground_truth": self.ground_truth,
            "response_a": self.response_a,
            "response_b": self.response_b,
            "a_is_correct": self.a_is_correct,
            "b_is_correct": self.b_is_correct,
            "correct_choice": self.correct_choice,
        }


# ===========================================================================
# Step 1: Generate diverse answer candidates
# ===========================================================================


def generate_answer_candidates(
    dataset_path: str,
    model_id: str,
    output_dir: str,
    k_samples: int = 16,
    temperature: float = 1.0,
    max_tokens: int = 256,
    batch_size: int = 32,
    max_questions: int = 0,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = 4,
    image_base_dir: str = "/outputs/image_base",
    seed: int = 42,
) -> str:
    """Generate K answer candidates per question using vLLM.

    Loads the LIVR dataset, batches questions through vLLM with temperature
    sampling, and saves raw answers with correctness labels.

    Args:
        dataset_path: Path to LIVR MCQ JSONL
        model_id: HuggingFace model ID or local checkpoint
        output_dir: Directory to save answer candidates
        k_samples: Number of answers per question
        temperature: Sampling temperature (1.0 for diversity)
        max_tokens: Maximum tokens per answer
        batch_size: vLLM batch size for concurrent requests
        max_questions: Maximum questions to process (0 = all)
        gpu_memory_utilization: vLLM GPU memory fraction
        tensor_parallel_size: Number of GPUs for tensor parallel
        image_base_dir: Base directory for resolving image paths
        seed: Random seed for reproducibility

    Returns:
        Path to the output JSONL file with answer candidates
    """
    # Lazy imports — vLLM and torch only needed when actually generating
    from vllm import LLM, SamplingParams

    from vlm_grpo.data import _load_jsonl, _resolve_image_path, load_image_safe
    from vlm_grpo.prompts import VL_ASSISTANT_SYSTEM_PROMPT
    from vlm_grpo.rewards.verifier import verify_answer

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "answer_candidates.jsonl")

    # Load dataset
    raw_samples = _load_jsonl(dataset_path, max_questions)
    logger.info(f"Loaded {len(raw_samples)} questions from {dataset_path}")

    # Initialize vLLM
    logger.info(
        f"Initializing vLLM with model={model_id}, "
        f"tp={tensor_parallel_size}, gpu_mem={gpu_memory_utilization}"
    )
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=4096,
        seed=seed,
    )

    sampling_params = SamplingParams(
        n=k_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95,
        seed=seed,
    )

    # Build prompts for all questions
    prompts = []
    valid_samples = []
    images_list = []

    for i, sample in enumerate(raw_samples):
        question = sample.get("question", "")
        image_path = _resolve_image_path(sample, image_base_dir)

        if not question or not image_path:
            logger.warning(f"Skipping sample {i}: missing question or image")
            continue

        question = question.replace("<image>", "").strip()

        # Load image
        image = load_image_safe(image_path, max_pixels=401408)
        if image is None:
            logger.warning(f"Skipping sample {i}: failed to load image {image_path}")
            continue

        # Build A1-style prompt (same as training, without think tags)
        prompt = _build_vllm_chat_prompt(question, VL_ASSISTANT_SYSTEM_PROMPT)
        prompts.append(prompt)
        images_list.append(image)
        valid_samples.append(
            {
                "index": i,
                "question": question,
                "image_path": image_path,
                "ground_truth": sample.get("ground_truth", ""),
                "answer_type": sample.get("answer_type", "mcq"),
                "choices": sample.get("choices", ""),
                "dataset_name": sample.get("dataset_name", "unknown"),
            }
        )

    logger.info(f"Prepared {len(prompts)} valid questions for generation")

    # Generate in batches
    all_results: list[QuestionAnswers] = []
    total_correct = 0
    total_generated = 0

    for batch_start in range(0, len(prompts), batch_size):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_images = images_list[batch_start:batch_end]
        batch_samples = valid_samples[batch_start:batch_end]

        logger.info(
            f"Generating batch {batch_start // batch_size + 1}/"
            f"{(len(prompts) + batch_size - 1) // batch_size} "
            f"({batch_end - batch_start} questions)"
        )

        # vLLM generation with images
        outputs = llm.generate(
            _build_vllm_inputs(batch_prompts, batch_images),
            sampling_params=sampling_params,
        )

        for output, sample_info in zip(outputs, batch_samples):
            candidates = []
            for completion in output.outputs:
                raw_text = completion.text.strip()

                # Score against ground truth
                result = verify_answer(
                    raw_text,
                    sample_info["ground_truth"],
                    sample_info["answer_type"],
                    sample_info["choices"],
                )

                candidate = AnswerCandidate(
                    text=raw_text,
                    extracted_letter=result.extracted,
                    is_correct=result.is_correct,
                )
                candidates.append(candidate)

                total_generated += 1
                if result.is_correct:
                    total_correct += 1

            qa = QuestionAnswers(
                sample_index=sample_info["index"],
                question=sample_info["question"],
                image_path=sample_info["image_path"],
                ground_truth=sample_info["ground_truth"],
                answer_type=sample_info["answer_type"],
                choices=sample_info["choices"],
                dataset_name=sample_info["dataset_name"],
                candidates=candidates,
            )
            all_results.append(qa)

    # Write results
    with open(output_path, "w") as f:
        for qa in all_results:
            f.write(json.dumps(qa.to_dict(), ensure_ascii=False) + "\n")

    # Summary statistics
    accuracy = total_correct / max(total_generated, 1)
    mixed_count = sum(1 for qa in all_results if qa.has_mixed)
    all_correct_count = sum(1 for qa in all_results if qa.n_correct == len(qa.candidates))
    all_wrong_count = sum(1 for qa in all_results if qa.n_wrong == len(qa.candidates))

    logger.info("=" * 60)
    logger.info("Generation Summary")
    logger.info("=" * 60)
    logger.info(f"Total questions:     {len(all_results)}")
    logger.info(f"Total answers:       {total_generated}")
    logger.info(f"Overall accuracy:    {accuracy:.1%}")
    mixed_pct = mixed_count / max(len(all_results), 1)
    logger.info(f"Mixed (usable):      {mixed_count} ({mixed_pct:.1%})")
    logger.info(f"All correct (skip):  {all_correct_count}")
    logger.info(f"All wrong (skip):    {all_wrong_count}")
    logger.info(f"Output: {output_path}")

    # Per-task breakdown
    task_stats: dict[str, dict[str, int]] = {}
    for qa in all_results:
        task = qa.dataset_name
        if task not in task_stats:
            task_stats[task] = {"total": 0, "mixed": 0, "all_correct": 0, "all_wrong": 0}
        task_stats[task]["total"] += 1
        if qa.has_mixed:
            task_stats[task]["mixed"] += 1
        elif qa.n_correct == len(qa.candidates):
            task_stats[task]["all_correct"] += 1
        else:
            task_stats[task]["all_wrong"] += 1

    logger.info("\nPer-task breakdown:")
    for task in sorted(task_stats.keys()):
        s = task_stats[task]
        logger.info(
            f"  {task:30s}: {s['total']:5d} total, "
            f"{s['mixed']:5d} mixed ({s['mixed'] / max(s['total'], 1):.0%}), "
            f"{s['all_correct']:5d} all-correct, "
            f"{s['all_wrong']:5d} all-wrong"
        )

    return output_path


def _build_vllm_chat_prompt(question: str, system_prompt: str) -> list[dict]:
    """Build a chat prompt for vLLM generation.

    Args:
        question: Visual question text
        system_prompt: System prompt text

    Returns:
        Chat-formatted message list for vLLM
    """
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]


def _build_vllm_inputs(
    prompts: list[list[dict]],
    images: list,
) -> list[dict]:
    """Build vLLM input dicts from prompts and images.

    Args:
        prompts: List of chat message lists
        images: List of PIL images

    Returns:
        List of vLLM input dicts with prompt and multi_modal_data
    """
    inputs = []
    for prompt, image in zip(prompts, images):
        inputs.append(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            }
        )
    return inputs


# ===========================================================================
# Step 2: Build pairwise preference pairs
# ===========================================================================


def build_preference_pairs(
    answers_dir: str,
    output_path: str,
    max_pairs_per_question: int = 3,
    seed: int = 42,
    min_answer_length: int = 1,
) -> str:
    """Build pairwise preference training data from answer candidates.

    For each question with mixed correct/incorrect answers:
    - Sample up to max_pairs_per_question (correct, incorrect) pairs
    - Randomly assign which is "Response A" vs "Response B"
    - 50/50 ensures no position bias in training

    Skips questions where ALL answers are correct or ALL wrong — these
    have no preference signal (both responses would be equivalent).

    Args:
        answers_dir: Directory containing answer_candidates.jsonl
        output_path: Output JSONL path for preference pairs
        max_pairs_per_question: Maximum pairs per question (3 is a good
            balance: enough pairs for training, not so many that easy
            questions dominate)
        seed: Random seed
        min_answer_length: Minimum answer length in characters to include

    Returns:
        Path to the output JSONL file
    """
    rng = random.Random(seed)
    answers_path = os.path.join(answers_dir, "answer_candidates.jsonl")

    if not os.path.exists(answers_path):
        logger.error(f"Answer candidates not found: {answers_path}")
        sys.exit(1)

    # Load answer candidates
    questions: list[QuestionAnswers] = []
    with open(answers_path) as f:
        for line in f:
            record = json.loads(line.strip())
            candidates = [AnswerCandidate(**c) for c in record.pop("candidates", [])]
            qa = QuestionAnswers(**record)
            qa.candidates = candidates
            questions.append(qa)

    logger.info(f"Loaded {len(questions)} questions with answer candidates")

    # Build pairs
    all_pairs: list[PairwisePreferenceSample] = []
    skipped_all_correct = 0
    skipped_all_wrong = 0

    for qa in questions:
        if not qa.has_mixed:
            if qa.n_correct == len(qa.candidates):
                skipped_all_correct += 1
            else:
                skipped_all_wrong += 1
            continue

        correct_candidates = [c for c in qa.candidates if c.is_correct]
        wrong_candidates = [c for c in qa.candidates if not c.is_correct]

        # Filter out too-short answers (model sometimes outputs empty)
        correct_candidates = [
            c for c in correct_candidates if len(c.text.strip()) >= min_answer_length
        ]
        wrong_candidates = [c for c in wrong_candidates if len(c.text.strip()) >= min_answer_length]

        if not correct_candidates or not wrong_candidates:
            continue

        # Sample pairs: each pair is (correct, wrong)
        # Strategy: sample diverse pairs, not all combinations
        # Use up to max_pairs_per_question unique (correct, wrong) pairs
        n_pairs = min(
            max_pairs_per_question,
            len(correct_candidates) * len(wrong_candidates),
        )

        # Build all possible pairs and sample
        possible_pairs = [(c, w) for c in correct_candidates for w in wrong_candidates]
        rng.shuffle(possible_pairs)
        selected_pairs = possible_pairs[:n_pairs]

        for correct, wrong in selected_pairs:
            # Randomize position: 50% chance correct is A, 50% correct is B
            if rng.random() < 0.5:
                response_a, response_b = correct.text, wrong.text
                a_is_correct, b_is_correct = True, False
                correct_choice = "A"
            else:
                response_a, response_b = wrong.text, correct.text
                a_is_correct, b_is_correct = False, True
                correct_choice = "B"

            pair = PairwisePreferenceSample(
                question=qa.question,
                image_path=qa.image_path,
                ground_truth=qa.ground_truth,
                response_a=response_a,
                response_b=response_b,
                correct_choice=correct_choice,
                a_is_correct=a_is_correct,
                b_is_correct=b_is_correct,
                dataset_name=qa.dataset_name,
                source_sample_index=qa.sample_index,
            )
            all_pairs.append(pair)

    # Shuffle all pairs so training isn't ordered by question
    rng.shuffle(all_pairs)

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair.to_grpo_record(), ensure_ascii=False) + "\n")

    # Summary
    a_correct_count = sum(1 for p in all_pairs if p.correct_choice == "A")
    b_correct_count = sum(1 for p in all_pairs if p.correct_choice == "B")

    logger.info("=" * 60)
    logger.info("Preference Pair Summary")
    logger.info("=" * 60)
    logger.info(f"Total pairs:         {len(all_pairs)}")
    a_pct = a_correct_count / max(len(all_pairs), 1)
    b_pct = b_correct_count / max(len(all_pairs), 1)
    logger.info(f"Correct is A:        {a_correct_count} ({a_pct:.1%})")
    logger.info(f"Correct is B:        {b_correct_count} ({b_pct:.1%})")
    logger.info(f"Skipped all-correct: {skipped_all_correct}")
    logger.info(f"Skipped all-wrong:   {skipped_all_wrong}")
    logger.info(f"Output: {output_path}")

    # Per-task breakdown
    task_pairs: dict[str, int] = {}
    for pair in all_pairs:
        task_pairs[pair.dataset_name] = task_pairs.get(pair.dataset_name, 0) + 1

    logger.info("\nPer-task pair counts:")
    for task in sorted(task_pairs.keys()):
        logger.info(f"  {task:30s}: {task_pairs[task]:5d} pairs")

    return output_path


# ===========================================================================
# Step 3: Evaluate critic calibration after warm-up
# ===========================================================================


def evaluate_critic_calibration(
    dataset_path: str,
    model_id: str,
    output_path: str,
    n_samples: int = 500,
    gpu_memory_utilization: float = 0.85,
    tensor_parallel_size: int = 4,
    image_base_dir: str = "/outputs/image_base",
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate critic calibration: does the model correctly assess A1?

    For each question:
    1. Generate A1 at temp=0 (greedy — deterministic answer)
    2. Generate F1 (critic feedback) at temp=0
    3. Check if F1 says "correct"/"incorrect" and whether A1 is actually right
    4. Compute calibration metrics

    Key metrics:
    - sycophancy_rate: says "correct" when A1 is wrong (target: < 50%)
    - false_negative_rate: says "incorrect" when A1 is right (target: < 30%)
    - calibration_accuracy: overall correctness of the critic's assessment

    Args:
        dataset_path: Path to LIVR MCQ JSONL (or any MCQ dataset)
        model_id: Model checkpoint to evaluate
        output_path: Path to write evaluation results JSON
        n_samples: Number of questions to evaluate
        gpu_memory_utilization: vLLM GPU memory fraction
        tensor_parallel_size: Number of GPUs for tensor parallel
        image_base_dir: Base directory for image paths
        seed: Random seed

    Returns:
        Dict of metric name to value
    """
    from vllm import LLM, SamplingParams

    from vlm_grpo.data import _load_jsonl, _resolve_image_path, load_image_safe
    from vlm_grpo.prompts import (
        FEEDBACK_CRITIC_SYSTEM_PROMPT,
        VL_ASSISTANT_SYSTEM_PROMPT,
    )
    from vlm_grpo.rewards.deterministic import (
        _HEDGED_POSITIVE_PATTERNS,
        _NEGATIVE_FEEDBACK_PATTERNS,
        _POSITIVE_FEEDBACK_PATTERNS,
    )
    from vlm_grpo.rewards.verifier import verify_answer

    rng = random.Random(seed)

    # Load and sample dataset
    raw_samples = _load_jsonl(dataset_path, 0)
    rng.shuffle(raw_samples)
    raw_samples = raw_samples[:n_samples]
    logger.info(f"Evaluating on {len(raw_samples)} samples")

    # Initialize vLLM
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=4096,
        seed=seed,
    )

    # Phase 1: Generate A1 (greedy)
    a1_params = SamplingParams(n=1, temperature=0.0, max_tokens=128, seed=seed)

    a1_prompts = []
    a1_images = []
    valid_samples = []

    for sample in raw_samples:
        question = sample.get("question", "").replace("<image>", "").strip()
        image_path = _resolve_image_path(sample, image_base_dir)
        if not question or not image_path:
            continue

        image = load_image_safe(image_path, max_pixels=401408)
        if image is None:
            continue

        a1_prompts.append(_build_vllm_chat_prompt(question, VL_ASSISTANT_SYSTEM_PROMPT))
        a1_images.append(image)
        valid_samples.append(
            {
                "question": question,
                "image_path": image_path,
                "image": image,
                "ground_truth": sample.get("ground_truth", ""),
                "answer_type": sample.get("answer_type", "mcq"),
                "choices": sample.get("choices", ""),
            }
        )

    logger.info(f"Generating A1 for {len(a1_prompts)} samples...")
    a1_outputs = llm.generate(
        _build_vllm_inputs(a1_prompts, a1_images),
        sampling_params=a1_params,
    )

    # Phase 2: Generate F1 (critic feedback) for each A1
    f1_params = SamplingParams(n=1, temperature=0.0, max_tokens=256, seed=seed)

    f1_prompts = []
    f1_images = []
    a1_results = []

    for output, sample_info in zip(a1_outputs, valid_samples):
        a1_text = output.outputs[0].text.strip()

        # Score A1
        a1_result = verify_answer(
            a1_text,
            sample_info["ground_truth"],
            sample_info["answer_type"],
            sample_info["choices"],
        )

        # Build critic prompt (role-flipped, Qwen2VL format)
        f1_prompt = [
            {"role": "system", "content": FEEDBACK_CRITIC_SYSTEM_PROMPT},
            {
                "role": "assistant",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": sample_info["question"]},
                ],
            },
            {"role": "user", "content": a1_text},
        ]

        f1_prompts.append(f1_prompt)
        f1_images.append(sample_info["image"])
        a1_results.append(
            {
                "a1_text": a1_text,
                "a1_is_correct": a1_result.is_correct,
                "question": sample_info["question"],
                "ground_truth": sample_info["ground_truth"],
            }
        )

    logger.info(f"Generating F1 for {len(f1_prompts)} samples...")
    f1_outputs = llm.generate(
        _build_vllm_inputs(f1_prompts, f1_images),
        sampling_params=f1_params,
    )

    # Phase 3: Analyze calibration
    total = len(f1_outputs)
    a1_correct_count = 0
    a1_wrong_count = 0

    # When A1 is correct
    critic_says_correct_given_correct = 0
    critic_says_wrong_given_correct = 0
    critic_says_unclear_given_correct = 0

    # When A1 is wrong
    critic_says_correct_given_wrong = 0
    critic_says_wrong_given_wrong = 0
    critic_says_unclear_given_wrong = 0

    detailed_results = []

    for f1_output, a1_info in zip(f1_outputs, a1_results):
        f1_text = f1_output.outputs[0].text.strip()

        # Classify critic stance
        stance = _classify_feedback_stance(
            f1_text,
            _HEDGED_POSITIVE_PATTERNS,
            _POSITIVE_FEEDBACK_PATTERNS,
            _NEGATIVE_FEEDBACK_PATTERNS,
        )

        if a1_info["a1_is_correct"]:
            a1_correct_count += 1
            if stance == "positive":
                critic_says_correct_given_correct += 1
            elif stance == "negative":
                critic_says_wrong_given_correct += 1
            else:
                critic_says_unclear_given_correct += 1
        else:
            a1_wrong_count += 1
            if stance == "positive":
                critic_says_correct_given_wrong += 1
            elif stance == "negative":
                critic_says_wrong_given_wrong += 1
            else:
                critic_says_unclear_given_wrong += 1

        detailed_results.append(
            {
                "question": a1_info["question"],
                "ground_truth": a1_info["ground_truth"],
                "a1_text": a1_info["a1_text"],
                "a1_is_correct": a1_info["a1_is_correct"],
                "f1_text": f1_text,
                "critic_stance": stance,
            }
        )

    # Compute metrics
    metrics: dict[str, float] = {
        "total_evaluated": float(total),
        "a1_accuracy": a1_correct_count / max(total, 1),
        "a1_correct_count": float(a1_correct_count),
        "a1_wrong_count": float(a1_wrong_count),
    }

    if a1_wrong_count > 0:
        metrics["sycophancy_rate"] = critic_says_correct_given_wrong / a1_wrong_count
        metrics["correct_rejection_rate"] = critic_says_wrong_given_wrong / a1_wrong_count
        metrics["unclear_given_wrong"] = critic_says_unclear_given_wrong / a1_wrong_count
    else:
        metrics["sycophancy_rate"] = 0.0
        metrics["correct_rejection_rate"] = 0.0

    if a1_correct_count > 0:
        metrics["false_negative_rate"] = critic_says_wrong_given_correct / a1_correct_count
        metrics["true_positive_rate"] = critic_says_correct_given_correct / a1_correct_count
    else:
        metrics["false_negative_rate"] = 0.0
        metrics["true_positive_rate"] = 0.0

    # Overall calibration accuracy
    correct_assessments = critic_says_correct_given_correct + critic_says_wrong_given_wrong
    metrics["calibration_accuracy"] = correct_assessments / max(total, 1)

    # Write results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {"metrics": metrics, "detailed_results": detailed_results},
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("=" * 60)
    logger.info("Critic Calibration Evaluation")
    logger.info("=" * 60)
    logger.info(f"Total evaluated:         {total}")
    logger.info(f"A1 accuracy:             {metrics['a1_accuracy']:.1%}")
    logger.info(f"Sycophancy rate:         {metrics['sycophancy_rate']:.1%} (target: < 50%)")
    logger.info(f"Correct rejection rate:  {metrics['correct_rejection_rate']:.1%} (target: > 50%)")
    logger.info(f"False negative rate:     {metrics['false_negative_rate']:.1%} (target: < 30%)")
    logger.info(f"Calibration accuracy:    {metrics['calibration_accuracy']:.1%}")
    logger.info(f"Output: {output_path}")

    return metrics


def _classify_feedback_stance(
    feedback_text: str,
    hedged_positive_patterns: list,
    positive_patterns: list,
    negative_patterns: list,
) -> str:
    """Classify whether feedback says the answer is correct, wrong, or unclear.

    Uses the same keyword patterns as the calibration reward to ensure
    consistent measurement.

    Args:
        feedback_text: Critic feedback text
        hedged_positive_patterns: Patterns for hedged positive (counted as negative)
        positive_patterns: Patterns for positive assessment
        negative_patterns: Patterns for negative assessment

    Returns:
        "positive", "negative", or "unclear"
    """
    # Check hedged positive first (these count as negative)
    has_hedged = any(p.search(feedback_text) for p in hedged_positive_patterns)
    has_positive = any(p.search(feedback_text) for p in positive_patterns)
    has_negative = any(p.search(feedback_text) for p in negative_patterns)

    if has_hedged:
        return "negative"
    if has_negative and not has_positive:
        return "negative"
    if has_positive and not has_negative:
        return "positive"
    if has_positive and has_negative:
        # Mixed signals — classify as negative (conservative)
        return "negative"

    return "unclear"


# ===========================================================================
# Pairwise Preference Reward Function (for GRPO training)
# ===========================================================================


def compute_pairwise_preference_reward(
    model_output: str,
    correct_choice: str,
) -> float:
    """Compute reward for a pairwise preference judgment.

    Checks if the model selected the correct response (A or B) by
    extracting from \\boxed{A} or \\boxed{B} pattern.

    Reward scheme:
        Correct choice: +1.0
        Wrong choice:    0.0
        No valid choice: 0.0 (model must learn to use the format)

    Args:
        model_output: Raw model output text
        correct_choice: "A" or "B"

    Returns:
        Reward value (0.0 or 1.0)
    """
    import re

    match = re.search(BOXED_PATTERN_STR, model_output)
    if not match:
        # Fallback: look for standalone "A" or "B" at end of text
        fallback = re.search(r"\b([AB])\s*[.!]?\s*$", model_output.strip())
        if fallback:
            chosen = fallback.group(1)
        else:
            return 0.0
    else:
        chosen = match.group(1)

    return 1.0 if chosen == correct_choice else 0.0


# ===========================================================================
# CLI
# ===========================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build pairwise preference data for critic GRPO warm-up",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Sub-command")

    # --- generate ---
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate K answer candidates per question using vLLM",
    )
    gen_parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to LIVR MCQ JSONL",
    )
    gen_parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model ID or local checkpoint",
    )
    gen_parser.add_argument(
        "--output_dir",
        type=str,
        default="/outputs/pairwise_preference/raw_answers",
        help="Directory to save answer candidates",
    )
    gen_parser.add_argument("--k_samples", type=int, default=16)
    gen_parser.add_argument("--temperature", type=float, default=1.0)
    gen_parser.add_argument("--max_tokens", type=int, default=256)
    gen_parser.add_argument("--batch_size", type=int, default=32)
    gen_parser.add_argument("--max_questions", type=int, default=0)
    gen_parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    gen_parser.add_argument("--tensor_parallel_size", type=int, default=4)
    gen_parser.add_argument("--image_base_dir", type=str, default="/outputs/image_base")
    gen_parser.add_argument("--seed", type=int, default=42)

    # --- build ---
    build_parser = subparsers.add_parser(
        "build",
        help="Build preference pairs from answer candidates",
    )
    build_parser.add_argument(
        "--answers_dir",
        type=str,
        required=True,
        help="Directory containing answer_candidates.jsonl",
    )
    build_parser.add_argument(
        "--output_path",
        type=str,
        default="/outputs/pairwise_preference/pairwise_grpo.jsonl",
        help="Output JSONL path for preference pairs",
    )
    build_parser.add_argument("--max_pairs_per_question", type=int, default=3)
    build_parser.add_argument("--seed", type=int, default=42)
    build_parser.add_argument(
        "--min_answer_length",
        type=int,
        default=1,
        help="Minimum answer length in characters",
    )

    # --- eval ---
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate critic calibration after warm-up",
    )
    eval_parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to LIVR MCQ JSONL",
    )
    eval_parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model checkpoint to evaluate",
    )
    eval_parser.add_argument(
        "--output_path",
        type=str,
        default="/outputs/pairwise_preference/eval_calibration.json",
        help="Path for evaluation results JSON",
    )
    eval_parser.add_argument("--n_samples", type=int, default=500)
    eval_parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    eval_parser.add_argument("--tensor_parallel_size", type=int, default=4)
    eval_parser.add_argument("--image_base_dir", type=str, default="/outputs/image_base")
    eval_parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    """Main entry point dispatching to subcommands."""
    args = parse_args()

    if args.command == "generate":
        generate_answer_candidates(
            dataset_path=args.dataset_path,
            model_id=args.model_id,
            output_dir=args.output_dir,
            k_samples=args.k_samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            max_questions=args.max_questions,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            image_base_dir=args.image_base_dir,
            seed=args.seed,
        )
    elif args.command == "build":
        build_preference_pairs(
            answers_dir=args.answers_dir,
            output_path=args.output_path,
            max_pairs_per_question=args.max_pairs_per_question,
            seed=args.seed,
            min_answer_length=args.min_answer_length,
        )
    elif args.command == "eval":
        evaluate_critic_calibration(
            dataset_path=args.dataset_path,
            model_id=args.model_id,
            output_path=args.output_path,
            n_samples=args.n_samples,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            image_base_dir=args.image_base_dir,
            seed=args.seed,
        )
    else:
        logger.error("No command specified. Use: generate, build, or eval")
        sys.exit(1)


if __name__ == "__main__":
    main()
