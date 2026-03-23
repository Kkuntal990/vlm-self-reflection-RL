#!/usr/bin/env python3
"""
K-sample rollout engine for two-trajectory GRPO training.

Orchestrates the generation of K feedback (F1) and refined answer (A2)
pairs for each training sample. Used by the critic's custom GRPO loop
to compute downstream-aware rewards.

The rollout pipeline for each sample:
1. Build critic prompt → Generate K F1 completions (temperature sampling)
2. For each F1, build refiner prompt → Generate A2 (greedy)
3. Compute critic rewards from (F1, A2, ground_truth) pairs

All generation runs under torch.no_grad() to save memory.

Usage:
    from vlm_grpo.rollout import RolloutEngine, CriticRolloutResult

    engine = RolloutEngine(model, processor, config)
    results = engine.generate_critic_rollout(samples)
"""

import logging
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from vlm_grpo.config import CriticRewardWeights, RolloutConfig
from vlm_grpo.prompts import (
    build_critic_prompt,
    build_initial_answer_prompt,
    build_refiner_prompt,
)
from vlm_grpo.rewards.composition import (
    CriticRewardBreakdown,
    TrajectoryFeedbackRewardBreakdown,
    TrajectoryResponseRewardBreakdown,
    compute_critic_reward_breakdown,
    compute_feedback_reward_breakdown,
    compute_response_reward_breakdown,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Rollout Data Structures
# =============================================================================


@dataclass
class CriticRolloutResult:
    """Result of a full critic rollout for one sample.

    Contains K feedback completions and their corresponding A2 completions,
    along with pre-computed rewards for each (F1, A2) pair.

    Attributes:
        sample_index: Index in the original dataset
        question: Visual question text
        image_path: Path to the image
        ground_truth: Ground truth answer
        answer1: Initial answer (A1)
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        choices: MCQ choices string
        dataset_name: Source dataset name
        a1_is_correct: Whether A1 was correct
        feedbacks: K feedback texts from the critic
        answer2s: K A2 texts (one per feedback)
        rewards: K composite rewards
        reward_breakdowns: K full reward breakdowns
    """

    sample_index: int
    question: str
    image_path: str
    ground_truth: str
    answer1: str
    answer_type: str
    choices: str
    dataset_name: str
    a1_is_correct: bool
    feedbacks: list[str] = field(default_factory=list)
    answer2s: list[str] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    reward_breakdowns: list[CriticRewardBreakdown] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class RefinerRolloutResult:
    """Result of a refiner rollout for one (sample, feedback) pair.

    Contains K A2 completions for a fixed feedback.

    Attributes:
        sample_index: Index in the original dataset
        question: Visual question text
        image_path: Path to the image
        ground_truth: Ground truth answer
        answer1: Initial answer (A1)
        feedback1: The fixed feedback used
        answer_type: Answer type
        choices: MCQ choices string
        dataset_name: Source dataset name
        a1_is_correct: Whether A1 was correct
        answer2s: K A2 texts
    """

    sample_index: int
    question: str
    image_path: str
    ground_truth: str
    answer1: str
    feedback1: str
    answer_type: str
    choices: str
    dataset_name: str
    a1_is_correct: bool
    answer2s: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# Rollout Engine
# =============================================================================


class RolloutEngine:
    """Orchestrates K-sample rollout for two-trajectory GRPO.

    Generates K feedback and A2 pairs per sample using the model's
    generate() method. All generation runs under torch.no_grad().

    The engine is model-agnostic: it requires a model and processor
    that support the HuggingFace generate() interface.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        config: RolloutConfig,
        reward_weights: CriticRewardWeights,
        device: str = "cuda",
        model_type: str = "llava",
    ) -> None:
        """Initialize the rollout engine.

        Args:
            model: HuggingFace model with generate() method
            processor: Tokenizer/processor for the model
            config: Rollout configuration
            reward_weights: Critic reward weights for scoring
            device: Device for generation
            model_type: Model family ("llava" or "qwen2vl")
        """
        self.model = model
        self.processor = processor
        self.config = config
        self.reward_weights = reward_weights
        self.device = device
        self.model_type = model_type

    def generate_critic_rollout(
        self,
        samples: list[dict],
    ) -> list[CriticRolloutResult]:
        """Generate K F1+A2 pairs for each sample and compute rewards.

        For each sample:
        1. Build critic prompt → generate K F1 completions with temperature sampling
        2. For each F1, build refiner prompt → generate A2 with greedy decoding
        3. Compute critic rewards from (F1, A2, ground_truth)

        All generation runs under torch.no_grad().

        Args:
            samples: List of dataset sample dicts with fields: question,
                image_path, ground_truth, answer1, answer_type, choices,
                dataset_name, a1_is_correct

        Returns:
            List of CriticRolloutResult, one per sample
        """
        # Lazy import for heavy ML libraries
        import torch

        results = []
        batch_size = self.config.batch_size
        k = self.config.k_samples

        for batch_start in range(0, len(samples), batch_size):
            batch = samples[batch_start : batch_start + batch_size]

            for sample in batch:
                question = sample.get("question", "").replace("<image>", "").strip()
                image_path = sample.get("image_path", "")
                ground_truth = sample.get("ground_truth", "")
                answer1 = sample.get("answer1", "")
                answer_type = sample.get("answer_type", "open")
                choices_str = sample.get("choices", "")
                dataset_name = sample.get("dataset_name", "unknown")
                a1_is_correct = sample.get("a1_is_correct", True)
                sample_index = sample.get("sample_index", batch_start)
                image = sample.get("image")

                result = CriticRolloutResult(
                    sample_index=sample_index,
                    question=question,
                    image_path=image_path,
                    ground_truth=ground_truth,
                    answer1=answer1,
                    answer_type=answer_type,
                    choices=choices_str,
                    dataset_name=dataset_name,
                    a1_is_correct=a1_is_correct,
                )

                # Step 1: Generate K feedbacks
                with torch.no_grad():
                    feedbacks = self._generate_k_feedbacks(
                        question=question,
                        answer1=answer1,
                        answer_type=answer_type,
                        choices=choices_str,
                        image=image,
                        k=k,
                    )

                # Step 2: For each feedback, generate A2
                answer2s = []
                with torch.no_grad():
                    for f1 in feedbacks:
                        a2 = self._generate_a2(
                            question=question,
                            answer1=answer1,
                            feedback1=f1,
                            answer_type=answer_type,
                            choices=choices_str,
                            image=image,
                        )
                        answer2s.append(a2)

                # Step 3: Compute rewards
                rewards = []
                breakdowns = []
                for f1, a2 in zip(feedbacks, answer2s):
                    bd = compute_critic_reward_breakdown(
                        feedback_text=f1,
                        a2_text=a2,
                        ground_truth=ground_truth,
                        answer1=answer1,
                        a1_is_correct=a1_is_correct,
                        answer_type=answer_type,
                        choices=choices_str,
                        weights=self.reward_weights,
                    )
                    rewards.append(bd.total_reward)
                    breakdowns.append(bd)

                result.feedbacks = feedbacks
                result.answer2s = answer2s
                result.rewards = rewards
                result.reward_breakdowns = breakdowns
                results.append(result)

            processed = min(batch_start + batch_size, len(samples))
            logger.info(f"Rollout progress: {processed}/{len(samples)} samples")

        return results

    def _generate_k_feedbacks(
        self,
        question: str,
        answer1: str,
        answer_type: str,
        choices: str,
        image: Any,
        k: int,
    ) -> list[str]:
        """Generate K feedback completions for a single sample.

        Uses temperature sampling for diversity.

        Args:
            question: Visual question text
            answer1: Initial answer
            answer_type: Answer type
            choices: MCQ choices
            image: PIL Image
            k: Number of completions

        Returns:
            List of K feedback text strings
        """

        messages = build_critic_prompt(
            question, answer1, answer_type, choices, model_type=self.model_type
        )

        # Build input from messages using the processor
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Re-inject <image> if template stripped it from non-user roles (LLaVA only)
        if self.model_type != "qwen2vl" and image is not None and "<image>" not in text:
            text = "<image>\n" + text

        if image is not None:
            inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        feedbacks = []
        for _ in range(k):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_completion_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
            )
            # Decode only the generated tokens
            generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
            text_out = self.processor.decode(generated_ids, skip_special_tokens=True)
            feedbacks.append(text_out.strip())

        return feedbacks

    def _generate_a2(
        self,
        question: str,
        answer1: str,
        feedback1: str,
        answer_type: str,
        choices: str,
        image: Any,
    ) -> str:
        """Generate a single A2 completion for a (question, A1, F1) triple.

        Uses greedy decoding for deterministic A2.

        Args:
            question: Visual question text
            answer1: Initial answer
            feedback1: Feedback from the critic
            answer_type: Answer type
            choices: MCQ choices
            image: PIL Image

        Returns:
            A2 text string
        """

        messages = build_refiner_prompt(question, answer1, feedback1, answer_type, choices)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if image is not None:
            inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)

        temp = self.config.a2_temperature
        do_sample = temp > 0

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_completion_length,
            temperature=temp if do_sample else None,
            do_sample=do_sample,
        )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        text_out = self.processor.decode(generated_ids, skip_special_tokens=True)
        return text_out.strip()


# =============================================================================
# Rollout Metrics Computation
# =============================================================================


def compute_rollout_metrics(
    results: list[CriticRolloutResult],
) -> dict[str, float]:
    """Compute aggregate metrics from rollout results.

    Args:
        results: List of CriticRolloutResult from generate_critic_rollout

    Returns:
        Dict of metric name → value
    """
    if not results:
        return {}

    total_pairs = 0
    rr_count = 0
    rw_count = 0
    wr_count = 0
    ww_count = 0
    reward_sum = 0.0
    format_valid_count = 0
    feedback_lengths = []

    for r in results:
        for bd in r.reward_breakdowns:
            total_pairs += 1
            reward_sum += bd.total_reward
            if bd.format_valid:
                format_valid_count += 1
            feedback_lengths.append(len(bd.feedback_text.split()))

            if r.a1_is_correct:
                if bd.a2_correct is True:
                    rr_count += 1
                elif bd.a2_correct is False:
                    rw_count += 1
            else:
                if bd.a2_correct is True:
                    wr_count += 1
                elif bd.a2_correct is False:
                    ww_count += 1

    n = max(total_pairs, 1)
    metrics = {
        "rollout/total_pairs": float(total_pairs),
        "rollout/reward_mean": reward_sum / n,
        "rollout/rr_rate": rr_count / n,
        "rollout/rw_rate": rw_count / n,
        "rollout/wr_rate": wr_count / n,
        "rollout/ww_rate": ww_count / n,
        "rollout/format_valid_rate": format_valid_count / n,
        "rollout/feedback_length_mean": (
            sum(feedback_lengths) / len(feedback_lengths) if feedback_lengths else 0.0
        ),
    }

    return metrics


# =============================================================================
# Full Self-Reflection Rollout
# =============================================================================


@dataclass
class SelfReflectionRolloutResult:
    """Result of a full self-reflection rollout for one sample.

    Contains K trajectories of (A1, F1, A2) with two separate reward
    breakdowns per trajectory: one for response quality, one for feedback.

    Attributes:
        sample_index: Index in the original dataset
        question: Visual question text
        image_path: Path to the image
        ground_truth: Ground truth answer
        answer_type: Answer type
        choices: MCQ choices string
        dataset_name: Source dataset name
        answer1s: K initial answers
        feedbacks: K feedback texts
        answer2s: K refined answers
        response_rewards: K response reward scalars
        feedback_rewards: K feedback reward scalars
        response_breakdowns: K response reward breakdowns
        feedback_breakdowns: K feedback reward breakdowns
    """

    sample_index: int
    question: str
    image_path: str
    ground_truth: str
    answer_type: str
    choices: str
    dataset_name: str
    answer1s: list[str] = field(default_factory=list)
    feedbacks: list[str] = field(default_factory=list)
    answer2s: list[str] = field(default_factory=list)
    response_rewards: list[float] = field(default_factory=list)
    feedback_rewards: list[float] = field(default_factory=list)
    response_breakdowns: list[TrajectoryResponseRewardBreakdown] = field(default_factory=list)
    feedback_breakdowns: list[TrajectoryFeedbackRewardBreakdown] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def _generate_completions(
    model: Any,
    processor: Any,
    messages: list[dict],
    image: Any,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    k: int,
    model_type: str = "llava",
) -> list[str]:
    """Generate K completions for given messages.

    Args:
        model: HuggingFace model with generate() method
        processor: Tokenizer/processor for the model
        messages: Conversation messages
        image: PIL Image (or None)
        device: Device string
        max_new_tokens: Maximum tokens per generation
        temperature: Sampling temperature
        top_p: Top-p sampling
        k: Number of completions
        model_type: Model family ("llava" or "qwen2vl")

    Returns:
        List of K text completions
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Re-inject <image> token for LLaVA when template strips it from
    # non-user roles. Qwen2.5-VL handles images natively.
    if model_type != "qwen2vl" and image is not None and "<image>" not in text:
        text = "<image>\n" + text

    if image is not None:
        inputs = processor(text=text, images=image, return_tensors="pt").to(device)
    else:
        inputs = processor(text=text, return_tensors="pt").to(device)

    do_sample = temperature > 0
    prompt_len = inputs["input_ids"].shape[1]

    if k == 1:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
        )
        return [processor.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()]

    # TRL-style: expand the single prompt k times along the batch dim so all
    # k completions are generated in one parallel model.generate() call.
    batched_inputs = {
        key: val.expand(k, *val.shape[1:]).contiguous() for key, val in inputs.items()
    }
    logger.info(
        f"[generate A1] batch={batched_inputs['input_ids'].shape[0]} "
        f"seq_len={batched_inputs['input_ids'].shape[1]}"
    )
    outputs = model.generate(
        **batched_inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        do_sample=do_sample,
    )
    return [
        processor.decode(outputs[i][prompt_len:], skip_special_tokens=True).strip()
        for i in range(k)
    ]


def _generate_batch_completions(
    model: Any,
    processor: Any,
    messages_list: list[list[dict]],
    images: list[Any],
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    model_type: str = "llava",
) -> list[str]:
    """Generate one completion per message sequence, batched in one generate call.

    Supports different images per sequence, enabling full cross-sample batching
    (e.g. all N*K trajectories in one call). Mirrors TRL's repeat_interleave
    pattern where the entire rollout batch is generated in one shot.

    Args:
        model: HuggingFace model with generate() method
        processor: Tokenizer/processor for the model
        messages_list: N message sequences, one per trajectory
        images: N PIL Images (or Nones), one per sequence
        device: Device string
        max_new_tokens: Maximum tokens per generation
        temperature: Sampling temperature
        top_p: Top-p sampling
        model_type: Model family ("llava" or "qwen2vl")

    Returns:
        List of N text completions, one per input sequence
    """
    n = len(messages_list)
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]
    has_image = any(img is not None for img in images)
    # Re-inject <image> token for LLaVA only; Qwen handles images natively
    if has_image and model_type != "qwen2vl":
        texts = ["<image>\n" + t if "<image>" not in t else t for t in texts]

    # Left-pad for batched autoregressive generation so real tokens sit at
    # the right edge and generation continues from the correct position.
    orig_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        if has_image:
            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(device)
        else:
            inputs = processor(
                text=texts,
                return_tensors="pt",
                padding=True,
            ).to(device)

        do_sample = temperature > 0
        prompt_len = inputs["input_ids"].shape[1]  # same for all after left-padding

        logger.info(
            f"[generate batch] batch={inputs['input_ids'].shape[0]} "
            f"seq_len={inputs['input_ids'].shape[1]}"
        )
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
        )
    finally:
        processor.tokenizer.padding_side = orig_padding_side

    return [
        processor.decode(outputs[i][prompt_len:], skip_special_tokens=True).strip()
        for i in range(n)
    ]


def generate_self_reflection_rollout(
    model: Any,
    processor: Any,
    samples: list[dict],
    config: RolloutConfig,
    response_weights: Any,
    feedback_weights: Any,
    device: str = "cuda",
    model_type: str = "llava",
) -> list[SelfReflectionRolloutResult]:
    """Generate K full self-reflection trajectories per sample.

    For each sample, generates K independent chains:
        A1 (temperature) -> F1 (temperature) -> A2 (greedy)
    Then computes two separate rewards per trajectory.

    All generation runs under torch.no_grad().

    Args:
        model: HuggingFace model with generate() method
        processor: Tokenizer/processor
        samples: List of sample dicts with: question, image_path,
            ground_truth, answer_type, choices, dataset_name, image
        config: Rollout configuration
        response_weights: ResponseRewardWeights for scoring A1+A2
        feedback_weights: FeedbackRewardWeights for scoring F1
        device: Device for generation
        model_type: Model family ("llava" or "qwen2vl")

    Returns:
        List of SelfReflectionRolloutResult, one per sample
    """
    import torch

    n = len(samples)
    k = config.k_samples
    gen_batch = config.batch_size  # chunk size to avoid OOM on very large batches

    # Pre-extract per-sample fields
    questions = [s.get("question", "").replace("<image>", "").strip() for s in samples]
    images = [s.get("image") for s in samples]
    ground_truths = [s.get("ground_truth", "") for s in samples]
    answer_types = [s.get("answer_type", "open") for s in samples]
    choices_list = [s.get("choices", "") for s in samples]
    dataset_names = [s.get("dataset_name", "unknown") for s in samples]
    sample_indices = [s.get("sample_index", i) for i, s in enumerate(samples)]

    # Flat lists of all N*K trajectories, grouped repeat_interleave style:
    # [s0_t0, s0_t1, ..., s0_t(k-1), s1_t0, ..., s(n-1)_t(k-1)]
    all_a1s: list[str] = []
    all_f1s: list[str] = []
    all_a2s: list[str] = []

    with torch.no_grad():
        # Process in generation chunks to bound peak VRAM usage.
        # Each chunk generates gen_batch*k sequences per call.
        for chunk_start in range(0, n, gen_batch):
            chunk_end = min(chunk_start + gen_batch, n)
            chunk_qs = questions[chunk_start:chunk_end]
            chunk_imgs = images[chunk_start:chunk_end]
            chunk_size = chunk_end - chunk_start

            # Step 1: Generate K A1s for every sample in the chunk.
            # repeat_interleave: [q0]*k + [q1]*k + ... → chunk_size*k prompts
            a1_prompts = [build_initial_answer_prompt(q) for q in chunk_qs for _ in range(k)]
            imgs_expanded = [img for img in chunk_imgs for _ in range(k)]

            chunk_a1s = _generate_batch_completions(
                model,
                processor,
                a1_prompts,
                imgs_expanded,
                device,
                max_new_tokens=config.a1_max_completion_length,
                temperature=config.temperature,
                top_p=config.top_p,
                model_type=model_type,
            )
            all_a1s.extend(chunk_a1s)

            # Step 2: Generate F1 for each trajectory.
            f1_prompts = [
                build_critic_prompt(chunk_qs[i], chunk_a1s[i * k + j], model_type=model_type)
                for i in range(chunk_size)
                for j in range(k)
            ]
            chunk_f1s = _generate_batch_completions(
                model,
                processor,
                f1_prompts,
                imgs_expanded,
                device,
                max_new_tokens=config.f1_max_completion_length,
                temperature=config.feedback_temperature,
                top_p=config.top_p,
                model_type=model_type,
            )
            all_f1s.extend(chunk_f1s)

            # Step 3: Generate A2 for each trajectory.
            a2_prompts = [
                build_refiner_prompt(chunk_qs[i], chunk_a1s[i * k + j], chunk_f1s[i * k + j])
                for i in range(chunk_size)
                for j in range(k)
            ]
            chunk_a2s = _generate_batch_completions(
                model,
                processor,
                a2_prompts,
                imgs_expanded,
                device,
                max_new_tokens=config.a2_max_completion_length,
                temperature=config.a2_temperature,
                top_p=config.top_p,
                model_type=model_type,
            )
            all_a2s.extend(chunk_a2s)

            logger.info(
                f"Self-reflection rollout: {chunk_end}/{n} samples (gen_batch={chunk_size}, k={k})"
            )

    # Step 4: Pre-warm the LLM judge cache with one batched generate() call
    # covering all (a1, gt) and (a2, gt) pairs. Subsequent per-trajectory calls
    # inside verify_answer() will hit the shared _score_cache dict (O(1) lookup)
    # instead of running N individual generate() calls.
    from vlm_grpo.rewards.judge_llm import is_enabled, llm_judge_score_batch

    if is_enabled():
        judge_pairs: list[tuple[str, str]] = []
        for i in range(n):
            gt = ground_truths[i]
            for j in range(k):
                judge_pairs.append((all_a1s[i * k + j], gt))
                judge_pairs.append((all_a2s[i * k + j], gt))
        llm_judge_score_batch(judge_pairs)

    # Step 5: Compute rewards and assemble results, one per sample.
    results = []
    for i in range(n):
        traj_slice = slice(i * k, (i + 1) * k)
        answer1s = all_a1s[traj_slice]
        feedbacks = all_f1s[traj_slice]
        answer2s = all_a2s[traj_slice]

        result = SelfReflectionRolloutResult(
            sample_index=sample_indices[i],
            question=questions[i],
            image_path=samples[i].get("image_path", ""),
            ground_truth=ground_truths[i],
            answer_type=answer_types[i],
            choices=choices_list[i],
            dataset_name=dataset_names[i],
        )

        for a1, f1, a2 in zip(answer1s, feedbacks, answer2s):
            resp_bd = compute_response_reward_breakdown(
                a1_text=a1,
                a2_text=a2,
                ground_truth=ground_truths[i],
                answer_type=answer_types[i],
                choices=choices_list[i],
                weights=response_weights,
            )
            fb_bd = compute_feedback_reward_breakdown(
                feedback_text=f1,
                a1_text=a1,
                a2_text=a2,
                ground_truth=ground_truths[i],
                answer_type=answer_types[i],
                choices=choices_list[i],
                weights=feedback_weights,
            )
            result.response_rewards.append(resp_bd.total_reward)
            result.feedback_rewards.append(fb_bd.total_reward)
            result.response_breakdowns.append(resp_bd)
            result.feedback_breakdowns.append(fb_bd)

        result.answer1s = list(answer1s)
        result.feedbacks = list(feedbacks)
        result.answer2s = list(answer2s)
        results.append(result)

    return results


def compute_self_reflection_metrics(
    results: list[SelfReflectionRolloutResult],
) -> dict[str, float]:
    """Compute aggregate metrics from self-reflection rollout results.

    Args:
        results: List of SelfReflectionRolloutResult

    Returns:
        Dict of metric name -> value
    """
    if not results:
        return {}

    total = 0
    a1_correct_count = 0
    rr_count = 0
    rw_count = 0
    wr_count = 0
    ww_count = 0
    resp_reward_sum = 0.0
    fb_reward_sum = 0.0
    fb_format_valid_count = 0

    for r in results:
        for resp_bd, fb_bd in zip(r.response_breakdowns, r.feedback_breakdowns):
            total += 1
            resp_reward_sum += resp_bd.total_reward
            fb_reward_sum += fb_bd.total_reward
            if resp_bd.a1_correct:
                a1_correct_count += 1
            if fb_bd.feedback_format_valid:
                fb_format_valid_count += 1

            if resp_bd.a1_correct:
                if resp_bd.a2_correct:
                    rr_count += 1
                else:
                    rw_count += 1
            else:
                if resp_bd.a2_correct:
                    wr_count += 1
                else:
                    ww_count += 1

    n = max(total, 1)
    return {
        "sr/total_trajectories": float(total),
        "sr/a1_accuracy": a1_correct_count / n,
        "sr/rr_rate": rr_count / n,
        "sr/rw_rate": rw_count / n,
        "sr/wr_rate": wr_count / n,
        "sr/ww_rate": ww_count / n,
        "sr/response_reward_mean": resp_reward_sum / n,
        "sr/feedback_reward_mean": fb_reward_sum / n,
        "sr/feedback_format_valid_rate": fb_format_valid_count / n,
    }
