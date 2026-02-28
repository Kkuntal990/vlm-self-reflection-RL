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
from vlm_grpo.prompts import build_critic_prompt, build_refiner_prompt
from vlm_grpo.rewards.composition import (
    CriticRewardBreakdown,
    compute_critic_reward_breakdown,
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
    ) -> None:
        """Initialize the rollout engine.

        Args:
            model: HuggingFace model with generate() method
            processor: Tokenizer/processor for the model
            config: Rollout configuration
            reward_weights: Critic reward weights for scoring
            device: Device for generation
        """
        self.model = model
        self.processor = processor
        self.config = config
        self.reward_weights = reward_weights
        self.device = device

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

        messages = build_critic_prompt(question, answer1, answer_type, choices)

        # Build input from messages using the processor
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if image is not None:
            inputs = self.processor(
                text=text, images=image, return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.processor(
                text=text, return_tensors="pt"
            ).to(self.device)

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

        messages = build_refiner_prompt(
            question, answer1, feedback1, answer_type, choices
        )

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if image is not None:
            inputs = self.processor(
                text=text, images=image, return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.processor(
                text=text, return_tensors="pt"
            ).to(self.device)

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
