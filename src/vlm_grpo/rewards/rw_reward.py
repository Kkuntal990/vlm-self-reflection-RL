#!/usr/bin/env python3
"""
RW-focused reward composition for GRPO training.

Provides 5 separate reward functions that TRL's GRPOTrainer calls.
Each function receives completions + dataset columns as kwargs and
returns a list of scalar rewards.

Designed to be passed as a list to GRPOTrainer's reward_funcs parameter,
with corresponding reward_weights in GRPOConfig.

Usage:
    from vlm_grpo.rewards.rw_reward import get_reward_functions

    reward_fns = get_reward_functions()
    # Pass to GRPOTrainer:
    # trainer = GRPOTrainer(reward_funcs=reward_fns, ...)
"""

import logging
import sys
from typing import Any

from vlm_grpo.rewards.base import RewardBreakdown
from vlm_grpo.rewards.deterministic import (
    compute_feedback_calibration_reward,
    compute_final_correct_reward,
    compute_format_reward,
    compute_minimal_edit_reward,
    compute_no_regression_reward,
)
from vlm_grpo.trajectory import (
    extract_answer_from_text,
    extract_completion_text,
    parse_trajectory,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Shared parsing cache to avoid redundant work across reward functions.
# Maps completion text hash to (ParsedTrajectory, extracted_answer, format_valid).
_parse_cache: dict[int, tuple[Any, str, bool]] = {}
_parse_cache_max_size: int = 10000


def _get_parsed(
    completion_text: str,
    answer_type: str,
    choices: str,
) -> tuple[Any, str, bool]:
    """Get or compute parsed trajectory with caching.

    Args:
        completion_text: Raw completion text
        answer_type: Expected answer type
        choices: MCQ choices

    Returns:
        Tuple of (ParsedTrajectory, extracted_answer, format_valid)
    """
    cache_key = hash((completion_text, answer_type, choices))

    if cache_key in _parse_cache:
        return _parse_cache[cache_key]

    trajectory = parse_trajectory(completion_text)
    extracted = extract_answer_from_text(trajectory.final_answer, answer_type, choices)
    fmt_valid = compute_format_reward(trajectory, extracted, answer_type) > 0

    result = (trajectory, extracted, fmt_valid)

    if len(_parse_cache) < _parse_cache_max_size:
        _parse_cache[cache_key] = result

    return result


def _extract_text(completion: Any) -> str:
    """Safely extract text from a TRL completion object."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return extract_completion_text(completion)
    return str(completion)


# =============================================================================
# Individual Reward Functions (TRL-compatible signatures)
# =============================================================================


def format_reward_fn(
    completions: list,
    answer_type: list[str] | None = None,
    choices: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """R_format: Validate FEEDBACK:/FINAL_ANSWER: markers and answer type.

    Args:
        completions: Generated completions from GRPOTrainer
        answer_type: Answer type classifications (from dataset column)
        choices: MCQ choices (from dataset column)
        **kwargs: Additional kwargs from TRL

    Returns:
        List of format reward floats
    """
    rewards = []
    for i, comp in enumerate(completions):
        text = _extract_text(comp)
        a_type = answer_type[i] if answer_type else "open"
        ch = choices[i] if choices else ""

        trajectory, extracted, _ = _get_parsed(text, a_type, ch)
        reward = compute_format_reward(trajectory, extracted, a_type)
        rewards.append(reward)

    return rewards


def correctness_reward_fn(
    completions: list,
    ground_truth: list[str] | None = None,
    answer_type: list[str] | None = None,
    choices: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """R_final_correct: Check if final answer matches ground truth.

    Gated by format validity.

    Args:
        completions: Generated completions from GRPOTrainer
        ground_truth: Ground truth answers (from dataset column)
        answer_type: Answer type classifications (from dataset column)
        choices: MCQ choices (from dataset column)
        **kwargs: Additional kwargs from TRL

    Returns:
        List of correctness reward floats
    """
    rewards = []
    for i, comp in enumerate(completions):
        text = _extract_text(comp)
        gt = ground_truth[i] if ground_truth else ""
        a_type = answer_type[i] if answer_type else "open"
        ch = choices[i] if choices else ""

        _, extracted, fmt_valid = _get_parsed(text, a_type, ch)
        reward = compute_final_correct_reward(extracted, gt, a_type, fmt_valid)
        rewards.append(reward)

    return rewards


def no_regression_reward_fn(
    completions: list,
    ground_truth: list[str] | None = None,
    answer_type: list[str] | None = None,
    choices: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """R_no_regression: Penalize RW flips heavily.

    +1.0 for RR (maintained correctness), -3.0 for RW (regression).

    Args:
        completions: Generated completions from GRPOTrainer
        ground_truth: Ground truth answers (from dataset column)
        answer_type: Answer type classifications (from dataset column)
        choices: MCQ choices (from dataset column)
        **kwargs: Additional kwargs from TRL

    Returns:
        List of no-regression reward floats
    """
    rewards = []
    for i, comp in enumerate(completions):
        text = _extract_text(comp)
        gt = ground_truth[i] if ground_truth else ""
        a_type = answer_type[i] if answer_type else "open"
        ch = choices[i] if choices else ""

        _, extracted, fmt_valid = _get_parsed(text, a_type, ch)
        reward = compute_no_regression_reward(extracted, gt, a_type, fmt_valid)
        rewards.append(reward)

    return rewards


def minimal_edit_reward_fn(
    completions: list,
    ground_truth: list[str] | None = None,
    answer1: list[str] | None = None,
    answer_type: list[str] | None = None,
    choices: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """R_minimal_edit: Reward minimal edits when both answers are correct.

    Args:
        completions: Generated completions from GRPOTrainer
        ground_truth: Ground truth answers (from dataset column)
        answer1: Precomputed initial answers (from dataset column)
        answer_type: Answer type classifications (from dataset column)
        choices: MCQ choices (from dataset column)
        **kwargs: Additional kwargs from TRL

    Returns:
        List of minimal edit reward floats
    """
    rewards = []
    for i, comp in enumerate(completions):
        text = _extract_text(comp)
        gt = ground_truth[i] if ground_truth else ""
        a1 = answer1[i] if answer1 else ""
        a_type = answer_type[i] if answer_type else "open"
        ch = choices[i] if choices else ""

        _, extracted, fmt_valid = _get_parsed(text, a_type, ch)
        reward = compute_minimal_edit_reward(extracted, gt, a1, a_type, fmt_valid)
        rewards.append(reward)

    return rewards


def feedback_calibration_reward_fn(
    completions: list,
    answer_type: list[str] | None = None,
    choices: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """R_feedback_calibration: Reward calibrated feedback.

    Since Answer1 is always correct: rewards feedback saying "correct" /
    "no change needed" and penalizes feedback suggesting changes.

    Args:
        completions: Generated completions from GRPOTrainer
        answer_type: Answer type classifications (from dataset column)
        choices: MCQ choices (from dataset column)
        **kwargs: Additional kwargs from TRL

    Returns:
        List of feedback calibration reward floats
    """
    rewards = []
    for i, comp in enumerate(completions):
        text = _extract_text(comp)
        a_type = answer_type[i] if answer_type else "open"
        ch = choices[i] if choices else ""

        trajectory, _, _ = _get_parsed(text, a_type, ch)
        reward = compute_feedback_calibration_reward(trajectory.feedback)
        rewards.append(reward)

    return rewards


# =============================================================================
# Bundle and Utility Functions
# =============================================================================


def get_reward_functions() -> list:
    """Get all 5 reward functions as a list for TRL's reward_funcs param.

    Returns:
        Ordered list of reward functions:
        [format, correctness, no_regression, minimal_edit, feedback_calibration]
    """
    return [
        format_reward_fn,
        correctness_reward_fn,
        no_regression_reward_fn,
        minimal_edit_reward_fn,
        feedback_calibration_reward_fn,
    ]


def compute_full_breakdown(
    completion_text: str,
    ground_truth: str,
    answer1: str,
    answer_type: str,
    choices: str,
    dataset_name: str,
) -> RewardBreakdown:
    """Compute full reward breakdown for a single completion.

    Useful for sanity check mode and detailed logging.

    Args:
        completion_text: Raw completion text
        ground_truth: Ground truth answer
        answer1: Precomputed initial answer
        answer_type: Answer type classification
        choices: MCQ choices
        dataset_name: Source dataset name

    Returns:
        RewardBreakdown with all component values
    """
    trajectory, extracted, fmt_valid = _get_parsed(completion_text, answer_type, choices)

    components = {
        "format": compute_format_reward(trajectory, extracted, answer_type),
        "final_correct": compute_final_correct_reward(
            extracted, ground_truth, answer_type, fmt_valid
        ),
        "no_regression": compute_no_regression_reward(
            extracted, ground_truth, answer_type, fmt_valid
        ),
        "minimal_edit": compute_minimal_edit_reward(
            extracted, ground_truth, answer1, answer_type, fmt_valid
        ),
        "feedback_calibration": compute_feedback_calibration_reward(trajectory.feedback),
    }

    # Import here to avoid circular dependency
    from vlm_grpo.config import RewardWeights

    weights = RewardWeights()
    weight_map = {
        "format": weights.w_format,
        "final_correct": weights.w_final,
        "no_regression": weights.w_rw,
        "minimal_edit": weights.w_edit,
        "feedback_calibration": weights.w_fb,
    }

    weighted = {k: v * weight_map[k] for k, v in components.items()}
    total = sum(weighted.values())

    return RewardBreakdown(
        total_reward=total,
        components=components,
        weighted_components=weighted,
        format_valid=fmt_valid,
        parse_success=trajectory.parse_success,
        final_answer_extracted=extracted,
    )


def clear_parse_cache() -> None:
    """Clear the shared parsing cache. Call between training epochs."""
    _parse_cache.clear()
