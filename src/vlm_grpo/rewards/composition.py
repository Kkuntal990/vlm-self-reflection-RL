#!/usr/bin/env python3
"""
Reward composition for two-trajectory GRPO training.

Aggregates individual reward components into composite scores for both
critic and refiner training. Provides TRL-compatible reward function
wrappers for the refiner (which uses standard GRPOTrainer).

Usage:
    from vlm_grpo.rewards.composition import (
        CriticRewardWeights,
        RefinerRewardWeights,
        compute_critic_reward_breakdown,
        get_refiner_reward_functions,
    )

    # Critic: compute full breakdown (used in custom training loop)
    breakdown = compute_critic_reward_breakdown(
        feedback_text="The answer is correct.",
        a2_text="A",
        ground_truth="A", answer1="A", a1_is_correct=True,
        answer_type="mcq", choices="",
        weights=CriticRewardWeights(),
    )

    # Refiner: get TRL-compatible reward functions
    reward_fns = get_refiner_reward_functions()
"""

import logging
import re
import sys
from dataclasses import asdict, dataclass
from typing import Any

from vlm_grpo.rewards.correctness import compute_a2_correctness_reward
from vlm_grpo.rewards.feedback import (
    compute_downstream_aware_reward,
    compute_feedback_calibration_reward,
)
from vlm_grpo.rewards.stability import (
    compute_minimal_edit_reward,
    compute_no_regression_reward,
)
from vlm_grpo.rewards.verifier import verify_answer
from vlm_grpo.trajectory import (
    detect_hedging,
    extract_answer_from_text,
    extract_completion_text,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Weight Dataclasses
# =============================================================================


@dataclass
class CriticRewardWeights:
    """Weights for critic reward composition.

    reward = w_downstream * R_downstream
           + w_calibration * R_calibration
           + w_format * R_format

    Attributes:
        w_downstream: Weight for downstream-aware reward (dominant)
        w_calibration: Weight for feedback calibration
        w_format: Weight for format compliance
    """

    w_downstream: float = 2.0
    w_calibration: float = 1.0
    w_format: float = 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_list(self) -> list[float]:
        """Return weights as ordered list: [format, calibration, downstream]."""
        return [self.w_format, self.w_calibration, self.w_downstream]


@dataclass
class RefinerRewardWeights:
    """Weights for refiner reward composition.

    reward = w_correctness * R_correctness
           + w_no_regression * R_no_regression
           + w_minimal_edit * R_minimal_edit
           + w_format * R_format

    Attributes:
        w_correctness: Weight for A2 correctness
        w_no_regression: Weight for no-regression penalty (dominant)
        w_minimal_edit: Weight for minimal edit reward
        w_format: Weight for format compliance
    """

    w_correctness: float = 1.0
    w_no_regression: float = 2.0
    w_minimal_edit: float = 0.3
    w_format: float = 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_list(self) -> list[float]:
        """Return weights as ordered list: [format, correctness, no_regression, minimal_edit]."""
        return [
            self.w_format,
            self.w_correctness,
            self.w_no_regression,
            self.w_minimal_edit,
        ]


# =============================================================================
# Reward Breakdown Dataclasses
# =============================================================================


@dataclass
class CriticRewardBreakdown:
    """Full reward breakdown for one F1 completion.

    Attributes:
        total_reward: Weighted sum of all components
        components: Dict mapping component name to raw reward value
        weighted_components: Dict mapping component name to weighted value
        feedback_text: The feedback text from the critic
        a2_text: The A2 text generated from this feedback
        a2_extracted: Normalized A2 answer
        a2_correct: Whether A2 is correct
        format_valid: Whether the feedback format was valid
    """

    total_reward: float
    components: dict[str, float]
    weighted_components: dict[str, float]
    feedback_text: str
    a2_text: str
    a2_extracted: str
    a2_correct: bool
    format_valid: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class RefinerRewardBreakdown:
    """Full reward breakdown for one A2 completion.

    Attributes:
        total_reward: Weighted sum of all components
        components: Dict mapping component name to raw reward value
        weighted_components: Dict mapping component name to weighted value
        a2_extracted: Normalized A2 answer
        a2_correct: Whether A2 is correct (None if undetermined)
        format_valid: Whether the A2 format was valid
    """

    total_reward: float
    components: dict[str, float]
    weighted_components: dict[str, float]
    a2_extracted: str
    a2_correct: bool
    format_valid: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# Critic Format Reward
# =============================================================================

# Stance keywords: feedback must take a position on A1's correctness
_CRITIC_FORMAT_STANCE_KEYWORDS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bcorrect\b",
        r"\bwrong\b",
        r"\bno change\b",
        r"\bshould be\b",
        r"\bincorrect\b",
        r"\baccurate\b",
    ]
]


def compute_critic_format_reward(feedback_text: str) -> float:
    """R_format for critic: validate feedback format with stance detection.

    Three-tier scoring:
        Empty or <3 words            → -2.0  (heavy penalty for safe empty)
        3-4 words, no stance keyword → -1.0  (weak/vague feedback)
        >=5 words OR stance keyword  → +1.0  (substantive feedback)

    Args:
        feedback_text: Feedback text from the critic

    Returns:
        Format reward in {-2.0, -1.0, +1.0}
    """
    stripped = feedback_text.strip()
    if not stripped:
        return -2.0

    word_count = len(stripped.split())
    if word_count < 3:
        return -2.0

    has_stance = any(p.search(stripped) for p in _CRITIC_FORMAT_STANCE_KEYWORDS)

    if word_count >= 5 or has_stance:
        return 1.0

    return -1.0


def compute_critic_reward_breakdown(
    feedback_text: str,
    a2_text: str,
    ground_truth: str,
    answer1: str,
    a1_is_correct: bool,
    answer_type: str,
    choices: str,
    weights: CriticRewardWeights,
) -> CriticRewardBreakdown:
    """Compute full reward breakdown for one critic completion.

    Combines: downstream-aware, calibration, and format rewards.

    Args:
        feedback_text: Feedback text from the critic
        a2_text: A2 text generated from this feedback
        ground_truth: Ground truth answer
        answer1: Initial answer (A1) text
        a1_is_correct: Whether A1 was correct
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        choices: MCQ choices string (empty for non-MCQ)
        weights: Reward weight configuration

    Returns:
        CriticRewardBreakdown with all component scores
    """
    # Extract normalized A2 answer
    a2_extracted = extract_answer_from_text(a2_text, answer_type, choices)

    # Format reward
    r_format = compute_critic_format_reward(feedback_text)
    format_valid = r_format > 0

    # Downstream-aware reward
    r_downstream = compute_downstream_aware_reward(
        feedback_text=feedback_text,
        a2_extracted=a2_extracted,
        ground_truth=ground_truth,
        answer_type=answer_type,
        a1=answer1,
        a1_is_correct=a1_is_correct,
    )

    # Calibration reward
    r_calibration = compute_feedback_calibration_reward(
        feedback_text=feedback_text,
        a1_is_correct=a1_is_correct,
    )

    # Determine A2 correctness for logging
    a2_result = verify_answer(a2_extracted, ground_truth, answer_type)
    a2_correct = a2_result.is_correct

    # Compose weighted reward
    components = {
        "format": r_format,
        "downstream": r_downstream,
        "calibration": r_calibration,
    }
    weighted_components = {
        "format": r_format * weights.w_format,
        "downstream": r_downstream * weights.w_downstream,
        "calibration": r_calibration * weights.w_calibration,
    }
    total_reward = sum(weighted_components.values())

    return CriticRewardBreakdown(
        total_reward=total_reward,
        components=components,
        weighted_components=weighted_components,
        feedback_text=feedback_text,
        a2_text=a2_text,
        a2_extracted=a2_extracted,
        a2_correct=a2_correct,
        format_valid=format_valid,
    )


# =============================================================================
# Refiner Reward Computation (for sanity checks and logging)
# =============================================================================


def compute_refiner_reward_breakdown(
    a2_text: str,
    ground_truth: str,
    answer1: str,
    a1_is_correct: bool,
    answer_type: str,
    choices: str,
    weights: RefinerRewardWeights,
) -> RefinerRewardBreakdown:
    """Compute full reward breakdown for one refiner completion.

    Combines: correctness, no-regression, minimal-edit, and format rewards.

    Args:
        a2_text: A2 text from the refiner
        ground_truth: Ground truth answer
        answer1: Initial answer (A1) text
        a1_is_correct: Whether A1 was correct
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        choices: MCQ choices string (empty for non-MCQ)
        weights: Reward weight configuration

    Returns:
        RefinerRewardBreakdown with all component scores
    """
    # Extract normalized A2 answer
    a2_extracted = extract_answer_from_text(a2_text, answer_type, choices)

    # Format reward (A2 should be a valid answer)
    r_format = _compute_refiner_format_reward(a2_extracted, a2_text, answer_type)
    format_valid = r_format > 0

    # Correctness reward
    r_correctness = compute_a2_correctness_reward(
        a2_extracted=a2_extracted,
        ground_truth=ground_truth,
        answer_type=answer_type,
        format_valid=format_valid,
    )

    # No-regression reward
    r_no_regression = compute_no_regression_reward(
        a2_extracted=a2_extracted,
        ground_truth=ground_truth,
        answer_type=answer_type,
        a1_is_correct=a1_is_correct,
        format_valid=format_valid,
    )

    # Minimal-edit reward
    r_minimal_edit = compute_minimal_edit_reward(
        a1=answer1,
        a2_extracted=a2_extracted,
        ground_truth=ground_truth,
        answer_type=answer_type,
        format_valid=format_valid,
    )

    # Determine A2 correctness for logging
    a2_result = verify_answer(a2_extracted, ground_truth, answer_type)
    a2_correct = a2_result.is_correct

    # Compose weighted reward
    components = {
        "format": r_format,
        "correctness": r_correctness,
        "no_regression": r_no_regression,
        "minimal_edit": r_minimal_edit,
    }
    weighted_components = {
        "format": r_format * weights.w_format,
        "correctness": r_correctness * weights.w_correctness,
        "no_regression": r_no_regression * weights.w_no_regression,
        "minimal_edit": r_minimal_edit * weights.w_minimal_edit,
    }
    total_reward = sum(weighted_components.values())

    return RefinerRewardBreakdown(
        total_reward=total_reward,
        components=components,
        weighted_components=weighted_components,
        a2_extracted=a2_extracted,
        a2_correct=a2_correct,
        format_valid=format_valid,
    )


# =============================================================================
# TRL-Compatible Reward Functions (for refiner GRPOTrainer)
# =============================================================================


def _extract_a2_text(completion: Any) -> str:
    """Safely extract A2 text from TRL completion object.

    Args:
        completion: Completion in TRL format (str or list[dict])

    Returns:
        Plain text string
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return extract_completion_text(completion)
    return str(completion)


def _compute_refiner_format_reward(
    a2_extracted: str,
    a2_text: str,
    answer_type: str,
) -> float:
    """R_format for refiner: validate A2 format.

    Checks that A2 is non-empty and appropriate for the answer type.

    Args:
        a2_extracted: Normalized extracted A2 answer
        a2_text: Raw A2 text
        answer_type: Expected answer type

    Returns:
        +1.0 if valid, -1.0 if invalid
    """
    if not a2_text.strip():
        return -1.0
    if not a2_extracted:
        return -1.0
    if answer_type == "yesno" and detect_hedging(a2_text):
        return -1.0
    return 1.0


def refiner_format_reward_fn(
    completions: list,
    answer_type: list[str] | None = None,
    choices: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """TRL-compatible format reward for refiner A2 completions.

    Args:
        completions: List of K completions from GRPOTrainer
        answer_type: List of answer types (from dataset columns)
        choices: List of MCQ choices (from dataset columns)
        **kwargs: Additional dataset columns (ignored)

    Returns:
        List of format rewards, one per completion
    """
    rewards = []
    for i, comp in enumerate(completions):
        at = answer_type[i] if answer_type else "open"
        ch = choices[i] if choices else ""
        a2_text = _extract_a2_text(comp)
        a2_extracted = extract_answer_from_text(a2_text, at, ch)
        r = _compute_refiner_format_reward(a2_extracted, a2_text, at)
        rewards.append(r)
    return rewards


def refiner_correctness_reward_fn(
    completions: list,
    ground_truth: list[str] | None = None,
    answer_type: list[str] | None = None,
    choices: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """TRL-compatible correctness reward for refiner A2 completions.

    Args:
        completions: List of K completions from GRPOTrainer
        ground_truth: List of ground truths (from dataset columns)
        answer_type: List of answer types (from dataset columns)
        choices: List of MCQ choices (from dataset columns)
        **kwargs: Additional dataset columns (ignored)

    Returns:
        List of correctness rewards, one per completion
    """
    rewards = []
    for i, comp in enumerate(completions):
        gt = ground_truth[i] if ground_truth else ""
        at = answer_type[i] if answer_type else "open"
        ch = choices[i] if choices else ""

        a2_text = _extract_a2_text(comp)
        a2_extracted = extract_answer_from_text(a2_text, at, ch)
        format_valid = bool(a2_extracted)

        r = compute_a2_correctness_reward(
            a2_extracted=a2_extracted,
            ground_truth=gt,
            answer_type=at,
            format_valid=format_valid,
        )
        rewards.append(r)
    return rewards


def refiner_no_regression_reward_fn(
    completions: list,
    ground_truth: list[str] | None = None,
    answer_type: list[str] | None = None,
    choices: list[str] | None = None,
    a1_is_correct: list[bool] | None = None,
    **kwargs: Any,
) -> list[float]:
    """TRL-compatible no-regression reward for refiner A2 completions.

    Args:
        completions: List of K completions from GRPOTrainer
        ground_truth: List of ground truths (from dataset columns)
        answer_type: List of answer types (from dataset columns)
        choices: List of MCQ choices (from dataset columns)
        a1_is_correct: List of A1 correctness flags (from dataset columns)
        **kwargs: Additional dataset columns (ignored)

    Returns:
        List of no-regression rewards, one per completion
    """
    rewards = []
    for i, comp in enumerate(completions):
        gt = ground_truth[i] if ground_truth else ""
        at = answer_type[i] if answer_type else "open"
        ch = choices[i] if choices else ""
        a1_correct = a1_is_correct[i] if a1_is_correct else True

        a2_text = _extract_a2_text(comp)
        a2_extracted = extract_answer_from_text(a2_text, at, ch)
        format_valid = bool(a2_extracted)

        r = compute_no_regression_reward(
            a2_extracted=a2_extracted,
            ground_truth=gt,
            answer_type=at,
            a1_is_correct=a1_correct,
            format_valid=format_valid,
        )
        rewards.append(r)
    return rewards


def refiner_minimal_edit_reward_fn(
    completions: list,
    ground_truth: list[str] | None = None,
    answer1: list[str] | None = None,
    answer_type: list[str] | None = None,
    choices: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """TRL-compatible minimal-edit reward for refiner A2 completions.

    Args:
        completions: List of K completions from GRPOTrainer
        ground_truth: List of ground truths (from dataset columns)
        answer1: List of A1 answers (from dataset columns)
        answer_type: List of answer types (from dataset columns)
        choices: List of MCQ choices (from dataset columns)
        **kwargs: Additional dataset columns (ignored)

    Returns:
        List of minimal-edit rewards, one per completion
    """
    rewards = []
    for i, comp in enumerate(completions):
        gt = ground_truth[i] if ground_truth else ""
        a1 = answer1[i] if answer1 else ""
        at = answer_type[i] if answer_type else "open"
        ch = choices[i] if choices else ""

        a2_text = _extract_a2_text(comp)
        a2_extracted = extract_answer_from_text(a2_text, at, ch)
        format_valid = bool(a2_extracted)

        r = compute_minimal_edit_reward(
            a1=a1,
            a2_extracted=a2_extracted,
            ground_truth=gt,
            answer_type=at,
            format_valid=format_valid,
        )
        rewards.append(r)
    return rewards


def get_refiner_reward_functions() -> list:
    """Return ordered list of TRL-compatible reward functions for refiner.

    Order matches RefinerRewardWeights.to_list():
    [format, correctness, no_regression, minimal_edit]

    Returns:
        List of reward function callables
    """
    return [
        refiner_format_reward_fn,
        refiner_correctness_reward_fn,
        refiner_no_regression_reward_fn,
        refiner_minimal_edit_reward_fn,
    ]
