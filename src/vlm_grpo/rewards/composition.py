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

from vlm_grpo.rewards.correctness import (
    compute_a1_correctness_01,
    compute_a2_correctness_reward,
)
from vlm_grpo.rewards.feedback import compute_downstream_aware_reward
from vlm_grpo.rewards.stability import compute_no_regression_reward
from vlm_grpo.rewards.verifier import verify_answer
from vlm_grpo.trajectory import (
    extract_answer_from_text,
    extract_completion_text,
    normalize_answer,
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

    Attributes:
        w_downstream: Weight for downstream-aware reward
    """

    w_downstream: float = 2.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_list(self) -> list[float]:
        """Return weights as ordered list: [downstream]."""
        return [self.w_downstream]


@dataclass
class RefinerRewardWeights:
    """Weights for refiner reward composition.

    reward = w_correctness * R_correctness
           + w_no_regression * R_no_regression
           + w_format * R_format

    Attributes:
        w_correctness: Weight for A2 correctness
        w_no_regression: Weight for no-regression penalty (dominant)
        w_format: Weight for format compliance
    """

    w_correctness: float = 1.0
    w_no_regression: float = 2.0
    w_format: float = 0.15

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_list(self) -> list[float]:
        """Return weights as ordered list: [format, correctness, no_regression]."""
        return [
            self.w_format,
            self.w_correctness,
            self.w_no_regression,
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
    """

    total_reward: float
    components: dict[str, float]
    weighted_components: dict[str, float]
    feedback_text: str
    a2_text: str
    a2_extracted: str
    a2_correct: bool

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
# F1 Verification Accuracy
# =============================================================================


def compute_verification_accuracy_reward(
    feedback_text: str,
    a1_is_correct: bool,
) -> float:
    """R_verification: does F1's verdict match A1's actual correctness?

    Verdict is extracted ONLY from `\\boxed{...}` (LLaVA-Critic-R1 style).
    Missing boxed → -1 (treat as wrong verdict). No keyword-fallback on
    `<think>` prose — that path was a noise source (~3% of trajectories
    inherited a verdict from think text rather than the explicit boxed
    output).

    Args:
        feedback_text: Verification output from the model
        a1_is_correct: Whether A1 was actually correct

    Returns:
        +1.0 if boxed verdict matches A1 truth,
        -1.0 if boxed verdict contradicts A1 truth, or boxed verdict is
              missing/unparseable (treat as wrong verdict).
    """
    from vlm_grpo.trajectory import extract_from_boxed

    boxed = extract_from_boxed(feedback_text).upper()
    if boxed == "INCORRECT":
        return 1.0 if not a1_is_correct else -1.0
    if boxed == "CORRECT":
        return 1.0 if a1_is_correct else -1.0
    # Missing or unparseable boxed verdict → wrong (no extraction → wrong answer).
    return -1.0


def compute_feedback_format_reward(feedback_text: str) -> float:
    """R_fb_format: structure + clean verdict check for F1.

    Binary {0, +1}. Two conditions must both hold for +1:
      1. `<think>...</think>...\\boxed{...}` structure present (in order).
      2. Inner content of `\\boxed{}` is exactly `CORRECT` or `INCORRECT`
         (case-insensitive) — no descriptor, no garbage like `\\boxed{(A)}`.

    Verdict correctness (vs A1 truth) is handled separately by
    R_verification; this is a pure format compliance bonus.

    Args:
        feedback_text: Raw F1 output

    Returns:
        +1.0 if structure present AND boxed content is CORRECT/INCORRECT.
         0.0 otherwise (no penalty, just no bonus).
    """
    from vlm_grpo.trajectory import extract_from_boxed, has_think_boxed

    if not has_think_boxed(feedback_text):
        return 0.0
    verdict = extract_from_boxed(feedback_text).upper()
    return 1.0 if verdict in ("CORRECT", "INCORRECT") else 0.0


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

    Combines downstream-aware reward only.

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

    # Downstream-aware reward (pass raw text for MCQ answer text matching)
    r_downstream = compute_downstream_aware_reward(
        feedback_text=feedback_text,
        a2_extracted=a2_text,
        ground_truth=ground_truth,
        answer_type=answer_type,
        a1=answer1,
        a1_is_correct=a1_is_correct,
    )

    # Determine A2 correctness for logging
    a2_result = verify_answer(a2_text, ground_truth, answer_type)
    a2_correct = a2_result.is_correct

    # Compose weighted reward
    components = {
        "downstream": r_downstream,
    }
    weighted_components = {
        "downstream": r_downstream * weights.w_downstream,
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
    use_think_answer_tags: bool = False,
    use_answer_tag_only: bool = False,
) -> RefinerRewardBreakdown:
    """Compute full reward breakdown for one refiner completion.

    Combines: correctness, no-regression, and format rewards.

    Args:
        a2_text: A2 text from the refiner
        ground_truth: Ground truth answer
        answer1: Initial answer (A1) text
        a1_is_correct: Whether A1 was correct
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        choices: MCQ choices string (empty for non-MCQ)
        weights: Reward weight configuration
        use_think_answer_tags: Whether to check for tag format

    Returns:
        RefinerRewardBreakdown with all component scores
    """
    # Format reward
    r_format = _compute_refiner_format_reward(
        a2_text, answer_type, ground_truth, use_think_answer_tags, use_answer_tag_only
    )
    format_valid = r_format >= 0

    # Extract A2 for correctness (liberal extraction, separate from format)
    a2_extracted = extract_answer_from_text(a2_text, answer_type, choices)

    # Correctness reward (pass raw text for MCQ answer text matching)
    r_correctness = compute_a2_correctness_reward(
        a2_extracted=a2_text,
        ground_truth=ground_truth,
        answer_type=answer_type,
    )

    # No-regression reward
    r_no_regression = compute_no_regression_reward(
        a2_extracted=a2_text,
        ground_truth=ground_truth,
        answer_type=answer_type,
        a1_is_correct=a1_is_correct,
    )

    # Determine A2 correctness for logging
    a2_result = verify_answer(a2_text, ground_truth, answer_type)
    a2_correct = a2_result.is_correct

    # Compose weighted reward
    components = {
        "format": r_format,
        "correctness": r_correctness,
        "no_regression": r_no_regression,
    }
    weighted_components = {
        "format": r_format * weights.w_format,
        "correctness": r_correctness * weights.w_correctness,
        "no_regression": r_no_regression * weights.w_no_regression,
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


# Threshold for LLM format fallback (0-1 scale, 0.7 = score 7/10)
_LLM_FORMAT_THRESHOLD: float = 0.7


def _llm_format_fallback(
    a2_text: str,
    ground_truth: str,
    answer_type: str,
) -> float:
    """LLM fallback for format check when deterministic check fails.

    Only invoked when ground_truth is available and LLM judge is enabled.

    Args:
        a2_text: Raw A2 text
        ground_truth: Ground truth answer
        answer_type: Answer type string

    Returns:
        0.0 if LLM says format matches, -1.0 otherwise
    """
    if not ground_truth:
        return -1.0

    try:
        from vlm_grpo.rewards.judge_llm import is_enabled, llm_format_judge

        if not is_enabled():
            return -1.0

        score = llm_format_judge(a2_text.strip(), ground_truth, answer_type)
        if score >= _LLM_FORMAT_THRESHOLD:
            return 0.0
        return -1.0
    except Exception as e:
        logger.warning(f"LLM format fallback failed: {e}")
        return -1.0


def _compute_refiner_format_reward(
    a2_text: str,
    answer_type: str,
    ground_truth: str = "",
    use_think_answer_tags: bool = False,
    use_answer_tag_only: bool = False,
) -> float:
    """R_format for refiner: format compliance check.

    Three modes:

    **Think+Answer tag mode** (use_think_answer_tags=True):
        Checks for <think>...</think><answer>...</answer> structure.
        Then validates the inner answer content by type.
        +0.5 fully compliant, -0.5 tags but bad inner, -1.0 tags missing.

    **Answer-tag-only mode** (use_answer_tag_only=True):
        Checks for <answer>...</answer> only (no <think> required).
        Validates inner answer content by type.
        +0.5 fully compliant, -0.5 tag but bad inner, -1.0 tag missing.

    **Bare mode** (both False):
        Original normalize-first pipeline.
        0.0 when compliant, -1.0 when not.

    Args:
        a2_text: Raw A2 text from the refiner
        answer_type: Expected answer type
        ground_truth: Ground truth answer (for LLM format fallback)
        use_think_answer_tags: Whether to check for <think>+<answer> tags
        use_answer_tag_only: Whether to check for <answer> tag only

    Returns:
        Format reward score
    """
    if use_think_answer_tags:
        return _compute_tag_format_reward(a2_text, answer_type, ground_truth)

    if use_answer_tag_only:
        return _compute_answer_tag_only_format_reward(a2_text, answer_type)

    return _compute_bare_format_reward(a2_text, answer_type, ground_truth)


def _compute_tag_format_reward(
    a2_text: str,
    answer_type: str,
    ground_truth: str = "",
) -> float:
    """Format reward for A2 tag mode: structure + clean atomic inner.

    Binary {0, +1}. Two conditions must both hold for +1:
      1. Both `<think>...</think>` and `<answer>...</answer>` present.
      2. Inner content of `<answer>` is exactly an atomic answer for the
         expected type (MCQ letter, integer, yes/no, etc.) — no descriptor
         text, no prose.

    Correctness extraction is more lenient (accepts `(A) descriptor`),
    but format reward demands the clean canonical form.

    Args:
        a2_text: Raw A2 text.
        answer_type: Expected answer type.
        ground_truth: Ground truth (unused, kept for API compat).

    Returns:
        +1.0 if both tags present AND inner is a clean atomic answer.
         0.0 otherwise.
    """
    from vlm_grpo.trajectory import extract_from_answer_tags, has_think_answer_tags

    if not has_think_answer_tags(a2_text):
        return 0.0
    inner = extract_from_answer_tags(a2_text).strip()
    if not inner:
        return 0.0
    if not _is_clean_atomic_answer(inner, answer_type):
        return 0.0
    return 1.0


def _is_clean_atomic_answer(inner: str, answer_type: str) -> bool:
    """Whether `inner` is exactly an atomic answer with no extra content."""
    if answer_type == "mcq":
        return bool(re.match(r"^\([A-Da-d]\)$", inner) or re.match(r"^[A-Da-d]\.?$", inner))
    if answer_type == "yesno":
        return inner.lower().rstrip(".,;:") in ("yes", "no")
    if answer_type == "counting":
        return bool(re.match(r"^\d+$", inner))
    if answer_type == "numeric":
        num_text = inner.replace(",", "").rstrip("%")
        try:
            float(num_text.replace("/", "."))
            return True
        except ValueError:
            return False
    # Open-ended: any non-empty content counts as "clean"
    return bool(inner)


def _compute_answer_tag_only_format_reward(
    a2_text: str,
    answer_type: str,
) -> float:
    """Format reward for `<answer>`-only mode: tag + clean atomic inner.

    Binary {0, +1}. Same as `_compute_tag_format_reward` but only requires
    `<answer>` (no `<think>`). Inner content must still be a clean atomic
    answer for the expected type.

    Args:
        a2_text: Raw A2 text.
        answer_type: Expected answer type.

    Returns:
        +1.0 if `<answer>` present AND inner is a clean atomic answer.
         0.0 otherwise.
    """
    from vlm_grpo.trajectory import extract_from_answer_tags

    if not re.search(r"<answer>", a2_text, re.IGNORECASE):
        return 0.0
    inner = extract_from_answer_tags(a2_text).strip()
    if not inner:
        return 0.0
    if not _is_clean_atomic_answer(inner, answer_type):
        return 0.0
    return 1.0


def _compute_bare_format_reward(
    a2_text: str,
    answer_type: str,
    ground_truth: str = "",
) -> float:
    """Original format reward: normalize-first, penalty-only.

    Args:
        a2_text: Raw A2 text.
        answer_type: Expected answer type.
        ground_truth: Ground truth (for LLM fallback).

    Returns:
        0.0 if compliant, -1.0 if not.
    """
    normalized = normalize_answer(a2_text)

    if not normalized:
        return -1.0

    if answer_type == "mcq":
        if len(normalized) != 1 or normalized not in "abcdef":
            return _llm_format_fallback(a2_text, ground_truth, answer_type)
    elif answer_type == "yesno":
        if normalized not in ("yes", "no"):
            return _llm_format_fallback(a2_text, ground_truth, answer_type)
    elif answer_type == "counting":
        try:
            int(normalized)
        except ValueError:
            return -1.0
    elif answer_type == "numeric":
        num_text = normalized.replace(",", "")
        if "/" in num_text:
            parts = num_text.split("/")
            if len(parts) != 2:
                return -1.0
            try:
                float(parts[0])
                float(parts[1])
            except ValueError:
                return -1.0
        else:
            num_text = num_text.rstrip("%")
            try:
                float(num_text)
            except ValueError:
                return -1.0

    return 0.0


def refiner_format_reward_fn(
    completions: list,
    answer_type: list[str] | None = None,
    choices: list[str] | None = None,
    ground_truth: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """TRL-compatible format reward for refiner A2 completions.

    Args:
        completions: List of K completions from GRPOTrainer
        answer_type: List of answer types (from dataset columns)
        choices: List of MCQ choices (from dataset columns)
        ground_truth: List of ground truths (for LLM format fallback)
        **kwargs: Additional dataset columns (ignored)

    Returns:
        List of format rewards, one per completion
    """
    rewards = []
    for i, comp in enumerate(completions):
        at = answer_type[i] if answer_type else "open"
        gt = ground_truth[i] if ground_truth else ""
        a2_text = _extract_a2_text(comp)
        r = _compute_refiner_format_reward(a2_text, at, gt)
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

        a2_text = _extract_a2_text(comp)

        r = compute_a2_correctness_reward(
            a2_extracted=a2_text,
            ground_truth=gt,
            answer_type=at,
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
        a1_correct = a1_is_correct[i] if a1_is_correct else True

        a2_text = _extract_a2_text(comp)

        r = compute_no_regression_reward(
            a2_extracted=a2_text,
            ground_truth=gt,
            answer_type=at,
            a1_is_correct=a1_correct,
        )
        rewards.append(r)
    return rewards


def get_refiner_reward_functions() -> list:
    """Return ordered list of TRL-compatible reward functions for refiner.

    Order matches RefinerRewardWeights.to_list():
    [format, correctness, no_regression]

    Returns:
        List of reward function callables
    """
    return [
        refiner_format_reward_fn,
        refiner_correctness_reward_fn,
        refiner_no_regression_reward_fn,
    ]


# =============================================================================
# Full Self-Reflection Trajectory Rewards (two separate breakdowns)
# =============================================================================


@dataclass
class TrajectoryResponseRewardBreakdown:
    """Reward breakdown for response quality (drives A1+A2 GRPO update).

    Attributes:
        total_reward: Weighted sum of response components
        components: Dict mapping component name to raw reward value
        weighted_components: Dict mapping component name to weighted value
        a1_correct: Whether A1 is correct
        a2_correct: Whether A2 is correct
        a2_extracted: Normalized extracted A2 answer
        a2_format_valid: Whether A2 format is valid
    """

    total_reward: float
    components: dict[str, float]
    weighted_components: dict[str, float]
    a1_correct: bool
    a2_correct: bool
    a2_extracted: str
    a2_format_valid: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class TrajectoryFeedbackRewardBreakdown:
    """Reward breakdown for feedback quality (drives F1 GRPO update).

    Attributes:
        total_reward: Weighted sum of feedback components
        components: Dict mapping component name to raw reward value
        weighted_components: Dict mapping component name to weighted value
    """

    total_reward: float
    components: dict[str, float]
    weighted_components: dict[str, float]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# Single-turn A1 baseline reward (used when RolloutConfig.single_turn_a1=True)
# =============================================================================


def compute_a1_format_01(a1_text: str, answer_type: str) -> float:
    """R_a1_format_01: binary {0.0, 1.0} format check for the single-turn baseline.

    1.0 only when BOTH conditions hold:
      1. The output contains both ``<think>...</think>`` and ``<answer>...</answer>``
         (any order, any inner content).
      2. The inner content of ``<answer>`` is a clean atomic answer for the
         expected type — same definition as ``_is_clean_atomic_answer`` (MCQ
         letter, integer, yes/no, numeric, or any non-empty string for open).

    Uses the same atomic-answer notion as the multi-turn ``_compute_tag_format_reward``
    so the baseline's format signal is consistent with the eval pipeline (which
    extracts via ``<answer>`` tags).

    Args:
        a1_text: Raw A1 output text.
        answer_type: Expected answer type ("mcq", "yesno", "numeric",
            "counting", "open").

    Returns:
        1.0 if both tags present AND inner is a clean atomic answer.
        0.0 otherwise.
    """
    from vlm_grpo.trajectory import extract_from_answer_tags, has_think_answer_tags

    if not has_think_answer_tags(a1_text):
        return 0.0
    inner = extract_from_answer_tags(a1_text).strip()
    if not inner:
        return 0.0
    if not _is_clean_atomic_answer(inner, answer_type):
        return 0.0
    return 1.0


@dataclass
class BaselineA1RewardBreakdown:
    """Reward breakdown for single-turn A1 baseline GRPO.

    Reward = ``w_a1_correctness * R_a1_correct_01 + w_a1_format * R_a1_format_01``.

    Both components are in [0, 1] and weights default to 0.9 / 0.1, so
    ``total_reward`` lives in [0, 1].

    Attributes:
        total_reward: Weighted sum of the two components.
        components: Raw component values (each in {0.0, 1.0}).
        weighted_components: Component values multiplied by their weights.
        a1_correct: Whether A1 matched the ground truth.
        a1_format_valid: Whether A1 passed the strict tag-format check.
        a1_extracted: Strict atomic-answer extraction from ``<answer>`` tags
            (empty string when format invalid).
    """

    total_reward: float
    components: dict[str, float]
    weighted_components: dict[str, float]
    a1_correct: bool
    a1_format_valid: bool
    a1_extracted: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def compute_baseline_a1_reward_breakdown(
    a1_text: str,
    ground_truth: str,
    answer_type: str,
    choices: str,
    weights: Any,
) -> BaselineA1RewardBreakdown:
    """Compute the 2-component A1-only reward used by the single-turn baseline.

    This is the lone reward path when ``RolloutConfig.single_turn_a1=True``;
    no F1 / A2 / no_regression / shaped-improvement components participate.

    Args:
        a1_text: Raw A1 output text.
        ground_truth: Ground truth answer.
        answer_type: Answer type ("mcq", "yesno", "numeric", "counting", "open").
        choices: MCQ choices string (kept for API parity, unused here).
        weights: ``BaselineA1RewardWeights`` instance.

    Returns:
        BaselineA1RewardBreakdown with both components, weighted sum, and
        derived flags (a1_correct, a1_format_valid, a1_extracted).
    """
    r_correct = compute_a1_correctness_01(a1_text, ground_truth, answer_type)
    r_format = compute_a1_format_01(a1_text, answer_type)
    a1_extracted = extract_answer_from_text(a1_text, answer_type, choices, strict=True)
    a1_correct = r_correct == 1.0
    a1_format_valid = r_format == 1.0

    components = {
        "a1_correctness": r_correct,
        "a1_format": r_format,
    }
    weighted_components = {
        "a1_correctness": r_correct * weights.w_a1_correctness,
        "a1_format": r_format * weights.w_a1_format,
    }
    total_reward = sum(weighted_components.values())

    return BaselineA1RewardBreakdown(
        total_reward=total_reward,
        components=components,
        weighted_components=weighted_components,
        a1_correct=a1_correct,
        a1_format_valid=a1_format_valid,
        a1_extracted=a1_extracted,
    )


def compute_response_reward_breakdown(
    a1_text: str,
    a2_text: str,
    ground_truth: str,
    answer_type: str,
    choices: str,
    weights: Any,
    use_think_answer_tags: bool = False,
    use_answer_tag_only: bool = False,
    reward_shaping_alpha: float = 0.0,
) -> TrajectoryResponseRewardBreakdown:
    """Compute response reward for a single trajectory.

    Evaluates answer quality: A1 correctness, A2 correctness,
    no-regression, and format.

    Args:
        a1_text: Initial answer text
        a2_text: Refined answer text
        ground_truth: Ground truth answer
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        choices: MCQ choices string
        weights: ResponseRewardWeights instance
        use_think_answer_tags: Whether to check for tag format

    Returns:
        TrajectoryResponseRewardBreakdown with all component scores
    """
    # In tag mode, strict extraction requires <answer> tag presence.
    # Missing tag → extracted="" → MatchResult.is_correct=False naturally,
    # so the correctness path handles "no extractable answer" without any
    # special short-circuit. Format reward is independent and binary.
    tag_mode = use_think_answer_tags or use_answer_tag_only

    a1_result = verify_answer(a1_text, ground_truth, answer_type, strict=tag_mode)
    a1_correct = a1_result.is_correct

    a2_result = verify_answer(a2_text, ground_truth, answer_type, strict=tag_mode)
    a2_correct = a2_result.is_correct

    # Format rewards — pure structural check (binary {0, +1}) per turn.
    # Both A1 and A2 are evaluated independently for tag presence + clean
    # atomic inner content. Symmetry across turns prevents the model from
    # learning "format only matters for A2".
    r_a1_format = _compute_refiner_format_reward(
        a1_text, answer_type, ground_truth, use_think_answer_tags, use_answer_tag_only
    )
    r_a2_format = _compute_refiner_format_reward(
        a2_text, answer_type, ground_truth, use_think_answer_tags, use_answer_tag_only
    )
    a2_format_valid = r_a2_format > 0

    # Extract A2 for display (uses same strictness/tag-requirement as scoring)
    a2_extracted = extract_answer_from_text(a2_text, answer_type, choices, strict=tag_mode)

    # A1 / A2 correctness rewards
    r_a1 = 1.0 if a1_correct else -1.0
    if answer_type in ("counting", "open") and a2_result.score is not None:
        r_a2 = 2.0 * a2_result.score - 1.0
    else:
        r_a2 = 1.0 if a2_correct else -1.0

    # No-regression / improvement reward for A2.
    # When reward_shaping_alpha > 0: use shaped improvement term α*(R(A2)-R(A1)).
    # This replaces transition-based no_regression with a continuous signal
    # that reduces RR stabilization (less conservative A2) while amplifying
    # WR/RW gradient. Use w_no_regression=1.0 when alpha > 0.
    # When alpha = 0: fall back to transition-based no_regression.
    from vlm_grpo.rewards.verifier import DETERMINISTIC_TYPES

    if reward_shaping_alpha > 0:
        r_no_reg = reward_shaping_alpha * (r_a2 - r_a1)
    elif answer_type in DETERMINISTIC_TYPES:
        if a1_correct:
            r_no_reg = 1.0 if a2_correct else -2.0
        else:
            r_no_reg = 3.0 if a2_correct else 0.0
    else:
        if a1_correct:
            r_no_reg = 1.0 if a2_correct else -3.0
        else:
            r_no_reg = 2.0 if a2_correct else 0.0

    components = {
        "a1_correctness": r_a1,
        "a1_format": r_a1_format,
        "a2_correctness": r_a2,
        "a2_format": r_a2_format,
        "no_regression": r_no_reg,
    }
    weighted_components = {
        "a1_correctness": r_a1 * weights.w_a1_correctness,
        "a1_format": r_a1_format * weights.w_a1_format,
        "a2_correctness": r_a2 * weights.w_a2_correctness,
        "a2_format": r_a2_format * weights.w_a2_format,
        "no_regression": r_no_reg * weights.w_no_regression,
    }
    total_reward = sum(weighted_components.values())

    return TrajectoryResponseRewardBreakdown(
        total_reward=total_reward,
        components=components,
        weighted_components=weighted_components,
        a1_correct=a1_correct,
        a2_correct=a2_correct,
        a2_extracted=a2_extracted,
        a2_format_valid=a2_format_valid,
    )


def compute_feedback_reward_breakdown(
    feedback_text: str,
    a1_text: str,
    a2_text: str,
    ground_truth: str,
    answer_type: str,
    choices: str,
    weights: Any,
    use_improvement_reward: bool = False,
    reward_shaping_alpha: float = 0.0,
) -> TrajectoryFeedbackRewardBreakdown:
    """Compute feedback reward for a single trajectory.

    F1 outputs a CORRECT/INCORRECT verdict with brief explanation. Reward
    components:
    - downstream: `r_a2 + α·(r_a2 − r_a1)` — judge F1 by the A2 it produces
    - verification: ±1 based on whether F1's verdict matches A1's truth

    Args:
        feedback_text: F1 output text (should contain CORRECT or INCORRECT)
        a1_text: Initial answer text
        a2_text: Refined answer text
        ground_truth: Ground truth answer
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        choices: MCQ choices string (unused, kept for API compat)
        weights: FeedbackRewardWeights instance
        use_improvement_reward: If True, use R(A2)-R(A1) instead of shaped downstream
        reward_shaping_alpha: SCoRe-style shaped reward alpha

    Returns:
        TrajectoryFeedbackRewardBreakdown with all component scores
    """
    # Verify A1 and A2 ONCE (downstream needs both)
    a1_result = verify_answer(a1_text, ground_truth, answer_type)
    a2_result = verify_answer(a2_text, ground_truth, answer_type)
    a1_correct = a1_result.is_correct
    a2_correct = a2_result.is_correct

    # No short-circuit on missing F1 format tags. Missing `\boxed{}` is
    # handled naturally:
    #   - verification_reward returns -1 (treat as wrong verdict)
    #   - downstream is gated by `r_verification > 0`, so it becomes 0
    #   - format_reward returns 0 (no bonus for missing structure)
    # The natural extraction path produces the same "no credit" outcome
    # that the short-circuit used to enforce, without the large -2.0
    # sentinel that created an inverted gradient vs tagged-wrong F1s.

    from vlm_grpo.rewards.verifier import DETERMINISTIC_TYPES

    if not feedback_text.strip():
        r_downstream = 0.0
    elif reward_shaping_alpha > 0:
        # SCoRe-style shaped reward: R(A2) + α × (R(A2) - R(A1))
        # With α=5: WR=+11, RW=-11, RR=+1, WW=-1
        # Adds RR stabilization signal (+1) missing from pure improvement,
        # and eliminates ~33% dead K-groups (RR≠WW, always has variance).
        r_a1 = 1.0 if a1_correct else -1.0
        r_a2 = 1.0 if a2_correct else -1.0
        r_downstream = r_a2 + reward_shaping_alpha * (r_a2 - r_a1)
    elif use_improvement_reward:
        # Improvement-based: R(A2) - R(A1) ∈ {-2, 0, +2}
        # RR=0, WW=0, WR=+2, RW=-2. Group mean → 0, WR/RW dominate advantage.
        r_a1 = 1.0 if a1_correct else -1.0
        r_a2 = 1.0 if a2_correct else -1.0
        r_downstream = r_a2 - r_a1
    elif answer_type in DETERMINISTIC_TYPES:
        if a1_correct:
            r_downstream = 1.0 if a2_correct else -1.5
        else:
            r_downstream = 3.0 if a2_correct else -1.0
    else:
        if a1_correct:
            r_downstream = 1.0 if a2_correct else -2.0
        else:
            r_downstream = 2.0 if a2_correct else -1.0

    r_verification = compute_verification_accuracy_reward(feedback_text, a1_correct)
    r_fb_format = compute_feedback_format_reward(feedback_text)

    # Asymmetric gate on downstream credit.
    #
    # When F1's verdict is calibrated (r_verification > 0): full bidirectional
    # downstream signal flows — WR rewards +11, RW penalises -11, etc.
    #
    # When F1's verdict is wrong (r_verification <= 0): clamp to NEGATIVE
    # downstream only. Positive downstream is gated to 0 to prevent
    # sycophancy farming (e.g., \boxed{CORRECT} on a wrong A1 can't earn
    # the +11 WR bonus when A2 happens to variance-flip right), but NEGATIVE
    # downstream still flows through so F1 is penalised proportionally when
    # its bad verdict caused actual harm:
    #   • RW + \boxed{INCORRECT} on a right A1 (F1 pushed A2 off correct
    #     answer, A2 regressed):  downstream = -11, total ≈ -5.3
    #   • WW + sycophantic \boxed{CORRECT} on wrong A1 (F1 reinforced wrong,
    #     A2 stayed wrong):       downstream = -1,  total ≈ -0.8
    #   • RR + wrong \boxed{INCORRECT} (A2 ignored bad advice, stayed
    #     right): downstream = +1 gated to 0, total = -0.35  (ineffectual,
    #     small penalty — no harm done)
    #
    # The asymmetry discriminates "F1 actively caused harm" from "F1
    # was ineffectual"; the symmetric gate collapsed both to the same
    # -0.35 signal.
    if r_verification > 0:
        r_downstream_gated = r_downstream
    else:
        r_downstream_gated = min(r_downstream, 0.0)

    components = {
        "downstream": r_downstream_gated,
        "verification": r_verification,
        "format": r_fb_format,
    }
    weighted_components = {
        "downstream": r_downstream_gated * weights.w_downstream,
        "verification": r_verification * weights.w_verification_accuracy,
        "format": r_fb_format * weights.w_format,
    }
    total_reward = sum(weighted_components.values())

    return TrajectoryFeedbackRewardBreakdown(
        total_reward=total_reward,
        components=components,
        weighted_components=weighted_components,
    )


# =============================================================================
# Per-component [0, 1] rescaled rewards (used when
# RolloutConfig.use_rescaled_rewards=True)
# =============================================================================


def _to_unit(x: float, lo: float, hi: float) -> float:
    """Map ``x`` from ``[lo, hi]`` into ``[0, 1]`` via ``(x - lo) / (hi - lo)``.

    Out-of-range inputs are clipped to ``[0, 1]``. Used by the rescaled
    reward path to renormalize each multi-turn component to a unit range so
    per-unit-weight gradient magnitude is equalized across components.

    Args:
        x: Raw value to rescale.
        lo: Lower bound of the raw range (maps to 0).
        hi: Upper bound of the raw range (maps to 1).

    Returns:
        Float in ``[0, 1]``.
    """
    if hi <= lo:
        return 0.0
    val = (x - lo) / (hi - lo)
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


# Raw component ranges as implemented in ``compute_response_reward_breakdown``
# and ``compute_feedback_reward_breakdown``.
#
# These are inline values inside the existing breakdowns (NOT the values from
# the standalone ``stability.compute_no_regression_reward`` /
# ``feedback.compute_downstream_aware_reward`` functions, which differ).
#
# IMPORTANT: keep these in sync with the inline values if either function is
# edited. See the per-component rationale in the docstrings below.
_A1_CORR_RANGE = (-1.0, 1.0)
_A2_CORR_RANGE = (-1.0, 1.0)
# no_regression: deterministic det:{-2,+3}; open:{-3,+2}. Combined min/max
# across both branches gives the safest unified rescale.
_NO_REG_DET_RANGE = (-2.0, 3.0)
_NO_REG_OPEN_RANGE = (-3.0, 2.0)
# downstream (default, transition mode): deterministic det:{-1.5,+3};
# open:{-2,+2}. After asymmetric gating (clamp to <=0 when verification
# fails), the negative arm doesn't change, so the raw range bounds are still
# the right normalizer.
_DOWNSTREAM_DET_RANGE = (-1.5, 3.0)
_DOWNSTREAM_OPEN_RANGE = (-2.0, 2.0)
# verification: ±1
_VERIFY_RANGE = (-1.0, 1.0)
# format rewards (a1, a2, fb) are already binary {0, +1} in the active
# multi-turn path (see ``_compute_tag_format_reward``,
# ``_compute_answer_tag_only_format_reward``, ``compute_feedback_format_reward``)
# so they need no rescaling. The bare-mode {-1, 0} branch is rescaled to
# {0, 1} via ``_FORMAT_BARE_RANGE`` only when called.
_FORMAT_BARE_RANGE = (-1.0, 0.0)


def _format_to_01(raw: float) -> float:
    """Rescale a format reward to [0, 1].

    The active tag-mode and answer-tag-only paths already return {0, +1}, so
    this is a no-op for them. The bare path returns {-1, 0}, which gets
    rescaled to {0, 1}.

    Args:
        raw: Raw format reward.

    Returns:
        Format reward in ``[0, 1]``.
    """
    if raw < 0.0:
        return _to_unit(raw, _FORMAT_BARE_RANGE[0], _FORMAT_BARE_RANGE[1])
    # Already binary {0, +1} from tag-mode paths.
    if raw > 1.0:
        return 1.0
    return raw


def compute_a2_correctness_01(
    a2_extracted: str,
    ground_truth: str,
    answer_type: str,
    tolerance: float = 0.01,
) -> float:
    """[0, 1]-rescaled A2 correctness reward.

    For deterministic types (MCQ / yesno / numeric): binary {0, 1}.
    For counting / open: continuous [0, 1] via raw_score (already in [0, 1]
    inside ``compute_a2_correctness_reward`` before the ``2*score - 1``
    expansion — we recover it via ``(raw + 1) / 2`` to avoid duplicating the
    cascade).

    Args:
        a2_extracted: Normalized extracted A2 answer (or raw text accepted by
            ``compute_a2_correctness_reward``).
        ground_truth: Ground truth answer.
        answer_type: Answer type ("mcq", "yesno", "numeric", "open",
            "counting").
        tolerance: Numeric tolerance for comparison.

    Returns:
        Float in ``[0, 1]``.
    """
    raw = compute_a2_correctness_reward(
        a2_extracted=a2_extracted,
        ground_truth=ground_truth,
        answer_type=answer_type,
        tolerance=tolerance,
    )
    return _to_unit(raw, _A2_CORR_RANGE[0], _A2_CORR_RANGE[1])


def compute_no_regression_01(
    a2_extracted: str,
    ground_truth: str,
    answer_type: str,
    a1_is_correct: bool,
    tolerance: float = 0.01,
) -> float:
    """[0, 1]-rescaled no-regression reward (matches the inline values in
    ``compute_response_reward_breakdown``).

    Inline raw values (NOT the stability.compute_no_regression_reward values):
        Deterministic: RR=+1, RW=-2, WR=+3, WW=0  (range [-2, +3])
        Open-ended:    RR=+1, RW=-3, WR=+2, WW=0  (range [-3, +2])

    Args:
        a2_extracted: Normalized extracted A2 answer.
        ground_truth: Ground truth answer.
        answer_type: Answer type ("mcq", "yesno", "numeric", "open",
            "counting").
        a1_is_correct: Whether A1 was correct.
        tolerance: Numeric tolerance for comparison.

    Returns:
        Float in ``[0, 1]``.
    """
    from vlm_grpo.rewards.verifier import DETERMINISTIC_TYPES

    a2_result = verify_answer(a2_extracted, ground_truth, answer_type, tolerance=tolerance)
    a2_correct = a2_result.is_correct

    if answer_type in DETERMINISTIC_TYPES:
        if a1_is_correct:
            raw = 1.0 if a2_correct else -2.0
        else:
            raw = 3.0 if a2_correct else 0.0
        return _to_unit(raw, _NO_REG_DET_RANGE[0], _NO_REG_DET_RANGE[1])
    if a1_is_correct:
        raw = 1.0 if a2_correct else -3.0
    else:
        raw = 2.0 if a2_correct else 0.0
    return _to_unit(raw, _NO_REG_OPEN_RANGE[0], _NO_REG_OPEN_RANGE[1])


def compute_a2_format_01(
    a2_text: str,
    answer_type: str,
    ground_truth: str = "",
    use_think_answer_tags: bool = False,
    use_answer_tag_only: bool = False,
) -> float:
    """[0, 1]-rescaled A2 format reward.

    Reuses ``_compute_refiner_format_reward`` which returns {0, +1} in tag
    modes (already in [0, 1]) and {-1, 0} in bare mode (needs rescaling).

    Args:
        a2_text: Raw A2 text from the refiner.
        answer_type: Expected answer type.
        ground_truth: Ground truth (for LLM format fallback).
        use_think_answer_tags: Whether to check for <think>+<answer> tags.
        use_answer_tag_only: Whether to check for <answer> tag only.

    Returns:
        Float in ``[0, 1]``.
    """
    raw = _compute_refiner_format_reward(
        a2_text, answer_type, ground_truth, use_think_answer_tags, use_answer_tag_only
    )
    return _format_to_01(raw)


def compute_downstream_01(
    feedback_text: str,
    a1_text: str,
    a2_text: str,
    ground_truth: str,
    answer_type: str,
    use_improvement_reward: bool = False,
    reward_shaping_alpha: float = 0.0,
    tolerance: float = 0.01,
) -> float:
    """[0, 1]-rescaled downstream-aware reward (transition mode only).

    Mirrors the inline computation in ``compute_feedback_reward_breakdown``:
        Deterministic: RR=+1, RW=-1.5, WR=+3, WW=-1  (range [-1.5, +3])
        Open-ended:    RR=+1, RW=-2,   WR=+2, WW=-1  (range [-2, +2])

    Empty F1 → 0.0 raw → rescaled to the lower-bound mid-value (treated as
    "no contribution"). Then the same asymmetric gate (``min(raw, 0)`` when
    verification fails) is applied BEFORE rescaling so the [0, 1] mapping
    captures the gated value.

    Improvement-mode and shaped-alpha modes are NOT supported by this
    rescaler (their raw range varies with α). When either is set, falls
    back to scaling the raw value with the open-ended bounds for safety
    (the user's frozen-a1-mt-r01 run uses neither).

    Args:
        feedback_text: F1 text (used for the empty-check short-circuit).
        a1_text: Initial answer text.
        a2_text: Refined answer text.
        ground_truth: Ground truth answer.
        answer_type: Answer type.
        use_improvement_reward: If True, returns ``(raw + 2) / 4`` to put
            ``r_a2 - r_a1`` ∈ {-2, 0, +2} into [0, 1].
        reward_shaping_alpha: SCoRe-style alpha. If > 0, scales raw with the
            theoretical envelope ``[-1 - 2α, 1 + 2α]``.
        tolerance: Numeric tolerance for comparison.

    Returns:
        Float in ``[0, 1]``.
    """
    from vlm_grpo.rewards.verifier import DETERMINISTIC_TYPES

    if not feedback_text.strip():
        # Empty F1 raw=0; map to the mid of the deterministic range so the
        # K-group baseline still varies meaningfully.
        return _to_unit(0.0, _DOWNSTREAM_DET_RANGE[0], _DOWNSTREAM_DET_RANGE[1])

    a1_result = verify_answer(a1_text, ground_truth, answer_type, tolerance=tolerance)
    a2_result = verify_answer(a2_text, ground_truth, answer_type, tolerance=tolerance)
    a1_correct = a1_result.is_correct
    a2_correct = a2_result.is_correct

    if reward_shaping_alpha > 0:
        # r_a2 + α(r_a2 - r_a1), with r_* in {-1, +1}
        # range: [-1 - 2α, 1 + 2α]
        r_a1 = 1.0 if a1_correct else -1.0
        r_a2 = 1.0 if a2_correct else -1.0
        raw = r_a2 + reward_shaping_alpha * (r_a2 - r_a1)
        lo = -1.0 - 2.0 * reward_shaping_alpha
        hi = 1.0 + 2.0 * reward_shaping_alpha
        return _to_unit(raw, lo, hi)

    if use_improvement_reward:
        r_a1 = 1.0 if a1_correct else -1.0
        r_a2 = 1.0 if a2_correct else -1.0
        raw = r_a2 - r_a1
        return _to_unit(raw, -2.0, 2.0)

    if answer_type in DETERMINISTIC_TYPES:
        if a1_correct:
            raw = 1.0 if a2_correct else -1.5
        else:
            raw = 3.0 if a2_correct else -1.0
        return _to_unit(raw, _DOWNSTREAM_DET_RANGE[0], _DOWNSTREAM_DET_RANGE[1])
    if a1_correct:
        raw = 1.0 if a2_correct else -2.0
    else:
        raw = 2.0 if a2_correct else -1.0
    return _to_unit(raw, _DOWNSTREAM_OPEN_RANGE[0], _DOWNSTREAM_OPEN_RANGE[1])


def compute_verification_01(feedback_text: str, a1_is_correct: bool) -> float:
    """[0, 1]-rescaled verification (calibration) reward.

    Wraps ``compute_verification_accuracy_reward`` which returns ±1.

    Args:
        feedback_text: F1 text.
        a1_is_correct: Whether A1 was actually correct.

    Returns:
        Float in ``[0, 1]``.
    """
    raw = compute_verification_accuracy_reward(feedback_text, a1_is_correct)
    return _to_unit(raw, _VERIFY_RANGE[0], _VERIFY_RANGE[1])


def compute_fb_format_01(feedback_text: str) -> float:
    """[0, 1] feedback-format reward.

    The active path (``compute_feedback_format_reward``) already returns
    {0, +1} so this is effectively a pass-through, included for symmetry
    with the rest of the ``_01`` family.

    Args:
        feedback_text: Raw F1 output.

    Returns:
        Float in ``[0, 1]``.
    """
    raw = compute_feedback_format_reward(feedback_text)
    if raw < 0.0:
        return 0.0
    if raw > 1.0:
        return 1.0
    return raw


def compute_response_reward_breakdown_01(
    a1_text: str,
    a2_text: str,
    ground_truth: str,
    answer_type: str,
    choices: str,
    weights: Any,
    use_think_answer_tags: bool = False,
    use_answer_tag_only: bool = False,
    reward_shaping_alpha: float = 0.0,
) -> TrajectoryResponseRewardBreakdown:
    """[0, 1]-rescaled response reward (parallel to
    ``compute_response_reward_breakdown``).

    Each component is in [0, 1]; with ResponseRewardWeights summing to 1.0,
    ``total_reward`` lands in [0, 1] (not [0, sum_of_weights] strictly,
    because the inline ``no_regression`` raw values don't all reach the
    upper bound from a given a1_correct branch — see
    ``_NO_REG_DET_RANGE``).

    Args:
        a1_text: Initial answer text.
        a2_text: Refined answer text.
        ground_truth: Ground truth answer.
        answer_type: Answer type ("mcq", "yesno", "numeric", "open",
            "counting").
        choices: MCQ choices string.
        weights: ResponseRewardWeights instance.
        use_think_answer_tags: Whether to check for tag format.
        use_answer_tag_only: Whether to check for <answer>-only format.
        reward_shaping_alpha: SCoRe-style shaped reward alpha (passed
            through but currently does NOT replace the inline transition
            values — the rescaled path uses the discrete RR/RW/WR/WW
            values, matching the ``alpha=0`` branch of the raw breakdown).

    Returns:
        ``TrajectoryResponseRewardBreakdown`` with each ``components[k]`` in
        [0, 1] and ``weighted_components[k] = weight_k × component_k``.
    """
    from vlm_grpo.rewards.verifier import DETERMINISTIC_TYPES

    tag_mode = use_think_answer_tags or use_answer_tag_only
    a1_result = verify_answer(a1_text, ground_truth, answer_type, strict=tag_mode)
    a1_correct = a1_result.is_correct
    a2_result = verify_answer(a2_text, ground_truth, answer_type, strict=tag_mode)
    a2_correct = a2_result.is_correct

    r_a1_format = compute_a2_format_01(
        a1_text, answer_type, ground_truth, use_think_answer_tags, use_answer_tag_only
    )
    r_a2_format = compute_a2_format_01(
        a2_text, answer_type, ground_truth, use_think_answer_tags, use_answer_tag_only
    )
    a2_format_valid = r_a2_format > 0.5

    a2_extracted = extract_answer_from_text(a2_text, answer_type, choices, strict=tag_mode)

    # A1 / A2 correctness (rescaled to [0, 1])
    r_a1 = 1.0 if a1_correct else 0.0
    if answer_type in ("counting", "open") and a2_result.score is not None:
        r_a2 = a2_result.score  # already in [0, 1]
    else:
        r_a2 = 1.0 if a2_correct else 0.0

    # No-regression (rescaled). When alpha > 0, the inline raw path uses
    # ``α * (r_a2_raw - r_a1_raw)`` with r_*_raw in {-1, +1}; we mirror
    # that and rescale via the symmetric envelope.
    if reward_shaping_alpha > 0:
        r_a1_raw = 1.0 if a1_correct else -1.0
        r_a2_raw = 1.0 if a2_correct else -1.0
        raw_no_reg = reward_shaping_alpha * (r_a2_raw - r_a1_raw)
        # range: [-2α, +2α]
        r_no_reg = _to_unit(raw_no_reg, -2.0 * reward_shaping_alpha, 2.0 * reward_shaping_alpha)
    elif answer_type in DETERMINISTIC_TYPES:
        if a1_correct:
            raw_no_reg = 1.0 if a2_correct else -2.0
        else:
            raw_no_reg = 3.0 if a2_correct else 0.0
        r_no_reg = _to_unit(raw_no_reg, _NO_REG_DET_RANGE[0], _NO_REG_DET_RANGE[1])
    else:
        if a1_correct:
            raw_no_reg = 1.0 if a2_correct else -3.0
        else:
            raw_no_reg = 2.0 if a2_correct else 0.0
        r_no_reg = _to_unit(raw_no_reg, _NO_REG_OPEN_RANGE[0], _NO_REG_OPEN_RANGE[1])

    components = {
        "a1_correctness": r_a1,
        "a1_format": r_a1_format,
        "a2_correctness": r_a2,
        "a2_format": r_a2_format,
        "no_regression": r_no_reg,
    }
    weighted_components = {
        "a1_correctness": r_a1 * weights.w_a1_correctness,
        "a1_format": r_a1_format * weights.w_a1_format,
        "a2_correctness": r_a2 * weights.w_a2_correctness,
        "a2_format": r_a2_format * weights.w_a2_format,
        "no_regression": r_no_reg * weights.w_no_regression,
    }
    total_reward = sum(weighted_components.values())

    return TrajectoryResponseRewardBreakdown(
        total_reward=total_reward,
        components=components,
        weighted_components=weighted_components,
        a1_correct=a1_correct,
        a2_correct=a2_correct,
        a2_extracted=a2_extracted,
        a2_format_valid=a2_format_valid,
    )


def compute_feedback_reward_breakdown_01(
    feedback_text: str,
    a1_text: str,
    a2_text: str,
    ground_truth: str,
    answer_type: str,
    choices: str,
    weights: Any,
    use_improvement_reward: bool = False,
    reward_shaping_alpha: float = 0.0,
) -> TrajectoryFeedbackRewardBreakdown:
    """[0, 1]-rescaled feedback reward (parallel to
    ``compute_feedback_reward_breakdown``).

    Each component is in [0, 1]; with FeedbackRewardWeights summing to 1.0,
    ``total_reward`` lands in [0, 1].

    The asymmetric gate from the raw breakdown (clamp downstream to <=0
    when verification fails) becomes a clamp to ``<= mid_zero`` after
    rescaling — i.e., when verification fails, downstream is clamped to
    ``min(value, _to_unit(0.0, lo, hi))``. This preserves the "F1 was
    ineffectual" vs "F1 actively caused harm" discrimination.

    Args:
        feedback_text: F1 output text.
        a1_text: Initial answer text.
        a2_text: Refined answer text.
        ground_truth: Ground truth answer.
        answer_type: Answer type.
        choices: MCQ choices string (unused, kept for API compat).
        weights: FeedbackRewardWeights instance.
        use_improvement_reward: If True, use R(A2)-R(A1)-style downstream.
        reward_shaping_alpha: SCoRe-style shaped reward alpha.

    Returns:
        ``TrajectoryFeedbackRewardBreakdown`` with each ``components[k]`` in
        [0, 1] and ``weighted_components[k] = weight_k × component_k``.
    """
    a1_result = verify_answer(a1_text, ground_truth, answer_type)
    a1_correct = a1_result.is_correct

    r_verification = compute_verification_01(feedback_text, a1_correct)
    r_fb_format = compute_fb_format_01(feedback_text)

    raw_downstream_01 = compute_downstream_01(
        feedback_text=feedback_text,
        a1_text=a1_text,
        a2_text=a2_text,
        ground_truth=ground_truth,
        answer_type=answer_type,
        use_improvement_reward=use_improvement_reward,
        reward_shaping_alpha=reward_shaping_alpha,
    )

    # Asymmetric gate: when verification rescaled <= 0.5 (i.e. raw <= 0,
    # meaning F1's verdict was wrong), clamp positive downstream to the
    # rescaled "raw=0" midpoint to suppress sycophancy farming. Negative
    # downstream (rescaled value below the midpoint) still flows through.
    if answer_type in (
        "mcq",
        "yesno",
        "numeric",
    ):
        downstream_zero = _to_unit(0.0, _DOWNSTREAM_DET_RANGE[0], _DOWNSTREAM_DET_RANGE[1])
    else:
        downstream_zero = _to_unit(0.0, _DOWNSTREAM_OPEN_RANGE[0], _DOWNSTREAM_OPEN_RANGE[1])
    if r_verification <= 0.5:
        r_downstream_gated = min(raw_downstream_01, downstream_zero)
    else:
        r_downstream_gated = raw_downstream_01

    components = {
        "downstream": r_downstream_gated,
        "verification": r_verification,
        "format": r_fb_format,
    }
    weighted_components = {
        "downstream": r_downstream_gated * weights.w_downstream,
        "verification": r_verification * weights.w_verification_accuracy,
        "format": r_fb_format * weights.w_format,
    }
    total_reward = sum(weighted_components.values())

    return TrajectoryFeedbackRewardBreakdown(
        total_reward=total_reward,
        components=components,
        weighted_components=weighted_components,
    )
