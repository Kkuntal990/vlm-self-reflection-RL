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
from vlm_grpo.rewards.feedback import compute_downstream_aware_reward
from vlm_grpo.rewards.stability import (
    compute_minimal_edit_reward,
    compute_no_regression_reward,
)
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
           + w_format * R_format

    Attributes:
        w_downstream: Weight for downstream-aware reward (dominant)
        w_format: Weight for format compliance
    """

    w_downstream: float = 2.0
    w_format: float = 0.15

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_list(self) -> list[float]:
        """Return weights as ordered list: [format, downstream]."""
        return [self.w_format, self.w_downstream]


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
    w_format: float = 0.15

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

# CJK character detection for proper word counting in Chinese/Japanese/Korean
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")


def _contains_cjk(text: str) -> bool:
    """Check if text contains CJK characters.

    Args:
        text: Input text string

    Returns:
        True if any CJK characters are found
    """
    return bool(_CJK_CHAR_RE.search(text))


def _count_content_units(text: str) -> int:
    """Count content units: CJK characters + whitespace-delimited non-CJK tokens.

    For pure English text, behaves like len(text.split()).
    For CJK text, each CJK character counts as one unit (≈ one word).
    For mixed text, combines both counts.

    Args:
        text: Input text string

    Returns:
        Content unit count
    """
    if not _contains_cjk(text):
        return len(text.split())

    count = 0
    non_cjk_buf: list[str] = []
    for ch in text:
        if _CJK_CHAR_RE.match(ch):
            if non_cjk_buf:
                token = "".join(non_cjk_buf).strip()
                if token:
                    count += len(token.split())
                non_cjk_buf = []
            count += 1
        else:
            non_cjk_buf.append(ch)

    if non_cjk_buf:
        token = "".join(non_cjk_buf).strip()
        if token:
            count += len(token.split())

    return count


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
] + [
    # Chinese stance keywords (no word boundaries needed for CJK)
    re.compile(p)
    for p in [
        r"正确",
        r"错误",
        r"不正确",
        r"准确",
        r"应该",
        r"修改",
    ]
]


def compute_critic_format_reward(feedback_text: str) -> float:
    """R_format for critic: pure structure check (no keyword overlap with calibration).

    Penalty-only, based solely on word count. Stance keyword detection is
    handled by calibration reward — keeping them separate avoids redundant
    signals that reduce K-group variance.

    Three-tier scoring:
        Empty or <3 units  → -2.0  (heavy penalty for safe empty)
        3-6 units          → -1.0  (too terse for useful feedback)
        >6 units           →  0.0  (acceptable structure)

    For CJK text, each character counts as one unit (≈ one word).

    Args:
        feedback_text: Feedback text from the critic

    Returns:
        Format reward in {-2.0, -1.0, 0.0}
    """
    stripped = feedback_text.strip()
    if not stripped:
        return -2.0

    word_count = _count_content_units(stripped)
    if word_count < 3:
        return -2.0

    if word_count <= 6:
        return -1.0

    return 0.0


# Pattern for detecting think/answer tag leakage in F1 outputs.
# The critic prompt has no tag instructions — tags here indicate
# behavioral contamination from A1/A2 tag training.
_F1_TAG_LEAKAGE_PATTERN = re.compile(r"</?(?:think|answer)>", re.IGNORECASE)


def compute_f1_tag_penalty(feedback_text: str) -> float:
    """Penalty for F1 using think/answer tags (role contamination).

    The critic system prompt does not instruct tag usage. When F1 generates
    <think>/<answer> tags, it's behavioral leakage from A1/A2 training that
    wastes tokens on internal reasoning instead of direct feedback, producing
    worse downstream outcomes (27.7% R→R vs 43.5% for plain F1).

    Args:
        feedback_text: Feedback text from the critic

    Returns:
        -2.0 if any think/answer tags found, 0.0 otherwise
    """
    if _F1_TAG_LEAKAGE_PATTERN.search(feedback_text):
        return -2.0
    return 0.0


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

    Combines: downstream-aware and format rewards.

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

    # Format reward (penalty-only: 0.0 = valid, negative = invalid)
    r_format = compute_critic_format_reward(feedback_text)
    format_valid = r_format >= 0

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
        "format": r_format,
        "downstream": r_downstream,
    }
    weighted_components = {
        "format": r_format * weights.w_format,
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
    use_think_answer_tags: bool = False,
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
        use_think_answer_tags: Whether to check for tag format

    Returns:
        RefinerRewardBreakdown with all component scores
    """
    # Format reward
    r_format = _compute_refiner_format_reward(
        a2_text, answer_type, ground_truth, use_think_answer_tags
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

    # Minimal-edit reward
    r_minimal_edit = compute_minimal_edit_reward(
        a1=answer1,
        a2_extracted=a2_text,
        ground_truth=ground_truth,
        answer_type=answer_type,
    )

    # Determine A2 correctness for logging
    a2_result = verify_answer(a2_text, ground_truth, answer_type)
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
) -> float:
    """R_format for refiner: format compliance check.

    Two modes controlled by use_think_answer_tags:

    **Tag mode** (use_think_answer_tags=True):
        Checks for <think>...</think><answer>...</answer> structure.
        Then validates the inner answer content by type.
        - Both tags present + valid inner answer → 0.0
        - Tags present but inner answer invalid → -0.5 (partial credit)
        - Tags missing entirely → -1.0

    **Bare mode** (use_think_answer_tags=False):
        Original normalize-first pipeline:
        1. normalize_answer(a2_text)
        2. Type-specific check (MCQ=letter, YesNo=yes/no, etc.)
        3. LLM fallback if deterministic check fails

    Penalty-only: 0.0 when compliant, -1.0 (or -0.5) when not.

    Args:
        a2_text: Raw A2 text from the refiner
        answer_type: Expected answer type
        ground_truth: Ground truth answer (for LLM format fallback)
        use_think_answer_tags: Whether to check for tag format

    Returns:
        0.0 if format-compliant, negative if not
    """
    if use_think_answer_tags:
        return _compute_tag_format_reward(a2_text, answer_type, ground_truth)

    return _compute_bare_format_reward(a2_text, answer_type, ground_truth)


def _compute_tag_format_reward(
    a2_text: str,
    answer_type: str,
    ground_truth: str = "",
) -> float:
    """Format reward for think+answer tag mode. Binary: 1.0 or 0.0.

    Following DeepSeek-R1 convention: simple binary format check.
    1.0 if both <think>+<answer> tags present with valid inner content.
    0.0 otherwise.

    Args:
        a2_text: Raw A2 text.
        answer_type: Expected answer type.
        ground_truth: Ground truth (unused, kept for API compat).

    Returns:
        1.0 if fully compliant, 0.0 otherwise.
    """
    from vlm_grpo.trajectory import extract_from_answer_tags, has_think_answer_tags

    if not has_think_answer_tags(a2_text):
        return 0.0

    # Tags present — check inner answer content strictly
    inner = extract_from_answer_tags(a2_text).strip()
    if not inner:
        return 0.0

    if answer_type == "mcq":
        if not re.match(r"^\([A-Da-d]\)$", inner):
            return 0.0
    elif answer_type == "yesno":
        if inner.lower() not in ("yes", "no"):
            return 0.0
    elif answer_type == "counting":
        if not re.match(r"^\d+$", inner):
            return 0.0
    elif answer_type == "numeric":
        try:
            float(inner.replace(",", "").rstrip("%").replace("/", "."))
        except ValueError:
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

        a2_text = _extract_a2_text(comp)

        r = compute_minimal_edit_reward(
            a1=a1,
            a2_extracted=a2_text,
            ground_truth=gt,
            answer_type=at,
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
        feedback_format_valid: Whether feedback is substantive
    """

    total_reward: float
    components: dict[str, float]
    weighted_components: dict[str, float]
    feedback_format_valid: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def compute_response_reward_breakdown(
    a1_text: str,
    a2_text: str,
    ground_truth: str,
    answer_type: str,
    choices: str,
    weights: Any,
    use_think_answer_tags: bool = False,
    reward_shaping_alpha: float = 0.0,
) -> TrajectoryResponseRewardBreakdown:
    """Compute response reward for a single trajectory.

    Evaluates answer quality: A1 correctness, A2 correctness,
    no-regression, format, and minimal edit.

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
    # Verify A1 and A2 ONCE each.
    a1_result = verify_answer(a1_text, ground_truth, answer_type)
    a1_correct = a1_result.is_correct

    # When think+answer tags are required and A2 has no <answer> tag:
    # short-circuit to penalty-only. Prevents false positives from stray
    # letters in prose being extracted as MCQ answers.
    requires_tags = use_think_answer_tags
    a2_has_tag = bool(re.search(r"<answer>", a2_text, re.IGNORECASE))

    if requires_tags and not a2_has_tag:
        # No <answer> tag: strong negative to push model toward tags.
        # -2.0 raw × w_a2_format ensures no-tag is always worse than
        # the worst tagged outcome (WW with tags), creating clear
        # gradient signal for tag adoption separate from correctness.
        _NO_TAG_PENALTY = -2.0
        components = {
            "a1_correctness": 0.0,
            "a2_correctness": 0.0,
            "no_regression": 0.0,
            "a2_format": _NO_TAG_PENALTY,
            "minimal_edit": 0.0,
        }
        weighted_components = {
            "a1_correctness": 0.0,
            "a2_correctness": 0.0,
            "no_regression": 0.0,
            "a2_format": _NO_TAG_PENALTY * weights.w_a2_format,
            "minimal_edit": 0.0,
        }
        return TrajectoryResponseRewardBreakdown(
            total_reward=_NO_TAG_PENALTY * weights.w_a2_format,
            components=components,
            weighted_components=weighted_components,
            a1_correct=a1_correct,
            a2_correct=False,
            a2_extracted="",
            a2_format_valid=False,
        )
    else:
        a2_result = verify_answer(a2_text, ground_truth, answer_type)
        a2_correct = a2_result.is_correct

        # Format reward
        r_a2_format = _compute_refiner_format_reward(
            a2_text, answer_type, ground_truth, use_think_answer_tags
        )
        a2_format_valid = r_a2_format > 0

        # Extract A2 for display
        a2_extracted = extract_answer_from_text(
            a2_text, answer_type, choices, require_answer_tag=requires_tags
        )

        # A1 correctness reward
        r_a1 = 1.0 if a1_correct else -1.0

        # A2 correctness reward
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

    # Minimal edit reward (only when both correct)
    if a1_correct and a2_correct:
        from vlm_grpo.rewards.verifier import DETERMINISTIC_TYPES
        from vlm_grpo.utils import normalized_edit_distance

        if answer_type in DETERMINISTIC_TYPES:
            a1_cmp = a1_result.extracted
            a2_cmp = a2_result.extracted
        else:
            a1_cmp = a1_text.strip().lower()
            a2_cmp = a2_text.strip().lower()
        edit_dist = normalized_edit_distance(a1_cmp, a2_cmp)
        r_edit = max(1.0 - 0.5 * edit_dist, 0.0)
    else:
        r_edit = 0.0

    components = {
        "a1_correctness": r_a1,
        "a2_correctness": r_a2,
        "no_regression": r_no_reg,
        "a2_format": r_a2_format,
        "minimal_edit": r_edit,
    }
    weighted_components = {
        "a1_correctness": r_a1 * weights.w_a1_correctness,
        "a2_correctness": r_a2 * weights.w_a2_correctness,
        "no_regression": r_no_reg * weights.w_no_regression,
        "a2_format": r_a2_format * weights.w_a2_format,
        "minimal_edit": r_edit * weights.w_minimal_edit,
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

    Components:
    1. Downstream-aware (did F1 help A2?) — transition-shaped or improvement-based
    2. Calibration — keyword-based assessment (variance-breaking tiebreaker)
    3. Format — is F1 substantive?
    4. Tag penalty — punish F1 using think/answer tags

    Args:
        feedback_text: Feedback text from the critic
        a1_text: Initial answer text
        a2_text: Refined answer text
        ground_truth: Ground truth answer
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        choices: MCQ choices string
        weights: FeedbackRewardWeights instance
        use_improvement_reward: If True, use R(A2)-R(A1) instead of
            transition-shaped downstream constants

    Returns:
        TrajectoryFeedbackRewardBreakdown with all component scores
    """
    from vlm_grpo.rewards.feedback import compute_feedback_calibration_reward

    # Verify A1 and A2 ONCE (downstream needs both)
    a1_result = verify_answer(a1_text, ground_truth, answer_type)
    a2_result = verify_answer(a2_text, ground_truth, answer_type)
    a1_correct = a1_result.is_correct
    a2_correct = a2_result.is_correct

    # Feedback format reward (penalty-only: 0.0 = valid, negative = invalid)
    r_format = compute_critic_format_reward(feedback_text)
    format_valid = r_format >= 0

    # Downstream-aware reward (computed directly, no extra verify_answer call)
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

    # Calibration: keyword-based assessment of whether F1 correctly
    # identifies A1 correctness. Used as variance-breaking tiebreaker
    # since feedback TEXT varies across K trajectories.
    r_calibration = compute_feedback_calibration_reward(feedback_text, a1_correct)

    # Tag leakage penalty: punish F1 that uses <think>/<answer> tags
    # from A1/A2 training. Only applied when w_tag_penalty > 0.
    r_tag_penalty = compute_f1_tag_penalty(feedback_text)

    components = {
        "downstream": r_downstream,
        "calibration": r_calibration,
        "format": r_format,
        "tag_penalty": r_tag_penalty,
    }
    w_tag = getattr(weights, "w_tag_penalty", 0.0)
    weighted_components = {
        "downstream": r_downstream * weights.w_downstream,
        "calibration": r_calibration * weights.w_calibration,
        "format": r_format * weights.w_format,
        "tag_penalty": r_tag_penalty * w_tag,
    }
    total_reward = sum(weighted_components.values())

    return TrajectoryFeedbackRewardBreakdown(
        total_reward=total_reward,
        components=components,
        weighted_components=weighted_components,
        feedback_format_valid=format_valid,
    )
