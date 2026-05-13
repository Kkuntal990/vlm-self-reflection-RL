#!/usr/bin/env python3
"""
Reward composition for the multi-turn self-reflection GRPO loop.

Active entry points used by the trainer (``critic_grpo.py``) and rollout
(``rollout.py``):

- ``compute_response_reward_breakdown[_01]`` — drives A1+A2 log-prob update
- ``compute_feedback_reward_breakdown[_01]`` — drives F1 log-prob update
- ``compute_baseline_a1_reward_breakdown`` — single-turn A1 baseline

The ``_01`` variants rescale each raw component to [0, 1] before weighting
(``--use_rescaled_rewards``). The non-``_01`` variants emit raw values. See
``docs/rewards.md`` for the per-component landscape.
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
from vlm_grpo.rewards.verifier import verify_answer
from vlm_grpo.trajectory import extract_answer_from_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


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


def _compute_tag_format_reward(
    a2_text: str,
    answer_type: str,
    ground_truth: str = "",
) -> float:
    """R_format for A1/A2: think+answer tag compliance check.

    Binary {0, +1}. Two conditions must both hold for +1:
      1. Both `<think>...</think>` and `<answer>...</answer>` present.
      2. Inner content of `<answer>` is exactly an atomic answer for the
         expected type (MCQ letter, integer, yes/no, etc.) — no descriptor
         text, no prose.

    Correctness extraction is more lenient (accepts `(A) descriptor`),
    but format reward demands the clean canonical form.

    Args:
        a2_text: Raw A1 or A2 text (used by both turns).
        answer_type: Expected answer type
        ground_truth: Ground truth (unused, kept for API compat — older
            callers passed this for an LLM-fallback path that has been
            removed along with the non-tag mode).

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
        # Accept A-F (uppercase + lowercase). trajectory.py and the rest of
        # the MCQ matching code already accept A-F; restricting the format
        # reward to A-D zeroed format credit on E/F options.
        return bool(re.match(r"^\([A-Fa-f]\)$", inner) or re.match(r"^[A-Fa-f]\.?$", inner))
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
    no F1 / A2 / shaped-improvement components participate.

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
    reward_shaping_alpha: float = 0.0,
) -> TrajectoryResponseRewardBreakdown:
    """Compute response reward for a single trajectory.

    Evaluates answer quality: A1 correctness, A2 correctness, format,
    and the optional additive WR bonus.

    Args:
        a1_text: Initial answer text
        a2_text: Refined answer text
        ground_truth: Ground truth answer
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        choices: MCQ choices string
        weights: ResponseRewardWeights instance
        reward_shaping_alpha: Kept for API parity; unused in this path
            since the transition / shaped reward component was deleted.

    Returns:
        TrajectoryResponseRewardBreakdown with all component scores
    """
    del reward_shaping_alpha  # parity arg, no longer used here
    # Tag-mode is the only supported extraction path: <answer> tag presence
    # is required. Missing tag → extracted="" → MatchResult.is_correct=False
    # naturally, so the correctness path handles "no extractable answer"
    # without any special short-circuit. Format reward is independent and
    # binary.
    a1_result = verify_answer(a1_text, ground_truth, answer_type, strict=True)
    a1_correct = a1_result.is_correct

    a2_result = verify_answer(a2_text, ground_truth, answer_type, strict=True)
    a2_correct = a2_result.is_correct

    # Format rewards — pure structural check (binary {0, +1}) per turn.
    # Both A1 and A2 are evaluated independently for tag presence + clean
    # atomic inner content. Symmetry across turns prevents the model from
    # learning "format only matters for A2".
    r_a1_format = _compute_tag_format_reward(a1_text, answer_type, ground_truth)
    r_a2_format = _compute_tag_format_reward(a2_text, answer_type, ground_truth)
    a2_format_valid = r_a2_format > 0

    # Extract A2 for display (uses same strictness/tag-requirement as scoring)
    a2_extracted = extract_answer_from_text(a2_text, answer_type, choices, strict=True)

    # A1 / A2 correctness rewards
    r_a1 = 1.0 if a1_correct else -1.0
    if answer_type in ("counting", "open") and a2_result.score is not None:
        r_a2 = 2.0 * a2_result.score - 1.0
    else:
        r_a2 = 1.0 if a2_correct else -1.0

    # WR bonus: Bernoulli {0, 1} indicator that fires when A1 was wrong
    # and A2 corrected to right (the WR quadrant). Additive — does NOT
    # penalise RW. Off by default (w_wr_bonus=0.0). See
    # ResponseRewardWeights docstring.
    r_wr_bonus = 1.0 if (not a1_correct and a2_correct) else 0.0

    components = {
        "a1_correctness": r_a1,
        "a1_format": r_a1_format,
        "a2_correctness": r_a2,
        "a2_format": r_a2_format,
        "wr_bonus": r_wr_bonus,
    }
    weighted_components = {
        "a1_correctness": r_a1 * weights.w_a1_correctness,
        "a1_format": r_a1_format * weights.w_a1_format,
        "a2_correctness": r_a2 * weights.w_a2_correctness,
        "a2_format": r_a2_format * weights.w_a2_format,
        "wr_bonus": r_wr_bonus * weights.w_wr_bonus,
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
# PAG-faithful per-segment binary rewards (arXiv:2506.10406)
# =============================================================================
#
# Matches the binary {0, 1} per-segment reward placement implemented in PAG's
# released code (``verl/workers/reward_manager/pag.py``):
#
#   reward_tensor[i, last_A1_tok]  =  R_y(A1)                            ∈ {0,1}
#   reward_tensor[i, last_F1_tok]  =  R_v(F1, A1)                        ∈ {0,1}
#   reward_tensor[i, last_A2_tok]  =  R_y(A2) + α·(R_y(A2) − R_y(A1))    ∈ ℝ
#
# Our GRPO scaffold uses one scalar per turn (broadcast to that turn's tokens
# by the trainer's per-segment loss aggregation) rather than a per-token reward
# tensor; the two are equivalent under γ=1 within-turn / γ=0 between-turn (PAG's
# ``multiturn_mask``). The composer below emits ``r_a1`` and ``r_a2`` as
# separate fields on the response breakdown, and the trainer (with
# ``use_pag_segment_rewards=True``) reads them via the existing
# ``separate_turn_loss=True`` per-segment K-group advantage path.
#
# Format reward kept at 0.1 weight as a small structural anchor — PAG has none
# (their verdict uses a trailing-sentence regex), but our extraction is
# ``\boxed{}``-based and benefits from a binary {0,1} format component to
# discourage missing-tag F1 outputs. With weights summing to 1.0, the per-turn
# reward stays in [0, 1] (A1 / F1) or [-α, 1+α] (A2 after shaping).


@dataclass
class PAGSegmentRewardBreakdown:
    """Per-segment PAG-style breakdown (arXiv:2506.10406).

    Unlike the legacy ``TrajectoryResponseRewardBreakdown`` (one pooled scalar
    over A1 + A2), this breakdown carries r_a1 and r_a2 separately so the
    trainer can drive two independent K-group baselines.

    Attributes:
        r_a1: A1's per-segment reward, ``w_a1_corr·R_a1_corr_01 +
            w_a1_fmt·R_a1_fmt_01``. ∈ [0, 1] for convex weights.
        r_a2: A2's per-segment reward, ``w_a2_corr·R_a2_corr_01 +
            w_a2_fmt·R_a2_fmt_01 + α·(R_a2_corr_01 − R_a1_corr_01)``. ∈ ℝ.
            None when the trajectory was gated (selective revision stopped
            at F1) — signals downstream code to exclude this trajectory from
            the A2 K-group baseline and from the A2 policy loss.
        r_a1_corr: Raw binary correctness ∈ {0, 1} (used for diagnostics +
            for the A2 shaping baseline; do not weight here, the weighted
            value is already in r_a1).
        r_a2_corr: Raw binary correctness ∈ {0, 1}.
        a1_format: Raw binary {0, 1} format compliance for A1.
        a2_format: Raw binary {0, 1} format compliance for A2.
        a1_correct / a2_correct: Boolean correctness flags.
        a2_extracted: Strict atomic extraction from ``<answer>`` tag for A2.
        a2_format_valid: Whether A2 passed the format check.
        gated: Whether the selective-revision gate stopped this trajectory
            at F1 (i.e. F1 said ``CORRECT``). When True, A2 is empty and
            r_a2 is None.
        shaping_alpha: α used in b_y. Stored for diagnostics.
        total_reward: Sum of segment rewards (skip-A2 → r_a1 only). Used
            only for logging — the trainer reads r_a1 / r_a2 directly.
        components / weighted_components: Compatibility dicts so the existing
            metric-logger code (which iterates breakdown.components) works
            unchanged. ``components`` holds raw values; ``weighted_components``
            holds the weighted values that get summed.
    """

    r_a1: float
    r_a2: float | None
    r_a1_corr: float
    r_a2_corr: float
    a1_format: float
    a2_format: float
    a1_correct: bool
    a2_correct: bool
    a2_extracted: str
    a2_format_valid: bool
    gated: bool
    shaping_alpha: float
    total_reward: float
    components: dict[str, float]
    weighted_components: dict[str, float]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def compute_pag_response_breakdown(
    a1_text: str,
    a2_text: str,
    ground_truth: str,
    answer_type: str,
    choices: str,
    weights: Any,
    pag_shaping_alpha: float = 1.0,
    gated: bool = False,
) -> PAGSegmentRewardBreakdown:
    """Compute the PAG-style per-segment response reward for one trajectory.

    Mirrors PAG's ``pag.py`` reward placement: r_a1 lives on the A1 segment,
    r_a2 lives on the A2 segment with the shaping bonus baked in. Binary
    correctness {0, 1} (mirrors PAG's ``first_result["acc"]`` / ``policy_result["acc"]``).

    **Shaping bonus formulation.** This composer uses RAW BINARY correctness
    for the b_y term (``bonus = α·(R_a2_corr_01 − R_a1_corr_01)``), matching
    PAG's released code at ``verl/workers/reward_manager/pag.py``:

        reward_value += self.rs_coef * (policy_result["acc"] - prev_acc)

    where ``policy_result["acc"]`` and ``prev_acc`` are the binary
    accuracies, NOT the full weighted segment rewards. The PAG paper's
    Eq. (5) is written more abstractly as ``α·(R(ŷ_2) − R(ŷ_1))``, but the
    released implementation is unambiguous: the bonus is over correctness
    accuracies only, so the format component does NOT enter b_y. When both
    A1 and A2 format pass (the common case in tag mode), this differs from
    a "full weighted reward" formulation by exactly the
    ``w_a2_correctness`` factor — at α=1 and 0.9/0.1 weights, that's an
    11% larger WR/RW shaping signal than the full-reward formulation.
    Intentional: match the released code, not the paper's abstract form.

    When ``gated=True`` (F1 verdict said CORRECT in the selective-revision
    rollout), A2 was not generated; r_a2 is set to None and the trajectory
    contributes nothing to the A2 K-group baseline downstream.

    Args:
        a1_text: Initial answer text.
        a2_text: Refined answer text (empty string when gated).
        ground_truth: Ground truth answer.
        answer_type: Answer type ("mcq", "yesno", "numeric", "open",
            "counting").
        choices: MCQ choices string (kept for API parity with the legacy
            composers).
        weights: ``ResponseRewardWeights`` instance. ``w_a1_correctness`` and
            ``w_a2_correctness`` should be 0.9 and the format weights 0.1
            for PAG-faithful runs; the shaping bonus uses
            ``pag_shaping_alpha`` directly.
        pag_shaping_alpha: α coefficient on the b_y bonus.
        gated: Whether the selective-revision gate stopped this trajectory
            at F1 (i.e. F1 said CORRECT).

    Returns:
        ``PAGSegmentRewardBreakdown`` with r_a1 and r_a2 as separate scalars
        (r_a2 = None when gated).
    """
    # A1: binary correctness + binary format. Tag-mode strict extraction is
    # the only supported path.
    a1_result = verify_answer(a1_text, ground_truth, answer_type, strict=True)
    a1_correct = a1_result.is_correct
    r_a1_corr_01 = 1.0 if a1_correct else 0.0
    r_a1_fmt_01 = _compute_tag_format_reward(a1_text, answer_type, ground_truth)

    r_a1 = r_a1_corr_01 * weights.w_a1_correctness + r_a1_fmt_01 * weights.w_a1_format

    # A2: only compute when the trajectory was NOT gated.
    if gated:
        r_a2_corr_01 = 0.0
        r_a2_fmt_01 = 0.0
        a2_correct = False
        a2_extracted = ""
        a2_format_valid = False
        r_a2: float | None = None
        bonus = 0.0
    else:
        a2_result = verify_answer(a2_text, ground_truth, answer_type, strict=True)
        a2_correct = a2_result.is_correct
        r_a2_corr_01 = 1.0 if a2_correct else 0.0
        r_a2_fmt_01 = _compute_tag_format_reward(a2_text, answer_type, ground_truth)
        a2_extracted = extract_answer_from_text(a2_text, answer_type, choices, strict=True)
        # Match legacy composer's threshold: > 0 → valid. Binary {0, 1} reward
        # makes the choice moot in practice, but `> 0` keeps the convention.
        a2_format_valid = r_a2_fmt_01 > 0
        bonus = pag_shaping_alpha * (r_a2_corr_01 - r_a1_corr_01)
        r_a2 = r_a2_corr_01 * weights.w_a2_correctness + r_a2_fmt_01 * weights.w_a2_format + bonus

    # ``gated`` lives on the dataclass as its own field (see
    # PAGSegmentRewardBreakdown); we do NOT smuggle it into ``components`` /
    # ``weighted_components`` to avoid polluting the wandb component metric
    # loop in critic_grpo.py with a non-reward value.
    components = {
        "a1_correctness": r_a1_corr_01,
        "a1_format": r_a1_fmt_01,
        "a2_correctness": r_a2_corr_01,
        "a2_format": r_a2_fmt_01,
        "shaping_bonus": bonus,
    }
    weighted_components = {
        "a1_correctness": r_a1_corr_01 * weights.w_a1_correctness,
        "a1_format": r_a1_fmt_01 * weights.w_a1_format,
        "a2_correctness": r_a2_corr_01 * weights.w_a2_correctness,
        "a2_format": r_a2_fmt_01 * weights.w_a2_format,
        "shaping_bonus": bonus,
    }
    total_reward = r_a1 + (0.0 if r_a2 is None else r_a2)

    return PAGSegmentRewardBreakdown(
        r_a1=r_a1,
        r_a2=r_a2,
        r_a1_corr=r_a1_corr_01,
        r_a2_corr=r_a2_corr_01,
        a1_format=r_a1_fmt_01,
        a2_format=r_a2_fmt_01,
        a1_correct=a1_correct,
        a2_correct=a2_correct,
        a2_extracted=a2_extracted,
        a2_format_valid=a2_format_valid,
        gated=gated,
        shaping_alpha=pag_shaping_alpha,
        total_reward=total_reward,
        components=components,
        weighted_components=weighted_components,
    )


def compute_pag_feedback_breakdown(
    feedback_text: str,
    a1_text: str,
    ground_truth: str,
    answer_type: str,
    weights: Any,
) -> TrajectoryFeedbackRewardBreakdown:
    """Compute the PAG-style binary feedback reward for one trajectory.

    Matches PAG's ``pag.py`` verifier placement: R_v(F1, A1) ∈ {0, 1} where
    1 means F1's verdict matches A1's actual correctness. NO downstream
    component (PAG's turn-independent γ=0 means F1 sees no downstream signal).

    Args:
        feedback_text: F1 output text (with ``<think>...</think>`` and
            ``\\boxed{CORRECT|INCORRECT}``).
        a1_text: Initial answer text (needed to evaluate verdict correctness).
        ground_truth: Ground truth answer.
        answer_type: Answer type.
        weights: ``FeedbackRewardWeights`` instance. ``w_verification_accuracy``
            should be 0.9 and ``w_format`` 0.1 for PAG-faithful runs.
            ``w_downstream`` is ignored — PAG has no downstream term.

    Returns:
        ``TrajectoryFeedbackRewardBreakdown`` with the binary verification +
        format components and ``downstream`` recorded as 0.0 for logging.
    """
    a1_result = verify_answer(a1_text, ground_truth, answer_type)
    a1_correct = a1_result.is_correct

    # Binary verification reward: 1 if F1's boxed verdict matches A1's truth.
    r_verification_pm = compute_verification_accuracy_reward(feedback_text, a1_correct)
    r_verification_01 = 1.0 if r_verification_pm > 0 else 0.0

    # Binary format reward (already {0, 1}).
    r_fb_format = compute_feedback_format_reward(feedback_text)

    components = {
        "downstream": 0.0,
        "verification": r_verification_01,
        "format": r_fb_format,
    }
    weighted_components = {
        # Downstream is zeroed in PAG (turn-independent γ=0). Even if a YAML
        # accidentally sets w_downstream > 0, the PAG path emits a 0
        # contribution to keep the reward composition bit-for-bit faithful to
        # the paper.
        "downstream": 0.0,
        "verification": r_verification_01 * weights.w_verification_accuracy,
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
# the standalone ``feedback.compute_downstream_aware_reward`` function, which
# differs).
#
# IMPORTANT: keep these in sync with the inline values if either function is
# edited. See the per-component rationale in the docstrings below.
_A1_CORR_RANGE = (-1.0, 1.0)
_A2_CORR_RANGE = (-1.0, 1.0)
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
# ``compute_feedback_format_reward``) so they need no rescaling.


def _format_to_01(raw: float) -> float:
    """Clamp a format reward to [0, 1].

    The think+answer tag-mode path already returns {0, +1}, so this is
    effectively a no-op. Defensive bounds-check kept in case a future
    helper emits a value outside [0, 1].

    Args:
        raw: Raw format reward.

    Returns:
        Format reward in ``[0, 1]``.
    """
    if raw < 0.0:
        return 0.0
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


def compute_a2_format_01(
    a2_text: str,
    answer_type: str,
    ground_truth: str = "",
) -> float:
    """[0, 1]-rescaled A2 format reward.

    Reuses ``_compute_tag_format_reward`` which already returns
    {0, +1} (the only supported think+answer tag-mode path), so this is
    effectively a clamp-to-[0, 1] pass-through.

    Args:
        a2_text: Raw A2 text from the refiner.
        answer_type: Expected answer type.
        ground_truth: Ground truth (unused, kept for API compat).

    Returns:
        Float in ``[0, 1]``.
    """
    raw = _compute_tag_format_reward(a2_text, answer_type, ground_truth)
    return _format_to_01(raw)


def compute_downstream_01(
    feedback_text: str,
    a1_text: str,
    a2_text: str,
    ground_truth: str,
    answer_type: str,
    reward_shaping_alpha: float = 0.0,
    tolerance: float = 0.01,
) -> float:
    """[0, 1]-rescaled downstream-aware reward (transition mode only).

    Mirrors the inline computation in ``compute_feedback_reward_breakdown``:
        Deterministic: RR=+1, RW=-1.5, WR=+3, WW=-1  (range [-1.5, +3])
        Open-ended:    RR=+1, RW=-2,   WR=+2, WW=-1  (range [-2, +2])

    Empty F1 → 0.0 raw → rescaled to the lower-bound mid-value (treated as
    "no contribution"). The asymmetric gate (``min(raw, midpoint)`` when
    verification fails) is applied by the caller after this rescaling.

    Args:
        feedback_text: F1 text (used for the empty-check short-circuit).
        a1_text: Initial answer text.
        a2_text: Refined answer text.
        ground_truth: Ground truth answer.
        answer_type: Answer type.
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
    reward_shaping_alpha: float = 0.0,
) -> TrajectoryResponseRewardBreakdown:
    """[0, 1]-rescaled response reward (parallel to
    ``compute_response_reward_breakdown``).

    Each component is in [0, 1]; with ResponseRewardWeights summing to 1.0,
    ``total_reward`` lands in [0, 1].

    Args:
        a1_text: Initial answer text.
        a2_text: Refined answer text.
        ground_truth: Ground truth answer.
        answer_type: Answer type ("mcq", "yesno", "numeric", "open",
            "counting").
        choices: MCQ choices string.
        weights: ResponseRewardWeights instance.
        reward_shaping_alpha: Kept for API parity; the transition / shaped
            reward component was deleted, so this argument is now unused.

    Returns:
        ``TrajectoryResponseRewardBreakdown`` with each ``components[k]`` in
        [0, 1] and ``weighted_components[k] = weight_k × component_k``.
    """
    del reward_shaping_alpha  # parity arg, no longer used here

    a1_result = verify_answer(a1_text, ground_truth, answer_type, strict=True)
    a1_correct = a1_result.is_correct
    a2_result = verify_answer(a2_text, ground_truth, answer_type, strict=True)
    a2_correct = a2_result.is_correct

    r_a1_format = compute_a2_format_01(a1_text, answer_type, ground_truth)
    r_a2_format = compute_a2_format_01(a2_text, answer_type, ground_truth)
    a2_format_valid = r_a2_format > 0.5

    a2_extracted = extract_answer_from_text(a2_text, answer_type, choices, strict=True)

    # A1 / A2 correctness (rescaled to [0, 1])
    r_a1 = 1.0 if a1_correct else 0.0
    if answer_type in ("counting", "open") and a2_result.score is not None:
        r_a2 = a2_result.score  # already in [0, 1]
    else:
        r_a2 = 1.0 if a2_correct else 0.0

    # WR bonus (rescaled path): same Bernoulli {0, 1} indicator. Already
    # in [0, 1] so no rescaling needed. Matches the raw path component.
    r_wr_bonus = 1.0 if (not a1_correct and a2_correct) else 0.0

    components = {
        "a1_correctness": r_a1,
        "a1_format": r_a1_format,
        "a2_correctness": r_a2,
        "a2_format": r_a2_format,
        "wr_bonus": r_wr_bonus,
    }
    weighted_components = {
        "a1_correctness": r_a1 * weights.w_a1_correctness,
        "a1_format": r_a1_format * weights.w_a1_format,
        "a2_correctness": r_a2 * weights.w_a2_correctness,
        "a2_format": r_a2_format * weights.w_a2_format,
        "wr_bonus": r_wr_bonus * weights.w_wr_bonus,
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
