#!/usr/bin/env python3
"""Tests for the PAG-faithful per-segment binary reward composer.

Mirrors PAG's released ``verl/workers/reward_manager/pag.py`` reward
placement: r_a1, r_a2 emitted as separate scalars; A2 carries the
α·(R(A2) − R(A1)) shaping bonus, A1 does not. Gated trajectories
(F1 said CORRECT in the selective-revision rollout) emit r_a2 = None.
"""

import pytest

from vlm_grpo.config import FeedbackRewardWeights, ResponseRewardWeights
from vlm_grpo.rewards.composition import (
    PAGSegmentRewardBreakdown,
    compute_pag_feedback_breakdown,
    compute_pag_response_breakdown,
)

# Tag-mode A1/A2 strings for an MCQ where the ground truth is "(A)".
A_RIGHT = "<think>looks like A</think><answer>(A)</answer>"
A_WRONG = "<think>looks like B</think><answer>(B)</answer>"
GT = "(A)"

# F1 outputs paired against an A1 truth value. CORRECT means F1 says A1 is right.
F1_VERDICT_CORRECT = "<think>checks out</think>\\boxed{CORRECT}"
F1_VERDICT_WRONG = "<think>I see an error</think>\\boxed{INCORRECT}"
F1_NO_BOX = "<think>no boxed verdict</think>"


def _pag_response_weights() -> ResponseRewardWeights:
    """User spec: 0.9 correctness + 0.1 format per turn, no_regression
    + wr_bonus zeroed (the PAG composer ignores both and applies its
    own shaping bonus via pag_shaping_alpha).
    """
    return ResponseRewardWeights(
        w_a1_correctness=0.9,
        w_a1_format=0.1,
        w_a2_correctness=0.9,
        w_a2_format=0.1,
        w_no_regression=0.0,
        w_wr_bonus=0.0,
    )


def _pag_feedback_weights() -> FeedbackRewardWeights:
    """User spec: 0.9 verification + 0.1 format. Downstream zeroed (PAG
    is turn-independent; F1 sees no downstream signal).
    """
    return FeedbackRewardWeights(
        w_downstream=0.0,
        w_verification_accuracy=0.9,
        w_format=0.1,
    )


# ---------------------------------------------------------------------------
# Response head: r_a1 and r_a2 per-segment, four A1/A2 quadrants × α=1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a1_text,a2_text,a1_corr_01,a2_corr_01",
    [
        (A_RIGHT, A_RIGHT, 1.0, 1.0),  # RR
        (A_WRONG, A_RIGHT, 0.0, 1.0),  # WR
        (A_RIGHT, A_WRONG, 1.0, 0.0),  # RW
        (A_WRONG, A_WRONG, 0.0, 0.0),  # WW
    ],
)
def test_pag_response_ungated_quadrants(a1_text, a2_text, a1_corr_01, a2_corr_01):
    """All four A1/A2 quadrants under α=1, format always passes (tag-mode).

    Expected per the spec:
      r_a1 = 0.9·R_a1 + 0.1·R_a1_fmt = 0.9·R_a1 + 0.1
      r_a2 = 0.9·R_a2 + 0.1·R_a2_fmt + α·(R_a2 − R_a1)
           = 0.9·R_a2 + 0.1 + (R_a2 − R_a1)
    """
    bd = compute_pag_response_breakdown(
        a1_text=a1_text,
        a2_text=a2_text,
        ground_truth=GT,
        answer_type="mcq",
        choices="",
        weights=_pag_response_weights(),
        use_think_answer_tags=True,
        pag_shaping_alpha=1.0,
        gated=False,
    )

    assert isinstance(bd, PAGSegmentRewardBreakdown)
    assert not bd.gated
    assert bd.r_a2 is not None

    expected_r_a1 = 0.9 * a1_corr_01 + 0.1 * 1.0
    expected_bonus = 1.0 * (a2_corr_01 - a1_corr_01)
    expected_r_a2 = 0.9 * a2_corr_01 + 0.1 * 1.0 + expected_bonus

    assert bd.r_a1_corr == pytest.approx(a1_corr_01)
    assert bd.r_a2_corr == pytest.approx(a2_corr_01)
    assert bd.r_a1 == pytest.approx(expected_r_a1)
    assert bd.r_a2 == pytest.approx(expected_r_a2)
    assert bd.shaping_alpha == 1.0


def test_pag_response_gated_zeroes_a2():
    """gated=True: A2 is not generated → r_a2 = None, A1 reward intact."""
    bd = compute_pag_response_breakdown(
        a1_text=A_RIGHT,
        a2_text="",  # gated trajectory has empty A2 completion
        ground_truth=GT,
        answer_type="mcq",
        choices="",
        weights=_pag_response_weights(),
        use_think_answer_tags=True,
        pag_shaping_alpha=1.0,
        gated=True,
    )

    assert bd.gated
    assert bd.r_a2 is None
    # A1 still gets its standard per-segment reward — PAG §A.4 mandates
    # an A1 reward at every turn the policy acts; the gate stopping A2
    # doesn't affect this.
    assert bd.r_a1 == pytest.approx(0.9 * 1.0 + 0.1 * 1.0)
    assert bd.r_a1_corr == 1.0
    # A2 reward components zeroed for safety in the breakdown.
    assert bd.r_a2_corr == 0.0
    assert bd.a2_format == 0.0


def test_pag_response_alpha_zero_drops_shaping():
    """α=0 → b_y = 0; A2 reward reduces to corr + format only.

    Sanity check that the shaping math is gated cleanly behind the alpha
    coefficient.
    """
    bd = compute_pag_response_breakdown(
        a1_text=A_WRONG,
        a2_text=A_RIGHT,  # WR — would normally bonus by +1 at α=1
        ground_truth=GT,
        answer_type="mcq",
        choices="",
        weights=_pag_response_weights(),
        use_think_answer_tags=True,
        pag_shaping_alpha=0.0,
        gated=False,
    )

    expected_r_a2_no_bonus = 0.9 * 1.0 + 0.1 * 1.0  # 1.0
    assert bd.r_a2 == pytest.approx(expected_r_a2_no_bonus)
    assert bd.components["shaping_bonus"] == 0.0


def test_pag_response_format_missing():
    """Missing <answer> tag → format reward = 0; correctness still scores."""
    a1_no_tags = "I think the answer is (A)"  # no <think> + <answer> structure
    bd = compute_pag_response_breakdown(
        a1_text=a1_no_tags,
        a2_text=A_RIGHT,
        ground_truth=GT,
        answer_type="mcq",
        choices="",
        weights=_pag_response_weights(),
        use_think_answer_tags=True,
        pag_shaping_alpha=1.0,
        gated=False,
    )
    # tag_mode=True with missing tags → strict extraction returns "" → wrong.
    # That, in turn, means a1_corr = 0 and r_a1 = 0 (correctness 0 × 0.9 +
    # format 0 × 0.1).
    assert bd.r_a1_corr == 0.0
    assert bd.a1_format == 0.0
    assert bd.r_a1 == pytest.approx(0.0)
    # WR shaping bonus fires because a1_corr=0 and a2_corr=1.
    assert bd.r_a2 is not None
    assert bd.r_a2 == pytest.approx(0.9 * 1.0 + 0.1 * 1.0 + 1.0 * (1.0 - 0.0))


# ---------------------------------------------------------------------------
# Feedback head: binary verification + format, no downstream
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a1_text,f1_text,expected_verif,expected_fmt",
    [
        # A1 right + F1 says CORRECT → verification 1, format 1
        (A_RIGHT, F1_VERDICT_CORRECT, 1.0, 1.0),
        # A1 right + F1 says INCORRECT → verification 0 (wrong verdict)
        (A_RIGHT, F1_VERDICT_WRONG, 0.0, 1.0),
        # A1 wrong + F1 says INCORRECT → verification 1
        (A_WRONG, F1_VERDICT_WRONG, 1.0, 1.0),
        # A1 wrong + F1 says CORRECT → verification 0
        (A_WRONG, F1_VERDICT_CORRECT, 0.0, 1.0),
        # A1 right + F1 missing boxed → verification 0, format 0
        (A_RIGHT, F1_NO_BOX, 0.0, 0.0),
    ],
)
def test_pag_feedback_quadrants(a1_text, f1_text, expected_verif, expected_fmt):
    """Per PAG: r_f1 = 0.9·R_verification_01 + 0.1·R_fb_format.

    Verification is 1 iff F1's boxed verdict matches A1's actual correctness.
    Format is 1 iff `<think>…</think>…\\boxed{CORRECT|INCORRECT}` is present.
    """
    bd = compute_pag_feedback_breakdown(
        feedback_text=f1_text,
        a1_text=a1_text,
        ground_truth=GT,
        answer_type="mcq",
        weights=_pag_feedback_weights(),
    )

    expected_total = 0.9 * expected_verif + 0.1 * expected_fmt
    assert bd.components["verification"] == pytest.approx(expected_verif)
    assert bd.components["format"] == pytest.approx(expected_fmt)
    # Downstream is always 0 in the PAG feedback path (PAG is turn-independent).
    assert bd.components["downstream"] == 0.0
    assert bd.weighted_components["downstream"] == 0.0
    assert bd.total_reward == pytest.approx(expected_total)


def test_pag_feedback_ignores_downstream_weight():
    """Even if a YAML accidentally sets w_downstream > 0, the PAG feedback
    composer emits a 0 downstream contribution (PAG paper §3.2: turn-
    independent γ=0).
    """
    weights = FeedbackRewardWeights(
        w_downstream=0.5,  # accidentally non-zero
        w_verification_accuracy=0.4,
        w_format=0.1,
    )
    bd = compute_pag_feedback_breakdown(
        feedback_text=F1_VERDICT_CORRECT,
        a1_text=A_RIGHT,
        ground_truth=GT,
        answer_type="mcq",
        weights=weights,
    )
    # Downstream component is hard-zeroed.
    assert bd.weighted_components["downstream"] == 0.0
    # Total = 0.4·1 + 0.1·1 = 0.5 (NOT 0.5·1 + 0.4·1 + 0.1·1 = 1.0).
    assert bd.total_reward == pytest.approx(0.5)
