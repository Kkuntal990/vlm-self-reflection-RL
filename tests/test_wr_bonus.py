#!/usr/bin/env python3
"""Tests for the WR-bonus response reward component.

The WR bonus fires when A1 is wrong AND A2 is right (the "wrong→right"
quadrant). It's an additive Bernoulli {0, 1} indicator multiplied by
``w_wr_bonus`` — no symmetric RW penalty.

Covers both the raw breakdown (``compute_response_reward_breakdown``)
and the rescaled breakdown (``compute_response_reward_breakdown_01``)
for all four A1/A2 quadrants {RR, WR, RW, WW}.
"""

import pytest

from vlm_grpo.config import ResponseRewardWeights
from vlm_grpo.rewards.composition import (
    compute_response_reward_breakdown,
    compute_response_reward_breakdown_01,
)

# Tag-mode A1/A2 strings for an MCQ where the ground truth is "(A)".
A_RIGHT = "<think>looks like A</think><answer>(A)</answer>"
A_WRONG = "<think>looks like B</think><answer>(B)</answer>"
GT = "(A)"


def _weights_for_wr_bonus_job() -> ResponseRewardWeights:
    """Match the WR-bonus YAML's response weights.

    Defaults zero out everything except the four levers the WR-bonus
    job uses. The combination (w_a2_correctness=0.9 + formats=0.05+0.05
    + wr_bonus=1.0) is the spec from the conversation: RR=1.0, WR=2.0,
    RW=0.1, WW=0.1 when format always passes.
    """
    return ResponseRewardWeights(
        w_a1_correctness=0.0,
        w_a1_format=0.05,
        w_a2_correctness=0.9,
        w_a2_format=0.05,
        w_wr_bonus=1.0,
    )


# ---------------------------------------------------------------------------
# Component values (raw and rescaled): wr_bonus is Bernoulli {0, 1}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a1_text,a2_text,expected_wr_bonus",
    [
        (A_RIGHT, A_RIGHT, 0.0),  # RR: A1 right → no bonus
        (A_WRONG, A_RIGHT, 1.0),  # WR: A1 wrong, A2 right → bonus fires
        (A_RIGHT, A_WRONG, 0.0),  # RW: A2 wrong → no bonus
        (A_WRONG, A_WRONG, 0.0),  # WW: A2 wrong → no bonus
    ],
)
def test_wr_bonus_raw_breakdown_component(a1_text, a2_text, expected_wr_bonus):
    """Raw path: the wr_bonus component value is 1 only in the WR quadrant."""
    weights = _weights_for_wr_bonus_job()
    breakdown = compute_response_reward_breakdown(
        a1_text=a1_text,
        a2_text=a2_text,
        ground_truth=GT,
        answer_type="mcq",
        choices="(A) (B)",
        weights=weights,
    )
    assert breakdown.components["wr_bonus"] == expected_wr_bonus


@pytest.mark.parametrize(
    "a1_text,a2_text,expected_wr_bonus",
    [
        (A_RIGHT, A_RIGHT, 0.0),
        (A_WRONG, A_RIGHT, 1.0),
        (A_RIGHT, A_WRONG, 0.0),
        (A_WRONG, A_WRONG, 0.0),
    ],
)
def test_wr_bonus_rescaled_breakdown_component(a1_text, a2_text, expected_wr_bonus):
    """Rescaled path: same Bernoulli {0, 1} value — already in [0, 1]."""
    weights = _weights_for_wr_bonus_job()
    breakdown = compute_response_reward_breakdown_01(
        a1_text=a1_text,
        a2_text=a2_text,
        ground_truth=GT,
        answer_type="mcq",
        choices="(A) (B)",
        weights=weights,
    )
    assert breakdown.components["wr_bonus"] == expected_wr_bonus


# ---------------------------------------------------------------------------
# Total-reward quadrant table (the spec the user signed off on)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a1_text,a2_text,expected_total",
    [
        # In rescaled mode with format always passing (tag mode + atomic answer):
        # RR: 0.9·1 + 0.05·1 + 0.05·1 + 1.0·0 = 1.0
        # WR: 0.9·1 + 0.05·1 + 0.05·1 + 1.0·1 = 2.0
        # RW: 0.9·0 + 0.05·1 + 0.05·1 + 1.0·0 = 0.1
        # WW: 0.9·0 + 0.05·1 + 0.05·1 + 1.0·0 = 0.1
        (A_RIGHT, A_RIGHT, 1.0),
        (A_WRONG, A_RIGHT, 2.0),
        (A_RIGHT, A_WRONG, 0.1),
        (A_WRONG, A_WRONG, 0.1),
    ],
)
def test_wr_bonus_quadrant_table_rescaled(a1_text, a2_text, expected_total):
    """Rescaled total reward matches the {RR=1.0, WR=2.0, RW=0.1, WW=0.1} spec."""
    weights = _weights_for_wr_bonus_job()
    breakdown = compute_response_reward_breakdown_01(
        a1_text=a1_text,
        a2_text=a2_text,
        ground_truth=GT,
        answer_type="mcq",
        choices="(A) (B)",
        weights=weights,
    )
    assert abs(breakdown.total_reward - expected_total) < 1e-9


# ---------------------------------------------------------------------------
# Regression guard: w_wr_bonus=0 (default) leaves existing behavior unchanged
# ---------------------------------------------------------------------------


def test_wr_bonus_disabled_by_default():
    """With w_wr_bonus=0.0 (default), the bonus contributes nothing to total.

    Regression guard so existing experiments are unaffected by the new
    component. Compare two breakdowns: one with the default weight set
    (w_wr_bonus=0) and the same prompts/answers — total_reward should be
    independent of whether the WR quadrant fires.
    """
    weights = ResponseRewardWeights(
        w_a1_correctness=0.0,
        w_a1_format=0.0,
        w_a2_correctness=1.0,
        w_a2_format=0.0,
        # w_wr_bonus defaults to 0.0
    )
    rr = compute_response_reward_breakdown_01(A_RIGHT, A_RIGHT, GT, "mcq", "(A) (B)", weights)
    wr = compute_response_reward_breakdown_01(A_WRONG, A_RIGHT, GT, "mcq", "(A) (B)", weights)
    # Both end up with a2_correct=True → total=1.0 regardless of A1.
    assert rr.total_reward == wr.total_reward == 1.0
    # And the wr_bonus *component* still fires for WR — it's just
    # multiplied by zero in weighted_components.
    assert wr.components["wr_bonus"] == 1.0
    assert wr.weighted_components["wr_bonus"] == 0.0
