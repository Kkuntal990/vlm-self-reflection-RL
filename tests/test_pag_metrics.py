#!/usr/bin/env python3
"""Tests for the PAG-aware metric aggregation in ``compute_self_reflection_metrics``.

Covers:
  • Per-segment reward means (r_a1, r_a2, r_f1, shaping_bonus) over a mixed
    batch of gated + non-gated trajectories.
  • Effective accuracy = A1 for gated, A2 for non-gated (the inference-time
    final-answer accuracy under selective revision).
  • Gate quality: productive_gate (gated AND A1 right) vs sycophantic_gate
    (gated AND A1 wrong).
  • F1 verdict-precision metrics.
  • A2-related metrics exclude gated trajectories from the denominator.
"""

import pytest

from vlm_grpo.rewards.composition import (
    PAGSegmentRewardBreakdown,
    TrajectoryFeedbackRewardBreakdown,
)
from vlm_grpo.rollout import SelfReflectionRolloutResult, compute_self_reflection_metrics


def _resp_bd(
    r_a1: float,
    r_a2: float | None,
    a1_correct: bool,
    a2_correct: bool,
    gated: bool,
    shaping_bonus: float = 0.0,
    a1_fmt: float = 1.0,
    a2_fmt: float = 1.0,
) -> PAGSegmentRewardBreakdown:
    """Build a minimal PAG response breakdown for a single trajectory."""
    components = {
        "a1_correctness": 1.0 if a1_correct else 0.0,
        "a1_format": a1_fmt,
        "a2_correctness": 1.0 if a2_correct else 0.0,
        "a2_format": a2_fmt,
        "shaping_bonus": shaping_bonus,
    }
    return PAGSegmentRewardBreakdown(
        r_a1=r_a1,
        r_a2=r_a2,
        r_a1_corr=1.0 if a1_correct else 0.0,
        r_a2_corr=1.0 if a2_correct else 0.0,
        a1_format=a1_fmt,
        a2_format=a2_fmt,
        a1_correct=a1_correct,
        a2_correct=a2_correct,
        a2_extracted="(A)" if a2_correct else "(B)",
        a2_format_valid=a2_fmt > 0,
        gated=gated,
        shaping_alpha=1.0,
        total_reward=r_a1 + (0.0 if r_a2 is None else r_a2),
        components=components,
        weighted_components=components,
    )


def _fb_bd(
    r_f1: float,
    fmt: float = 1.0,
    verification: float = 1.0,
) -> TrajectoryFeedbackRewardBreakdown:
    components = {
        "downstream": 0.0,
        "verification": verification,
        "format": fmt,
    }
    return TrajectoryFeedbackRewardBreakdown(
        total_reward=r_f1,
        components=components,
        weighted_components=components,
    )


def _wrap_in_result(
    resp_bds: list[PAGSegmentRewardBreakdown],
    fb_bds: list[TrajectoryFeedbackRewardBreakdown],
) -> SelfReflectionRolloutResult:
    """Pack per-trajectory breakdowns into a single rollout result."""
    return SelfReflectionRolloutResult(
        sample_index=0,
        question="q",
        image_path="",
        ground_truth="(A)",
        answer_type="mcq",
        choices="",
        dataset_name="test",
        answer1s=[""] * len(resp_bds),
        feedbacks=[""] * len(fb_bds),
        answer2s=[""] * len(resp_bds),
        response_rewards=[float(bd.total_reward) for bd in resp_bds],
        feedback_rewards=[float(bd.total_reward) for bd in fb_bds],
        response_breakdowns=resp_bds,
        feedback_breakdowns=fb_bds,
    )


# ---------------------------------------------------------------------------
# Mixed-batch scenarios
# ---------------------------------------------------------------------------


def test_metrics_mixed_batch_pag():
    """K=4 batch with one of each gate × correctness combination.

    Construction:
      • #0: gated, A1 correct (PRODUCTIVE gate, no A2)            → effective right
      • #1: gated, A1 wrong   (SYCOPHANTIC gate, no A2)           → effective wrong
      • #2: not gated, A1 wrong → A2 right                        → WR, effective right
      • #3: not gated, A1 right → A2 wrong                        → RW, effective wrong
    """
    resp_bds = [
        _resp_bd(r_a1=1.0, r_a2=None, a1_correct=True, a2_correct=False, gated=True),
        _resp_bd(r_a1=0.1, r_a2=None, a1_correct=False, a2_correct=False, gated=True),
        _resp_bd(
            r_a1=0.1, r_a2=2.0, a1_correct=False, a2_correct=True, gated=False, shaping_bonus=1.0
        ),
        _resp_bd(
            r_a1=1.0,
            r_a2=0.1,
            a1_correct=True,
            a2_correct=False,
            gated=False,
            shaping_bonus=-1.0,
        ),
    ]
    fb_bds = [
        _fb_bd(r_f1=1.0, verification=1.0),  # F1 said CORRECT and was right
        _fb_bd(r_f1=0.1, verification=0.0),  # F1 said CORRECT but A1 was wrong
        _fb_bd(r_f1=1.0, verification=1.0),  # F1 said WRONG (correctly)
        _fb_bd(r_f1=0.1, verification=0.0),  # F1 said WRONG but A1 was right
    ]
    metrics = compute_self_reflection_metrics([_wrap_in_result(resp_bds, fb_bds)])

    # Totals + gating split
    assert metrics["sr/total_trajectories"] == 4.0
    assert metrics["sr/gated_rate"] == pytest.approx(0.5)  # 2 / 4
    assert metrics["sr/productive_gate_rate"] == pytest.approx(0.25)  # 1 / 4
    assert metrics["sr/sycophantic_gate_rate"] == pytest.approx(0.25)  # 1 / 4

    # Transition rates denominate by non-gated trajectories (= 2)
    assert metrics["sr/wr_rate"] == pytest.approx(0.5)  # 1 / 2
    assert metrics["sr/rw_rate"] == pytest.approx(0.5)  # 1 / 2
    assert metrics["sr/rr_rate"] == 0.0
    assert metrics["sr/ww_rate"] == 0.0

    # A1 accuracy uses full denominator (every trajectory generates A1)
    assert metrics["sr/a1_accuracy"] == pytest.approx(0.5)  # 2 / 4 right
    # A2 accuracy excludes gated: among non-gated, 1 right (WR), 1 wrong (RW)
    assert metrics["sr/a2_accuracy"] == pytest.approx(0.5)  # 1 / 2
    # Effective: gated → A1 (1 right), non-gated → A2 (1 right) → 2 / 4
    assert metrics["sr/effective_accuracy"] == pytest.approx(0.5)

    # Per-segment rewards
    assert metrics["sr/r_a1_mean"] == pytest.approx((1.0 + 0.1 + 0.1 + 1.0) / 4)
    # r_a2_mean: only over non-gated samples (#2 = 2.0, #3 = 0.1)
    assert metrics["sr/r_a2_mean"] == pytest.approx((2.0 + 0.1) / 2)
    # r_f1_mean uses full denominator (F1 always runs)
    assert metrics["sr/r_f1_mean"] == pytest.approx((1.0 + 0.1 + 1.0 + 0.1) / 4)
    # shaping bonus mean over non-gated: (+1) + (-1) = 0 / 2
    assert metrics["sr/shaping_bonus_mean"] == pytest.approx(0.0)

    # F1 verdict precision:
    #   CORRECT verdict precision = P(A1 right | F1=CORRECT) = 1 / 2 (one productive, one syco)
    #   WRONG verdict precision   = P(A1 wrong | F1=WRONG)   = 1 / 2 (one WR-honest, one spurious)
    assert metrics["sr/f1_correct_verdict_precision"] == pytest.approx(0.5)
    assert metrics["sr/f1_wrong_verdict_precision"] == pytest.approx(0.5)


def test_metrics_all_gated_pag():
    """Every K-group trajectory gated — no A2 ran. RR/RW/WR/WW rates should
    NOT divide by zero (n_transitions guards with max(..., 1)), and
    effective accuracy collapses to A1 accuracy.
    """
    resp_bds = [
        _resp_bd(r_a1=1.0, r_a2=None, a1_correct=True, a2_correct=False, gated=True),
        _resp_bd(r_a1=0.1, r_a2=None, a1_correct=False, a2_correct=False, gated=True),
        _resp_bd(r_a1=1.0, r_a2=None, a1_correct=True, a2_correct=False, gated=True),
    ]
    fb_bds = [_fb_bd(r_f1=1.0)] * 3
    metrics = compute_self_reflection_metrics([_wrap_in_result(resp_bds, fb_bds)])

    assert metrics["sr/gated_rate"] == pytest.approx(1.0)
    assert metrics["sr/a1_accuracy"] == pytest.approx(2 / 3)
    assert metrics["sr/effective_accuracy"] == pytest.approx(2 / 3)
    # Transition rates all 0 (no transitions happened) — guard prevents div-by-zero.
    assert metrics["sr/wr_rate"] == 0.0
    assert metrics["sr/rw_rate"] == 0.0
    assert metrics["sr/rr_rate"] == 0.0
    assert metrics["sr/ww_rate"] == 0.0
    # A2 mean isn't meaningfully defined when no A2 ran, but the n_transitions
    # max-guard keeps it finite (= 0.0).
    assert metrics["sr/r_a2_mean"] == 0.0
    assert metrics["sr/shaping_bonus_mean"] == 0.0


def test_metrics_no_gating_pag():
    """No gate fired (selective_revision disabled or F1 always says WRONG).

    Behaves identically to the legacy path: effective_accuracy == a2_accuracy,
    transition denominators = n.
    """
    resp_bds = [
        # WR: A1 wrong → A2 right
        _resp_bd(
            r_a1=0.1, r_a2=2.0, a1_correct=False, a2_correct=True, gated=False, shaping_bonus=1.0
        ),
        # RR: both right
        _resp_bd(r_a1=1.0, r_a2=1.0, a1_correct=True, a2_correct=True, gated=False),
    ]
    fb_bds = [_fb_bd(r_f1=1.0)] * 2
    metrics = compute_self_reflection_metrics([_wrap_in_result(resp_bds, fb_bds)])

    assert metrics["sr/gated_rate"] == 0.0
    assert metrics["sr/a2_accuracy"] == 1.0
    assert metrics["sr/effective_accuracy"] == 1.0
    assert metrics["sr/wr_rate"] == 0.5
    assert metrics["sr/rr_rate"] == 0.5
    # Both F1 verdicts said WRONG; one was honest (about WR), one was spurious (about RR).
    assert metrics["sr/f1_wrong_verdict_precision"] == pytest.approx(0.5)
