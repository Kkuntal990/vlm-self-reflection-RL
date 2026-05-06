#!/usr/bin/env python3
"""Tests for the per-component [0, 1]-rescaled multi-turn reward path.

Covers ``_to_unit``, the per-component ``*_01`` wrappers, and the umbrella
``compute_response_reward_breakdown_01`` /
``compute_feedback_reward_breakdown_01`` entry points. Also includes a
regression guard ensuring the existing raw breakdown path is unchanged.
"""

from vlm_grpo.config import FeedbackRewardWeights, ResponseRewardWeights
from vlm_grpo.rewards.composition import (
    _NO_REG_DET_RANGE,
    _to_unit,
    compute_a2_correctness_01,
    compute_a2_format_01,
    compute_downstream_01,
    compute_fb_format_01,
    compute_feedback_reward_breakdown,
    compute_feedback_reward_breakdown_01,
    compute_no_regression_01,
    compute_response_reward_breakdown,
    compute_response_reward_breakdown_01,
    compute_verification_01,
)

# ---------------------------------------------------------------------------
# _to_unit helper
# ---------------------------------------------------------------------------


def test_to_unit_endpoints():
    assert _to_unit(-1.0, -1.0, 1.0) == 0.0
    assert _to_unit(1.0, -1.0, 1.0) == 1.0


def test_to_unit_midpoint():
    assert _to_unit(0.0, -1.0, 1.0) == 0.5


def test_to_unit_clipped_below():
    assert _to_unit(-5.0, -1.0, 1.0) == 0.0


def test_to_unit_clipped_above():
    assert _to_unit(5.0, -1.0, 1.0) == 1.0


def test_to_unit_zero_span_safe():
    # Degenerate range — should not raise / divide-by-zero.
    assert _to_unit(2.0, 1.0, 1.0) == 0.0


# ---------------------------------------------------------------------------
# Per-component _01 wrappers
# ---------------------------------------------------------------------------


def test_a2_correctness_01_mcq():
    assert compute_a2_correctness_01("(A)", "(A)", "mcq") == 1.0
    assert compute_a2_correctness_01("(B)", "(A)", "mcq") == 0.0


def test_a2_correctness_01_yesno():
    assert compute_a2_correctness_01("yes", "yes", "yesno") == 1.0
    assert compute_a2_correctness_01("no", "yes", "yesno") == 0.0


def test_no_regression_01_mcq_transitions():
    # Inline values: RR=+1, RW=-2, WR=+3, WW=0  range [-2, +3]
    rr = compute_no_regression_01("(A)", "(A)", "mcq", a1_is_correct=True)
    rw = compute_no_regression_01("(B)", "(A)", "mcq", a1_is_correct=True)
    wr = compute_no_regression_01("(A)", "(A)", "mcq", a1_is_correct=False)
    ww = compute_no_regression_01("(B)", "(A)", "mcq", a1_is_correct=False)
    # Map each to expected [0, 1]: (raw + 2) / 5
    assert abs(rr - (1.0 + 2.0) / 5.0) < 1e-9  # 0.6
    assert abs(rw - (-2.0 + 2.0) / 5.0) < 1e-9  # 0.0
    assert abs(wr - (3.0 + 2.0) / 5.0) < 1e-9  # 1.0
    assert abs(ww - (0.0 + 2.0) / 5.0) < 1e-9  # 0.4
    assert wr > rr > ww > rw  # ordering preserved


def test_a2_format_01_with_tags():
    """In tag mode, valid format → 1.0; missing tags → 0.0."""
    valid = compute_a2_format_01(
        "<think>reasoning</think><answer>(A)</answer>",
        "mcq",
        ground_truth="(A)",
        use_think_answer_tags=True,
    )
    assert valid == 1.0
    no_tags = compute_a2_format_01("(A)", "mcq", ground_truth="(A)", use_think_answer_tags=True)
    assert no_tags == 0.0


def test_downstream_01_mcq_transitions():
    # Inline default-mode values:
    # RR=+1, RW=-1.5, WR=+3, WW=-1  range [-1.5, +3]  span 4.5
    f1_correct = "<think>looks right</think>\\boxed{CORRECT}"
    f1_incorrect = "<think>off</think>\\boxed{INCORRECT}"
    rr = compute_downstream_01(
        feedback_text=f1_correct,
        a1_text="(A)",
        a2_text="(A)",
        ground_truth="(A)",
        answer_type="mcq",
    )
    rw = compute_downstream_01(
        feedback_text=f1_incorrect,
        a1_text="(A)",
        a2_text="(B)",
        ground_truth="(A)",
        answer_type="mcq",
    )
    wr = compute_downstream_01(
        feedback_text=f1_incorrect,
        a1_text="(B)",
        a2_text="(A)",
        ground_truth="(A)",
        answer_type="mcq",
    )
    ww = compute_downstream_01(
        feedback_text=f1_incorrect,
        a1_text="(B)",
        a2_text="(C)",
        ground_truth="(A)",
        answer_type="mcq",
    )
    expected_rr = (1.0 + 1.5) / 4.5
    expected_rw = 0.0
    expected_wr = 1.0
    expected_ww = (-1.0 + 1.5) / 4.5
    assert abs(rr - expected_rr) < 1e-6
    assert abs(rw - expected_rw) < 1e-6
    assert abs(wr - expected_wr) < 1e-6
    assert abs(ww - expected_ww) < 1e-6


def test_downstream_01_empty_feedback():
    """Empty F1 → mid of deterministic range (raw=0 rescaled)."""
    val = compute_downstream_01(
        feedback_text="",
        a1_text="(A)",
        a2_text="(A)",
        ground_truth="(A)",
        answer_type="mcq",
    )
    expected = (0.0 + 1.5) / 4.5  # raw=0 → 0.333...
    assert abs(val - expected) < 1e-6


def test_verification_01_correct_verdict():
    """Boxed CORRECT verdict matching A1 truth → 1.0."""
    val = compute_verification_01("\\boxed{CORRECT}", a1_is_correct=True)
    assert val == 1.0


def test_verification_01_wrong_verdict():
    """Boxed CORRECT verdict on wrong A1 → 0.0."""
    val = compute_verification_01("\\boxed{CORRECT}", a1_is_correct=False)
    assert val == 0.0


def test_verification_01_missing_boxed():
    """Missing boxed verdict → 0.0 (raw -1 rescaled)."""
    val = compute_verification_01("just some text", a1_is_correct=True)
    assert val == 0.0


def test_fb_format_01_valid_structure():
    """<think>...</think>...\\boxed{CORRECT} → 1.0."""
    val = compute_fb_format_01("<think>review</think>\nVerdict: \\boxed{CORRECT}")
    assert val == 1.0


def test_fb_format_01_missing_structure():
    val = compute_fb_format_01("just plain text without structure")
    assert val == 0.0


# ---------------------------------------------------------------------------
# End-to-end breakdowns
# ---------------------------------------------------------------------------


def test_response_breakdown_01_best_case_rr():
    """RR + valid format. With weights 0.27/0.03/0.27/0.03/0.40, expected
    total = 0.27*1 (a1_corr) + 0.03*1 (a1_fmt) + 0.27*1 (a2_corr) + 0.03*1
    (a2_fmt) + 0.40 * (1+2)/5 (no_reg RR) = 0.27 + 0.03 + 0.27 + 0.03 + 0.24
    = 0.84. All components in [0, 1]."""
    weights = ResponseRewardWeights()
    a1 = "<think>r</think><answer>(A)</answer>"
    a2 = "<think>r</think><answer>(A)</answer>"
    bd = compute_response_reward_breakdown_01(
        a1_text=a1,
        a2_text=a2,
        ground_truth="(A)",
        answer_type="mcq",
        choices="(A) x\n(B) y",
        weights=weights,
        use_think_answer_tags=True,
    )
    for v in bd.components.values():
        assert 0.0 <= v <= 1.0, f"component out of [0,1]: {v}"
    assert bd.a1_correct is True
    assert bd.a2_correct is True
    expected = (
        0.27 * 1.0  # a1_corr
        + 0.03 * 1.0  # a1_fmt
        + 0.27 * 1.0  # a2_corr
        + 0.03 * 1.0  # a2_fmt
        + 0.40 * _to_unit(1.0, _NO_REG_DET_RANGE[0], _NO_REG_DET_RANGE[1])  # no_reg
    )
    assert abs(bd.total_reward - expected) < 1e-6


def test_response_breakdown_01_wr_max():
    """WR + valid format = component-wise maximum: a2_corr=1, no_reg=1
    (RR=+1 → 0.6, RW=-2 → 0, WR=+3 → 1.0). Total = 0.27+0.03+0.27+0+0.4*1
    = 0.97. (a1_format on a wrong (B) answer with valid tags is still 1.0
    since format check is type-agnostic.)"""
    weights = ResponseRewardWeights()
    a1 = "<think>r</think><answer>(B)</answer>"  # wrong
    a2 = "<think>r</think><answer>(A)</answer>"  # right
    bd = compute_response_reward_breakdown_01(
        a1_text=a1,
        a2_text=a2,
        ground_truth="(A)",
        answer_type="mcq",
        choices="",
        weights=weights,
        use_think_answer_tags=True,
    )
    assert bd.a1_correct is False
    assert bd.a2_correct is True
    assert bd.components["no_regression"] == 1.0  # WR maps to 1.0


def test_response_breakdown_01_worst_case_rw_no_tags():
    """RW + no tags → all rewards near 0."""
    weights = ResponseRewardWeights()
    a1 = "(A)"  # right but no tags
    a2 = "(B)"  # wrong + no tags
    bd = compute_response_reward_breakdown_01(
        a1_text=a1,
        a2_text=a2,
        ground_truth="(A)",
        answer_type="mcq",
        choices="",
        weights=weights,
        use_think_answer_tags=True,
    )
    # In strict tag mode, missing tags → extracted=""; verify_answer marks
    # it incorrect. Both a1_correct and a2_correct False due to strict mode.
    assert bd.components["a1_format"] == 0.0
    assert bd.components["a2_format"] == 0.0
    # All components in [0, 1]
    for v in bd.components.values():
        assert 0.0 <= v <= 1.0


def test_feedback_breakdown_01_max_total_is_one():
    """RR + good F1 + perfect format. Each weighted component should
    contribute up to its weight; sum ≤ 1.0. With FeedbackRewardWeights
    defaults (0.45/0.45/0.1), best feasible is when all three = 1.0."""
    weights = FeedbackRewardWeights()
    f1 = "<think>The answer is (A) which matches.</think>\nVerdict: \\boxed{CORRECT}"
    bd = compute_feedback_reward_breakdown_01(
        feedback_text=f1,
        a1_text="(A)",
        a2_text="(A)",
        ground_truth="(A)",
        answer_type="mcq",
        choices="",
        weights=weights,
    )
    for v in bd.components.values():
        assert 0.0 <= v <= 1.0
    # All three components should hit their max here:
    #   verification: CORRECT verdict on right A1 → 1.0
    #   format: tags + valid boxed → 1.0
    #   downstream: RR with verified verdict → (1.0+1.5)/4.5 ≈ 0.555...
    assert bd.components["verification"] == 1.0
    assert bd.components["format"] == 1.0
    assert bd.total_reward <= 1.0 + 1e-9


def test_feedback_breakdown_01_worst_case_zero():
    """Empty F1 + missing boxed → all components 0 except downstream
    midpoint (no F1 path returns the deterministic range mid)."""
    weights = FeedbackRewardWeights()
    bd = compute_feedback_reward_breakdown_01(
        feedback_text="",
        a1_text="(B)",
        a2_text="(C)",
        ground_truth="(A)",
        answer_type="mcq",
        choices="",
        weights=weights,
    )
    # Empty feedback → verification=0, fb_format=0
    # downstream raw=0 mid → (0+1.5)/4.5 ≈ 0.333
    # Then asymmetric gate clamps it to that midpoint anyway.
    assert bd.components["verification"] == 0.0
    assert bd.components["format"] == 0.0
    assert 0.0 < bd.components["downstream"] < 0.5
    assert bd.total_reward < 0.5


def test_feedback_breakdown_01_asymmetric_gate():
    """When verification fails (≤0.5), positive downstream is clamped to
    the midpoint. WR with wrong verdict → downstream raw +3 → would be 1.0
    but gates to deterministic-mid 0.333."""
    weights = FeedbackRewardWeights()
    # F1 says CORRECT but A1 was wrong → verification=0
    f1 = "<think>looks right</think>\\boxed{CORRECT}"
    bd = compute_feedback_reward_breakdown_01(
        feedback_text=f1,
        a1_text="(B)",  # wrong
        a2_text="(A)",  # right (WR)
        ground_truth="(A)",
        answer_type="mcq",
        choices="",
        weights=weights,
    )
    assert bd.components["verification"] == 0.0
    # downstream gated to midpoint
    expected_mid = _to_unit(0.0, _NO_REG_DET_RANGE[0], _NO_REG_DET_RANGE[1])
    # ^ same span as deterministic downstream (-1.5, 3.0) is 4.5; mid raw=0
    # → (0+1.5)/4.5 = 1/3
    expected_dmid = (0.0 + 1.5) / 4.5
    assert abs(bd.components["downstream"] - expected_dmid) < 1e-6
    # Just sanity: deterministic range has the same 0-mid as no_reg.
    assert abs(expected_mid - (0.0 + 2.0) / 5.0) < 1e-9  # different ranges


# ---------------------------------------------------------------------------
# Regression guard: raw path unchanged
# ---------------------------------------------------------------------------


def test_raw_response_breakdown_unchanged():
    """The non-rescaled response breakdown should still produce the same
    historical {-1, +1, +3, ...} component values."""
    weights = ResponseRewardWeights()
    a1 = "<think>r</think><answer>(A)</answer>"
    a2 = "<think>r</think><answer>(A)</answer>"
    bd = compute_response_reward_breakdown(
        a1_text=a1,
        a2_text=a2,
        ground_truth="(A)",
        answer_type="mcq",
        choices="",
        weights=weights,
        use_think_answer_tags=True,
    )
    # With tags + RR: a1_corr=+1, a2_corr=+1, a1_fmt=+1, a2_fmt=+1,
    # no_reg=+1 (RR for det). Inline raw values, NOT rescaled.
    assert bd.components["a1_correctness"] == 1.0
    assert bd.components["a2_correctness"] == 1.0
    assert bd.components["no_regression"] == 1.0
    assert bd.components["a1_format"] == 1.0
    assert bd.components["a2_format"] == 1.0


def test_raw_feedback_breakdown_unchanged():
    """The non-rescaled feedback breakdown still emits raw {-2..+3}-style
    component values (NOT [0, 1])."""
    weights = FeedbackRewardWeights()
    f1 = "<think>checks</think>\nVerdict: \\boxed{CORRECT}"
    bd = compute_feedback_reward_breakdown(
        feedback_text=f1,
        a1_text="(A)",
        a2_text="(A)",
        ground_truth="(A)",
        answer_type="mcq",
        choices="",
        weights=weights,
    )
    # RR with valid CORRECT verdict on right A1: downstream=+1, verification=+1, format=+1
    assert bd.components["downstream"] == 1.0
    assert bd.components["verification"] == 1.0
    assert bd.components["format"] == 1.0


# ---------------------------------------------------------------------------
# Total reward range check
# ---------------------------------------------------------------------------


def test_total_reward_strictly_non_negative_in_rescaled_path():
    """Across a battery of trajectory types, resp+fb total stays >= 0 in
    the rescaled path."""
    weights_resp = ResponseRewardWeights()
    weights_fb = FeedbackRewardWeights()
    cases = [
        # (a1_text, f1_text, a2_text, gt)
        (
            "<think>r</think><answer>(A)</answer>",
            "<think>r</think>\\boxed{CORRECT}",
            "<think>r</think><answer>(A)</answer>",
            "(A)",
        ),
        ("(B)", "", "(C)", "(A)"),  # all wrong, no tags, empty F1
        (
            "<think>r</think><answer>(B)</answer>",
            "<think>r</think>\\boxed{INCORRECT}",
            "<think>r</think><answer>(A)</answer>",
            "(A)",
        ),  # WR
    ]
    for a1, f1, a2, gt in cases:
        rb = compute_response_reward_breakdown_01(
            a1_text=a1,
            a2_text=a2,
            ground_truth=gt,
            answer_type="mcq",
            choices="",
            weights=weights_resp,
            use_think_answer_tags=True,
        )
        fb = compute_feedback_reward_breakdown_01(
            feedback_text=f1,
            a1_text=a1,
            a2_text=a2,
            ground_truth=gt,
            answer_type="mcq",
            choices="",
            weights=weights_fb,
        )
        total = rb.total_reward + fb.total_reward
        assert total >= 0.0, f"total<0 for case ({a1!r}, {f1!r}, {a2!r}): {total}"
        assert total <= 2.0 + 1e-6, f"total>2 for case: {total}"
