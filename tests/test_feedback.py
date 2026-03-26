#!/usr/bin/env python3
"""Tests for feedback reward functions."""

from vlm_grpo.rewards.feedback import (
    compute_downstream_aware_reward,
    compute_feedback_calibration_reward,
)

# =============================================================================
# compute_feedback_calibration_reward
# =============================================================================


class TestFeedbackCalibrationReward:
    """Tests for compute_feedback_calibration_reward."""

    # A1 correct: positive feedback should be rewarded
    def test_a1_correct_positive_feedback(self) -> None:
        r = compute_feedback_calibration_reward(
            "The answer is correct and matches the image.", a1_is_correct=True
        )
        assert r == 1.0

    def test_a1_correct_negative_feedback(self) -> None:
        r = compute_feedback_calibration_reward(
            "The answer is incorrect and should be changed.", a1_is_correct=True
        )
        assert r == -1.0

    # A1 wrong: negative feedback should be rewarded
    def test_a1_wrong_negative_feedback(self) -> None:
        r = compute_feedback_calibration_reward(
            "The answer is wrong and needs to be corrected.", a1_is_correct=False
        )
        assert r == 1.0

    def test_a1_wrong_positive_feedback(self) -> None:
        r = compute_feedback_calibration_reward(
            "The answer is correct. Well done!", a1_is_correct=False
        )
        assert r == -1.0

    # Neutral
    def test_empty_feedback(self) -> None:
        r = compute_feedback_calibration_reward("", a1_is_correct=True)
        assert r == 0.0

    def test_neutral_feedback(self) -> None:
        r = compute_feedback_calibration_reward(
            "The image shows a cat on a table.", a1_is_correct=True
        )
        assert r == 0.0

    # Mixed feedback
    def test_a1_correct_mixed_positive(self) -> None:
        r = compute_feedback_calibration_reward(
            "The answer is mostly correct but has a minor error.",
            a1_is_correct=True,
        )
        # Has both "correct" and "error" → depends on counts
        assert -1.0 <= r <= 1.0

    def test_no_change_needed(self) -> None:
        r = compute_feedback_calibration_reward(
            "No change needed, the answer is fine.", a1_is_correct=True
        )
        assert r > 0

    def test_should_be_changed(self) -> None:
        r = compute_feedback_calibration_reward("The answer should be revised.", a1_is_correct=True)
        assert r < 0

    def test_soft_doubt_a1_correct(self) -> None:
        """Soft doubt on correct answer → -0.3."""
        r = compute_feedback_calibration_reward(
            "You might want to double-check this.", a1_is_correct=True
        )
        assert r == -0.3

    def test_soft_doubt_a1_wrong(self) -> None:
        """Soft doubt on wrong answer → +0.3."""
        r = compute_feedback_calibration_reward(
            "You might want to double-check this.", a1_is_correct=False
        )
        assert r == 0.3


# =============================================================================
# compute_downstream_aware_reward
# =============================================================================


class TestDownstreamAwareReward:
    """Tests for compute_downstream_aware_reward."""

    # rw_first phase (A1 correct)
    def test_rr_a1_correct(self) -> None:
        """Good feedback: A1 correct → A2 correct (RR) → +1.0."""
        r = compute_downstream_aware_reward(
            feedback_text="The answer is correct.",
            a2_extracted="A",
            ground_truth="A",
            answer_type="mcq",
            a1="A",
            a1_is_correct=True,
        )
        assert r == 1.0

    def test_rw_a1_correct(self) -> None:
        """Bad feedback: A1 correct → A2 wrong (RW) → -2.0."""
        r = compute_downstream_aware_reward(
            feedback_text="The answer is wrong, it should be B.",
            a2_extracted="B",
            ground_truth="A",
            answer_type="mcq",
            a1="A",
            a1_is_correct=True,
        )
        assert r == -2.0

    # full phase (A1 wrong)
    def test_wr_a1_wrong(self) -> None:
        """Great feedback: A1 wrong → A2 correct (WR) → +3.0."""
        r = compute_downstream_aware_reward(
            feedback_text="The answer is incorrect, it should be A.",
            a2_extracted="A",
            ground_truth="A",
            answer_type="mcq",
            a1="B",
            a1_is_correct=False,
        )
        assert r == 2.0

    def test_ww_a1_wrong(self) -> None:
        """Failed feedback: A1 wrong → A2 wrong (WW) → -1.0."""
        r = compute_downstream_aware_reward(
            feedback_text="The answer might need some adjustment.",
            a2_extracted="C",
            ground_truth="A",
            answer_type="mcq",
            a1="B",
            a1_is_correct=False,
        )
        assert r == -1.0

    # Edge cases
    def test_empty_feedback(self) -> None:
        r = compute_downstream_aware_reward(
            feedback_text="",
            a2_extracted="A",
            ground_truth="A",
            answer_type="mcq",
            a1="A",
            a1_is_correct=True,
        )
        assert r == 0.0

    def test_open_synonym_rr(self) -> None:
        """Open-ended synonym match → CORRECT → RR → +1.0."""
        r = compute_downstream_aware_reward(
            feedback_text="Some feedback.",
            a2_extracted="automobile",
            ground_truth="car",
            answer_type="open",
            a1="car",
            a1_is_correct=True,
        )
        assert r == 1.0

    def test_yesno_wr(self) -> None:
        r = compute_downstream_aware_reward(
            feedback_text="The answer should be Yes.",
            a2_extracted="Yes",
            ground_truth="Yes",
            answer_type="yesno",
            a1="No",
            a1_is_correct=False,
        )
        assert r == 2.0

    def test_numeric_rr(self) -> None:
        r = compute_downstream_aware_reward(
            feedback_text="The calculation is correct.",
            a2_extracted="3.14",
            ground_truth="3.14",
            answer_type="numeric",
            a1="3.14",
            a1_is_correct=True,
        )
        assert r == 1.0
