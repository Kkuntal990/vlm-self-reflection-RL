#!/usr/bin/env python3
"""Tests for feedback reward functions."""

from vlm_grpo.rewards.feedback import compute_downstream_aware_reward


class TestDownstreamAwareReward:
    """Tests for compute_downstream_aware_reward."""

    def test_rr_a1_correct(self) -> None:
        """Good feedback: A1 correct → A2 correct (RR) → +3.0 (deterministic;
        raised to match WR so the feedback head is also tied between RR and WR)."""
        r = compute_downstream_aware_reward(
            feedback_text="CORRECT. Matches the image.",
            a2_extracted="A",
            ground_truth="A",
            answer_type="mcq",
            a1="A",
            a1_is_correct=True,
        )
        assert r == 3.0

    def test_rw_a1_correct(self) -> None:
        """Bad feedback: A1 correct → A2 wrong (RW) → -1.5 (deterministic)."""
        r = compute_downstream_aware_reward(
            feedback_text="INCORRECT. Should be B.",
            a2_extracted="B",
            ground_truth="A",
            answer_type="mcq",
            a1="A",
            a1_is_correct=True,
        )
        assert r == -1.5

    def test_wr_a1_wrong(self) -> None:
        """Great feedback: A1 wrong → A2 correct (WR) → +3.0 (deterministic)."""
        r = compute_downstream_aware_reward(
            feedback_text="INCORRECT. Should be A.",
            a2_extracted="A",
            ground_truth="A",
            answer_type="mcq",
            a1="B",
            a1_is_correct=False,
        )
        assert r == 3.0

    def test_ww_a1_wrong(self) -> None:
        """Failed feedback: A1 wrong → A2 wrong (WW) → -1.0."""
        r = compute_downstream_aware_reward(
            feedback_text="CORRECT. Looks fine.",
            a2_extracted="C",
            ground_truth="A",
            answer_type="mcq",
            a1="B",
            a1_is_correct=False,
        )
        assert r == -1.0

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
        """Open-ended synonym match → CORRECT → RR → +2.0 (raised to match WR)."""
        r = compute_downstream_aware_reward(
            feedback_text="Some feedback.",
            a2_extracted="automobile",
            ground_truth="car",
            answer_type="open",
            a1="car",
            a1_is_correct=True,
        )
        assert r == 2.0

    def test_yesno_wr(self) -> None:
        """YesNo is deterministic type → WR = +3.0."""
        r = compute_downstream_aware_reward(
            feedback_text="INCORRECT. Should be Yes.",
            a2_extracted="Yes",
            ground_truth="Yes",
            answer_type="yesno",
            a1="No",
            a1_is_correct=False,
        )
        assert r == 3.0

    def test_numeric_rr(self) -> None:
        """Numeric is deterministic → RR = +3.0 (raised to match WR for tie)."""
        r = compute_downstream_aware_reward(
            feedback_text="CORRECT.",
            a2_extracted="3.14",
            ground_truth="3.14",
            answer_type="numeric",
            a1="3.14",
            a1_is_correct=True,
        )
        assert r == 3.0
