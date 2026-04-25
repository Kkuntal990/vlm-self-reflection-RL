#!/usr/bin/env python3
"""Tests for stability reward functions."""

from vlm_grpo.rewards.stability import compute_no_regression_reward


class TestNoRegressionReward:
    """Tests for compute_no_regression_reward."""

    def test_rr_a1_correct(self) -> None:
        """A1 correct, A2 correct → RR → +1.0."""
        r = compute_no_regression_reward("A", "A", "mcq", a1_is_correct=True)
        assert r == 1.0

    def test_rw_a1_correct(self) -> None:
        """A1 correct, A2 wrong → RW → -2.0 (deterministic type)."""
        r = compute_no_regression_reward("B", "A", "mcq", a1_is_correct=True)
        assert r == -2.0

    def test_wr_a1_wrong(self) -> None:
        """A1 wrong, A2 correct → WR → +3.0 (deterministic type)."""
        r = compute_no_regression_reward("A", "A", "mcq", a1_is_correct=False)
        assert r == 3.0

    def test_ww_a1_wrong(self) -> None:
        """A1 wrong, A2 wrong → WW → -0.5 (small penalty for "stable wrong")."""
        r = compute_no_regression_reward("B", "A", "mcq", a1_is_correct=False)
        assert r == -0.5

    def test_ww_open(self) -> None:
        """Open-ended WW also -0.5."""
        r = compute_no_regression_reward("wrong", "cat", "open", a1_is_correct=False)
        assert r == -0.5

    def test_yesno_rw(self) -> None:
        """YesNo is deterministic type → RW = -2.0."""
        r = compute_no_regression_reward("No", "Yes", "yesno", a1_is_correct=True)
        assert r == -2.0

    def test_numeric_rr(self) -> None:
        r = compute_no_regression_reward("3.14", "3.14", "numeric", a1_is_correct=True)
        assert r == 1.0

    def test_empty_mcq_is_rw(self) -> None:
        """Empty MCQ → parse failure → WRONG → RW → -2.0 (deterministic type)."""
        r = compute_no_regression_reward("", "A", "mcq", a1_is_correct=True)
        assert r == -2.0

    def test_open_rw_keeps_heavy_penalty(self) -> None:
        """Open-ended RW keeps -3.0 (large answer space)."""
        r = compute_no_regression_reward("wrong", "cat", "open", a1_is_correct=True)
        assert r == -3.0

    def test_open_wr_keeps_original_reward(self) -> None:
        """Open-ended WR keeps +2.0."""
        r = compute_no_regression_reward("cat", "cat", "open", a1_is_correct=False)
        assert r == 2.0
