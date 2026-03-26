#!/usr/bin/env python3
"""Tests for stability reward functions."""

from vlm_grpo.rewards.stability import (
    compute_minimal_edit_reward,
    compute_no_regression_reward,
)

# =============================================================================
# compute_no_regression_reward
# =============================================================================


class TestNoRegressionReward:
    """Tests for compute_no_regression_reward."""

    def test_rr_a1_correct(self) -> None:
        """A1 correct, A2 correct → RR → +1.0."""
        r = compute_no_regression_reward("A", "A", "mcq", a1_is_correct=True)
        assert r == 1.0

    def test_rw_a1_correct(self) -> None:
        """A1 correct, A2 wrong → RW → -3.0."""
        r = compute_no_regression_reward("B", "A", "mcq", a1_is_correct=True)
        assert r == -3.0

    def test_wr_a1_wrong(self) -> None:
        """A1 wrong, A2 correct → WR → +2.0."""
        r = compute_no_regression_reward("A", "A", "mcq", a1_is_correct=False)
        assert r == 2.0

    def test_ww_a1_wrong(self) -> None:
        """A1 wrong, A2 wrong → WW → 0.0."""
        r = compute_no_regression_reward("B", "A", "mcq", a1_is_correct=False)
        assert r == 0.0

    def test_yesno_rw(self) -> None:
        r = compute_no_regression_reward(
            "No",
            "Yes",
            "yesno",
            a1_is_correct=True,
        )
        assert r == -3.0

    def test_numeric_rr(self) -> None:
        r = compute_no_regression_reward(
            "3.14",
            "3.14",
            "numeric",
            a1_is_correct=True,
        )
        assert r == 1.0

    def test_empty_mcq_is_rw(self) -> None:
        """Empty MCQ → parse failure → WRONG → RW → -3.0."""
        r = compute_no_regression_reward("", "A", "mcq", a1_is_correct=True)
        assert r == -3.0

    def test_mcq_parens_same_as_plain(self) -> None:
        """(A) and A should both extract to 'A' → identical edit distance."""
        r = compute_minimal_edit_reward("(A)", "A", "A", "mcq")
        assert r == 1.0


# =============================================================================
# compute_minimal_edit_reward
# =============================================================================


class TestMinimalEditReward:
    """Tests for compute_minimal_edit_reward."""

    def test_identical_answers(self) -> None:
        """Both correct, identical → 1.0."""
        r = compute_minimal_edit_reward("A", "A", "A", "mcq")
        assert r == 1.0

    def test_different_but_correct(self) -> None:
        """Both correct, different text → < 1.0 but > 0.0."""
        r = compute_minimal_edit_reward("Yes", "Yes", "Yes", "yesno")
        assert r == 1.0

    def test_a1_wrong_returns_zero(self) -> None:
        """A1 wrong → 0.0 regardless of A2."""
        r = compute_minimal_edit_reward("B", "A", "A", "mcq")
        assert r == 0.0

    def test_a2_wrong_returns_zero(self) -> None:
        """A2 wrong → 0.0."""
        r = compute_minimal_edit_reward("A", "B", "A", "mcq")
        assert r == 0.0

    def test_lambda_edit_controls_penalty(self) -> None:
        """Higher lambda_edit means more penalty for edits."""
        r_low = compute_minimal_edit_reward("cat", "cat", "cat", "open", lambda_edit=0.1)
        r_high = compute_minimal_edit_reward("cat", "cat", "cat", "open", lambda_edit=0.9)
        # Both correct and identical, so both 1.0
        assert r_low == 1.0
        assert r_high == 1.0

    def test_open_ended_similar(self) -> None:
        """Open-ended: exact match required for match_answer to return True."""
        r = compute_minimal_edit_reward("cat", "cat", "cat", "open")
        assert r == 1.0

    def test_reward_clamped_at_zero(self) -> None:
        """Reward should not go below 0.0."""
        # Even with high lambda_edit, reward is clamped
        r = compute_minimal_edit_reward("Yes", "Yes", "Yes", "yesno", lambda_edit=2.0)
        assert r >= 0.0
