#!/usr/bin/env python3
"""Tests for correctness reward functions."""

from vlm_grpo.rewards.correctness import (
    compute_a2_correctness_reward,
    compute_downstream_improvement_reward,
)

# =============================================================================
# compute_a2_correctness_reward
# =============================================================================


class TestA2CorrectnessReward:
    """Tests for compute_a2_correctness_reward."""

    def test_mcq_correct(self) -> None:
        r = compute_a2_correctness_reward("A", "A", "mcq")
        assert r == 1.0

    def test_mcq_incorrect(self) -> None:
        r = compute_a2_correctness_reward("B", "A", "mcq")
        assert r == -1.0

    def test_yesno_correct(self) -> None:
        r = compute_a2_correctness_reward("Yes", "yes", "yesno")
        assert r == 1.0

    def test_yesno_incorrect(self) -> None:
        r = compute_a2_correctness_reward("No", "yes", "yesno")
        assert r == -1.0

    def test_numeric_correct(self) -> None:
        r = compute_a2_correctness_reward("3.14", "3.14", "numeric")
        assert r == 1.0

    def test_numeric_within_tolerance(self) -> None:
        r = compute_a2_correctness_reward("3.14", "3.15", "numeric", tolerance=0.01)
        assert r == 1.0

    def test_numeric_incorrect(self) -> None:
        r = compute_a2_correctness_reward("5", "3.14", "numeric")
        assert r == -1.0

    def test_open_exact_match(self) -> None:
        r = compute_a2_correctness_reward("cat", "cat", "open")
        assert r == 1.0

    def test_open_mismatch_returns_negative(self) -> None:
        """Unrelated open-ended answer → low score → negative continuous reward."""
        r = compute_a2_correctness_reward("airplane", "cat", "open")
        assert r < 0  # Continuous: low similarity maps to negative reward

    def test_open_substring_returns_high(self) -> None:
        """Substring match → high score → near +1.0 continuous reward."""
        r = compute_a2_correctness_reward("a domestic cat", "cat", "open")
        assert r > 0.8  # Continuous: score=0.95 → reward=0.9

    def test_empty_mcq_a2_returns_wrong(self) -> None:
        """Empty MCQ → parse failure → WRONG → -1.0."""
        r = compute_a2_correctness_reward("", "A", "mcq")
        assert r == -1.0


# =============================================================================
# compute_downstream_improvement_reward
# =============================================================================


class TestDownstreamImprovementReward:
    """Tests for compute_downstream_improvement_reward."""

    def test_rr_transition(self) -> None:
        """A1 correct, A2 correct → RR → +1.0."""
        r = compute_downstream_improvement_reward(
            a1="A",
            a2_extracted="A",
            ground_truth="A",
            answer_type="mcq",
            a1_is_correct=True,
        )
        assert r == 1.0

    def test_rw_transition(self) -> None:
        """A1 correct, A2 wrong → RW → -2.0."""
        r = compute_downstream_improvement_reward(
            a1="A",
            a2_extracted="B",
            ground_truth="A",
            answer_type="mcq",
            a1_is_correct=True,
        )
        assert r == -2.0

    def test_wr_transition(self) -> None:
        """A1 wrong, A2 correct → WR → +2.0."""
        r = compute_downstream_improvement_reward(
            a1="B",
            a2_extracted="A",
            ground_truth="A",
            answer_type="mcq",
            a1_is_correct=False,
        )
        assert r == 2.0

    def test_ww_transition(self) -> None:
        """A1 wrong, A2 wrong → WW → -0.5."""
        r = compute_downstream_improvement_reward(
            a1="B",
            a2_extracted="C",
            ground_truth="A",
            answer_type="mcq",
            a1_is_correct=False,
        )
        assert r == -0.5

    def test_yesno_rw(self) -> None:
        """YesNo RW transition."""
        r = compute_downstream_improvement_reward(
            a1="Yes",
            a2_extracted="No",
            ground_truth="Yes",
            answer_type="yesno",
            a1_is_correct=True,
        )
        assert r == -2.0

    def test_numeric_wr(self) -> None:
        """Numeric WR transition."""
        r = compute_downstream_improvement_reward(
            a1="5",
            a2_extracted="3.14",
            ground_truth="3.14",
            answer_type="numeric",
            a1_is_correct=False,
        )
        assert r == 2.0

    def test_open_mismatch_downstream(self) -> None:
        """Open-ended mismatch → WRONG → WW → -0.5."""
        r = compute_downstream_improvement_reward(
            a1="a dog",
            a2_extracted="airplane",
            ground_truth="cat",
            answer_type="open",
            a1_is_correct=False,
        )
        assert r == -0.5
