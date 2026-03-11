#!/usr/bin/env python3
"""Tests for two-trajectory reward composition."""

from vlm_grpo.rewards.composition import (
    CriticRewardWeights,
    RefinerRewardWeights,
    compute_critic_reward_breakdown,
    compute_refiner_reward_breakdown,
    get_refiner_reward_functions,
    refiner_correctness_reward_fn,
    refiner_format_reward_fn,
    refiner_minimal_edit_reward_fn,
    refiner_no_regression_reward_fn,
)

# =============================================================================
# Weight Dataclasses
# =============================================================================


class TestCriticRewardWeights:
    """Tests for CriticRewardWeights."""

    def test_defaults(self) -> None:
        w = CriticRewardWeights()
        assert w.w_downstream == 2.0
        assert w.w_calibration == 1.0
        assert w.w_format == 0.5

    def test_to_dict(self) -> None:
        w = CriticRewardWeights()
        d = w.to_dict()
        assert "w_downstream" in d
        assert d["w_downstream"] == 2.0

    def test_to_list(self) -> None:
        w = CriticRewardWeights()
        lst = w.to_list()
        assert len(lst) == 3
        assert lst == [0.5, 1.0, 2.0]  # [format, calibration, downstream]

    def test_custom_weights(self) -> None:
        w = CriticRewardWeights(w_downstream=3.0, w_calibration=0.5, w_format=0.1)
        assert w.w_downstream == 3.0


class TestRefinerRewardWeights:
    """Tests for RefinerRewardWeights."""

    def test_defaults(self) -> None:
        w = RefinerRewardWeights()
        assert w.w_correctness == 1.0
        assert w.w_no_regression == 2.0
        assert w.w_minimal_edit == 0.3
        assert w.w_format == 0.5

    def test_to_list(self) -> None:
        w = RefinerRewardWeights()
        lst = w.to_list()
        assert len(lst) == 4
        assert lst == [0.5, 1.0, 2.0, 0.3]


# =============================================================================
# Critic Reward Breakdown
# =============================================================================


class TestCriticRewardBreakdown:
    """Tests for compute_critic_reward_breakdown."""

    def test_perfect_rr(self) -> None:
        """Perfect scenario: calibrated feedback, A2 correct, A1 correct."""
        bd = compute_critic_reward_breakdown(
            feedback_text="The answer is correct and matches the image.",
            a2_text="A",
            ground_truth="A",
            answer1="A",
            a1_is_correct=True,
            answer_type="mcq",
            choices="",
            weights=CriticRewardWeights(),
        )
        assert bd.a2_correct is True
        assert bd.format_valid is True
        assert bd.components["downstream"] == 1.0
        assert bd.components["calibration"] == 1.0
        assert bd.total_reward > 0

    def test_rw_regression(self) -> None:
        """Bad feedback causes RW regression."""
        bd = compute_critic_reward_breakdown(
            feedback_text="The answer is wrong, it should be B.",
            a2_text="B",
            ground_truth="A",
            answer1="A",
            a1_is_correct=True,
            answer_type="mcq",
            choices="",
            weights=CriticRewardWeights(),
        )
        assert bd.a2_correct is False
        assert bd.components["downstream"] == -2.0
        assert bd.total_reward < 0

    def test_empty_feedback_penalized(self) -> None:
        """Empty feedback gets heavy format penalty (-2.0)."""
        bd = compute_critic_reward_breakdown(
            feedback_text="",
            a2_text="A",
            ground_truth="A",
            answer1="A",
            a1_is_correct=True,
            answer_type="mcq",
            choices="",
            weights=CriticRewardWeights(),
        )
        assert bd.format_valid is False
        assert bd.components["format"] == -2.0

    def test_single_word_minus_two(self) -> None:
        """Single-word feedback (<3 words) → -2.0."""
        bd = compute_critic_reward_breakdown(
            feedback_text="OK",
            a2_text="A",
            ground_truth="A",
            answer1="A",
            a1_is_correct=True,
            answer_type="mcq",
            choices="",
            weights=CriticRewardWeights(),
        )
        assert bd.components["format"] == -2.0

    def test_stance_keyword_plus_one(self) -> None:
        """3-word feedback with stance keyword → +1.0."""
        bd = compute_critic_reward_breakdown(
            feedback_text="Answer is correct.",
            a2_text="A",
            ground_truth="A",
            answer1="A",
            a1_is_correct=True,
            answer_type="mcq",
            choices="",
            weights=CriticRewardWeights(),
        )
        assert bd.components["format"] == 1.0

    def test_weak_feedback_minus_one(self) -> None:
        """3-4 words, no stance keyword → -1.0."""
        bd = compute_critic_reward_breakdown(
            feedback_text="Maybe try again.",
            a2_text="A",
            ground_truth="A",
            answer1="A",
            a1_is_correct=True,
            answer_type="mcq",
            choices="",
            weights=CriticRewardWeights(),
        )
        assert bd.components["format"] == -1.0

    def test_five_words_plus_one(self) -> None:
        """>=5 words even without stance keyword → +1.0."""
        bd = compute_critic_reward_breakdown(
            feedback_text="I think you might want to reconsider.",
            a2_text="A",
            ground_truth="A",
            answer1="A",
            a1_is_correct=True,
            answer_type="mcq",
            choices="",
            weights=CriticRewardWeights(),
        )
        assert bd.components["format"] == 1.0

    def test_to_dict(self) -> None:
        bd = compute_critic_reward_breakdown(
            feedback_text="The answer looks good.",
            a2_text="A",
            ground_truth="A",
            answer1="A",
            a1_is_correct=True,
            answer_type="mcq",
            choices="",
            weights=CriticRewardWeights(),
        )
        d = bd.to_dict()
        assert "total_reward" in d
        assert "components" in d


# =============================================================================
# Refiner Reward Breakdown
# =============================================================================


class TestRefinerRewardBreakdown:
    """Tests for compute_refiner_reward_breakdown."""

    def test_perfect_rr(self) -> None:
        """Perfect refiner: A2 correct, same as A1."""
        bd = compute_refiner_reward_breakdown(
            a2_text="A",
            ground_truth="A",
            answer1="A",
            a1_is_correct=True,
            answer_type="mcq",
            choices="",
            weights=RefinerRewardWeights(),
        )
        assert bd.a2_correct is True
        assert bd.components["correctness"] == 1.0
        assert bd.components["no_regression"] == 1.0
        assert bd.total_reward > 0

    def test_rw_regression(self) -> None:
        """A2 wrong → heavy penalty."""
        bd = compute_refiner_reward_breakdown(
            a2_text="B",
            ground_truth="A",
            answer1="A",
            a1_is_correct=True,
            answer_type="mcq",
            choices="",
            weights=RefinerRewardWeights(),
        )
        assert bd.a2_correct is False
        assert bd.components["no_regression"] == -3.0
        assert bd.total_reward < 0

    def test_to_dict(self) -> None:
        bd = compute_refiner_reward_breakdown(
            a2_text="A",
            ground_truth="A",
            answer1="A",
            a1_is_correct=True,
            answer_type="mcq",
            choices="",
            weights=RefinerRewardWeights(),
        )
        d = bd.to_dict()
        assert "total_reward" in d


# =============================================================================
# TRL-Compatible Reward Functions
# =============================================================================


class TestRefinerTRLFunctions:
    """Tests for TRL-compatible refiner reward functions."""

    def test_format_reward_fn(self) -> None:
        rewards = refiner_format_reward_fn(
            completions=["A", "B", ""],
            answer_type=["mcq", "mcq", "mcq"],
            choices=["", "", ""],
        )
        assert len(rewards) == 3
        assert rewards[0] == 0.0  # Valid MCQ (penalty-only: 0.0 = compliant)
        assert rewards[1] == 0.0  # Valid MCQ (penalty-only: 0.0 = compliant)
        assert rewards[2] == -1.0  # Empty

    def test_correctness_reward_fn(self) -> None:
        rewards = refiner_correctness_reward_fn(
            completions=["A", "B"],
            ground_truth=["A", "A"],
            answer_type=["mcq", "mcq"],
            choices=["", ""],
        )
        assert len(rewards) == 2
        assert rewards[0] == 1.0  # Correct
        assert rewards[1] == -1.0  # Incorrect

    def test_no_regression_reward_fn(self) -> None:
        rewards = refiner_no_regression_reward_fn(
            completions=["A", "B"],
            ground_truth=["A", "A"],
            answer_type=["mcq", "mcq"],
            choices=["", ""],
            a1_is_correct=[True, True],
        )
        assert len(rewards) == 2
        assert rewards[0] == 1.0  # RR
        assert rewards[1] == -3.0  # RW

    def test_minimal_edit_reward_fn(self) -> None:
        rewards = refiner_minimal_edit_reward_fn(
            completions=["A", "B"],
            ground_truth=["A", "A"],
            answer1=["A", "A"],
            answer_type=["mcq", "mcq"],
            choices=["", ""],
        )
        assert len(rewards) == 2
        assert rewards[0] == 1.0  # Identical correct answers
        assert rewards[1] == 0.0  # A2 wrong, no edit reward

    def test_get_refiner_reward_functions(self) -> None:
        fns = get_refiner_reward_functions()
        assert len(fns) == 4
        assert callable(fns[0])
        assert callable(fns[1])
        assert callable(fns[2])
        assert callable(fns[3])

    def test_reward_functions_handle_none_kwargs(self) -> None:
        """Reward functions should handle None gracefully."""
        rewards = refiner_correctness_reward_fn(
            completions=["A"],
            ground_truth=None,
            answer_type=None,
        )
        assert len(rewards) == 1
