#!/usr/bin/env python3
"""Tests for two-trajectory reward composition."""

from vlm_grpo.rewards.composition import (
    CriticRewardWeights,
    RefinerRewardWeights,
    _compute_refiner_format_reward,
    compute_critic_reward_breakdown,
    compute_refiner_reward_breakdown,
    get_refiner_reward_functions,
    refiner_correctness_reward_fn,
    refiner_format_reward_fn,
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

    def test_to_dict(self) -> None:
        w = CriticRewardWeights()
        d = w.to_dict()
        assert "w_downstream" in d
        assert d["w_downstream"] == 2.0

    def test_to_list(self) -> None:
        w = CriticRewardWeights()
        lst = w.to_list()
        assert len(lst) == 1
        assert lst == [2.0]

    def test_custom_weights(self) -> None:
        w = CriticRewardWeights(w_downstream=3.0)
        assert w.w_downstream == 3.0


class TestRefinerRewardWeights:
    """Tests for RefinerRewardWeights."""

    def test_defaults(self) -> None:
        w = RefinerRewardWeights()
        assert w.w_correctness == 1.0
        assert w.w_no_regression == 2.0
        assert w.w_format == 0.15

    def test_to_list(self) -> None:
        w = RefinerRewardWeights()
        lst = w.to_list()
        assert len(lst) == 3
        assert lst == [0.15, 1.0, 2.0]


# =============================================================================
# Critic Reward Breakdown
# =============================================================================


class TestCriticRewardBreakdown:
    """Tests for compute_critic_reward_breakdown."""

    def test_perfect_rr(self) -> None:
        """RR: calibrated feedback, A2 correct, A1 correct."""
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
        assert bd.components["downstream"] == 3.0
        assert "format" not in bd.components
        assert "calibration" not in bd.components
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
        assert bd.components["downstream"] == -1.5  # deterministic type
        assert bd.total_reward < 0

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
        assert "minimal_edit" not in bd.components
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
        assert bd.components["no_regression"] == -2.0  # deterministic type
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
            ground_truth=["A", "B", "A"],
        )
        assert len(rewards) == 3
        assert rewards[0] == 0.0
        assert rewards[1] == 0.0
        assert rewards[2] == -1.0

    def test_correctness_reward_fn(self) -> None:
        rewards = refiner_correctness_reward_fn(
            completions=["A", "B"],
            ground_truth=["A", "A"],
            answer_type=["mcq", "mcq"],
            choices=["", ""],
        )
        assert len(rewards) == 2
        assert rewards[0] == 1.0
        assert rewards[1] == -1.0

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
        assert rewards[1] == -2.0  # RW

    def test_get_refiner_reward_functions(self) -> None:
        fns = get_refiner_reward_functions()
        assert len(fns) == 3
        for fn in fns:
            assert callable(fn)

    def test_reward_functions_handle_none_kwargs(self) -> None:
        """Reward functions should handle None gracefully."""
        rewards = refiner_correctness_reward_fn(
            completions=["A"],
            ground_truth=None,
            answer_type=None,
        )
        assert len(rewards) == 1


# =============================================================================
# Refiner Format LLM Fallback
# =============================================================================


class TestRefinerFormatFallback:
    """Tests for refiner format reward with LLM fallback paths."""

    def test_mcq_single_letter_passes(self) -> None:
        assert _compute_refiner_format_reward("A", "mcq") == 0.0
        assert _compute_refiner_format_reward("B", "mcq", "B. 24") == 0.0

    def test_mcq_letter_dot_text_fails_without_gt(self) -> None:
        assert _compute_refiner_format_reward("B. 24", "mcq") == -1.0

    def test_mcq_letter_dot_text_fails_without_llm(self) -> None:
        import os

        os.environ.pop("VLM_USE_LLM_JUDGE", None)
        assert _compute_refiner_format_reward("B. 24", "mcq", "B. 24") == -1.0

    def test_yesno_standard_passes(self) -> None:
        assert _compute_refiner_format_reward("Yes", "yesno") == 0.0
        assert _compute_refiner_format_reward("no", "yesno") == 0.0

    def test_yesno_sentence_fails_without_gt(self) -> None:
        assert _compute_refiner_format_reward("The fence is in front", "yesno") == -1.0

    def test_yesno_sentence_fails_without_llm(self) -> None:
        import os

        os.environ.pop("VLM_USE_LLM_JUDGE", None)
        assert (
            _compute_refiner_format_reward(
                "The fence is in front",
                "yesno",
                "The fence is in front of the boy.",
            )
            == -1.0
        )

    def test_empty_always_fails(self) -> None:
        assert _compute_refiner_format_reward("", "mcq", "B") == -1.0
        assert _compute_refiner_format_reward("", "yesno", "Yes") == -1.0

    def test_numeric_no_fallback(self) -> None:
        assert _compute_refiner_format_reward("42", "numeric") == 0.0
        assert _compute_refiner_format_reward("not a number", "numeric") == -1.0

    def test_open_always_passes(self) -> None:
        assert _compute_refiner_format_reward("anything", "open") == 0.0
