#!/usr/bin/env python3
"""Tests for end-to-end reward composition."""

import pytest

from vlm_grpo.rewards.rw_reward import (
    clear_parse_cache,
    compute_full_breakdown,
    correctness_reward_fn,
    feedback_calibration_reward_fn,
    format_reward_fn,
    minimal_edit_reward_fn,
    no_regression_reward_fn,
)


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    """Clear parse cache between tests."""
    clear_parse_cache()


# =============================================================================
# Full breakdown tests
# =============================================================================


class TestComputeFullBreakdown:
    """Tests for compute_full_breakdown()."""

    def test_perfect_rr(self) -> None:
        """Perfect completion: correct answer, calibrated feedback."""
        completion = "FEEDBACK:\nThe answer is correct and matches the image.\nFINAL_ANSWER:\nA"
        breakdown = compute_full_breakdown(
            completion_text=completion,
            ground_truth="A",
            answer1="A",
            answer_type="mcq",
            choices="",
            dataset_name="test",
        )

        assert breakdown.parse_success is True
        assert breakdown.format_valid is True
        assert breakdown.final_answer_extracted == "A"
        assert breakdown.components["format"] == 1.0
        assert breakdown.components["final_correct"] == 1.0
        assert breakdown.components["no_regression"] == 1.0
        assert breakdown.components["minimal_edit"] == 1.0
        assert breakdown.components["feedback_calibration"] == 1.0
        assert breakdown.total_reward > 0

    def test_rw_flip(self) -> None:
        """RW flip: wrong answer, high penalty."""
        completion = "FEEDBACK:\nThe answer is incorrect.\nFINAL_ANSWER:\nB"
        breakdown = compute_full_breakdown(
            completion_text=completion,
            ground_truth="A",
            answer1="A",
            answer_type="mcq",
            choices="",
            dataset_name="test",
        )

        assert breakdown.format_valid is True
        assert breakdown.components["final_correct"] == -1.0
        assert breakdown.components["no_regression"] == -3.0
        assert breakdown.components["minimal_edit"] == 0.0
        assert breakdown.total_reward < 0

    def test_malformed_no_markers(self) -> None:
        """Malformed: no markers at all."""
        completion = "Just some random text."
        breakdown = compute_full_breakdown(
            completion_text=completion,
            ground_truth="A",
            answer1="A",
            answer_type="mcq",
            choices="",
            dataset_name="test",
        )

        assert breakdown.parse_success is False
        assert breakdown.format_valid is False
        assert breakdown.components["format"] == -1.0
        # Gated components
        assert breakdown.components["final_correct"] == 0.0
        assert breakdown.components["no_regression"] == 0.0

    def test_missing_final_answer(self) -> None:
        """Only FEEDBACK marker, missing FINAL_ANSWER."""
        completion = "FEEDBACK:\nLooks good but no final answer."
        breakdown = compute_full_breakdown(
            completion_text=completion,
            ground_truth="A",
            answer1="A",
            answer_type="mcq",
            choices="",
            dataset_name="test",
        )

        assert breakdown.parse_success is False
        assert breakdown.format_valid is False

    def test_yesno_correct(self) -> None:
        """Yes/No correct answer."""
        completion = "FEEDBACK:\nThe answer is correct.\nFINAL_ANSWER:\nYes"
        breakdown = compute_full_breakdown(
            completion_text=completion,
            ground_truth="Yes",
            answer1="Yes",
            answer_type="yesno",
            choices="",
            dataset_name="test",
        )

        assert breakdown.format_valid is True
        assert breakdown.components["final_correct"] == 1.0
        assert breakdown.components["no_regression"] == 1.0

    def test_numeric_correct(self) -> None:
        """Numeric correct answer."""
        completion = "FEEDBACK:\nThe answer is accurate.\nFINAL_ANSWER:\n42"
        breakdown = compute_full_breakdown(
            completion_text=completion,
            ground_truth="42",
            answer1="42",
            answer_type="numeric",
            choices="",
            dataset_name="test",
        )

        assert breakdown.format_valid is True
        assert breakdown.components["final_correct"] == 1.0


# =============================================================================
# Individual reward function tests (TRL-compatible signature)
# =============================================================================


class TestRewardFunctions:
    """Tests for individual TRL-compatible reward functions."""

    def test_format_reward_fn_batch(self) -> None:
        """Format reward handles batch of completions."""
        completions = [
            "FEEDBACK:\nOK\nFINAL_ANSWER:\nA",
            "No markers here",
            "FEEDBACK:\nOK\nFINAL_ANSWER:\nB",
        ]
        rewards = format_reward_fn(
            completions=completions,
            answer_type=["mcq", "mcq", "mcq"],
            choices=["", "", ""],
        )
        assert len(rewards) == 3
        assert rewards[0] == 1.0  # valid
        assert rewards[1] == -1.0  # invalid
        assert rewards[2] == 1.0  # valid

    def test_correctness_reward_fn_batch(self) -> None:
        """Correctness reward handles batch."""
        completions = [
            "FEEDBACK:\nOK\nFINAL_ANSWER:\nA",
            "FEEDBACK:\nOK\nFINAL_ANSWER:\nB",
        ]
        rewards = correctness_reward_fn(
            completions=completions,
            ground_truth=["A", "A"],
            answer_type=["mcq", "mcq"],
            choices=["", ""],
        )
        assert len(rewards) == 2
        assert rewards[0] == 1.0  # correct
        assert rewards[1] == -1.0  # incorrect

    def test_no_regression_reward_fn_batch(self) -> None:
        """No regression reward handles batch."""
        completions = [
            "FEEDBACK:\nOK\nFINAL_ANSWER:\nA",
            "FEEDBACK:\nOK\nFINAL_ANSWER:\nB",
        ]
        rewards = no_regression_reward_fn(
            completions=completions,
            ground_truth=["A", "A"],
            answer_type=["mcq", "mcq"],
            choices=["", ""],
        )
        assert rewards[0] == 1.0  # RR
        assert rewards[1] == -3.0  # RW

    def test_minimal_edit_reward_fn_batch(self) -> None:
        """Minimal edit reward handles batch."""
        completions = [
            "FEEDBACK:\nOK\nFINAL_ANSWER:\nA",
            "FEEDBACK:\nOK\nFINAL_ANSWER:\nB",
        ]
        rewards = minimal_edit_reward_fn(
            completions=completions,
            ground_truth=["A", "A"],
            answer1=["A", "A"],
            answer_type=["mcq", "mcq"],
            choices=["", ""],
        )
        assert rewards[0] == 1.0  # Same answer, both correct
        assert rewards[1] == 0.0  # Answer2 wrong, not applicable

    def test_feedback_calibration_reward_fn_batch(self) -> None:
        """Feedback calibration reward handles batch."""
        completions = [
            "FEEDBACK:\nThe answer is correct.\nFINAL_ANSWER:\nA",
            "FEEDBACK:\nThe answer is wrong, should be changed.\nFINAL_ANSWER:\nA",
        ]
        rewards = feedback_calibration_reward_fn(
            completions=completions,
            answer_type=["mcq", "mcq"],
            choices=["", ""],
        )
        assert rewards[0] > 0  # Calibrated
        assert rewards[1] < 0  # Miscalibrated


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in reward composition."""

    def test_conversational_format(self) -> None:
        """TRL conversational format completions."""
        completions = [
            [{"role": "assistant", "content": "FEEDBACK:\nOK\nFINAL_ANSWER:\nA"}],
        ]
        rewards = format_reward_fn(
            completions=completions,
            answer_type=["mcq"],
            choices=[""],
        )
        assert rewards[0] == 1.0

    def test_none_kwargs_defaults(self) -> None:
        """Functions handle None kwargs gracefully."""
        rewards = format_reward_fn(
            completions=["FEEDBACK:\nOK\nFINAL_ANSWER:\nA"],
        )
        assert len(rewards) == 1

    def test_empty_completion(self) -> None:
        """Empty string completion."""
        rewards = format_reward_fn(
            completions=[""],
            answer_type=["mcq"],
            choices=[""],
        )
        assert rewards[0] == -1.0

    def test_format_gates_correctness(self) -> None:
        """When format is invalid, correctness should be gated (0.0)."""
        malformed = "No markers at all"
        fmt_rewards = format_reward_fn(
            completions=[malformed],
            answer_type=["mcq"],
            choices=[""],
        )
        corr_rewards = correctness_reward_fn(
            completions=[malformed],
            ground_truth=["A"],
            answer_type=["mcq"],
            choices=[""],
        )
        assert fmt_rewards[0] == -1.0
        assert corr_rewards[0] == 0.0  # Gated
