#!/usr/bin/env python3
"""Tests for rollout data structures and metrics computation."""

from vlm_grpo.config import RolloutConfig
from vlm_grpo.rewards.composition import CriticRewardBreakdown
from vlm_grpo.rollout import (
    CriticRolloutResult,
    RefinerRolloutResult,
    compute_rollout_metrics,
)

# =============================================================================
# Data Structure Tests
# =============================================================================


class TestCriticRolloutResult:
    """Tests for CriticRolloutResult dataclass."""

    def test_construction(self) -> None:
        result = CriticRolloutResult(
            sample_index=0,
            question="What is in the image?",
            image_path="/path/to/img.jpg",
            ground_truth="A",
            answer1="A",
            answer_type="mcq",
            choices="(A) cat (B) dog",
            dataset_name="test",
            a1_is_correct=True,
        )
        assert result.sample_index == 0
        assert result.feedbacks == []
        assert result.answer2s == []
        assert result.rewards == []

    def test_with_rollout_data(self) -> None:
        result = CriticRolloutResult(
            sample_index=0,
            question="Q",
            image_path="/img.jpg",
            ground_truth="A",
            answer1="A",
            answer_type="mcq",
            choices="",
            dataset_name="test",
            a1_is_correct=True,
            feedbacks=["good", "bad"],
            answer2s=["A", "B"],
            rewards=[1.0, -2.0],
        )
        assert len(result.feedbacks) == 2
        assert result.rewards[0] == 1.0

    def test_to_dict(self) -> None:
        result = CriticRolloutResult(
            sample_index=0,
            question="Q",
            image_path="/img.jpg",
            ground_truth="A",
            answer1="A",
            answer_type="mcq",
            choices="",
            dataset_name="test",
            a1_is_correct=True,
        )
        d = result.to_dict()
        assert "sample_index" in d
        assert "feedbacks" in d


class TestRefinerRolloutResult:
    """Tests for RefinerRolloutResult dataclass."""

    def test_construction(self) -> None:
        result = RefinerRolloutResult(
            sample_index=0,
            question="Q",
            image_path="/img.jpg",
            ground_truth="A",
            answer1="A",
            feedback1="The answer looks correct.",
            answer_type="mcq",
            choices="",
            dataset_name="test",
            a1_is_correct=True,
        )
        assert result.feedback1 == "The answer looks correct."
        assert result.answer2s == []

    def test_to_dict(self) -> None:
        result = RefinerRolloutResult(
            sample_index=0,
            question="Q",
            image_path="/img.jpg",
            ground_truth="A",
            answer1="A",
            feedback1="feedback",
            answer_type="mcq",
            choices="",
            dataset_name="test",
            a1_is_correct=True,
            answer2s=["A", "B", "A"],
        )
        d = result.to_dict()
        assert len(d["answer2s"]) == 3


# =============================================================================
# Config Tests
# =============================================================================


class TestRolloutConfig:
    """Tests for RolloutConfig."""

    def test_defaults(self) -> None:
        config = RolloutConfig()
        assert config.k_samples == 4
        assert config.temperature == 1.0
        assert config.a2_temperature == 1.0
        assert config.batch_size == 8

    def test_custom(self) -> None:
        config = RolloutConfig(k_samples=8, temperature=1.0)
        assert config.k_samples == 8
        assert config.temperature == 1.0

    def test_to_dict(self) -> None:
        config = RolloutConfig()
        d = config.to_dict()
        assert "k_samples" in d


# =============================================================================
# Metrics Computation Tests
# =============================================================================


class TestComputeRolloutMetrics:
    """Tests for compute_rollout_metrics."""

    def _make_breakdown(
        self,
        total_reward: float,
        a2_correct: bool | None,
        format_valid: bool = True,
        feedback_text: str = "Some feedback here.",
    ) -> CriticRewardBreakdown:
        return CriticRewardBreakdown(
            total_reward=total_reward,
            components={"format": 1.0, "downstream": 1.0, "calibration": 1.0},
            weighted_components={"format": 0.5, "downstream": 2.0, "calibration": 1.0},
            feedback_text=feedback_text,
            a2_text="A",
            a2_extracted="A",
            a2_correct=a2_correct,
            format_valid=format_valid,
        )

    def test_empty_results(self) -> None:
        metrics = compute_rollout_metrics([])
        assert metrics == {}

    def test_all_rr(self) -> None:
        """All RR transitions."""
        results = [
            CriticRolloutResult(
                sample_index=0,
                question="Q",
                image_path="/img.jpg",
                ground_truth="A",
                answer1="A",
                answer_type="mcq",
                choices="",
                dataset_name="test",
                a1_is_correct=True,
                feedbacks=["good"] * 4,
                answer2s=["A"] * 4,
                rewards=[1.0] * 4,
                reward_breakdowns=[self._make_breakdown(1.0, True) for _ in range(4)],
            )
        ]
        metrics = compute_rollout_metrics(results)
        assert metrics["rollout/rr_rate"] == 1.0
        assert metrics["rollout/rw_rate"] == 0.0

    def test_mixed_transitions(self) -> None:
        """Mix of RR and RW."""
        breakdowns = [
            self._make_breakdown(1.0, True),  # RR
            self._make_breakdown(-2.0, False),  # RW
        ]
        results = [
            CriticRolloutResult(
                sample_index=0,
                question="Q",
                image_path="/img.jpg",
                ground_truth="A",
                answer1="A",
                answer_type="mcq",
                choices="",
                dataset_name="test",
                a1_is_correct=True,
                feedbacks=["good", "bad"],
                answer2s=["A", "B"],
                rewards=[1.0, -2.0],
                reward_breakdowns=breakdowns,
            )
        ]
        metrics = compute_rollout_metrics(results)
        assert metrics["rollout/rr_rate"] == 0.5
        assert metrics["rollout/rw_rate"] == 0.5

    def test_format_valid_rate(self) -> None:
        breakdowns = [
            self._make_breakdown(1.0, True, format_valid=True),
            self._make_breakdown(-1.0, False, format_valid=False),
        ]
        results = [
            CriticRolloutResult(
                sample_index=0,
                question="Q",
                image_path="/img.jpg",
                ground_truth="A",
                answer1="A",
                answer_type="mcq",
                choices="",
                dataset_name="test",
                a1_is_correct=True,
                feedbacks=["good", ""],
                answer2s=["A", "B"],
                rewards=[1.0, -1.0],
                reward_breakdowns=breakdowns,
            )
        ]
        metrics = compute_rollout_metrics(results)
        assert metrics["rollout/format_valid_rate"] == 0.5
