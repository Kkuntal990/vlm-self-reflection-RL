"""Tests for binary verification reward functions (v9: <feedback> tags)."""

import pytest

from vlm_grpo.rewards.composition import (
    compute_feedback_reward_breakdown,
    compute_verification_accuracy_reward,
    compute_verification_format_reward,
)
from vlm_grpo.config import FeedbackRewardWeights
from vlm_grpo.prompts import (
    FEEDBACK_VERIFIER_SYSTEM_PROMPT,
    VL_ASSISTANT_SYSTEM_PROMPT,
    build_critic_prompt,
    build_initial_answer_prompt,
)
from vlm_grpo.trajectory import extract_from_feedback_tags, has_feedback_tags


# =============================================================================
# Feedback tag extraction
# =============================================================================


class TestFeedbackTagExtraction:
    """Test <feedback> tag helpers in trajectory.py."""

    def test_extract_correct(self) -> None:
        assert extract_from_feedback_tags("<feedback>CORRECT</feedback> Good job.") == "CORRECT"

    def test_extract_incorrect(self) -> None:
        assert (
            extract_from_feedback_tags("<feedback>INCORRECT</feedback> Should be (B).")
            == "INCORRECT"
        )

    def test_case_insensitive(self) -> None:
        assert extract_from_feedback_tags("<Feedback>correct</Feedback>") == "correct"

    def test_no_tags_returns_empty(self) -> None:
        assert extract_from_feedback_tags("The answer is correct") == ""

    def test_has_tags_true(self) -> None:
        assert has_feedback_tags("<feedback>CORRECT</feedback>") is True

    def test_has_tags_false(self) -> None:
        assert has_feedback_tags("CORRECT") is False

    def test_whitespace_inside_tags(self) -> None:
        assert extract_from_feedback_tags("<feedback> INCORRECT </feedback>") == "INCORRECT"


# =============================================================================
# Verification format reward (binary: 1.0 with tags, 0.0 without)
# =============================================================================


class TestVerificationFormatReward:
    """Test compute_verification_format_reward with <feedback> tags."""

    def test_correct_with_tags(self) -> None:
        assert compute_verification_format_reward("<feedback>CORRECT</feedback> Good.") == 1.0

    def test_incorrect_with_tags(self) -> None:
        assert compute_verification_format_reward("<feedback>INCORRECT</feedback> Bad.") == 1.0

    def test_no_tags_returns_zero(self) -> None:
        assert compute_verification_format_reward("CORRECT") == 0.0

    def test_no_tags_verbose_returns_zero(self) -> None:
        assert compute_verification_format_reward("The answer is correct") == 0.0

    def test_empty_returns_zero(self) -> None:
        assert compute_verification_format_reward("") == 0.0

    def test_invalid_verdict_in_tags(self) -> None:
        assert compute_verification_format_reward("<feedback>MAYBE</feedback>") == 0.0

    def test_case_insensitive_verdict(self) -> None:
        assert compute_verification_format_reward("<feedback>correct</feedback>") == 1.0


# =============================================================================
# Verification accuracy (primary: tags, fallback: text search)
# =============================================================================


class TestVerificationAccuracyReward:
    """Test compute_verification_accuracy_reward."""

    # Primary path: <feedback> tags
    def test_tag_correct_a1_right(self) -> None:
        assert (
            compute_verification_accuracy_reward(
                "<feedback>CORRECT</feedback> Well done.", a1_is_correct=True
            )
            == 1.0
        )

    def test_tag_correct_a1_wrong(self) -> None:
        assert (
            compute_verification_accuracy_reward(
                "<feedback>CORRECT</feedback> Looks good.", a1_is_correct=False
            )
            == -1.0
        )

    def test_tag_incorrect_a1_wrong(self) -> None:
        assert (
            compute_verification_accuracy_reward(
                "<feedback>INCORRECT</feedback> Should be (B).", a1_is_correct=False
            )
            == 1.0
        )

    def test_tag_incorrect_a1_right(self) -> None:
        assert (
            compute_verification_accuracy_reward(
                "<feedback>INCORRECT</feedback> Wrong.", a1_is_correct=True
            )
            == -1.0
        )

    def test_tag_invalid_verdict(self) -> None:
        assert (
            compute_verification_accuracy_reward("<feedback>MAYBE</feedback>", a1_is_correct=True)
            == 0.0
        )

    # Fallback path: text search (no tags)
    def test_fallback_correct(self) -> None:
        assert compute_verification_accuracy_reward("CORRECT", a1_is_correct=True) == 1.0

    def test_fallback_incorrect(self) -> None:
        assert compute_verification_accuracy_reward("INCORRECT", a1_is_correct=False) == 1.0

    def test_fallback_embedded(self) -> None:
        assert (
            compute_verification_accuracy_reward(
                "The answer appears incorrect", a1_is_correct=False
            )
            == 1.0
        )

    def test_no_verdict_returns_zero(self) -> None:
        assert (
            compute_verification_accuracy_reward(
                "The painting shows blue tones", a1_is_correct=True
            )
            == 0.0
        )

    def test_empty_returns_zero(self) -> None:
        assert compute_verification_accuracy_reward("", a1_is_correct=True) == 0.0


# =============================================================================
# Feedback breakdown integration
# =============================================================================


class TestBinaryFeedbackBreakdown:
    """Test compute_feedback_reward_breakdown with binary verification."""

    @pytest.fixture()
    def weights(self) -> FeedbackRewardWeights:
        return FeedbackRewardWeights(
            w_downstream=1.0,
            w_calibration=0.0,
            w_format=0.5,
            w_tag_penalty=0.0,
            w_verification_accuracy=1.0,
        )

    def test_wr_correct_with_tags(self, weights: FeedbackRewardWeights) -> None:
        """F1 says INCORRECT (correct), A2 corrects → high reward."""
        bd = compute_feedback_reward_breakdown(
            feedback_text="<feedback>INCORRECT</feedback> Should be (A).",
            a1_text="ZZZ_WRONG",
            a2_text="A",
            ground_truth="A",
            answer_type="mcq",
            choices="(A) (B) (C) (D)",
            weights=weights,
            use_binary_verification=True,
            reward_shaping_alpha=3.0,
        )
        # downstream: WR with α=3 → 1 + 3*(1-(-1)) = 7
        # verification: INCORRECT when A1 wrong → +1.0
        # format: <feedback> tag present → 1.0 * 0.5 = 0.5
        assert bd.total_reward == pytest.approx(7.0 + 1.0 + 0.5, abs=0.01)

    def test_rr_correct_with_tags(self, weights: FeedbackRewardWeights) -> None:
        """F1 says CORRECT (correct), A2 maintains."""
        bd = compute_feedback_reward_breakdown(
            feedback_text="<feedback>CORRECT</feedback> Good.",
            a1_text="A",
            a2_text="A",
            ground_truth="A",
            answer_type="mcq",
            choices="(A) (B) (C) (D)",
            weights=weights,
            use_binary_verification=True,
            reward_shaping_alpha=3.0,
        )
        # downstream: RR with α=3 → 1 + 3*(0) = 1
        # verification: CORRECT when A1 right → +1.0
        # format: tag present → 0.5
        assert bd.total_reward == pytest.approx(1.0 + 1.0 + 0.5, abs=0.01)

    def test_no_tags_fallback(self, weights: FeedbackRewardWeights) -> None:
        """F1 without tags — format=0.0, accuracy via text fallback."""
        bd = compute_feedback_reward_breakdown(
            feedback_text="INCORRECT. The answer should be (A).",
            a1_text="ZZZ_WRONG",
            a2_text="A",
            ground_truth="A",
            answer_type="mcq",
            choices="(A) (B) (C) (D)",
            weights=weights,
            use_binary_verification=True,
            reward_shaping_alpha=3.0,
        )
        # format: no tags → 0.0 * 0.5 = 0.0
        assert bd.components["format"] == 0.0
        # accuracy should still work via fallback
        assert bd.components["calibration"] == 1.0


# =============================================================================
# Prompt tests
# =============================================================================


class TestCriticPromptBinary:
    """Test build_critic_prompt with binary verification."""

    def test_binary_prompt_uses_feedback_verifier(self) -> None:
        msgs = build_critic_prompt(
            "What color?",
            "Blue",
            model_type="qwen2vl",
            use_binary_verification=True,
        )
        system_text = msgs[0]["content"][0]["text"]
        assert system_text == FEEDBACK_VERIFIER_SYSTEM_PROMPT
        assert "<feedback>" in system_text

    def test_standard_prompt_uses_critic_system(self) -> None:
        msgs = build_critic_prompt(
            "What color?",
            "Blue",
            model_type="qwen2vl",
            use_binary_verification=False,
        )
        system_text = msgs[0]["content"][0]["text"]
        assert "constructive feedback" in system_text

    def test_role_flipping_preserved_in_binary(self) -> None:
        msgs = build_critic_prompt(
            "What color?",
            "Blue",
            model_type="qwen2vl",
            use_binary_verification=True,
        )
        assert msgs[1]["role"] == "assistant"  # question as assistant
        assert msgs[2]["role"] == "user"  # answer as user


class TestA1PromptUnchanged:
    """Confirm A1/A2 use VL_ASSISTANT_SYSTEM_PROMPT in bare mode."""

    def test_a1_prompt_bare_mode(self) -> None:
        msgs = build_initial_answer_prompt("What?", use_think_answer_tags=False)
        system_text = msgs[0]["content"][0]["text"]
        assert system_text == VL_ASSISTANT_SYSTEM_PROMPT
