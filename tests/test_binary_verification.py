"""Tests for v8 binary verification reward functions."""

import pytest

from vlm_grpo.rewards.composition import (
    compute_feedback_reward_breakdown,
    compute_verification_accuracy_reward,
    compute_verification_format_reward,
)
from vlm_grpo.config import FeedbackRewardWeights
from vlm_grpo.prompts import (
    BINARY_VERIFICATION_SYSTEM_PROMPT,
    VL_ASSISTANT_SYSTEM_PROMPT,
    build_critic_prompt,
    build_initial_answer_prompt,
)


class TestVerificationFormatReward:
    """Test compute_verification_format_reward."""

    def test_correct_uppercase(self) -> None:
        assert compute_verification_format_reward("CORRECT") == 0.0

    def test_incorrect_uppercase(self) -> None:
        assert compute_verification_format_reward("INCORRECT") == 0.0

    def test_correct_lowercase(self) -> None:
        assert compute_verification_format_reward("correct") == 0.0

    def test_incorrect_mixed_case(self) -> None:
        assert compute_verification_format_reward("Incorrect") == 0.0

    def test_with_whitespace(self) -> None:
        assert compute_verification_format_reward("  CORRECT  ") == 0.0

    def test_verbose_answer_penalized(self) -> None:
        assert compute_verification_format_reward("The answer is correct") == -2.0

    def test_empty_penalized(self) -> None:
        assert compute_verification_format_reward("") == -2.0

    def test_partial_word_penalized(self) -> None:
        assert compute_verification_format_reward("CORR") == -2.0

    def test_extra_words_penalized(self) -> None:
        assert compute_verification_format_reward("CORRECT answer") == -2.0


class TestVerificationAccuracyReward:
    """Test compute_verification_accuracy_reward."""

    def test_correct_when_a1_right(self) -> None:
        assert compute_verification_accuracy_reward("CORRECT", a1_is_correct=True) == 1.0

    def test_correct_when_a1_wrong(self) -> None:
        assert compute_verification_accuracy_reward("CORRECT", a1_is_correct=False) == -1.0

    def test_incorrect_when_a1_wrong(self) -> None:
        assert compute_verification_accuracy_reward("INCORRECT", a1_is_correct=False) == 1.0

    def test_incorrect_when_a1_right(self) -> None:
        assert compute_verification_accuracy_reward("INCORRECT", a1_is_correct=True) == -1.0

    def test_no_verdict_returns_zero(self) -> None:
        assert compute_verification_accuracy_reward("The painting shows blue tones", a1_is_correct=True) == 0.0

    def test_embedded_correct_matches(self) -> None:
        """Text containing 'correct' anywhere should match."""
        assert compute_verification_accuracy_reward("maybe correct", a1_is_correct=True) == 1.0

    def test_ambiguous_both_returns_zero(self) -> None:
        """Text with both CORRECT and INCORRECT returns 0 (ambiguous)."""
        assert compute_verification_accuracy_reward("correct, not incorrect", a1_is_correct=True) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert compute_verification_accuracy_reward("", a1_is_correct=True) == 0.0

    def test_case_insensitive(self) -> None:
        assert compute_verification_accuracy_reward("correct", a1_is_correct=True) == 1.0
        assert compute_verification_accuracy_reward("Incorrect", a1_is_correct=False) == 1.0


class TestBinaryFeedbackBreakdown:
    """Test compute_feedback_reward_breakdown with binary verification."""

    @pytest.fixture()
    def weights(self) -> FeedbackRewardWeights:
        return FeedbackRewardWeights(
            w_downstream=1.0,
            w_calibration=0.0,
            w_format=0.15,
            w_tag_penalty=0.0,
            w_verification_accuracy=1.0,
        )

    def test_wr_correct_verification(self, weights: FeedbackRewardWeights) -> None:
        """F1 says INCORRECT (correct), A2 corrects → high reward."""
        bd = compute_feedback_reward_breakdown(
            feedback_text="INCORRECT",
            a1_text="ZZZ_WRONG",
            a2_text="A",
            ground_truth="A",
            answer_type="mcq",
            choices="(A) (B) (C) (D)",
            weights=weights,
            use_binary_verification=True,
            reward_shaping_alpha=10.0,
        )
        # downstream: WR with α=10 → 1 + 10*(1-(-1)) = 21
        # verification: INCORRECT when A1 wrong → +1.0
        # format: valid → 0.0
        assert bd.total_reward == pytest.approx(21.0 + 1.0 + 0.0, abs=0.01)

    def test_ww_sycophantic_verification(self, weights: FeedbackRewardWeights) -> None:
        """F1 says CORRECT (wrong — sycophantic), A2 stays wrong."""
        bd = compute_feedback_reward_breakdown(
            feedback_text="CORRECT",
            a1_text="ZZZ_WRONG",
            a2_text="ZZZ_STILL_WRONG",
            ground_truth="A",
            answer_type="mcq",
            choices="(A) (B) (C) (D)",
            weights=weights,
            use_binary_verification=True,
            reward_shaping_alpha=10.0,
        )
        # downstream: WW with α=10 → -1 + 10*(-1-(-1)) = -1
        # verification: CORRECT when A1 wrong → -1.0
        # format: valid → 0.0
        assert bd.total_reward == pytest.approx(-1.0 + (-1.0) + 0.0, abs=0.01)

    def test_rr_correct_verification(self, weights: FeedbackRewardWeights) -> None:
        """F1 says CORRECT (correct), A2 maintains."""
        bd = compute_feedback_reward_breakdown(
            feedback_text="CORRECT",
            a1_text="A",
            a2_text="A",
            ground_truth="A",
            answer_type="mcq",
            choices="(A) (B) (C) (D)",
            weights=weights,
            use_binary_verification=True,
            reward_shaping_alpha=10.0,
        )
        # downstream: RR with α=10 → 1 + 10*(1-1) = 1
        # verification: CORRECT when A1 right → +1.0
        # format: valid → 0.0
        assert bd.total_reward == pytest.approx(1.0 + 1.0 + 0.0, abs=0.01)

    def test_invalid_format_penalized(self, weights: FeedbackRewardWeights) -> None:
        """F1 outputs verbose text instead of CORRECT/INCORRECT."""
        bd = compute_feedback_reward_breakdown(
            feedback_text="The answer looks good to me",
            a1_text="A",
            a2_text="A",
            ground_truth="A",
            answer_type="mcq",
            choices="(A) (B) (C) (D)",
            weights=weights,
            use_binary_verification=True,
            reward_shaping_alpha=10.0,
        )
        # format: invalid → -2.0 * 0.15 = -0.3
        assert bd.components["format"] == -2.0
        assert not bd.feedback_format_valid


class TestCriticPromptBinary:
    """Test build_critic_prompt with binary verification."""

    def test_binary_prompt_uses_verification_system(self) -> None:
        msgs = build_critic_prompt(
            "What color?",
            "Blue",
            model_type="qwen2vl",
            use_binary_verification=True,
        )
        system_text = msgs[0]["content"][0]["text"]
        assert system_text == BINARY_VERIFICATION_SYSTEM_PROMPT

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
