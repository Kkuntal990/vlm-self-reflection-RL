#!/usr/bin/env python3
"""Tests for two-trajectory reward composition."""

from vlm_grpo.rewards.composition import (
    CriticRewardWeights,
    RefinerRewardWeights,
    _compute_refiner_format_reward,
    _contains_cjk,
    _count_content_units,
    compute_critic_format_reward,
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
        assert w.w_format == 0.15

    def test_to_dict(self) -> None:
        w = CriticRewardWeights()
        d = w.to_dict()
        assert "w_downstream" in d
        assert d["w_downstream"] == 2.0

    def test_to_list(self) -> None:
        w = CriticRewardWeights()
        lst = w.to_list()
        assert len(lst) == 2
        assert lst == [0.15, 2.0]  # [format, downstream]

    def test_custom_weights(self) -> None:
        w = CriticRewardWeights(w_downstream=3.0, w_format=0.1)
        assert w.w_downstream == 3.0


class TestRefinerRewardWeights:
    """Tests for RefinerRewardWeights."""

    def test_defaults(self) -> None:
        w = RefinerRewardWeights()
        assert w.w_correctness == 1.0
        assert w.w_no_regression == 2.0
        assert w.w_minimal_edit == 0.3
        assert w.w_format == 0.15

    def test_to_list(self) -> None:
        w = RefinerRewardWeights()
        lst = w.to_list()
        assert len(lst) == 4
        assert lst == [0.15, 1.0, 2.0, 0.3]


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
            ground_truth=["A", "B", "A"],
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


# =============================================================================
# CJK Helper Functions
# =============================================================================


class TestCJKHelpers:
    """Tests for CJK detection and content unit counting."""

    def test_contains_cjk_chinese(self) -> None:
        assert _contains_cjk("这是中文") is True

    def test_contains_cjk_english(self) -> None:
        assert _contains_cjk("This is English") is False

    def test_contains_cjk_mixed(self) -> None:
        assert _contains_cjk("Answer is 正确") is True

    def test_contains_cjk_empty(self) -> None:
        assert _contains_cjk("") is False

    def test_count_english(self) -> None:
        assert _count_content_units("The answer is correct") == 4

    def test_count_chinese(self) -> None:
        # 6 CJK characters: 答案是正确的
        assert _count_content_units("答案是正确的") == 6

    def test_count_mixed(self) -> None:
        # "Answer" + "is" (2 words) + 正确 (2 chars) = 4
        assert _count_content_units("Answer is 正确") == 4

    def test_count_empty(self) -> None:
        assert _count_content_units("") == 0

    def test_count_chinese_long_sentence(self) -> None:
        # Real Chinese feedback: many chars → large count
        feedback = "你的回答是24，但这个答案是错误的"
        count = _count_content_units(feedback)
        assert count >= 10  # Should be well above 5-word threshold


# =============================================================================
# Critic Format CJK
# =============================================================================


class TestCriticFormatCJK:
    """Tests for CJK support in critic format reward."""

    def test_chinese_substantive_feedback(self) -> None:
        """Chinese feedback with many characters should get +1.0."""
        feedback = "答案是正确的，与图像匹配。"
        assert compute_critic_format_reward(feedback) == 1.0

    def test_chinese_with_stance_keyword(self) -> None:
        """Chinese feedback with stance keyword should get +1.0."""
        feedback = "答案正确"  # 4 chars with 正确
        assert compute_critic_format_reward(feedback) == 1.0

    def test_chinese_short_no_stance(self) -> None:
        """Very short Chinese without stance should get -2.0."""
        feedback = "好"  # 1 char
        assert compute_critic_format_reward(feedback) == -2.0

    def test_chinese_medium_no_stance(self) -> None:
        """3-4 Chinese chars without stance keyword should get -1.0."""
        feedback = "还可以吧"  # 4 chars, no stance keyword
        assert compute_critic_format_reward(feedback) == -1.0

    def test_english_unchanged(self) -> None:
        """English feedback still works correctly."""
        assert compute_critic_format_reward("Answer is correct.") == 1.0
        assert compute_critic_format_reward("OK") == -2.0
        assert compute_critic_format_reward("Maybe try again.") == -1.0

    def test_real_chinese_feedback(self) -> None:
        """Real Chinese feedback from training logs should get +1.0."""
        feedback = (
            "你的回答是24，但这个答案是错误的。你需要考虑PA=12这个信息，"
            "因为这个长度给了我们一个比例关系。"
        )
        assert compute_critic_format_reward(feedback) == 1.0


# =============================================================================
# Refiner Format LLM Fallback
# =============================================================================


class TestRefinerFormatFallback:
    """Tests for refiner format reward with LLM fallback paths."""

    def test_mcq_single_letter_passes(self) -> None:
        """Single letter passes deterministic check, no LLM needed."""
        assert _compute_refiner_format_reward("A", "mcq") == 0.0
        assert _compute_refiner_format_reward("B", "mcq", "B. 24") == 0.0

    def test_mcq_letter_dot_text_fails_without_gt(self) -> None:
        """'B. 24' fails without ground_truth (no fallback possible)."""
        assert _compute_refiner_format_reward("B. 24", "mcq") == -1.0

    def test_mcq_letter_dot_text_fails_without_llm(self) -> None:
        """'B. 24' with GT but LLM disabled still fails."""
        import os

        os.environ.pop("VLM_USE_LLM_JUDGE", None)
        assert _compute_refiner_format_reward("B. 24", "mcq", "B. 24") == -1.0

    def test_yesno_standard_passes(self) -> None:
        """Standard yes/no passes deterministic check."""
        assert _compute_refiner_format_reward("Yes", "yesno") == 0.0
        assert _compute_refiner_format_reward("no", "yesno") == 0.0

    def test_yesno_sentence_fails_without_gt(self) -> None:
        """Sentence answer fails yesno without ground_truth."""
        assert _compute_refiner_format_reward("The fence is in front", "yesno") == -1.0

    def test_yesno_sentence_fails_without_llm(self) -> None:
        """Sentence with GT but LLM disabled still fails."""
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
        """Empty answer always gets -1.0 regardless of GT."""
        assert _compute_refiner_format_reward("", "mcq", "B") == -1.0
        assert _compute_refiner_format_reward("", "yesno", "Yes") == -1.0

    def test_numeric_no_fallback(self) -> None:
        """Numeric format has no LLM fallback — deterministic only."""
        assert _compute_refiner_format_reward("42", "numeric") == 0.0
        assert _compute_refiner_format_reward("not a number", "numeric") == -1.0

    def test_open_always_passes(self) -> None:
        """Open format passes for any non-empty text."""
        assert _compute_refiner_format_reward("anything", "open") == 0.0
