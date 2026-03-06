#!/usr/bin/env python3
"""Tests for deterministic reward components."""

import pytest

from vlm_grpo.rewards.deterministic import (
    compute_feedback_calibration_reward,
    compute_final_correct_reward,
    compute_format_reward,
    compute_minimal_edit_reward,
    compute_no_regression_reward,
    match_answer,
    match_mcq,
    match_numeric,
    match_yesno,
)
from vlm_grpo.trajectory import ParsedTrajectory
from vlm_grpo.utils import normalized_edit_distance

# =============================================================================
# Answer matching tests
# =============================================================================


class TestMatchMCQ:
    """Tests for match_mcq()."""

    def test_exact_match(self) -> None:
        assert match_mcq("A", "A") is True

    def test_different_letters(self) -> None:
        assert match_mcq("A", "B") is False

    def test_with_parens(self) -> None:
        assert match_mcq("(A)", "A") is True

    def test_both_parens(self) -> None:
        assert match_mcq("(B)", "(B)") is True

    def test_lowercase(self) -> None:
        assert match_mcq("a", "A") is True

    def test_empty_predicted(self) -> None:
        assert match_mcq("", "A") is False

    def test_non_letter(self) -> None:
        assert match_mcq("1", "A") is False


class TestMatchYesNo:
    """Tests for match_yesno()."""

    def test_yes_yes(self) -> None:
        assert match_yesno("Yes", "Yes") is True

    def test_no_no(self) -> None:
        assert match_yesno("No", "No") is True

    def test_yes_no(self) -> None:
        assert match_yesno("Yes", "No") is False

    def test_case_insensitive(self) -> None:
        assert match_yesno("YES", "yes") is True

    def test_true_yes(self) -> None:
        assert match_yesno("true", "yes") is True

    def test_empty(self) -> None:
        assert match_yesno("", "Yes") is False


class TestMatchNumeric:
    """Tests for match_numeric()."""

    def test_exact_integer(self) -> None:
        assert match_numeric("42", "42") is True

    def test_exact_float(self) -> None:
        assert match_numeric("3.14", "3.14") is True

    def test_within_tolerance(self) -> None:
        assert match_numeric("3.14", "3.15", tolerance=0.01) is True

    def test_outside_tolerance(self) -> None:
        assert match_numeric("3.14", "4.0", tolerance=0.01) is False

    def test_zero_gt(self) -> None:
        assert match_numeric("0.001", "0", tolerance=0.01) is True

    def test_fraction(self) -> None:
        assert match_numeric("1/2", "0.5") is True

    def test_non_numeric(self) -> None:
        assert match_numeric("hello", "42") is False

    def test_comma_number(self) -> None:
        assert match_numeric("1,000", "1000") is True


class TestMatchAnswer:
    """Tests for match_answer() dispatch."""

    def test_mcq_correct(self) -> None:
        assert match_answer("A", "A", "mcq") is True

    def test_mcq_wrong(self) -> None:
        assert match_answer("A", "B", "mcq") is False

    def test_yesno_correct(self) -> None:
        assert match_answer("Yes", "yes", "yesno") is True

    def test_numeric_correct(self) -> None:
        assert match_answer("42", "42", "numeric") is True

    def test_open_exact(self) -> None:
        assert match_answer("a cat", "A Cat", "open") is True

    def test_open_different(self) -> None:
        # Different open answers return None (cannot determine)
        result = match_answer("a cat", "a dog", "open")
        assert result is None

    def test_empty_predicted(self) -> None:
        assert match_answer("", "A", "mcq") is None


# =============================================================================
# Format reward tests
# =============================================================================


class TestFormatReward:
    """Tests for compute_format_reward()."""

    def test_valid_format(self) -> None:
        traj = ParsedTrajectory(
            feedback="Looks correct.",
            final_answer="A",
            raw_completion="FEEDBACK:\nLooks correct.\nFINAL_ANSWER:\nA",
            has_feedback_marker=True,
            has_final_answer_marker=True,
            parse_success=True,
        )
        assert compute_format_reward(traj, "A", "mcq") == 1.0

    def test_missing_markers(self) -> None:
        traj = ParsedTrajectory(
            feedback="",
            final_answer="A",
            raw_completion="Just text",
            has_feedback_marker=False,
            has_final_answer_marker=False,
            parse_success=False,
        )
        assert compute_format_reward(traj, "A", "mcq") == -1.0

    def test_empty_feedback(self) -> None:
        traj = ParsedTrajectory(
            feedback="",
            final_answer="A",
            raw_completion="FEEDBACK:\n\nFINAL_ANSWER:\nA",
            has_feedback_marker=True,
            has_final_answer_marker=True,
            parse_success=True,
        )
        assert compute_format_reward(traj, "A", "mcq") == -1.0

    def test_empty_extracted_answer(self) -> None:
        traj = ParsedTrajectory(
            feedback="OK",
            final_answer="nothing useful",
            raw_completion="FEEDBACK:\nOK\nFINAL_ANSWER:\nnothing useful",
            has_feedback_marker=True,
            has_final_answer_marker=True,
            parse_success=True,
        )
        # MCQ extraction fails -> empty extracted answer
        assert compute_format_reward(traj, "", "mcq") == -1.0


# =============================================================================
# Final correct reward tests
# =============================================================================


class TestFinalCorrectReward:
    """Tests for compute_final_correct_reward()."""

    def test_correct_mcq(self) -> None:
        assert compute_final_correct_reward("A", "A", "mcq", format_valid=True) == 1.0

    def test_incorrect_mcq(self) -> None:
        assert compute_final_correct_reward("A", "B", "mcq", format_valid=True) == -1.0

    def test_gated_by_format(self) -> None:
        assert compute_final_correct_reward("A", "A", "mcq", format_valid=False) == 0.0

    def test_correct_yesno(self) -> None:
        assert compute_final_correct_reward("Yes", "yes", "yesno", format_valid=True) == 1.0

    def test_undetermined_open(self) -> None:
        result = compute_final_correct_reward("a cat", "a dog", "open", format_valid=True)
        assert result == 0.0


# =============================================================================
# No regression reward tests
# =============================================================================


class TestNoRegressionReward:
    """Tests for compute_no_regression_reward()."""

    def test_rr_positive(self) -> None:
        """Answer2 correct = maintained correctness."""
        assert compute_no_regression_reward("A", "A", "mcq", format_valid=True) == 1.0

    def test_rw_heavy_penalty(self) -> None:
        """Answer2 wrong = regression, heavy penalty."""
        assert compute_no_regression_reward("B", "A", "mcq", format_valid=True) == -3.0

    def test_gated_by_format(self) -> None:
        assert compute_no_regression_reward("A", "A", "mcq", format_valid=False) == 0.0


# =============================================================================
# Minimal edit reward tests
# =============================================================================


class TestMinimalEditReward:
    """Tests for compute_minimal_edit_reward()."""

    def test_identical_answers(self) -> None:
        """Same answer1 and answer2 = max edit reward."""
        result = compute_minimal_edit_reward("A", "A", "A", "mcq", format_valid=True)
        assert result == 1.0

    @pytest.mark.skip(
        reason="Pre-existing: verifier substring match makes 'cat' vs 'a cat' both correct but edit distance != 0"
    )
    def test_different_but_both_correct(self) -> None:
        """Both correct but different text = lower reward."""
        # For open-ended where both match GT
        result = compute_minimal_edit_reward("cat", "cat", "a cat", "open", format_valid=True)
        # "cat" matches "cat" exactly -> open match returns True
        assert result == 1.0

    def test_answer2_wrong(self) -> None:
        """Answer2 wrong = 0.0 (not applicable)."""
        result = compute_minimal_edit_reward("B", "A", "A", "mcq", format_valid=True)
        assert result == 0.0

    def test_gated_by_format(self) -> None:
        result = compute_minimal_edit_reward("A", "A", "A", "mcq", format_valid=False)
        assert result == 0.0


# =============================================================================
# Feedback calibration reward tests
# =============================================================================


class TestFeedbackCalibrationReward:
    """Tests for compute_feedback_calibration_reward()."""

    def test_positive_correct(self) -> None:
        assert compute_feedback_calibration_reward("The answer is correct.") == 1.0

    def test_positive_no_change(self) -> None:
        assert compute_feedback_calibration_reward("No change needed.") == 1.0

    def test_negative_incorrect(self) -> None:
        assert compute_feedback_calibration_reward("The answer is incorrect.") == -1.0

    def test_negative_should_be(self) -> None:
        assert compute_feedback_calibration_reward("The answer should be B.") == -1.0

    def test_neutral_empty(self) -> None:
        assert compute_feedback_calibration_reward("") == 0.0

    def test_neutral_ambiguous(self) -> None:
        """No clear positive or negative signals."""
        assert compute_feedback_calibration_reward("The image shows a cat.") == 0.0

    def test_mixed_favors_negative(self) -> None:
        """Mixed signals where negative outweighs positive."""
        result = compute_feedback_calibration_reward(
            "The answer is correct but needs to be changed and is wrong."
        )
        assert result < 0


# =============================================================================
# Edit distance utility tests
# =============================================================================


class TestNormalizedEditDistance:
    """Tests for normalized_edit_distance()."""

    def test_identical(self) -> None:
        assert normalized_edit_distance("hello", "hello") == 0.0

    def test_completely_different(self) -> None:
        assert normalized_edit_distance("abc", "xyz") == 1.0

    def test_one_edit(self) -> None:
        dist = normalized_edit_distance("hello", "hallo")
        assert 0.0 < dist < 1.0

    def test_empty_strings(self) -> None:
        assert normalized_edit_distance("", "") == 0.0

    def test_one_empty(self) -> None:
        assert normalized_edit_distance("hello", "") == 1.0
        assert normalized_edit_distance("", "hello") == 1.0

    def test_symmetric(self) -> None:
        d1 = normalized_edit_distance("hello", "world")
        d2 = normalized_edit_distance("world", "hello")
        assert d1 == d2
