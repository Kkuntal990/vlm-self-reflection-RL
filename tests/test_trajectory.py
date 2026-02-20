#!/usr/bin/env python3
"""Tests for trajectory parsing and answer extraction."""

import pytest

from vlm_grpo.trajectory import (
    detect_hedging,
    extract_answer_from_text,
    extract_completion_text,
    parse_trajectory,
)


# =============================================================================
# parse_trajectory tests
# =============================================================================


class TestParseTrajectory:
    """Tests for parse_trajectory()."""

    def test_valid_both_markers(self) -> None:
        """Both FEEDBACK: and FINAL_ANSWER: present."""
        text = "FEEDBACK:\nThe answer looks correct.\nFINAL_ANSWER:\nA"
        result = parse_trajectory(text)
        assert result.parse_success is True
        assert result.has_feedback_marker is True
        assert result.has_final_answer_marker is True
        assert result.feedback == "The answer looks correct."
        assert result.final_answer == "A"

    def test_valid_multiline_feedback(self) -> None:
        """Feedback spans multiple lines."""
        text = "FEEDBACK:\nLine one.\nLine two.\nLine three.\nFINAL_ANSWER:\nB"
        result = parse_trajectory(text)
        assert result.parse_success is True
        assert "Line one." in result.feedback
        assert "Line three." in result.feedback
        assert result.final_answer == "B"

    def test_missing_feedback_marker(self) -> None:
        """Only FINAL_ANSWER marker present."""
        text = "Some random text\nFINAL_ANSWER:\nC"
        result = parse_trajectory(text)
        assert result.parse_success is False
        assert result.has_feedback_marker is False
        assert result.has_final_answer_marker is True
        assert result.final_answer == "C"

    def test_missing_final_answer_marker(self) -> None:
        """Only FEEDBACK marker present."""
        text = "FEEDBACK:\nThe answer is wrong."
        result = parse_trajectory(text)
        assert result.parse_success is False
        assert result.has_feedback_marker is True
        assert result.has_final_answer_marker is False
        assert result.feedback == "The answer is wrong."

    def test_no_markers(self) -> None:
        """No markers at all."""
        text = "Just some random text without any markers."
        result = parse_trajectory(text)
        assert result.parse_success is False
        assert result.has_feedback_marker is False
        assert result.has_final_answer_marker is False
        assert result.feedback == ""
        assert result.final_answer == ""

    def test_empty_string(self) -> None:
        """Empty completion."""
        result = parse_trajectory("")
        assert result.parse_success is False

    def test_case_insensitive_markers(self) -> None:
        """Markers should be case-insensitive."""
        text = "feedback:\nLooks good.\nfinal_answer:\nD"
        result = parse_trajectory(text)
        assert result.parse_success is True
        assert result.feedback == "Looks good."
        assert result.final_answer == "D"

    def test_extra_whitespace(self) -> None:
        """Whitespace around markers."""
        text = "FEEDBACK :  \n  Correct.  \n  FINAL_ANSWER :  \n  A  "
        result = parse_trajectory(text)
        assert result.parse_success is True
        assert result.feedback == "Correct."
        assert result.final_answer == "A"

    def test_multiple_final_answer_uses_last(self) -> None:
        """Multiple FINAL_ANSWER markers - use last one."""
        text = "FEEDBACK:\nWait.\nFINAL_ANSWER:\nA\nFINAL_ANSWER:\nB"
        result = parse_trajectory(text)
        assert result.parse_success is True
        assert result.final_answer == "B"

    def test_raw_completion_preserved(self) -> None:
        """Raw completion is preserved."""
        text = "FEEDBACK:\nOK\nFINAL_ANSWER:\nA"
        result = parse_trajectory(text)
        assert result.raw_completion == text


# =============================================================================
# extract_completion_text tests
# =============================================================================


class TestExtractCompletionText:
    """Tests for extract_completion_text()."""

    def test_string_input(self) -> None:
        """String passes through."""
        assert extract_completion_text("hello") == "hello"

    def test_conversational_format(self) -> None:
        """TRL conversational format."""
        completion = [{"role": "assistant", "content": "FEEDBACK:\nOK\nFINAL_ANSWER:\nA"}]
        result = extract_completion_text(completion)
        assert "FEEDBACK:" in result
        assert "FINAL_ANSWER:" in result

    def test_structured_content(self) -> None:
        """Structured content with text items."""
        completion = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "FEEDBACK: OK"},
                    {"type": "text", "text": "FINAL_ANSWER: A"},
                ],
            }
        ]
        result = extract_completion_text(completion)
        assert "FEEDBACK" in result
        assert "FINAL_ANSWER" in result

    def test_empty_list(self) -> None:
        """Empty list returns empty string."""
        result = extract_completion_text([])
        assert result == ""


# =============================================================================
# extract_answer_from_text tests
# =============================================================================


class TestExtractAnswerFromText:
    """Tests for extract_answer_from_text()."""

    # MCQ tests
    def test_mcq_single_letter(self) -> None:
        assert extract_answer_from_text("A", "mcq") == "A"

    def test_mcq_with_parens(self) -> None:
        assert extract_answer_from_text("(B)", "mcq") == "B"

    def test_mcq_with_paren_right(self) -> None:
        assert extract_answer_from_text("C)", "mcq") == "C"

    def test_mcq_multiple_letters_rejected(self) -> None:
        """Multiple different letters should be rejected (anti-hacking)."""
        assert extract_answer_from_text("A or B", "mcq") == ""

    def test_mcq_same_letter_repeated(self) -> None:
        """Same letter repeated is OK."""
        assert extract_answer_from_text("A A", "mcq") == "A"

    def test_mcq_lowercase(self) -> None:
        assert extract_answer_from_text("d", "mcq") == "D"

    def test_mcq_empty(self) -> None:
        assert extract_answer_from_text("", "mcq") == ""

    def test_mcq_no_letter(self) -> None:
        assert extract_answer_from_text("hello world", "mcq") == ""

    # YesNo tests
    def test_yesno_yes(self) -> None:
        assert extract_answer_from_text("Yes", "yesno") == "Yes"

    def test_yesno_no(self) -> None:
        assert extract_answer_from_text("No", "yesno") == "No"

    def test_yesno_case_insensitive(self) -> None:
        assert extract_answer_from_text("YES", "yesno") == "Yes"

    def test_yesno_hedging_rejected(self) -> None:
        """Hedging language should cause rejection."""
        assert extract_answer_from_text("I think yes, maybe", "yesno") == ""

    def test_yesno_possibly(self) -> None:
        assert extract_answer_from_text("Possibly no", "yesno") == ""

    # Numeric tests
    def test_numeric_integer(self) -> None:
        assert extract_answer_from_text("42", "numeric") == "42"

    def test_numeric_float(self) -> None:
        assert extract_answer_from_text("3.14", "numeric") == "3.14"

    def test_numeric_negative(self) -> None:
        assert extract_answer_from_text("-5", "numeric") == "-5"

    def test_numeric_in_text(self) -> None:
        assert extract_answer_from_text("The answer is 7", "numeric") == "7"

    def test_numeric_empty(self) -> None:
        assert extract_answer_from_text("no number here", "numeric") == ""

    # Open tests
    def test_open_passthrough(self) -> None:
        assert extract_answer_from_text("a cat", "open") == "a cat"

    def test_open_strip(self) -> None:
        assert extract_answer_from_text("  a cat  ", "open") == "a cat"


# =============================================================================
# detect_hedging tests
# =============================================================================


class TestDetectHedging:
    """Tests for detect_hedging()."""

    def test_no_hedging(self) -> None:
        assert detect_hedging("Yes") is False

    def test_maybe(self) -> None:
        assert detect_hedging("Maybe yes") is True

    def test_probably(self) -> None:
        assert detect_hedging("Probably not") is True

    def test_i_think(self) -> None:
        assert detect_hedging("I think the answer is yes") is True

    def test_not_sure(self) -> None:
        assert detect_hedging("I'm not sure but yes") is True

    def test_clean_no(self) -> None:
        assert detect_hedging("No") is False

    def test_it_could_be(self) -> None:
        assert detect_hedging("It could be yes") is True
