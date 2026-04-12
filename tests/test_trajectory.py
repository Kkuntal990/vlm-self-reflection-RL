#!/usr/bin/env python3
"""Tests for answer extraction, normalization, and hedging detection."""

from vlm_grpo.trajectory import (
    detect_hedging,
    extract_answer_from_text,
    extract_completion_text,
    extract_from_answer_tags,
    has_think_answer_tags,
)

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
        completion = [{"role": "assistant", "content": "The answer is A"}]
        result = extract_completion_text(completion)
        assert result == "The answer is A"

    def test_structured_content(self) -> None:
        """Structured content with text items."""
        completion = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Part one"},
                    {"type": "text", "text": "Part two"},
                ],
            }
        ]
        result = extract_completion_text(completion)
        assert "Part one" in result
        assert "Part two" in result

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


# =============================================================================
# Think/Answer tag extraction tests
# =============================================================================


class TestExtractFromAnswerTags:
    """Tests for extract_from_answer_tags."""

    def test_extracts_from_tags(self) -> None:
        text = "<think>The cat is top-left.</think><answer>(A)</answer>"
        assert extract_from_answer_tags(text) == "(A)"

    def test_extracts_multiline(self) -> None:
        text = "<think>Looking at the image,\nthe object is in box C.</think>\n<answer>(C)</answer>"
        assert extract_from_answer_tags(text) == "(C)"

    def test_fallback_no_tags(self) -> None:
        """Without tags, returns original text."""
        assert extract_from_answer_tags("(B)") == "(B)"

    def test_fallback_bare_letter(self) -> None:
        assert extract_from_answer_tags("A") == "A"

    def test_strips_whitespace_inside(self) -> None:
        text = "<think>reason</think><answer>  (B)  </answer>"
        assert extract_from_answer_tags(text) == "(B)"

    def test_case_insensitive(self) -> None:
        text = "<Think>reason</Think><Answer>(A)</Answer>"
        assert extract_from_answer_tags(text) == "(A)"

    def test_verbose_answer_inside_tags(self) -> None:
        """If model puts verbose text in answer tags, we still extract it."""
        text = "<think>reason</think><answer>The answer is (B)</answer>"
        assert extract_from_answer_tags(text) == "The answer is (B)"


class TestHasThinkAnswerTags:
    """Tests for has_think_answer_tags."""

    def test_both_present(self) -> None:
        assert has_think_answer_tags("<think>x</think><answer>y</answer>") is True

    def test_only_answer(self) -> None:
        assert has_think_answer_tags("<answer>y</answer>") is False

    def test_only_think(self) -> None:
        assert has_think_answer_tags("<think>x</think>") is False

    def test_no_tags(self) -> None:
        assert has_think_answer_tags("(A)") is False

    def test_bare_text(self) -> None:
        assert has_think_answer_tags("") is False
