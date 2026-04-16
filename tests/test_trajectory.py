#!/usr/bin/env python3
"""Tests for answer extraction, normalization, and hedging detection.

Tests cover all realistic Qwen2.5-VL output patterns:
- Extraction: liberal recovery of answers from messy model outputs
- Format reward: strict validation of expected output structure
- Correctness matching: deterministic matching for MCQ and counting
"""

import pytest

from vlm_grpo.rewards.composition import (
    _compute_tag_format_reward,
)
from vlm_grpo.rewards.deterministic import match_answer
from vlm_grpo.trajectory import (
    detect_hedging,
    extract_answer_from_text,
    extract_completion_text,
    extract_from_answer_tags,
    has_think_answer_tags,
    normalize_answer,
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

    # MCQ format variations — extraction should be liberal
    def test_mcq_paren_letter_paren(self) -> None:
        """(A) format."""
        assert extract_answer_from_text("(A)", "mcq") == "A"

    def test_mcq_lowercase_paren(self) -> None:
        """(a) format."""
        assert extract_answer_from_text("(a)", "mcq") == "A"

    def test_mcq_letter_dot(self) -> None:
        """A. format."""
        assert extract_answer_from_text("A.", "mcq") == "A"

    def test_mcq_bare_letter(self) -> None:
        """Bare A format."""
        assert extract_answer_from_text("A", "mcq") == "A"

    def test_mcq_bare_lowercase(self) -> None:
        """Bare a format."""
        assert extract_answer_from_text("a", "mcq") == "A"

    def test_mcq_letter_with_text(self) -> None:
        """(A) the second image."""
        assert extract_answer_from_text("(A) the second image", "mcq") == "A"

    def test_mcq_answer_is_pattern(self) -> None:
        """The answer is (B)."""
        assert extract_answer_from_text("The answer is (B)", "mcq") == "B"

    def test_mcq_answer_colon(self) -> None:
        """Answer: C."""
        assert extract_answer_from_text("Answer: C", "mcq") == "C"

    def test_mcq_from_answer_tags(self) -> None:
        """<answer>(A)</answer> — extraction goes through tags first."""
        assert extract_answer_from_text("<answer>(A)</answer>", "mcq") == "A"

    def test_mcq_from_think_answer_tags(self) -> None:
        """Full think/answer tag output."""
        text = "<think>I see a cat</think><answer>(B)</answer>"
        assert extract_answer_from_text(text, "mcq") == "B"

    def test_mcq_verbose_in_tags(self) -> None:
        """<answer>A. the second image</answer>."""
        assert extract_answer_from_text("<answer>A. the second image</answer>", "mcq") == "A"

    # Counting tests — extraction should handle digits, words, and tags
    def test_counting_bare_digit(self) -> None:
        assert extract_answer_from_text("6", "counting") == "6"

    def test_counting_in_prose(self) -> None:
        """The count is 6."""
        assert extract_answer_from_text("The count is 6", "counting") == "6"

    def test_counting_number_word(self) -> None:
        """six -> 6."""
        assert extract_answer_from_text("six", "counting") == "6"

    def test_counting_word_in_prose(self) -> None:
        """There are three cups."""
        assert extract_answer_from_text("There are three cups", "counting") == "3"

    def test_counting_from_answer_tags(self) -> None:
        """<answer>6</answer>."""
        assert extract_answer_from_text("<answer>6</answer>", "counting") == "6"

    def test_counting_word_in_answer_tags(self) -> None:
        """<answer>six</answer>."""
        assert extract_answer_from_text("<answer>six</answer>", "counting") == "6"

    def test_counting_from_think_answer(self) -> None:
        """Full think/answer output with digit."""
        text = "<think>I count 6 people</think><answer>6</answer>"
        assert extract_answer_from_text(text, "counting") == "6"

    def test_counting_digit_preferred_over_word(self) -> None:
        """Digit in text takes priority over word."""
        assert extract_answer_from_text("There are 6 items, six total", "counting") == "6"

    def test_counting_no_number(self) -> None:
        """No number at all."""
        assert extract_answer_from_text("many objects", "counting") == ""

    def test_counting_numeric_fallback(self) -> None:
        """Numeric type does NOT match words (only counting does)."""
        assert extract_answer_from_text("six", "numeric") == ""

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


# =============================================================================
# End-to-end: extraction + correctness + strict format for all model outputs
# =============================================================================


class TestMCQEndToEnd:
    """Test all realistic MCQ outputs from Qwen2.5-VL.

    For each output pattern, verify:
    1. Extraction recovers the letter (liberal)
    2. Correctness matching works
    3. Format reward is strict: only <think>...<answer>(X)</answer> gets +0.5
    """

    # --- Perfect format: <think>...<answer>(X)</answer> ---

    def test_perfect_format_correct(self) -> None:
        text = "<think>The second image matches the Impressionist style.</think><answer>(A)</answer>"
        assert extract_answer_from_text(text, "mcq") == "A"
        extracted = normalize_answer(extract_from_answer_tags(text))
        assert match_answer(extracted, "a", "mcq") is True
        assert _compute_tag_format_reward(text, "mcq") == 0.5

    def test_perfect_format_wrong(self) -> None:
        text = "<think>I think it's the third image.</think><answer>(B)</answer>"
        assert extract_answer_from_text(text, "mcq") == "B"
        extracted = normalize_answer(extract_from_answer_tags(text))
        assert match_answer(extracted, "a", "mcq") is False
        assert _compute_tag_format_reward(text, "mcq") == 0.5  # format is fine

    def test_perfect_format_lowercase(self) -> None:
        text = "<think>reason</think><answer>(a)</answer>"
        assert extract_answer_from_text(text, "mcq") == "A"
        assert _compute_tag_format_reward(text, "mcq") == 0.5

    # --- Tags present but bad content inside <answer> ---

    def test_tags_bare_letter_inside(self) -> None:
        """<answer>A</answer> — missing parentheses."""
        text = "<think>reason</think><answer>A</answer>"
        assert extract_answer_from_text(text, "mcq") == "A"  # extraction works
        assert _compute_tag_format_reward(text, "mcq") == -0.5  # format fails

    def test_tags_letter_dot_text_inside(self) -> None:
        """<answer>A. the second image</answer> — verbose."""
        text = "<think>reason</think><answer>A. the second image</answer>"
        assert extract_answer_from_text(text, "mcq") == "A"  # extraction works
        assert _compute_tag_format_reward(text, "mcq") == -0.5  # format fails

    def test_tags_full_sentence_inside(self) -> None:
        """<answer>The answer is (B)</answer> — sentence."""
        text = "<think>reason</think><answer>The answer is (B)</answer>"
        assert extract_answer_from_text(text, "mcq") == "B"  # extraction works
        assert _compute_tag_format_reward(text, "mcq") == -0.5  # format fails

    def test_tags_letter_with_period(self) -> None:
        """<answer>A.</answer> — letter with period."""
        text = "<think>reason</think><answer>A.</answer>"
        assert extract_answer_from_text(text, "mcq") == "A"
        assert _compute_tag_format_reward(text, "mcq") == -0.5

    def test_tags_empty_answer(self) -> None:
        """<answer></answer> — empty."""
        text = "<think>reason</think><answer></answer>"
        assert extract_answer_from_text(text, "mcq") == ""
        assert _compute_tag_format_reward(text, "mcq") == -0.5

    # --- No tags at all ---

    def test_no_tags_bare_letter(self) -> None:
        text = "A"
        assert extract_answer_from_text(text, "mcq") == "A"
        assert _compute_tag_format_reward(text, "mcq") == -1.0

    def test_no_tags_paren_letter(self) -> None:
        text = "(B)"
        assert extract_answer_from_text(text, "mcq") == "B"
        assert _compute_tag_format_reward(text, "mcq") == -1.0

    def test_no_tags_verbose_response(self) -> None:
        """Long verbose response like base model outputs."""
        text = "To solve this pattern recognition problem, let's analyze the sequence of images presented. The correct answer is (C)."
        assert extract_answer_from_text(text, "mcq") == "C"
        assert _compute_tag_format_reward(text, "mcq") == -1.0

    def test_no_tags_answer_is_pattern(self) -> None:
        text = "The answer is B"
        assert extract_answer_from_text(text, "mcq") == "B"
        assert _compute_tag_format_reward(text, "mcq") == -1.0

    def test_no_tags_letter_dot_with_text(self) -> None:
        text = "A. the second image"
        assert extract_answer_from_text(text, "mcq") == "A"
        assert _compute_tag_format_reward(text, "mcq") == -1.0

    # --- Edge cases ---

    def test_think_only_no_answer_tag(self) -> None:
        """<think> present but no <answer> tag."""
        text = "<think>I see the pattern</think>The answer is (A)"
        assert extract_answer_from_text(text, "mcq") == "A"
        assert _compute_tag_format_reward(text, "mcq") == -1.0

    def test_answer_tag_only_no_think(self) -> None:
        """<answer> present but no <think> tag."""
        text = "<answer>(A)</answer>"
        assert extract_answer_from_text(text, "mcq") == "A"
        assert _compute_tag_format_reward(text, "mcq") == -1.0  # needs both tags

    def test_multiple_letters_in_reasoning(self) -> None:
        """Think section mentions multiple letters but answer is clear."""
        text = "<think>Option A shows realism, B shows impressionism. B matches.</think><answer>(B)</answer>"
        assert extract_answer_from_text(text, "mcq") == "B"
        assert _compute_tag_format_reward(text, "mcq") == 0.5


class TestCountingEndToEnd:
    """Test all realistic counting outputs from Qwen2.5-VL.

    For each output pattern, verify:
    1. Extraction recovers the number (liberal, incl. words)
    2. Correctness matching works (exact integer)
    3. Format reward is strict: only <think>...<answer>N</answer> gets +0.5
    """

    # --- Perfect format: <think>...<answer>6</answer> ---

    def test_perfect_format_correct(self) -> None:
        text = "<think>I count 6 people in the image.</think><answer>6</answer>"
        assert extract_answer_from_text(text, "counting") == "6"
        assert match_answer("6", "6", "counting") is True
        assert _compute_tag_format_reward(text, "counting") == 0.5

    def test_perfect_format_wrong(self) -> None:
        text = "<think>I see 5 people.</think><answer>5</answer>"
        assert extract_answer_from_text(text, "counting") == "5"
        assert match_answer("5", "6", "counting") is False
        assert _compute_tag_format_reward(text, "counting") == 0.5  # format fine

    def test_perfect_format_two_digit(self) -> None:
        text = "<think>There are 10 birds.</think><answer>10</answer>"
        assert extract_answer_from_text(text, "counting") == "10"
        assert match_answer("10", "10", "counting") is True
        assert _compute_tag_format_reward(text, "counting") == 0.5

    # --- Tags present but bad content inside <answer> ---

    def test_tags_number_word_inside(self) -> None:
        """<answer>six</answer> — word instead of digit."""
        text = "<think>I count six people.</think><answer>six</answer>"
        assert extract_answer_from_text(text, "counting") == "6"  # extraction works
        assert _compute_tag_format_reward(text, "counting") == -0.5  # format fails

    def test_tags_number_with_text_inside(self) -> None:
        """<answer>6 people</answer> — extra text."""
        text = "<think>reason</think><answer>6 people</answer>"
        assert extract_answer_from_text(text, "counting") == "6"  # extraction works
        assert _compute_tag_format_reward(text, "counting") == -0.5  # format fails

    def test_tags_sentence_inside(self) -> None:
        """<answer>The count is 6</answer> — sentence."""
        text = "<think>reason</think><answer>The count is 6</answer>"
        assert extract_answer_from_text(text, "counting") == "6"
        assert _compute_tag_format_reward(text, "counting") == -0.5

    def test_tags_float_inside(self) -> None:
        """<answer>6.0</answer> — float instead of int."""
        text = "<think>reason</think><answer>6.0</answer>"
        assert extract_answer_from_text(text, "counting") == "6.0"
        assert _compute_tag_format_reward(text, "counting") == -0.5  # not bare int

    def test_tags_empty(self) -> None:
        text = "<think>reason</think><answer></answer>"
        assert extract_answer_from_text(text, "counting") == ""
        assert _compute_tag_format_reward(text, "counting") == -0.5

    def test_tags_mcq_letter_inside(self) -> None:
        """<answer>(D)</answer> — model confused, outputs MCQ letter for counting."""
        text = "<think>I think 4 cups.</think><answer>(D)</answer>"
        # Tag extraction gets "(D)", numeric extraction finds no digit -> ""
        assert extract_answer_from_text(text, "counting") == ""
        assert _compute_tag_format_reward(text, "counting") == -0.5  # (D) is not int

    # --- No tags at all ---

    def test_no_tags_bare_number(self) -> None:
        text = "6"
        assert extract_answer_from_text(text, "counting") == "6"
        assert _compute_tag_format_reward(text, "counting") == -1.0

    def test_no_tags_prose_with_number(self) -> None:
        text = "The count is 6."
        assert extract_answer_from_text(text, "counting") == "6"
        assert _compute_tag_format_reward(text, "counting") == -1.0

    def test_no_tags_number_word(self) -> None:
        text = "There are three cups in the image."
        assert extract_answer_from_text(text, "counting") == "3"
        assert _compute_tag_format_reward(text, "counting") == -1.0

    def test_no_tags_verbose(self) -> None:
        """Long verbose response — model just explains."""
        text = "In the image, there are two people in the foreground and one person partially visible in the background, making a total of three people."
        assert extract_answer_from_text(text, "counting") == "2"  # first digit found
        assert _compute_tag_format_reward(text, "counting") == -1.0

    # --- Edge cases ---

    def test_counting_zero(self) -> None:
        text = "<think>No objects visible.</think><answer>0</answer>"
        assert extract_answer_from_text(text, "counting") == "0"
        assert match_answer("0", "0", "counting") is True
        assert _compute_tag_format_reward(text, "counting") == 0.5

    def test_counting_exact_match_required(self) -> None:
        """Counting uses tolerance=0.0 — no rounding."""
        assert match_answer("5", "6", "counting") is False
        assert match_answer("6", "6", "counting") is True

    def test_counting_word_to_number_all(self) -> None:
        """Verify all supported number words."""
        for word, digit in [
            ("zero", "0"), ("one", "1"), ("two", "2"), ("three", "3"),
            ("four", "4"), ("five", "5"), ("six", "6"), ("seven", "7"),
            ("eight", "8"), ("nine", "9"), ("ten", "10"),
        ]:
            assert extract_answer_from_text(word, "counting") == digit, f"Failed for {word}"

    def test_numeric_does_not_match_words(self) -> None:
        """Numeric type should NOT match number words — only counting does."""
        assert extract_answer_from_text("three", "numeric") == ""
        assert extract_answer_from_text("three", "counting") == "3"


class TestCorrectnessMatching:
    """Test match_answer for all answer types."""

    # MCQ
    def test_mcq_match(self) -> None:
        assert match_answer("a", "a", "mcq") is True

    def test_mcq_no_match(self) -> None:
        assert match_answer("a", "b", "mcq") is False

    def test_mcq_empty(self) -> None:
        assert match_answer("", "a", "mcq") is None

    # Counting
    def test_counting_exact(self) -> None:
        assert match_answer("6", "6", "counting") is True

    def test_counting_off_by_one(self) -> None:
        assert match_answer("5", "6", "counting") is False

    def test_counting_string_match(self) -> None:
        assert match_answer("10", "10", "counting") is True

    def test_counting_non_numeric(self) -> None:
        """Non-parseable string."""
        assert match_answer("many", "6", "counting") is False

    # Numeric (with tolerance)
    def test_numeric_exact(self) -> None:
        assert match_answer("3.14", "3.14", "numeric") is True

    def test_numeric_within_tolerance(self) -> None:
        assert match_answer("3.14", "3.15", "numeric") is True  # within 1%

    def test_numeric_outside_tolerance(self) -> None:
        assert match_answer("3", "6", "numeric") is False
