#!/usr/bin/env python3
"""Tests for deterministic answer matching."""

from vlm_grpo.rewards.deterministic import (
    match_answer,
    match_mcq,
    match_numeric,
    match_yesno,
)
from vlm_grpo.utils import normalized_edit_distance

# =============================================================================
# Answer matching tests
# =============================================================================


class TestMatchMCQ:
    """Tests for match_mcq() with pre-normalized (lowercase) inputs."""

    def test_exact_match(self) -> None:
        assert match_mcq("a", "a") is True

    def test_different_letters(self) -> None:
        assert match_mcq("a", "b") is False

    def test_same_letter(self) -> None:
        assert match_mcq("c", "c") is True

    def test_empty_predicted(self) -> None:
        assert match_mcq("", "a") is False

    def test_non_letter(self) -> None:
        assert match_mcq("1", "a") is False


class TestMatchYesNo:
    """Tests for match_yesno() with pre-normalized (lowercase) inputs."""

    def test_yes_yes(self) -> None:
        assert match_yesno("yes", "yes") is True

    def test_no_no(self) -> None:
        assert match_yesno("no", "no") is True

    def test_yes_no(self) -> None:
        assert match_yesno("yes", "no") is False

    def test_empty(self) -> None:
        assert match_yesno("", "yes") is False


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
    """Tests for match_answer() with pre-normalized (lowercase) inputs."""

    def test_mcq_correct(self) -> None:
        assert match_answer("a", "a", "mcq") is True

    def test_mcq_wrong(self) -> None:
        assert match_answer("a", "b", "mcq") is False

    def test_yesno_correct(self) -> None:
        assert match_answer("yes", "yes", "yesno") is True

    def test_numeric_correct(self) -> None:
        assert match_answer("42", "42", "numeric") is True

    def test_open_exact(self) -> None:
        assert match_answer("a cat", "a cat", "open") is True

    def test_open_different(self) -> None:
        result = match_answer("a cat", "a dog", "open")
        assert result is None

    def test_empty_predicted(self) -> None:
        assert match_answer("", "a", "mcq") is None


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
