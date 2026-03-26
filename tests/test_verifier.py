#!/usr/bin/env python3
"""Tests for structured answer verification (verifier.py)."""

import pytest

from vlm_grpo.rewards.verifier import (  # noqa: I001
    CORRECT,
    DETERMINISTIC_TYPES,
    WRONG,
    MatchResult,
    verify_answer,
)

# =============================================================================
# MatchResult dataclass
# =============================================================================


class TestMatchResult:
    """Tests for MatchResult properties and methods."""

    def test_is_correct(self) -> None:
        r = MatchResult("mcq", True, CORRECT, "A", None)
        assert r.is_correct is True
        assert r.is_wrong is False

    def test_is_wrong(self) -> None:
        r = MatchResult("mcq", True, WRONG, "B", None)
        assert r.is_correct is False
        assert r.is_wrong is True

    def test_to_dict(self) -> None:
        r = MatchResult("mcq", True, CORRECT, "A", None)
        d = r.to_dict()
        assert d["answer_type"] == "mcq"
        assert d["parse_ok"] is True
        assert d["verdict"] == CORRECT
        assert d["extracted"] == "A"
        assert d["score"] is None

    def test_deterministic_types(self) -> None:
        assert "mcq" in DETERMINISTIC_TYPES
        assert "yesno" in DETERMINISTIC_TYPES
        assert "numeric" in DETERMINISTIC_TYPES
        assert "open" not in DETERMINISTIC_TYPES


# =============================================================================
# MCQ verification
# =============================================================================


class TestMCQVerification:
    """Tests for MCQ answer verification."""

    def test_correct(self) -> None:
        r = verify_answer("A", "A", "mcq")
        assert r.verdict == CORRECT
        assert r.parse_ok is True

    def test_wrong(self) -> None:
        r = verify_answer("B", "A", "mcq")
        assert r.verdict == WRONG
        assert r.parse_ok is True

    def test_empty_is_wrong(self) -> None:
        r = verify_answer("", "A", "mcq")
        assert r.verdict == WRONG
        assert r.parse_ok is False

    def test_garbled_is_wrong(self) -> None:
        r = verify_answer("asdf jkl;", "A", "mcq")
        assert r.verdict == WRONG

    def test_parens_correct(self) -> None:
        r = verify_answer("(A)", "A", "mcq")
        assert r.verdict == CORRECT
        assert r.extracted == "A"


# =============================================================================
# YesNo verification
# =============================================================================


class TestYesNoVerification:
    """Tests for YesNo answer verification."""

    def test_correct(self) -> None:
        r = verify_answer("Yes", "yes", "yesno")
        assert r.verdict == CORRECT

    def test_wrong(self) -> None:
        r = verify_answer("No", "yes", "yesno")
        assert r.verdict == WRONG

    def test_empty_is_wrong(self) -> None:
        r = verify_answer("", "yes", "yesno")
        assert r.verdict == WRONG
        assert r.parse_ok is False

    def test_hedging_is_wrong(self) -> None:
        r = verify_answer("maybe yes maybe no", "yes", "yesno")
        assert r.verdict == WRONG


# =============================================================================
# Numeric verification
# =============================================================================


class TestNumericVerification:
    """Tests for numeric answer verification."""

    def test_correct(self) -> None:
        r = verify_answer("3.14", "3.14", "numeric")
        assert r.verdict == CORRECT

    def test_wrong(self) -> None:
        r = verify_answer("5", "3.14", "numeric")
        assert r.verdict == WRONG

    def test_empty_is_wrong(self) -> None:
        r = verify_answer("", "3.14", "numeric")
        assert r.verdict == WRONG
        assert r.parse_ok is False

    def test_tolerance(self) -> None:
        r = verify_answer("3.14", "3.15", "numeric", tolerance=0.02)
        assert r.verdict == CORRECT

    def test_fraction(self) -> None:
        r = verify_answer("1/2", "0.5", "numeric")
        assert r.verdict == CORRECT


# =============================================================================
# Open-ended verification (deterministic stages only -- no embedding)
# =============================================================================


class TestOpenEndedDeterministicStages:
    """Tests for open-ended cascade stages that don't need embeddings."""

    def test_exact_match(self) -> None:
        r = verify_answer("cat", "cat", "open")
        assert r.verdict == CORRECT
        assert r.score == 1.0

    def test_case_insensitive_match(self) -> None:
        r = verify_answer("Cat", "cat", "open")
        assert r.verdict == CORRECT

    def test_substring_containment(self) -> None:
        r = verify_answer("a domestic cat", "cat", "open")
        assert r.verdict == CORRECT
        assert r.score == 0.95

    def test_token_f1_match(self) -> None:
        """High token overlap should match."""
        r = verify_answer("red sports car", "red car sports", "open")
        assert r.verdict == CORRECT

    def test_anls_match(self) -> None:
        """Small edit distance should match via ANLS."""
        r = verify_answer("colur", "color", "open")
        assert r.verdict == CORRECT

    def test_sentence_level_color_conflict_is_wrong(self) -> None:
        r = verify_answer(
            "The little girl's dress is red.",
            "The little girl's dress is pink.",
            "open",
        )
        assert r.verdict == WRONG


# =============================================================================
# Open-ended verification (embedding stage)
# =============================================================================


class TestOpenEndedEmbeddingStage:
    """Tests for embedding-based verification.

    These tests require sentence-transformers to be installed.
    """

    @pytest.fixture(autouse=True)
    def _check_sentence_transformers(self) -> None:
        """Skip if sentence-transformers is not installed."""
        pytest.importorskip("sentence_transformers")

    def test_synonym_correct(self) -> None:
        r = verify_answer("automobile", "car", "open")
        assert r.verdict == CORRECT

    def test_unrelated_wrong(self) -> None:
        r = verify_answer("airplane", "cat", "open")
        assert r.verdict == WRONG

    def test_short_phrase_mismatch_does_not_pass_via_anls(self) -> None:
        """Completely different multi-word answers should not pass ANLS."""
        r = verify_answer("red sports car", "blue wooden table", "open")
        assert r.verdict == WRONG


# =============================================================================
# Antonym contradiction detection
# =============================================================================


class TestAntonymContradiction:
    """Tests for antonym contradiction detection in open-ended matching.

    Sentences that differ only by an antonym pair (left/right, yes/no, etc.)
    should be marked WRONG despite high surface similarity.
    """

    def test_left_right_contradiction(self) -> None:
        """Descriptive VQA: 'left' vs 'right' must be WRONG."""
        r = verify_answer(
            "The pepper is on the left side of the image.",
            "The pepper is on the right of the image.",
            "open",
        )
        assert r.verdict == WRONG

    def test_above_below_contradiction(self) -> None:
        r = verify_answer(
            "The cat is above the table.",
            "The cat is below the table.",
            "open",
        )
        assert r.verdict == WRONG

    def test_top_bottom_contradiction(self) -> None:
        r = verify_answer(
            "The text is at the top of the image.",
            "The text is at the bottom of the image.",
            "open",
        )
        assert r.verdict == WRONG

    def test_no_false_positive_on_elaboration(self) -> None:
        """Adding extra words (not antonyms) should still match."""
        r = verify_answer(
            "The pepper is on the right side of the image.",
            "The pepper is on the right of the image.",
            "open",
        )
        assert r.verdict == CORRECT

    def test_increase_decrease_contradiction(self) -> None:
        r = verify_answer(
            "The trend shows an increase over time.",
            "The trend shows a decrease over time.",
            "open",
        )
        assert r.verdict == WRONG
