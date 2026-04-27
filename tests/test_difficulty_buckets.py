#!/usr/bin/env python3
"""Tests for the A1-difficulty bucket predicate.

The bucketing decides which prompts get dropped from training; off-by-one
mistakes here would silently corrupt the training set, so the predicate is
worth pinning with explicit cases for K=8 (production) and K=16 (sanity).
"""

import importlib.util
import pathlib

# Load the script as a module without exposing it as a package import path.
_SCRIPT_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "scripts"
    / "analysis"
    / "difficulty_buckets.py"
)
_spec = importlib.util.spec_from_file_location("difficulty_buckets", _SCRIPT_PATH)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_bucket_for = _mod._bucket_for


class TestBucketForK8:
    """K=8 is the production setting. These thresholds match the buckets
    documented in the script docstring (>=7 trivial, 4..6 easy, 1..3 medium,
    0 brick_wall).
    """

    def test_zero_correct_is_brick_wall(self) -> None:
        assert _bucket_for(0.0, 8) == "brick_wall"

    def test_one_correct_is_medium(self) -> None:
        assert _bucket_for(1 / 8, 8) == "medium"

    def test_three_correct_is_medium(self) -> None:
        assert _bucket_for(3 / 8, 8) == "medium"

    def test_four_correct_is_easy(self) -> None:
        assert _bucket_for(4 / 8, 8) == "easy"

    def test_six_correct_is_easy(self) -> None:
        assert _bucket_for(6 / 8, 8) == "easy"

    def test_seven_correct_is_trivial(self) -> None:
        assert _bucket_for(7 / 8, 8) == "trivial"

    def test_eight_correct_is_trivial(self) -> None:
        assert _bucket_for(1.0, 8) == "trivial"


class TestBucketForK16:
    """Predicate must scale with K (thresholds are fractions of K)."""

    def test_zero_correct_is_brick_wall(self) -> None:
        assert _bucket_for(0.0, 16) == "brick_wall"

    def test_one_correct_is_medium(self) -> None:
        assert _bucket_for(1 / 16, 16) == "medium"

    def test_seven_correct_is_medium(self) -> None:
        # 7/16 = 0.4375 < 8/16, still medium
        assert _bucket_for(7 / 16, 16) == "medium"

    def test_eight_correct_is_easy(self) -> None:
        assert _bucket_for(8 / 16, 16) == "easy"

    def test_fourteen_correct_is_trivial(self) -> None:
        # 14/16 == 7/8 → trivial threshold
        assert _bucket_for(14 / 16, 16) == "trivial"

    def test_sixteen_correct_is_trivial(self) -> None:
        assert _bucket_for(1.0, 16) == "trivial"


class TestEdgeCases:
    """Defensive: malformed K must not raise; empty rows should be brick_wall."""

    def test_zero_k_returns_brick_wall(self) -> None:
        assert _bucket_for(0.0, 0) == "brick_wall"

    def test_negative_k_returns_brick_wall(self) -> None:
        assert _bucket_for(0.5, -1) == "brick_wall"
