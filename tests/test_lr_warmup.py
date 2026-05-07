"""Tests for the linear LR warmup helper.

Verifies the math at boundary cases (step 0, mid-warmup, end-of-warmup,
post-warmup) and ensures the no-warmup default returns the peak LR
unchanged so non-warmup runs are bit-for-bit identical.
"""

from __future__ import annotations

import pytest

from vlm_grpo.utils import compute_warmup_lr


class TestNoWarmup:
    """warmup_steps=0 must return peak_lr at every step (regression guard)."""

    def test_returns_peak_at_step_zero(self) -> None:
        assert compute_warmup_lr(global_step=0, peak_lr=1e-5, warmup_steps=0) == 1e-5

    def test_returns_peak_at_arbitrary_step(self) -> None:
        for step in (1, 10, 100, 1000, 99999):
            assert compute_warmup_lr(global_step=step, peak_lr=1e-5, warmup_steps=0) == 1e-5

    def test_returns_peak_for_negative_warmup(self) -> None:
        # Defensive: warmup_steps < 0 should be treated like no-warmup.
        assert compute_warmup_lr(global_step=5, peak_lr=2e-5, warmup_steps=-1) == 2e-5


class TestLinearRamp:
    """warmup_steps=100, peak_lr=1e-5: full boundary table."""

    PEAK = 1e-5
    WARMUP = 100

    def test_step_0_is_zero(self) -> None:
        assert compute_warmup_lr(0, self.PEAK, self.WARMUP) == 0.0

    def test_step_50_is_half_peak(self) -> None:
        assert compute_warmup_lr(50, self.PEAK, self.WARMUP) == pytest.approx(5e-6)

    def test_step_99_is_just_under_peak(self) -> None:
        assert compute_warmup_lr(99, self.PEAK, self.WARMUP) == pytest.approx(9.9e-6)

    def test_step_100_is_full_peak(self) -> None:
        # Boundary: step == warmup_steps -> ramp complete.
        assert compute_warmup_lr(100, self.PEAK, self.WARMUP) == self.PEAK

    def test_step_500_still_full_peak(self) -> None:
        assert compute_warmup_lr(500, self.PEAK, self.WARMUP) == self.PEAK


class TestEdgeCases:
    """Single-step warmup and arbitrary peak scaling."""

    def test_warmup_steps_one_step_zero_is_zero(self) -> None:
        assert compute_warmup_lr(0, peak_lr=1e-5, warmup_steps=1) == 0.0

    def test_warmup_steps_one_step_one_is_full(self) -> None:
        assert compute_warmup_lr(1, peak_lr=1e-5, warmup_steps=1) == 1e-5

    def test_warmup_steps_one_step_2_is_full(self) -> None:
        assert compute_warmup_lr(2, peak_lr=1e-5, warmup_steps=1) == 1e-5

    def test_warmup_scales_with_peak_lr(self) -> None:
        # warmup_steps=10, peak_lr=1.0 -> step 5 is exactly 0.5
        assert compute_warmup_lr(5, peak_lr=1.0, warmup_steps=10) == pytest.approx(0.5)

    def test_warmup_with_zero_peak(self) -> None:
        # peak_lr=0 -> always 0, regardless of step.
        for step in (0, 50, 100, 500):
            assert compute_warmup_lr(step, peak_lr=0.0, warmup_steps=100) == 0.0
