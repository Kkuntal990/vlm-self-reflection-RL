#!/usr/bin/env python3
"""Tests for the Dr.GRPO-normalized k3 KL estimator (`_kl_term_drgrpo`).

The k3 estimator is `exp(Δ) − Δ − 1` where Δ = log_ref − log_new
(Schulman 2020). It is unbiased and always non-negative. This helper
aggregates with `sum / max_len` to match the Dr.GRPO policy-loss
normalization (arXiv:2503.20783) — the previous `.mean()` aggregation
was length-biased and silently weakened the A1 anchor on long
completions.
"""

import math

import torch

from vlm_grpo.critic_grpo import _kl_term_drgrpo


class TestZeroKL:
    """When the policy and reference distributions are identical, KL must
    be exactly zero (not just numerically small) — this anchors all the
    other test cases."""

    def test_identical_lps_returns_zero(self) -> None:
        ref = torch.tensor([-1.0, -2.0, -0.5, -3.0])
        new = torch.tensor([-1.0, -2.0, -0.5, -3.0])
        kl = _kl_term_drgrpo(ref, new, max_len=10.0)
        assert kl.item() == 0.0

    def test_identical_with_grad_tracking_on_new(self) -> None:
        # `new_lps` carries grad in production; result still zero.
        ref = torch.tensor([-1.0, -2.0, -0.5])
        new = torch.tensor([-1.0, -2.0, -0.5], requires_grad=True)
        kl = _kl_term_drgrpo(ref, new, max_len=5.0)
        assert kl.item() == 0.0


class TestNonNegativity:
    """k3 is provably non-negative for any input (Schulman 2020)."""

    def test_positive_drift(self) -> None:
        ref = torch.tensor([-1.0, -1.0, -1.0])
        new = torch.tensor([-2.0, -2.0, -2.0])
        kl = _kl_term_drgrpo(ref, new, max_len=10.0)
        assert kl.item() > 0.0

    def test_negative_drift(self) -> None:
        # Reverse drift direction — KL is still non-negative.
        ref = torch.tensor([-2.0, -2.0, -2.0])
        new = torch.tensor([-1.0, -1.0, -1.0])
        kl = _kl_term_drgrpo(ref, new, max_len=10.0)
        assert kl.item() > 0.0

    def test_mixed_drift(self) -> None:
        ref = torch.tensor([-1.0, -2.0, -3.0, -1.5])
        new = torch.tensor([-2.0, -1.0, -2.5, -3.0])
        kl = _kl_term_drgrpo(ref, new, max_len=20.0)
        assert kl.item() >= 0.0


class TestDrGRPONormalization:
    """Dr.GRPO normalization (`sum / max_len`) is the central design fix
    — it ensures the KL gradient and the policy gradient share the same
    per-token scale, so `kl_coeff * a1_kl_coeff` is a true gradient-magnitude
    knob rather than a length-coupled one."""

    def test_normalization_is_sum_over_max_len(self) -> None:
        # 4 tokens, each contributing the same per-token KL.
        ref = torch.tensor([-1.0, -1.0, -1.0, -1.0])
        new = torch.tensor([-2.0, -2.0, -2.0, -2.0])
        max_len = 10.0
        kl = _kl_term_drgrpo(ref, new, max_len=max_len)

        # k3 per token: exp(Δ) − Δ − 1 with Δ = -1 - (-2) = 1
        per_tok = math.exp(1.0) - 1.0 - 1.0
        expected = (4 * per_tok) / max_len
        assert abs(kl.item() - expected) < 1e-5

    def test_short_completion_does_not_inflate_kl_per_max_len(self) -> None:
        # Two completions with the SAME per-token Δ: the short one used to
        # produce a larger `.mean()` KL than the long one (length bias).
        # Under sum/max_len with the same max_len, they contribute
        # proportionally to their actual length — short=less, long=more.
        ref = torch.tensor([-1.0, -1.0])
        new = torch.tensor([-2.0, -2.0])
        kl_short = _kl_term_drgrpo(ref, new, max_len=10.0)

        ref_long = torch.tensor([-1.0] * 8)
        new_long = torch.tensor([-2.0] * 8)
        kl_long = _kl_term_drgrpo(ref_long, new_long, max_len=10.0)

        # Long completion now contributes 4× more KL (8/2 tokens), instead
        # of being equal under .mean() (which averaged out the difference).
        assert kl_long.item() > kl_short.item()
        assert abs(kl_long.item() / kl_short.item() - 4.0) < 1e-5

    def test_doubling_max_len_halves_kl(self) -> None:
        # Same per-token KL, doubling max_len should halve the result.
        ref = torch.tensor([-1.0, -1.0, -1.0])
        new = torch.tensor([-2.0, -2.0, -2.0])
        kl_a = _kl_term_drgrpo(ref, new, max_len=10.0)
        kl_b = _kl_term_drgrpo(ref, new, max_len=20.0)
        assert abs(kl_b.item() * 2.0 - kl_a.item()) < 1e-6


class TestNumericalSafety:
    """Defensive: reference and policy log-probs may carry NaN/inf from
    masked positions or extreme model outputs. Helper must not propagate."""

    def test_nan_in_input_clamped(self) -> None:
        ref = torch.tensor([-1.0, float("nan"), -1.0])
        new = torch.tensor([-2.0, -2.0, -2.0])
        kl = _kl_term_drgrpo(ref, new, max_len=10.0)
        assert math.isfinite(kl.item())

    def test_posinf_in_diff_clamped(self) -> None:
        # Δ = +∞ would explode exp(Δ); helper clamps to +20 instead.
        ref = torch.tensor([float("inf"), -1.0])
        new = torch.tensor([-2.0, -2.0])
        kl = _kl_term_drgrpo(ref, new, max_len=10.0)
        assert math.isfinite(kl.item())

    def test_neginf_in_diff_clamped(self) -> None:
        ref = torch.tensor([float("-inf"), -1.0])
        new = torch.tensor([-2.0, -2.0])
        kl = _kl_term_drgrpo(ref, new, max_len=10.0)
        assert math.isfinite(kl.item())

    def test_extreme_drift_clamped(self) -> None:
        # |Δ| = 100 must clamp to 20 to keep exp() finite.
        ref = torch.tensor([50.0])
        new = torch.tensor([-50.0])
        kl = _kl_term_drgrpo(ref, new, max_len=10.0)
        assert math.isfinite(kl.item())

    def test_empty_input_returns_zero(self) -> None:
        ref = torch.tensor([])
        new = torch.tensor([])
        kl = _kl_term_drgrpo(ref, new, max_len=10.0)
        assert kl.item() == 0.0


class TestGradientFlow:
    """The KL term is the gradient backbone of the SCoRe Stage I anchor —
    grad MUST flow through `new_lps` into the policy parameters. This is
    the property that makes A1_KL_COEFF actually do anything."""

    def test_grad_flows_through_new_lps(self) -> None:
        ref = torch.tensor([-1.0, -1.0])
        new = torch.tensor([-2.0, -2.0], requires_grad=True)
        kl = _kl_term_drgrpo(ref, new, max_len=10.0)
        kl.backward()
        assert new.grad is not None
        assert torch.all(torch.isfinite(new.grad))
        # ∂(exp(Δ)−Δ−1)/∂new = -exp(Δ)+1; for Δ=1 this is 1-e ≈ -1.718.
        # Per-token grad / max_len = (1 - e) / 10 ≈ -0.1718.
        expected_per_tok = (1.0 - math.exp(1.0)) / 10.0
        assert abs(new.grad[0].item() - expected_per_tok) < 1e-5

    def test_no_grad_flows_through_ref_lps(self) -> None:
        # Reference lps must be detached upstream; helper's .detach() is
        # belt-and-suspenders. Even if ref carries grad, none should flow
        # back into it.
        ref = torch.tensor([-1.0, -1.0], requires_grad=True)
        new = torch.tensor([-2.0, -2.0], requires_grad=True)
        kl = _kl_term_drgrpo(ref, new, max_len=10.0)
        kl.backward()
        assert ref.grad is None or float(ref.grad.abs().sum()) == 0.0
