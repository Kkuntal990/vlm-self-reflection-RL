"""Unit tests for Bug #3: a2_reward_mean must not be diluted by gated zeros.

Under PAG selective revision, gated trajectories (F1 said CORRECT → A2 skipped)
carry a 0.0 placeholder in a2_rewards_t. Naive mean over all N·K would depress
the headline a2_reward_mean metric as F1's gate rate climbs, disconnecting the
W&B chart from actual A2 quality.

The fix in critic_grpo.py:2016+ denominates the global a2_reward_mean over the
active (non-gated) trajectories only, using globally-correct
sum_across_ranks / count_across_ranks aggregation.
"""

from __future__ import annotations

import torch


def _compute_a2_reward_mean_pag(a2_rewards: torch.Tensor, a2_active_mask: torch.Tensor) -> float:
    """Mirror the PAG-path single-rank logic in critic_grpo._train_step."""
    a2_sum = float(a2_rewards[a2_active_mask].sum().item())
    a2_count = float(a2_active_mask.sum().item())
    return a2_sum / a2_count if a2_count > 0 else float("nan")


def _compute_a2_reward_mean_naive(a2_rewards: torch.Tensor) -> float:
    """Pre-fix behavior: mean over all rows including gated 0.0 placeholders."""
    return float(a2_rewards.mean().item())


def test_no_gating_matches_naive_mean():
    """When nothing is gated, the masked mean equals the naive mean."""
    rewards = torch.tensor([0.9, 0.6, 0.3, 1.0])
    mask = torch.tensor([True, True, True, True])
    assert _compute_a2_reward_mean_pag(rewards, mask) == _compute_a2_reward_mean_naive(rewards)


def test_half_gating_naive_dilutes_by_50pct():
    """6 of 12 trajectories gated → naive mean is half of masked mean
    when the 6 active rows all score 1.0. Demonstrates the bug magnitude.
    """
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    mask = torch.tensor([True] * 6 + [False] * 6)
    masked = _compute_a2_reward_mean_pag(rewards, mask)
    naive = _compute_a2_reward_mean_naive(rewards)
    assert masked == 1.0
    assert naive == 0.5
    assert masked - naive == 0.5  # naive depressed by exactly the gate rate × masked


def test_gate_rate_drift_doesnt_change_masked_mean():
    """Same per-active-trajectory A2 quality at two different gate rates.
    Masked mean is invariant; naive mean shifts with gate rate.
    """
    # Low gate rate: 2 of 12 gated, 10 active rows each scoring 0.6
    low = torch.tensor([0.6] * 10 + [0.0] * 2)
    low_mask = torch.tensor([True] * 10 + [False] * 2)
    # High gate rate: 8 of 12 gated, 4 active rows each scoring 0.6
    high = torch.tensor([0.6] * 4 + [0.0] * 8)
    high_mask = torch.tensor([True] * 4 + [False] * 8)

    masked_low = _compute_a2_reward_mean_pag(low, low_mask)
    masked_high = _compute_a2_reward_mean_pag(high, high_mask)
    naive_low = _compute_a2_reward_mean_naive(low)
    naive_high = _compute_a2_reward_mean_naive(high)

    # Masked metric is stable across gate-rate change
    assert abs(masked_low - masked_high) < 1e-6
    assert abs(masked_low - 0.6) < 1e-6
    # Naive metric mechanically moves with gate rate (the bug)
    assert naive_low > naive_high
    assert abs(naive_low - 0.5) < 1e-6  # 10 * 0.6 / 12
    assert abs(naive_high - 0.2) < 1e-6  # 4 * 0.6 / 12


def test_all_gated_returns_nan():
    """When every trajectory is gated there is no A2 signal — return NaN."""
    rewards = torch.tensor([0.0, 0.0, 0.0])
    mask = torch.tensor([False, False, False])
    import math

    assert math.isnan(_compute_a2_reward_mean_pag(rewards, mask))


def test_cross_rank_aggregation_formula():
    """Globally-correct mean = sum_across_ranks / count_across_ranks.

    The fix at critic_grpo.py:2025+ all_reduces a2_sum_local and a2_count_local
    separately and divides. This test mirrors that aggregation against
    the per-rank-mean-then-average formula it replaces, and verifies the
    new formula is correct when ranks have different gate rates.
    """
    # Rank 0: 8 active, mean 0.8 → sum=6.4, count=8
    # Rank 1: 2 active, mean 0.2 → sum=0.4, count=2
    # True global mean over the 10 active trajectories = 6.8 / 10 = 0.68
    rank0_sum, rank0_count = 6.4, 8.0
    rank1_sum, rank1_count = 0.4, 2.0

    # The (incorrect) old method — mean of per-rank means — would compute:
    old_method = (rank0_sum / rank0_count + rank1_sum / rank1_count) / 2  # 0.5
    # The (correct) new method — sum then divide:
    new_method = (rank0_sum + rank1_sum) / (rank0_count + rank1_count)  # 0.68

    assert abs(old_method - 0.5) < 1e-6
    assert abs(new_method - 0.68) < 1e-6
    # The two diverge whenever gate rates differ across ranks (the typical case)
    assert abs(new_method - old_method) > 0.15
