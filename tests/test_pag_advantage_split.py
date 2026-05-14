#!/usr/bin/env python3
"""Tests for the PAG per-segment K-group advantage split.

The trainer's ``_compute_pag_a2_advantages`` helper computes A2's
group baseline ONLY over the trajectories that actually generated A2
(i.e. those NOT gated by F1). Gated trajectories receive advantage = 0
so their (empty) A2 completion contributes zero to the A2 policy loss
even via the K-group baseline subtraction.

These tests use ``SelfReflectionGRPOTrainer.__new__`` to instantiate
the class without running ``__init__`` (which would require a real
model + processor + config); the helper is a self-contained method
that only touches its arguments, so the lightweight stub is safe.
"""

import torch

from vlm_grpo.critic_grpo import SelfReflectionGRPOTrainer


def _make_trainer() -> SelfReflectionGRPOTrainer:
    """Bypass __init__ — the advantage helper only consults its args."""
    t = SelfReflectionGRPOTrainer.__new__(SelfReflectionGRPOTrainer)
    return t


def test_pag_a2_advantage_no_gating_matches_standard_baseline():
    """When no trajectory is gated, the result should match the standard
    K-group baseline (mean subtraction in dr_grpo mode).
    """
    trainer = _make_trainer()
    rewards = torch.tensor([0.9, 1.0, 2.0, 0.1])
    active = torch.tensor([True, True, True, True])

    adv = trainer._compute_pag_a2_advantages(rewards, active, k=4, loss_type="dr_grpo")

    expected_mean = rewards.mean()
    expected = rewards - expected_mean
    assert torch.allclose(adv, expected, atol=1e-6)


def test_pag_a2_advantage_excludes_gated_from_baseline():
    """The K-group mean must be computed over active samples only.

    Group of 4: 2 active (rewards 1.0 and 2.0), 2 gated (placeholders 0.0).
    Active mean = 1.5. Active advantages = -0.5, +0.5. Gated advantages = 0.
    """
    trainer = _make_trainer()
    rewards = torch.tensor([1.0, 0.0, 2.0, 0.0])  # gated entries = placeholder 0.0
    active = torch.tensor([True, False, True, False])

    adv = trainer._compute_pag_a2_advantages(rewards, active, k=4, loss_type="dr_grpo")

    # Active mean = (1 + 2) / 2 = 1.5
    assert adv[0].item() == -0.5  # 1.0 - 1.5 = -0.5
    assert adv[2].item() == 0.5  # 2.0 - 1.5 = +0.5
    assert adv[1].item() == 0.0  # gated
    assert adv[3].item() == 0.0  # gated


def test_pag_a2_advantage_all_gated_yields_zero():
    """Every trajectory in the K-group gated out at F1 → no A2 gradient.

    The group contributes zero to the policy update — correct behaviour
    when the model is uniformly confident enough to skip revision.
    """
    trainer = _make_trainer()
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0])  # all placeholders
    active = torch.tensor([False, False, False, False])

    adv = trainer._compute_pag_a2_advantages(rewards, active, k=4, loss_type="dr_grpo")

    assert torch.allclose(adv, torch.zeros(4))


def test_pag_a2_advantage_single_active_yields_zero():
    """K-group with one active sample → std=0 in the standard GRPO path,
    or just (reward - itself) = 0 in dr_grpo mode. Either way the single
    active sample gets 0 advantage (no within-group contrast).
    """
    trainer = _make_trainer()
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0])
    active = torch.tensor([True, False, False, False])

    adv = trainer._compute_pag_a2_advantages(rewards, active, k=4, loss_type="dr_grpo")

    # The lone active sample's advantage = 1.0 - 1.0 = 0.0.
    assert adv[0].item() == 0.0
    assert adv[1].item() == 0.0
    assert adv[2].item() == 0.0
    assert adv[3].item() == 0.0


def test_pag_a2_advantage_grpo_with_std_division():
    """Standard GRPO mode (mean + Bessel sample std, ddof=1). With 3 active
    samples of {0, 1, 2}: mean=1, std=sqrt(1)=1.0. Advantages: -1, 0, +1.
    Gated entry stays at zero. Uses default `.std()` (ddof=1) to match TRL
    GRPOTrainer + PAG verl reference.
    """
    trainer = _make_trainer()
    rewards = torch.tensor([0.0, 1.0, 2.0, 0.0])
    active = torch.tensor([True, True, True, False])

    adv = trainer._compute_pag_a2_advantages(rewards, active, k=4, loss_type="grpo")

    active_vals = rewards[active]
    mean = active_vals.mean()
    std = active_vals.std()  # ddof=1 (Bessel sample std)
    expected_0 = (rewards[0] - mean) / (std + 1e-8)
    expected_1 = (rewards[1] - mean) / (std + 1e-8)
    expected_2 = (rewards[2] - mean) / (std + 1e-8)

    assert torch.allclose(adv[0], expected_0, atol=1e-6)
    assert torch.allclose(adv[1], expected_1, atol=1e-6)
    assert torch.allclose(adv[2], expected_2, atol=1e-6)
    assert adv[3].item() == 0.0  # gated


def test_pag_a2_advantage_multiple_groups_independent():
    """K-group baselines must be computed independently per group.

    Two groups of 2: group 0 has {1, 3} (mean=2, advantages -1/+1),
    group 1 has {10, 20} (mean=15, advantages -5/+5). The advantages
    must NOT bleed across groups.
    """
    trainer = _make_trainer()
    rewards = torch.tensor([1.0, 3.0, 10.0, 20.0])
    active = torch.tensor([True, True, True, True])

    adv = trainer._compute_pag_a2_advantages(rewards, active, k=2, loss_type="dr_grpo")

    assert adv[0].item() == -1.0
    assert adv[1].item() == 1.0
    assert adv[2].item() == -5.0
    assert adv[3].item() == 5.0
