"""Tests for GDPO per-component K-group advantage normalization.

Reference: Liu et al. (NVIDIA), "Group reward-Decoupled Normalization Policy
Optimization for Multi-reward RL Optimization" (2026), arXiv:2601.05242.

Equations 4–7:
  Eq 4: A_k_j = (r_k_j - mean_k_in_group) / (std_k_in_group + eps)
  Eq 5/7: A_sum_j = sum_k(w_k * A_k_j)
  Eq 6: Â_j = (A_sum_j - batch_mean) / (batch_std + eps)
"""

from __future__ import annotations

import pytest
import torch

from vlm_grpo.critic_grpo import SelfReflectionGRPOTrainer


def _stub_trainer() -> SelfReflectionGRPOTrainer:
    """Build a SelfReflectionGRPOTrainer instance with only the attributes the
    advantage helpers touch — bypassing __init__ since we don't need the model
    or optimizer for these unit tests.
    """
    inst = SelfReflectionGRPOTrainer.__new__(SelfReflectionGRPOTrainer)
    inst.device = "cpu"
    return inst


# --------------------------------------------------------------------------- #
# Test 1 — hand-computed example                                              #
# --------------------------------------------------------------------------- #
def test_gdpo_hand_computed_example_k4_two_components():
    """Hand-computed Eq.4-7 for K=4, 2 components, equal weights.

    Inputs:
      a2_correct: [0, 1, 0, 1]   mean=0.5, std(unbiased ddof=1)=0.5774
      a2_format:  [0, 0, 1, 1]   mean=0.5, std(unbiased ddof=1)=0.5774

    With torch.std default (Bessel correction, ddof=1):
      A1 (a2_correct) ≈ [-0.866, +0.866, -0.866, +0.866]
      A2 (a2_format)  ≈ [-0.866, -0.866, +0.866, +0.866]
      A_sum (w=1,1)   ≈ [-1.732, 0, 0, +1.732]
      batch_mean = 0
      batch_std  ≈ 1.4142
      Â           ≈ [-1.225, 0, 0, +1.225]
    """
    trainer = _stub_trainer()
    components = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    weights = torch.tensor([1.0, 1.0])

    out = trainer._compute_gdpo_advantages(components, weights, k=4)

    # Each per-component A_j has values approx ±1/std_unbiased ≈ ±√3 ≈ ±1.732
    # Sum: [-2/std, 0, 0, +2/std] = [-3.464, 0, 0, +3.464] / 1 = ...
    # Actually: per-component normalize A1 = (x - 0.5)/0.5774 with values
    #   x=0 -> -0.866, x=1 -> +0.866. Sum gives ±1.732 / 0.
    # Then batch_std of [-1.732, 0, 0, +1.732] = sqrt(2.0) * unbiased ddof=1
    #   = sqrt(sum_of_squared_deviations / (N-1)) = sqrt(6/3) = sqrt(2) ≈ 1.4142
    # final = [-1.732, 0, 0, +1.732] / 1.4142 ≈ [-1.225, 0, 0, +1.225]
    expected = torch.tensor([-1.2247, 0.0, 0.0, 1.2247])
    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-3), f"got {out}"


# --------------------------------------------------------------------------- #
# Test 2 — regression guard (use_gdpo_normalization=False is unchanged)       #
# --------------------------------------------------------------------------- #
def test_grpo_path_unchanged_when_gdpo_disabled():
    """When the flag is OFF, the trainer must use the existing GRPO advantage.
    This test calls _compute_group_advantages directly with a fixed reward
    array and verifies the output matches the expected GRPO formula
    (group-mean subtract, divide by std).
    """
    trainer = _stub_trainer()
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0])  # K=4, single group
    out = trainer._compute_group_advantages(rewards, k=4, loss_type="grpo")

    mean = rewards.mean()
    std = rewards.std()
    expected = (rewards - mean) / (std + 1e-8)
    assert torch.allclose(out, expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# Test 3 — std=0 component contributes 0 to advantage                         #
# --------------------------------------------------------------------------- #
def test_zero_std_component_contributes_zero():
    """Component 2 is saturated at 1.0 across the K-group (std=0). Its
    contribution to A_sum should be 0; the final advantage should match what
    you get with only the variant component.
    """
    trainer = _stub_trainer()
    components_with_zero = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    components_only = torch.tensor(
        [
            [0.0],
            [1.0],
            [0.0],
            [1.0],
        ]
    )
    weights_two = torch.tensor([1.0, 1.0])
    weights_one = torch.tensor([1.0])

    out_two = trainer._compute_gdpo_advantages(components_with_zero, weights_two, k=4)
    out_one = trainer._compute_gdpo_advantages(components_only, weights_one, k=4)

    # The std=0 component contributes 0 across the board, so the two outputs
    # should be identical.
    assert torch.allclose(out_two, out_one, atol=1e-6), f"two={out_two} vs one={out_one}"


# --------------------------------------------------------------------------- #
# Test 4 — all components std=0 -> all advantages 0                           #
# --------------------------------------------------------------------------- #
def test_all_components_zero_variance_all_zero():
    """If every reward component is constant across the K-group, no gradient
    signal exists; all advantages should be 0.
    """
    trainer = _stub_trainer()
    components = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    weights = torch.tensor([1.0, 1.0])

    out = trainer._compute_gdpo_advantages(components, weights, k=4)
    expected = torch.zeros(4)
    assert torch.allclose(out, expected, atol=1e-6), f"got {out}"


# --------------------------------------------------------------------------- #
# Test 5 — two heads independent                                              #
# --------------------------------------------------------------------------- #
def test_two_heads_independent():
    """Response head and feedback head should compute advantages independently.
    Calling _compute_gdpo_advantages with response data should not be
    affected by what feedback data exists (we test by calling twice with
    different inputs and verifying outputs are independent, not by mocking).
    """
    trainer = _stub_trainer()
    resp_components = torch.tensor(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    fb_components = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ]
    )
    weights = torch.tensor([1.0, 1.0])

    resp_out = trainer._compute_gdpo_advantages(resp_components, weights, k=4)
    fb_out = trainer._compute_gdpo_advantages(fb_components, weights, k=4)

    # Each output should have shape [4] and the two should not be equal.
    assert resp_out.shape == (4,)
    assert fb_out.shape == (4,)
    assert not torch.allclose(resp_out, fb_out, atol=1e-3), "two heads must be independent"

    # And: re-running resp with the same input must give the same output
    # (deterministic; no leakage from a previous call).
    resp_out_again = trainer._compute_gdpo_advantages(resp_components, weights, k=4)
    assert torch.allclose(resp_out, resp_out_again, atol=1e-9)


# --------------------------------------------------------------------------- #
# Test 6 — weight=0 component has no effect                                   #
# --------------------------------------------------------------------------- #
def test_weight_zero_component_has_no_effect():
    """The simplified-rewards run sets no_regression and downstream weights
    to 0. Even if those components have variance across the K-group, they
    must not affect the final advantage.
    """
    trainer = _stub_trainer()
    components = torch.tensor(
        [
            [0.0, 5.0],
            [1.0, -3.0],
            [0.0, 2.0],
            [1.0, 100.0],
        ]
    )
    weights_zeroed = torch.tensor([1.0, 0.0])  # second component zeroed
    weights_only = torch.tensor([1.0])
    components_only = components[:, :1]

    out_zeroed = trainer._compute_gdpo_advantages(components, weights_zeroed, k=4)
    out_only = trainer._compute_gdpo_advantages(components_only, weights_only, k=4)

    assert torch.allclose(out_zeroed, out_only, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
