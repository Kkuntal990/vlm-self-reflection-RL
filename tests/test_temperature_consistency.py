"""Unit tests for Bug #1: HF logprob temperature divisor.

The trainer's HF forward pass must divide ``shift_logits`` by the per-turn
sampling temperature before ``log_softmax`` so that ``new_lp`` lives on the
same distribution as vLLM's sample-time ``old_lp``. Without this the A2 PPO
ratio is biased ≠ 1 at step 0 and the surrogate gradient is attenuated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _hf_log_softmax_with_temperature(
    shift_logits: torch.Tensor, sampling_temperature: float
) -> torch.Tensor:
    """Mirror the production divisor logic in
    ``critic_grpo._forward_from_pretokenized_multi``.
    """
    if sampling_temperature != 1.0:
        shift_logits = shift_logits / sampling_temperature
    return F.log_softmax(shift_logits, dim=-1)


def _vllm_style_logprob_at_temperature(
    logits: torch.Tensor, sampled_ids: torch.Tensor, temperature: float
) -> torch.Tensor:
    """vLLM returns ``log softmax(logits / T)`` for each sampled token."""
    return (
        F.log_softmax(logits / temperature, dim=-1)
        .gather(-1, sampled_ids.unsqueeze(-1))
        .squeeze(-1)
    )


def test_a1_temperature_noop_at_t1():
    """A1 sampled at T=1.0 — divisor is a no-op."""
    torch.manual_seed(1)
    logits = torch.randn(5, 32) * 2.0
    sampled = torch.randint(0, 32, (5,))
    hf = (
        _hf_log_softmax_with_temperature(logits, sampling_temperature=1.0)
        .gather(-1, sampled.unsqueeze(-1))
        .squeeze(-1)
    )
    vllm = _vllm_style_logprob_at_temperature(logits, sampled, temperature=1.0)
    assert torch.allclose(hf, vllm, atol=1e-6)


def test_f1_temperature_noop_at_t1():
    """F1 sampled at T=1.0 — divisor is a no-op."""
    torch.manual_seed(2)
    logits = torch.randn(7, 32) * 2.0
    sampled = torch.randint(0, 32, (7,))
    hf = (
        _hf_log_softmax_with_temperature(logits, sampling_temperature=1.0)
        .gather(-1, sampled.unsqueeze(-1))
        .squeeze(-1)
    )
    vllm = _vllm_style_logprob_at_temperature(logits, sampled, temperature=1.0)
    assert torch.allclose(hf, vllm, atol=1e-6)


def test_a2_temperature_divisor_applied_at_t07():
    """A2 sampled at T=0.7 — divisor MUST be applied for HF/vLLM agreement."""
    torch.manual_seed(3)
    logits = torch.randn(6, 32) * 3.0
    sampled = torch.randint(0, 32, (6,))
    hf_fixed = (
        _hf_log_softmax_with_temperature(logits, sampling_temperature=0.7)
        .gather(-1, sampled.unsqueeze(-1))
        .squeeze(-1)
    )
    vllm = _vllm_style_logprob_at_temperature(logits, sampled, temperature=0.7)
    assert torch.allclose(hf_fixed, vllm, atol=1e-6), (
        f"HF logprobs with T=0.7 divisor must match vLLM logprobs at T=0.7. "
        f"max gap = {(hf_fixed - vllm).abs().max().item():.2e}"
    )


def test_a2_buggy_path_diverges_from_vllm():
    """Without the divisor, HF (T=1) diverges from vLLM (T=0.7) — this is the bug."""
    torch.manual_seed(4)
    logits = torch.randn(6, 32) * 3.0
    sampled = torch.randint(0, 32, (6,))
    hf_buggy = (
        _hf_log_softmax_with_temperature(logits, sampling_temperature=1.0)
        .gather(-1, sampled.unsqueeze(-1))
        .squeeze(-1)
    )
    vllm = _vllm_style_logprob_at_temperature(logits, sampled, temperature=0.7)
    gap = (hf_buggy - vllm).abs().max().item()
    assert gap > 1e-3, (
        f"Expected the buggy path to disagree with vLLM at T=0.7. "
        f"If gap is {gap:.2e} ~ 0, the test is no longer exercising the bug."
    )


def test_ppo_ratio_unbiased_at_step_zero_after_fix():
    """PPO ratio exp(new_lp - old_lp) must equal 1.0 elementwise at step 0
    when the policy hasn't moved AND the temperature divisor is applied.
    """
    torch.manual_seed(5)
    logits = torch.randn(8, 32) * 2.5
    sampled = torch.randint(0, 32, (8,))
    for temp in (1.0, 0.7):
        old_lp = _vllm_style_logprob_at_temperature(logits, sampled, temperature=temp)
        new_lp = (
            _hf_log_softmax_with_temperature(logits, sampling_temperature=temp)
            .gather(-1, sampled.unsqueeze(-1))
            .squeeze(-1)
        )
        ratio = (new_lp - old_lp).exp()
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5), (
            f"PPO ratio at T={temp} step 0 should be all 1s, got "
            f"min={ratio.min().item():.4f} max={ratio.max().item():.4f}"
        )
