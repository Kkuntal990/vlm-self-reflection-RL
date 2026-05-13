#!/usr/bin/env python3
"""Standalone verifier: HF logprobs match vLLM sample-time logprobs only when the
sampling temperature divisor is applied to HF shift_logits before log_softmax.

Demonstrates Bug #1 (A2 PPO ratio temperature mismatch) and confirms the fix.

vLLM at sample time returns ``Logprob.logprob`` = ``log softmax(logits / T)`` where
T is the SamplingParams temperature. For A2 we sample at T=0.7. The GRPO PPO ratio
``exp(new_lp - old_lp)`` is only well-defined when ``new_lp`` is computed on the
SAME distribution. Computing HF ``new_lp`` at T=1.0 (the default) biases the
ratio away from 1.0 at step 0 and attenuates A2 policy gradient.

Run:
    uv run python scripts/verify/verify_temperature_consistency.py
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F


def vllm_style_logprob(
    logits: torch.Tensor, sampled_ids: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Mimic what vLLM's sampler returns for ``Logprob.logprob`` at temperature T."""
    return (
        F.log_softmax(logits / temperature, dim=-1)
        .gather(-1, sampled_ids.unsqueeze(-1))
        .squeeze(-1)
    )


def hf_buggy_logprob(logits: torch.Tensor, sampled_ids: torch.Tensor) -> torch.Tensor:
    """Pre-fix HF code path: ignores temperature -> mismatched distribution."""
    return F.log_softmax(logits, dim=-1).gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)


def hf_fixed_logprob(
    logits: torch.Tensor, sampled_ids: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Post-fix HF code path: divides logits by sampling temperature before log_softmax."""
    return (
        F.log_softmax(logits / temperature, dim=-1)
        .gather(-1, sampled_ids.unsqueeze(-1))
        .squeeze(-1)
    )


def main() -> int:
    torch.manual_seed(0)
    vocab = 32
    seq_len = 5
    # Synthesise logits with a few peaks to make the temperature effect visible.
    logits = torch.randn(seq_len, vocab) * 3.0
    sampled = torch.randint(0, vocab, (seq_len,))

    failures: list[str] = []

    print("=" * 70)
    print("Temperature consistency check: HF new_lp vs vLLM old_lp")
    print("=" * 70)

    for label, temp in [("A1", 1.0), ("F1", 1.0), ("A2", 0.7)]:
        vllm_lp = vllm_style_logprob(logits, sampled, temperature=temp)
        hf_fixed = hf_fixed_logprob(logits, sampled, temperature=temp)
        hf_buggy = hf_buggy_logprob(logits, sampled)
        gap_fixed = (hf_fixed - vllm_lp).abs().max().item()
        gap_buggy = (hf_buggy - vllm_lp).abs().max().item()
        ratio_drift_buggy = (hf_buggy - vllm_lp).exp().mean().item()
        ratio_drift_fixed = (hf_fixed - vllm_lp).exp().mean().item()

        print(
            f"\n  {label} turn  (T={temp})\n"
            f"    max |hf_fixed - vllm| = {gap_fixed:.2e}   (target: ~0)\n"
            f"    max |hf_buggy - vllm| = {gap_buggy:.2e}   ({'bug visible' if gap_buggy > 1e-6 else 'noop (T=1)'})\n"
            f"    mean exp(new-old) fixed = {ratio_drift_fixed:.4f}   (target: 1.0)\n"
            f"    mean exp(new-old) buggy = {ratio_drift_buggy:.4f}"
        )

        if gap_fixed > 1e-6:
            failures.append(f"{label}: fixed path diverges from vLLM by {gap_fixed:.2e}")
        if temp != 1.0 and gap_buggy < 1e-3:
            failures.append(
                f"{label}: buggy path looks identical to fixed at T={temp} — verifier itself is broken"
            )

    print("\n" + "=" * 70)
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS — HF logprobs match vLLM logprobs when temperature divisor is applied.")
    print("       Pre-fix A2 path diverges from vLLM (mean ratio shifts away from 1.0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
