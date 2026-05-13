# GRPO Training Troubleshooting

Diagnostic methods and metrics for detecting whether GRPO training is actually
learning. Every section lists: what to check, how to check it, what healthy vs
broken looks like.

## Quick diagnostic flow

When metrics are flat, run these checks in order. Each one rules out or
confirms a specific failure mode.

1. **Weight-update probe** — is the optimizer actually changing parameters?
2. **Effective delta magnitude** — are updates large enough to change behavior?
3. **Zero-variance fraction** — is there any gradient signal per step?
4. **Reward/accuracy window trend** — is the signal actually driving change?
5. **vLLM weight sync** — are rollouts using the trained policy?

---

## 1. LoRA weight-update probe

**Symptom**: metrics flat across many steps; can't tell if gradients flow.

LoRA is initialized with `A ~ Gaussian(small)`, `B = 0`. Any optimizer step that
passes gradient to LoRA moves `B` away from zero. A still-zero `B` after
training means the optimizer never updated these params.

```python
# Run on a saved checkpoint's adapter_model.safetensors:
from safetensors.torch import load_file
sd = load_file("<ckpt>/adapter_model.safetensors")
B = sum((sd[k].float()**2).sum().item() for k in sd if "lora_B" in k)
nz = sum(1 for k in sd if "lora_B" in k and (sd[k]**2).sum() > 1e-12)
total = sum(1 for k in sd if "lora_B" in k)
print(f"B norm {B**0.5:.4f}  non-zero layers {nz}/{total}")
```

| Output | Interpretation |
|---|---|
| `B norm ~0.0`, `0/N nonzero` | **Optimizer not stepping.** Check `optimizer.step()` in train loop, `requires_grad` on LoRA, DDP grad-sync, or grad clipping to zero. |
| `B norm > 0`, `N/N nonzero` | Gradients flow. Go to §2. |

## 2. Effective LoRA delta magnitude

Even with `B > 0`, the effective weight perturbation is `(α/r) · B @ A`. If
this is orders of magnitude smaller than base weights, the model hasn't
behaviorally moved.

```python
# Same adapter file. For each layer pair A,B:
delta = (alpha / r) * (B @ A)             # B: (out, r), A: (r, in)
# Compare with base weight scale. Qwen2.5-VL-7B base weights are ~[0.01, 0.1]
# element-wise. A delta with max_abs < 1e-4 is ~1000× smaller than base.
```

| Max abs delta element | Interpretation |
|---|---|
| < 1e-4 | **LR too low or signal too weak.** Behaviorally no-op. Raise LR 5-10×. |
| 1e-4 to 1e-2 | Updates in the noticeable range. Behavioral change expected. |
| > 0.1 | Risk of catastrophic forgetting / mode collapse. |

**Sanity**: for LoRA RL on a 7B model, healthy runs typically hit
`max_abs ~ 1e-3` after a few hundred steps. Under that, training isn't
really training.

## 3. Zero-variance K-group fraction

GRPO's advantage = `reward - group_mean(reward)`. If all K trajectories for a
prompt produce the same reward, advantages are zero and no gradient flows for
that prompt.

```bash
grep -oE "frac_zero_std=[0-9.]+" <output>/training.log | tail -20
```

| frac_zero_std | Interpretation |
|---|---|
| < 0.2 | Healthy — most prompts produce gradient. |
| 0.3–0.5 | Lots of wasted compute but training still works. |
| > 0.5 | **Most prompts contribute nothing.** Enable SSR, raise K, raise rollout temperature, or filter dataset. |

## 4. Accuracy / reward window trend

The right thing to track is not mean `resp_reward` (which sums aggregates and
can mask movement) but per-window **A1 accuracy, A2 accuracy, and WR−RW**
across the training log.

```python
# In-log: extract 'a1_correct=X a2_correct=Y' per trajectory, bucket into
# 10 equal windows, compute accuracy per window.
```

| Pattern across 10 windows | Interpretation |
|---|---|
| A1/A2 climb monotonically, WR−RW trends positive | Healthy learning. |
| A1/A2 climb then collapse back to baseline | **Training instability.** Often α too high (±4 spikes), or KL not anchoring. |
| Flat throughout | No learning. Go to §1/§2. |
| A2 systematically below A1 | Refiner hurting; under PAG selective revision check `sr/sycophantic_gate_rate` — if F1 gates wrong A1s as CORRECT, A2 never gets a chance to recover. Also check `pag_shaping_alpha` and that the temperature-divisor fix (§5) is in place. |

## 5. vLLM weight sync

Training updates LoRA. vLLM has its own weight copy. If sync fails, rollouts
use stale weights and the policy effectively never changes.

**Check**: in `vllm_rollout.py`, confirm `merge_adapter()` → vLLM
`load_weights()` → `unmerge_adapter()` is called **every rollout step**, not
lazily.

**Also**: capture vLLM's returned logprobs at sample time, compare with
HF-path `new_lps` on the same tokens. Drift > 0.01 per token indicates
kernel/precision mismatch — or, more commonly, a temperature mismatch
between vLLM sampling and HF `log_softmax`. Run
`scripts/verify/verify_temperature_consistency.py` to confirm the divisor
is applied (Bug #1 fix); pre-fix, A2 PPO ratio drifts ~6.86× off 1.0 at
step 0.

---

## Known failure modes & fixes

| Failure | Indicator | Root cause | Fix |
|---|---|---|---|
| **Dead LoRA** | LoRA-B norm = 0 | optimizer not stepping LoRA params | Check param-group construction, DDP sync |
| **Toy updates** | Effective delta max abs < 1e-4 | LR too low for signal scale | Raise LR 5-10× |
| **Signal starvation** | frac_zero_std > 0.5 | K too small or reward too discrete | Raise K, raise temperature; (DAPO `--use_dynamic_sampling` was the old SSR knob — now off in both active runs) |
| **Shaped-reward spike domination** | RW/WR advantages dominate group, collapse after peak | α too high on binary rewards | α=1 (SCoRe paper / PAG default) |
| **Length bias** | Short answers get more gradient than long reasoning | Per-sample loss mean | Use Dr.GRPO `sum / max_comp_len` |
| **KL pinning** | kl_loss ≈ policy_loss in magnitude | KL coefficient too strong | Lower per-turn `*_kl_coeff` — uniform 220 across A1/A2/F1 is the current default; the legacy 2500× A1 anchor was tuned for anchor-only frozen-A1 mode and over-suppresses an actively-trained A1 |
| **Dataset noise** | Best run plateaus well below 80% | GT labels wrong or ambiguous | Run per-task VLM oracle audit, drop low-trust tasks |
| **vLLM/HF logprob drift at iter 0** | Non-zero clip_frac at inner epoch 0; ratio ≠ 1 on synthetic step-0 batch | vLLM samples at T<1, HF `new_lp` was computed at T=1 — distributions mismatch | **Fixed** in commit `c825376`: HF forward divides `shift_logits` by per-turn `sampling_temperature` before `log_softmax`. Use `scripts/verify/verify_temperature_consistency.py` to confirm. |
| **`reward/a2_mean` flat as gate rate climbs** | Headline metric drops even while conditional A2 quality improves | Naive `.mean()` over all N·K averages in `0.0` placeholders for gated trajectories | **Fixed** in commit `c825376`: denominate over `a2_active_mask` (PAG); cross-rank reduce sums and counts separately. Check `sr/r_a2_mean` (always correct) against `reward/a2_mean`; they should now agree. |
| **F1 sycophancy collapse** | `sr/sycophantic_gate_rate` climbs while `sr/verification_accuracy` stalls or drops | Under selective revision + `w_downstream=0`, F1 pays no cost for emitting `\boxed{CORRECT}` indiscriminately | Architectural — reintroduce a calibration signal (e.g. `w_downstream>0`, or a verdict-mismatch penalty on F1). Tracked under the "literature pivots" section of `CLAUDE.md`. |

---

## Metrics dashboard (what to plot in wandb)

Drop these at every step:

- `reward/resp_mean`, `reward/fb_mean` — aggregate
- `reward/a1_mean`, `reward/a2_mean` — per-turn (needs `separate_turn_loss=True`; under PAG, `a2_mean` is denominated over non-gated trajectories only)
- `sr/r_a1_mean`, `sr/r_a2_mean`, `sr/r_f1_mean` — PAG segment rewards
- `sr/effective_accuracy` — A1 for gated rows, A2 for non-gated (the inference-time final answer)
- `sr/gated_rate`, `sr/productive_gate_rate`, `sr/sycophantic_gate_rate` — gate quality
- `sr/rw_rate`, `sr/wr_rate` — self-correction rates among non-gated (target: WR > RW)
- `sr/f1_correct_verdict_precision`, `sr/f1_wrong_verdict_precision` — F1 calibration
- `grpo/frac_reward_zero_std`, `grpo/frac_fb_zero_std` — keep < 0.3
- `grpo/resp_adv_abs_mean`, `grpo/fb_adv_abs_mean` — advantage **magnitude** (mean of `|adv|`), NOT signed mean. Healthy K-group baselining produces `|adv|` around `E|Z|` for a binary-reward distribution (~0.5–0.9); the centered signed mean is ~0 by construction.
- `grpo/clip_frac` — fraction of tokens hitting PPO clip (keep < 0.2)
- `grpo/grad_norm` — per-step gradient norm (should be non-zero and stable)
- `grpo/entropy` — policy entropy (watch for collapse to 0)
- `reward/a1_accuracy`, `reward/a2_accuracy` — rolling windows

**Golden-path curve**: resp_reward and a1/a2 accuracy should climb
monotonically (noisy but trending up). WR-RW should trend positive. If both
a1 and a2 stay flat after 100 steps, training is NOT learning regardless of
loss values.

---

## One-off diagnostic commands

```bash
# Last N step summaries
grep "Step [0-9]\+:" <out>/training.log | tail -30

# RR/WW/RW/WR distribution per training window
python scripts/ops/analyze_trajectories.py <out>/training.log --windows 10

# Checkpoint LoRA probe
python scripts/ops/lora_probe.py <out>/checkpoint-<N>/adapter_model.safetensors

# Frozen-vs-moving weights
python scripts/ops/lora_delta.py <out>/checkpoint-<N>/adapter_model.safetensors --r 64 --alpha 128
```

*(Create these scripts if they don't exist — one-off ad-hoc snippets work too.)*

---

## Rules of thumb

- **If loss is ~0 but accuracy doesn't move, look at LoRA-B norm first.**
- **If accuracy peaks then collapses, look at α and KL second.**
- **If a1 and a2 track identically, the refiner isn't refining — look at feedback reward structure.**
- **If WR-RW is persistently negative, feedback is actively harmful — check the verdict/gate logic.**
- **Don't trust mean reward curves alone; they sum over heterogeneous outcomes. Per-window accuracy is more informative.**
- **`resp_adv` in the inner-epoch log is `mean(|adv|)`, not `mean(adv)`. A value of 0.7 is centered K-group baselining producing healthy magnitudes, not "uncentered advantages." Look at the variable name carefully.**
- **Under PAG, `reward/a2_mean` and `sr/r_a2_mean` are now both denominated over non-gated trajectories and should agree. If they don't, the metric path is broken.**
