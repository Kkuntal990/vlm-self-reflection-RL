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
| A2 systematically below A1 | Refiner hurting; check no_regression shape and feedback gate. |

## 5. vLLM weight sync

Training updates LoRA. vLLM has its own weight copy. If sync fails, rollouts
use stale weights and the policy effectively never changes.

**Check**: in `vllm_rollout.py`, confirm `merge_adapter()` → vLLM
`load_weights()` → `unmerge_adapter()` is called **every rollout step**, not
lazily.

**Also**: capture vLLM's returned logprobs at sample time, compare with
HF-path `old_lps` on the same tokens. Drift > 0.01 per token indicates
kernel/precision mismatch that can blow up PPO ratios at iter 0.

---

## Known failure modes & fixes

| Failure | Indicator | Root cause | Fix |
|---|---|---|---|
| **Dead LoRA** | LoRA-B norm = 0 | optimizer not stepping LoRA params | Check param-group construction, DDP sync |
| **Toy updates** | Effective delta max abs < 1e-4 | LR too low for signal scale | Raise LR 5-10× |
| **Signal starvation** | frac_zero_std > 0.5 | K too small or reward too discrete | Enable SSR, raise K, raise temperature |
| **Shaped-reward spike domination** | RW/WR advantages dominate group, collapse after peak | α too high on binary rewards | α=1 (SCoRe paper default) |
| **Length bias** | Short answers get more gradient than long reasoning | Per-sample loss mean | Use Dr.GRPO `sum / max_comp_len` |
| **KL pinning** | kl_loss ≈ policy_loss in magnitude | KL coefficient too strong | Lower kl_coeff, or remove A1 KL anchor post-freeze |
| **Dataset noise** | Best run plateaus well below 80% | GT labels wrong or ambiguous | Run per-task VLM oracle audit, drop low-trust tasks |
| **vLLM drift** | Non-zero clip_frac at inner epoch 0 | vLLM rollout logprobs ≠ HF path | Add IS correction using vLLM logprobs |

---

## Metrics dashboard (what to plot in wandb)

Drop these at every step:

- `reward/resp_mean`, `reward/fb_mean` — aggregate
- `reward/a1_mean`, `reward/a2_mean` — per-turn (needs `separate_turn_loss=True`)
- `sr/rw_rate`, `sr/wr_rate` — self-correction rates (target: WR > RW)
- `grpo/frac_reward_zero_std`, `grpo/frac_fb_zero_std` — keep < 0.3
- `grpo/resp_adv_abs_mean`, `grpo/fb_adv_abs_mean` — advantage magnitude
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
