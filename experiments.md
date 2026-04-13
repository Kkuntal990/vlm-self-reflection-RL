# Experiment Log

## Experiment 1: LIVR 9K MCQ — Think/Answer Tags (v1)

**Job**: `qwen-grpo-livr-9k-think-tags`
**Branch**: `feature/vllm-deepspeed` (commit `2fc4dd2`)
**Date**: 2026-04-11 to 2026-04-12
**Duration**: 22h on 4x A100-80GB
**Dataset**: LIVR 9K perception MCQ (9 tasks, 1000 each)
**Base model**: Qwen2.5-VL-7B-Instruct (no SFT warmup)

### Config

| Parameter | Value |
|-----------|-------|
| K samples | 8 |
| Batch / Grad Acc / GPUs | 2 / 2 / 4 (effective=16) |
| Learning rate | 1e-6 |
| Loss type | dr_grpo |
| KL coeff | 0.0 |
| LoRA r / alpha | 64 / 128 (scaling=2.0) |
| Max completion | 256 tokens (A1/F1/A2) |
| Temperature | A1=1.0, F1=0.7, A2=1.0 |
| Think/answer tags | Enabled (A1/A2 only) |
| vLLM gpu_mem | 0.50 |

### Reward Weights

| Response | Weight | Feedback | Weight |
|----------|--------|----------|--------|
| a1_correctness | 1.0 | downstream | 2.0 |
| a2_correctness | 1.0 | calibration | 0.2 |
| no_regression | 2.0 | format | 0.15 |
| a2_format | 0.5 | | |
| minimal_edit | 0.3 | | |

MCQ-aware asymmetry: no_regression WR=+3/RW=-2, downstream WR=+3/RW=-1.5

### Final Metrics (2250/2250 steps)

| Metric | Early 10% | Late 10% | Delta |
|--------|-----------|----------|-------|
| A1 Accuracy | 45.9% | 50.7% | +4.8pp |
| A2 Accuracy | 43.8% | 50.1% | +6.4pp |
| WR (self-correction) | 4.9% | 4.9% | 0.0pp |
| RW (regression) | 7.0% | 5.5% | -1.5pp |
| RR (maintain) | 38.9% | 45.2% | +6.3pp |
| WW (stuck wrong) | 49.2% | 44.4% | -4.8pp |
| Net (WR-RW) | -2.1pp | -0.6pp | +1.5pp |
| Resp Reward | +0.75 | +1.24 | +0.49 |
| FB Reward | -0.14 | +0.15 | +0.29 |
| Entropy | 0.965 | 0.920 | -0.045 |
| Grad Norm | 0.70 | 0.97 | +0.27 |
| Zero-Std Frac | 12.7% | 17.3% | +4.6pp |
| A1 Tokens | 142 | 116 | -26 |
| F1 Tokens | 130 | 112 | -18 |
| A2 Tokens | 136 | 108 | -28 |

### What Worked

- A1 accuracy improved +4.8pp (model learns to answer MCQs better)
- A2 accuracy improved +6.4pp
- Regression rate dropped from 7.0% to 5.5%
- Model learned think/answer tag format (a2_format +0.09 to +0.20)
- Outputs became more concise (-26 tokens for A1)
- Both resp and fb rewards improved and turned positive

### Problems Identified

1. **WR (self-correction) stuck at 4.9%** — never improved from start to finish. The model learned to get A1 right (easier path) instead of learning to correct wrong A1s.

2. **F1 think tag leakage**: 1% early → 51% late training. Think tags bled from A1/A2 into F1 despite critic prompt having no tag instructions. Think-tagged F1 produced worse outcomes (R→R: 27.7% vs 43.5% for plain F1).

3. **Confirmatory feedback**: F1 almost always agrees with A1 ("your answer is correct"), even when A1 is wrong. This prevents self-correction since A2 has no signal to change.

4. **Entropy declining**: 0.965 → 0.920 (-5.4%). Not collapsed but steady erosion of policy diversity.

5. **Zero-variance K-groups increasing**: 12.7% → 17.3%. More K-groups where all trajectories get the same reward, producing zero gradient.

### Root Cause Analysis

**Joint A1+A2 loss is the core problem.** In `critic_grpo.py:612`, A1 and A2 log-probs are concatenated and share a single advantage scalar. When W→R happens (+3.0 advantage), it reinforces both the wrong A1 and the correct A2 equally. Improving A1 directly yields the same reward as self-correction but is strictly easier. This is the "direct solution collapse" identified by SCoRe (arXiv 2409.12917).

**No A1 KL anchor.** SCoRe applies strong KL to the first turn to prevent improving A1 directly. Our KL is 0.0 for all turns.

**Cold start for feedback.** The WR=+3 feedback bonus exists but is rarely triggered because confirmatory feedback dominates, creating a chicken-and-egg problem.

### Artifacts

| Asset | Path |
|-------|------|
| Final checkpoint | `/outputs/grpo_qwen_livr_v1/final/` |
| Checkpoints | `/outputs/grpo_qwen_livr_v1/checkpoint-{250,500,750,1000}/` |
| Training log | `/outputs/grpo_qwen_livr_v1/training.log` |
| WandB | `grpo-livr-9k-think-tags-v1` in `vlm-self-reflection-grpo` |
| K8s YAML | `k8s/job-qwen-grpo-livr-9k.yaml` |

---

## Experiment 2: LIVR 9K MCQ — Tag Penalty Fix (v2)

**Job**: `qwen-grpo-livr-9k-v2-tagfix`
**Branch**: `feature/vllm-deepspeed` (commit `9f52e22`)
**Date**: 2026-04-12 (started)
**Duration**: In progress
**Dataset**: Same LIVR 9K
**Base model**: Qwen2.5-VL-7B-Instruct (fresh, no v1 checkpoint)

### Changes from v1

1. **F1 tag penalty**: `compute_f1_tag_penalty()` returns -2.0 when F1 contains `<think>/<answer>` tags. Weight `w_fb_tag_penalty=0.5` (effective -1.0 penalty).
2. **Critic prompt hardened**: Added "Write your feedback as plain text. Do NOT use XML tags." to `FEEDBACK_CRITIC_SYSTEM_PROMPT`.
3. **MCQ extraction bugfix**: `extract_answer_from_text()` now calls `extract_from_answer_tags()` first, fixing false `a2_extracted='E'` from matching stray letters in `<think>` sections (cosmetic — rewards were already correct).

### Config

Same as v1 except:

| Parameter | v1 | v2 |
|-----------|----|----|
| w_fb_tag_penalty | 0.0 (not present) | 0.5 |
| Critic prompt | No tag instruction | "Do NOT use XML tags" |
| Output dir | `/outputs/grpo_qwen_livr_v1/` | `/outputs/grpo_qwen_livr_v2/` |
| Resume from | Fresh | Fresh |

### Hypothesis

The F1 tag penalty will suppress think tag leakage in feedback, producing cleaner F1 that leads to better downstream outcomes. Specifically:
- F1 tag leakage should stay near 0% (vs 51% in v1)
- R→R maintain rate should be higher (tagged F1 dropped R→R from 43.5% to 27.7%)
- Feedback reward should improve faster

### What This Does NOT Fix

The fundamental joint A1+A2 loss problem and lack of A1 KL anchor remain. WR rate may still not improve because the core incentive structure still rewards improving A1 over learning self-correction.

### Early Results (step 114/2250, 5%)

- F1 tag leakage: **0/928 (0%)** vs 1% at same point in v1
- Training running stable

### Metrics (to be filled after completion)

_In progress_

### Artifacts

| Asset | Path |
|-------|------|
| Output dir | `/outputs/grpo_qwen_livr_v2/` |
| WandB | `grpo-livr-9k-v2-tagfix` in `vlm-self-reflection-grpo` |
| K8s YAML | `k8s/job-qwen-grpo-livr-9k-v2.yaml` |

---

## Planned: Experiment 3 — Separate A1/A2 Loss (SCoRe-style)

### Motivation

Experiments 1-2 use joint A1+A2 loss where a single advantage scalar is applied to all tokens. This causes "direct solution collapse" (SCoRe, arXiv 2409.12917) — the model improves A1 instead of learning self-correction.

### Proposed Changes

1. **Separate A1 and A2 loss terms**: A1 gets advantage from `a1_correctness` only. A2 gets advantage from `a2_correctness + no_regression + format + minimal_edit`. Modify `critic_grpo.py` to compute separate advantages and loss for each turn.

2. **Strong A1 KL anchor**: `a1_kl_coeff = 10x`, `a2_kl_coeff = 1x`. Prevents A1 from improving directly, forcing the model to maintain base-model-like mistakes that A2 must learn to correct.

3. **Explicit correction bonus** in A2-only reward: +2.0 for W→R, -2.0 for R→W, applied only to A2 tokens.

4. **Dynamic calibration weight**: Increase `w_calibration` from 0.2 → 1.0 when A1 is wrong, to incentivize error detection in feedback.

### Key Files to Modify

- `src/vlm_grpo/critic_grpo.py` — Separate A1/A2 log-probs, advantages, and loss
- `src/vlm_grpo/rewards/composition.py` — Split response reward into A1 and A2 scalars
- `src/vlm_grpo/config.py` — Add `a1_kl_coeff`, `a2_kl_coeff`, `w_correction_bonus`

### References

- SCoRe (arXiv 2409.12917) — Two-stage RL for self-correction with first-turn KL anchor
- Murphy (arXiv 2511.07833) — Multi-turn GRPO for self-correcting code generation
- TL-GRPO (arXiv 2601.16480) — Turn-level RL with per-turn reward design
- Learning from Failures (arXiv 2503.04808) — Multi-attempt RL with attempt-specific rewards
