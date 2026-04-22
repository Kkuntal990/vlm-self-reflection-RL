# Experiment Log

## v1: LIVR 9K — Think/Answer Tags

**Job**: `qwen-grpo-livr-9k-think-tags` | **YAML**: `k8s/job-qwen-grpo-livr-9k.yaml`
**Date**: 2026-04-11–12 | 22h, 4x A100 | Base Qwen2.5-VL-7B | Commit `2fc4dd2`
**Output**: `/outputs/grpo_qwen_livr_v1/` | **WandB**: `grpo-livr-9k-think-tags-v1`

| Metric | Early 10% | Late 10% | Delta |
|--------|-----------|----------|-------|
| A1 Acc | 45.9% | 50.7% | +4.8pp |
| A2 Acc | 43.8% | 50.1% | +6.4pp |
| WR | 4.9% | 4.9% | 0.0pp |
| RW | 7.0% | 5.5% | -1.5pp |
| Entropy | 0.965 | 0.920 | -0.045 |
| Resp Reward | +0.75 | +1.24 | +0.49 |
| FB Reward | -0.14 | +0.15 | +0.29 |
| F1 Tag Leak | ~1% | 51% | exploded |

**Worked**: A1/A2 accuracy up, regression down, concise outputs, tag format learned.
**Failed**: WR stuck at 4.9%. F1 tag leakage 1%→51%. Confirmatory feedback dominates.
**Root cause**: Joint A1+A2 loss shares single advantage — model improves A1 (easy) instead of learning correction (hard). No A1 KL anchor (SCoRe arXiv:2409.12917).

---

## v2: LIVR 9K — Tag Penalty Fix

**Job**: `qwen-grpo-livr-9k-v2-tagfix` | **YAML**: `k8s/job-qwen-grpo-livr-9k-v2.yaml`
**Date**: 2026-04-12–13 | Base Qwen2.5-VL-7B (fresh) | Commit `9f52e22`
**Output**: `/outputs/grpo_qwen_livr_v2/` | **WandB**: `grpo-livr-9k-v2-tagfix`

**Changes from v1**: F1 tag penalty (w=0.5, raw=-2.0), hardened critic prompt ("Do NOT use XML tags"), MCQ extraction bugfix (`extract_from_answer_tags` before MCQ matching).

**Incomplete**: Killed at ~27% (4,958/18,000 trajectories, checkpoint-250 only). Prior table was misleading — compared v2's 27% endpoint against v1's 100% endpoint, with different extraction logic.

| Metric | v1 Q5 (21-27%) | v2 Q5 (21-27%) | Notes |
|--------|-----------------|-----------------|-------|
| A1 Acc | 51.1% | 51.3% | Equivalent |
| A2 Acc | 48.1% | 47.7% | Equivalent |
| WR rate | 9.1% | 7.7% | Within noise |
| RW rate | 14.7% | 14.4% | Within noise |
| resp_rwd | 1.009 | 0.990 | Equivalent |
| F1 Tag Leak | N/A | 0.0% | Fixed |

**Worked**: F1 tag leakage completely eliminated (0%). Tag penalty + hardened prompt sufficient.
**Not failed**: At equivalent training windows, v2 tracks v1 closely. The previously reported "accuracy degradation" was an artifact of (1) comparing different training percentages, and (2) the MCQ extraction bugfix changing how `a1_correct`/`a2_correct` were computed (v1 could match stray letters in `<think>` sections).

---

## v3: Separate A1/A2 Loss (SCoRe-style)

**Job**: `qwen-grpo-livr-9k-v3-score` | **YAML**: `k8s/job-qwen-grpo-livr-9k-v3.yaml`
**Date**: 2026-04-13 (in progress, 69%) | Base Qwen2.5-VL-7B (fresh) | Commit `5342176`
**Output**: `/outputs/grpo_qwen_livr_v3/`

**Changes from v2**: Separate A1/A2 loss, A1 KL=0.2 (10x anchor), A2/F1 KL=0.02, w_calibration=1.0.

| Metric | Early 10% | Recent 10% | Delta |
|--------|-----------|------------|-------|
| A1 Acc | 63.8% | 50.9% | -12.9pp |
| A2 Acc | 62.0% | 41.1% | -20.9pp |
| WR | 2.5% | 1.2% | -1.3pp |
| RW | 4.3% | 11.0% | +6.7pp |
| Entropy | 0.819 | 0.839 | +0.020 |
| F1 Tag Leak | 0% | 0% | clean |

**Worked**: Non-zero loss (KL producing gradient), F1 tags clean without prompt instruction.
**Failed**: WR declined to 1.2%. RW spiked to 11.0% — model regressing heavily. A2 accuracy dropping fast.
**Root cause**: Sycophantic critic + broken keyword calibration. Log analysis (n=2128): 81% of WW have F1 saying "correct" (calibration=-1.0). 53% of RW have calibration=+1.0 (false positive — "partially correct, but correct answer is (D)"). Same-model critique cannot identify errors it couldn't avoid in A1.

---

## v3 Final Analysis: The Real Problem

**Job**: `qwen-grpo-livr-9k-v3-score` | **Complete**: 25h, 18,000 trajectories (2,250 samples)

| Metric | First 10% | Last 10% | Delta |
|--------|-----------|----------|-------|
| A1 Acc | 46.2% | 49.5% | +3.3pp |
| A2 Acc | 45.0% | 46.1% | +1.1pp |
| WR | 5.8% | 6.1% | +0.3pp |
| RW | 7.0% | 9.5% | +2.5pp |
| Resp Reward | 0.855 | 0.916 | +0.06 |
| FB Reward | -0.083 | 0.020 | +0.10 |
| F1 Tag Leak | 0.2% | 0.4% | ~0 |

### Key Findings (overturning prior assumptions)

**1. F1 is NOT overwhelmingly sycophantic.** When A1 is wrong (n=9,273): F1 honest 63.0%, sycophantic 27.7%, neutral 9.3%. The critic identifies errors more often than not.

**2. Zero-variance K-groups are NOT the bottleneck.** Only 2.8% of fb K-groups have zero variance. Mean fb reward gap within groups = 6.07. Gradient signal exists.

**3. Honest critique does NOT lead to correction — it makes things WORSE.**

```
P(A2 correct | A1 wrong, F1 honest)      =  8.1%
P(A2 correct | A1 wrong, F1 sycophantic) = 15.2%
Lift from honest F1 = -7.1pp
```

This is the **verification-generation gap**. For MCQ with C=4 choices, the model that chose (B) wrongly gains zero information from knowing "(B) is wrong" about which of (A)/(C)/(D) is correct. Honest critique sends the model into confused re-reasoning that fails 91.9% of the time.

**4. GRPO advantage is INVERTED for WR trajectories.**

| Transition | Mean fb_reward | Mean advantage | Positive adv % |
|---|---|---|---|
| WR (want highest) | 0.084 | **-0.054** | 44.8% |
| RW (want lowest) | 0.054 | **+0.014** | 45.4% |

WR has **negative mean advantage** — GRPO is pushing the policy AWAY from corrective feedback. Only 25.8% of WR trajectories beat all WW trajectories in their K-group.

**5. The sycophancy gradient points the wrong way.**

```
R(sycophantic F1) ≈ 0.152·(+3) + 0.848·(-1) = -0.392
R(critical F1)    ≈ 0.081·(+3) + 0.919·(-1) = -0.676
∂E[R_fb]/∂s = R(syco) - R(crit) = +0.284 > 0
```

The policy has positive gradient toward MORE sycophancy because P(WR|syco)=15.2% > P(WR|honest)=8.1%.

### Root Cause

The transition-shaped downstream reward {WR:+3, RR:+1, WW:-1, RW:-1.5} is dominated by RR(+1) and WW(-1) within each K-group. These make the group mean noisy around 0, and a lone WR(+3) gets diluted. The honest_WR vs syco_WW fb_reward gap is only **+0.029** — essentially zero signal.

### Mathematical Framework

The core problem is that R_downstream is a function of the **transition (a1_correct, a2_correct)**, and within a K-group where all 8 trajectories share the same question, the transitions are highly correlated. The advantage `Â_k = R_k - μ_G` is dominated by the majority transition type, not the minority WR.

The improvement-based reward `R_improve = R(A2) - R(A1)` fixes this by giving RR=0 and WW=0, making the group mean converge to 0 and letting WR(+2) and RW(-2) dominate the advantage.

---

## Experiment Diff Table

Quick reference for what changed between versions. All share: Qwen2.5-VL-7B, LIVR 9K MCQ, K=8, dr_grpo, LoRA r=64, separate_turn_loss, 4x A100.

| | v1 | v2 | v3 | v4 | v5 | v6/v6b | v7 |
|---|---|---|---|---|---|---|---|
| **Key change** | Baseline + tags | Tag penalty | SCoRe KL | Drop cal/edit, shuffle | SSR buffer | Improve reward + freeze A1 | **Shaped fb reward** |
| w_calibration | 0.2 | 0.2 | 1.0 | **0.0** | 0.0 | 0.0 | 0.0 |
| w_minimal_edit | 0.3 | 0.3 | 0.3 | **0.0** | 0.0 | 0.0 | 0.0 |
| w_tag_penalty | 0.0 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| KL (A1/A2/FB) | 0/0/0 | 0/0/0 | 0.2/0.01/0.01 | 0.2/0.02/0.02 | 0.2/0.02/0.02 | **20/0.02/0.02** | 20/0.02/0.02 |
| Data shuffle | No | No | No | **Yes** | Yes | Yes | Yes |
| KL /3.0 bug | Yes | Yes | Yes | **Fixed** | Fixed | Fixed | Fixed |
| SSR | — | — | — | — | **buffer=64** | — | — |
| Fb reward type | Transition | Transition | Transition | Transition | Transition | **Improvement** | **Shaped α=5** |
| Freeze A1 | — | — | — | — | — | **280 steps** | 280 steps |
| Status | Complete | Incomplete | Complete | Running | Crashed | **Running** | Planned |

---

## v4: Drop Calibration, Drop Minimal Edit, Shuffle Data, KL Fix

**Job**: `qwen-grpo-livr-9k-v4-downstream` | **YAML**: `k8s/job-qwen-grpo-livr-9k-v4.yaml`
**Date**: 2026-04-14 | Commit `999f684` → `0a2aa96` | **Output**: `/outputs/grpo_qwen_livr_v4/`

**Changes from v3**: w_calibration=0.0, w_minimal_edit=0.0, shuffle training data, fix KL /3.0 normalization bug (A1 anchor was 3x weaker than intended). NCCL timeout 1800s. Resumed from checkpoint-250 after NCCL crash at ~7%.

---

## v5: SSR (Selective Sample Replay)

**Job**: `qwen-grpo-livr-9k-v5-ssr` | **YAML**: `k8s/job-qwen-grpo-livr-9k-v5.yaml`
**Date**: 2026-04-14 | **Output**: `/outputs/grpo_qwen_livr_v5/`

**Changes from v4**: +SSR (buffer=64, alpha=1.0, VL-Rethinker 2504.08837). Crashed at ~5% (NCCL timeout). **Deprioritized** — v3 analysis showed zero-var K-groups are only 2.8%, not the bottleneck.

---

## v6: Improvement-Based Reward + SCoRe Stage I

**Job**: `qwen-grpo-livr-9k-v6-improve` | **YAML**: `k8s/job-qwen-grpo-livr-9k-v6.yaml`
**Date**: 2026-04-14 | Commit `d2011d9` | **Output**: `/outputs/grpo_qwen_livr_v6/`

**Changes from v4**: Two mathematically-motivated fixes for the inverted WR advantage.

1. **Improvement-based feedback reward** (Critique-GRPO, 2506.03106): `R_improve = R(A2) - R(A1)`. WR=+2, RW=-2, RR=0, WW=0. Group mean → 0, so WR/RW dominate the advantage instead of being diluted by RR/WW majority.
2. **SCoRe Stage I** (2409.12917): Freeze A1 policy loss for 100 steps (A1 KL anchor preserved). Prevents A1 from co-adapting with F1 during initial learning.

**Hypothesis**: v3 analysis showed WR mean advantage was -0.054 (GRPO pushes AWAY from correction). The improvement reward directly fixes this by zeroing out RR/WW. If WR advantage flips positive, self-correction should improve.

**Refs**: Critique-GRPO (2506.03106), SCoRe (2409.12917), Huang et al. (2310.01798).

### v6b: Strong A1 KL + 50% Freeze

v6 crashed (NCCL). v6b re-ran with stronger settings: A1 KL=1000x (effective 20.0), freeze=280 steps (50%), fresh start. Output: `/outputs/grpo_qwen_livr_v6b/`.

| Metric | First 10% | Latest 10% | Delta |
|--------|-----------|------------|-------|
| A1 Acc | 43.9% | 46.5% | +2.6pp |
| A2 Acc | 41.4% | 44.7% | +3.3pp |
| WR rate | 6.4% | 7.4% | +1.0pp |
| RW rate | 13.8% | 12.3% | -1.4pp |
| Entropy | 2.026 | 1.956 | -0.07 |

**Freeze vs post-freeze**: RW dropped 14.3%→11.8% (good), WR flat 7.9%→7.5% (not improving). No A1 spike after unfreeze — strong KL anchor works. A2-A1 gap narrowing (-2.3pp→-1.6pp).

**Root cause of flat WR**: Pure improvement reward (RR=0, WW=0) gives 33.6% dead K-groups and no RR stabilization signal. RW gradient (0.143) is 30% stronger than WR gradient (0.110), so the policy reduces regression without learning to correct. Need shaped reward to add RR signal and center group mean to 0.

---

## v7: SCoRe-Style Shaped Reward for Both F1 and A2 (Planned)

**Changes from v6b**: Shaped reward `α×(R(A2)-R(A1))` replaces both F1 improvement reward AND A2 no_regression, α=5. Dataset: `livr_perception_mixed.jsonl` (11K).

```
                 v6b                    v7 (shaped α=5)
F1 reward:       R(A2)-R(A1)           R(A2) + α×(R(A2)-R(A1))
  WR:            +2                     +11
  RW:            -2                     -11
  RR:             0                     +1
  WW:             0                     -1
  Dead K-groups: 53.8%                  <1%

A2 reward:       a2_corr + 2×noreg     a2_corr + 1×α×(a2_corr - a1_corr)
  WR:            +7                     +11
  RW:            -5                     -11
  RR:            +3                     +1   (less conservative → more exploration)
  RW:            -1                     -1
```

**Why replace no_regression for A2**: Current RR stabilization (Â=+2.3) makes A2 too conservative. RR gradient dominates WR gradient 3.8:1 (∇_RR=0.94 vs ∇_WR=0.25) because RR is 10x more frequent. With shaped α=5: ratio drops to 1.2:1 (∇_RR=0.53 vs ∇_WR=0.45). A2 becomes willing to attempt correction while RW penalty is stronger (-10.71 vs -5.70).

**F1 fix**: Dead K-groups 53.8%→<1%. RR(+1)/WW(-1) always differ, providing variance for F1 gradient.

**v6b final analysis (99%, 17,824 traj)**: Three problems identified:
1. **88% F1 sycophancy** — honest F1 gives 33% WR but only produced 9% of the time
2. **53.8% dead fb K-groups** — improvement reward RR=WW=0 kills gradient
3. **WR flat at 7.5%** despite RW dropping 13.8%→8.1% — A2 too conservative

v7 directly fixes #2 and #3. Problem #1 (sycophancy) requires architectural changes (v8: verify-then-correct).

**Architecture**: Same as v6b (separate_turn_loss, A1 KL=1000x, freeze=280 steps). `--reward_shaping_alpha 5.0`, `w_no_regression=1.0`, `w_downstream=1.0`.

**Refs**: SCoRe (2409.12917), Tyen et al. (2311.08516), Huang et al. (2310.01798).

---

## v7b: α=5 Shaped Reward + Rebalanced Weights

**Job**: `qwen-grpo-livr-9k-v7-shaped` | **YAML**: `k8s/job-qwen-grpo-livr-9k-v7.yaml`
**Date**: 2026-04-17 | Base Qwen2.5-VL-7B | Commits `88107b5` → `202719b`
**Output**: `/outputs/grpo_qwen_livr_v7b/` | **WandB**: `grpo-livr-9k-v7b-shaped`

**Changes from v7 plan**: α reduced 10→5 (avoid RR negative advantage), w_a2_format 0.5→2.0 (format signal visible), no-tag penalty -1.0→-2.0 raw, A1_KL 1000x→150x. Used `<think>+<answer>` tags.

| Metric | v7b late | Notes |
|--------|---|---|
| A1 Acc | 47.4% | Flat (KL strong) |
| A2 Acc | 46.5% | Slightly below A1 |
| **Self-reflection Δ** | **-0.9pp** | Negative |
| WR rate | 2.9% | Low |
| RW rate | 3.7% | Still > WR |

**Root cause of regression**: `<think>` tags let the model reason itself out of correct answers. 97% of RW cases had positive F1 ("Your answer is correct") but the model overthought in `<think>` and flipped anyway. Job deleted in favor of v8 (answer-tag-only).

---

## v8b: Binary Verification + Answer-Tag-Only (FULL RUN COMPLETED)

**Job**: `qwen-grpo-livr-9k-v8-verify` | **YAML**: `k8s/job-qwen-grpo-livr-9k-v8.yaml`
**Date**: 2026-04-17→18 | 24h+ total | Commit `a02b147`
**Output**: `/outputs/grpo_qwen_livr_v8b/final/`
**WandB**: `grpo-livr-9k-v8b-verify` ([link](https://wandb.ai/braindecode/vlm-self-reflection-grpo/runs/c1sgeeqp))

**Changes from v7b**: `<answer>` tag only (no `<think>`), binary verification F1 (CORRECT/INCORRECT), a2_temp=0.7, w_minimal_edit=0.3. α=5, w_a2=1, w_fmt=2.0, A1_KL=150.

**Completed**: Original crashed at step 1190 (SIGABRT rank 3); resumed from checkpoint-1000 pinned to commit a02b147, finished full 2763 steps.

| Metric | Late 1600 trajectories | Notes |
|--------|---|---|
| A1 Acc | 50.9% | |
| A2 Acc | 54.2% | |
| **Self-reflection Δ** | **+3.3pp** | Best so far |
| WR rate (of A1-wrong) | 21.9% | Big jump from v7b |
| RW rate (of A1-correct) | 14.7% | Higher but offset by WR |
| WR > RW? | **Yes** (172 > 120) | First experiment where this holds |

**Worked**: no-tag short-circuit, `a2_temp=0.7`, `minimal_edit=0.3`, binary verifier simpler than freeform.
**Failed**: 44% over-harshness (F1 cries wolf), sycophancy 29%→17% but still meaningful, rw_rate elevated.

**Post-run bug discovery**: `train_self_reflection.py` auto-reduced F1 token cap from yaml-set 256 to **16 tokens** whenever `--use_binary_verification` was active. F1 explanations truncated mid-sentence. v8b was affected by this bug. Fix: remove auto-reduce (commit `f52b38d`).

---

## v9: Rebalanced Rewards + `<feedback>` Tags (RUNNING, α=3 w_a2=3)

**Job**: `qwen-grpo-livr-9k-v9-balanced` | **YAML**: `k8s/job-qwen-grpo-livr-9k-v9.yaml`
**Date**: 2026-04-18 (fresh restart after F1 token cap bug fix) | Commit `f52b38d`
**Output**: `/outputs/grpo_qwen_livr_v9_fixed/`
**WandB**: `grpo-livr-9k-v9-balanced-fixed`

**Hypothesis**: Rebalance WR:RR ratio to ~2:1 (was 4.3:1 in v7b) via:
- α reduced 5→3 (halves shaped amplification)
- w_a2_correctness 1→3 (raises RR base so it stays positive in advantage)
- `<feedback>` tags for deterministic F1 verdict extraction
- w_fb_format 0→0.5 (incentivize `<feedback>` tag usage + explanations)
- Added separate `response_alpha` / `feedback_alpha` params (currently both 3)

**Expected A2 reward table**:
| Transition | v9 | v8b/v9b (α=5) |
|---|---|---|
| RR same + tag | +5.3 | +1.5 |
| WR + tag | +11.0 | +12.0 |
| RW + tag | -7.0 | -9.0 |
| WW + tag | -1.0 | -0.5 |
| No tag | -4.0 | -4.0 |

**WR:RR ratio: 2.1:1** (both positive on hard Q).

**Key infrastructure fixes from v8**:
1. F1 token cap bug (auto-reduce to 16) removed → F1 explanations can reach 50-80 tokens
2. Env-var prompts (commit `c9f6b7a`) → yaml is source of truth for prompts
3. Commit pinning in yaml → reproducible runs

**Progress at step ~670 (24%)**:
| Window | resp_r | fb_r | rw_rate |
|--------|---|---|---|
| Pre-freeze (10-280) | +1.22 | +0.34 | 0.163 |
| Post-freeze (280+) | +1.65 | +0.64 | 0.122 |

Loss spikes: 2 (step 240, step 620) vs v9b's 4. Smoother training.

---

## v9b: v8b Ablation with Bug Fix (RUNNING, in parallel)

**Job**: `qwen-grpo-livr-9k-v9b-v8params` | **YAML**: `k8s/job-qwen-grpo-livr-9k-v9b.yaml`
**Date**: 2026-04-18 | Commit `f52b38d`
**Output**: `/outputs/grpo_qwen_livr_v9b/`
**WandB**: `grpo-livr-9k-v9b-v8params-fixed`

**Purpose**: Isolate the impact of the F1 token cap bug on v8b's design. Same parameters as v8b (α=5, w_a2=1, BINARY_VERIFIER prompt without `<feedback>` tags) but with the token cap bug fixed.

**v8b vs v9b byte-for-byte identical except**:
- Code commit: a02b147 (bug) → f52b38d (fixed)
- Effective F1 token cap: 16 → 256

**3-way comparison goal**:
- **v9 vs v9b**: does rebalanced reward design beat v8b-style once the bug is fixed?
- **v9b vs v8b-final**: how much did the bug hurt v8b?

**Progress at step ~670 (24%)**:
| Metric | v9 | v9b | Gap |
|--------|------|------|------|
| Cumulative resp | +1.47 | +1.29 | +0.18 (v9) |
| Cumulative fb | **+0.51** | **-0.13** | **+0.64** (v9) |
| rw_rate avg | 0.139 | 0.132 | tied |
| Loss spikes | 2 | 4 | v9 more stable |

---

## Experimental Infrastructure Notes

### EMA Smoothing

Per-step metrics in wandb are smoothed with EMA:
```python
ema_alpha = 0.05  # weight for new value
ema = 0.95 * prev + 0.05 * new
```

**Half-life ≈ 13.5 steps** (60 steps → 95% new info). EMA lags the true trend by ~14 steps. If the EMA of WR-RW stays flat over many steps, the underlying metric is genuinely not improving — this is an accurate signal, not a display artifact.

### Known Bug Fixed in Commit `f52b38d`

Prior to `f52b38d`, `train_self_reflection.py` had an auto-reduce that silently overrode yaml-set token caps to 16/32 tokens when `--use_binary_verification` was active. This truncated F1 explanations mid-sentence. Fixed by removing the auto-reduce and respecting yaml values.

### Are Our Rewards Actually Training Self-Reflection?

**Observation**: Rewards are outcome-based (did A2 improve over A1?), not process-based (did the model actually reflect on F1's content?). A WR transition gets the same reward whether:
- Model correctly identified the error and fixed it (true reflection), OR
- Model happened to sample a different answer that was correct (variance)

Evidence from v8b trajectory analysis (77k trajectories):
- When F1 says "CORRECT" and A1 IS correct: 19% of A2s CHANGE anyway
- When F1 (miscalibrated) says "INCORRECT" and A1 IS correct: 52% follow → RW
- Over-harshness rate: 44% (F1 defaults to "INCORRECT")
- Sycophancy: early 29% → late 17% (did improve)

This suggests the model learns **which tokens correlate with reward**, not necessarily the ACT of reflection. Self-reflection Δ of +3.3pp is real but may plateau without process-level signals.

**Potential future rewards for true self-reflection** (v10+ ideas):
- Gate A2 reward on whether A2's content cites F1's specific error
- Reward F1 for citing visual evidence (not just verdict)
- Add a "consistency" reward: A1 and A2 should agree when F1 says CORRECT
- Supervised pre-training of F1 on calibrated critiques before joint RL

**Refs**: SCoRe (2409.12917), Self-Rewarding Correction (2502.19613), Critique-GRPO (2506.03106), DAPO (2503.14476), Dr.GRPO (2503.20783).
