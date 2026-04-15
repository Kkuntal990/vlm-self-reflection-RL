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

## v7: SCoRe-Style Shaped Feedback Reward (Planned)

**Changes from v6b**: Replace pure improvement `R(A2)-R(A1)` with shaped `R(A2) + α×(R(A2)-R(A1))`, α=5.

```
                 v6b (improvement)    v7 (shaped α=5)
F1 reward WR:         +2                  +11
F1 reward RW:         -2                  -11
F1 reward RR:          0                   +1  ← new stabilization signal
F1 reward WW:          0                   -1  ← new penalty
Dead K-groups:       33.6%                0.3%
```

**Mathematical justification**: With v6b's improvement reward, E[μ_group]=-0.038 and |∇_RW|/|∇_WR|=1.30 (RW gradient 30% stronger). With shaped α=5, E[μ_group]≈0 and ∇_WR becomes dominant because RR(+1) and WW(-1) no longer dilute the mean.

**A2 response reward unchanged**: no_regression stays (RR=+3, RW=-5, WR=+7, WW=-1). Analysis confirmed no_regression is necessary — without it, WR=RW=0 in A2's advantage and A2 loses all self-correction signal. Shaped A2 reward deferred to v8 (marginal benefit: WR advantage +6→+7, 17% improvement).

**Architecture**: Same as v6b (separate_turn_loss, A1 KL=1000x, freeze=280 steps). Only the F1 reward function changes. New flag: `--reward_shaping_alpha 5.0`.
