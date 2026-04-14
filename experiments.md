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

**Changes from v1**: F1 tag penalty (w=0.5, raw=-2.0), hardened critic prompt ("Do NOT use XML tags"), MCQ extraction bugfix.

| Metric | Early 10% | Late 10% | Delta |
|--------|-----------|----------|-------|
| A1 Acc | 53.6% | 42.9% | -10.7pp |
| A2 Acc | 54.5% | 41.1% | -13.4pp |
| WR | 7.1% | 2.7% | -4.4pp |
| RW | 6.2% | 4.5% | -1.7pp |
| Entropy | 0.839 | 0.864 | +0.025 |
| F1 Tag Leak | 0% | 0% | fixed |

**Worked**: F1 tag leakage completely eliminated (0/560). Tag penalty + hardened prompt sufficient.
**Failed**: Accuracy degraded. WR declined 7.1%→2.7%. No separate turn loss = same WR stagnation as v1.

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

## v4 (planned): Effectiveness Reward + Reflection-Focused Training

**Goal**: Fix sycophantic critic via outcome-based feedback reward + remove conservative bias.

**Changes**:
1. Replace keyword calibration with effectiveness indicator: WR=+0.5, RR=+0.25, WW=0.0, RW=-0.25 (SRPO 2506.01713)
2. Remove minimal_edit reward (w=0.0) — stops rewarding "don't change" behavior
3. Add SSR (Selective Sample Replay) for zero-variance K-groups (VL-Rethinker 2504.08837)
4. Reward only F1 tokens when A2 succeeds — isolate feedback learning signal (Reflect/Retry/Reward 2505.24726)

**Future**: Pairwise preference GRPO warm-up (LLaVA-Critic-R1 2509.00676), rollout recombination (Octopus 2602.08503).

**Refs**: SRPO (2506.01713), VL-Rethinker (2504.08837), Reflect/Retry/Reward (2505.24726), LLaVA-Critic-R1 (2509.00676), Octopus (2602.08503), Critique-GRPO (2506.03106), S2R (2502.12853).
