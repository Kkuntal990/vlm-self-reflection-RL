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
**Date**: 2026-04-12 (in progress) | Base Qwen2.5-VL-7B (fresh) | Commit `9f52e22`
**Output**: `/outputs/grpo_qwen_livr_v2/` | **WandB**: `grpo-livr-9k-v2-tagfix`

**Changes from v1**: F1 tag penalty (w=0.5, raw=-2.0), hardened critic prompt ("Do NOT use XML tags"), MCQ extraction bugfix.
**Hypothesis**: 0% F1 tag leakage → better R→R rate, cleaner feedback. WR likely still flat (joint loss unfixed).
**Early**: F1 tag leakage 0/928 (0%) at step 114.

_Final metrics TBD._

---

## v3 (planned): Separate A1/A2 Loss (SCoRe-style)

**Goal**: Fix WR stagnation by giving A1 and A2 separate advantages.
**Changes**: (1) Separate A1/A2 loss in `critic_grpo.py`. (2) Strong A1 KL anchor (10x). (3) Correction bonus in A2-only reward. (4) Dynamic calibration weight when A1 wrong.
**Refs**: SCoRe (2409.12917), Murphy (2511.07833), TL-GRPO (2601.16480), Learning from Failures (2503.04808).
