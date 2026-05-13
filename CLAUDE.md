# VLM Self-Reflection GRPO Training Pipeline

## Overview

Multi-turn GRPO training pipeline for VLM self-reflection, using a custom SelfReflectionGRPOTrainer with two-reward design.

**Models**: Qwen2.5-VL-7B-Instruct (base) or SFT-v2 checkpoint
**Framework**: Custom GRPO trainer + vLLM (sleep mode for GPU sharing)
**Approach**: Multi-turn A1 → F1 → A2 flow with separate response and feedback rewards

## Data

### GRPO Training Data
- **Path**: `/outputs/grpo_data/balanced_70k.jsonl`
- **Fields used**: `question`, `ground_truth`, `answer_type`, `images` (for image path)
- **Fields IGNORED**: `messages` (contains SFT conversation with thought/answer system prompt — not used by GRPO)
- The `messages` field has a system prompt with `Thought: / Answer:` format instructions, but GRPO uses its own hardcoded prompts from `src/vlm_grpo/prompts.py`
- Images are resolved from the `images` array (or `image_path` field), NOT extracted from messages
- `data.py` only falls back to parsing `messages` when `question` is NOT present as a top-level field

### SFT Training Data
- **Path**: `/outputs/mixed_training_v1/` (multiple JSONL files)
- Source datasets: `fire_messages_filtered.jsonl`, `fire_messages_single_turn.jsonl`, `lvlm_nlf_multiturn.jsonl`, `lvlm_nlf_single_turn.jsonl`, `vqa_aokvqa.jsonl`, `vqa_scienceqa.jsonl`, `vqa_tallyqa.jsonl`
- SFT uses the `messages` field directly (full conversation format with system prompt)
- SFT system prompt matches the shorter VL_ASSISTANT_SYSTEM_PROMPT (no thought/answer format)

### GRPO Prompts (hardcoded in `src/vlm_grpo/prompts.py`)

**A1 (Initial Answer) — `build_initial_answer_prompt(question)`**:
- System: `VL_ASSISTANT_SYSTEM_PROMPT` — "You are a helpful vision-language assistant..."
- User: [image] + question
- Model generates A1

**F1 (Critic Feedback) — `build_critic_prompt(question, answer1, model_type="qwen2vl")`**:
- System: `FEEDBACK_CRITIC_SYSTEM_PROMPT` — "You are a visual question answering critic..."
- **Roles are FLIPPED** (matches SFT training data):
  - Assistant: [image] + question (for Qwen2VL; LLaVA hoists image to system)
  - User: answer1
- Model generates feedback as assistant

**A2 (Refined Answer) — `build_refiner_prompt(question, answer1, feedback1)`**:
- System: `VL_ASSISTANT_SYSTEM_PROMPT`
- User: [image] + question
- Assistant: answer1
- User: feedback1
- Model generates A2

### Two-Reward Design [SCoRe¹, Critique-GRPO²]

GRPO³ uses two separate reward signals per trajectory — one for response quality (drives A1+A2 log-prob update), one for feedback quality (drives F1 log-prob update)². Both are computed in `src/vlm_grpo/rewards/composition.py`. Full per-component walkthrough lives in [`docs/rewards.md`](docs/rewards.md); the section below is a summary that matches the code.

F1 emits `<think>...</think>` followed by `\boxed{CORRECT|INCORRECT}` (no `<answer>` tag). A1/A2 emit `<think>...</think><answer>(X)</answer>`. Extraction is tag-strict: missing tag → empty extraction → wrong outcome (no special-case overrides).

**Response Reward** (drives A1 + A2 log-probs)

```
r_response = w_a1_corr · R_a1_correctness
           + w_a1_fmt  · R_a1_format
           + w_a2_corr · R_a2_correctness
           + w_a2_fmt  · R_a2_format
```

- `R_a1_correctness`, `R_a2_correctness`: ±1 for deterministic types (MCQ / yesno / numeric); continuous `2·score − 1` for counting / open. SCoRe Stage I anchor¹ + standard GRPO outcome reward³
- `R_a1_format`, `R_a2_format`: binary {0, +1}. +1 only when **both** `<think>...</think>` and `<answer>...</answer>` are present **and** the inner content is a clean atomic answer (MCQ letter / int / yes-no / parseable numeric). DeepSeek-R1⁴

The `ResponseRewardWeights` dataclass still carries `w_no_regression` and `w_wr_bonus` fields (and the legacy `R_no_regression` / `R_wr_bonus` components remain in `composition.py`), but both active runs set them to 0 — they're vestigial and not documented further here. See `docs/rewards.md` and the git history for the legacy multi-turn-transition formulas.

**Feedback Reward** (drives F1 log-prob)

```
r_feedback = w_ver  · R_verification
           + w_fmt  · R_fb_format
```

- `R_verification`: ±1 from `\boxed{CORRECT|INCORRECT}` extraction matching A1's actual correctness. **No keyword fallback on `<think>` prose** (that path was a noise source). Missing / unparseable boxed → −1. LLaVA-Critic-R1 / CriticGPT-style verdict head
- `R_fb_format`: binary {0, +1}. +1 only when both `<think>...</think>` and `\boxed{...}` are present **and** the boxed inner is exactly `CORRECT` or `INCORRECT` (case-insensitive). DeepSeek-R1⁴

`FeedbackRewardWeights.w_downstream` is similarly vestigial — the Critique-GRPO downstream-aware reward and its asymmetric verification gate still exist in `compute_feedback_reward_breakdown` but `w_downstream=0` in both active runs (and the PAG path zeroes downstream unconditionally).

#### Reward Components

**Response Reward (per trajectory):**

| Component | Raw Range | Logic | Source |
|-----------|-----------|-------|--------|
| a1_correctness | {−1, +1} (det) / [−1, +1] (open) | Strict-tag extraction → match GT | SCoRe Stage I¹ |
| a1_format | {0, +1} | `<think>` + `<answer>` + clean atomic inner | DeepSeek-R1⁴ |
| a2_correctness | {−1, +1} (det) / [−1, +1] (open) | Strict-tag extraction → match GT (continuous via verifier score) | GRPO³ |
| a2_format | {0, +1} | Same structural+atomic check as A1 | DeepSeek-R1⁴ |

**Feedback Reward (per trajectory):**

| Component | Raw Range | Logic | Source |
|-----------|-----------|-------|--------|
| verification | {−1, +1} | `\boxed{CORRECT\|INCORRECT}` matches A1 truth (boxed-only, no keyword fallback) | LLaVA-Critic-R1 |
| fb_format | {0, +1} | `<think>` + `\boxed{}` present, boxed inner is exactly `CORRECT`/`INCORRECT` | DeepSeek-R1⁴ |

#### PAG segment rewards (active path)

The `pag-faithful` run uses a separate per-segment composer in `composition.py`:

- `compute_pag_response_breakdown` returns a `PAGSegmentRewardBreakdown` carrying `r_a1` and `r_a2` as **separate scalars** rather than one pooled response reward. Components are binary {0, 1} (PAG paper convention, arXiv:2506.10406):
  - `r_a1 = w_a1_corr · R_a1_corr_01 + w_a1_fmt · R_a1_fmt_01`
  - `r_a2 = w_a2_corr · R_a2_corr_01 + w_a2_fmt · R_a2_fmt_01 + α · (R_a2_corr_01 − R_a1_corr_01)` — the α shaping bonus is added **to A2 only**, matching the released PAG implementation (binary accuracies in the bonus, not full weighted rewards).
  - `pag_shaping_alpha=1.0` is the PAG paper default (`rs_coef=1`).
  - **Selective revision gate**: when `--use_selective_revision` is on, trajectories whose F1 emits `\boxed{CORRECT}` skip A2; the breakdown sets `r_a2=None` and `gated=True`. The trainer's PAG branch excludes gated trajectories from the A2 K-group baseline and contributes 0 A2 policy loss for them.
- `compute_pag_feedback_breakdown` returns a binary {0, 1} verification + format reward; PAG has no downstream component (γ=0 turn-independent), so `w_downstream` is ignored on this path.

#### Other reward modes

- **Rescaled rewards** (`--use_rescaled_rewards`): each raw component above is rescaled to [0, 1] by `_to_unit(raw, lo, hi)` before weighting, so per-unit-weight gradient magnitude is equalised across components and `total_reward ∈ [0, 1]` for convex weights. Used by the active `kl-fix-no-bonus` run. See `compute_response_reward_breakdown_01` / `compute_feedback_reward_breakdown_01`.
- **Single-turn A1 baseline** (`--single_turn_a1`): F1+A2 skipped; reward = `0.9·R_a1_correct_01 + 0.1·R_a1_format_01` ∈ [0, 1]. Code path retained for the baseline-A1 run that produced the ckpt-1000 init both active runs use, but no current run exercises it.

#### Implementation Details

- **Loss**: `--loss_type grpo` (vanilla GRPO) is used by both active runs. Dr. GRPO⁷ (`--loss_type dr_grpo`) — removes std normalization to avoid low-variance reward bias — is still available but not exercised in the active runs.
- **KL estimator**: Schulman k3⁸ unbiased estimator `exp(Δ) − Δ − 1`, aggregated as `sum / max_completion_length` (Dr. GRPO–consistent), applied per-turn with independent `a1_kl_coeff`, `a2_kl_coeff`, `fb_kl_coeff`. KL is computed against a **frozen reference model** (`--ref_model_init_from_checkpoint` loads baseline-A1 ckpt-1000 as the ref distribution, so step-0 KL is 0 against the init).
- **Empty-completion handling**: trajectories with 0 completion tokens (immediate EOS) emit empty tensors for both `old_lp` and `new_lp`, so the `sum / max_len` policy / KL aggregations cleanly evaluate to 0. Earlier code used a `[0.0]` sentinel that injected a spurious `A/max_len` term — fixed in commit `95f091b`.
- **GDPO normalization** (`--use_gdpo_normalization`): per-component K-group advantage normalization (Liu 2026, arXiv:2601.05242), then weighted sum, then batch-renormalize. Equalises per-component gradient contribution. Used by `kl-fix-no-bonus`; the PAG run uses the two-adapter routing for role-isolation at the parameter level and leaves GDPO off.
- **DAPO** (`--use_dynamic_sampling`, `--clip_high`): drops zero-variance K-groups and uses asymmetric PPO clipping (paper: 0.28 upper, 0.2 lower). Available but **off** in both active runs — `--clip_high 0.2` symmetric, no dynamic sampling.
- **A2 PPO temperature divisor** (Bug #1 fix, commit `c825376`): vLLM samples A2 at T=0.7 returning `log softmax(logits/0.7)`. The HF forward pass now divides `shift_logits` by the per-turn `sampling_temperature` before `log_softmax`, so `exp(new_lp − old_lp) = 1.0` at step 0. A1/F1 (T=1.0) are no-ops. Verified by `scripts/verify/verify_temperature_consistency.py` and `tests/test_temperature_consistency.py`.
- **`reward/a2_mean` denominator** (Bug #3 fix, commit `c825376`): under PAG selective revision, gated trajectories carry `r_a2=0.0` placeholders. The headline metric now denominates over `a2_active_mask.sum()` (PAG) or the full tensor (non-PAG), and the cross-rank all_reduce sums `(a2_sum_local, a2_count_local)` separately rather than averaging per-rank means. Verified by `tests/test_a2_reward_mean_dilution.py`.

#### References

1. **SCoRe** — Kumar et al., "Training Language Models to Self-Correct via Reinforcement Learning" (2024). [arXiv:2409.12917](https://arxiv.org/abs/2409.12917)
   → Shaped reward `α·(r_a2 − r_a1)`, Stage I first-turn KL anchor, Stage II self-correction regularizer
2. **Critique-GRPO** — Zhang et al., "Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback" (2025). [arXiv:2506.03106](https://arxiv.org/abs/2506.03106)
   → Two-reward design (separate response + feedback heads), downstream-aware feedback reward
3. **GRPO** — Shao et al., "DeepSeekMath" (2024). [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
   → Group Relative Policy Optimization, advantage = reward − group mean
4. **DeepSeek-R1** — Guo et al., "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL" (2025). [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
   → `<think>/<answer>` format reward, β=0.001 KL anchor
5. **ReST-MCTS** — Zhang et al., "ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search" (2024). [arXiv:2406.03816](https://arxiv.org/abs/2406.03816)
   → Asymmetric verification penalties (reward correction > penalize regression)
6. **Self-Refine** — Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback" (2023). [arXiv:2303.17651](https://arxiv.org/abs/2303.17651)
   → Over-refinement failure mode, stability when already correct
7. **Dr. GRPO** — Liu et al., "Understanding R1-Zero-Like Training: A Critical Perspective" (2025). [arXiv:2503.20783](https://arxiv.org/abs/2503.20783)
   → Remove std normalization from GRPO (fixes low-variance-group bias)
8. **Schulman k3 KL estimator** — Schulman, "Approximating KL Divergence" (2020). [joschu.net/blog/kl-approx](http://joschu.net/blog/kl-approx.html)
   → Unbiased, always-positive `exp(Δ) − Δ − 1` used in per-turn KL

### Prompt Mismatch Warning
The balanced_70k dataset's `messages` contain a system prompt with `Thought: [reasoning] / Answer: [final answer]` format. This is NOT the prompt used during GRPO training. GRPO hardcodes its own prompts in `prompts.py`. If you switch to using the dataset's messages, the prompt format will change.

## Current Experiments: LIVR-v2 9K with Pattern-A self-reflection

Both active runs init the response adapter from the `baseline-a1`
checkpoint-1000 (61.4% BLINK avg, +5.3pp over base) and KL-anchor against
that same checkpoint via `--ref_model_init_from_checkpoint` (so step-0 KL
is 0 against the init). They share the two-adapter routing, vLLM-native
loss path, K=12, vanilla GRPO with symmetric PPO clip 0.2, LR=1e-5 with
56-step warmup, grad_acc=8, and **uniform** `A1_KL_COEFF = A2_KL_COEFF =
FB_KL_COEFF = 220` (the older 2500× A1 anchor was tuned for an
anchored-A1-only regime that the active runs are not in).

| Job | YAML | Notes |
|---|---|---|
| `pag-faithful` | `job-...-pag-faithful.yaml` | PAG (arXiv:2506.10406) segment rewards via `compute_pag_response_breakdown` + `compute_pag_feedback_breakdown`; selective revision gate (`--use_selective_revision`) suppresses A2 when F1 emits `\boxed{CORRECT}`; A1 actively trained (no freeze); `pag_shaping_alpha=1.0`. Weights: `0.9·corr + 0.1·fmt` per segment for both response heads and F1. Branch `grpo-kl-ref-fix`. |
| `kl-fix-no-bonus` | `job-...-two-adapters-kl-fix-no-bonus.yaml` | Rescaled rewards (`--use_rescaled_rewards`) + GDPO normalization (`--use_gdpo_normalization`); two-adapter routing freezes A1 by setting `W_A1_correctness=0`. Response weights: `w_a2_corr=0.9, w_a1_fmt=0.05, w_a2_fmt=0.05`. Feedback weights: `w_verification=0.9, w_fb_format=0.1`. Both adapters r=64 α=128 (feedback matched to baseline-A1 capacity). Branch `grpo-kl-ref-fix`. |

Earlier runs (baseline-A1, the frozen-a1-mt-* line, the two-adapter
WR-bonus A/B, the single-fb-only arms, curriculum DAPO references) are
documented in `experiments.md` and the git history. Both active runs
inherit their `INIT_CHECKPOINT` and KL ref from the `baseline-a1` LoRA
that line produced.

**Active reward-weight regimes**:

| Regime | YAML | W_A1_corr | W_A1_fmt | W_A2_corr | W_A2_fmt | W_verification | W_fb_format | Response sum | Feedback sum |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PAG-faithful | `pag-faithful` | 0.9 | 0.1 | 0.9 | 0.1 | 0.9 | 0.1 | 1.0 / segment | 1.0 |
| kl-fix-no-bonus | `kl-fix-no-bonus` | 0.0 | 0.05 | 0.9 | 0.05 | 0.9 | 0.1 | 1.0 | 1.0 |

PAG sums "per segment" — `r_a1` and `r_a2` are independent K-group
baselines, not pooled. The α=1 shaping bonus on A2 (added on top of the
0.9/0.1 weighted reward) lets `r_a2` reach `1.0 + α = 2.0` on a WR
transition and dip to `−α = −1.0` on a RW; the released PAG code applies
α to raw binary accuracies, not weighted rewards.

### Dataset: LIVR-v2 9K
- **Build pipeline**: `k8s/job-build-livr-v2-sources.yaml` → `k8s/job-build-livr-v2.yaml`
- **Output**: `/outputs/livr_v2/data/livr_v2_9task_train.jsonl` (full) or
  `/outputs/livr_v2/data/livr_v2_9k_curriculum_filtered.jsonl` (curriculum subset)
- **Tasks (1000 each)**: Counting, Jigsaw, ObjLoc, ArtStyle, RelReflect, VisCorr, SemCorr, FunCorr, VisSim
- **Image format**: a SINGLE pre-composited PNG per sample with all candidate sub-images
  + region labels drawn in (matches LIVR paper's composite design but does the
  composing offline rather than at inference time). Driven by
  `scripts/data/livr_v2/livr_common.py` (`create_reference_and_options`,
  `create_side_by_side`, `create_grid_image`).
- All tasks are MCQ with ground truth in `(A)` format. `answer_type=mcq`,
  deterministic matching (`VLM_USE_LLM_JUDGE=0`).
- **Difficulty profiling** (for curriculum): `k8s/job-profile-livr-v2-9k-difficulty.yaml`
  drives `scripts/data/profile_difficulty_a1.py` →
  `scripts/analysis/difficulty_buckets.py`, then
  `scripts/data/filter_by_difficulty.py` drops un-learnable buckets.

### Pattern-A self-reflection (single user turn per turn)
A1, F1, A2 all use a single-user-message format with no custom system prompt
(Qwen's chat-template default fires). The candidate answer A1 and the feedback F1
are quoted as TEXT inside the next user message rather than chained as separate
assistant/user turns. Eval-side wrappers in the BLINK PatternA YAMLs mirror
`src/vlm_grpo/prompts.py` byte-for-byte. See `src/vlm_grpo/prompts.py` docstring
for the literature precedent (LLaVA-Critic-R1, Critique-GRPO, CriticGPT).

### Output format (think + answer / boxed)
Tag-format prompting is on unconditionally — the `--use_think_answer_tags` /
`--use_answer_tag_only` toggles were removed in the audit cleanup (commit
`c825376`).
- A1/A2 user message appends a tag-format instruction (`THINK_ANSWER_INSTRUCTION` env in YAMLs); expected output: `<think>reasoning</think><answer>(A)</answer>`.
- F1 user message appends a separate verifier instruction (`F1_VERIFIER_INSTRUCTION`); expected output: `<think>reasoning</think> \boxed{CORRECT|INCORRECT}`. F1 uses `\boxed{}` rather than `<answer>` for the verdict.
- Format rewards (`R_a1_format`, `R_a2_format`, `R_fb_format`) are all **binary {0, +1}** — +1 only when both tag pairs are present AND the inner content is a clean atomic answer (or `CORRECT`/`INCORRECT` for F1). No partial-credit scheme.

### Key Training Config (active runs)
Shared across `pag-faithful` and `kl-fix-no-bonus`:
- K=12 samples, batch=2, grad_acc=8, lr=1e-5 with 56-step linear warmup, **vanilla GRPO** loss
- KL: β=0.001, uniform per-turn coefficients `a1_kl_coeff = a2_kl_coeff = fb_kl_coeff = 220` against the baseline-A1 ckpt-1000 reference (k3 aggregated as `sum/max_completion_length`)
- LoRA r=64 α=128, all completions capped at 386 tokens, feedback_temp=1.0, a2_temp=0.7
- Symmetric PPO clip `--clip_range 0.2 --clip_high 0.2`. **No DAPO** (no dynamic sampling, no clip-higher).
- vLLM colocate with sleep mode, gpu_mem=0.30 (PAG) / 0.35 (kl-fix; second frozen ref model on each rank)
- A1 actively trained (no `--freeze_a1_steps`); A1 is anchored only through the uniform KL coefficient and (in `kl-fix-no-bonus`) through `W_A1_correctness=0`
- `save_steps=100` (PAG) / `250` (kl-fix), in global_step

### Known Issues
- Multi-view_Reasoning + Visual_Correspondence on BLINK score near random — these
  tasks aren't in LIVR training (Multi-view) or use a denser-marker composite
  (Correspondence) that diverges from LIVR's Source/Target layout.
- BLINK IQ_Test under `vlmevalkit` exact-matching is unscorable for verbose base
  predictions; needs the GPT judge enabled to extract a letter from prose.

## Conceptual concerns and pivots (literature audit, 2026-05)

A multi-agent audit reviewed whether **same-model in-loop self-critique
under RL** has positive precedent at our scale for visual reasoning.
The headline finding from the literature review:

> The conceptual frame — a single 7B VLM, no external verifier, in-loop
> critique-then-refine trained jointly with RL — is **not supported by
> the 2024-2026 literature for visual reasoning**.

Every comparable paper that ostensibly does "self-critique" has an
asymmetry we do not have:

- **SCoRe** (Kumar 2024, arXiv:2409.12917) — no critique step at all.
  A2 conditions on A1 plus a fixed "try again" prompt; our F1 turn is
  an **addition** not validated by SCoRe.
- **Critique-GRPO** (Zhang 2025, arXiv:2506.03106) — critiques generated
  by **GPT-4o (external teacher)**, not by the policy itself.
- **CriticGPT** (McAleese 2024) — separate trained critic on
  human-labeled inserted-bug data.
- **LLaVA-Critic-R1** (Yu 2025, arXiv:2509.00676) — same model, but
  self-critique applied only at **test time**, not in-loop during RL.
- **CRITIC** (Gou 2023) — external tool grounding.
- **Reflexion** (Shinn 2023) — external environmental success signal.

The "honest F1 hurts A2" pattern we measure (v3 trajectory analysis:
P(A2 right | A1 wrong, F1 honest) = 8.1% vs P(A2 right | A1 wrong,
F1 sycophantic) = 15.2%, a **−7.1pp lift** from honest critique) is
exactly the canonical **generation–verification-gap** failure mode
predicted by:

- Huang 2023, *LLMs Cannot Self-Correct Reasoning Yet* (arXiv:2310.01798)
- Tyen 2023, *LLMs find errors but need location* (arXiv:2311.08516)
- Stechly & Kambhampati 2024, *Self-Verification Limitations*
  (arXiv:2402.08115)
- Kambhampati et al. survey (arXiv:2406.01297)

**Recommended pivots with positive published evidence at our scale:**

1. **Critic-V** (Zhang CVPR 2025, arXiv:2411.18203) — separate
   DPO-trained VLM critic + vanilla GRPO reasoner. Beats GPT-4V on
   5/8 multimodal reasoning benchmarks. Most directly applicable.
2. **DPO over (A1_wrong, A2_corrected) pairs** — drop F1 entirely,
   mine self-correction pairs, train with DPO. Removes the
   verdict-calibration problem (LLaVA-SCo, ISR-DPO, 2025).
3. **Process Reward Models for visual reasoning** (PROPA
   arXiv:2511.10279, VRPRM arXiv:2508.03556) — step-level supervision
   instead of outcome + critique.
4. **External LLM judge as F1** — use a stronger model (GPT-4o /
   Claude-class) as the critic; matches Critique-GRPO's setting.
5. **Two-stage SFT critic → RL** — supervise F1 on gold critique
   pairs first, then RL only A1+A2. Industrial pattern from CriticGPT
   and the original LLaVA-Critic.

The literature consensus is that the **fix is architectural, not in
the reward weights**. No amount of α / weight tuning closes the
−7.1pp gap our 7B same-model setup produces.

**Update (post-audit, 2026-05):** an implementation audit in commit
`c825376` found and fixed two load-bearing bugs that were independent
of the conceptual concerns above — **Bug #1** (A2 PPO ratio temperature
mismatch: HF computed `new_lp` at T=1.0 while vLLM sampled at T=0.7,
biasing `exp(new_lp − old_lp)` ~6.86× off from 1.0 at step 0 and
attenuating A2 surrogate gradient) and **Bug #3** (`reward/a2_mean`
diluted by gated-trajectory zero placeholders under PAG selective
revision). The architectural pivots (Critic-V, Vision-SR1, PAG, etc.)
remain open. The current `pag-faithful` run is the first arm to test
one of those pivots (selective revision + per-segment rewards) on the
post-audit codebase.

## How to Run

Always use `uv` for running.

### Install

```bash
uv sync
uv sync --extra dev  # for testing
```

### Linting

```bash
ruff format src/ tests/ train_self_reflection.py scripts/
ruff check src/ tests/ train_self_reflection.py scripts/
ruff check src/ tests/ train_self_reflection.py scripts/ --fix
```

### Tests

```bash
uv run pytest tests/ -v
```

### Training

```bash
# Full training (4 GPUs, Qwen2.5-VL base)
accelerate launch --config_file k8s/multi_gpu.yaml --num_processes=4 \
    train_self_reflection.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --model_type qwen2vl \
    --dataset_path /outputs/grpo_data/balanced_70k.jsonl \
    --output_dir /outputs/grpo_qwen_sr_v9_20k \
    --image_base_dir /outputs/image_base \
    --use_vllm --freeze_vision_tower
```

## Experiment Log

All experiments are tracked in `experiments.md`. After each experiment:
1. Record final metrics table (early 10% vs late 10%): A1/A2 acc, WR, RW, entropy, rewards, F1 tag leak
2. List what worked, what failed, and root cause (1-2 sentences each)
3. Note artifacts (output dir, wandb run, k8s yaml, commit)
4. Keep entries concise — no config tables if same as prior experiment (just note diffs)

## Cluster Operations

**Helper Pod**: All `kubectl exec` commands for reading logs, checking outputs, or running scripts on the PVC must use the helper pod `vlm-jupyter-eval2`. If the pod is in `Completed` or `Failed` state, redeploy it first:

```bash
kubectl delete pod vlm-jupyter-eval2 --ignore-not-found
kubectl apply -f /Users/kuntalkokate/svcl-projects/vlm-self-reflection/k8s/jupyter-1gpu-test.yaml
kubectl wait --for=condition=Ready pod/vlm-jupyter-eval2 --timeout=120s
```

Do NOT exec into training job pods for log analysis — use the helper pod which has the shared PVC mounted at `/outputs/`.

## MUST Follow

1. **Imports**: Standard library first, then third-party, then local. Alphabetized within groups.
2. **Type Hints**: Required on all function parameters and return types.
3. **Docstrings**: Google-style with Args/Returns sections.
4. **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_SNAKE_CASE for constants.
5. **Logging**: Use `logging.basicConfig()` and `logger = logging.getLogger(__name__)`.
6. **Entry Points**: All scripts MUST use `if __name__ == "__main__": main()` guard.
7. **Dataclasses**: Use `@dataclass` for structured results with `to_dict()` methods.

## MUST NOT Change

1. **Lazy Imports in Model Classes**: Do NOT move imports from `__init__` methods to module level in model wrapper classes.

## Repository Structure

```
.
├── train_self_reflection.py      # Main training entry point
├── pyproject.toml                # Package + tooling config (ruff, mypy, pytest)
├── uv.lock                       # Locked dependency versions
├── CLAUDE.md                     # This file
├── experiments.md                # Experiment log (v1 → v10)
├── docs/
│   ├── rewards.md                # Per-component reward walkthrough + landscape tables
│   └── troubleshooting.md        # GRPO training diagnostics (LoRA probe, KL, DAPO, etc.)
│
├── src/vlm_grpo/                 # Installable package
│   ├── config.py                 # Dataclasses: RolloutConfig, ResponseRewardWeights, FeedbackRewardWeights, BaselineA1RewardWeights, SelfReflectionConfig
│   ├── critic_grpo.py            # SelfReflectionGRPOTrainer: GRPO loop, per-turn KL, policy update, GDPO normalization
│   ├── data.py                   # Dataset loading + answer_type detection
│   ├── prompts.py                # A1 / F1 / A2 prompt builders + system prompts
│   ├── rollout.py                # HF generate() rollout engine (3 turns per sample)
│   ├── vllm_rollout.py           # vLLM rollout engine with sleep-mode for GPU sharing (returns completion token_ids end-to-end)
│   ├── trajectory.py             # Answer extraction, tag parsing, MCQ letter normalization, boxed verdict extraction
│   ├── utils.py                  # Seeding, env setup, normalized edit distance
│   └── rewards/
│       ├── composition.py        # Response + Feedback breakdowns; raw and [0,1]-rescaled paths; baseline A1
│       ├── correctness.py        # A1 / A2 correctness rewards (binary / continuous)
│       ├── deterministic.py      # MCQ / YesNo / numeric answer matching
│       ├── judge_llm.py          # Optional LLM judge (enabled via VLM_USE_LLM_JUDGE=1)
│       └── verifier.py           # Top-level verify_answer dispatcher (deterministic + judge); DETERMINISTIC_TYPES set
│
├── tests/                        # pytest suite
│   ├── test_a2_reward_mean_dilution.py # Bug #3: a2_mean denominator under PAG gating + cross-rank reduce
│   ├── test_adapter_routing.py        # Per-turn LoRA adapter dispatch
│   ├── test_baseline_a1_rewards.py    # Single-turn baseline reward (binary {0,1}) — vestigial path
│   ├── test_config_wiring.py          # CLI args → SelfReflectionConfig plumbing
│   ├── test_correctness.py            # A1 / A2 correctness rewards
│   ├── test_deterministic.py          # MCQ / YesNo / numeric matchers
│   ├── test_difficulty_buckets.py     # Curriculum difficulty bucketing
│   ├── test_dynamic_sampling.py       # DAPO dynamic-sampling K-group filter (off in active runs)
│   ├── test_gdpo.py                   # Per-component K-group advantage normalization
│   ├── test_init_from_checkpoint.py   # LoRA init without inheriting global_step + optimizer state restore
│   ├── test_kl_ref_model.py           # `--ref_model_init_from_checkpoint` KL ref distribution
│   ├── test_kl_term.py                # Per-turn Schulman k3 KL (sum/max_len aggregation)
│   ├── test_lr_warmup.py              # Linear LR ramp 0 → peak over N steps
│   ├── test_pag_advantage_split.py    # PAG per-segment K-group baselines (A1 over all K, A2 over non-gated)
│   ├── test_pag_metrics.py            # PAG wandb signals + per-rank reduce
│   ├── test_pag_segment_rewards.py    # `compute_pag_response_breakdown` / `compute_pag_feedback_breakdown`
│   ├── test_pag_selective_revision.py # Gate F1=CORRECT → skip A2 → r_a2 = None
│   ├── test_rescaled_rewards.py       # [0,1]-rescaled reward path
│   ├── test_temperature_consistency.py # Bug #1: A2 PPO ratio when HF divides shift_logits by sampling T
│   ├── test_trajectory.py             # Tag parsing + answer/boxed extraction
│   ├── test_verification.py           # F1 verdict accuracy reward
│   ├── test_verifier.py               # Top-level verify_answer
│   ├── test_vllm_token_passthrough.py # vLLM completion token_ids audit
│   └── test_wr_bonus.py               # WR-bonus reward component (vestigial — w_wr_bonus=0 in active runs)
│
├── scripts/
│   ├── analysis/
│   │   └── difficulty_buckets.py        # Bucket samples by A1 pass-rate
│   ├── data/
│   │   ├── blink_composite_rebuild.py   # Composite BLINK images to match training format
│   │   ├── filter_by_difficulty.py      # Drop trivial / brick-wall buckets for curriculum
│   │   ├── profile_difficulty_a1.py     # Run A1 over training data, log per-sample correctness
│   │   └── livr_v2/                     # LIVR-v2 dataset builders
│   │       ├── build_*.py               #   9 per-task builders
│   │       ├── livr_common.py           #   shared composite/grid layout helpers
│   │       ├── download_sources.py      #   pull source datasets (COCO, NIGHTS, ...)
│   │       ├── dedup.py                 #   per-task dedup + cross-split overlap removal
│   │       ├── merge.py                 #   merge 9 tasks into one JSONL
│   │       ├── repair_counting_labels.py
│   │       └── rerun_after_pipeline.sh
│   └── verify/
│       ├── verify_pag_gradient.py            # Numerical gradient check for PAG policy + KL loss
│       ├── verify_temperature_consistency.py # Bug #1 harness: 6.86× ratio drift on buggy path vs 1.0 on fixed path
│       └── verify_vllm_native_loss.py        # Standalone vLLM⇄HF equivalence harness
│
└── k8s/                                 # Kubernetes configs
    │
    ├── ─── Training (active) ───
    ├── job-qwen-grpo-livr-v2-9k-pag-faithful.yaml                 # Active: PAG segment rewards + selective revision gate, A1 actively trained, uniform KL anchor 220
    ├── job-qwen-grpo-livr-v2-9k-two-adapters-kl-fix-no-bonus.yaml # Active: rescaled rewards + GDPO + two-adapter; A1 frozen via W_A1_corr=0; uniform KL anchor 220
    ├── # Earlier YAMLs (baseline-a1, frozen-a1-mt-*, two-adapter WR-bonus A/B, single-fb-only, curriculum) — see experiments.md and git history
    │
    ├── ─── Data builders ───
    ├── job-build-livr-v2-sources.yaml                    # Download LIVR source datasets
    ├── job-build-livr-v2.yaml                            # Build composite LIVR-v2 dataset
    ├── job-profile-livr-v2-9k-difficulty.yaml            # Difficulty profiler for curriculum
    │
    ├── ─── Eval (BLINK PatternA over composite TSVs) ───
    ├── job-eval-vlmevalkit-blink-base-vanilla-v2.yaml             # Vanilla base, single shard
    ├── job-eval-vlmevalkit-blink-base-vanilla-v2-shard{1,2}.yaml  # Sharded vanilla base
    ├── job-eval-vlmevalkit-blink-curriculum-final-v2-shard{1,2}.yaml          # Curr-final main
    ├── job-eval-vlmevalkit-blink-curriculum-final-v2-shard{1,2}-overflow.yaml # Slow-tail
    ├── job-eval-vlmevalkit-blink-curriculum-final-v2-tail-overflow.yaml       # IQ + SpatRel
    ├── job-eval-vlmevalkit-blink-no-dapo-k8-ckpt1000-v2-shard{1,2}.yaml          # No-dapo main
    ├── job-eval-vlmevalkit-blink-no-dapo-k8-ckpt1000-v2-shard{1,2}-overflow.yaml # Slow-tail
    ├── job-eval-vlmevalkit-blink-no-dapo-k8-ckpt1000-v2-tail-overflow.yaml       # IQ + SpatRel
    │
    └── ─── Infrastructure ───
        ├── jupyter-1gpu-dev.yaml         # Helper pod for PVC reads + scratch scripts
        ├── pod-vllm-native-verify.yaml   # A100 pod for running scripts/verify/verify_vllm_native_loss.py
        ├── multi_gpu.yaml                # Accelerate multi-GPU launch config
        ├── deepspeed_zero3.yaml          # Accelerate + DeepSpeed ZeRO-3 launch config
        └── ds_zero3_config.json          # DeepSpeed ZeRO-3 engine config
```
