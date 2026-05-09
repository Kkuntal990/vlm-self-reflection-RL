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
           + w_no_reg  · R_no_regression
```

- `R_a1_correctness`, `R_a2_correctness`: ±1 for deterministic types (MCQ / yesno / numeric); continuous `2·score − 1` for counting / open. SCoRe Stage I anchor¹ + standard GRPO outcome reward³
- `R_a1_format`, `R_a2_format`: binary {0, +1}. +1 only when **both** `<think>...</think>` and `<answer>...</answer>` are present **and** the inner content is a clean atomic answer (MCQ letter / int / yes-no / parseable numeric). DeepSeek-R1⁴
- `R_no_regression`: A1↔A2 transition reward. With `reward_shaping_alpha > 0` (active runs use α=1) it is **shaped**: `α · (R(A2) − R(A1))`. With α=0 it falls back to a discrete table — deterministic: `RR=+1, RW=−2, WR=+3, WW=0`; open: `RR=+1, RW=−3, WR=+2, WW=0`. SCoRe shaping¹ + ReST-MCTS asymmetry⁵

**Feedback Reward** (drives F1 log-prob)

```
r_feedback = w_down · R_downstream_gated
           + w_ver  · R_verification
           + w_fmt  · R_fb_format
```

- `R_downstream`: Did F1 lead to a correct A2? Critique-GRPO downstream-aware reward². With α > 0 (active α=1): `R(A2) + α·(R(A2) − R(A1))` → at α=1 yields `WR=+3, RW=−3, RR=+1, WW=−1`. With α=0 deterministic: `RR=+1, RW=−1.5, WR=+3, WW=−1`; open: `RR=+1, RW=−2, WR=+2, WW=−1`
- `R_verification`: ±1 from `\boxed{CORRECT|INCORRECT}` extraction matching A1's actual correctness. **No keyword fallback on `<think>` prose** (that path was a noise source). Missing / unparseable boxed → −1. **Replaces the old keyword-based `calibration` reward.** LLaVA-Critic-R1 / CriticGPT-style verdict head
- `R_fb_format`: binary {0, +1}. +1 only when both `<think>...</think>` and `\boxed{...}` are present **and** the boxed inner is exactly `CORRECT` or `INCORRECT` (case-insensitive). Pure structural anchor — no word-count component anymore. DeepSeek-R1⁴

**Asymmetric downstream gate** (`composition.py` `compute_feedback_reward_breakdown`):

```python
if r_verification > 0: r_downstream_gated = r_downstream         # full bidirectional
else:                   r_downstream_gated = min(r_downstream, 0)  # negative-only flow
```

When F1's verdict is calibrated, the full downstream signal flows. When the verdict is wrong, only the negative arm flows — F1 is still penalised for actively causing an `RW` regression, but cannot farm a positive bonus from a sycophantic `\boxed{CORRECT}` whose A2 happened to variance-flip right.

#### Reward Components

**Response Reward (per trajectory):**

| Component | Raw Range | Logic | Source |
|-----------|-----------|-------|--------|
| a1_correctness | {−1, +1} (det) / [−1, +1] (open) | Strict-tag extraction → match GT | SCoRe Stage I¹ |
| a1_format | {0, +1} | `<think>` + `<answer>` + clean atomic inner | DeepSeek-R1⁴ |
| a2_correctness | {−1, +1} (det) / [−1, +1] (open) | Strict-tag extraction → match GT (continuous via verifier score) | GRPO³ |
| a2_format | {0, +1} | Same structural+atomic check as A1 | DeepSeek-R1⁴ |
| no_regression | depends on α | α>0: `α·(R(A2)−R(A1))` (active α=1 → ±2). α=0 det: `RR=+1, RW=−2, WR=+3, WW=0`. α=0 open: `RR=+1, RW=−3, WR=+2, WW=0` | SCoRe shaping¹ + ReST-MCTS asymmetry⁵ |

**Feedback Reward (per trajectory):**

| Component | Raw Range | Logic | Source |
|-----------|-----------|-------|--------|
| downstream | depends on α (gated) | α>0: `R(A2) + α·(R(A2)−R(A1))` (active α=1 → WR=+3, RW=−3, RR=+1, WW=−1). α=0 det: `RR=+1, RW=−1.5, WR=+3, WW=−1`. α=0 open: `RR=+1, RW=−2, WR=+2, WW=−1`. Asymmetric gate clamps to ≤0 when verification fails | Critique-GRPO² + ReST-MCTS⁵ |
| verification | {−1, +1} | `\boxed{CORRECT\|INCORRECT}` matches A1 truth (boxed-only, no keyword fallback) | LLaVA-Critic-R1 |
| fb_format | {0, +1} | `<think>` + `\boxed{}` present, boxed inner is exactly `CORRECT`/`INCORRECT` | DeepSeek-R1⁴ |

#### Optional reward modes

- **Single-turn A1 baseline** (`--single_turn_a1`): F1+A2 are skipped entirely. Reward = `w_a1_correctness · R_a1_correct_01 + w_a1_format · R_a1_format_01`, both binary {0, 1}. Default 0.9 / 0.1 → total in [0, 1]. See `compute_baseline_a1_reward_breakdown`.
- **Rescaled rewards** (`--use_rescaled_rewards`): each raw component above is rescaled to [0, 1] by `_to_unit(raw, lo, hi)` before weighting, so per-unit-weight gradient magnitude is equalised across components and `total_reward ∈ [0, 1]` for convex weights. The asymmetric downstream gate becomes `min(value, midpoint)`. Used by all recent `frozen-a1-mt-*` runs except `frozen-a1-mt-full-raw`. See `compute_response_reward_breakdown_01` / `compute_feedback_reward_breakdown_01`.
- **Per-turn alpha override** (`--response_alpha`, `--feedback_alpha`): each defaults to `−1` meaning "use `--reward_shaping_alpha`". Set them independently to vary shaping strength on the response head vs feedback head.
- **Improvement-only feedback** (`--use_improvement_reward`): `R_downstream = R(A2) − R(A1)` ∈ {−2, 0, +2}. Mutually exclusive with shaped α; α takes precedence when both are set.

#### Implementation Details

- **Loss**: Dr. GRPO⁷ (`--loss_type dr_grpo`) — removes std normalization to avoid low-variance reward bias. Vanilla GRPO (`--loss_type grpo`) is used by the recent `vanilla-warmup` / `vanilla-tokid` runs.
- **KL estimator**: Schulman k3⁸ unbiased estimator `exp(Δ) − Δ − 1`, aggregated as `sum / max_completion_length` (Dr. GRPO–consistent), applied per-turn with independent `a1_kl_coeff`, `a2_kl_coeff`, `fb_kl_coeff`¹
- **GDPO normalization** (`--use_gdpo_normalization`): per-component K-group advantage normalization (Liu 2026, arXiv:2601.05242), then weighted sum, then batch-renormalize. Equalises per-component gradient contribution. Off by default.
- **DAPO** (`--use_dynamic_sampling`, `--clip_high`): drops zero-variance K-groups and uses asymmetric PPO clipping (paper: 0.28 upper, 0.2 lower). Independent of SSR.

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

The active line of work is **multi-turn with A1 frozen** — initialise from the
`baseline-a1` checkpoint-1000 (61.4% BLINK avg, +5.3pp over base) and train F1 +
A2 only. `FREEZE_A1_STEPS=1_000_000` keeps A1 frozen for the full epoch. The
baseline and the original DAPO/no-DAPO runs are kept around as references.

**Runs ordered roughly by recency (latest first):**

| Job | YAML | Train mode | Notes |
|---|---|---|---|
| `frozen-a1-mt-vanilla-tokid` | `job-...-frozen-a1-mt-vanilla-tokid.yaml` | Vanilla GRPO + LR warmup + rescaled rewards, K=12, no DAPO | Latest. Plumbs vLLM completion `token_ids` end-to-end (audit Bug 2 fix). Reward = A2 corr/fmt + verification + fb_fmt only (no_regression and downstream zeroed). Branch `grpo-tokid-fix`. |
| `frozen-a1-mt-vanilla-warmup` | `job-...-frozen-a1-mt-vanilla-warmup.yaml` | Vanilla GRPO + LR warmup + rescaled rewards | Sanity baseline for `vanilla-tokid` before the token-id fix. Same simplified weights. |
| `frozen-a1-mt-gdpo` / `gdpo-warmup` | `job-...-frozen-a1-mt-gdpo[-warmup].yaml` | GDPO per-component K-group advantage normalization (Liu 2026) + rescaled rewards | Tests whether per-component normalization fixes the response-vs-feedback gradient-magnitude imbalance. `gdpo-warmup` adds linear LR ramp to stabilise. |
| `frozen-a1-mt-full-simple` | `job-...-frozen-a1-mt-full-simple.yaml` | Rescaled rewards, simplified weights (drop no_reg + downstream) | Isolates verification/correctness signals only. |
| `frozen-a1-mt-full` | `job-...-frozen-a1-mt-full.yaml` | Rescaled rewards, full weight set (no_reg=0.50, downstream=0.45) | Full multi-turn reward set in rescaled mode. |
| `frozen-a1-mt-full-raw` | `job-...-frozen-a1-mt-full-raw.yaml` | **Raw** (un-rescaled) rewards, full weight set, LR 1e-6 | Direct counterpart of `full` — same weights, no [0,1] rescaling, lower LR to avoid blowing up the wider-range gradient. |
| `frozen-a1-mt-r01` | `job-...-frozen-a1-mt-r01.yaml` | Rescaled rewards, full weight set | First rescaled-rewards run; resume-vs-init bug fix landed here. |
| `frozen-a1-mt` | `job-...-frozen-a1-mt.yaml` | Raw rewards, full weight set | Original frozen-A1 multi-turn run. |
| `baseline-a1` | `job-...-baseline-a1.yaml` | **Single-turn** A1 only (`--single_turn_a1`) | Reward = `0.9·R_a1_correct_01 + 0.1·R_a1_format_01` ∈ [0, 1]. β=0.001 KL. Source of the ckpt-1000 LoRA used as `INIT_CHECKPOINT` by every frozen-a1-mt run. Branch: `grpo-baseline`. |
| `curriculum` | `job-...-curriculum.yaml` | DAPO (K=16, dynamic sampling + clip-higher) on curriculum-filtered subset | Earlier reference run. Drops trivial + brick-wall difficulty buckets. |
| `curriculum-no-dapo-k8` | `job-...-curriculum-no-dapo-k8.yaml` | Vanilla GRPO, K=8, full 9-task dataset | DAPO ablation reference. |

**Active reward-weight regimes** (all multi-turn frozen-a1-mt runs share
`α=1` for response and feedback shaping, `W_A1_*=0` since A1 is frozen,
`W_A2_correctness=0.45`, `W_A2_format=0.05`):

| Regime | YAMLs | W_no_reg | W_downstream | W_verification | W_fb_format | Sums |
|---|---|---:|---:|---:|---:|---|
| Full multi-turn | `frozen-a1-mt`, `r01`, `full`, `full-raw` | 0.50 | 0.45 | 0.45 | 0.10 | 1.00 / 1.00 |
| Simplified (no shaping) | `full-simple`, `gdpo`, `gdpo-warmup`, `vanilla-warmup`, `vanilla-tokid` | 0.00 | 0.00 | 0.45 | 0.05 | 0.50 / 0.50 |

The simplified regime intentionally lets the convex-combination warning fire
(`_validate_weight_sum` logs at startup) — dropping `no_regression` and
`downstream` keeps only the per-turn correctness/format signals plus F1
verification.

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

### Think/Answer/Boxed Tags (`--use_think_answer_tags`)
- A1/A2 user message appends a tag-format instruction (`THINK_ANSWER_INSTRUCTION` env in YAMLs)
- Expected A1/A2 output: `<think>reasoning</think><answer>(A)</answer>`
- F1 user message appends a separate verifier instruction (`F1_VERIFIER_INSTRUCTION`)
- Expected F1 output: `<think>reasoning</think> \boxed{CORRECT|INCORRECT}` — F1 has its own structural convention (`\boxed{}`), distinct from A1/A2's `<answer>`
- Format rewards (response head: `R_a1_format`, `R_a2_format`; feedback head: `R_fb_format`) are all **binary {0, +1}** — +1 only when both tag pairs are present AND the inner content is a clean atomic answer (or `CORRECT`/`INCORRECT` for F1). No partial-credit `+0.5 / −0.5 / −1.0` scheme; no separate F1 tag-leak penalty.

### Key Training Config (frozen-a1-mt-vanilla-tokid — latest)
- K=12 samples, batch=2, grad_acc=1, lr=1e-5 with 100-step linear warmup, **vanilla GRPO** loss
- KL: β=0.001, all turn coefficients = 1.0 (KL k3 aggregated as `sum/max_completion_length`)
- LoRA r=64 alpha=128, all completions capped at 386 tokens, feedback_temp=1.0, a2_temp=0.7
- DAPO **off** in this run (`USE_DYNAMIC_SAMPLING=""`, `CLIP_HIGH=0.0` symmetric)
- Rescaled rewards on (`--use_rescaled_rewards`), simplified weight regime (see table above)
- vLLM colocate with sleep mode, gpu_mem=0.40
- A1 frozen (`FREEZE_A1_STEPS=1_000_000`), init from `/outputs/grpo_qwen_livr_v2_9k_baseline_a1/checkpoint-1000`
- save_steps=250 (in global_step, not samples — first checkpoint ~sample 500)

### Known Issues
- Multi-view_Reasoning + Visual_Correspondence on BLINK score near random — these
  tasks aren't in LIVR training (Multi-view) or use a denser-marker composite
  (Correspondence) that diverges from LIVR's Source/Target layout.
- BLINK IQ_Test under `vlmevalkit` exact-matching is unscorable for verbose base
  predictions; needs the GPT judge enabled to extract a letter from prose.

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
│       ├── correctness.py        # A1 / A2 correctness rewards (binary / continuous), improvement reward
│       ├── deterministic.py      # MCQ / YesNo / numeric answer matching
│       ├── feedback.py           # Downstream-aware reward (legacy entry; current path lives in composition.py)
│       ├── judge_llm.py          # Optional LLM judge (enabled via VLM_USE_LLM_JUDGE=1)
│       ├── stability.py          # No-regression reward (legacy entry; current values inlined in composition.py)
│       └── verifier.py           # Top-level verify_answer dispatcher (deterministic + judge); DETERMINISTIC_TYPES set
│
├── tests/                        # pytest suite
│   ├── test_baseline_a1_rewards.py    # Single-turn baseline reward (binary {0,1})
│   ├── test_config_wiring.py          # CLI args → SelfReflectionConfig plumbing
│   ├── test_correctness.py            # A1 / A2 correctness rewards
│   ├── test_deterministic.py          # MCQ / YesNo / numeric matchers
│   ├── test_difficulty_buckets.py     # Curriculum difficulty bucketing
│   ├── test_dynamic_sampling.py       # DAPO dynamic-sampling K-group filter
│   ├── test_feedback.py               # Downstream-aware reward
│   ├── test_gdpo.py                   # Per-component K-group advantage normalization
│   ├── test_init_from_checkpoint.py   # LoRA init without inheriting global_step + optimizer state restore
│   ├── test_kl_term.py                # Per-turn Schulman k3 KL (sum/max_len aggregation)
│   ├── test_lr_warmup.py              # Linear LR ramp 0 → peak over N steps
│   ├── test_rescaled_rewards.py       # [0,1]-rescaled reward path
│   ├── test_rollout.py                # RolloutConfig + batch rollout
│   ├── test_stability.py              # No-regression reward
│   ├── test_trajectory.py             # Tag parsing + answer/boxed extraction
│   ├── test_two_traj_composition.py   # End-to-end response + feedback composition
│   ├── test_verification.py           # F1 verdict accuracy reward
│   ├── test_verifier.py               # Top-level verify_answer
│   └── test_vllm_token_passthrough.py # vLLM completion token_ids audit
│
├── scripts/
│   ├── analysis/
│   │   └── difficulty_buckets.py        # Bucket samples by A1 pass-rate
│   └── data/
│       ├── blink_composite_rebuild.py   # Composite BLINK images to match training format
│       ├── filter_by_difficulty.py      # Drop trivial / brick-wall buckets for curriculum
│       ├── profile_difficulty_a1.py     # Run A1 over training data, log per-sample correctness
│       └── livr_v2/                     # LIVR-v2 dataset builders
│           ├── build_*.py               #   9 per-task builders
│           ├── livr_common.py           #   shared composite/grid layout helpers
│           ├── download_sources.py      #   pull source datasets (COCO, NIGHTS, ...)
│           ├── dedup.py                 #   per-task dedup + cross-split overlap removal
│           ├── merge.py                 #   merge 9 tasks into one JSONL
│           ├── repair_counting_labels.py
│           └── rerun_after_pipeline.sh
│
└── k8s/                                 # Kubernetes configs
    │
    ├── ─── Training (most recent first) ───
    ├── job-qwen-grpo-livr-v2-9k-frozen-a1-mt-vanilla-tokid.yaml   # Latest: vanilla GRPO + token-id audit fix
    ├── job-qwen-grpo-livr-v2-9k-frozen-a1-mt-vanilla-warmup.yaml  # Vanilla GRPO + LR warmup
    ├── job-qwen-grpo-livr-v2-9k-frozen-a1-mt-gdpo.yaml            # GDPO per-component normalization
    ├── job-qwen-grpo-livr-v2-9k-frozen-a1-mt-gdpo-warmup.yaml     # GDPO + LR warmup
    ├── job-qwen-grpo-livr-v2-9k-frozen-a1-mt-full.yaml            # Full-weight rescaled rewards
    ├── job-qwen-grpo-livr-v2-9k-frozen-a1-mt-full-simple.yaml     # Simplified weights (drop no_reg + downstream)
    ├── job-qwen-grpo-livr-v2-9k-frozen-a1-mt-full-raw.yaml        # Full weights, raw (un-rescaled), LR 1e-6
    ├── job-qwen-grpo-livr-v2-9k-frozen-a1-mt-r01.yaml             # First rescaled-rewards run
    ├── job-qwen-grpo-livr-v2-9k-frozen-a1-mt.yaml                 # Original frozen-A1 multi-turn run
    ├── job-qwen-grpo-livr-v2-9k-baseline-a1.yaml                  # Single-turn A1-only baseline (source of INIT_CHECKPOINT)
    ├── job-qwen-grpo-livr-v2-9k-curriculum.yaml                   # Earlier reference: curriculum DAPO
    ├── job-qwen-grpo-livr-v2-9k-curriculum-no-dapo-k8.yaml        # Earlier reference: no-DAPO K=8 ablation
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
        ├── multi_gpu.yaml                # Accelerate multi-GPU launch config
        ├── deepspeed_zero3.yaml          # Accelerate + DeepSpeed ZeRO-3 launch config
        └── ds_zero3_config.json          # DeepSpeed ZeRO-3 engine config
```
