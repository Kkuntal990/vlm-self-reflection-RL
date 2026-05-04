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

GRPO³ uses two separate reward signals per trajectory — one for response quality (drives A1+A2 log-prob update), one for feedback quality (drives F1 log-prob update)². Rewards are computed in `src/vlm_grpo/rewards/composition.py`.

**Response Reward** = w_a1 * R_a1 + w_a2 * R_a2 + w_noreg * R_noreg + w_fmt * R_fmt + w_edit * R_edit

- `a1_correctness`: Binary — is A1 correct? (+1 / -1). First-turn accuracy anchor, prevents Stage I collapse¹
- `a2_correctness`: Binary for MCQ/YesNo/Numeric (+1 / -1), continuous for counting/open (2*score - 1). Standard GRPO outcome reward³
- `no_regression`: Did A2 maintain or improve on A1? Shaped improvement reward `α·(r_a2 − r_a1)`¹ with asymmetric RW/WR values⁵. MCQ-aware: deterministic types RW=-2/WR=+3, open-ended RW=-3/WR=+2
- `a2_format`: Without tags: penalty-only (0 / -1). With tags: +0.5 / -0.5 / -1.0. `<think>/<answer>` tag format reward⁴
- `minimal_edit`: Only when both A1 and A2 are correct — rewards keeping the answer stable (0 to +1). Over-refinement mitigation⁶

**Feedback Reward** = w_down * R_down + w_cal * R_cal + w_fmt * R_fmt

- `downstream`: Did F1 lead to a correct A2? F1's value is its effect on downstream A2 correctness². MCQ-aware: deterministic RW=-1.5/WR=+3, open-ended RW=-2/WR=+2⁵
- `calibration`: Does F1 correctly assess A1's correctness? Self-critique signal⁶. Keyword-based, 7 discrete values (-1 to +1). Key variance-breaker — without it, downstream alone causes 50-75% zero-variance K-groups⁷
- `fb_format`: De-coupled from calibration (pure word count): empty/<3 words=-2, 3-6 words=-1, >6 words=0. Format anchor⁴

#### Reward Components

**Response Reward:**

| Component | Raw Range | Logic | Source |
|-----------|-----------|-------|--------|
| a1_correctness | {-1, +1} | Binary correct/wrong | SCoRe Stage I¹ |
| a2_correctness | [-1, +1] | Binary for MCQ/YesNo/Numeric; continuous for counting/open | GRPO³ |
| no_regression | {-3..+2.35} | Deterministic: RW:-2, WW:-0.5, RR:+1, WR:+2.35. Open: RW:-3, WW:-0.5, RR:+1, WR:+2.35 | SCoRe shaping¹ + ReST-MCTS asymmetry⁵ |
| a2_format | {-1..+0.5} | No tags: 0/-1. With tags: +0.5 (valid), -0.5 (bad inner), -1.0 (no tags) | DeepSeek-R1⁴ |
| minimal_edit | [0, +1] | `max(1 - 0.5*edit_dist, 0)`, only when both A1 and A2 correct | Self-Refine⁶ |

**Feedback Reward:**

| Component | Raw Range | Logic | Source |
|-----------|-----------|-------|--------|
| downstream | {-2..+3} | Deterministic: RW:-1.5, WW:-1, RR:+3, WR:+3. Open: RW:-2, WW:-1, RR:+2, WR:+2 | Critique-GRPO² + ReST-MCTS⁵ |
| calibration | [-1, +1] | 7 discrete values from keyword matching (positive/negative/mixed/doubt/neutral) | Self-Refine⁶ |
| fb_format | {-2, -1, 0} | Pure word count (de-coupled from calibration): <3:-2, 3-6:-1, >6:0 | DeepSeek-R1⁴ |

#### Implementation Details

- **Loss**: Dr. GRPO⁷ (`--loss_type dr_grpo`) — removes std normalization to avoid low-variance reward bias
- **KL estimator**: Schulman k3⁸ unbiased estimator `exp(Δ) − Δ − 1`, applied per-turn with independent `a1_kl_coeff`, `a2_kl_coeff`, `fb_kl_coeff`¹

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

Two parallel training runs over the LIVR-v2 9-task dataset (1000 samples per task,
single composite image per sample):

| Job | YAML | DAPO | Notes |
|---|---|---|---|
| `qwen-grpo-livr-v2-9k-curriculum` | `k8s/job-qwen-grpo-livr-v2-9k-curriculum.yaml` | Yes (K=16, dynamic sampling + clip-higher) | Curriculum-filtered subset (drops trivial + brick-wall difficulties) |
| `qwen-grpo-livr-v2-9k-no-dapo-k8` | `k8s/job-qwen-grpo-livr-v2-9k-curriculum-no-dapo-k8.yaml` | No (K=8, vanilla GRPO) | Full 9-task dataset, ablation against DAPO |
| `qwen-grpo-livr-v2-9k-baseline-a1` | `k8s/job-qwen-grpo-livr-v2-9k-baseline-a1.yaml` | No (K=16, single-turn A1 only) | Single-turn baseline (`--single_turn_a1`) — strips F1+A2, trains GRPO on A1 with `0.9 * a1_correctness_01 + 0.1 * a1_format_01` (range [0,1]) and normal β=0.001 KL. Used to isolate algorithm bugs from multi-turn / two-reward composition issues. Branch: `grpo-baseline`. |

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

### Think/Answer Tags (`--use_think_answer_tags`)
- A1/A2 user message appends a tag-format instruction
- Expected output: `<think>reasoning</think><answer>(A)</answer>`
- F1 (critic) is freeform — no `<think>/<answer>` tags expected, with a tag-leak
  penalty (`compute_f1_tag_penalty` in `composition.py`) when leakage occurs
- Tag format reward: +0.5 (valid tags + valid inner), -0.5 (tags but bad inner), -1.0 (no tags)

### MCQ-Aware Reward Asymmetry
Deterministic types (MCQ/YesNo/Numeric) use different no-regression and downstream values than open-ended:
- **No-regression**: RR=+1, RW=-2, WR=+2.35, WW=-0.5 (WR=2.35 is the exact compensation that ties response-head RR and WR after the a1_correctness term, removing the sandbagging gradient bias)
- **Downstream**: RR=+3, RW=-1.5, WR=+3, WW=-1 (RR raised to match WR so feedback head is also tied — exact tie on combined reward between RR and WR)
- **Feedback format**: De-coupled from calibration — pure word count only (0/<3/-2, 3-6/-1, >6/0)

### Key Training Config (curriculum DAPO)
- K=16 samples, batch=2, grad_acc=2 (effective batch=16), lr=1e-5, dr_grpo loss
- DAPO: dynamic sampling (drops zero-variance K-groups) + clip-higher
- LoRA r=64 alpha=128, max_completion=256, feedback_temp=0.7, a2_temp=1.0
- vLLM colocate with sleep mode, gpu_mem=0.50
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
├── experiments.md                # Experiment log (v1 → v9b)
│
├── src/vlm_grpo/                 # Installable package
│   ├── config.py                 # Dataclasses: RolloutConfig, ResponseRewardWeights, FeedbackRewardWeights
│   ├── critic_grpo.py            # SelfReflectionGRPOTrainer: GRPO loop, per-turn KL, policy update
│   ├── data.py                   # Dataset loading + answer_type detection
│   ├── prompts.py                # A1 / F1 / A2 prompt builders + system prompts
│   ├── rollout.py                # HF generate() rollout engine (3 turns per sample)
│   ├── vllm_rollout.py           # vLLM rollout engine with sleep-mode for GPU sharing
│   ├── trajectory.py             # Answer extraction, tag parsing, MCQ letter normalization
│   ├── utils.py                  # Seeding, env setup, normalized edit distance
│   └── rewards/
│       ├── composition.py        # Combines all rewards into Response + Feedback breakdowns
│       ├── correctness.py        # A2 correctness reward (binary / continuous)
│       ├── deterministic.py      # MCQ / YesNo / numeric answer matching
│       ├── feedback.py           # F1 calibration + downstream-aware reward
│       ├── judge_llm.py          # Optional LLM judge (enabled via VLM_USE_LLM_JUDGE=1)
│       ├── stability.py          # No-regression + minimal-edit rewards
│       └── verifier.py           # Top-level verify_answer dispatcher (deterministic + judge)
│
├── tests/                        # pytest suite
│   ├── test_correctness.py          # A2 correctness reward
│   ├── test_deterministic.py        # MCQ / YesNo / numeric matchers
│   ├── test_difficulty_buckets.py   # Curriculum difficulty bucketing
│   ├── test_dynamic_sampling.py     # DAPO dynamic-sampling K-group filter
│   ├── test_feedback.py             # Calibration + downstream
│   ├── test_kl_term.py              # Per-turn Schulman k3 KL
│   ├── test_rollout.py              # RolloutConfig + batch rollout
│   ├── test_stability.py            # No-regression + minimal-edit
│   ├── test_trajectory.py           # Tag parsing + answer extraction
│   ├── test_two_traj_composition.py # End-to-end response + feedback composition
│   ├── test_verification.py         # Verifier dispatcher (deterministic + judge)
│   └── test_verifier.py             # Top-level verify_answer
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
    ├── ─── Active training & data ───
    ├── job-qwen-grpo-livr-v2-9k-curriculum.yaml          # Curriculum DAPO training
    ├── job-qwen-grpo-livr-v2-9k-curriculum-no-dapo-k8.yaml  # No-DAPO ablation training
    ├── job-build-livr-v2-sources.yaml                    # Download LIVR source datasets
    ├── job-build-livr-v2.yaml                            # Build composite LIVR-v2 dataset
    ├── job-profile-livr-v2-9k-difficulty.yaml            # Difficulty profiler for curriculum
    │
    ├── ─── Active eval (BLINK PatternA over composite TSVs) ───
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
