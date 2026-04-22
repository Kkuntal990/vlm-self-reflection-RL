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

### Two-Reward Design

GRPO uses two separate reward signals per trajectory — one for response quality (drives A1+A2 log-prob update), one for feedback quality (drives F1 log-prob update). Rewards are computed in `src/vlm_grpo/rewards/composition.py`.

**Response Reward** = w_a1 * R_a1 + w_a2 * R_a2 + w_noreg * R_noreg + w_fmt * R_fmt + w_edit * R_edit

- `a1_correctness`: Binary — is A1 correct? (+1 / -1)
- `a2_correctness`: Binary for MCQ/YesNo/Numeric (+1 / -1), continuous for counting/open (2*score - 1)
- `no_regression`: Did A2 maintain or improve on A1? MCQ-aware: deterministic types RW=-2/WR=+3, open-ended RW=-3/WR=+2
- `a2_format`: Without tags: penalty-only (0 / -1). With tags: +0.5 / -0.5 / -1.0
- `minimal_edit`: Only when both A1 and A2 are correct — rewards keeping the answer stable (0 to +1)

**Feedback Reward** = w_down * R_down + w_cal * R_cal + w_fmt * R_fmt

- `downstream`: Did F1 lead to a correct A2? MCQ-aware: deterministic RW=-1.5/WR=+3, open-ended RW=-2/WR=+2
- `calibration`: Does F1 correctly assess A1's correctness? Keyword-based, 7 discrete values (-1 to +1). Key variance-breaker — without it, downstream alone causes 50-75% zero-variance K-groups
- `fb_format`: De-coupled from calibration (pure word count): empty/<3 words=-2, 3-6 words=-1, >6 words=0

#### Reward Components

**Response Reward:**

| Component | Raw Range | Logic |
|-----------|-----------|-------|
| a1_correctness | {-1, +1} | Binary correct/wrong |
| a2_correctness | [-1, +1] | Binary for MCQ/YesNo/Numeric; continuous for counting/open |
| no_regression | {-3..+3} | Deterministic: RW:-2, WW:0, RR:+1, WR:+3. Open: RW:-3, WW:0, RR:+1, WR:+2 |
| a2_format | {-1..+0.5} | No tags: 0/-1. With tags: +0.5 (valid), -0.5 (bad inner), -1.0 (no tags) |
| minimal_edit | [0, +1] | `max(1 - 0.5*edit_dist, 0)`, only when both A1 and A2 correct |

**Feedback Reward:**

| Component | Raw Range | Logic |
|-----------|-----------|-------|
| downstream | {-2..+3} | Deterministic: RW:-1.5, WW:-1, RR:+1, WR:+3. Open: RW:-2, WW:-1, RR:+1, WR:+2 |
| calibration | [-1, +1] | 7 discrete values from keyword matching (positive/negative/mixed/doubt/neutral) |
| fb_format | {-2, -1, 0} | Pure word count (de-coupled from calibration): <3:-2, 3-6:-1, >6:0 |

### Prompt Mismatch Warning
The balanced_70k dataset's `messages` contain a system prompt with `Thought: [reasoning] / Answer: [final answer]` format. This is NOT the prompt used during GRPO training. GRPO hardcodes its own prompts in `prompts.py`. If you switch to using the dataset's messages, the prompt format will change.

## Current Experiment: LIVR 9K MCQ with Think/Answer Tags

**Job**: `qwen-grpo-livr-9k-think-tags` on pod `vlm-jupyter-eval2` (4x A100-80GB)
**K8s YAML**: `k8s/job-qwen-grpo-livr-9k.yaml`
**Branch**: `feature/vllm-deepspeed` (commit `2fc4dd2`)
**Started**: 2026-04-11, ~2250 samples/process, ~34s/sample

### Dataset: LIVR Perception MCQ (9K)
- **Path**: `/outputs/livr_data/livr_perception_mcq.jsonl`
- **Construction scripts**: `scripts/livr/build_*.py` (9 tasks, 1000 each)
- 9 tasks: Counting, Jigsaw, ObjLoc, VisCorr, ArtStyle, SemCorr, FunCorr, RelReflect, VisSim
- All MCQ with ground truth in `(A)` format (matches BLINK eval)
- `answer_type=mcq` only, `VLM_USE_LLM_JUDGE=0` (deterministic matching sufficient)

### Think/Answer Tags (`--use_think_answer_tags`)
- A1/A2 system prompt includes tag format instruction (`VL_ASSISTANT_SYSTEM_PROMPT_WITH_TAGS`)
- Expected output: `<think>reasoning</think><answer>(A)</answer>`
- F1 (critic) is freeform — no tags
- Tag format reward: +0.5 (valid tags + valid inner), -0.5 (tags but bad inner), -1.0 (no tags)
- `w_a2_format` auto-raised to 0.5 when tags enabled

### MCQ-Aware Reward Asymmetry
Deterministic types (MCQ/YesNo/Numeric) use different no-regression and downstream values than open-ended:
- **No-regression**: RR=+1, RW=-2, WR=+3, WW=0 (less punitive regression, more rewarding correction)
- **Downstream**: RR=+1, RW=-1.5, WR=+3, WW=-1
- **Feedback format**: De-coupled from calibration — pure word count only (0/<3/-2, 3-6/-1, >6/0)

### Key Training Config
- K=8 samples, batch=2, grad_acc=2 (effective batch=16), lr=1e-6, dr_grpo loss
- LoRA r=64 alpha=128, max_completion=256, feedback_temp=0.7, a2_temp=1.0
- vLLM colocate with sleep mode, gpu_mem=0.50
- save_steps=250 (in global_step, not samples — first checkpoint ~sample 500)

### F1 Tag Leakage Fix (v2)
- **Problem**: Think tags leak from A1/A2 into F1 (0.3% → 38% by step 500) despite critic prompt having no tag instructions
- Think-tagged F1 produces worse outcomes: R→R drops from 43.5% to 27.7%
- **Fix**: `compute_f1_tag_penalty()` in `composition.py` — returns -2.0 when F1 contains `<think>/<answer>` tags
- Weight `w_fb_tag_penalty=0.5` → effective -1.0 penalty per tagged F1
- Critic system prompt also strengthened with "Do NOT use XML tags"
- v2 job resumes from v1 checkpoint-250, writes to `/outputs/grpo_qwen_livr_v2/`

### Known Issues
- Model puts verbose text inside `<answer>` tags (e.g., `(B) full text` instead of `(B)`), causing a2_format=-0.5. Expected to improve with training.
- Relative reflectance task: model struggles consistently (COCO luminance proxy may be too hard)

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

## Key Files

- `train_self_reflection.py` - Main training entry point
- `src/vlm_grpo/critic_grpo.py` - SelfReflectionGRPOTrainer
- `src/vlm_grpo/rewards/composition.py` - Response + feedback reward composition
- `src/vlm_grpo/rewards/correctness.py` - A2 correctness reward
- `src/vlm_grpo/rewards/feedback.py` - Feedback calibration + downstream reward
- `src/vlm_grpo/rewards/stability.py` - No-regression + minimal edit rewards
- `src/vlm_grpo/rewards/verifier.py` - Deterministic answer verification
- `src/vlm_grpo/rewards/deterministic.py` - MCQ/YesNo/numeric answer matching
- `src/vlm_grpo/trajectory.py` - Answer extraction and normalization
- `src/vlm_grpo/prompts.py` - Prompt builders (A1, critic, refiner)
- `src/vlm_grpo/data.py` - Dataset loading
