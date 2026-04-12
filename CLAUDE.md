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
- `no_regression`: Did A2 maintain or improve on A1? Heaviest penalty for R→W (-3), highest reward for W→R (+2)
- `a2_format`: Penalty-only — is A2 in the expected format? (0 / -1)
- `minimal_edit`: Only when both A1 and A2 are correct — rewards keeping the answer stable (0 to +1)

**Feedback Reward** = w_down * R_down + w_cal * R_cal + w_fmt * R_fmt

- `downstream`: Did F1 lead to a correct A2? 4 discrete values based on A1→A2 transition (WR:+2, RR:+1, WW:-1, RW:-2)
- `calibration`: Does F1 correctly assess A1's correctness? Keyword-based, 7 discrete values (-1 to +1). Key variance-breaker — without it, downstream alone causes 50-75% zero-variance K-groups
- `fb_format`: Three-tier — empty/short (-2), vague (-1), substantive (+1)

#### Reward Components

**Response Reward:**

| Component | Raw Range | Logic |
|-----------|-----------|-------|
| a1_correctness | {-1, +1} | Binary correct/wrong |
| a2_correctness | [-1, +1] | Binary for MCQ/YesNo/Numeric; continuous for counting/open |
| no_regression | {-3, 0, +1, +2} | RW:-3, WW:0, RR:+1, WR:+2 |
| a2_format | {-1, 0} | Penalty-only (0=valid, -1=invalid) |
| minimal_edit | [0, +1] | `max(1 - 0.5*edit_dist, 0)`, only when both A1 and A2 correct |

**Feedback Reward:**

| Component | Raw Range | Logic |
|-----------|-----------|-------|
| downstream | {-2, -1, +1, +2} | RW:-2, WW:-1, RR:+1, WR:+2; 0 if empty feedback |
| calibration | [-1, +1] | 7 discrete values from keyword matching (positive/negative/mixed/doubt/neutral) |
| fb_format | {-2, -1, +1} | empty/short:-2, vague:-1, substantive:+1 |

### Prompt Mismatch Warning
The balanced_70k dataset's `messages` contain a system prompt with `Thought: [reasoning] / Answer: [final answer]` format. This is NOT the prompt used during GRPO training. GRPO hardcodes its own prompts in `prompts.py`. If you switch to using the dataset's messages, the prompt format will change.

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
