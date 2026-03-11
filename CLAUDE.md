# VLM Self-Reflection GRPO Training Pipeline

## Overview

Multi-turn GRPO training pipeline for VLM self-reflection, using a custom SelfReflectionGRPOTrainer with two-reward design.

**Model**: LLaVA-1.5 SFT checkpoint (from FIRE behavior cloning)
**Framework**: Custom GRPO trainer + vLLM
**Approach**: Multi-turn A1 → F1 → A2 flow with separate response and feedback rewards

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
# Full training (2 GPUs)
accelerate launch --num_processes=2 train_self_reflection.py \
    --model_id /outputs/llava-7b-fire-full-sft-checkpoint \
    --dataset_path /outputs/fire_preprocessed_v3/dataset.jsonl \
    --output_dir /outputs/grpo_self_reflection_v1
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
