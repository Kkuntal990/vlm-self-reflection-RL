# VLM GRPO RW Training Pipeline

## Overview

GRPO training pipeline for reducing RW flips (Right-to-Wrong transitions) in VLM self-reflection, using TRL's GRPOTrainer with custom reward functions.

**Model**: LLaVA-1.5 SFT checkpoint (from FIRE behavior cloning)
**Framework**: TRL GRPOTrainer + vLLM colocate
**Approach**: Single-turn GRPO where model generates FEEDBACK + FINAL_ANSWER in one completion

## How to Run

Always use `uv` for running.

### Install

```bash
uv sync
uv sync --extra dev  # for testing
```

### Linting

```bash
ruff format src/ tests/ train_grpo_rw.py scripts/
ruff check src/ tests/ train_grpo_rw.py scripts/
ruff check src/ tests/ train_grpo_rw.py scripts/ --fix
```

### Tests

```bash
uv run pytest tests/ -v
```

### Training

```bash
# Sanity check (no GPU needed for reward logic)
uv run python train_grpo_rw.py \
    --dataset_path /outputs/grpo_data/answer1_correct_train.jsonl \
    --sanity_check_samples 50

# Full training (4 GPUs)
accelerate launch --num_processes=4 train_grpo_rw.py \
    --model_id /outputs/llava-1.5-sft-checkpoint \
    --dataset_path /outputs/grpo_data/answer1_correct_train.jsonl \
    --output_dir /outputs/grpo_rw_v1
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
2. **Reward function signatures**: Must match TRL's expected `reward_fn(completions, **kwargs) -> list[float]`.

## Key Files

- `train_grpo_rw.py` - Main training entry point
- `src/vlm_grpo/rewards/rw_reward.py` - Reward functions for TRL
- `src/vlm_grpo/trajectory.py` - Parse FEEDBACK/FINAL_ANSWER markers
- `src/vlm_grpo/rewards/deterministic.py` - MCQ/YesNo/numeric scoring
- `src/vlm_grpo/data.py` - Dataset loading for TRL GRPOTrainer
