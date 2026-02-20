#!/usr/bin/env python3
"""
Configuration dataclasses for GRPO RW training.

Centralizes all configuration for reward weights, answer type detection,
and training hyperparameters.

Usage:
    from vlm_grpo.config import GRPORWConfig, RewardWeights

    config = GRPORWConfig(
        model_id="llava-hf/llava-1.5-7b-hf",
        dataset_path="/outputs/grpo_data/answer1_correct_train.jsonl",
        reward_weights=RewardWeights(w_rw=2.0),
    )
"""

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class RewardWeights:
    """Weights for reward composition.

    reward = w_final * R_final_correct
           + w_format * R_format
           + w_rw * R_no_regression
           + w_edit * R_minimal_edit
           + w_fb * R_feedback_calibration

    Attributes:
        w_final: Weight for final answer correctness
        w_format: Weight for format compliance
        w_rw: Weight for no-regression penalty (highest by default)
        w_edit: Weight for minimal edit reward
        w_fb: Weight for feedback calibration
    """

    w_final: float = 1.0
    w_format: float = 0.5
    w_rw: float = 2.0
    w_edit: float = 0.3
    w_fb: float = 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_list(self) -> list[float]:
        """Return weights as ordered list for TRL reward_weights param."""
        return [self.w_format, self.w_final, self.w_rw, self.w_edit, self.w_fb]


@dataclass
class AnswerTypeConfig:
    """Configuration for answer type detection and extraction.

    Attributes:
        mcq_pattern: Regex for MCQ answer extraction (single letter A-F)
        yesno_pattern: Regex for yes/no answer extraction
        numeric_tolerance: Relative tolerance for numeric comparison
    """

    mcq_pattern: str = r"\(?([A-F])\)?"
    yesno_pattern: str = r"\b(yes|no)\b"
    numeric_tolerance: float = 0.01

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class GRPORWConfig:
    """Top-level configuration for GRPO RW training.

    Attributes:
        model_id: HuggingFace model identifier or local checkpoint path
        dataset_path: Path to precomputed Answer1-correct JSONL dataset
        val_dataset_path: Optional path to validation JSONL dataset
        image_base_dir: Base directory for resolving relative image paths
        output_dir: Directory for checkpoints and logs
        reward_weights: Reward component weights
        answer_type_config: Answer type detection config
        num_generations: Number of completions per prompt for GRPO
        max_completion_length: Maximum tokens in generated completion
        max_prompt_length: Maximum tokens in prompt (None for VLMs)
        learning_rate: Training learning rate
        per_device_train_batch_size: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        num_train_epochs: Number of training epochs
        max_samples: Maximum training samples (0 = all)
        use_vllm: Whether to use vLLM for generation
        vllm_mode: vLLM mode ("colocate" or "server")
        vllm_gpu_memory_utilization: GPU memory fraction for vLLM
        use_peft: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_target_modules: Target modules for LoRA
        sanity_check_samples: Number of samples for sanity check mode (0=disabled)
        val_check_interval: Steps between validation RW-rate checks
        logging_steps: Steps between logging
        save_steps: Steps between checkpoint saves
        seed: Random seed
    """

    model_id: str = "llava-hf/llava-1.5-7b-hf"
    dataset_path: str = ""
    val_dataset_path: str = ""
    image_base_dir: str = "/outputs/image_base"
    output_dir: str = "./outputs/grpo_rw"
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    answer_type_config: AnswerTypeConfig = field(default_factory=AnswerTypeConfig)
    num_generations: int = 4
    max_completion_length: int = 512
    max_prompt_length: Optional[int] = None
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    max_samples: int = 0
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.3
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    sanity_check_samples: int = 0
    val_check_interval: int = 500
    logging_steps: int = 10
    save_steps: int = 500
    seed: int = 42

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
