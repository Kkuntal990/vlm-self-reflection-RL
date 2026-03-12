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


# =============================================================================
# Two-Trajectory GRPO Configuration
# =============================================================================


@dataclass
class CriticRewardWeights:
    """Weights for critic reward composition.

    reward = w_downstream * R_downstream
           + w_calibration * R_calibration
           + w_format * R_format

    Attributes:
        w_downstream: Weight for downstream-aware reward (dominant)
        w_calibration: Weight for feedback calibration
        w_format: Weight for format compliance
    """

    w_downstream: float = 2.0
    w_calibration: float = 1.0
    w_format: float = 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_list(self) -> list[float]:
        """Return weights as ordered list: [format, calibration, downstream]."""
        return [self.w_format, self.w_calibration, self.w_downstream]


@dataclass
class RefinerRewardWeights:
    """Weights for refiner reward composition.

    reward = w_correctness * R_correctness
           + w_no_regression * R_no_regression
           + w_minimal_edit * R_minimal_edit
           + w_format * R_format

    Attributes:
        w_correctness: Weight for A2 correctness
        w_no_regression: Weight for no-regression penalty (dominant)
        w_minimal_edit: Weight for minimal edit reward
        w_format: Weight for format compliance
    """

    w_correctness: float = 1.0
    w_no_regression: float = 2.0
    w_minimal_edit: float = 0.3
    w_format: float = 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_list(self) -> list[float]:
        """Return weights as ordered list: [format, correctness, no_regression, minimal_edit]."""
        return [
            self.w_format,
            self.w_correctness,
            self.w_no_regression,
            self.w_minimal_edit,
        ]


@dataclass
class RolloutConfig:
    """Configuration for K-sample rollout.

    Attributes:
        k_samples: Number of F1/A2 completions per sample
        max_completion_length: Maximum tokens per generation
        temperature: Sampling temperature for F1 generation
        top_p: Top-p sampling for F1 generation
        a2_temperature: Temperature for A2 generation (0.0 = greedy)
        batch_size: Samples to process in parallel during rollout
    """

    k_samples: int = 4
    max_completion_length: int = 512
    a1_max_completion_length: int = 200
    f1_max_completion_length: int = 512
    a2_max_completion_length: int = 200
    temperature: float = 0.7
    feedback_temperature: float = 0.9
    top_p: float = 0.9
    a2_temperature: float = 0.0
    batch_size: int = 8

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping during training.

    Attributes:
        patience: Validation checks without improvement before stopping
        metric: Metric to monitor (e.g., "val/rw_rate")
        mode: Whether lower ("min") or higher ("max") is better
        min_delta: Minimum change to qualify as improvement
    """

    patience: int = 5
    metric: str = "val/rw_rate"
    mode: str = "min"
    min_delta: float = 0.01

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class TwoTrajectoryConfig:
    """Top-level configuration for two-trajectory GRPO training.

    Attributes:
        model_id: HuggingFace model identifier or local checkpoint path
        dataset_path: Path to precomputed JSONL dataset
        val_dataset_path: Path to validation JSONL dataset
        image_base_dir: Base directory for resolving relative image paths
        output_dir: Directory for checkpoints and logs
        phase: Training phase ("rw_first" or "full")
        rollout: Rollout configuration
        critic_weights: Critic reward weights
        refiner_weights: Refiner reward weights
        learning_rate: Training learning rate
        per_device_train_batch_size: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        num_train_epochs: Number of training epochs
        max_samples: Maximum training samples (0 = all)
        use_peft: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_target_modules: Target modules for LoRA
        kl_coeff: KL divergence coefficient for GRPO
        clip_range: Policy ratio clipping range
        early_stopping: Early stopping configuration
        sanity_check_samples: Samples for sanity check mode (0=disabled)
        logging_steps: Steps between logging
        save_steps: Steps between checkpoint saves
        val_check_interval: Steps between validation
        seed: Random seed
    """

    model_id: str = "llava-hf/llava-1.5-7b-hf"
    dataset_path: str = ""
    val_dataset_path: str = ""
    image_base_dir: str = "/outputs/image_base"
    output_dir: str = "./outputs/grpo_two_traj"
    phase: str = "rw_first"
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    critic_weights: CriticRewardWeights = field(default_factory=CriticRewardWeights)
    refiner_weights: RefinerRewardWeights = field(default_factory=RefinerRewardWeights)
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    max_samples: int = 0
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    kl_coeff: float = 0.05
    clip_range: float = 0.2
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    sanity_check_samples: int = 0
    logging_steps: int = 10
    save_steps: int = 500
    val_check_interval: int = 500
    seed: int = 42

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# Single-Trajectory GRPO Configuration (backward-compatible)
# =============================================================================


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


# =============================================================================
# Full Self-Reflection GRPO Configuration
# =============================================================================


@dataclass
class ResponseRewardWeights:
    """Weights for response reward (applied to A1 + A2 log-probs).

    reward_resp = w_a1_correctness * R_a1_correct
               + w_a2_correctness * R_a2_correct
               + w_no_regression * R_no_regression
               + w_a2_format * R_a2_format
               + w_minimal_edit * R_minimal_edit

    Attributes:
        w_a1_correctness: Weight for A1 correctness
        w_a2_correctness: Weight for A2 correctness
        w_no_regression: Weight for no-regression penalty (dominant)
        w_a2_format: Weight for A2 format compliance
        w_minimal_edit: Weight for minimal edit reward
    """

    w_a1_correctness: float = 0.5
    w_a2_correctness: float = 1.0
    w_no_regression: float = 2.0
    w_a2_format: float = 0.5
    w_minimal_edit: float = 0.3

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class FeedbackRewardWeights:
    """Weights for feedback reward (applied to F1 log-prob).

    reward_fb = w_downstream * R_downstream
              + w_calibration * R_calibration
              + w_format * R_format

    Attributes:
        w_downstream: Weight for downstream-aware reward (dominant)
        w_calibration: Weight for feedback calibration
        w_format: Weight for format compliance
    """

    w_downstream: float = 2.0
    w_calibration: float = 1.0
    w_format: float = 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SelfReflectionConfig:
    """Top-level configuration for full self-reflection GRPO training.

    Single model generates A1 -> F1 -> A2, trained with two separate
    GRPO updates: one for response quality (A1+A2), one for feedback
    quality (F1). Both updates share the same LoRA adapter.

    Attributes:
        model_id: HuggingFace model identifier or local checkpoint path
        dataset_path: Path to JSONL dataset
        val_dataset_path: Path to validation JSONL dataset
        image_base_dir: Base directory for resolving relative image paths
        output_dir: Directory for checkpoints and logs
        rollout: Rollout configuration (k_samples, temperatures, etc.)
        response_weights: Response reward weights (for A1+A2 GRPO update)
        feedback_weights: Feedback reward weights (for F1 GRPO update)
        learning_rate: Training learning rate
        per_device_train_batch_size: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        num_train_epochs: Number of training epochs
        max_samples: Maximum training samples (0 = all)
        use_peft: Whether to use LoRA
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_target_modules: Target modules for LoRA
        kl_coeff: KL divergence coefficient for GRPO
        clip_range: Policy ratio clipping range
        early_stopping: Early stopping configuration
        sanity_check_samples: Samples for sanity check mode (0=disabled)
        logging_steps: Steps between logging
        save_steps: Steps between checkpoint saves
        val_check_interval: Steps between validation
        seed: Random seed
    """

    model_id: str = "llava-hf/llava-1.5-7b-hf"
    dataset_path: str = ""
    val_dataset_path: str = ""
    image_base_dir: str = "/outputs/image_base"
    output_dir: str = "./outputs/grpo_self_reflection"
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    response_weights: ResponseRewardWeights = field(default_factory=ResponseRewardWeights)
    feedback_weights: FeedbackRewardWeights = field(default_factory=FeedbackRewardWeights)
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    max_samples: int = 0
    use_peft: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    kl_coeff: float = 0.05
    clip_range: float = 0.2
    num_inner_epochs: int = 4
    inner_mini_batch_size: int = 8
    debug: bool = False
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    sanity_check_samples: int = 0
    logging_steps: int = 10
    save_steps: int = 500
    val_check_interval: int = 500
    seed: int = 42

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
