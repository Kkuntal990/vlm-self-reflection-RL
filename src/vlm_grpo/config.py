#!/usr/bin/env python3
"""
Configuration dataclasses for self-reflection GRPO training.

Centralizes all configuration for reward weights, answer type detection,
rollout parameters, and training hyperparameters.

Usage:
    from vlm_grpo.config import SelfReflectionConfig, ResponseRewardWeights

    config = SelfReflectionConfig(
        model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        dataset_path="/outputs/grpo_data/balanced_70k.jsonl",
    )
"""

from dataclasses import asdict, dataclass, field


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
    w_format: float = 0.15
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
    temperature: float = 1.0
    feedback_temperature: float = 1.0
    top_p: float = 0.9
    a2_temperature: float = 1.0
    batch_size: int = 8
    use_think_answer_tags: bool = False
    use_improvement_reward: bool = False
    reward_shaping_alpha: float = 0.0
    use_binary_verification: bool = False

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

    w_a1_correctness: float = 1.0
    w_a2_correctness: float = 1.0
    w_no_regression: float = 2.0
    w_a2_format: float = 0.15
    w_minimal_edit: float = 0.0

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
        w_downstream: Weight for downstream-aware reward (dominant).
            Transition-shaped: WR=+3, RR=+1, RW=-1.5, WW=-1.
        w_calibration: Weight for keyword-based calibration. Disabled by
            default (0.0) — literature consensus is outcome-based rewards
            only for feedback. Kept in code for experimentation.
        w_format: Weight for format compliance (word count penalty)
        w_tag_penalty: Weight for F1 tag leakage penalty
    """

    w_downstream: float = 2.0
    w_calibration: float = 0.0
    w_format: float = 0.15
    w_tag_penalty: float = 0.5
    w_verification_accuracy: float = 0.0

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
        model_type: Model family ("auto", "llava", "qwen2vl")
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
        kl_coeff: KL divergence coefficient for GRPO (0.0 disables KL)
        a1_kl_coeff: KL multiplier for A1 turn (SCoRe-style anchor, relative to kl_coeff)
        a2_kl_coeff: KL multiplier for A2 turn (relative to kl_coeff)
        fb_kl_coeff: KL multiplier for F1 turn (relative to kl_coeff)
        separate_turn_loss: If True, compute separate advantages for A1 vs A2
        clip_range: Policy ratio clipping range
        loss_type: GRPO loss variant ("grpo" or "dr_grpo")
        freeze_vision_tower: Whether to freeze the vision encoder
        max_pixels: Maximum total pixels per image (Qwen2.5-VL dynamic resolution)
        min_pixels: Minimum total pixels per image (Qwen2.5-VL dynamic resolution)
        early_stopping: Early stopping configuration
        sanity_check_samples: Samples for sanity check mode (0=disabled)
        logging_steps: Steps between logging
        save_steps: Steps between checkpoint saves
        val_check_interval: Steps between validation
        seed: Random seed
    """

    model_id: str = "llava-hf/llava-1.5-7b-hf"
    model_type: str = "auto"
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
    a1_kl_coeff: float = 1.0
    a2_kl_coeff: float = 1.0
    fb_kl_coeff: float = 1.0
    separate_turn_loss: bool = False
    use_ssr: bool = False
    ssr_buffer_size: int = 64
    ssr_alpha: float = 1.0
    use_improvement_reward: bool = False
    reward_shaping_alpha: float = 0.0
    freeze_a1_steps: int = 0
    clip_range: float = 0.2
    loss_type: str = "grpo"
    freeze_vision_tower: bool = False
    max_pixels: int = 401408
    min_pixels: int = 200704
    num_inner_epochs: int = 4
    inner_mini_batch_size: int = 4
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
