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

import logging
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

# Tolerance for weight-sum == 1.0 validation. Floating-point env-var parsing
# commonly lands off by a few 1e-9; 1e-6 is strict enough to catch real mistakes.
_WEIGHT_SUM_TOL = 1e-6


def _validate_weight_sum(name: str, weights: dict[str, float]) -> None:
    """Warn if the weights don't sum to 1.0.

    Reward-weight sets in this pipeline are expected to be convex
    combinations so the aggregate reward lives on a comparable scale
    across experiments. This emits a warning on mismatch rather than
    a hard error, so deliberate ablations remain possible.
    """
    total = sum(weights.values())
    if abs(total - 1.0) > _WEIGHT_SUM_TOL:
        logger.warning(
            "%s weights sum to %.4f, expected 1.0. Components: %s",
            name,
            total,
            weights,
        )


@dataclass
class RewardWeights:
    """Weights for reward composition.

    reward = w_final * R_final_correct
           + w_format * R_format
           + w_rw * R_no_regression
           + w_fb * R_feedback_calibration

    Attributes:
        w_final: Weight for final answer correctness
        w_format: Weight for format compliance
        w_rw: Weight for no-regression penalty (highest by default)
        w_fb: Weight for feedback calibration
    """

    w_final: float = 1.0
    w_format: float = 0.15
    w_rw: float = 2.0
    w_fb: float = 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def to_list(self) -> list[float]:
        """Return weights as ordered list for TRL reward_weights param."""
        return [self.w_format, self.w_final, self.w_rw, self.w_fb]


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
    use_answer_tag_only: bool = False
    use_improvement_reward: bool = False
    reward_shaping_alpha: float = 0.0
    response_alpha: float = -1.0  # -1 means "use reward_shaping_alpha"
    feedback_alpha: float = -1.0  # -1 means "use reward_shaping_alpha"

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
               + w_a1_format       * R_a1_format
               + w_a2_correctness * R_a2_correct
               + w_a2_format       * R_a2_format
               + w_no_regression  * R_no_regression

    Per-turn split convention: each turn carries a 0.9·corr + 0.1·fmt
    sub-reward. With turn weight 0.3 each and no_regression 0.4, this
    gives the defaults below (sum = 1.0).

    Weights must sum to 1.0 (convex combination). A `__post_init__`
    warning fires otherwise. Raw α is applied inside R_no_regression,
    so the *weighted* reward is not bounded by 1.0 — only the linear
    combination is convex.

    Attributes:
        w_a1_correctness: Weight for A1 correctness  (0.9 × turn_weight)
        w_a1_format:      Weight for A1 format       (0.1 × turn_weight)
        w_a2_correctness: Weight for A2 correctness  (0.9 × turn_weight)
        w_a2_format:      Weight for A2 format       (0.1 × turn_weight)
        w_no_regression:  Weight for transition / shaped reward
    """

    w_a1_correctness: float = 0.27
    w_a1_format: float = 0.03
    w_a2_correctness: float = 0.27
    w_a2_format: float = 0.03
    w_no_regression: float = 0.40

    def __post_init__(self) -> None:
        _validate_weight_sum("ResponseRewardWeights", self.to_dict())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class FeedbackRewardWeights:
    """Weights for feedback reward (applied to F1 log-prob).

    reward_fb = w_downstream * R_downstream
              + w_verification_accuracy * R_verification
              + w_format * R_format

    Weights must sum to 1.0 (convex combination). A `__post_init__`
    warning fires otherwise. Raw α is applied inside R_downstream,
    so the *weighted* reward is not bounded by 1.0.

    Attributes:
        w_downstream: Weight for downstream-aware reward.
            Shaped: r_a2 + α·(r_a2 − r_a1). Gated on verdict calibration.
        w_verification_accuracy: Weight for F1 verdict accuracy
            (±1 based on whether F1 says CORRECT/INCORRECT and A1 is actually
            right/wrong).
        w_format: Weight for F1 format compliance (<think></think> +
            \\boxed{VERDICT} structure, ±1).
    """

    w_downstream: float = 0.45
    w_verification_accuracy: float = 0.45
    w_format: float = 0.1

    def __post_init__(self) -> None:
        _validate_weight_sum("FeedbackRewardWeights", self.to_dict())

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
    ssr_buffer_size: int = 256
    ssr_alpha: float = 1.0
    # DAPO Dynamic Sampling (arXiv:2503.14476 §3.2): drop K-groups whose
    # rewards are zero-variance (advantage=0, gradient=0). Independent of
    # use_ssr — when use_ssr=True the dropped slots are refilled from the
    # SSR buffer; when SSR is off the policy update simply runs on the
    # smaller effective batch (every gradient step is still non-degenerate).
    use_dynamic_sampling: bool = False
    use_improvement_reward: bool = False
    reward_shaping_alpha: float = 0.0
    freeze_a1_steps: int = 0
    freeze_a2_steps: int = 0
    # Path to a LoRA checkpoint to load as frozen reference adapter (Stage II).
    # When set, KL is computed against that checkpoint's distribution instead of
    # the base model. Empty string = use base model as ref (current behavior).
    ref_adapter_path: str = ""
    clip_range: float = 0.2
    # DAPO Clip-Higher (arXiv:2503.14476 §3.1): asymmetric PPO clipping.
    # When > 0, upper clip becomes (1 + clip_high) instead of (1 + clip_range),
    # giving positive-advantage tokens more headroom before clipping kicks in.
    # Recommended: clip_high=0.28 (paper) with clip_range=0.2 for the lower bound.
    # 0.0 disables and falls back to symmetric clip_range.
    clip_high: float = 0.0
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
