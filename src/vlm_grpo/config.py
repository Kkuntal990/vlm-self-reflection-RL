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
    reward_shaping_alpha: float = 0.0
    response_alpha: float = -1.0  # -1 means "use reward_shaping_alpha"
    feedback_alpha: float = -1.0  # -1 means "use reward_shaping_alpha"
    # Baseline mode: skip F1 and A2 generation/loss entirely. Train GRPO on
    # A1 alone with a single 2-component [0,1] reward. Used to isolate
    # algorithm bugs from multi-turn / two-reward composition issues.
    single_turn_a1: bool = False
    # Multi-turn rescaled-reward mode: per-component [0, 1] normalization of
    # the multi-turn response + feedback reward composition. Equalizes
    # per-unit-weight gradient magnitude across components and produces
    # strictly non-negative resp_reward / fb_reward / total_reward. See
    # ``compute_response_reward_breakdown_01`` and
    # ``compute_feedback_reward_breakdown_01`` in
    # ``src/vlm_grpo/rewards/composition.py``.
    use_rescaled_rewards: bool = False
    # PAG-faithful arm (arXiv:2506.10406):
    #   - ``use_pag_segment_rewards`` switches to the binary {0, 1} per-segment
    #     reward composers in ``rewards/composition.py``. r_a1 and r_a2 are
    #     emitted as separate scalars; A2 carries the shaping bonus
    #     ``α·(R(A2)−R(A1))``. F1 reward = w_verification·R_verification_01 +
    #     w_format·R_fb_format. The trainer then drives the existing
    #     ``separate_turn_loss=True`` per-segment advantage path.
    #   - ``use_selective_revision`` activates the F1-verdict gate in the
    #     rollout: when F1's ``\boxed{}`` extraction is ``CORRECT``, A2 is
    #     emitted as an empty completion and excluded from the A2 K-group
    #     baseline. ``WRONG`` / missing verdict → A2 generated as today.
    #   - ``pag_shaping_alpha`` is α in b_y(ŷ_2)=α·(R(A2)−R(A1)). PAG sets 1.0.
    # All three are off by default — legacy paths are bit-for-bit unchanged.
    use_pag_segment_rewards: bool = False
    use_selective_revision: bool = False
    pag_shaping_alpha: float = 1.0

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
               + w_wr_bonus       * R_wr_bonus

    Weights must sum to 1.0 (convex combination). A `__post_init__`
    warning fires otherwise.

    Attributes:
        w_a1_correctness: Weight for A1 correctness  (0.9 × turn_weight)
        w_a1_format:      Weight for A1 format       (0.1 × turn_weight)
        w_a2_correctness: Weight for A2 correctness  (0.9 × turn_weight)
        w_a2_format:      Weight for A2 format       (0.1 × turn_weight)
        w_wr_bonus:       Weight for an additive bonus that fires when
            A1 is wrong AND A2 is right (the WR quadrant). The component
            is a Bernoulli {0, 1} indicator, so the contribution to
            total reward is either 0 or ``w_wr_bonus``. Designed as a
            "promote-WR without penalising-RW" knob that does NOT push
            the model to actively degrade A1 (Option 5 in the literature
            menu vs SCoRe / ReST-MCTS shaping). Defaults to 0.0 so
            existing experiments are unaffected.
    """

    w_a1_correctness: float = 0.45
    w_a1_format: float = 0.05
    w_a2_correctness: float = 0.45
    w_a2_format: float = 0.05
    w_wr_bonus: float = 0.0

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
class BaselineA1RewardWeights:
    """Weights for single-turn A1 baseline reward composition.

    reward = w_a1_correctness * R_a1_correct_01
           + w_a1_format      * R_a1_format_01

    Both components live in [0, 1], so the convex combination is in [0, 1].
    Default 0.9 / 0.1 makes correctness the dominant signal while keeping a
    small format-anchor bonus to nudge the model to follow the
    <think>/<answer> tag structure used by the eval pipeline.

    Used only when ``RolloutConfig.single_turn_a1=True``.

    Attributes:
        w_a1_correctness: Weight for binary {0,1} A1 correctness reward.
        w_a1_format: Weight for binary {0,1} A1 format-compliance reward.
    """

    w_a1_correctness: float = 0.9
    w_a1_format: float = 0.1

    def __post_init__(self) -> None:
        _validate_weight_sum("BaselineA1RewardWeights", self.to_dict())

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class AdapterSpec:
    """Specification for one LoRA adapter in the multi-adapter routing config.

    Attributes:
        name: Adapter handle used by PEFT's set_adapter / save_pretrained.
            Must be unique within the routing config.
        trainable: Whether this adapter receives gradients during training.
            When False, it is loaded with requires_grad=False — used for a
            frozen reference adapter (e.g. a frozen-A1 expert).
        init_from_checkpoint: Filesystem path to a saved PEFT checkpoint
            directory whose adapter_model.safetensors should be loaded as
            this adapter's initial weights. When None, the adapter is added
            from scratch using the run's LoraConfig (random LoRA init).
        warm_start_from_adapter: Name of another adapter (already loaded)
            whose weights should be copied into this one at init. Use when
            you want two adapters to share a starting point but diverge
            during training. Mutually exclusive with init_from_checkpoint.
    """

    name: str
    trainable: bool = True
    init_from_checkpoint: str | None = None
    warm_start_from_adapter: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AdapterRoutingConfig:
    """Multi-adapter routing for per-turn LoRA selection.

    The default empty-adapters list means single-adapter mode — the trainer
    uses one adapter named "default" for all turns, matching the existing
    behavior. To enable multi-adapter mode, supply at least one adapter spec
    and a turn→adapter mapping.

    Examples:
        Single adapter (default):
            AdapterRoutingConfig()
        Response/feedback split (this config's primary use case):
            AdapterRoutingConfig(
                turns={"a1": "response", "f1": "feedback", "a2": "response"},
                adapters=[
                    AdapterSpec(name="response", trainable=True),
                    AdapterSpec(name="feedback", trainable=True),
                ],
            )
        Frozen-A1 + trainable F1+A2 (legacy two-adapter pattern):
            AdapterRoutingConfig(
                turns={"a1": "a1_expert", "f1": "f1_a2_expert", "a2": "f1_a2_expert"},
                adapters=[
                    AdapterSpec(name="a1_expert", trainable=False,
                                init_from_checkpoint="/path/to/a1/ckpt"),
                    AdapterSpec(name="f1_a2_expert", trainable=True,
                                warm_start_from_adapter="a1_expert"),
                ],
            )

    Attributes:
        turns: Mapping of turn name ("a1" / "f1" / "a2") to adapter name.
            Every turn must map to an adapter listed in ``adapters``. If
            the dict is empty (single-adapter default), all turns implicitly
            route to "default".
        adapters: Ordered list of adapter specs. The first adapter is the
            one PEFT wraps the base model with; subsequent adapters are
            added via add_adapter. Empty list = single-adapter mode.
        frozen_lora_patterns: List of substring patterns that, when matched
            against a LoRA parameter's full name, force its ``requires_grad``
            to ``False`` across EVERY adapter — overriding the per-adapter
            ``trainable`` flag. Use this to keep specific module families
            from training even though their hosting adapter is otherwise
            trainable, e.g. ``["visual"]`` freezes vision-encoder and merger
            LoRA on both response and feedback while leaving language-decoder
            LoRA trainable. Matching is anchored on the LoRA-tensor segments
            ``.lora_A.<adapter>.<pattern>`` / ``.lora_B.<adapter>.<pattern>``
            implicitly — the pattern itself is a plain substring check
            against the full parameter name, but only LoRA params (those
            already containing ``.lora_A.<adapter>.`` or ``.lora_B.<adapter>.``)
            are considered. Empty list (default) = no extra freezing.
    """

    turns: dict[str, str] = field(default_factory=dict)
    adapters: list[AdapterSpec] = field(default_factory=list)
    frozen_lora_patterns: list[str] = field(default_factory=list)

    @property
    def enabled(self) -> bool:
        """True when multi-adapter routing is active (>=1 spec provided)."""
        return len(self.adapters) > 0

    def adapter_for_turn(self, turn: str) -> str:
        """Resolve the adapter name for a given turn.

        Falls back to "default" when routing is disabled.
        """
        if not self.enabled:
            return "default"
        if turn not in self.turns:
            raise KeyError(
                f"Turn {turn!r} not in adapter_routing.turns (have: {sorted(self.turns)})"
            )
        return self.turns[turn]

    def trainable_adapter_names(self) -> list[str]:
        """List adapter names where trainable=True."""
        return [a.name for a in self.adapters if a.trainable]

    def validate(self) -> None:
        """Sanity-check routing wiring; raises ValueError on misconfiguration.

        Catches:
          - duplicate adapter names
          - turn→adapter references to unknown adapters
          - missing routing for any of the three turns when routing is enabled
          - mutually exclusive init_from_checkpoint + warm_start_from_adapter
          - warm_start_from_adapter referencing an adapter that appears
            after this one (must be loaded first)
        """
        if not self.enabled:
            return
        seen: set[str] = set()
        for spec in self.adapters:
            if spec.name in seen:
                raise ValueError(f"Duplicate adapter name in routing: {spec.name}")
            seen.add(spec.name)
            if spec.init_from_checkpoint and spec.warm_start_from_adapter:
                raise ValueError(
                    f"Adapter {spec.name}: init_from_checkpoint and "
                    "warm_start_from_adapter are mutually exclusive"
                )
            if spec.warm_start_from_adapter and spec.warm_start_from_adapter not in seen - {
                spec.name
            }:
                raise ValueError(
                    f"Adapter {spec.name} warm-starts from "
                    f"{spec.warm_start_from_adapter!r}, which is not loaded "
                    "before it. Reorder ``adapters`` so the source comes first."
                )
        required_turns = {"a1", "f1", "a2"}
        missing = required_turns - set(self.turns)
        if missing:
            raise ValueError(f"adapter_routing.turns missing entries for: {sorted(missing)}")
        unknown_targets = set(self.turns.values()) - seen
        if unknown_targets:
            raise ValueError(
                f"adapter_routing.turns references unknown adapters: "
                f"{sorted(unknown_targets)}. Defined adapters: {sorted(seen)}"
            )
        if not self.trainable_adapter_names():
            raise ValueError(
                "adapter_routing has no trainable adapter — at least one "
                "spec must set trainable=True."
            )

    def to_dict(self) -> dict:
        return {
            "turns": dict(self.turns),
            "adapters": [a.to_dict() for a in self.adapters],
            "frozen_lora_patterns": list(self.frozen_lora_patterns),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AdapterRoutingConfig":
        """Build from a plain dict (typically parsed from JSON).

        Args:
            data: ``{"turns": {...}, "adapters": [{"name": ..., ...}, ...],
                "frozen_lora_patterns": [...]}``. The last key is optional.

        Returns:
            Validated AdapterRoutingConfig. Empty / missing data → disabled.
        """
        if not data:
            return cls()
        adapters_raw = data.get("adapters", []) or []
        adapters = [AdapterSpec(**spec) for spec in adapters_raw]
        turns = dict(data.get("turns", {}) or {})
        frozen = list(data.get("frozen_lora_patterns", []) or [])
        cfg = cls(turns=turns, adapters=adapters, frozen_lora_patterns=frozen)
        cfg.validate()
        return cfg


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
    baseline_weights: BaselineA1RewardWeights = field(default_factory=BaselineA1RewardWeights)
    # Multi-adapter routing. Default (empty) = single-adapter mode using the
    # PEFT "default" adapter for all turns. Populate adapters + turns to
    # enable per-turn LoRA selection (e.g. separate response/feedback adapters).
    adapter_routing: AdapterRoutingConfig = field(default_factory=AdapterRoutingConfig)
    learning_rate: float = 1e-5
    # Linear LR warmup: linearly ramps optimizer LR from 0 -> learning_rate
    # over the first N global_steps, then holds constant. 0 = no warmup
    # (constant LR from step 0, behavior unchanged).
    lr_warmup_steps: int = 0
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
    # Name of a frozen LoRA adapter on the POLICY model used as the KL
    # reference distribution. When set, the trainer swaps to this adapter
    # before each KL ref forward and restores the previous adapter after.
    # The base weights are shared with the policy, so the per-rank memory
    # cost is one extra LoRA (~50 MB) instead of a full 7B base (~15 GB).
    # Loaded from train_self_reflection.py via PEFT's ``load_adapter``.
    # Mutually exclusive with the legacy ``ref_model`` argument passed to
    # ``SelfReflectionGRPOTrainer.__init__``; the trainer asserts at most
    # one is set.
    kl_ref_adapter_name: str | None = None
    separate_turn_loss: bool = False
    # DAPO Dynamic Sampling (arXiv:2503.14476 §3.2): drop K-groups whose
    # rewards are zero-variance (advantage=0, gradient=0). The policy update
    # then runs on the smaller effective batch so every gradient step is
    # non-degenerate.
    use_dynamic_sampling: bool = False
    # GDPO per-component K-group advantage normalization (Liu 2026,
    # arXiv:2601.05242). When True, _compute_group_advantages normalizes
    # each reward component within its K-group separately, then takes a
    # weighted sum, then batch-renormalizes — equalizing per-component
    # gradient contribution. When False, falls back to standard GRPO
    # group-normalize-then-sum behavior (bit-for-bit unchanged).
    use_gdpo_normalization: bool = False
    reward_shaping_alpha: float = 0.0
    freeze_a1_steps: int = 0
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
