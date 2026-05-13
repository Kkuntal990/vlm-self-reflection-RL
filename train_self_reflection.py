#!/usr/bin/env python3
"""Full self-reflection GRPO training for VLM.

Single model generates the full chain: A1 -> F1 -> A2.
Two separate GRPO updates per step:
  - Response update: advantage from response reward on log_prob(A1) + log_prob(A2)
  - Feedback update: advantage from feedback reward on log_prob(F1)

Both updates share the same LoRA adapter. The conversation flow matches
the inference script (self_reflective_inference_v2.py) exactly:
  A1: System=VL_ASSISTANT, User=[image]+question
  F1: System=CRITIC, Assistant=[image]+question (flipped), User=A1
  A2: System=VL_ASSISTANT, User=[image]+question, Asst=A1, User=F1 (raw)

Usage:
    # Sanity check (no GPU needed for reward logic)
    uv run python train_self_reflection.py \
        --dataset_path /outputs/grpo_data/train.jsonl \
        --sanity_check_samples 10

    # Full training (4 GPUs)
    accelerate launch --num_processes=4 train_self_reflection.py \
        --model_id /outputs/llava-1.5-sft-checkpoint \
        --dataset_path /outputs/grpo_data/train.jsonl \
        --output_dir /outputs/grpo_self_reflection_v1

Reference:
    - GRPO: https://arxiv.org/abs/2402.03300
    - SCoRe: https://arxiv.org/abs/2409.12917
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Full self-reflection GRPO training (A1->F1->A2, two-reward)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model_id",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="HuggingFace model ID or local checkpoint path",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "llava", "qwen2vl"],
        help="Model family: 'auto' (detect from model_id), 'llava', or 'qwen2vl'",
    )

    # Dataset
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to JSONL dataset",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default="",
        help="Path to validation JSONL",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.0,
        help="Fraction of training data to hold out for validation (e.g. 0.1). "
        "Ignored when --val_dataset_path is provided.",
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="/outputs/image_base",
        help="Base directory for resolving relative image paths",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum training samples (0 = all)",
    )
    parser.add_argument(
        "--sample_indices",
        type=str,
        default="",
        help="Comma-separated JSONL line indices to run (e.g. '0,5,12'). Overrides max_samples.",
    )

    # Training
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/grpo_self_reflection",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument("--k_samples", type=int, default=4, help="Trajectories per sample")
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--a1_max_completion_length", type=int, default=200)
    parser.add_argument("--f1_max_completion_length", type=int, default=512)
    parser.add_argument("--a2_max_completion_length", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7, help="A1 sampling temperature")
    parser.add_argument(
        "--feedback_temperature", type=float, default=0.9, help="F1 sampling temperature"
    )
    parser.add_argument(
        "--a2_temperature", type=float, default=0.3, help="A2 temperature (0=greedy)"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help=(
            "Linearly ramp LR from 0 to learning_rate over the first N global_steps, "
            "then hold constant. 0 = no warmup (constant LR from step 0, default)."
        ),
    )
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--kl_coeff", type=float, default=0.05)
    parser.add_argument("--a1_kl_coeff", type=float, default=1.0)
    parser.add_argument("--a2_kl_coeff", type=float, default=1.0)
    parser.add_argument("--fb_kl_coeff", type=float, default=1.0)
    parser.add_argument("--separate_turn_loss", action="store_true")
    parser.add_argument(
        "--use_dynamic_sampling",
        action="store_true",
        help=(
            "Enable DAPO Dynamic Sampling (arXiv:2503.14476): drop K-groups with "
            "zero reward variance (advantage=0, gradient=0) before the policy "
            "update. The policy update then runs on the smaller effective batch."
        ),
    )
    parser.add_argument(
        "--use_gdpo_normalization",
        action="store_true",
        help=(
            "Enable GDPO per-component K-group advantage normalization "
            "(Liu 2026, arXiv:2601.05242). Each reward component is "
            "K-group-normalized independently, then a weighted sum is taken, "
            "then the result is batch-renormalized. Equalizes per-component "
            "gradient contribution — low-variance components (e.g. saturated "
            "format reward) no longer get drowned out by high-variance ones. "
            "Existing per-component weights remain as multipliers on the "
            "normalized advantages; uniform weights recover the paper's design."
        ),
    )
    parser.add_argument(
        "--reward_shaping_alpha",
        type=float,
        default=0.0,
        help="SCoRe-style shaped reward alpha: R(A2)+α*(R(A2)-R(A1)).",
    )
    parser.add_argument(
        "--response_alpha",
        type=float,
        default=-1.0,
        help="Shaped reward alpha for response (A2). -1 means use --reward_shaping_alpha.",
    )
    parser.add_argument(
        "--feedback_alpha",
        type=float,
        default=-1.0,
        help="Shaped reward alpha for feedback (F1). -1 means use --reward_shaping_alpha.",
    )
    parser.add_argument(
        "--w_verification_accuracy",
        type=float,
        default=0.45,
        help="Weight for F1 verification accuracy reward",
    )
    parser.add_argument(
        "--w_fb_format",
        type=float,
        default=0.1,
        help="Weight for F1 format reward (<think></think> + \\boxed{VERDICT})",
    )
    parser.add_argument(
        "--freeze_a1_steps",
        type=int,
        default=0,
        help="Freeze A1 policy loss for N steps (SCoRe Stage I)",
    )
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument(
        "--clip_high",
        type=float,
        default=0.0,
        help=(
            "DAPO Clip-Higher (arXiv:2503.14476): asymmetric upper clip bound. "
            "When > 0, upper PPO clip becomes (1+clip_high), lower stays (1+clip_range). "
            "Recommended 0.28 with clip_range=0.2. 0.0 disables (symmetric)."
        ),
    )
    parser.add_argument(
        "--num_inner_epochs",
        type=int,
        default=4,
        help="Inner optimization epochs per rollout batch (mu in GRPO)",
    )
    parser.add_argument("--seed", type=int, default=42)

    # LoRA
    parser.add_argument("--no_peft", action="store_true", help="Disable LoRA")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)

    # Qwen2.5-VL specific
    parser.add_argument(
        "--freeze_vision_tower", action="store_true", help="Freeze vision encoder weights"
    )
    parser.add_argument("--max_pixels", type=int, default=401408, help="Max pixels per image")
    parser.add_argument(
        "--use_rescaled_rewards",
        action="store_true",
        help=(
            "Use the per-component [0, 1]-rescaled multi-turn reward composition "
            "(`compute_response_reward_breakdown_01` / "
            "`compute_feedback_reward_breakdown_01`). Each reward component "
            "(a2_correctness, no_regression, a2_format, downstream, verification, "
            "fb_format) is normalized to [0, 1] from its raw range; with "
            "weights summing to 1.0, resp_reward / fb_reward / total_reward "
            "are all strictly non-negative. Equalizes per-unit-weight gradient "
            "magnitude across components — matches the design that worked in "
            "the single-turn baseline-a1 run. Ignored when --single_turn_a1 "
            "is also set (baseline path has its own [0, 1] reward)."
        ),
    )
    parser.add_argument(
        "--use_pag_segment_rewards",
        action="store_true",
        help=(
            "PAG-faithful arm (arXiv:2506.10406). Switches the reward composer "
            "to per-segment binary {0,1} rewards: r_a1 = w_a1_corr·R_a1_corr_01 + "
            "w_a1_fmt·R_a1_fmt; r_a2 = w_a2_corr·R_a2_corr_01 + w_a2_fmt·R_a2_fmt + "
            "α·(R_a2_corr_01 − R_a1_corr_01). r_a1 and r_a2 drive INDEPENDENT "
            "K-group baselines in the trainer (separate_turn_loss path is "
            "forced on). Feedback head emits R_v ∈ {0,1} (verdict matches A1 "
            "truth) + format only; the downstream component is zeroed to "
            "match PAG's turn-independent γ=0 design. See also "
            "--use_selective_revision and --pag_shaping_alpha."
        ),
    )
    parser.add_argument(
        "--use_selective_revision",
        action="store_true",
        help=(
            "PAG-style selective revision gate (arXiv:2506.10406 §3.1). After "
            "F1 generation, extract the `\\boxed{}` verdict. CORRECT → A2 is "
            "NOT generated (trajectory terminates at F1, A2 completion is "
            "empty, A2 is excluded from the A2 K-group baseline and "
            "contributes 0 to the A2 policy loss). WRONG / missing / "
            "unparseable → A2 is generated as usual. Independent of "
            "--use_pag_segment_rewards (the gate can be tested in isolation)."
        ),
    )
    parser.add_argument(
        "--pag_shaping_alpha",
        type=float,
        default=1.0,
        help=(
            "α coefficient on b_y(ŷ_2) = α·(R_a2_corr_01 − R_a1_corr_01), the "
            "PAG shaping bonus added ONLY to A2's per-segment reward (NOT to "
            "A1, NOT to F1). PAG paper uses 1.0; their ablation found α=5 "
            "and α=10 gave no further improvement. Only used when "
            "--use_pag_segment_rewards is set."
        ),
    )
    parser.add_argument(
        "--single_turn_a1",
        action="store_true",
        help=(
            "Single-turn A1-only baseline mode. Skips F1 (critic) and A2 (refiner) "
            "generation/loss entirely; trains GRPO on A1 with a 2-component [0,1] "
            "reward (0.9*correctness + 0.1*format by default). Used to isolate "
            "algorithm bugs from multi-turn / two-reward composition issues. "
            "When set, the --w_a1_correctness / --w_a1_format flags are interpreted "
            "as the baseline weights; --w_a2_*, --w_no_regression and feedback "
            "flags are ignored. KL is applied to A1 only."
        ),
    )
    parser.add_argument("--min_pixels", type=int, default=200704, help="Min pixels per image")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="grpo",
        choices=["grpo", "dr_grpo"],
        help="GRPO loss variant: 'grpo' (vanilla) or 'dr_grpo' (removes length/difficulty bias)",
    )

    # vLLM acceleration
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Use vLLM for rollout generation (3-5x faster). Requires vllm package.",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.50,
        help="Fraction of GPU memory for vLLM (freed during training via sleep mode)",
    )

    # Response reward weights (must sum to 1.0)
    # Per-turn split: each turn carries 0.9·corr + 0.1·fmt sub-reward.
    parser.add_argument("--w_a1_correctness", type=float, default=0.27)
    parser.add_argument("--w_a1_format", type=float, default=0.03)
    parser.add_argument("--w_a2_correctness", type=float, default=0.27)
    parser.add_argument("--w_a2_format", type=float, default=0.03)
    parser.add_argument("--w_no_regression", type=float, default=0.40)
    # WR-bonus: additive Bernoulli reward when A1 wrong AND A2 right.
    # Default 0 keeps existing experiments unaffected. Set to 1.0 to
    # promote refinement without penalising RW (Option 5 from the
    # reward-design menu — distinct from no_regression shaping which
    # also penalises RW).
    parser.add_argument("--w_wr_bonus", type=float, default=0.0)

    # Feedback reward weights (must sum to 1.0)
    parser.add_argument("--w_downstream", type=float, default=0.45)

    # Logging and checkpointing
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)

    # Resume from checkpoint
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="",
        help="Path to checkpoint dir to resume from (e.g. /outputs/.../checkpoint-500). "
        "Loads LoRA adapter weights + optimizer state, skips already-processed samples.",
    )
    parser.add_argument(
        "--init_from_checkpoint",
        type=str,
        default="",
        help="Path to checkpoint dir whose LoRA adapter weights should INITIALIZE this "
        "run. Loads adapter weights AND optimizer state (if optimizer.pt exists). "
        "Unlike --resume_from_checkpoint, this does NOT inherit global_step and does "
        "NOT skip samples. Use this to start a fresh training schedule on top of a "
        "previously trained adapter (e.g. start a new epoch with the weights from "
        "another run as the starting point). Should not be set together with "
        "--resume_from_checkpoint.",
    )
    parser.add_argument(
        "--ref_model_init_from_checkpoint",
        type=str,
        default="",
        help="Path to checkpoint dir whose LoRA adapter weights are merged into a "
        "second frozen copy of the base model, used as the KL reference distribution. "
        "When set, the trainer's per-turn KL anchors against THIS checkpoint instead "
        "of the raw base model. Required whenever you initialize a trainable adapter "
        "from a checkpoint (e.g. baseline-A1 ckpt-1000 as the starting point for "
        "frozen-a1-mt runs) — otherwise the default ``disable_adapter_layers()`` ref "
        "path anchors to the raw base model and pulls A1 away from the init "
        "distribution. Costs one extra forward-pass model copy on each rank.",
    )

    # Multi-adapter routing
    parser.add_argument(
        "--adapter_routing_json",
        type=str,
        default="",
        help=(
            "JSON blob describing per-turn LoRA-adapter routing. When empty "
            "(default), the trainer uses a single 'default' adapter for all "
            "turns. Schema: "
            '{"turns": {"a1": <name>, "f1": <name>, "a2": <name>}, '
            '"adapters": [{"name": <str>, "trainable": <bool>, '
            '"init_from_checkpoint": <path?>, '
            '"warm_start_from_adapter": <name?>}]}. '
            "Adapters are loaded in list order — sources for warm_start "
            "must come before consumers."
        ),
    )

    # Debug
    parser.add_argument(
        "--debug", action="store_true", help="Print generated trajectories and reward breakdowns"
    )

    # Sanity check
    parser.add_argument(
        "--sanity_check_samples",
        type=int,
        default=0,
        help="Run sanity check on N samples (0=disabled, skip training)",
    )

    return parser.parse_args()


def _load_adapter_weights_from_checkpoint(
    ckpt_path: "Path",  # noqa: F821 - forward ref, Path imported by callers
    model: object,
    accelerator: object,
    adapter_routing: object,
) -> None:
    """Load LoRA adapter weights from a checkpoint into ``model`` in place.

    Layout assumptions:
      * Single-adapter mode: weights live at
        ``<ckpt>/adapter_model.safetensors`` (legacy path; one adapter named
        "default"). The file is loaded directly via set_peft_model_state_dict
        on the "default" adapter.
      * Multi-adapter mode: each trainable adapter lives under
        ``<ckpt>/<adapter_name>/adapter_model.safetensors``. Each is loaded
        in turn onto its matching named adapter.

    Backwards-compat: in multi-adapter mode, if a per-adapter sub-directory
    is missing but a top-level adapter_model.safetensors exists AND there is
    exactly one trainable adapter, the top-level file is loaded into that
    adapter. Helps when a single-adapter checkpoint is being repurposed as
    init for a new multi-adapter run.

    Args:
        ckpt_path: Filesystem path to the checkpoint directory.
        model: The (DDP-wrapped) PEFT model whose adapter weights are
            being overwritten in place.
        accelerator: HuggingFace Accelerator (for unwrap + device).
        adapter_routing: AdapterRoutingConfig describing the adapters.

    Raises:
        FileNotFoundError: When no adapter file can be located for one or
            more trainable adapters.
    """
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    unwrapped = accelerator.unwrap_model(model)

    if not adapter_routing.enabled:
        adapter_file = ckpt_path / "adapter_model.safetensors"
        if not adapter_file.exists():
            raise FileNotFoundError(
                f"Checkpoint {ckpt_path} is missing adapter_model.safetensors. "
                "Aborting to avoid silently running on random LoRA init."
            )
        state = load_file(str(adapter_file), device=str(accelerator.device))
        set_peft_model_state_dict(unwrapped, state)
        logger.info(f"Loaded LoRA adapter from {adapter_file} (single-adapter mode)")
        return

    trainable_names = adapter_routing.trainable_adapter_names()
    top_level_file = ckpt_path / "adapter_model.safetensors"
    use_top_level_fallback = top_level_file.exists() and len(trainable_names) == 1

    for adapter_name in trainable_names:
        per_adapter_file = ckpt_path / adapter_name / "adapter_model.safetensors"
        if per_adapter_file.exists():
            src = per_adapter_file
        elif use_top_level_fallback:
            src = top_level_file
            logger.info(
                f"No {adapter_name}/ sub-dir at {ckpt_path}; falling back to "
                f"top-level adapter_model.safetensors for the only trainable "
                f"adapter."
            )
        else:
            raise FileNotFoundError(
                f"Checkpoint {ckpt_path} missing adapter weights for "
                f"adapter {adapter_name!r}. Expected file at "
                f"{per_adapter_file} (or a top-level adapter_model.safetensors "
                "for single-trainable-adapter back-compat)."
            )
        state = load_file(str(src), device=str(accelerator.device))
        set_peft_model_state_dict(unwrapped, state, adapter_name=adapter_name)
        logger.info(f"Loaded LoRA adapter {adapter_name!r} from {src}")


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    from datetime import timedelta

    from accelerate import Accelerator, InitProcessGroupKwargs

    from vlm_grpo.utils import set_seed, setup_environment

    setup_environment()
    set_seed(args.seed)

    ddp_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    # Suppress verbose logging on non-main processes
    if not accelerator.is_main_process:
        logging.getLogger().setLevel(logging.WARNING)

    # Auto-detect model type from model_id if set to "auto"
    model_type = args.model_type
    if model_type == "auto":
        model_id_lower = args.model_id.lower()
        if "qwen" in model_id_lower:
            model_type = "qwen2vl"
        else:
            model_type = "llava"
        logger.info(f"Auto-detected model_type: {model_type}")

    # Build configs
    from vlm_grpo.config import (
        AdapterRoutingConfig,
        BaselineA1RewardWeights,
        FeedbackRewardWeights,
        ResponseRewardWeights,
        RolloutConfig,
        SelfReflectionConfig,
    )

    # Parse adapter routing JSON early so any schema errors surface before
    # the heavy model load. Empty string → routing.enabled=False → single-
    # adapter mode (unchanged behavior).
    if args.adapter_routing_json.strip():
        import json as _json

        try:
            routing_data = _json.loads(args.adapter_routing_json)
        except _json.JSONDecodeError as e:
            raise SystemExit(f"--adapter_routing_json is not valid JSON: {e}") from e
        adapter_routing = AdapterRoutingConfig.from_dict(routing_data)
        logger.info(
            f"Multi-adapter routing enabled: turns={adapter_routing.turns}, "
            f"adapters={[a.name for a in adapter_routing.adapters]} "
            f"(trainable={adapter_routing.trainable_adapter_names()})"
        )
    else:
        adapter_routing = AdapterRoutingConfig()

    response_weights = ResponseRewardWeights(
        w_a1_correctness=args.w_a1_correctness,
        w_a1_format=args.w_a1_format,
        w_a2_correctness=args.w_a2_correctness,
        w_a2_format=args.w_a2_format,
        w_no_regression=args.w_no_regression,
        w_wr_bonus=args.w_wr_bonus,
    )
    feedback_weights = FeedbackRewardWeights(
        w_downstream=args.w_downstream,
        w_verification_accuracy=args.w_verification_accuracy,
        w_format=args.w_fb_format,
    )
    # Baseline weights reuse --w_a1_correctness / --w_a1_format only when
    # actually running the single-turn-A1 baseline. For multi-turn runs the
    # baseline-weights field is constructed but never consumed; in that case
    # keep the dataclass defaults (0.9 / 0.1) so the convex-combination
    # validator stays quiet rather than firing a misleading warning every
    # multi-turn launch.
    if args.single_turn_a1:
        baseline_weights = BaselineA1RewardWeights(
            w_a1_correctness=args.w_a1_correctness,
            w_a1_format=args.w_a1_format,
        )
    else:
        baseline_weights = BaselineA1RewardWeights()
    rollout_config = RolloutConfig(
        k_samples=args.k_samples,
        max_completion_length=args.max_completion_length,
        a1_max_completion_length=args.a1_max_completion_length,
        f1_max_completion_length=args.f1_max_completion_length,
        a2_max_completion_length=args.a2_max_completion_length,
        temperature=args.temperature,
        feedback_temperature=args.feedback_temperature,
        a2_temperature=args.a2_temperature,
        batch_size=args.per_device_train_batch_size,
        reward_shaping_alpha=args.reward_shaping_alpha,
        response_alpha=args.response_alpha,
        feedback_alpha=args.feedback_alpha,
        single_turn_a1=args.single_turn_a1,
        use_rescaled_rewards=args.use_rescaled_rewards,
        use_pag_segment_rewards=getattr(args, "use_pag_segment_rewards", False),
        use_selective_revision=getattr(args, "use_selective_revision", False),
        pag_shaping_alpha=float(getattr(args, "pag_shaping_alpha", 1.0)),
    )
    config = SelfReflectionConfig(
        model_id=args.model_id,
        model_type=model_type,
        dataset_path=args.dataset_path,
        val_dataset_path=args.val_dataset_path,
        image_base_dir=args.image_base_dir,
        output_dir=args.output_dir,
        rollout=rollout_config,
        response_weights=response_weights,
        feedback_weights=feedback_weights,
        baseline_weights=baseline_weights,
        adapter_routing=adapter_routing,
        learning_rate=args.learning_rate,
        lr_warmup_steps=args.lr_warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_samples=args.max_samples,
        use_peft=not args.no_peft,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        kl_coeff=args.kl_coeff,
        a1_kl_coeff=args.a1_kl_coeff,
        a2_kl_coeff=args.a2_kl_coeff,
        fb_kl_coeff=args.fb_kl_coeff,
        separate_turn_loss=args.separate_turn_loss,
        use_dynamic_sampling=args.use_dynamic_sampling,
        use_gdpo_normalization=args.use_gdpo_normalization,
        reward_shaping_alpha=args.reward_shaping_alpha,
        freeze_a1_steps=args.freeze_a1_steps,
        clip_range=args.clip_range,
        clip_high=args.clip_high,
        loss_type=args.loss_type,
        freeze_vision_tower=args.freeze_vision_tower,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        num_inner_epochs=args.num_inner_epochs,
        debug=args.debug,
        sanity_check_samples=args.sanity_check_samples,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        val_check_interval=args.save_steps,
        seed=args.seed,
    )

    logger.info(f"Response weights: {response_weights.to_dict()}")
    logger.info(f"Feedback weights: {feedback_weights.to_dict()}")
    if args.single_turn_a1:
        logger.info(
            f"Single-turn A1 baseline ENABLED. Baseline weights: {baseline_weights.to_dict()}"
        )

    # Load dataset
    from vlm_grpo.data import load_self_reflection_dataset

    # If specific indices requested, load enough lines to cover them
    if args.sample_indices:
        requested_indices = set(int(x.strip()) for x in args.sample_indices.split(","))
        load_limit = max(requested_indices) + 1
    else:
        requested_indices = None
        load_limit = args.sanity_check_samples or args.max_samples

    # Pass max_pixels for Qwen2.5-VL pixel-count-based image resizing
    img_max_pixels = args.max_pixels if model_type == "qwen2vl" else None
    train_dataset = load_self_reflection_dataset(
        args.dataset_path,
        image_base_dir=args.image_base_dir,
        max_samples=load_limit,
        max_pixels=img_max_pixels,
    )

    # Filter to requested indices
    if requested_indices is not None:
        train_dataset = [s for s in train_dataset if s["sample_index"] in requested_indices]
        logger.info(
            f"Filtered to {len(train_dataset)} samples at indices: {sorted(requested_indices)}"
        )
    else:
        logger.info(f"Training dataset: {len(train_dataset)} samples")

    # Shuffle training data to avoid task-type clustering.
    # Fixed seed ensures identical order across accelerate processes.
    import random

    random.Random(args.seed).shuffle(train_dataset)
    logger.info("Shuffled training dataset")

    val_dataset = None
    if args.val_dataset_path:
        val_dataset = load_self_reflection_dataset(
            args.val_dataset_path,
            image_base_dir=args.image_base_dir,
            max_samples=min(500, args.sanity_check_samples or 500),
            max_pixels=img_max_pixels,
        )
        logger.info(f"Validation dataset: {len(val_dataset)} samples")
    elif args.val_split > 0:
        n_val = max(1, int(len(train_dataset) * args.val_split))
        val_dataset = train_dataset[-n_val:]
        train_dataset = train_dataset[:-n_val]
        logger.info(
            f"Split: {len(train_dataset)} train, {len(val_dataset)} val "
            f"({args.val_split:.0%} held out)"
        )

    # Sanity check mode (main process only)
    if args.sanity_check_samples > 0:
        if accelerator.is_main_process:
            _run_sanity_check(
                train_dataset,
                response_weights,
                feedback_weights,
                reward_shaping_alpha=args.reward_shaping_alpha,
            )
        return

    # Load model + ref model
    logger.info(f"Loading model: {args.model_id} (type={model_type})")

    import torch
    from transformers import AutoProcessor

    # Load processor with pixel constraints for Qwen2.5-VL
    if model_type == "qwen2vl":
        processor = AutoProcessor.from_pretrained(
            args.model_id,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    else:
        processor = AutoProcessor.from_pretrained(args.model_id)

    # Load policy model (ref model not needed — PEFT adapter disable gives base weights)
    logger.info("Loading policy model...")
    if model_type == "qwen2vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            # sdpa (not flash_attention_2): the transformers flash_attention
            # integration has a None-guard bug on s_aux — crashes on any model
            # without attention sinks (e.g. Qwen2.5-VL ViT).
            # See transformers/integrations/flash_attention.py:84.
            # sdpa routes through torch.nn.functional.scaled_dot_product_attention
            # which auto-selects flash/memory-efficient backends without the wrapper.
            attn_implementation="sdpa",
        ).to(accelerator.device)
    else:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            # sdpa (not flash_attention_2): the transformers flash_attention
            # integration has a None-guard bug on s_aux — crashes on any model
            # without attention sinks (e.g. Qwen2.5-VL ViT).
            # See transformers/integrations/flash_attention.py:84.
            # sdpa routes through torch.nn.functional.scaled_dot_product_attention
            # which auto-selects flash/memory-efficient backends without the wrapper.
            attn_implementation="sdpa",
        ).to(accelerator.device)

    # Freeze vision tower if requested (saves memory, preserves visual features)
    if args.freeze_vision_tower:
        frozen_count = 0
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad = False
                frozen_count += 1
        logger.info(f"Froze {frozen_count} vision tower parameters")

    # Enable gradient checkpointing to reduce activation memory
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled on policy model")

    # Barrier before PEFT wrap: each rank loads the model independently from
    # HF cache; without a barrier, if load timing differs across ranks, PEFT's
    # module-order discovery in `get_peft_model` can produce inconsistent
    # trainable-param lists → DDP `_verify_params_across_processes` crashes on
    # shape mismatch (observed v10-base/v10-r1init 2026-04-22, NCCL timeout on
    # SeqNum=1/2 collective, `params[0] sizes [64, 1280]` — LoRA on ViT).
    if accelerator.num_processes > 1:
        logger.info("Pre-PEFT barrier: syncing all ranks on model load")
        accelerator.wait_for_everyone()

    # Apply LoRA
    if not args.no_peft:
        import re as _re

        from peft import LoraConfig, get_peft_model

        # Qwen2.5-VL: use all-linear targets. The visual merger MLP bridges
        # vision→language; removing the merger from LoRA targets unconditionally
        # caused entropy collapse (v6 runs) due to higher effective lr per
        # param, so we keep "all-linear" as the default discovery.
        # LLaVA: use explicit target modules from config.
        if model_type == "qwen2vl":
            target_modules = "all-linear"
        else:
            target_modules = config.lora_target_modules

        # When routing declares ``frozen_lora_patterns`` (e.g. Job A:
        # ["visual"]), translate those substring patterns into a regex for
        # PEFT's ``exclude_modules`` so the matching modules are not even
        # LoRA-wrapped for adapters we add fresh (the ``feedback`` adapter in
        # our standard routing). PEFT's exclude_modules with a STRING is
        # treated as a regex matched against the full module name; we wrap
        # each user-supplied pattern as ``.*<pattern>.*`` and join with |.
        #
        # NOTE: this affects only the LoRA config used for FRESH adapters
        # (``add_adapter`` inside init_multi_adapter_model). The first
        # adapter loaded via ``PeftModel.from_pretrained`` reads its own
        # ``adapter_config.json`` from the checkpoint and ignores our
        # exclude regex — e.g. the response adapter warm-started from
        # baseline-A1 still carries vision LoRA tensors, but they remain
        # frozen via ``frozen_lora_patterns`` enforcement in
        # ``_apply_trainable_flags`` / ``_enforce_trainable_grad_flags``.
        exclude_modules = None
        if adapter_routing.enabled and adapter_routing.frozen_lora_patterns:
            exclude_modules = "|".join(
                ".*" + _re.escape(p) + ".*" for p in adapter_routing.frozen_lora_patterns
            )
            logger.info(
                f"LoRA exclude_modules regex set to {exclude_modules!r} "
                f"(derived from frozen_lora_patterns={adapter_routing.frozen_lora_patterns})"
            )

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
            task_type="CAUSAL_LM",
        )

        if adapter_routing.enabled:
            from vlm_grpo.multi_adapter import init_multi_adapter_model

            model = init_multi_adapter_model(model, lora_config, adapter_routing)
        else:
            model = get_peft_model(model, lora_config)

        # PEFT's ``get_peft_model`` freezes ALL non-LoRA base parameters by
        # default — that is the entire purpose of LoRA fine-tuning. Vision
        # tower, merger, and language-decoder base weights all end up frozen
        # automatically. With ``exclude_modules`` set above (via
        # ``frozen_lora_patterns``), visual modules also have no LoRA wrapping,
        # so the vision side is fully frozen end-to-end without any extra
        # bookkeeping. This is the recipe that matches LIVR / VLM-R1's LoRA
        # mode: language-decoder LoRA only, everything else frozen.

        model.print_trainable_parameters()
        logger.info(f"LoRA applied (r={args.lora_r}, alpha={args.lora_alpha})")

        # Post-PEFT sanity barrier + shape hash: helps diagnose future
        # param-shape mismatches by logging a per-rank hash of (name, shape)
        # tuples. All ranks must produce the same hash, else DDP will crash.
        if accelerator.num_processes > 1:
            trainable_shapes = [
                (n, tuple(p.shape)) for n, p in model.named_parameters() if p.requires_grad
            ]
            shape_hash = hash(tuple(trainable_shapes))
            logger.info(
                f"Rank {accelerator.process_index}: "
                f"{len(trainable_shapes)} trainable params, shape_hash={shape_hash}"
            )
            accelerator.wait_for_everyone()

    # Load frozen KL reference model (separate copy of base + checkpoint LoRA merged in).
    #
    # Without this, the trainer's KL-ref pass falls back to
    # ``unwrapped_model.disable_adapter_layers()`` (see critic_grpo.py:1331), which
    # produces RAW BASE MODEL log-probs as the KL target — wrong whenever the
    # trainable adapter was initialized from a checkpoint (e.g. baseline-A1
    # ckpt-1000). A1 then drifts from its init distribution toward the raw base
    # model, dragging A2 with it through the shared LoRA. See the KL-audit notes
    # in the experiments log for the full diagnosis.
    ref_model: object | None = None
    if args.ref_model_init_from_checkpoint:
        logger.info(
            f"Loading frozen KL reference model: base={args.model_id}, "
            f"LoRA={args.ref_model_init_from_checkpoint} (will be merged into base)."
        )
        from peft import PeftModel

        if model_type == "qwen2vl":
            ref_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).to(accelerator.device)
        else:
            from transformers import AutoModelForVision2Seq

            ref_base = AutoModelForVision2Seq.from_pretrained(
                args.model_id,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            ).to(accelerator.device)

        ref_with_lora = PeftModel.from_pretrained(
            ref_base,
            args.ref_model_init_from_checkpoint,
            is_trainable=False,
        )
        ref_model = ref_with_lora.merge_and_unload()
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()
        logger.info(
            "Frozen KL reference model ready (LoRA merged; all params requires_grad=False)."
        )

        if accelerator.num_processes > 1:
            accelerator.wait_for_everyone()

    # Create optimizer and prepare for distributed training
    from torch.optim import AdamW

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01, fused=True)

    # DeepSpeed requires train_micro_batch_size_per_gpu to be set.
    # Since we use a custom training loop (no DataLoader), set it explicitly.
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            args.per_device_train_batch_size
        )

    model, optimizer = accelerator.prepare(model, optimizer)
    logger.info(
        f"Distributed training: {accelerator.num_processes} processes, device={accelerator.device}"
    )

    # Shard training dataset across processes (drop remainder for even distribution)
    n = len(train_dataset)
    per_process = n // accelerator.num_processes
    start = accelerator.process_index * per_process
    end = start + per_process
    train_dataset = train_dataset[start:end]
    logger.info(
        f"Process {accelerator.process_index}: samples {start}-{end} ({len(train_dataset)} samples)"
    )

    # Initialize vLLM engine on ALL ranks. With distributed_executor_backend=
    # "external_launcher", vLLM joins the existing torch.distributed group
    # (set up by accelerate/DeepSpeed) instead of spawning its own workers.
    # Each rank runs vLLM for its own data shard — no broadcast needed.
    # Reference: https://huggingface.co/blog/vllm-colocate
    vllm_engine = None
    if args.use_vllm:
        from vlm_grpo.vllm_rollout import VLLMRolloutEngine

        vllm_engine = VLLMRolloutEngine(
            model_id=args.model_id,
            processor=processor,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            max_model_len=8192,
            max_pixels=args.max_pixels,
            min_pixels=args.min_pixels,
            seed=accelerator.process_index,
        )
        vllm_engine.sleep()
        logger.info(f"vLLM engine initialized on rank {accelerator.process_index} (sleeping)")
        # Sync all ranks before continuing — prevents DeepSpeed hangs when
        # main process is still loading weights asynchronously.
        accelerator.wait_for_everyone()

    # Mutual-exclusion warn (don't hard-fail; let user decide).
    if args.resume_from_checkpoint and args.init_from_checkpoint:
        logger.warning(
            "Both --resume_from_checkpoint and --init_from_checkpoint are set. "
            "--init_from_checkpoint will be IGNORED; --resume_from_checkpoint takes "
            "precedence (it inherits global_step + optimizer state)."
        )

    # Init from checkpoint: load LoRA adapter weights, AND optimizer state if
    # present, but skip trainer_state.json (so global_step stays 0) and the
    # LR scheduler position (so we run a full fresh schedule). Use case:
    # warm-start a new run from a previously trained adapter when
    # --resume_from_checkpoint would inherit the source's step counter and
    # silently truncate the schedule (we hit this with the
    # baseline-a1 → frozen-a1-mt runs: resuming from baseline-a1's
    # checkpoint-1000 only ran 125 of the 1125 scheduled steps).
    #
    # Loading the optimizer state restores Adam's m/v moments so the first
    # gradient step doesn't overshoot baseline-a1's already-trained weights.
    # Without this (commit aed88ef), peak LR + cold Adam moments degraded
    # the starting weights instead of fine-tuning them.
    if args.init_from_checkpoint and not args.resume_from_checkpoint:
        from pathlib import Path

        init_path = Path(args.init_from_checkpoint)
        logger.info(f"Initializing from checkpoint weights at: {init_path}")
        _load_adapter_weights_from_checkpoint(
            ckpt_path=init_path,
            model=model,
            accelerator=accelerator,
            adapter_routing=adapter_routing,
        )

        # Optimizer state: load if present so Adam moments carry over. Without
        # this, peak LR on cold Adam (m=v=0) destroys the starting weights.
        # Falls back to fresh optimizer (with warning) on missing/corrupt file.
        optimizer_status = "fresh"
        optim_path = init_path / "optimizer.pt"
        if optim_path.exists():
            try:
                optim_state = torch.load(str(optim_path), map_location=str(accelerator.device))
                optimizer.load_state_dict(optim_state)
                optimizer_status = "inherited"
                logger.info(f"Loaded optimizer state from {optim_path}")
            except Exception as e:  # noqa: BLE001 — broad to keep training resilient
                logger.warning(
                    f"Failed to load optimizer state from {optim_path} ({e!r}); "
                    f"continuing with fresh optimizer state. "
                    f"Peak LR + cold Adam may degrade starting weights."
                )
        else:
            logger.warning(
                f"--init_from_checkpoint: optimizer.pt not found at {optim_path}. "
                f"Continuing with fresh optimizer state. "
                f"Peak LR + cold Adam may degrade starting weights."
            )

        logger.info(
            f"Initialized from checkpoint weights at {init_path} "
            f"(global_step=0, optimizer={optimizer_status}, fresh schedule)"
        )

    # Resume from checkpoint: load LoRA adapter + optimizer state + skip samples
    resume_step = 0
    if args.resume_from_checkpoint:
        from pathlib import Path

        ckpt_path = Path(args.resume_from_checkpoint)
        logger.info(f"Resuming from checkpoint: {ckpt_path}")

        _load_adapter_weights_from_checkpoint(
            ckpt_path=ckpt_path,
            model=model,
            accelerator=accelerator,
            adapter_routing=adapter_routing,
        )

        # Load optimizer state if saved
        optim_path = ckpt_path / "optimizer.pt"
        if optim_path.exists():
            optim_state = torch.load(str(optim_path), map_location=str(accelerator.device))
            optimizer.load_state_dict(optim_state)
            logger.info(f"Loaded optimizer state from {optim_path}")

        # Extract step number from checkpoint dir name (e.g. "checkpoint-500" → 500)
        ckpt_name = ckpt_path.name
        if ckpt_name.startswith("checkpoint-"):
            resume_step = int(ckpt_name.split("-")[1])
            logger.info(f"Resuming from step {resume_step}")

        # Skip already-processed samples. Each step processes batch_size samples,
        # and the dataset was already sharded per-process.
        samples_done = resume_step * config.rollout.batch_size
        if samples_done > 0 and samples_done < len(train_dataset):
            train_dataset = train_dataset[samples_done:]
            logger.info(
                f"Skipped {samples_done} already-processed samples, {len(train_dataset)} remaining"
            )
        elif samples_done >= len(train_dataset):
            logger.info("All samples already processed in this checkpoint. Nothing to train.")
            return

    # Create trainer
    from vlm_grpo.critic_grpo import SelfReflectionGRPOTrainer

    trainer = SelfReflectionGRPOTrainer(
        model=model,
        ref_model=ref_model,
        processor=processor,
        config=config,
        optimizer=optimizer,
        accelerator=accelerator,
        vllm_engine=vllm_engine,
    )

    # Set global_step to resume point so logging/checkpointing continues correctly
    if resume_step > 0:
        trainer.global_step = resume_step
        logger.info(f"Trainer global_step set to {resume_step}")

    # Initialize wandb on main process
    if accelerator.is_main_process:
        import os

        import wandb

        wandb_project = os.environ.get("WANDB_PROJECT", "vlm-self-reflection-grpo")
        wandb_run_name = os.environ.get("WANDB_RUN_NAME", None)
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "model_id": args.model_id,
                "model_type": model_type,
                "dataset_path": args.dataset_path,
                "max_samples": args.max_samples,
                "k_samples": args.k_samples,
                "loss_type": args.loss_type,
                "learning_rate": args.learning_rate,
                "kl_coeff": args.kl_coeff,
                "clip_range": args.clip_range,
                "clip_high": args.clip_high,
                "num_inner_epochs": args.num_inner_epochs,
                "temperature": args.temperature,
                "feedback_temperature": args.feedback_temperature,
                "a2_temperature": args.a2_temperature,
                "a1_max_completion_length": args.a1_max_completion_length,
                "f1_max_completion_length": args.f1_max_completion_length,
                "a2_max_completion_length": args.a2_max_completion_length,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "num_gpus": accelerator.num_processes,
                "response_weights": response_weights.to_dict(),
                "feedback_weights": feedback_weights.to_dict(),
                "seed": args.seed,
            },
            resume="allow" if resume_step > 0 else None,
        )
        trainer.wandb_run = wandb.run
        logger.info(f"wandb initialized: {wandb.run.url}")
    else:
        trainer.wandb_run = None

    # Train
    logger.info("Starting self-reflection GRPO training...")
    logger.info("  Two-reward design: response (A1+A2) + feedback (F1)")
    logger.info("  Single LoRA adapter, shared parameter updates")
    if vllm_engine:
        logger.info("  vLLM rollout acceleration: ENABLED")
    if resume_step > 0:
        logger.info(f"  Resuming from step {resume_step}")
    trainer.train(train_dataset, val_dataset)

    # Finish wandb
    if accelerator.is_main_process:
        wandb.finish()


def _run_sanity_check(
    dataset: list[dict],
    response_weights: object,
    feedback_weights: object,
    reward_shaping_alpha: float = 0.0,
) -> None:
    """Run sanity check with synthetic (A1, F1, A2) tuples.

    Creates test trajectories to verify both reward signals work correctly.

    Args:
        dataset: List of sample dicts
        response_weights: Response reward weights
        feedback_weights: Feedback reward weights
        reward_shaping_alpha: SCoRe-style shaped reward alpha
    """
    from vlm_grpo.rewards.composition import (
        compute_feedback_reward_breakdown,
        compute_feedback_reward_breakdown_01,  # noqa: F401  (importable for tests/sanity)
        compute_response_reward_breakdown,
        compute_response_reward_breakdown_01,  # noqa: F401
    )

    logger.info("=" * 70)
    logger.info("SANITY CHECK MODE (two-reward self-reflection)")
    logger.info("=" * 70)
    logger.info(f"Response weights: {response_weights.to_dict()}")
    logger.info(f"Feedback weights: {feedback_weights.to_dict()}")
    logger.info("")

    num_samples = min(len(dataset), 10)

    for i in range(num_samples):
        sample = dataset[i]
        gt = sample["ground_truth"]
        a_type = sample["answer_type"]
        ch = sample["choices"]
        ds = sample["dataset_name"]

        logger.info(f"--- Sample {i} ---")
        logger.info(f"  GT: {gt} | Type: {a_type} | Dataset: {ds}")

        test_cases = {
            "RR_correct_verify": {
                "a1": gt,
                "f1": "CORRECT. Matches the image.",
                "a2": gt,
            },
            "RW_wrong_verify": {
                "a1": gt,
                "f1": "INCORRECT. Revise.",
                "a2": "ZZZ_WRONG_ANSWER",
            },
            "WR_correct_verify": {
                "a1": "ZZZ_WRONG_ANSWER",
                "f1": "INCORRECT. Should be the other option.",
                "a2": gt,
            },
            "WW_sycophantic_verify": {
                "a1": "ZZZ_WRONG_ANSWER",
                "f1": "CORRECT. Looks fine.",
                "a2": "ZZZ_STILL_WRONG",
            },
        }

        for case_name, traj in test_cases.items():
            resp_bd = compute_response_reward_breakdown(
                a1_text=traj["a1"],
                a2_text=traj["a2"],
                ground_truth=gt,
                answer_type=a_type,
                choices=ch,
                weights=response_weights,
                reward_shaping_alpha=reward_shaping_alpha,
            )
            fb_bd = compute_feedback_reward_breakdown(
                feedback_text=traj["f1"],
                a1_text=traj["a1"],
                a2_text=traj["a2"],
                ground_truth=gt,
                answer_type=a_type,
                choices=ch,
                weights=feedback_weights,
                reward_shaping_alpha=reward_shaping_alpha,
            )

            logger.info(f"  [{case_name}]")
            logger.info(
                f"    Response reward: {resp_bd.total_reward:+.2f} "
                f"(A1={'R' if resp_bd.a1_correct else 'W'}"
                f"->A2={'R' if resp_bd.a2_correct else 'W'})"
            )
            logger.info(f"      Components: {resp_bd.components}")
            logger.info(f"    Feedback reward: {fb_bd.total_reward:+.2f}")
            logger.info(f"      Components: {fb_bd.components}")

        logger.info("")

    logger.info("=" * 70)
    logger.info("EXPECTED BEHAVIOR:")
    logger.info("  RR_correct_verify:     +resp, +fb (verdict calibrated, A2 stable)")
    logger.info("  RW_wrong_verify:       −resp (A2 regression), ~0 fb (miscalibrated)")
    logger.info("  WR_correct_verify:     +resp (big WR bonus), +fb (verdict calibrated)")
    logger.info("  WW_sycophantic_verify: −resp, −fb (verdict miscalibrated)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
