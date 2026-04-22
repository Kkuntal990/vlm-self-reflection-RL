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
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=None,
        help="Samples per rollout step. Defaults to per_device_train_batch_size if not set.",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--kl_coeff", type=float, default=0.05)
    parser.add_argument("--a1_kl_coeff", type=float, default=1.0)
    parser.add_argument("--a2_kl_coeff", type=float, default=1.0)
    parser.add_argument("--fb_kl_coeff", type=float, default=1.0)
    parser.add_argument("--separate_turn_loss", action="store_true")
    parser.add_argument(
        "--use_ssr", action="store_true", help="Enable Selective Sample Replay (VL-Rethinker)"
    )
    parser.add_argument("--ssr_buffer_size", type=int, default=64)
    parser.add_argument("--ssr_alpha", type=float, default=1.0)
    parser.add_argument(
        "--use_improvement_reward",
        action="store_true",
        help="Use R(A2)-R(A1) improvement reward for F1 (Critique-GRPO)",
    )
    parser.add_argument(
        "--reward_shaping_alpha",
        type=float,
        default=0.0,
        help="SCoRe-style shaped reward alpha: R(A2)+α*(R(A2)-R(A1)). "
        "Overrides --use_improvement_reward when > 0.",
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
        "--use_think_answer_tags",
        action="store_true",
        help="Use <think>...</think><answer>...</answer> tag format for A1/A2 generation. "
        "Enables structured reasoning with format reward for tags.",
    )
    parser.add_argument(
        "--use_answer_tag_only",
        action="store_true",
        help="Use <answer>...</answer> tag only (no <think>). Model reasons freely, "
        "wraps final answer in <answer> tags for structured extraction.",
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
    parser.add_argument("--w_a1_correctness", type=float, default=0.25)
    parser.add_argument("--w_a2_correctness", type=float, default=0.25)
    parser.add_argument("--w_no_regression", type=float, default=0.25)
    parser.add_argument("--w_a2_format", type=float, default=0.25)

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
        FeedbackRewardWeights,
        ResponseRewardWeights,
        RolloutConfig,
        SelfReflectionConfig,
    )

    response_weights = ResponseRewardWeights(
        w_a1_correctness=args.w_a1_correctness,
        w_a2_correctness=args.w_a2_correctness,
        w_no_regression=args.w_no_regression,
        w_a2_format=args.w_a2_format,
    )
    feedback_weights = FeedbackRewardWeights(
        w_downstream=args.w_downstream,
        w_verification_accuracy=args.w_verification_accuracy,
        w_format=args.w_fb_format,
    )
    rollout_config = RolloutConfig(
        k_samples=args.k_samples,
        max_completion_length=args.max_completion_length,
        a1_max_completion_length=args.a1_max_completion_length,
        f1_max_completion_length=args.f1_max_completion_length,
        a2_max_completion_length=args.a2_max_completion_length,
        temperature=args.temperature,
        feedback_temperature=args.feedback_temperature,
        a2_temperature=args.a2_temperature,
        batch_size=args.rollout_batch_size or args.per_device_train_batch_size,
        use_think_answer_tags=args.use_think_answer_tags,
        use_answer_tag_only=getattr(args, "use_answer_tag_only", False),
        use_improvement_reward=args.use_improvement_reward,
        reward_shaping_alpha=args.reward_shaping_alpha,
        response_alpha=args.response_alpha,
        feedback_alpha=args.feedback_alpha,
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
        learning_rate=args.learning_rate,
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
        use_ssr=args.use_ssr,
        ssr_buffer_size=args.ssr_buffer_size,
        ssr_alpha=args.ssr_alpha,
        use_improvement_reward=args.use_improvement_reward,
        reward_shaping_alpha=args.reward_shaping_alpha,
        freeze_a1_steps=args.freeze_a1_steps,
        clip_range=args.clip_range,
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
            attn_implementation="flash_attention_2",
        ).to(accelerator.device)
    else:
        from transformers import AutoModelForVision2Seq

        model = AutoModelForVision2Seq.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
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

    # Apply LoRA
    if not args.no_peft:
        from peft import LoraConfig, get_peft_model

        # Qwen2.5-VL: use all-linear targets. PEFT auto-excludes frozen
        # vision tower params (requires_grad=False), but includes the
        # visual merger MLP which bridges vision→language. Removing the
        # merger from LoRA targets caused entropy collapse (v6 runs) due
        # to higher effective lr per param. The dead vision LoRA_B params
        # (~350MB) are acceptable overhead for training stability.
        # LLaVA: use explicit target modules from config.
        if model_type == "qwen2vl":
            target_modules = "all-linear"
        else:
            target_modules = config.lora_target_modules

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info(f"LoRA applied (r={args.lora_r}, alpha={args.lora_alpha})")

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
            max_model_len=2048,
            max_pixels=args.max_pixels,
            min_pixels=args.min_pixels,
            seed=accelerator.process_index,
        )
        vllm_engine.sleep()
        logger.info(f"vLLM engine initialized on rank {accelerator.process_index} (sleeping)")
        # Sync all ranks before continuing — prevents DeepSpeed hangs when
        # main process is still loading weights asynchronously.
        accelerator.wait_for_everyone()

    # Resume from checkpoint: load LoRA adapter + optimizer state + skip samples
    resume_step = 0
    if args.resume_from_checkpoint:
        from pathlib import Path

        ckpt_path = Path(args.resume_from_checkpoint)
        logger.info(f"Resuming from checkpoint: {ckpt_path}")

        # Load LoRA adapter weights into the PEFT model
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        adapter_path = ckpt_path / "adapter_model.safetensors"
        if adapter_path.exists():
            adapter_state = load_file(str(adapter_path), device=str(accelerator.device))
            unwrapped = accelerator.unwrap_model(model)
            set_peft_model_state_dict(unwrapped, adapter_state)
            logger.info(f"Loaded LoRA adapter from {adapter_path}")
        else:
            logger.warning(f"No adapter_model.safetensors found at {ckpt_path}")

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
        ref_model=None,
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
        compute_response_reward_breakdown,
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
