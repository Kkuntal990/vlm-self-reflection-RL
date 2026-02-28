#!/usr/bin/env python3
"""
Entry point for critic GRPO training with downstream-aware rewards.

Trains the critic policy π(F1 | Q, I, A1) using a custom GRPO loop.
The critic generates feedback (F1) and is rewarded based on how the
resulting refined answer (A2) turns out — this is the "downstream-aware"
reward signal.

Training uses a separate LoRA adapter for the critic role.

Usage:
    # Sanity check (no GPU needed for reward computation):
    python grpo/train_critic.py \\
        --dataset_path /outputs/grpo_data/answer1_correct_train.jsonl \\
        --sanity_check_samples 5

    # Full training:
    python grpo/train_critic.py \\
        --model_id llava-hf/llava-1.5-7b-hf \\
        --dataset_path /outputs/grpo_data/answer1_correct_train.jsonl \\
        --val_dataset_path /outputs/grpo_data/answer1_correct_val.jsonl \\
        --output_dir ./outputs/grpo_critic \\
        --phase rw_first

Reference:
    - GRPO: https://arxiv.org/abs/2402.03300
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
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Train critic with downstream-aware GRPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model_id",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="HuggingFace model identifier or local checkpoint path",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to precomputed JSONL dataset",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default="",
        help="Path to validation JSONL dataset",
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

    # Training configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/grpo_critic",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="rw_first",
        choices=["rw_first", "full"],
        help="Training phase: rw_first (A1 correct only) or full",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )

    # Rollout configuration
    parser.add_argument(
        "--k_samples",
        type=int,
        default=4,
        help="Number of F1/A2 completions per sample",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=512,
        help="Maximum tokens per generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for F1 generation",
    )
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=8,
        help="Batch size for rollout generation",
    )

    # LoRA configuration
    parser.add_argument(
        "--no_peft",
        action="store_true",
        help="Disable LoRA (full fine-tuning)",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )

    # GRPO configuration
    parser.add_argument(
        "--kl_coeff",
        type=float,
        default=0.05,
        help="KL divergence coefficient",
    )
    parser.add_argument(
        "--clip_range",
        type=float,
        default=0.2,
        help="Policy ratio clipping range",
    )

    # Reward weights
    parser.add_argument(
        "--w_downstream", type=float, default=2.0, help="Downstream reward weight",
    )
    parser.add_argument(
        "--w_calibration", type=float, default=1.0, help="Calibration reward weight",
    )
    parser.add_argument(
        "--w_format", type=float, default=0.5, help="Format reward weight",
    )

    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=10, help="Log interval")
    parser.add_argument("--save_steps", type=int, default=500, help="Save interval")
    parser.add_argument(
        "--val_check_interval", type=int, default=500, help="Validation interval",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Sanity check
    parser.add_argument(
        "--sanity_check_samples",
        type=int,
        default=0,
        help="Number of samples for reward sanity check (0=disabled)",
    )

    return parser.parse_args()


def _run_sanity_check(args: argparse.Namespace) -> None:
    """Run sanity check: compute rewards on synthetic completions.

    Creates synthetic (F1, A2) pairs and prints reward breakdowns
    to verify reward logic before training.

    Args:
        args: Parsed CLI arguments
    """
    from vlm_grpo.config import CriticRewardWeights
    from vlm_grpo.data import _load_jsonl
    from vlm_grpo.rewards.composition import compute_critic_reward_breakdown

    weights = CriticRewardWeights(
        w_downstream=args.w_downstream,
        w_calibration=args.w_calibration,
        w_format=args.w_format,
    )

    samples = _load_jsonl(args.dataset_path, max_samples=args.sanity_check_samples)
    logger.info(f"Sanity check on {len(samples)} samples with weights: {weights}")

    for i, sample in enumerate(samples):
        gt = sample.get("ground_truth", "A")
        a1 = sample.get("answer1", "A")
        at = sample.get("answer_type", "mcq")
        ch = sample.get("choices", "")
        a1_correct = sample.get("a1_is_correct", True)

        logger.info(f"\n--- Sample {i}: gt={gt}, a1={a1}, type={at} ---")

        # Synthetic F1/A2 pairs
        wrong = "B" if gt != "B" else "C"
        test_cases = [
            ("Calibrated RR", "The answer is correct and matches the image.", gt),
            ("Miscalibrated RW", f"The answer is wrong, should be {wrong}.", wrong),
            ("Empty feedback", "", gt),
            ("Short feedback", "OK", gt),
            ("Negative RR", "The answer seems incorrect.", gt),
        ]

        for name, feedback, a2 in test_cases:
            bd = compute_critic_reward_breakdown(
                feedback_text=feedback,
                a2_text=a2,
                ground_truth=gt,
                answer1=a1,
                a1_is_correct=a1_correct,
                answer_type=at,
                choices=ch,
                weights=weights,
            )
            logger.info(
                f"  [{name}] total={bd.total_reward:+.2f} | "
                f"downstream={bd.components['downstream']:+.1f} "
                f"calib={bd.components['calibration']:+.1f} "
                f"format={bd.components['format']:+.1f} | "
                f"a2_correct={bd.a2_correct}"
            )

    logger.info("\nSanity check complete.")


def main() -> None:
    """Main function for critic GRPO training."""
    args = parse_args()

    from vlm_grpo.utils import set_seed, setup_environment

    setup_environment()
    set_seed(args.seed)

    # Sanity check mode
    if args.sanity_check_samples > 0:
        _run_sanity_check(args)
        return

    # Full training mode — lazy imports for heavy ML libraries
    import copy

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoProcessor

    from vlm_grpo.config import (
        CriticRewardWeights,
        RolloutConfig,
        TwoTrajectoryConfig,
    )
    from vlm_grpo.critic_grpo import CriticGRPOTrainer
    from vlm_grpo.data import _load_jsonl
    from vlm_grpo.rollout import RolloutEngine

    logger.info(f"Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Apply LoRA for critic adapter
    if not args.no_peft:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"Applied LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        model.print_trainable_parameters()

    # Create frozen reference model
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    logger.info("Created frozen reference model")

    # Build configuration
    reward_weights = CriticRewardWeights(
        w_downstream=args.w_downstream,
        w_calibration=args.w_calibration,
        w_format=args.w_format,
    )

    rollout_config = RolloutConfig(
        k_samples=args.k_samples,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        batch_size=args.rollout_batch_size,
    )

    config = TwoTrajectoryConfig(
        model_id=args.model_id,
        dataset_path=args.dataset_path,
        val_dataset_path=args.val_dataset_path,
        image_base_dir=args.image_base_dir,
        output_dir=args.output_dir,
        phase=args.phase,
        rollout=rollout_config,
        critic_weights=reward_weights,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_samples=args.max_samples,
        kl_coeff=args.kl_coeff,
        clip_range=args.clip_range,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        val_check_interval=args.val_check_interval,
        seed=args.seed,
    )

    # Load datasets as list[dict] for the custom training loop
    train_data = _load_jsonl(args.dataset_path, args.max_samples)
    val_data = _load_jsonl(args.val_dataset_path) if args.val_dataset_path else None

    logger.info(f"Train samples: {len(train_data)}")
    if val_data:
        logger.info(f"Val samples: {len(val_data)}")

    # Create rollout engine
    rollout_engine = RolloutEngine(
        model=model,
        processor=processor,
        config=rollout_config,
        reward_weights=reward_weights,
    )

    # Create trainer
    trainer = CriticGRPOTrainer(
        model=model,
        ref_model=ref_model,
        processor=processor,
        config=config,
        rollout_engine=rollout_engine,
    )

    # Train
    logger.info("Starting critic GRPO training...")
    metrics = trainer.train(train_data, val_data)

    logger.info(f"Training complete. Final metrics: {metrics}")


if __name__ == "__main__":
    main()
