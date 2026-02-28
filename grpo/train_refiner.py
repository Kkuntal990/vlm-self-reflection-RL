#!/usr/bin/env python3
"""
Entry point for refiner GRPO training using TRL's GRPOTrainer.

Trains the refiner policy π(A2 | Q, I, A1, F1) using standard GRPO.
The refiner generates a refined answer (A2) given the initial answer (A1)
and critic feedback (F1), and is rewarded for correctness, stability,
and minimal unnecessary edits.

Training uses a separate LoRA adapter for the refiner role.
Feedback (F1) is pre-computed using a frozen critic model.

Usage:
    # Sanity check:
    python grpo/train_refiner.py \\
        --dataset_path /outputs/grpo_data/answer1_correct_train.jsonl \\
        --feedback_path /outputs/grpo_data/critic_feedbacks.jsonl \\
        --sanity_check_samples 5

    # Full training:
    python grpo/train_refiner.py \\
        --model_id llava-hf/llava-1.5-7b-hf \\
        --dataset_path /outputs/grpo_data/answer1_correct_train.jsonl \\
        --feedback_path /outputs/grpo_data/critic_feedbacks.jsonl \\
        --output_dir ./outputs/grpo_refiner \\
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
        description="Train refiner with standard GRPO via TRL",
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
        "--feedback_path",
        type=str,
        default="",
        help="Path to JSONL with pre-computed critic feedbacks",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default="",
        help="Path to validation JSONL dataset",
    )
    parser.add_argument(
        "--val_feedback_path",
        type=str,
        default="",
        help="Path to JSONL with pre-computed val feedbacks",
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
        default="./outputs/grpo_refiner",
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
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )

    # GRPO configuration
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Number of A2 completions per prompt (K)",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=512,
        help="Maximum tokens per generation",
    )

    # vLLM configuration
    parser.add_argument("--no_vllm", action="store_true", help="Disable vLLM")
    parser.add_argument(
        "--vllm_mode",
        type=str,
        default="colocate",
        help="vLLM mode: colocate or server",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.3,
        help="GPU memory fraction for vLLM",
    )

    # LoRA configuration
    parser.add_argument("--no_peft", action="store_true", help="Disable LoRA")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")

    # Reward weights
    parser.add_argument(
        "--w_correctness", type=float, default=1.0, help="Correctness weight",
    )
    parser.add_argument(
        "--w_no_regression", type=float, default=2.0, help="No-regression weight",
    )
    parser.add_argument(
        "--w_minimal_edit", type=float, default=0.3, help="Minimal edit weight",
    )
    parser.add_argument(
        "--w_format", type=float, default=0.5, help="Format weight",
    )

    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=10, help="Steps between logging")
    parser.add_argument("--save_steps", type=int, default=500, help="Steps between saves")
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
    """Run sanity check: compute rewards on synthetic A2 completions.

    Args:
        args: Parsed CLI arguments
    """
    from vlm_grpo.data import _load_jsonl
    from vlm_grpo.rewards.composition import (
        RefinerRewardWeights,
        compute_refiner_reward_breakdown,
    )

    weights = RefinerRewardWeights(
        w_correctness=args.w_correctness,
        w_no_regression=args.w_no_regression,
        w_minimal_edit=args.w_minimal_edit,
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

        # Synthetic A2 completions
        wrong_answer = "B" if gt != "B" else "C"
        test_cases = [
            ("Perfect RR (same)", gt),
            ("RW regression", wrong_answer),
            ("Empty answer", ""),
        ]

        for name, a2 in test_cases:
            bd = compute_refiner_reward_breakdown(
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
                f"correct={bd.components['correctness']:+.1f} "
                f"regression={bd.components['no_regression']:+.1f} "
                f"edit={bd.components['minimal_edit']:+.1f} "
                f"format={bd.components['format']:+.1f} | "
                f"a2_correct={bd.a2_correct}"
            )

    logger.info("\nSanity check complete.")


def main() -> None:
    """Main function for refiner GRPO training."""
    args = parse_args()

    from vlm_grpo.utils import set_seed, setup_environment

    setup_environment()
    set_seed(args.seed)

    # Sanity check mode
    if args.sanity_check_samples > 0:
        _run_sanity_check(args)
        return

    # Full training mode — lazy imports for heavy ML libraries
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from vlm_grpo.data import load_refiner_dataset
    from vlm_grpo.rewards.composition import (
        RefinerRewardWeights,
        get_refiner_reward_functions,
    )

    # Build reward weights
    reward_weights = RefinerRewardWeights(
        w_correctness=args.w_correctness,
        w_no_regression=args.w_no_regression,
        w_minimal_edit=args.w_minimal_edit,
        w_format=args.w_format,
    )

    # Load datasets
    train_dataset = load_refiner_dataset(
        dataset_path=args.dataset_path,
        image_base_dir=args.image_base_dir,
        feedback_path=args.feedback_path,
        max_samples=args.max_samples,
    )

    eval_dataset = None
    if args.val_dataset_path:
        eval_dataset = load_refiner_dataset(
            dataset_path=args.val_dataset_path,
            image_base_dir=args.image_base_dir,
            feedback_path=args.val_feedback_path,
        )

    logger.info(f"Train dataset: {len(train_dataset)} samples")
    if eval_dataset:
        logger.info(f"Eval dataset: {len(eval_dataset)} samples")

    # Get TRL-compatible reward functions
    reward_funcs = get_refiner_reward_functions()
    logger.info(f"Using {len(reward_funcs)} reward functions")
    logger.info(f"Reward weights: {reward_weights.to_list()}")

    # LoRA config
    peft_config = None
    if not args.no_peft:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        logger.info(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")

    # GRPO config
    use_vllm = not args.no_vllm
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        reward_weights=reward_weights.to_list(),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        use_vllm=use_vllm,
        vllm_mode=args.vllm_mode if use_vllm else None,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization if use_vllm else None,
    )

    # Create trainer
    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting refiner GRPO training...")
    trainer.train()

    # Save
    trainer.save_model(args.output_dir)
    logger.info(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
