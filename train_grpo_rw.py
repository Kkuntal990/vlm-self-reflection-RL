#!/usr/bin/env python3
"""
GRPO training for reducing RW flips in VLM self-reflection.

Uses TRL's GRPOTrainer with 5 custom reward functions to train a
vision-language model to maintain correctness during self-reflection
(reduce Right-to-Wrong transitions).

The model receives a prompt with image + question + Answer1 and generates
a single completion containing free-form FEEDBACK and FINAL_ANSWER.

Usage:
    # Sanity check mode (prints reward breakdowns, no GPU needed for rewards)
    python train_grpo_rw.py \
        --dataset_path /outputs/grpo_data/answer1_correct_train.jsonl \
        --sanity_check_samples 50

    # Full training (4 GPUs, vLLM colocate)
    accelerate launch --num_processes=4 train_grpo_rw.py \
        --model_id /outputs/llava-1.5-sft-checkpoint \
        --dataset_path /outputs/grpo_data/answer1_correct_train.jsonl \
        --val_dataset_path /outputs/grpo_data/answer1_correct_test.jsonl \
        --output_dir /outputs/grpo_rw_v1

    # Without vLLM (single GPU dev mode)
    python train_grpo_rw.py \
        --model_id llava-hf/llava-1.5-7b-hf \
        --dataset_path /outputs/grpo_data/answer1_correct_train.jsonl \
        --no_vllm --max_samples 20

Reference:
    - GRPO: https://arxiv.org/abs/2402.03300
    - SCoRe: https://arxiv.org/abs/2409.12917
    - TRL VLM Alignment: https://huggingface.co/blog/trl-vlm-alignment
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GRPO training for VLM self-reflection RW reduction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model_id",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="HuggingFace model ID or local checkpoint path",
    )

    # Dataset configuration
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to precomputed Answer1-correct JSONL",
    )
    parser.add_argument(
        "--val_dataset_path",
        type=str,
        default="",
        help="Path to validation JSONL (for periodic RW rate logging)",
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
        default="./outputs/grpo_rw",
        help="Output directory for checkpoints and logs",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Number of completions per prompt for GRPO (K)",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=512,
        help="Maximum tokens in generated completion",
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
        help="Per-GPU batch size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # vLLM configuration
    parser.add_argument(
        "--no_vllm",
        action="store_true",
        help="Disable vLLM (use standard HF generation)",
    )
    parser.add_argument(
        "--vllm_mode",
        type=str,
        default="colocate",
        choices=["colocate", "server"],
        help="vLLM mode",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.3,
        help="GPU memory fraction for vLLM",
    )

    # LoRA configuration
    parser.add_argument(
        "--no_peft",
        action="store_true",
        help="Disable LoRA (use full fine-tuning)",
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

    # Reward weights
    parser.add_argument("--w_final", type=float, default=1.0, help="Weight for R_final_correct")
    parser.add_argument("--w_format", type=float, default=0.5, help="Weight for R_format")
    parser.add_argument("--w_rw", type=float, default=2.0, help="Weight for R_no_regression")
    parser.add_argument("--w_edit", type=float, default=0.3, help="Weight for R_minimal_edit")
    parser.add_argument(
        "--w_fb", type=float, default=0.5, help="Weight for R_feedback_calibration"
    )

    # Logging and checkpointing
    parser.add_argument("--logging_steps", type=int, default=10, help="Steps between logging")
    parser.add_argument(
        "--save_steps", type=int, default=500, help="Steps between checkpoint saves"
    )
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=500,
        help="Steps between validation RW rate checks",
    )

    # Sanity check
    parser.add_argument(
        "--sanity_check_samples",
        type=int,
        default=0,
        help="Run sanity check on N samples (0 = disabled, skip training)",
    )

    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()

    # Setup environment and seed
    from vlm_grpo.utils import set_seed, setup_environment

    setup_environment()
    set_seed(args.seed)

    # Build reward weights config
    from vlm_grpo.config import RewardWeights

    reward_weights = RewardWeights(
        w_final=args.w_final,
        w_format=args.w_format,
        w_rw=args.w_rw,
        w_edit=args.w_edit,
        w_fb=args.w_fb,
    )

    logger.info(f"Reward weights: {reward_weights.to_dict()}")

    # Load datasets
    from vlm_grpo.data import load_grpo_dataset

    train_dataset = load_grpo_dataset(
        args.dataset_path,
        image_base_dir=args.image_base_dir,
        max_samples=args.sanity_check_samples or args.max_samples,
    )
    logger.info(f"Training dataset: {len(train_dataset)} samples")

    val_dataset = None
    if args.val_dataset_path:
        val_dataset = load_grpo_dataset(
            args.val_dataset_path,
            image_base_dir=args.image_base_dir,
            max_samples=min(500, args.sanity_check_samples or 500),
        )
        logger.info(f"Validation dataset: {len(val_dataset)} samples")

    # Sanity check mode
    if args.sanity_check_samples > 0:
        _run_sanity_check(train_dataset, reward_weights)
        return

    # Import TRL components (lazy to keep sanity check fast)
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from vlm_grpo.rewards.rw_reward import get_reward_functions

    # Get reward functions
    reward_fns = get_reward_functions()
    logger.info(f"Using {len(reward_fns)} reward functions")

    # LoRA config
    peft_config = None
    if not args.no_peft:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        logger.info(f"Using LoRA (r={args.lora_r}, alpha={args.lora_alpha})")

    # GRPOConfig
    use_vllm = not args.no_vllm
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        # max_prompt_length removed in TRL >=0.29
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        remove_unused_columns=False,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        log_completions=True,
        report_to="wandb",
        reward_weights=reward_weights.to_list(),
        **(
            {
                "use_vllm": True,
                "vllm_mode": args.vllm_mode,
                "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
            }
            if use_vllm
            else {}
        ),
    )

    logger.info(f"GRPOConfig: num_generations={args.num_generations}, use_vllm={use_vllm}")
    logger.info(f"Output directory: {args.output_dir}")

    # Create trainer
    trainer = GRPOTrainer(
        model=args.model_id,
        reward_funcs=reward_fns,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting GRPO training...")
    trainer.train()

    # Save
    trainer.save_model(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")

    # Final metrics
    if val_dataset is not None:
        _log_final_metrics(trainer, val_dataset)


def _run_sanity_check(dataset, reward_weights: "RewardWeights") -> None:
    """Run sanity check with synthetic completions and print reward breakdowns.

    Creates several types of synthetic completions to verify reward logic:
    1. Perfect (correct answer, calibrated feedback)
    2. RW flip (wrong answer, miscalibrated feedback)
    3. Malformed (missing markers)
    4. Hedging (ambiguous answer)

    Args:
        dataset: Loaded HF dataset
        reward_weights: Reward component weights
    """
    from vlm_grpo.rewards.rw_reward import compute_full_breakdown

    logger.info("=" * 70)
    logger.info("SANITY CHECK MODE")
    logger.info("=" * 70)
    logger.info(f"Reward weights: {reward_weights.to_dict()}")
    logger.info("")

    num_samples = min(len(dataset), 10)

    for i in range(num_samples):
        row = dataset[i]
        gt = row["ground_truth"]
        a1 = row["answer1"]
        a_type = row["answer_type"]
        ch = row["choices"]
        ds = row["dataset_name"]

        logger.info(f"--- Sample {i} ---")
        logger.info(f"  GT: {gt} | A1: {a1} | Type: {a_type} | Dataset: {ds}")

        # Synthetic completions
        test_cases = {
            "perfect_rr": f"FEEDBACK:\nThe answer is correct and well-supported by the image.\nFINAL_ANSWER:\n{a1}",
            "rw_flip": f"FEEDBACK:\nThe answer is incorrect, should be changed.\nFINAL_ANSWER:\nZZZ_WRONG",
            "malformed": "This is just some text without markers.",
            "feedback_only": f"FEEDBACK:\nLooks good.\n",
            "answer_only": f"FINAL_ANSWER:\n{a1}",
        }

        for case_name, completion in test_cases.items():
            breakdown = compute_full_breakdown(completion, gt, a1, a_type, ch, ds)
            logger.info(f"  [{case_name}]")
            logger.info(f"    Total: {breakdown.total_reward:+.2f}")
            logger.info(f"    Components: {breakdown.components}")
            logger.info(f"    Weighted:   {breakdown.weighted_components}")
            logger.info(f"    Format OK: {breakdown.format_valid} | Parse OK: {breakdown.parse_success}")
            logger.info(f"    Extracted: '{breakdown.final_answer_extracted}'")

        logger.info("")

    # Summary statistics
    logger.info("=" * 70)
    logger.info("EXPECTED BEHAVIOR:")
    logger.info("  perfect_rr:   High positive reward (all components positive)")
    logger.info("  rw_flip:      Large negative reward (no_regression=-3.0)")
    logger.info("  malformed:    Negative reward (format=-1.0, all others gated)")
    logger.info("  feedback_only: Negative reward (format=-1.0, missing FINAL_ANSWER)")
    logger.info("  answer_only:  Negative reward (format=-1.0, missing FEEDBACK)")
    logger.info("=" * 70)


def _log_final_metrics(trainer, val_dataset) -> None:
    """Log final transition metrics on validation set.

    Args:
        trainer: Trained GRPOTrainer
        val_dataset: Validation dataset
    """
    logger.info("Computing final validation metrics...")
    # This would require generating completions on val_dataset
    # which happens naturally during eval - just log a note
    logger.info("Run evaluation separately with scripts/precompute_answer1.py for detailed metrics")


if __name__ == "__main__":
    main()
