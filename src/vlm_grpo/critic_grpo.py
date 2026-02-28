#!/usr/bin/env python3
"""
Custom GRPO training loop for the critic model.

Implements Group Relative Policy Optimization for the critic role in
two-trajectory self-reflection training. The critic generates feedback (F1)
and is trained with downstream-aware rewards based on how the resulting
refined answer (A2) turns out.

This module provides a custom training loop because TRL's standard
GRPOTrainer cannot handle downstream-aware rewards (which require
generating A2 to evaluate F1).

GRPO loss:
    L = -E[min(r(θ)*A, clip(r(θ), 1-ε, 1+ε)*A)] + β*KL(π_θ || π_ref)
    where r(θ) = π_θ(F1|prompt) / π_old(F1|prompt)
    and A = group-normalized advantage

Usage:
    from vlm_grpo.critic_grpo import CriticGRPOTrainer

    trainer = CriticGRPOTrainer(
        model=model, ref_model=ref_model, processor=processor,
        config=config, rollout_engine=engine,
    )
    trainer.train(train_dataset, val_dataset)
"""

import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from vlm_grpo.config import (
    TwoTrajectoryConfig,
)
from vlm_grpo.rollout import (
    RolloutEngine,
    compute_rollout_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class CriticTrainStepResult:
    """Result of a single critic training step.

    Attributes:
        loss: Total loss value
        policy_loss: Clipped policy gradient loss
        kl_loss: KL divergence from reference model
        reward_mean: Mean reward across the batch
        reward_std: Standard deviation of rewards
        rollout_metrics: Dict of rollout statistics
        global_step: Current global training step
    """

    loss: float
    policy_loss: float
    kl_loss: float
    reward_mean: float
    reward_std: float
    rollout_metrics: dict[str, float]
    global_step: int

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return asdict(self)


class CriticGRPOTrainer:
    """Custom GRPO trainer for critic with downstream-aware rewards.

    Implements the full training loop:
    1. ROLLOUT: Generate K F1+A2 pairs per sample
    2. REWARD: Compute critic rewards from (F1, A2, ground_truth)
    3. ADVANTAGE: Group-relative normalization within K samples
    4. LOSS: Clipped policy gradient + KL divergence
    5. BACKWARD + STEP: Update model parameters
    """

    def __init__(
        self,
        model: Any,
        ref_model: Any,
        processor: Any,
        config: TwoTrajectoryConfig,
        rollout_engine: RolloutEngine,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
    ) -> None:
        """Initialize the critic GRPO trainer.

        Uses lazy imports for heavy ML libraries.

        Args:
            model: The critic policy model (with LoRA adapter)
            ref_model: Frozen reference model for KL regularization
            processor: Tokenizer/processor for the model
            config: Two-trajectory training configuration
            rollout_engine: Pre-configured rollout engine
            optimizer: Optional pre-configured optimizer
            scheduler: Optional learning rate scheduler
        """
        # Lazy imports for heavy ML libraries
        from torch.optim import AdamW

        self.model = model
        self.ref_model = ref_model
        self.processor = processor
        self.config = config
        self.rollout_engine = rollout_engine
        self.device = next(model.parameters()).device

        # Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=0.01,
            )

        self.scheduler = scheduler
        self.global_step = 0
        self.best_metric = float("inf") if config.early_stopping.mode == "min" else float("-inf")
        self.patience_counter = 0

    def train(
        self,
        train_dataset: list[dict],
        val_dataset: Optional[list[dict]] = None,
    ) -> dict[str, float]:
        """Run the full critic GRPO training loop.

        Args:
            train_dataset: List of training sample dicts
            val_dataset: Optional list of validation sample dicts

        Returns:
            Dict of final training metrics
        """

        config = self.config
        batch_size = config.rollout.batch_size
        num_epochs = config.num_train_epochs
        grad_acc_steps = config.gradient_accumulation_steps
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_steps = (
            math.ceil(len(train_dataset) / batch_size) * num_epochs
        ) // grad_acc_steps
        logger.info(f"Starting critic GRPO training: {total_steps} optimization steps")
        logger.info(f"  Epochs: {num_epochs}, Batch size: {batch_size}")
        logger.info(f"  Gradient accumulation: {grad_acc_steps}")
        logger.info(f"  Dataset size: {len(train_dataset)}")

        all_metrics = {}
        should_stop = False

        for epoch in range(num_epochs):
            if should_stop:
                break

            logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")
            epoch_loss = 0.0
            epoch_steps = 0

            # Process in batches
            for batch_start in range(0, len(train_dataset), batch_size):
                batch = train_dataset[batch_start : batch_start + batch_size]

                step_result = self.train_step(batch)
                epoch_loss += step_result.loss
                epoch_steps += 1

                # Log
                if self.global_step % config.logging_steps == 0:
                    logger.info(
                        f"Step {self.global_step}: loss={step_result.loss:.4f}, "
                        f"reward_mean={step_result.reward_mean:.3f}, "
                        f"rw_rate={step_result.rollout_metrics.get('rollout/rw_rate', 0):.3f}"
                    )

                # Validate
                if (
                    val_dataset
                    and config.val_check_interval > 0
                    and self.global_step % config.val_check_interval == 0
                    and self.global_step > 0
                ):
                    val_metrics = self.validate(val_dataset)
                    all_metrics.update(val_metrics)
                    logger.info(f"Validation: {val_metrics}")

                    # Early stopping
                    should_stop = self._check_early_stopping(val_metrics)
                    if should_stop:
                        logger.info("Early stopping triggered!")
                        break

                # Save checkpoint
                if (
                    config.save_steps > 0
                    and self.global_step % config.save_steps == 0
                    and self.global_step > 0
                ):
                    self._save_checkpoint(output_dir / f"checkpoint-{self.global_step}")

            avg_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # Final save
        self._save_checkpoint(output_dir / "final")
        logger.info(f"Training complete. Model saved to {output_dir / 'final'}")

        return all_metrics

    def train_step(self, batch: list[dict]) -> CriticTrainStepResult:
        """Execute a single critic GRPO training step.

        Steps:
        1. ROLLOUT: Generate K F1s and A2s for each sample
        2. REWARD: Compute downstream-aware critic rewards
        3. ADVANTAGE: Group-relative normalization
        4. LOSS: Clipped policy gradient + KL
        5. BACKWARD + STEP

        Args:
            batch: List of sample dicts for this batch

        Returns:
            CriticTrainStepResult with loss and metrics
        """
        import torch

        self.model.train()

        # Step 1: Rollout (generates F1+A2, computes rewards)
        rollout_results = self.rollout_engine.generate_critic_rollout(batch)
        rollout_metrics = compute_rollout_metrics(rollout_results)

        # Collect all (prompt, completion, reward) triples
        all_rewards = []
        all_log_probs = []
        all_ref_log_probs = []

        for result in rollout_results:
            k = len(result.feedbacks)
            if k == 0:
                continue

            rewards_group = result.rewards
            all_rewards.extend(rewards_group)

            # Compute log probabilities for each feedback
            for f1 in result.feedbacks:
                log_prob = self._compute_log_prob(
                    question=result.question,
                    answer1=result.answer1,
                    answer_type=result.answer_type,
                    choices=result.choices,
                    completion=f1,
                    image=batch[0].get("image") if batch else None,
                    model=self.model,
                )
                ref_log_prob = self._compute_log_prob(
                    question=result.question,
                    answer1=result.answer1,
                    answer_type=result.answer_type,
                    choices=result.choices,
                    completion=f1,
                    image=batch[0].get("image") if batch else None,
                    model=self.ref_model,
                )
                all_log_probs.append(log_prob)
                all_ref_log_probs.append(ref_log_prob)

        if not all_rewards:
            self.global_step += 1
            return CriticTrainStepResult(
                loss=0.0, policy_loss=0.0, kl_loss=0.0,
                reward_mean=0.0, reward_std=0.0,
                rollout_metrics=rollout_metrics,
                global_step=self.global_step,
            )

        # Step 3: Group-relative advantage normalization
        rewards_tensor = torch.tensor(all_rewards, device=self.device)
        advantages = self._compute_group_advantages(
            rewards_tensor, self.config.rollout.k_samples
        )

        # Step 4: Compute GRPO loss
        log_probs = torch.stack(all_log_probs)
        ref_log_probs = torch.stack(all_ref_log_probs)

        # Policy ratio: r(θ) = exp(log π_θ - log π_old)
        # For GRPO, π_old is from the rollout (same model), so ratio starts at 1.0
        # We use the current log_probs vs detached rollout log_probs
        ratio = torch.exp(log_probs - log_probs.detach())

        # Clipped surrogate loss
        clip_range = self.config.clip_range
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL divergence loss
        kl = (log_probs - ref_log_probs).mean()
        kl_loss = self.config.kl_coeff * kl

        # Total loss
        loss = policy_loss + kl_loss

        # Step 5: Backward + step
        loss.backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.scheduler is not None:
                self.scheduler.step()

        self.global_step += 1

        return CriticTrainStepResult(
            loss=loss.item(),
            policy_loss=policy_loss.item(),
            kl_loss=kl_loss.item(),
            reward_mean=rewards_tensor.mean().item(),
            reward_std=rewards_tensor.std().item(),
            rollout_metrics=rollout_metrics,
            global_step=self.global_step,
        )

    def validate(self, val_dataset: list[dict]) -> dict[str, float]:
        """Run validation and return metrics.

        Args:
            val_dataset: List of validation sample dicts

        Returns:
            Dict of validation metrics
        """
        import torch

        self.model.eval()
        with torch.no_grad():
            rollout_results = self.rollout_engine.generate_critic_rollout(val_dataset)

        metrics = compute_rollout_metrics(rollout_results)
        # Prefix with "val/"
        val_metrics = {f"val/{k.split('/')[-1]}": v for k, v in metrics.items()}

        self.model.train()
        return val_metrics

    def _compute_log_prob(
        self,
        question: str,
        answer1: str,
        answer_type: str,
        choices: str,
        completion: str,
        image: Any,
        model: Any,
    ) -> Any:
        """Compute log probability of a completion under a model.

        Args:
            question: Visual question
            answer1: Initial answer
            answer_type: Answer type
            choices: MCQ choices
            completion: The feedback text to score
            image: PIL Image
            model: The model to compute log probs with

        Returns:
            Scalar tensor of total log probability
        """
        import torch

        messages = build_critic_prompt_with_completion(
            question, answer1, answer_type, choices, completion
        )

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        if image is not None:
            inputs = self.processor(
                text=text, images=image, return_tensors="pt"
            ).to(self.device)
        else:
            inputs = self.processor(
                text=text, return_tensors="pt"
            ).to(self.device)

        with torch.no_grad() if model is self.ref_model else torch.enable_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        # outputs.loss is cross-entropy averaged over tokens
        # Convert to total log prob: -loss * num_tokens
        num_tokens = inputs["input_ids"].shape[1]
        total_log_prob = -outputs.loss * num_tokens

        return total_log_prob

    def _compute_group_advantages(
        self,
        rewards: Any,
        k: int,
    ) -> Any:
        """Compute group-relative advantages for GRPO.

        Normalizes rewards within each group of K samples.

        Args:
            rewards: Tensor of all rewards (shape: [N * K])
            k: Group size

        Returns:
            Tensor of advantages (same shape as rewards)
        """
        import torch

        n_groups = len(rewards) // k
        advantages = torch.zeros_like(rewards)

        for i in range(n_groups):
            start = i * k
            end = start + k
            group = rewards[start:end]

            mean = group.mean()
            std = group.std()
            if std > 0:
                advantages[start:end] = (group - mean) / (std + 1e-8)
            else:
                advantages[start:end] = 0.0

        # Handle remaining samples (if not divisible by k)
        remaining = len(rewards) - n_groups * k
        if remaining > 0:
            start = n_groups * k
            group = rewards[start:]
            mean = group.mean()
            std = group.std()
            if std > 0:
                advantages[start:] = (group - mean) / (std + 1e-8)

        return advantages

    def _check_early_stopping(self, val_metrics: dict[str, float]) -> bool:
        """Check if training should stop based on validation metrics.

        Args:
            val_metrics: Dict of validation metrics

        Returns:
            True if training should stop
        """
        es_config = self.config.early_stopping
        metric_key = es_config.metric
        # Map metric key to val format
        val_key = f"val/{metric_key.split('/')[-1]}"

        if val_key not in val_metrics:
            return False

        current = val_metrics[val_key]

        if es_config.mode == "min":
            improved = current < (self.best_metric - es_config.min_delta)
        else:
            improved = current > (self.best_metric + es_config.min_delta)

        if improved:
            self.best_metric = current
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= es_config.patience

    def _save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint.

        Args:
            path: Directory to save to
        """
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

        # Save config
        config_path = path / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Saved checkpoint to {path}")


def build_critic_prompt_with_completion(
    question: str,
    answer1: str,
    answer_type: str,
    choices: str,
    completion: str,
) -> list[dict]:
    """Build critic prompt with completion appended for log prob computation.

    Args:
        question: Visual question
        answer1: Initial answer
        answer_type: Answer type
        choices: MCQ choices
        completion: The feedback text

    Returns:
        Full message list with assistant completion
    """
    from vlm_grpo.prompts import CRITIC_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
        {
            "role": "assistant",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
        {"role": "user", "content": answer1},
        {"role": "assistant", "content": completion},
    ]
    return messages
