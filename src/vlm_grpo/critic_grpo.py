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

from tqdm import tqdm

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

        total_steps = (math.ceil(len(train_dataset) / batch_size) * num_epochs) // grad_acc_steps
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
        all_old_log_probs = []
        all_ref_log_probs = []

        for result in rollout_results:
            k = len(result.feedbacks)
            if k == 0:
                continue

            rewards_group = result.rewards
            all_rewards.extend(rewards_group)

            sample_image = (
                result.image
                if hasattr(result, "image")
                else (batch[0].get("image") if batch else None)
            )

            # Compute log probabilities for each feedback
            for f1 in result.feedbacks:
                kwargs = dict(
                    question=result.question,
                    answer1=result.answer1,
                    answer_type=result.answer_type,
                    choices=result.choices,
                    completion=f1,
                    image=sample_image,
                )
                # π_old: log prob under current model weights, no grad (rollout snapshot)
                with torch.no_grad():
                    old_log_prob = self._compute_log_prob(**kwargs, model=self.model)
                # π_θ: log prob under current model weights, with grad (for backprop)
                log_prob = self._compute_log_prob(**kwargs, model=self.model)
                # π_ref: log prob under frozen reference model
                with torch.no_grad():
                    ref_log_prob = self._compute_log_prob(**kwargs, model=self.ref_model)

                all_log_probs.append(log_prob)
                all_old_log_probs.append(old_log_prob)
                all_ref_log_probs.append(ref_log_prob)

        if not all_rewards:
            self.global_step += 1
            return CriticTrainStepResult(
                loss=0.0,
                policy_loss=0.0,
                kl_loss=0.0,
                reward_mean=0.0,
                reward_std=0.0,
                rollout_metrics=rollout_metrics,
                global_step=self.global_step,
            )

        # Step 3: Group-relative advantage normalization
        rewards_tensor = torch.tensor(all_rewards, device=self.device)
        loss_type = getattr(self.config, "loss_type", "grpo")
        advantages = self._compute_group_advantages(
            rewards_tensor, self.config.rollout.k_samples, loss_type=loss_type
        )

        # Step 4: Compute GRPO loss
        log_probs = torch.stack(all_log_probs)
        old_log_probs = torch.stack(all_old_log_probs)
        ref_log_probs = torch.stack(all_ref_log_probs)

        # Policy ratio: r(θ) = π_θ(F1) / π_old(F1)
        # π_old is the no-grad snapshot from before this backward pass
        ratio = torch.exp(log_probs - old_log_probs)

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

        messages = build_critic_prompt_with_completion(
            question, answer1, answer_type, choices, completion
        )

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Re-inject <image> if template stripped it from non-user roles
        if image is not None and "<image>" not in text:
            text = "<image>\n" + text

        if image is not None:
            inputs = self.processor(text=text, images=image, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)

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
        loss_type: str = "grpo",
    ) -> Any:
        """Compute group-relative advantages for GRPO.

        Normalizes rewards within each group of K samples.

        Args:
            rewards: Tensor of all rewards (shape: [N * K])
            k: Group size
            loss_type: GRPO variant. "grpo" uses std normalization,
                "dr_grpo" uses only mean subtraction (removes length
                and difficulty bias per arXiv:2503.20783).

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
            if loss_type == "dr_grpo":
                # Dr. GRPO: no std normalization (removes difficulty bias)
                advantages[start:end] = group - mean
            else:
                # Vanilla GRPO: normalize by std
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
            if loss_type == "dr_grpo":
                advantages[start:] = group - mean
            else:
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
        {
            "role": "system",
            "content": [
                {"type": "image"},
                {"type": "text", "text": CRITIC_SYSTEM_PROMPT},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": question}]},
        {"role": "user", "content": [{"type": "text", "text": answer1}]},
        {"role": "assistant", "content": [{"type": "text", "text": completion}]},
    ]
    return messages


# =============================================================================
# Full Self-Reflection GRPO Trainer (two-reward, single model)
# =============================================================================


@dataclass
class SelfReflectionTrainStepResult:
    """Result of a single self-reflection training step.

    Attributes:
        loss: Total loss (response + feedback)
        response_loss: Clipped policy gradient loss for A1+A2
        feedback_loss: Clipped policy gradient loss for F1
        kl_loss: KL divergence from reference model
        response_reward_mean: Mean response reward across batch
        feedback_reward_mean: Mean feedback reward across batch
        rollout_metrics: Dict of rollout statistics
        global_step: Current global training step
    """

    loss: float
    response_loss: float
    feedback_loss: float
    kl_loss: float
    response_reward_mean: float
    feedback_reward_mean: float
    rollout_metrics: dict[str, float]
    global_step: int

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return asdict(self)


class SelfReflectionGRPOTrainer:
    """GRPO trainer for full self-reflection with two separate reward signals.

    Single model generates A1 -> F1 -> A2. Two GRPO losses:
    1. Response loss: advantage from response reward, applied to
       log_prob(A1) + log_prob(A2)
    2. Feedback loss: advantage from feedback reward, applied to
       log_prob(F1)

    Both losses backprop into the same model weights (shared LoRA).
    """

    def __init__(
        self,
        model: Any,
        ref_model: Optional[Any],
        processor: Any,
        config: Any,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        accelerator: Optional[Any] = None,
        vllm_engine: Optional[Any] = None,
    ) -> None:
        """Initialize the self-reflection GRPO trainer.

        Uses lazy imports for heavy ML libraries.

        If ref_model is None and the policy model is a PEFT model, ref
        log-probs are computed by disabling the LoRA adapter (zero GPU
        overhead). A separate ref_model is only needed for non-PEFT runs.

        Args:
            model: The policy model (with LoRA adapter)
            ref_model: Frozen reference model for KL, or None to use adapter disable
            processor: Tokenizer/processor for the model
            config: SelfReflectionConfig
            optimizer: Optional pre-configured optimizer
            scheduler: Optional learning rate scheduler
            accelerator: Optional HuggingFace Accelerator for distributed training
            vllm_engine: Optional VLLMRolloutEngine for faster generation
        """
        from torch.optim import AdamW

        self.model = model
        self.ref_model = ref_model
        self.processor = processor
        self.config = config
        self.accelerator = accelerator
        self.vllm_engine = vllm_engine

        if accelerator is not None:
            self.device = accelerator.device
        else:
            self.device = next(model.parameters()).device

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
        """Run the full self-reflection GRPO training loop.

        Args:
            train_dataset: List of training sample dicts
            val_dataset: Optional list of validation sample dicts

        Returns:
            Dict of final training metrics
        """

        config = self.config
        batch_size = config.rollout.batch_size
        num_epochs = config.num_train_epochs
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        is_main = self.accelerator is None or self.accelerator.is_main_process

        total_steps = math.ceil(len(train_dataset) / batch_size) * num_epochs
        logger.info(f"Starting self-reflection GRPO training: {total_steps} rollout steps")
        logger.info(f"  Epochs: {num_epochs}, Batch size: {batch_size}")
        logger.info(f"  Inner optimization epochs per step: {config.num_inner_epochs}")
        logger.info(f"  Dataset size: {len(train_dataset)}")
        logger.info("  Two-reward design: response (A1+A2) + feedback (F1)")

        all_metrics = {}
        should_stop = False

        total_samples = len(train_dataset) * num_epochs
        pbar = tqdm(
            total=total_samples,
            desc="Training",
            unit="sample",
            disable=not is_main,
            dynamic_ncols=True,
            file=sys.stdout,  # same stream as logger so it shows in k8s logs
            mininterval=30.0,  # don't flood logs; print at most every 30s
        )

        for epoch in range(num_epochs):
            if should_stop:
                break

            logger.info(f"=== Epoch {epoch + 1}/{num_epochs} ===")
            epoch_loss = 0.0
            epoch_steps = 0

            for batch_start in range(0, len(train_dataset), batch_size):
                batch = train_dataset[batch_start : batch_start + batch_size]

                # Load images for the batch
                from vlm_grpo.data import load_image_safe

                for sample in batch:
                    if "image" not in sample:
                        sample["image"] = load_image_safe(sample["image_path"])

                step_result = self.train_step(batch)
                epoch_loss += step_result.loss
                epoch_steps += 1

                rw_rate = step_result.rollout_metrics.get("sr/rw_rate", 0)
                pbar.set_postfix(
                    epoch=f"{epoch + 1}/{num_epochs}",
                    loss=f"{step_result.loss:.4f}",
                    resp_r=f"{step_result.response_reward_mean:.3f}",
                    fb_r=f"{step_result.feedback_reward_mean:.3f}",
                    rw_rate=f"{rw_rate:.3f}",
                )
                pbar.update(len(batch))

                if self.global_step % config.logging_steps == 0:
                    logger.info(
                        f"Step {self.global_step}: loss={step_result.loss:.4f}, "
                        f"resp_loss={step_result.response_loss:.4f}, "
                        f"fb_loss={step_result.feedback_loss:.4f}, "
                        f"resp_reward={step_result.response_reward_mean:.3f}, "
                        f"fb_reward={step_result.feedback_reward_mean:.3f}, "
                        f"rw_rate={rw_rate:.3f}"
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

                    should_stop = self._check_early_stopping(val_metrics)
                    if should_stop:
                        logger.info("Early stopping triggered!")
                        break

                # Save checkpoint (main process only)
                if (
                    config.save_steps > 0
                    and self.global_step % config.save_steps == 0
                    and self.global_step > 0
                    and is_main
                ):
                    self._save_checkpoint(output_dir / f"checkpoint-{self.global_step}")

            avg_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        pbar.close()

        # Final save (main process only)
        if is_main:
            self._save_checkpoint(output_dir / "final")
        logger.info(f"Training complete. Model saved to {output_dir / 'final'}")

        return all_metrics

    def train_step(self, batch: list[dict]) -> SelfReflectionTrainStepResult:
        """Execute a single self-reflection GRPO training step.

        Steps:
        1. ROLLOUT: Generate K (A1, F1, A2) trajectories per sample
        2. REWARD: Compute response rewards and feedback rewards separately
        3. ADVANTAGE: Group-relative normalization for each reward type
        4. OLD/REF LOG-PROBS: Compute once (frozen)
        5. INNER EPOCHS: Recompute log_probs mu times, each time
           computing clipped loss, backward, and optimizer step.
           After each step the model weights change so ratio diverges
           from 1.0 and the clipping mechanism activates.

        Args:
            batch: List of sample dicts for this batch

        Returns:
            SelfReflectionTrainStepResult with loss and metrics
        """
        import torch

        from vlm_grpo.prompts import (
            build_critic_prompt,
            build_initial_answer_prompt,
            build_prompt_with_completion,
            build_refiner_prompt,
        )
        from vlm_grpo.rollout import (
            compute_self_reflection_metrics,
            generate_self_reflection_rollout,
        )

        self.model.train()

        # Step 1: Rollout (generates A1->F1->A2, computes rewards)
        # Use unwrapped model for generation (DDP wrapper lacks .generate())
        gen_model = (
            self.accelerator.unwrap_model(self.model)
            if self.accelerator is not None
            else self.model
        )

        # Disable gradient checkpointing during generation — it forces
        # use_cache=False which breaks autoregressive KV caching and
        # produces garbage output.
        had_grad_ckpt = gen_model.is_gradient_checkpointing
        if had_grad_ckpt:
            gen_model.gradient_checkpointing_disable()

        model_type = getattr(self.config, "model_type", "llava")

        # vLLM sleep/wake cycle: wake → sync weights → generate → sleep.
        # The refcount leak (~1000 cycles) is handled by frequent
        # checkpointing (every 500 steps) + auto-resume on crash.
        if self.vllm_engine is not None:
            self.vllm_engine.wake_up()
            self.vllm_engine.update_weights_from_peft(gen_model, accelerator=self.accelerator)

        rollout_results = generate_self_reflection_rollout(
            model=gen_model,
            processor=self.processor,
            samples=batch,
            config=self.config.rollout,
            response_weights=self.config.response_weights,
            feedback_weights=self.config.feedback_weights,
            device=str(self.device),
            model_type=model_type,
            vllm_engine=self.vllm_engine,
        )
        rollout_metrics = compute_self_reflection_metrics(rollout_results)

        # Put vLLM to sleep to free GPU memory for training
        if self.vllm_engine is not None:
            self.vllm_engine.sleep()

        # Re-enable gradient checkpointing for the training forward passes
        if had_grad_ckpt:
            gen_model.gradient_checkpointing_enable()

        # Step 2: Collect trajectory data (prompts, images, rewards)
        trajectory_data = []
        all_resp_rewards = []
        all_fb_rewards = []

        for result in rollout_results:
            k = len(result.answer1s)
            if k == 0:
                continue

            all_resp_rewards.extend(result.response_rewards)
            all_fb_rewards.extend(result.feedback_rewards)

            # Debug: print generated trajectories and reward breakdowns
            if self.config.debug:
                logger.info("=" * 80)
                logger.info(
                    f"[DEBUG] Sample: question='{result.question[:80]}...' "
                    f"GT={result.ground_truth} type={result.answer_type}"
                )
                for j in range(k):
                    logger.info(f"  --- Trajectory {j + 1}/{k} ---")
                    logger.info(f"  A1: {result.answer1s[j]}")
                    logger.info(f"  F1: {result.feedbacks[j]}")
                    logger.info(f"  A2: {result.answer2s[j]}")
                    logger.info(
                        f"  resp_reward={result.response_rewards[j]:.3f} "
                        f"fb_reward={result.feedback_rewards[j]:.3f}"
                    )
                    if result.response_breakdowns:
                        rb = result.response_breakdowns[j]
                        logger.info(f"  resp_components: {rb.components}")
                        logger.info(
                            f"  a1_correct={rb.a1_correct} a2_correct={rb.a2_correct} "
                            f"a2_extracted='{rb.a2_extracted}'"
                        )
                    if result.feedback_breakdowns:
                        fb = result.feedback_breakdowns[j]
                        logger.info(f"  fb_components: {fb.components}")
                logger.info("=" * 80)

            image = None
            for s in batch:
                if s.get("question", "").replace("<image>", "").strip() == result.question:
                    image = s.get("image")
                    break

            for a1, f1, a2 in zip(result.answer1s, result.feedbacks, result.answer2s):
                a1_prompt = build_initial_answer_prompt(result.question)
                a1_full = build_prompt_with_completion(a1_prompt, a1)

                f1_prompt = build_critic_prompt(result.question, a1, model_type=model_type)
                f1_full = build_prompt_with_completion(f1_prompt, f1)

                a2_prompt = build_refiner_prompt(result.question, a1, f1)
                a2_full = build_prompt_with_completion(a2_prompt, a2)

                trajectory_data.append(
                    {"a1_full": a1_full, "f1_full": f1_full, "a2_full": a2_full, "image": image}
                )

        if not all_resp_rewards:
            self.global_step += 1
            return SelfReflectionTrainStepResult(
                loss=0.0,
                response_loss=0.0,
                feedback_loss=0.0,
                kl_loss=0.0,
                response_reward_mean=0.0,
                feedback_reward_mean=0.0,
                rollout_metrics=rollout_metrics,
                global_step=self.global_step,
            )

        k = self.config.rollout.k_samples

        # Step 3: Group-relative advantages (computed once, frozen)
        resp_rewards_t = torch.tensor(all_resp_rewards, device=self.device)
        fb_rewards_t = torch.tensor(all_fb_rewards, device=self.device)
        loss_type = getattr(self.config, "loss_type", "grpo")
        resp_advantages = self._compute_group_advantages(resp_rewards_t, k, loss_type=loss_type)
        fb_advantages = self._compute_group_advantages(fb_rewards_t, k, loss_type=loss_type)

        # Compute frac_reward_zero_std for BOTH response and feedback rewards:
        # fraction of K-groups where all trajectories got identical rewards
        # (zero variance → zero learning signal). Key TRL diagnostic metric.
        n_groups = len(resp_rewards_t) // k
        n_resp_zero = 0
        n_fb_zero = 0
        for gi in range(n_groups):
            s, e = gi * k, (gi + 1) * k
            if resp_rewards_t[s:e].std().item() < 1e-4:
                n_resp_zero += 1
            if fb_rewards_t[s:e].std().item() < 1e-4:
                n_fb_zero += 1
        frac_resp_zero_std = n_resp_zero / max(n_groups, 1)
        frac_fb_zero_std = n_fb_zero / max(n_groups, 1)

        # Step 4: Pre-tokenize once, then compute old/ref log-probs in mini-batches.
        #
        # Pre-tokenization caches apply_chat_template strings and prompt/full
        # token lengths so they are NOT recomputed on every inner epoch call.
        # Mini-batching (inner_mini_bs) prevents OOM from large logits tensors
        # (64 × 1000 × 32000 × 2 bytes ≈ 4 GB for a full-batch forward pass).
        unwrapped_model = (
            self.accelerator.unwrap_model(self.model)
            if self.accelerator is not None
            else self.model
        )
        n_traj = len(trajectory_data)
        inner_mini_bs = self.config.inner_mini_batch_size
        logger.info(f"  Pre-tokenizing {n_traj} trajectories (mini_bs={inner_mini_bs})...")

        imgs = [t["image"] for t in trajectory_data]
        a1_pretok = self._preprocess_trajectory_texts([t["a1_full"] for t in trajectory_data], imgs)
        a2_pretok = self._preprocess_trajectory_texts([t["a2_full"] for t in trajectory_data], imgs)
        f1_pretok = self._preprocess_trajectory_texts([t["f1_full"] for t in trajectory_data], imgs)

        # Completion token lengths from pre-tokenized data (token count, not word count).
        # full_lens - prompt_lens = number of completion tokens per trajectory.
        # Tracks length hacking where wrong answers grow longer (Dr. GRPO key finding).
        a1_toks = [f - p for f, p in zip(a1_pretok["full_lens"], a1_pretok["prompt_lens"])]
        f1_toks = [f - p for f, p in zip(f1_pretok["full_lens"], f1_pretok["prompt_lens"])]
        a2_toks = [f - p for f, p in zip(a2_pretok["full_lens"], a2_pretok["prompt_lens"])]
        avg_a1_toks = sum(a1_toks) / max(len(a1_toks), 1)
        avg_f1_toks = sum(f1_toks) / max(len(f1_toks), 1)
        avg_a2_toks = sum(a2_toks) / max(len(a2_toks), 1)

        logger.info(f"  Computing old/ref log-probs for {n_traj} trajectories...")

        old_a1_lps_list: list[Any] = []
        old_a2_lps_list: list[Any] = []
        old_fb_lps_list: list[Any] = []
        ref_a1_lps_list: list[Any] = []
        ref_a2_lps_list: list[Any] = []
        ref_fb_lps_list: list[Any] = []

        with torch.no_grad():
            # Combine a1/a2/f1 into one forward pass per mini-batch (3× fewer GPU ops)
            for mb_s in range(0, n_traj, inner_mini_bs):
                mb_e = min(mb_s + inner_mini_bs, n_traj)
                a1_lps, a2_lps, fb_lps = self._forward_from_pretokenized_multi(
                    [a1_pretok, a2_pretok, f1_pretok], unwrapped_model, mb_s, mb_e
                )
                old_a1_lps_list += a1_lps
                old_a2_lps_list += a2_lps
                old_fb_lps_list += fb_lps

            # Ref log-probs: only needed when kl_coeff > 0
            if self.config.kl_coeff > 0:
                if self.ref_model is None:
                    unwrapped_model.disable_adapter_layers()
                    ref_m = unwrapped_model
                else:
                    ref_m = self.ref_model

                for mb_s in range(0, n_traj, inner_mini_bs):
                    mb_e = min(mb_s + inner_mini_bs, n_traj)
                    a1_lps, a2_lps, fb_lps = self._forward_from_pretokenized_multi(
                        [a1_pretok, a2_pretok, f1_pretok], ref_m, mb_s, mb_e
                    )
                    ref_a1_lps_list += a1_lps
                    ref_a2_lps_list += a2_lps
                    ref_fb_lps_list += fb_lps

                if self.ref_model is None:
                    unwrapped_model.enable_adapter_layers()

            # Per-token log-probs: keep as lists of variable-length tensors.
            # For response, concatenate A1 and A2 tokens per trajectory.
            old_resp_lps = [torch.cat([a1, a2]) for a1, a2 in zip(old_a1_lps_list, old_a2_lps_list)]
            old_fb_lps = old_fb_lps_list  # already list of per-token tensors

            if self.config.kl_coeff > 0:
                ref_resp_lps = [
                    torch.cat([a1, a2]) for a1, a2 in zip(ref_a1_lps_list, ref_a2_lps_list)
                ]
                ref_fb_lps = ref_fb_lps_list
            else:
                ref_resp_lps = None
                ref_fb_lps = None

        # No explicit cache clear — PyTorch reuses memory automatically.

        clip_range = self.config.clip_range
        num_inner = self.config.num_inner_epochs

        # Step 5: Inner optimization epochs — use pre-tokenized data to skip
        # redundant apply_chat_template / tokenizer calls on every pass.
        total_resp_loss = 0.0
        total_fb_loss = 0.0
        total_kl_loss = 0.0
        n_traj_inner = n_traj

        # Reset entropy accumulators — token-weighted average across all
        # training mini-batches (not old/ref passes).
        self._entropy_sum = 0.0
        self._entropy_tokens = 0

        for inner_epoch in range(num_inner):
            self.optimizer.zero_grad()
            epoch_resp_loss = 0.0
            epoch_fb_loss = 0.0
            epoch_kl_loss = 0.0

            # Use unwrapped model to avoid DDP hooks firing per mini-batch.
            # Gradients still flow to the underlying parameters; we
            # manually all-reduce after all mini-batches.
            inner_model = (
                self.accelerator.unwrap_model(self.model)
                if self.accelerator is not None
                else self.model
            )

            # 3 batched forward passes per mini-batch (a1, a2, f1), reusing
            # pre-tokenized strings so no chat-template overhead per pass.
            for mb_start in range(0, n_traj_inner, inner_mini_bs):
                mb_end = min(mb_start + inner_mini_bs, n_traj_inner)

                mb_a1_lps, mb_a2_lps, mb_fb_lps = self._forward_from_pretokenized_multi(
                    [a1_pretok, a2_pretok, f1_pretok],
                    inner_model,
                    mb_start,
                    mb_end,
                    accumulate_entropy=True,
                )

                # Token-level GRPO loss (standard formulation).
                # Per-token ratios produce non-zero gradients even with
                # inner_epochs=1 because each token's gradient flows through
                # its own path. The scalar advantage broadcasts to all tokens
                # of a trajectory (outcome supervision).
                # References:
                #   - DeepSeekMath GRPO: arXiv:2402.03300
                #   - TRL GRPOTrainer: github.com/huggingface/trl
                #   - DAPO token-level loss: arXiv:2503.14476
                mb_loss = None
                for j in range(mb_end - mb_start):
                    ti = mb_start + j

                    # Per-token log-probs: cat A1+A2 for response
                    resp_new = torch.cat([mb_a1_lps[j], mb_a2_lps[j]])
                    fb_new = mb_fb_lps[j]

                    # Per-token ratios
                    resp_ratio_raw = torch.nan_to_num(
                        resp_new - old_resp_lps[ti].detach(),
                        nan=0.0,
                        posinf=20.0,
                        neginf=-20.0,
                    )
                    fb_ratio_raw = torch.nan_to_num(
                        fb_new - old_fb_lps[ti].detach(),
                        nan=0.0,
                        posinf=20.0,
                        neginf=-20.0,
                    )

                    # Response: per-token clipped surrogate, advantage broadcasts
                    resp_log_ratio = torch.clamp(resp_ratio_raw, min=-20.0, max=20.0)
                    resp_ratio = torch.exp(resp_log_ratio)
                    resp_surr1 = resp_ratio * resp_advantages[ti]
                    resp_surr2 = (
                        torch.clamp(resp_ratio, 1 - clip_range, 1 + clip_range)
                        * resp_advantages[ti]
                    )
                    traj_resp_loss = -torch.min(resp_surr1, resp_surr2).mean()

                    # Feedback: per-token clipped surrogate
                    fb_log_ratio = torch.clamp(fb_ratio_raw, min=-20.0, max=20.0)
                    fb_ratio = torch.exp(fb_log_ratio)
                    fb_surr1 = fb_ratio * fb_advantages[ti]
                    fb_surr2 = (
                        torch.clamp(fb_ratio, 1 - clip_range, 1 + clip_range) * fb_advantages[ti]
                    )
                    traj_fb_loss = -torch.min(fb_surr1, fb_surr2).mean()

                    # KL loss (per-token, averaged)
                    if self.config.kl_coeff > 0 and ref_resp_lps is not None:
                        resp_kl_raw = torch.nan_to_num(
                            ref_resp_lps[ti].detach() - resp_new,
                            nan=0.0,
                            posinf=20.0,
                            neginf=-20.0,
                        )
                        resp_kl = (
                            torch.exp(torch.clamp(resp_kl_raw, -20.0, 20.0))
                            - torch.clamp(resp_kl_raw, -20.0, 20.0)
                            - 1
                        ).mean()
                        fb_kl_raw = torch.nan_to_num(
                            ref_fb_lps[ti].detach() - fb_new,
                            nan=0.0,
                            posinf=20.0,
                            neginf=-20.0,
                        )
                        fb_kl = (
                            torch.exp(torch.clamp(fb_kl_raw, -20.0, 20.0))
                            - torch.clamp(fb_kl_raw, -20.0, 20.0)
                            - 1
                        ).mean()
                        traj_kl_loss = self.config.kl_coeff * (resp_kl + fb_kl) / 2.0
                    else:
                        traj_kl_loss = torch.tensor(0.0, device=resp_new.device)

                    traj_loss = (traj_resp_loss + traj_fb_loss + traj_kl_loss) / n_traj_inner
                    mb_loss = traj_loss if mb_loss is None else mb_loss + traj_loss

                    epoch_resp_loss += traj_resp_loss.item() / n_traj_inner
                    epoch_fb_loss += traj_fb_loss.item() / n_traj_inner
                    epoch_kl_loss += traj_kl_loss.item() / n_traj_inner

                if mb_loss is not None:
                    mb_loss.backward()

            # Manually all-reduce gradients across DDP processes
            if self.accelerator is not None and torch.distributed.is_initialized():
                for param in inner_model.parameters():
                    if param.grad is not None:
                        torch.distributed.all_reduce(
                            param.grad,
                            op=torch.distributed.ReduceOp.AVG,
                        )

            # Clip grad norm and log it to verify gradients are flowing.
            # With GRPO inner_epochs=1 and kl=0, loss is mathematically 0
            # but gradients are non-zero (REINFORCE gradient estimator).
            # Confirmed by TRL issue #3452: loss=0 with grad_norm>0 is expected.
            grad_norm = torch.nn.utils.clip_grad_norm_(inner_model.parameters(), 1.0).item()

            # Skip optimizer step if any gradient contains NaN/inf
            has_nan_grad = not math.isfinite(grad_norm)
            if has_nan_grad:
                logger.warning(
                    f"  [inner epoch {inner_epoch + 1}] NaN/inf gradient detected — "
                    "skipping optimizer step to preserve model weights"
                )
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()

            total_resp_loss += epoch_resp_loss
            total_fb_loss += epoch_fb_loss
            total_kl_loss += epoch_kl_loss

        # Log advantage statistics to detect the std=0 problem
        # (all K trajectories getting identical rewards → zero advantages → zero gradients)
        resp_adv_abs_mean = resp_advantages.abs().mean().item()
        fb_adv_abs_mean = fb_advantages.abs().mean().item()
        n_zero_adv = (resp_advantages == 0).sum().item()
        n_zero_fb_adv = (fb_advantages == 0).sum().item()

        # Token-weighted entropy from training forward passes (accumulated
        # across all inner-loop mini-batches, NOT from old/ref log-prob passes).
        entropy = self._entropy_sum / self._entropy_tokens if self._entropy_tokens > 0 else 0.0

        logger.info(
            f"  Inner epochs done: resp_loss={total_resp_loss / num_inner:.4f}, "
            f"fb_loss={total_fb_loss / num_inner:.4f}, "
            f"grad_norm={grad_norm:.4f}, "
            f"resp_adv={resp_adv_abs_mean:.4f}(z={n_zero_adv}/{len(resp_advantages)}), "
            f"fb_adv={fb_adv_abs_mean:.4f}(z={n_zero_fb_adv}/{len(fb_advantages)}), "
            f"entropy={entropy:.3f}, "
            f"frac_zero_std={frac_resp_zero_std:.2f}/{frac_fb_zero_std:.2f}, "
            f"tok={avg_a1_toks:.0f}/{avg_f1_toks:.0f}/{avg_a2_toks:.0f}"
        )

        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

        return SelfReflectionTrainStepResult(
            loss=(total_resp_loss + total_fb_loss + total_kl_loss) / num_inner,
            response_loss=total_resp_loss / num_inner,
            feedback_loss=total_fb_loss / num_inner,
            kl_loss=total_kl_loss / num_inner,
            response_reward_mean=resp_rewards_t.mean().item(),
            feedback_reward_mean=fb_rewards_t.mean().item(),
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

        from vlm_grpo.data import load_image_safe
        from vlm_grpo.rollout import (
            compute_self_reflection_metrics,
            generate_self_reflection_rollout,
        )

        self.model.eval()

        # Load images for validation
        for sample in val_dataset:
            if "image" not in sample:
                sample["image"] = load_image_safe(sample["image_path"])

        gen_model = (
            self.accelerator.unwrap_model(self.model)
            if self.accelerator is not None
            else self.model
        )

        # Disable gradient checkpointing for generation (needs KV cache)
        had_grad_ckpt = gen_model.is_gradient_checkpointing
        if had_grad_ckpt:
            gen_model.gradient_checkpointing_disable()

        model_type = getattr(self.config, "model_type", "llava")
        with torch.no_grad():
            rollout_results = generate_self_reflection_rollout(
                model=gen_model,
                processor=self.processor,
                samples=val_dataset,
                config=self.config.rollout,
                response_weights=self.config.response_weights,
                feedback_weights=self.config.feedback_weights,
                device=str(self.device),
                model_type=model_type,
            )

        if had_grad_ckpt:
            gen_model.gradient_checkpointing_enable()

        metrics = compute_self_reflection_metrics(rollout_results)
        val_metrics = {f"val/{k.split('/')[-1]}": v for k, v in metrics.items()}

        self.model.train()
        return val_metrics

    def _compute_log_prob(
        self,
        messages: list[dict],
        image: Any,
        model: Any,
    ) -> Any:
        """Compute log probability of completion tokens under a model.

        Masks prompt tokens so only generated tokens contribute to the
        log probability sum.

        Args:
            messages: Full message list (prompt + assistant completion)
            image: PIL Image (or None)
            model: The model to compute log probs with

        Returns:
            Scalar tensor of total log probability over completion tokens
        """
        import torch
        import torch.nn.functional as F

        # Build prompt-only messages (all except last assistant turn)
        prompt_messages = messages[:-1]

        # Tokenize full sequence (prompt + completion)
        full_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        # Tokenize prompt text only (for computing prompt length — no image needed)
        prompt_text = self.processor.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # apply_chat_template may strip <image> from non-user roles (e.g.,
        # critic prompt with <image> in assistant content).  Re-inject so the
        # processor can map the token to the PIL image.
        if image is not None and "<image>" not in full_text:
            full_text = "<image>\n" + full_text
        if image is not None and "<image>" not in prompt_text:
            prompt_text = "<image>\n" + prompt_text

        if image is not None:
            full_inputs = self.processor(text=full_text, images=image, return_tensors="pt").to(
                self.device
            )
        else:
            full_inputs = self.processor(text=full_text, return_tensors="pt").to(self.device)

        # Get prompt length by tokenizing text only (skip expensive image processing)
        prompt_ids = self.processor.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]

        # Forward pass on full sequence
        outputs = model(**full_inputs)
        logits = outputs.logits  # (1, seq_len, vocab_size)

        # Shift: logits[t] predicts token[t+1]
        # We want log P(token[t]) for t in [prompt_len, seq_len)
        # That means logits[prompt_len-1 : seq_len-1] predicting tokens[prompt_len : seq_len]
        shift_logits = logits[:, prompt_len - 1 : -1, :]
        shift_labels = full_inputs["input_ids"][:, prompt_len:]

        # Guard against inf/nan logits from fp16 + gradient checkpointing overflow
        shift_logits = torch.nan_to_num(shift_logits, nan=0.0, posinf=1e4, neginf=-1e4)

        # Compute per-token log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        if token_log_probs.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        # Sum over completion tokens — gives true log P(sequence).
        # KL overflow is already handled by the clamp(min=-20, max=20) in the
        # training loop, so we don't need length-normalization here.
        return token_log_probs.sum()

    def _preprocess_trajectory_texts(
        self,
        messages_list: list[list[dict]],
        images: list[Any],
    ) -> dict:
        """Pre-compute chat-template strings and token lengths for a trajectory batch.

        Called once before the inner optimization loop so that
        apply_chat_template + per-sequence tokenizer calls are not repeated on
        every inner epoch mini-batch forward pass.

        Args:
            messages_list: N full message lists (prompt + assistant completion)
            images: N PIL Images (one per sequence, may be None)

        Returns:
            dict with keys:
                full_texts: list[str] of N full chat-template strings
                prompt_lens: list[int] of N prompt token counts
                full_lens: list[int] of N full sequence token counts
                images: the same images list (stored for convenience)
        """
        has_image = any(img is not None for img in images)
        full_texts: list[str] = []
        prompt_texts: list[str] = []
        for messages in messages_list:
            prompt_messages = messages[:-1]
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            if has_image and "<image>" not in full_text:
                full_text = "<image>\n" + full_text
            if has_image and "<image>" not in prompt_text:
                prompt_text = "<image>\n" + prompt_text
            full_texts.append(full_text)
            prompt_texts.append(prompt_text)
        # Batch tokenize to avoid N sequential tokenizer calls
        prompt_enc = self.processor.tokenizer(
            prompt_texts, padding=False, return_attention_mask=False
        )
        prompt_lens = [len(ids) for ids in prompt_enc["input_ids"]]
        full_enc = self.processor.tokenizer(full_texts, padding=False, return_attention_mask=False)
        full_lens = [len(ids) for ids in full_enc["input_ids"]]
        return {
            "full_texts": full_texts,
            "prompt_lens": prompt_lens,
            "full_lens": full_lens,
            "images": images,
        }

    def _forward_from_pretokenized(
        self,
        pretok: dict,
        model: Any,
        mb_start: int = 0,
        mb_end: Optional[int] = None,
    ) -> list[Any]:
        """Batched forward pass using pre-computed text strings and lengths.

        Skips apply_chat_template and per-sequence tokenizer overhead.
        Passes use_cache=False to prevent KV cache allocation during training.

        Args:
            pretok: Output of _preprocess_trajectory_texts
            model: Model to run forward pass on
            mb_start: Start index for mini-batch slice (default 0)
            mb_end: End index for mini-batch slice (default: all)

        Returns:
            List of scalar log-prob tensors for the mini-batch slice
        """
        import torch
        import torch.nn.functional as F

        if mb_end is None:
            mb_end = len(pretok["full_texts"])

        full_texts = pretok["full_texts"][mb_start:mb_end]
        prompt_lens = pretok["prompt_lens"][mb_start:mb_end]
        full_lens = pretok["full_lens"][mb_start:mb_end]
        imgs = pretok["images"][mb_start:mb_end]
        n = len(full_texts)
        has_image = any(img is not None for img in imgs)

        orig_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"
        try:
            if has_image:
                batch_inputs = self.processor(
                    text=full_texts,
                    images=imgs,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
            else:
                batch_inputs = self.processor(
                    text=full_texts,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
        finally:
            self.processor.tokenizer.padding_side = orig_padding_side

        outputs = model(**batch_inputs, use_cache=False)
        logits = outputs.logits  # (n, padded_seq_len, vocab_size)

        total_len = batch_inputs["input_ids"].shape[1]
        log_probs_list = []
        for i in range(n):
            pad_len = total_len - full_lens[i]
            real_prompt_start = pad_len + prompt_lens[i]
            shift_logits = logits[i, real_prompt_start - 1 : -1, :]
            shift_labels = batch_inputs["input_ids"][i, real_prompt_start:]
            shift_logits = torch.nan_to_num(shift_logits, nan=0.0, posinf=1e4, neginf=-1e4)
            lp = F.log_softmax(shift_logits, dim=-1)
            token_lp = lp.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
            # Return per-token log-probs (not summed) for token-level GRPO loss.
            # Standard GRPO (DeepSeekMath arXiv:2402.03300, TRL GRPOTrainer)
            # computes per-token ratios for non-zero gradients with inner_epochs=1.
            log_probs_list.append(
                token_lp if token_lp.numel() > 0 else torch.tensor([0.0], device=self.device)
            )

        return log_probs_list

    def _forward_from_pretokenized_multi(
        self,
        pretok_list: list[dict],
        model: Any,
        mb_start: int = 0,
        mb_end: Optional[int] = None,
        accumulate_entropy: bool = False,
    ) -> list[list[Any]]:
        """Single forward pass for multiple pretokenized trajectory sets.

        Concatenates sequences from all sets into one batch, runs one GPU
        forward pass, then splits results back per set. Reduces forward-pass
        count (and CLIP vision encoder invocations) by len(pretok_list)×
        compared to calling _forward_from_pretokenized separately for each set.

        Args:
            pretok_list: List of pretok dicts from _preprocess_trajectory_texts.
            model: Model to run the forward pass on.
            mb_start: Mini-batch start index.
            mb_end: Mini-batch end index (default: full slice for each set).
            accumulate_entropy: If True, accumulate per-token entropy into
                self._entropy_tokens and self._entropy_sum for the caller
                to compute a token-weighted average. Only set True during the
                training inner loop, NOT during old/ref log-prob passes.

        Returns:
            List of len(pretok_list) lists, each containing scalar log-prob
            tensors for the corresponding set's mini-batch slice.
        """
        import torch
        import torch.nn.functional as F

        all_full_texts: list[str] = []
        all_prompt_lens: list[int] = []
        all_full_lens: list[int] = []
        all_imgs: list[Any] = []
        set_sizes: list[int] = []

        for pretok in pretok_list:
            _end = mb_end if mb_end is not None else len(pretok["full_texts"])
            sl = slice(mb_start, _end)
            texts = pretok["full_texts"][sl]
            set_sizes.append(len(texts))
            all_full_texts.extend(texts)
            all_prompt_lens.extend(pretok["prompt_lens"][sl])
            all_full_lens.extend(pretok["full_lens"][sl])
            all_imgs.extend(pretok["images"][sl])

        n = len(all_full_texts)
        has_image = any(img is not None for img in all_imgs)

        orig_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"
        try:
            if has_image:
                batch_inputs = self.processor(
                    text=all_full_texts,
                    images=all_imgs,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
            else:
                batch_inputs = self.processor(
                    text=all_full_texts,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
        finally:
            self.processor.tokenizer.padding_side = orig_side

        outputs = model(**batch_inputs, use_cache=False)
        logits = outputs.logits  # (n, padded_seq_len, vocab_size)
        del outputs  # free non-logit activations early

        total_len = batch_inputs["input_ids"].shape[1]
        all_lps: list[Any] = []
        for i in range(n):
            pad_len = total_len - all_full_lens[i]
            real_prompt_start = pad_len + all_prompt_lens[i]
            shift_logits = logits[i, real_prompt_start - 1 : -1, :]
            shift_labels = batch_inputs["input_ids"][i, real_prompt_start:]
            shift_logits = torch.nan_to_num(shift_logits, nan=0.0, posinf=1e4, neginf=-1e4)
            lp = F.log_softmax(shift_logits, dim=-1)
            token_lp = lp.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
            all_lps.append(
                token_lp if token_lp.numel() > 0 else torch.tensor([0.0], device=self.device)
            )
            # Accumulate per-token entropy only during training forward passes.
            # Token-weighted: each token contributes equally (matches TRL's
            # masked_batch_mean). entropy = -sum(p * log(p)) per token.
            # Collapse below ln(2)≈0.693 signals diversity loss (GTPO, DAPO).
            if accumulate_entropy and lp.shape[0] > 0:
                n_tokens = lp.shape[0]
                token_entropy_sum = -(lp.exp() * lp).sum(dim=-1).sum().item()
                self._entropy_sum += token_entropy_sum
                self._entropy_tokens += n_tokens

        result: list[list[Any]] = []
        offset = 0
        for size in set_sizes:
            result.append(all_lps[offset : offset + size])
            offset += size
        return result

    def _compute_log_probs_batched(
        self,
        messages_list: list[list[dict]],
        images: list[Any],
        model: Any,
    ) -> list[Any]:
        """Compute log probs for N sequences in one batched forward pass.

        Sequences are left-padded to the same length. Per-sequence prompt
        lengths are tracked so only completion tokens contribute to each
        sequence's log prob sum. This replaces N sequential forward passes
        with one batched call — the same efficiency pattern used by TRL.

        Args:
            messages_list: N full message lists (prompt + assistant completion)
            images: N PIL Images (or Nones), one per sequence
            model: The model to run the forward pass on

        Returns:
            List of N scalar tensors, each the total log prob over completion tokens
        """
        import torch
        import torch.nn.functional as F

        n = len(messages_list)
        has_image = any(img is not None for img in images)

        full_texts = []
        prompt_lens = []
        for messages in messages_list:
            prompt_messages = messages[:-1]
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            if has_image and "<image>" not in full_text:
                full_text = "<image>\n" + full_text
            if has_image and "<image>" not in prompt_text:
                prompt_text = "<image>\n" + prompt_text
            full_texts.append(full_text)
            prompt_lens.append(
                self.processor.tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
            )

        # Full sequence lengths (unpadded) to compute left-pad offsets later
        full_lens = [
            self.processor.tokenizer(ft, return_tensors="pt")["input_ids"].shape[1]
            for ft in full_texts
        ]

        # Left-pad so sequences can be batched for one forward pass
        orig_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"
        try:
            if has_image:
                batch_inputs = self.processor(
                    text=full_texts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
            else:
                batch_inputs = self.processor(
                    text=full_texts,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
        finally:
            self.processor.tokenizer.padding_side = orig_padding_side

        outputs = model(**batch_inputs, use_cache=False)
        logits = outputs.logits  # (n, padded_seq_len, vocab_size)

        total_len = batch_inputs["input_ids"].shape[1]
        log_probs_list = []
        for i in range(n):
            pad_len = total_len - full_lens[i]
            real_prompt_start = pad_len + prompt_lens[i]
            shift_logits = logits[i, real_prompt_start - 1 : -1, :]
            shift_labels = batch_inputs["input_ids"][i, real_prompt_start:]
            shift_logits = torch.nan_to_num(shift_logits, nan=0.0, posinf=1e4, neginf=-1e4)
            lp = F.log_softmax(shift_logits, dim=-1)
            token_lp = lp.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
            # Return per-token log-probs (not summed) for token-level GRPO loss.
            # Standard GRPO (DeepSeekMath arXiv:2402.03300, TRL GRPOTrainer)
            # computes per-token ratios for non-zero gradients with inner_epochs=1.
            log_probs_list.append(
                token_lp if token_lp.numel() > 0 else torch.tensor([0.0], device=self.device)
            )

        return log_probs_list

    def _compute_group_advantages(
        self,
        rewards: Any,
        k: int,
        loss_type: str = "grpo",
    ) -> Any:
        """Compute group-relative advantages for GRPO.

        Normalizes rewards within each group of K samples.

        Args:
            rewards: Tensor of all rewards (shape: [N * K])
            k: Group size
            loss_type: GRPO variant. "grpo" uses std normalization,
                "dr_grpo" uses only mean subtraction (removes length
                and difficulty bias per arXiv:2503.20783).

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
            if loss_type == "dr_grpo":
                advantages[start:end] = group - mean
            else:
                std = group.std()
                if std > 0:
                    advantages[start:end] = (group - mean) / (std + 1e-8)
                else:
                    advantages[start:end] = 0.0

        # Handle remaining samples
        remaining = len(rewards) - n_groups * k
        if remaining > 0:
            start = n_groups * k
            group = rewards[start:]
            mean = group.mean()
            if loss_type == "dr_grpo":
                advantages[start:] = group - mean
            else:
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
        """Save model checkpoint + optimizer state for resume.

        Args:
            path: Directory to save to
        """
        import torch

        path.mkdir(parents=True, exist_ok=True)
        unwrapped = (
            self.accelerator.unwrap_model(self.model)
            if self.accelerator is not None
            else self.model
        )
        unwrapped.save_pretrained(path)
        self.processor.save_pretrained(path)

        # Save optimizer state for resume
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")

        config_path = path / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Saved checkpoint to {path}")
