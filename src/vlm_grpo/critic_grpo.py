#!/usr/bin/env python3
"""
SelfReflectionGRPOTrainer: two-reward GRPO for VLM self-reflection.

Single model generates the full chain: A1 -> F1 -> A2. Two separate
GRPO updates per step:
  - Response update: advantage from response reward on log_prob(A1) + log_prob(A2)
  - Feedback update: advantage from feedback reward on log_prob(F1)

Both updates share the same LoRA adapter. The conversation flow matches
the inference script (self_reflective_inference_v2.py) exactly:
  A1: System=VL_ASSISTANT, User=[image]+question
  F1: System=CRITIC, Assistant=[image]+question (flipped), User=A1
  A2: System=VL_ASSISTANT, User=[image]+question, Asst=A1, User=F1 (raw)

GRPO loss (per-token, clipped surrogate):
    L = -E[min(r(theta)*A, clip(r(theta), 1-eps, 1+eps)*A)] + beta*KL
    where r(theta) = exp(log_pi_theta - log_pi_old) per token
    and A = group-normalized advantage

Usage:
    from vlm_grpo.critic_grpo import SelfReflectionGRPOTrainer

    trainer = SelfReflectionGRPOTrainer(
        model=model, ref_model=None, processor=processor,
        config=config, optimizer=optimizer, accelerator=accelerator,
        vllm_engine=vllm_engine,
    )
    trainer.train(train_dataset, val_dataset)
"""

import json
import logging
import math
import random
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Cross-rank metric aggregation (DDP)
# =============================================================================

# Metrics that represent cluster-wide counts and should be SUMmed across ranks
# rather than averaged. Everything else is treated as a per-rank rate/mean and
# averaged across ranks (valid because DDP gives equal batch sizes per rank).
_SUM_REDUCE_KEYS: frozenset[str] = frozenset({"sr/total_trajectories"})


def _dist_is_initialized() -> bool:
    """Check whether torch.distributed is usable (multi-rank training).

    Returns:
        True if torch.distributed is available, initialized, and world_size > 1.
    """
    import torch

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return False
    return torch.distributed.get_world_size() > 1


def _reduce_metrics_across_ranks(metrics: dict[str, float]) -> dict[str, float]:
    """All-reduce scalar metrics across DDP ranks.

    Averages rate/mean-type metrics and sums count-type metrics (per
    `_SUM_REDUCE_KEYS`). Non-scalar values are passed through unchanged.

    Packs all scalars into one tensor for a single all_reduce call instead
    of N per-key reductions — dramatically faster when there are many keys.

    Safe on single-GPU runs (returns metrics unchanged).

    Args:
        metrics: Dict of metric name -> value (per-rank).

    Returns:
        Dict with identical keys; scalar values averaged (or summed) across
        ranks. Non-scalar values (lists, strings) preserved as-is.
    """
    if not _dist_is_initialized():
        return metrics

    import torch

    scalar_keys = [k for k, v in metrics.items() if isinstance(v, (int, float))]
    if not scalar_keys:
        return metrics

    world_size = torch.distributed.get_world_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values = torch.tensor(
        [float(metrics[k]) for k in scalar_keys],
        device=device,
        dtype=torch.float32,
    )
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)

    reduced = dict(metrics)
    for i, k in enumerate(scalar_keys):
        summed = float(values[i].item())
        reduced[k] = summed if k in _SUM_REDUCE_KEYS else summed / world_size

    # Visibility log: surface cross-rank reduction in the text log so PVC
    # inspection can confirm it's firing. Only log on rank 0 to avoid spam.
    if torch.distributed.get_rank() == 0:
        total = reduced.get("sr/total_trajectories", float("nan"))
        wr = reduced.get("sr/wr_rate", float("nan"))
        rw = reduced.get("sr/rw_rate", float("nan"))
        logger.info(
            f"  [cross-rank reduce] world={world_size} "
            f"total_trajectories={total:.0f} "
            f"wr={wr:.3f} rw={rw:.3f}"
        )
    return reduced


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


@dataclass
class SSREntry:
    """A stored K-group for Selective Sample Replay (VL-Rethinker).

    Stores a complete rollout result for one sample (all K trajectories)
    along with the sample dict needed to rebuild prompts and images.

    Attributes:
        sample: Original sample dict (question, image_path, etc.)
        rollout_result: Full SelfReflectionRolloutResult with K trajectories
        fb_reward_std: Feedback reward std within this K-group
    """

    sample: dict
    rollout_result: Any
    fb_reward_std: float


class SSRBuffer:
    """Selective Sample Replay buffer (VL-Rethinker, arXiv:2504.08837).

    Stores K-groups with non-zero feedback reward variance and replays
    them when the current batch has zero-variance groups, ensuring the
    model always receives gradient signal for feedback learning.

    Args:
        max_size: Maximum K-groups in the buffer (FIFO eviction)
        alpha: Exponent for priority sampling (|std|^alpha)
    """

    def __init__(self, max_size: int = 64, alpha: float = 1.0) -> None:
        self._buffer: deque[SSREntry] = deque(maxlen=max_size)
        self.alpha = alpha

    def store(self, entries: list[SSREntry]) -> None:
        """Add non-zero-variance K-groups to the buffer.

        Args:
            entries: SSREntry objects to store
        """
        for entry in entries:
            self._buffer.append(entry)

    def sample(self, n: int) -> list[SSREntry]:
        """Sample n entries WITHOUT replacement, weighted by |fb_reward_std|^alpha.

        Args:
            n: Number of entries to sample

        Returns:
            List of sampled SSREntry objects (unique)
        """
        if not self._buffer or n <= 0:
            return []

        n = min(n, len(self._buffer))
        buf_list = list(self._buffer)
        weights = [abs(e.fb_reward_std) ** self.alpha for e in buf_list]
        total = sum(weights)
        if total < 1e-8:
            return random.sample(buf_list, n)

        # Without-replacement weighted sampling
        selected = []
        available = list(range(len(buf_list)))
        for _ in range(n):
            remaining_w = [weights[i] for i in available]
            r = random.random() * sum(remaining_w)
            cumulative = 0.0
            chosen_idx = len(available) - 1  # fallback for float rounding
            for idx, i in enumerate(available):
                cumulative += weights[i]
                if r <= cumulative:
                    chosen_idx = idx
                    break
            selected.append(buf_list[available[chosen_idx]])
            available.pop(chosen_idx)

        return selected

    def __len__(self) -> int:
        return len(self._buffer)


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
        self.wandb_run = None  # Set by train_self_reflection.py after init
        self.best_metric = float("inf") if config.early_stopping.mode == "min" else float("-inf")
        self.patience_counter = 0

        # EMA tracking for wandb metrics (per-step values are too noisy with
        # only batch_size * K = 16 trajectories per step).
        self._ema: dict[str, float] = {}
        self._ema_alpha = 0.05  # ~20-step half-life

        # Per-rank trajectory log for feedback preference data extraction.
        # Written by ALL ranks (not just rank 0), so all 9K questions are saved.
        self._trajectory_log = None

        # Selective Sample Replay buffer (VL-Rethinker)
        if getattr(config, "use_ssr", False):
            self.ssr_buffer: Optional[SSRBuffer] = SSRBuffer(
                max_size=getattr(config, "ssr_buffer_size", 64),
                alpha=getattr(config, "ssr_alpha", 1.0),
            )
        else:
            self.ssr_buffer = None

    def _update_ema(self, key: str, value: float) -> float:
        """Update exponential moving average and return smoothed value."""
        if key not in self._ema:
            self._ema[key] = value
        else:
            self._ema[key] = (1 - self._ema_alpha) * self._ema[key] + self._ema_alpha * value
        return self._ema[key]

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

        # Open per-rank trajectory log (all ranks write, not just rank 0)
        rank = getattr(self.accelerator, "process_index", 0) if self.accelerator else 0
        traj_log_path = output_dir / f"trajectories_rank{rank}.jsonl"
        self._trajectory_log = open(traj_log_path, "a")
        logger.info(f"Trajectory log: {traj_log_path}")

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

                # wandb logging (every step, not just logging_steps)
                if self.wandb_run is not None:
                    metrics = step_result.rollout_metrics
                    wandb_dict = {
                        # Core losses
                        "train/loss": step_result.loss,
                        "train/resp_loss": step_result.response_loss,
                        "train/fb_loss": step_result.feedback_loss,
                        "train/kl_loss": step_result.kl_loss,
                        # Rewards
                        "reward/resp_mean": step_result.response_reward_mean,
                        "reward/fb_mean": step_result.feedback_reward_mean,
                        "reward/resp_std": step_result.rollout_metrics.get(
                            "sr/response_reward_std", 0
                        ),
                        "reward/fb_std": step_result.rollout_metrics.get(
                            "sr/feedback_reward_std", 0
                        ),
                        # Accuracy (raw)
                        "accuracy/a1": metrics.get("sr/a1_accuracy", 0),
                        "accuracy/a2": metrics.get("sr/a2_accuracy", 0),
                        # Transition rates (raw)
                        "transitions/rr_rate": metrics.get("sr/rr_rate", 0),
                        "transitions/rw_rate": rw_rate,
                        "transitions/wr_rate": metrics.get("sr/wr_rate", 0),
                        "transitions/ww_rate": metrics.get("sr/ww_rate", 0),
                        # Training stability
                        "stability/entropy": step_result.rollout_metrics.get("sr/entropy", 0),
                        "stability/grad_norm": step_result.rollout_metrics.get("sr/grad_norm", 0),
                        "stability/frac_zero_std_resp": step_result.rollout_metrics.get(
                            "sr/frac_zero_std_resp", 0
                        ),
                        "stability/frac_zero_std_fb": step_result.rollout_metrics.get(
                            "sr/frac_zero_std_fb", 0
                        ),
                        "stability/resp_clip_frac": step_result.rollout_metrics.get(
                            "sr/resp_clip_frac", 0
                        ),
                        "stability/fb_clip_frac": step_result.rollout_metrics.get(
                            "sr/fb_clip_frac", 0
                        ),
                        # Advantages
                        "advantages/resp_abs_mean": step_result.rollout_metrics.get(
                            "sr/resp_adv_abs_mean", 0
                        ),
                        "advantages/fb_abs_mean": step_result.rollout_metrics.get(
                            "sr/fb_adv_abs_mean", 0
                        ),
                        # Token lengths
                        "lengths/a1_tokens": step_result.rollout_metrics.get("sr/avg_a1_tokens", 0),
                        "lengths/f1_tokens": step_result.rollout_metrics.get("sr/avg_f1_tokens", 0),
                        "lengths/a2_tokens": step_result.rollout_metrics.get("sr/avg_a2_tokens", 0),
                        # Format-violation rates (hard short-circuit hit rate).
                        # Disambiguates "model failed to correct" from
                        # "model failed to follow tag format" — critical
                        # when reading rr/rw/wr/ww rates since short-circuit
                        # forces a2_correct=False.
                        "format_violations/a2_rate": metrics.get(
                            "sr/a2_format_violation_rate", 0
                        ),
                        "format_violations/fb_rate": metrics.get(
                            "sr/fb_format_violation_rate", 0
                        ),
                    }

                    # EMA-smoothed versions of noisy per-step metrics
                    # (raw values have only batch_size * K = 16 samples per step)
                    ema_sources = {
                        "ema/a1_acc": metrics.get("sr/a1_accuracy", 0),
                        "ema/a2_acc": metrics.get("sr/a2_accuracy", 0),
                        "ema/wr_rate": metrics.get("sr/wr_rate", 0),
                        "ema/rw_rate": rw_rate,
                        "ema/rr_rate": metrics.get("sr/rr_rate", 0),
                        "ema/ww_rate": metrics.get("sr/ww_rate", 0),
                        "ema/resp_reward": step_result.response_reward_mean,
                        "ema/fb_reward": step_result.feedback_reward_mean,
                        "ema/entropy": step_result.rollout_metrics.get("sr/entropy", 0),
                        "ema/resp_adv_abs": step_result.rollout_metrics.get(
                            "sr/resp_adv_abs_mean", 0
                        ),
                        "ema/fb_adv_abs": step_result.rollout_metrics.get("sr/fb_adv_abs_mean", 0),
                        "ema/a2_fmt_violation": metrics.get(
                            "sr/a2_format_violation_rate", 0
                        ),
                        "ema/fb_fmt_violation": metrics.get(
                            "sr/fb_format_violation_rate", 0
                        ),
                    }
                    for ema_key, raw_val in ema_sources.items():
                        wandb_dict[ema_key] = self._update_ema(ema_key, raw_val)

                    # Derived EMA metrics
                    wandb_dict["ema/a2_minus_a1"] = self._ema.get("ema/a2_acc", 0) - self._ema.get(
                        "ema/a1_acc", 0
                    )
                    wandb_dict["ema/wr_minus_rw"] = self._ema.get("ema/wr_rate", 0) - self._ema.get(
                        "ema/rw_rate", 0
                    )

                    # Add reward component breakdown if available
                    for key in [
                        "sr/resp_a1_correctness_mean",
                        "sr/resp_a2_correctness_mean",
                        "sr/resp_no_regression_mean",
                        "sr/resp_a2_format_mean",
                        "sr/fb_downstream_mean",
                        "sr/fb_verification_mean",
                        "sr/fb_format_mean",
                    ]:
                        if key in metrics:
                            wandb_key = "components/" + key.split("/")[-1]
                            wandb_dict[wandb_key] = metrics[key]

                    self.wandb_run.log(wandb_dict, step=self.global_step)

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

        if self._trajectory_log is not None:
            self._trajectory_log.close()
            self._trajectory_log = None

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

        # Disable gradient checkpointing during generation -- it forces
        # use_cache=False which breaks autoregressive KV caching and
        # produces garbage output.
        had_grad_ckpt = gen_model.is_gradient_checkpointing
        if had_grad_ckpt:
            gen_model.gradient_checkpointing_disable()

        model_type = getattr(self.config, "model_type", "llava")

        # vLLM sleep/wake lifecycle: every step, wake for weights, sync,
        # wake for generation, generate, then sleep. The selective
        # wake_up(tags=...) pattern avoids the refcount crash that affects
        # wake_up() without tags (vLLM issues #20431, #16993, #24879).
        if self.vllm_engine is not None:
            self.vllm_engine.wake_up_for_weights()
            self.vllm_engine.update_weights_from_peft(gen_model, accelerator=self.accelerator)
            self.vllm_engine.wake_up_for_generation()

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

        # Sleep every step to free GPU memory for training
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

            # Save trajectory data to per-rank JSONL for feedback preference
            # extraction. Unlike logger output, this is written by ALL ranks.
            if self._trajectory_log is not None:
                for j in range(k):
                    rb = result.response_breakdowns[j] if result.response_breakdowns else None
                    fb = result.feedback_breakdowns[j] if result.feedback_breakdowns else None
                    record = {
                        "question": result.question,
                        "image_path": result.image_path,
                        "ground_truth": result.ground_truth,
                        "answer_type": result.answer_type,
                        "dataset_name": result.dataset_name,
                        "a1_text": result.answer1s[j],
                        "f1_text": result.feedbacks[j],
                        "a2_text": result.answer2s[j],
                        "a1_correct": rb.a1_correct if rb else None,
                        "a2_correct": rb.a2_correct if rb else None,
                        "a2_extracted": rb.a2_extracted if rb else None,
                        "resp_reward": result.response_rewards[j],
                        "fb_reward": result.feedback_rewards[j],
                        "resp_components": rb.components if rb else {},
                        "fb_components": fb.components if fb else {},
                        "global_step": self.global_step,
                    }
                    self._trajectory_log.write(json.dumps(record) + "\n")
                self._trajectory_log.flush()

            image = None
            for s in batch:
                if s.get("question", "").replace("<image>", "").strip() == result.question:
                    image = s.get("image")
                    break

            use_tags = self.config.rollout.use_think_answer_tags
            for a1, f1, a2 in zip(result.answer1s, result.feedbacks, result.answer2s):
                a1_prompt = build_initial_answer_prompt(
                    result.question, use_think_answer_tags=use_tags
                )
                a1_full = build_prompt_with_completion(a1_prompt, a1)

                f1_prompt = build_critic_prompt(
                    result.question,
                    a1,
                    model_type=model_type,
                )
                f1_full = build_prompt_with_completion(f1_prompt, f1)

                a2_prompt = build_refiner_prompt(
                    result.question, a1, f1, use_think_answer_tags=use_tags
                )
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

        # SSR: store non-zero-variance K-groups, augment with replayed ones
        if self.ssr_buffer is not None:
            import torch as _torch

            n_groups = len(rollout_results)
            n_zero_var = 0
            new_entries = []

            for result in rollout_results:
                fb_rews = result.feedback_rewards
                # Only store complete K-groups (partial groups break advantage alignment)
                if len(fb_rews) != k or len(result.response_breakdowns) != k:
                    continue
                fb_std = _torch.tensor(fb_rews).std().item()
                if fb_std > 1e-4:
                    # Store image_path (not PIL Image) to avoid memory leak
                    sample_copy = {"image_path": result.image_path, "question": result.question}
                    new_entries.append(
                        SSREntry(
                            sample=sample_copy,
                            rollout_result=result,
                            fb_reward_std=fb_std,
                        )
                    )
                else:
                    n_zero_var += 1

            self.ssr_buffer.store(new_entries)

            # Augment: for each zero-variance group, pull one from buffer
            if n_zero_var > 0 and len(self.ssr_buffer) > 0:
                replayed = self.ssr_buffer.sample(n_zero_var)
                logger.info(
                    f"  SSR: {n_zero_var}/{n_groups} zero-var fb groups, "
                    f"replaying {len(replayed)} from buffer "
                    f"(size={len(self.ssr_buffer)})"
                )
                use_tags = self.config.rollout.use_think_answer_tags
                for entry in replayed:
                    rr = entry.rollout_result
                    all_resp_rewards.extend(rr.response_rewards)
                    all_fb_rewards.extend(rr.feedback_rewards)

                    # Reload image from path (not stored PIL reference)
                    from vlm_grpo.data import load_image_safe

                    image = load_image_safe(entry.sample["image_path"])

                    for a1, f1, a2 in zip(rr.answer1s, rr.feedbacks, rr.answer2s):
                        a1_prompt = build_initial_answer_prompt(
                            rr.question, use_think_answer_tags=use_tags
                        )
                        a1_full = build_prompt_with_completion(a1_prompt, a1)
                        f1_prompt = build_critic_prompt(
                            rr.question,
                            a1,
                            model_type=model_type,
                        )
                        f1_full = build_prompt_with_completion(f1_prompt, f1)
                        a2_prompt = build_refiner_prompt(
                            rr.question,
                            a1,
                            f1,
                            use_think_answer_tags=use_tags,
                        )
                        a2_full = build_prompt_with_completion(a2_prompt, a2)
                        trajectory_data.append(
                            {
                                "a1_full": a1_full,
                                "f1_full": f1_full,
                                "a2_full": a2_full,
                                "image": image,
                            }
                        )

                    rollout_results.append(rr)

        # Step 3: Group-relative advantages (computed once, frozen)
        resp_rewards_t = torch.tensor(all_resp_rewards, device=self.device)
        fb_rewards_t = torch.tensor(all_fb_rewards, device=self.device)
        loss_type = getattr(self.config, "loss_type", "grpo")
        separate_turns = getattr(self.config, "separate_turn_loss", False)

        if separate_turns:
            # SCoRe-style: separate advantages for A1 and A2.
            # A1 reward = a1_correctness only (so wrong A1 gets negative advantage).
            # A2 reward = everything else (correctness, no_regression, format).
            a1_rewards_list = []
            a2_rewards_list = []
            for result in rollout_results:
                for rb in result.response_breakdowns:
                    a1_r = rb.weighted_components.get("a1_correctness", 0.0)
                    a2_r = (
                        rb.weighted_components.get("a2_correctness", 0.0)
                        + rb.weighted_components.get("no_regression", 0.0)
                        + rb.weighted_components.get("a2_format", 0.0)
                    )
                    a1_rewards_list.append(a1_r)
                    a2_rewards_list.append(a2_r)
            a1_rewards_t = torch.tensor(a1_rewards_list, device=self.device)
            a2_rewards_t = torch.tensor(a2_rewards_list, device=self.device)
            a1_advantages = self._compute_group_advantages(a1_rewards_t, k, loss_type=loss_type)
            a2_advantages = self._compute_group_advantages(a2_rewards_t, k, loss_type=loss_type)
            # Joint resp_advantages still used for logging/metrics
            resp_advantages = self._compute_group_advantages(resp_rewards_t, k, loss_type=loss_type)
        else:
            a1_advantages = None
            a2_advantages = None
            resp_advantages = self._compute_group_advantages(resp_rewards_t, k, loss_type=loss_type)

        fb_advantages = self._compute_group_advantages(fb_rewards_t, k, loss_type=loss_type)

        # Compute frac_reward_zero_std for BOTH response and feedback rewards:
        # fraction of K-groups where all trajectories got identical rewards
        # (zero variance -> zero learning signal). Key TRL diagnostic metric.
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
        # (64 x 1000 x 32000 x 2 bytes = 4 GB for a full-batch forward pass).
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
            # Combine a1/a2/f1 into one forward pass per mini-batch (3x fewer GPU ops)
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
            if separate_turns:
                # Separate A1 and A2 for per-turn loss
                old_a1_lps = old_a1_lps_list
                old_a2_lps = old_a2_lps_list
            else:
                old_a1_lps = None
                old_a2_lps = None
            old_resp_lps = [torch.cat([a1, a2]) for a1, a2 in zip(old_a1_lps_list, old_a2_lps_list)]
            old_fb_lps = old_fb_lps_list  # already list of per-token tensors

            if self.config.kl_coeff > 0:
                ref_a1_lps = ref_a1_lps_list
                ref_a2_lps = ref_a2_lps_list
                ref_fb_lps = ref_fb_lps_list
            else:
                ref_a1_lps = None
                ref_a2_lps = None
                ref_fb_lps = None

        # No explicit cache clear -- PyTorch reuses memory automatically.

        clip_range = self.config.clip_range
        num_inner = self.config.num_inner_epochs

        # Step 5: Inner optimization epochs -- use pre-tokenized data to skip
        # redundant apply_chat_template / tokenizer calls on every pass.
        total_resp_loss = 0.0
        total_fb_loss = 0.0
        total_kl_loss = 0.0
        n_traj_inner = n_traj

        # Reset entropy accumulators -- token-weighted average across all
        # training mini-batches (not old/ref passes).
        self._entropy_sum = 0.0
        self._entropy_tokens = 0
        # Reset clip fraction accumulators
        self._resp_clipped_tokens = 0
        self._resp_total_tokens = 0
        self._fb_clipped_tokens = 0
        self._fb_total_tokens = 0

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

                    fb_new = mb_fb_lps[j]

                    if separate_turns and a1_advantages is not None:
                        # SCoRe-style: separate A1 and A2 loss with per-turn advantages
                        a1_new = mb_a1_lps[j]
                        a2_new = mb_a2_lps[j]

                        a1_ratio_raw = torch.nan_to_num(
                            a1_new - old_a1_lps[ti].detach(), nan=0.0, posinf=20.0, neginf=-20.0
                        )
                        a2_ratio_raw = torch.nan_to_num(
                            a2_new - old_a2_lps[ti].detach(), nan=0.0, posinf=20.0, neginf=-20.0
                        )

                        a1_ratio = torch.exp(torch.clamp(a1_ratio_raw, -20.0, 20.0))
                        a2_ratio = torch.exp(torch.clamp(a2_ratio_raw, -20.0, 20.0))
                        a1_clipped = torch.clamp(a1_ratio, 1 - clip_range, 1 + clip_range)
                        a2_clipped = torch.clamp(a2_ratio, 1 - clip_range, 1 + clip_range)

                        a1_loss = -torch.min(
                            a1_ratio * a1_advantages[ti], a1_clipped * a1_advantages[ti]
                        ).mean()
                        a2_loss = -torch.min(
                            a2_ratio * a2_advantages[ti], a2_clipped * a2_advantages[ti]
                        ).mean()

                        # SCoRe Stage I: freeze A1 policy loss for initial steps.
                        # A1 KL anchor is kept (computed separately below) so A1
                        # stays near the reference distribution while F1/A2 learn.
                        freeze_a1 = getattr(self.config, "freeze_a1_steps", 0)
                        if freeze_a1 > 0 and self.global_step < freeze_a1:
                            a1_loss = torch.zeros_like(a1_loss)

                        traj_resp_loss = a1_loss + a2_loss

                        # Track clip fraction (combined)
                        resp_ratio_combined = torch.cat([a1_ratio, a2_ratio])
                        resp_clipped_combined = torch.cat([a1_clipped, a2_clipped])
                        self._resp_total_tokens += resp_ratio_combined.numel()
                        self._resp_clipped_tokens += (
                            (resp_ratio_combined != resp_clipped_combined).sum().item()
                        )
                    else:
                        # Original: joint A1+A2 response loss
                        resp_new = torch.cat([mb_a1_lps[j], mb_a2_lps[j]])
                        resp_ratio_raw = torch.nan_to_num(
                            resp_new - old_resp_lps[ti].detach(), nan=0.0, posinf=20.0, neginf=-20.0
                        )
                        resp_log_ratio = torch.clamp(resp_ratio_raw, min=-20.0, max=20.0)
                        resp_ratio = torch.exp(resp_log_ratio)
                        resp_clipped_ratio = torch.clamp(resp_ratio, 1 - clip_range, 1 + clip_range)
                        resp_surr1 = resp_ratio * resp_advantages[ti]
                        resp_surr2 = resp_clipped_ratio * resp_advantages[ti]
                        traj_resp_loss = -torch.min(resp_surr1, resp_surr2).mean()

                        self._resp_total_tokens += resp_ratio.numel()
                        self._resp_clipped_tokens += (resp_ratio != resp_clipped_ratio).sum().item()

                    # Feedback: per-token clipped surrogate
                    fb_ratio_raw = torch.nan_to_num(
                        fb_new - old_fb_lps[ti].detach(), nan=0.0, posinf=20.0, neginf=-20.0
                    )
                    fb_log_ratio = torch.clamp(fb_ratio_raw, min=-20.0, max=20.0)
                    fb_ratio = torch.exp(fb_log_ratio)
                    fb_clipped_ratio = torch.clamp(fb_ratio, 1 - clip_range, 1 + clip_range)
                    fb_surr1 = fb_ratio * fb_advantages[ti]
                    fb_surr2 = fb_clipped_ratio * fb_advantages[ti]
                    traj_fb_loss = -torch.min(fb_surr1, fb_surr2).mean()

                    # Track clip fraction
                    self._fb_total_tokens += fb_ratio.numel()
                    self._fb_clipped_tokens += (fb_ratio != fb_clipped_ratio).sum().item()

                    # KL loss (per-token, averaged) with per-turn coefficients.
                    # SCoRe insight: strong A1 KL anchors first turn near base model,
                    # preventing "direct solution collapse" where the model just
                    # improves A1 instead of learning self-correction.
                    if self.config.kl_coeff > 0 and ref_a1_lps is not None:

                        def _kl_term(ref_lps: torch.Tensor, new_lps: torch.Tensor) -> torch.Tensor:
                            raw = torch.nan_to_num(
                                ref_lps.detach() - new_lps, nan=0.0, posinf=20.0, neginf=-20.0
                            )
                            clamped = torch.clamp(raw, -20.0, 20.0)
                            return (torch.exp(clamped) - clamped - 1).mean()

                        a1_kl = _kl_term(ref_a1_lps[ti], mb_a1_lps[j])
                        a2_kl = _kl_term(ref_a2_lps[ti], mb_a2_lps[j])
                        f1_kl = _kl_term(ref_fb_lps[ti], fb_new)

                        a1_kl_c = self.config.kl_coeff * getattr(self.config, "a1_kl_coeff", 1.0)
                        a2_kl_c = self.config.kl_coeff * getattr(self.config, "a2_kl_coeff", 1.0)
                        fb_kl_c = self.config.kl_coeff * getattr(self.config, "fb_kl_coeff", 1.0)

                        traj_kl_loss = a1_kl_c * a1_kl + a2_kl_c * a2_kl + fb_kl_c * f1_kl
                    else:
                        traj_kl_loss = torch.tensor(0.0, device=fb_new.device)

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
                    f"  [inner epoch {inner_epoch + 1}] NaN/inf gradient detected -- "
                    "skipping optimizer step to preserve model weights"
                )
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()

            total_resp_loss += epoch_resp_loss
            total_fb_loss += epoch_fb_loss
            total_kl_loss += epoch_kl_loss

        # Log advantage statistics to detect the std=0 problem
        # (all K trajectories getting identical rewards -> zero advantages -> zero gradients)
        resp_adv_abs_mean = resp_advantages.abs().mean().item()
        fb_adv_abs_mean = fb_advantages.abs().mean().item()
        n_zero_adv = (resp_advantages == 0).sum().item()
        n_zero_fb_adv = (fb_advantages == 0).sum().item()

        # Token-weighted entropy from training forward passes (accumulated
        # across all inner-loop mini-batches, NOT from old/ref log-prob passes).
        entropy = self._entropy_sum / self._entropy_tokens if self._entropy_tokens > 0 else 0.0

        # Compute clip fraction: fraction of tokens where ratio was clipped
        resp_clip_frac = self._resp_clipped_tokens / max(self._resp_total_tokens, 1)
        fb_clip_frac = self._fb_clipped_tokens / max(self._fb_total_tokens, 1)

        logger.info(
            f"  Inner epochs done: resp_loss={total_resp_loss / num_inner:.4f}, "
            f"fb_loss={total_fb_loss / num_inner:.4f}, "
            f"grad_norm={grad_norm:.4f}, "
            f"resp_adv={resp_adv_abs_mean:.4f}(z={n_zero_adv}/{len(resp_advantages)}), "
            f"fb_adv={fb_adv_abs_mean:.4f}(z={n_zero_fb_adv}/{len(fb_advantages)}), "
            f"entropy={entropy:.3f}, "
            f"frac_zero_std={frac_resp_zero_std:.2f}/{frac_fb_zero_std:.2f}, "
            f"clip_frac={resp_clip_frac:.3f}/{fb_clip_frac:.3f}, "
            f"tok={avg_a1_toks:.0f}/{avg_f1_toks:.0f}/{avg_a2_toks:.0f}"
        )

        # Inject training-only metrics into rollout_metrics for wandb
        rollout_metrics["sr/entropy"] = entropy
        rollout_metrics["sr/grad_norm"] = grad_norm
        rollout_metrics["sr/frac_zero_std_resp"] = frac_resp_zero_std
        rollout_metrics["sr/frac_zero_std_fb"] = frac_fb_zero_std
        rollout_metrics["sr/resp_adv_abs_mean"] = resp_adv_abs_mean
        rollout_metrics["sr/fb_adv_abs_mean"] = fb_adv_abs_mean
        rollout_metrics["sr/resp_clip_frac"] = resp_clip_frac
        rollout_metrics["sr/fb_clip_frac"] = fb_clip_frac
        rollout_metrics["sr/avg_a1_tokens"] = avg_a1_toks
        rollout_metrics["sr/avg_f1_tokens"] = avg_f1_toks
        rollout_metrics["sr/avg_a2_tokens"] = avg_a2_toks
        rollout_metrics["sr/response_reward_std"] = resp_rewards_t.std().item()
        rollout_metrics["sr/feedback_reward_std"] = fb_rewards_t.std().item()

        # Reward component breakdown (averaged across trajectories)
        for result in rollout_results:
            if result.response_breakdowns:
                for bd in result.response_breakdowns:
                    for comp_name, comp_val in bd.weighted_components.items():
                        key = f"sr/resp_{comp_name}_mean"
                        rollout_metrics.setdefault(key, [])
                        rollout_metrics[key].append(comp_val)
            if result.feedback_breakdowns:
                for bd in result.feedback_breakdowns:
                    for comp_name, comp_val in bd.weighted_components.items():
                        key = f"sr/fb_{comp_name}_mean"
                        rollout_metrics.setdefault(key, [])
                        rollout_metrics[key].append(comp_val)
        # Average the component lists
        for key in list(rollout_metrics.keys()):
            if key.startswith("sr/resp_") and key.endswith("_mean"):
                val = rollout_metrics[key]
                if isinstance(val, list) and val:
                    rollout_metrics[key] = sum(val) / len(val)
            if key.startswith("sr/fb_") and key.endswith("_mean"):
                val = rollout_metrics[key]
                if isinstance(val, list) and val:
                    rollout_metrics[key] = sum(val) / len(val)

        # Aggregate metrics across DDP ranks so rank 0 logs the cluster-wide
        # view (64 trajectories) instead of its local 16-trajectory slice.
        # Per-rank view has ±10pp binomial noise on rate metrics; cross-rank
        # averaging halves that before it reaches wandb / EMA.
        rollout_metrics = _reduce_metrics_across_ranks(rollout_metrics)

        if self.scheduler is not None:
            self.scheduler.step()

        self.global_step += 1

        # Also reduce scalar reward means so response/feedback_reward_mean
        # reflect the global view (matches rollout_metrics).
        resp_reward_global = resp_rewards_t.mean().item()
        fb_reward_global = fb_rewards_t.mean().item()
        if _dist_is_initialized():
            import torch as _torch

            vals = _torch.tensor(
                [resp_reward_global, fb_reward_global],
                device=self.device, dtype=_torch.float32,
            )
            _torch.distributed.all_reduce(vals, op=_torch.distributed.ReduceOp.SUM)
            world_size = _torch.distributed.get_world_size()
            resp_reward_global = float(vals[0].item()) / world_size
            fb_reward_global = float(vals[1].item()) / world_size

        return SelfReflectionTrainStepResult(
            loss=(total_resp_loss + total_fb_loss + total_kl_loss) / num_inner,
            response_loss=total_resp_loss / num_inner,
            feedback_loss=total_fb_loss / num_inner,
            kl_loss=total_kl_loss / num_inner,
            response_reward_mean=resp_reward_global,
            feedback_reward_mean=fb_reward_global,
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

        # Log validation metrics to wandb
        if self.wandb_run is not None:
            self.wandb_run.log(val_metrics, step=self.global_step)

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
        # Tokenize prompt text only (for computing prompt length -- no image needed)
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

        # Sum over completion tokens -- gives true log P(sequence).
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
        count (and CLIP vision encoder invocations) by len(pretok_list)x
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
            # Collapse below ln(2)=0.693 signals diversity loss (GTPO, DAPO).
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
