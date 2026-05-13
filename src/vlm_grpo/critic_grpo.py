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
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

from vlm_grpo.utils import compute_warmup_lr

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

    import json

    import torch

    world_size = torch.distributed.get_world_size()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Synchronize key set across ranks. If two ranks emit different keys
    # (e.g., one rank's K-group was entirely gated and skipped an A2
    # component while another rank's wasn't), packing per-rank tensors
    # blindly would produce different-sized tensors and NCCL would
    # deadlock on the size mismatch. We use rank 0's key set as the
    # canonical schema and zero-fill missing keys on other ranks.
    local_scalar_keys = sorted(k for k, v in metrics.items() if isinstance(v, (int, float)))
    if torch.distributed.get_rank() == 0:
        encoded = json.dumps(local_scalar_keys).encode("utf-8")
        size_t = torch.tensor([len(encoded)], device=device, dtype=torch.long)
    else:
        encoded = b""
        size_t = torch.tensor([0], device=device, dtype=torch.long)
    torch.distributed.broadcast(size_t, src=0)
    payload = torch.zeros(int(size_t.item()), device=device, dtype=torch.uint8)
    if torch.distributed.get_rank() == 0:
        payload[: len(encoded)] = torch.tensor(list(encoded), device=device, dtype=torch.uint8)
    torch.distributed.broadcast(payload, src=0)
    canonical_keys: list[str] = json.loads(bytes(payload.tolist()).decode("utf-8"))
    if not canonical_keys:
        return metrics

    values = torch.tensor(
        [float(metrics[k]) if k in metrics else 0.0 for k in canonical_keys],
        device=device,
        dtype=torch.float32,
    )
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    scalar_keys = canonical_keys

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
        a1_reward_mean: Mean A1 per-turn reward (SCoRe-style). NaN when
            separate_turn_loss is disabled.
        a2_reward_mean: Mean A2 per-turn reward incl. shaped bonus
            α·(r_a2-r_a1). NaN when separate_turn_loss is disabled.
        rollout_metrics: Dict of rollout statistics
        global_step: Current global training step
    """

    loss: float
    response_loss: float
    feedback_loss: float
    kl_loss: float
    response_reward_mean: float
    feedback_reward_mean: float
    a1_reward_mean: float
    a2_reward_mean: float
    rollout_metrics: dict[str, float]
    global_step: int

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return asdict(self)


# =============================================================================
# Zero-variance group filtering (DAPO Dynamic Sampling helper)
# =============================================================================


def _mean_abs_advantage(rewards: list[float]) -> float:
    """Compute mean(|(r - mean) / std|) over a K-group.

    Used as the DAPO Dynamic Sampling drop predicate: ==0 ⇒ degenerate group
    (advantage and gradient are exactly zero for every trajectory in the group).

    Returns 0.0 when std is below 1e-6 (all-equal rewards: gradient is
    mathematically zero, no point keeping or training on the group).

    Args:
        rewards: K reward values for one rollout group.

    Returns:
        Non-negative mean |Â|. 0.0 means the group is degenerate.
    """
    if not rewards:
        return 0.0
    import torch as _torch

    t = _torch.tensor(rewards, dtype=_torch.float32)
    std = t.std().item()
    if std < 1e-6:
        return 0.0
    return ((t - t.mean()).abs() / (std + 1e-6)).mean().item()


def _identify_zero_var_groups(
    rollout_results: list[Any],
    k: int,
) -> tuple[list[bool], int]:
    """Mark each K-group as kept (non-degenerate) or dropped (degenerate).

    A group is dropped when BOTH the response head and the feedback head
    have zero variance (mean |Â| == 0) — meaning every trajectory in the
    group earns identical reward on both heads, so the GRPO advantage is
    zero on both updates and the gradient contribution is zero. A group
    that is degenerate on only one head is still kept, because it
    contributes a non-zero gradient on the other head.

    Malformed groups (response_rewards / feedback_rewards / response_breakdowns
    not exactly length K) are defensively dropped to avoid downstream
    tensor-shape errors.

    Args:
        rollout_results: List of SelfReflectionRolloutResult-like objects,
            each with .response_rewards, .feedback_rewards,
            .response_breakdowns of length K.
        k: Number of trajectories per group (config.rollout.k_samples).

    Returns:
        (kept_mask, n_dropped) where kept_mask[i]=True means group i has
        non-zero advantage on at least one head.
    """
    kept_mask: list[bool] = []
    n_dropped = 0
    for result in rollout_results:
        if (
            len(result.response_rewards) != k
            or len(result.feedback_rewards) != k
            or len(result.response_breakdowns) != k
        ):
            kept_mask.append(False)
            n_dropped += 1
            continue
        resp_priority = _mean_abs_advantage(result.response_rewards)
        fb_priority = _mean_abs_advantage(result.feedback_rewards)
        if (resp_priority + fb_priority) > 1e-4:
            kept_mask.append(True)
        else:
            kept_mask.append(False)
            n_dropped += 1
    return kept_mask, n_dropped


def _kl_term_drgrpo(
    ref_lps: Any,
    new_lps: Any,
    max_len: float,
) -> Any:
    """Schulman k3 KL estimator with Dr.GRPO normalization.

    The k3 estimator is `exp(Δ) − Δ − 1` where Δ = log_ref − log_new. It is
    unbiased and always non-negative. The standard `.mean()` aggregation
    divides by the actual completion length, which makes shorter completions
    contribute disproportionately more KL per token — a length bias that
    fights the Dr.GRPO policy loss (which uses `sum / max_len` for length
    invariance, arXiv:2503.20783).

    This helper aggregates with `sum / max_len` so that:
      - The KL gradient and the policy gradient share the same per-token
        scale, so the KL coefficient acts as a true gradient-magnitude knob.
      - A short completion does not get an inflated anchor pull.

    Inputs are NaN/inf-clamped to ±20 before exponentiation to prevent
    log-prob blowups from corrupting the gradient. Empty input returns 0.

    Args:
        ref_lps: Reference per-token log-probs (1-D tensor, detached upstream).
        new_lps: Current per-token log-probs (1-D tensor with grad).
        max_len: Per-turn max completion length (a1/a2/f1_max_completion_length).

    Returns:
        Scalar KL loss, normalized by max_len.
    """
    import torch

    if new_lps.numel() == 0:
        return torch.zeros((), device=new_lps.device, dtype=new_lps.dtype)
    raw = torch.nan_to_num(ref_lps.detach() - new_lps, nan=0.0, posinf=20.0, neginf=-20.0)
    clamped = torch.clamp(raw, -20.0, 20.0)
    return (torch.exp(clamped) - clamped - 1).sum() / max_len


def _filter_kept_groups(
    rollout_results: list[Any],
    all_resp_rewards: list[float],
    all_fb_rewards: list[float],
    trajectory_data: list[dict],
    group_slices: list[tuple[int, int]],
    kept_mask: list[bool],
) -> tuple[list[Any], list[float], list[float], list[dict]]:
    """Atomically filter the four parallel arrays by `kept_mask`.

    The reward arrays and trajectory_data are flat across trajectories;
    `group_slices[i] = (start, end)` gives the trajectory range for
    rollout_results[i]. All four are filtered in lock-step so downstream
    tensors and prompt batches stay aligned.

    Args:
        rollout_results: One entry per K-group.
        all_resp_rewards: One entry per trajectory.
        all_fb_rewards: One entry per trajectory.
        trajectory_data: One entry per trajectory.
        group_slices: One (start, end) per K-group, indexing the per-trajectory lists.
        kept_mask: One bool per K-group; True means keep.

    Returns:
        (kept_results, kept_resp_rewards, kept_fb_rewards, kept_trajectory_data)
        with the dropped groups' trajectories removed.
    """
    kept_results: list[Any] = []
    kept_resp: list[float] = []
    kept_fb: list[float] = []
    kept_traj: list[dict] = []
    for keep, (s, e), result in zip(kept_mask, group_slices, rollout_results):
        if keep:
            kept_results.append(result)
            kept_resp.extend(all_resp_rewards[s:e])
            kept_fb.extend(all_fb_rewards[s:e])
            kept_traj.extend(trajectory_data[s:e])
    return kept_results, kept_resp, kept_fb, kept_traj


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

        # Multi-adapter routing. Default = single-adapter mode using PEFT's
        # "default" adapter for all turns (matches pre-routing behavior).
        # When enabled, each turn (a1 / f1 / a2) is routed to a named
        # adapter; the trainer switches adapters before generation and
        # before each per-turn training forward pass.
        from vlm_grpo.config import AdapterRoutingConfig

        self._routing: AdapterRoutingConfig = getattr(
            config, "adapter_routing", AdapterRoutingConfig()
        )
        if self._routing.enabled:
            logger.info(
                f"  Multi-adapter routing: turns={self._routing.turns}, "
                f"trainable={self._routing.trainable_adapter_names()}"
            )

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
        # ``buffering=1`` enables line buffering — every ``\n``-terminated
        # write flushes to disk automatically. Without this, an OOM kill
        # or pod eviction during a training forward pass can lose records
        # that were written but still sat in the OS buffer. The trajectory
        # log is the primary post-crash diagnostic artifact (carries every
        # A1/F1/A2 with rewards), so durable-on-write matters more than
        # the tiny per-write syscall overhead.
        self._trajectory_log = open(traj_log_path, "a", buffering=1)
        logger.info(f"Trajectory log: {traj_log_path}")

        total_steps = math.ceil(len(train_dataset) / batch_size) * num_epochs
        logger.info(f"Starting self-reflection GRPO training: {total_steps} rollout steps")
        logger.info(f"  Epochs: {num_epochs}, Batch size: {batch_size}")
        logger.info(f"  Inner optimization epochs per step: {config.num_inner_epochs}")
        logger.info(f"  Dataset size: {len(train_dataset)}")
        logger.info("  Two-reward design: response (A1+A2) + feedback (F1)")
        if getattr(config, "use_gdpo_normalization", False):
            logger.info(
                "  GDPO normalization ENABLED (arXiv:2601.05242): per-component "
                "K-group normalize -> weighted sum -> batch-renormalize"
            )

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
                total_reward_mean = (
                    step_result.response_reward_mean + step_result.feedback_reward_mean
                )
                pbar.set_postfix(
                    epoch=f"{epoch + 1}/{num_epochs}",
                    loss=f"{step_result.loss:.4f}",
                    resp_r=f"{step_result.response_reward_mean:.3f}",
                    fb_r=f"{step_result.feedback_reward_mean:.3f}",
                    total_r=f"{total_reward_mean:.3f}",
                    rw_rate=f"{rw_rate:.3f}",
                )
                pbar.update(len(batch))

                if self.global_step % config.logging_steps == 0:
                    a1_r_str = (
                        f"{step_result.a1_reward_mean:.3f}"
                        if not math.isnan(step_result.a1_reward_mean)
                        else "nan"
                    )
                    a2_r_str = (
                        f"{step_result.a2_reward_mean:.3f}"
                        if not math.isnan(step_result.a2_reward_mean)
                        else "nan"
                    )
                    logger.info(
                        f"Step {self.global_step}: loss={step_result.loss:.4f}, "
                        f"resp_loss={step_result.response_loss:.4f}, "
                        f"fb_loss={step_result.feedback_loss:.4f}, "
                        f"resp_reward={step_result.response_reward_mean:.3f}, "
                        f"a1_reward={a1_r_str}, a2_reward={a2_r_str}, "
                        f"fb_reward={step_result.feedback_reward_mean:.3f}, "
                        f"total_reward={total_reward_mean:.3f}, "
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
                        "reward/total_mean": total_reward_mean,
                        "reward/a1_mean": step_result.a1_reward_mean,
                        "reward/a2_mean": step_result.a2_reward_mean,
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
                        # Format-violation rates: fraction of trajectories
                        # where required tags were missing (format reward = 0).
                        # Disambiguates "model failed to correct" from
                        # "model failed to follow tag format". With binary
                        # format rewards, missing tags marks the answer wrong
                        # via the natural extraction-failure path, not a
                        # short-circuit override.
                        "format_violations/a2_rate": metrics.get("sr/a2_format_violation_rate", 0),
                        "format_violations/fb_rate": metrics.get("sr/fb_format_violation_rate", 0),
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
                        "ema/total_reward": total_reward_mean,
                        "ema/entropy": step_result.rollout_metrics.get("sr/entropy", 0),
                        "ema/resp_adv_abs": step_result.rollout_metrics.get(
                            "sr/resp_adv_abs_mean", 0
                        ),
                        "ema/fb_adv_abs": step_result.rollout_metrics.get("sr/fb_adv_abs_mean", 0),
                        "ema/a2_fmt_violation": metrics.get("sr/a2_format_violation_rate", 0),
                        "ema/fb_fmt_violation": metrics.get("sr/fb_format_violation_rate", 0),
                    }
                    # PAG-only EMA metrics. Skipped silently on legacy runs
                    # where the per-segment reward / gated fields are absent.
                    # Selective revision (arXiv:2506.10406): the gate decides
                    # whether A2 runs, so these tell you whether the model is
                    # making sensible revise/skip decisions.
                    if "sr/gated_rate" in metrics:
                        ema_sources["ema/gated_rate"] = metrics["sr/gated_rate"]
                    if "sr/effective_accuracy" in metrics:
                        ema_sources["ema/effective_acc"] = metrics["sr/effective_accuracy"]
                    if "sr/r_a1_mean" in metrics:
                        ema_sources["ema/r_a1"] = metrics["sr/r_a1_mean"]
                    if "sr/r_a2_mean" in metrics:
                        ema_sources["ema/r_a2"] = metrics["sr/r_a2_mean"]
                    if "sr/r_f1_mean" in metrics:
                        ema_sources["ema/r_f1"] = metrics["sr/r_f1_mean"]
                    if "sr/shaping_bonus_mean" in metrics:
                        ema_sources["ema/shaping_bonus"] = metrics["sr/shaping_bonus_mean"]
                    if "sr/productive_gate_rate" in metrics:
                        ema_sources["ema/productive_gate_rate"] = metrics["sr/productive_gate_rate"]
                    if "sr/sycophantic_gate_rate" in metrics:
                        ema_sources["ema/sycophantic_gate_rate"] = metrics[
                            "sr/sycophantic_gate_rate"
                        ]
                    if "sr/f1_correct_verdict_precision" in metrics:
                        ema_sources["ema/f1_correct_verdict_precision"] = metrics[
                            "sr/f1_correct_verdict_precision"
                        ]
                    if "sr/f1_wrong_verdict_precision" in metrics:
                        ema_sources["ema/f1_wrong_verdict_precision"] = metrics[
                            "sr/f1_wrong_verdict_precision"
                        ]
                    for ema_key, raw_val in ema_sources.items():
                        wandb_dict[ema_key] = self._update_ema(ema_key, raw_val)

                    # Derived EMA metrics
                    wandb_dict["ema/a2_minus_a1"] = self._ema.get("ema/a2_acc", 0) - self._ema.get(
                        "ema/a1_acc", 0
                    )
                    wandb_dict["ema/wr_minus_rw"] = self._ema.get("ema/wr_rate", 0) - self._ema.get(
                        "ema/rw_rate", 0
                    )
                    if "ema/effective_acc" in self._ema and "ema/a1_acc" in self._ema:
                        # Selective revision should LIFT effective accuracy
                        # above A1 alone (the gate stops bad revisions while
                        # productive revisions still fire). Watch this go
                        # positive over training.
                        wandb_dict["ema/effective_minus_a1"] = (
                            self._ema["ema/effective_acc"] - self._ema["ema/a1_acc"]
                        )

                    # Add reward component breakdown if available
                    for key in [
                        "sr/resp_a1_correctness_mean",
                        "sr/resp_a1_format_mean",
                        "sr/resp_a2_correctness_mean",
                        "sr/resp_a2_format_mean",
                        "sr/resp_no_regression_mean",
                        "sr/resp_shaping_bonus_mean",
                        "sr/resp_wr_bonus_mean",
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
        #
        # We ALWAYS run the wake+sync prelim here, regardless of routing
        # mode: vLLM is in sleep state at step start (after last step's
        # final sleep()), so generation cannot proceed without the wake.
        # In multi-adapter mode we additionally activate the A1 adapter
        # BEFORE the sync so the merged weights pushed to vLLM correspond
        # to the head that will generate A1. The per-turn callback then
        # handles subsequent F1 / A2 adapter switches (re-merging the
        # newly-activated adapter into vLLM) — and no-ops when the next
        # turn routes to the same adapter as the current one.
        adapter_callback = self._build_adapter_callback(gen_model)
        if self.vllm_engine is not None:
            if self._routing.enabled:
                a1_adapter = self._routing.adapter_for_turn("a1")
                self._set_active_adapter(gen_model, a1_adapter)
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
            baseline_weights=getattr(self.config, "baseline_weights", None),
            adapter_callback=adapter_callback,
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
        # Track per-K-group slices into the aggregated lists so SSR can drop
        # zero-variance groups and replace them with buffer replays (paper Alg. 1 L16).
        group_slices: list[tuple[int, int]] = []

        for result in rollout_results:
            k = len(result.answer1s)
            if k == 0:
                continue

            slice_start = len(all_resp_rewards)
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
                    total_r = result.response_rewards[j] + result.feedback_rewards[j]
                    logger.info(
                        f"  resp_reward={result.response_rewards[j]:.3f} "
                        f"fb_reward={result.feedback_rewards[j]:.3f} "
                        f"total_reward={total_r:.3f}"
                    )
                    if result.response_breakdowns:
                        rb = result.response_breakdowns[j]
                        logger.info(f"  resp_components: {rb.components}")
                        logger.info(
                            f"  a1_correct={getattr(rb, 'a1_correct', None)} "
                            f"a2_correct={getattr(rb, 'a2_correct', None)} "
                            f"a2_extracted='{getattr(rb, 'a2_extracted', '')}'"
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
                        "a1_correct": getattr(rb, "a1_correct", None) if rb else None,
                        "a2_correct": getattr(rb, "a2_correct", None) if rb else None,
                        "a2_extracted": getattr(rb, "a2_extracted", None) if rb else None,
                        "resp_reward": result.response_rewards[j],
                        "fb_reward": result.feedback_rewards[j],
                        "total_reward": result.response_rewards[j] + result.feedback_rewards[j],
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

            single_turn_a1 = bool(getattr(self.config.rollout, "single_turn_a1", False))
            # rollout result token-id / logprob lists may be empty when
            # rolling back to legacy rollout paths; default to None per turn
            # so pre-tokenize falls back to the retokenize path and old_lp
            # is recomputed by the HF forward pass.
            a1_id_list = result.answer1_token_ids or [None] * len(result.answer1s)
            f1_id_list = result.feedback_token_ids or [None] * len(result.answer1s)
            a2_id_list = result.answer2_token_ids or [None] * len(result.answer1s)
            a1_lp_list = result.answer1_logprobs or [None] * len(result.answer1s)
            f1_lp_list = result.feedback_logprobs or [None] * len(result.answer1s)
            a2_lp_list = result.answer2_logprobs or [None] * len(result.answer1s)
            for a1, f1, a2, a1_ids, f1_ids, a2_ids, a1_lps, f1_lps, a2_lps in zip(
                result.answer1s,
                result.feedbacks,
                result.answer2s,
                a1_id_list,
                f1_id_list,
                a2_id_list,
                a1_lp_list,
                f1_lp_list,
                a2_lp_list,
            ):
                a1_prompt = build_initial_answer_prompt(result.question)
                a1_full = build_prompt_with_completion(a1_prompt, a1)

                if single_turn_a1:
                    # Single-turn baseline: skip F1 / A2 entirely. The trainer's
                    # per-turn forward / KL / loss will short-circuit the F1 and
                    # A2 paths when it sees ``f1_full is None`` / ``a2_full is None``.
                    trajectory_data.append(
                        {
                            "a1_full": a1_full,
                            "f1_full": None,
                            "a2_full": None,
                            "image": image,
                            "a1_completion_ids": a1_ids,
                            "f1_completion_ids": None,
                            "a2_completion_ids": None,
                            "a1_completion_logprobs": a1_lps,
                            "f1_completion_logprobs": None,
                            "a2_completion_logprobs": None,
                        }
                    )
                else:
                    f1_prompt = build_critic_prompt(
                        result.question,
                        a1,
                        model_type=model_type,
                    )
                    f1_full = build_prompt_with_completion(f1_prompt, f1)

                    a2_prompt = build_refiner_prompt(result.question, a1, f1)
                    a2_full = build_prompt_with_completion(a2_prompt, a2)

                    trajectory_data.append(
                        {
                            "a1_full": a1_full,
                            "f1_full": f1_full,
                            "a2_full": a2_full,
                            "image": image,
                            "a1_completion_ids": a1_ids,
                            "f1_completion_ids": f1_ids,
                            "a2_completion_ids": a2_ids,
                            "a1_completion_logprobs": a1_lps,
                            "f1_completion_logprobs": f1_lps,
                            "a2_completion_logprobs": a2_lps,
                        }
                    )

            group_slices.append((slice_start, len(all_resp_rewards)))

        if not all_resp_rewards:
            self.global_step += 1
            return SelfReflectionTrainStepResult(
                loss=0.0,
                response_loss=0.0,
                feedback_loss=0.0,
                kl_loss=0.0,
                response_reward_mean=0.0,
                feedback_reward_mean=0.0,
                a1_reward_mean=float("nan"),
                a2_reward_mean=float("nan"),
                rollout_metrics=rollout_metrics,
                global_step=self.global_step,
            )

        k = self.config.rollout.k_samples

        # DAPO Dynamic Sampling (arXiv:2503.14476 §3.2) + SSR (VL-Rethinker
        # Alg. 1, Eq. 1) share the same drop predicate: a K-group with zero
        # reward variance on BOTH response and feedback heads has |Â|==0 on
        # both updates and contributes zero gradient — it is identified and
        # removed here so the policy update only sees groups with real signal.
        # We train on the smaller effective batch (every gradient step is
        # still non-degenerate; we pay the rollout cost for some groups that
        # don't contribute).
        use_ds = bool(getattr(self.config, "use_dynamic_sampling", False))
        if use_ds:
            n_groups = len(rollout_results)

            kept_mask, n_dropped = _identify_zero_var_groups(rollout_results, k=k)

            # Filter all parallel arrays in lock-step.
            if n_dropped > 0:
                rollout_results, all_resp_rewards, all_fb_rewards, trajectory_data = (
                    _filter_kept_groups(
                        rollout_results=rollout_results,
                        all_resp_rewards=all_resp_rewards,
                        all_fb_rewards=all_fb_rewards,
                        trajectory_data=trajectory_data,
                        group_slices=group_slices,
                        kept_mask=kept_mask,
                    )
                )
                logger.info(
                    f"  DS: dropped {n_dropped}/{n_groups} zero-var groups, training on kept subset"
                )

            # If everything got dropped (no buffer to refill from), short-circuit
            # the policy update — there is no signal to learn from this batch.
            if not all_resp_rewards:
                self.global_step += 1
                return SelfReflectionTrainStepResult(
                    loss=0.0,
                    response_loss=0.0,
                    feedback_loss=0.0,
                    kl_loss=0.0,
                    response_reward_mean=0.0,
                    feedback_reward_mean=0.0,
                    a1_reward_mean=float("nan"),
                    a2_reward_mean=float("nan"),
                    rollout_metrics=rollout_metrics,
                    global_step=self.global_step,
                )

        # Step 3: Group-relative advantages (computed once, frozen)
        resp_rewards_t = torch.tensor(all_resp_rewards, device=self.device)
        fb_rewards_t = torch.tensor(all_fb_rewards, device=self.device)
        loss_type = getattr(self.config, "loss_type", "grpo")
        separate_turns = getattr(self.config, "separate_turn_loss", False)
        use_gdpo = getattr(self.config, "use_gdpo_normalization", False)
        use_pag = bool(getattr(self.config.rollout, "use_pag_segment_rewards", False))

        # PAG-faithful arm: per-segment binary rewards from
        # ``PAGSegmentRewardBreakdown``. ``r_a1`` and ``r_a2`` are read
        # directly (not from the weighted_components sum). Gated trajectories
        # (F1 said CORRECT → A2 skipped) carry r_a2=None; they're excluded
        # from the A2 K-group baseline and their A2 advantage is forced to 0.
        # Always supersedes use_gdpo / separate_turn_loss when enabled.
        if use_pag:
            a1_rewards_list: list[float] = []
            a2_rewards_list: list[float] = []
            a2_active_mask_list: list[bool] = []
            for result in rollout_results:
                # Tolerate legacy (non-PAG) breakdowns landing here — the
                # rollout path is supposed to emit PAGSegmentRewardBreakdown
                # when use_pag_segment_rewards is on, but a stray dispatch
                # would silently produce garbage advantages otherwise.
                for rb in result.response_breakdowns:
                    if not hasattr(rb, "r_a1"):
                        raise TypeError(
                            "use_pag_segment_rewards=True but the rollout "
                            "produced a non-PAG response breakdown "
                            f"({type(rb).__name__}); check that rollout.py "
                            "routes to compute_pag_response_breakdown."
                        )
                    a1_rewards_list.append(float(rb.r_a1))
                    if rb.r_a2 is None or rb.gated:
                        # Gated: zero placeholder, masked out below.
                        a2_rewards_list.append(0.0)
                        a2_active_mask_list.append(False)
                    else:
                        a2_rewards_list.append(float(rb.r_a2))
                        a2_active_mask_list.append(True)
            a1_rewards_t = torch.tensor(a1_rewards_list, device=self.device)
            a2_rewards_t = torch.tensor(a2_rewards_list, device=self.device)
            a2_active_mask = torch.tensor(a2_active_mask_list, dtype=torch.bool, device=self.device)

            # A1 K-group baseline: standard mean subtraction across the full
            # K-group (every trajectory generates A1).
            a1_advantages = self._compute_group_advantages(a1_rewards_t, k, loss_type=loss_type)
            # A2 K-group baseline: mean only over the trajectories that
            # actually generated A2 (i.e. those NOT gated by F1). Gated
            # samples get advantage = 0 (no A2 loss contribution).
            a2_advantages = self._compute_pag_a2_advantages(
                a2_rewards_t, a2_active_mask, k, loss_type=loss_type
            )
            # F1's reward is already per-segment (binary verdict + format).
            # Use the existing K-group baseline path — no skipping needed
            # since F1 is generated unconditionally.
            fb_advantages = self._compute_group_advantages(fb_rewards_t, k, loss_type=loss_type)
            # Joint resp_advantages kept only for the abs/zero-adv logging
            # block at the end of train_step. The actual policy loss path
            # (separate_turns=True branch) reads a1_advantages and
            # a2_advantages directly.
            resp_advantages = self._compute_group_advantages(resp_rewards_t, k, loss_type=loss_type)
            # Force the per-segment loss path on for the PAG arm.
            separate_turns = True
        elif use_gdpo:
            # GDPO (Liu 2026, arXiv:2601.05242, Eqs. 4-7): per-component
            # K-group normalize, weighted sum, then batch-renormalize.
            # We assemble the per-component matrix from the breakdowns and
            # call _compute_gdpo_advantages for each head.
            resp_components_keys = [
                "a1_correctness",
                "a1_format",
                "a2_correctness",
                "a2_format",
                "no_regression",
            ]
            fb_components_keys = ["downstream", "verification", "format"]
            resp_components_rows: list[list[float]] = []
            fb_components_rows: list[list[float]] = []
            for result in rollout_results:
                for rb in result.response_breakdowns:
                    resp_components_rows.append(
                        [float(rb.components.get(key, 0.0)) for key in resp_components_keys]
                    )
                for fb in result.feedback_breakdowns:
                    fb_components_rows.append(
                        [float(fb.components.get(key, 0.0)) for key in fb_components_keys]
                    )
            resp_components_t = torch.tensor(
                resp_components_rows, dtype=resp_rewards_t.dtype, device=self.device
            )
            fb_components_t = torch.tensor(
                fb_components_rows, dtype=fb_rewards_t.dtype, device=self.device
            )
            rw = self.config.response_weights
            fw = self.config.feedback_weights
            resp_weights_t = torch.tensor(
                [
                    rw.w_a1_correctness,
                    rw.w_a1_format,
                    rw.w_a2_correctness,
                    rw.w_a2_format,
                    rw.w_no_regression,
                ],
                dtype=resp_rewards_t.dtype,
                device=self.device,
            )
            fb_weights_t = torch.tensor(
                [fw.w_downstream, fw.w_verification_accuracy, fw.w_format],
                dtype=fb_rewards_t.dtype,
                device=self.device,
            )
            resp_advantages = self._compute_gdpo_advantages(resp_components_t, resp_weights_t, k)
            fb_advantages = self._compute_gdpo_advantages(fb_components_t, fb_weights_t, k)
            # GDPO doesn't currently support separate-turn A1/A2 advantages;
            # leave them None so the trainer falls back to the joint resp path.
            a1_advantages = None
            a2_advantages = None
            a1_rewards_t = None
            a2_rewards_t = None
        elif separate_turns:
            # SCoRe-style: separate advantages for A1 and A2.
            #   A1 reward = a1_correctness + a1_format       (turn-local signal)
            #   A2 reward = a2_correctness + a2_format       (turn-local signal)
            #             + no_regression                    (SCoRe shaping bonus α·(r_a2-r_a1))
            # Matches Stage II (Eq. 4) of arXiv:2409.12917 where the shaping
            # bonus α·(r_a2-r_a1) is added ONLY to the A2 reward, never A1.
            a1_rewards_list = []
            a2_rewards_list = []
            for result in rollout_results:
                for rb in result.response_breakdowns:
                    a1_r = rb.weighted_components.get(
                        "a1_correctness", 0.0
                    ) + rb.weighted_components.get("a1_format", 0.0)
                    a2_r = (
                        rb.weighted_components.get("a2_correctness", 0.0)
                        + rb.weighted_components.get("a2_format", 0.0)
                        + rb.weighted_components.get("no_regression", 0.0)
                    )
                    a1_rewards_list.append(a1_r)
                    a2_rewards_list.append(a2_r)
            a1_rewards_t = torch.tensor(a1_rewards_list, device=self.device)
            a2_rewards_t = torch.tensor(a2_rewards_list, device=self.device)
            a1_advantages = self._compute_group_advantages(a1_rewards_t, k, loss_type=loss_type)
            a2_advantages = self._compute_group_advantages(a2_rewards_t, k, loss_type=loss_type)
            # Joint resp_advantages still used for logging/metrics
            resp_advantages = self._compute_group_advantages(resp_rewards_t, k, loss_type=loss_type)
            fb_advantages = self._compute_group_advantages(fb_rewards_t, k, loss_type=loss_type)
        else:
            a1_advantages = None
            a2_advantages = None
            a1_rewards_t = None
            a2_rewards_t = None
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
        single_turn_a1 = bool(getattr(self.config.rollout, "single_turn_a1", False))
        a1_pretok = self._preprocess_trajectory_texts(
            [t["a1_full"] for t in trajectory_data],
            imgs,
            completion_token_ids=[t.get("a1_completion_ids") for t in trajectory_data],
            completion_logprobs=[t.get("a1_completion_logprobs") for t in trajectory_data],
            sampling_temperature=float(self.config.rollout.temperature),
        )
        if single_turn_a1:
            # Skip F1/A2 pretokenization entirely in baseline mode.
            a2_pretok = None
            f1_pretok = None
        else:
            a2_pretok = self._preprocess_trajectory_texts(
                [t["a2_full"] for t in trajectory_data],
                imgs,
                completion_token_ids=[t.get("a2_completion_ids") for t in trajectory_data],
                completion_logprobs=[t.get("a2_completion_logprobs") for t in trajectory_data],
                sampling_temperature=float(self.config.rollout.a2_temperature),
            )
            f1_pretok = self._preprocess_trajectory_texts(
                [t["f1_full"] for t in trajectory_data],
                imgs,
                completion_token_ids=[t.get("f1_completion_ids") for t in trajectory_data],
                completion_logprobs=[t.get("f1_completion_logprobs") for t in trajectory_data],
                sampling_temperature=float(self.config.rollout.feedback_temperature),
            )

        # Completion token lengths from pre-tokenized data (token count, not word count).
        # Tracks length hacking where wrong answers grow longer (Dr. GRPO key finding).
        #
        # In legacy mode: full_lens - prompt_lens (both from text-only tokenizer).
        # In native mode: full_lens stores text_only_prompt_len + len(completion_ids),
        #   so subtracting the SAME text_only prompt_lens recovers the completion
        #   length verbatim. Equivalently, len(completion_token_ids[i]) is always
        #   the true completion length regardless of path — preferred for clarity.
        def _completion_lens(pretok: dict) -> list[int]:
            comp_ids = pretok.get("completion_token_ids")
            if pretok.get("native_path", False) and comp_ids is not None:
                return [len(c) if c is not None else 0 for c in comp_ids]
            return [f - p for f, p in zip(pretok["full_lens"], pretok["prompt_lens"])]

        a1_toks = _completion_lens(a1_pretok)
        avg_a1_toks = sum(a1_toks) / max(len(a1_toks), 1)
        if single_turn_a1:
            avg_f1_toks = 0.0
            avg_a2_toks = 0.0
        else:
            f1_toks = _completion_lens(f1_pretok)
            a2_toks = _completion_lens(a2_pretok)
            avg_f1_toks = sum(f1_toks) / max(len(f1_toks), 1)
            avg_a2_toks = sum(a2_toks) / max(len(a2_toks), 1)

        logger.info(f"  Computing old/ref log-probs for {n_traj} trajectories...")

        old_a1_lps_list: list[Any] = []
        old_a2_lps_list: list[Any] = []
        old_fb_lps_list: list[Any] = []
        ref_a1_lps_list: list[Any] = []
        ref_a2_lps_list: list[Any] = []
        ref_fb_lps_list: list[Any] = []

        # Native path lets us skip the HF old-lp forward pass entirely:
        # vLLM emitted the per-token sampled logprob at rollout time, so we
        # take old_lp directly from those cached values. The remaining HF
        # work in this block is just the ref-lp pass (still required —
        # vLLM doesn't know about the reference adapter / base model).
        # Each pretok carries a ``native_path`` flag set in
        # ``_preprocess_trajectory_texts``; when all participating pretoks
        # are native AND ``completion_logprobs`` is fully populated, we use
        # the shortcut. Otherwise we fall back to the HF forward pass.
        def _logprobs_to_tensors(pretok: dict) -> Optional[list[Any]]:
            """Convert per-trajectory completion_logprobs lists to tensors,
            or return None when the shortcut is unsafe.

            Returns None (forcing the HF forward-pass fallback) when:
              - the pretok is not in native mode
              - completion_logprobs is missing entirely
              - any per-trajectory logprob list is None
              - any per-trajectory logprob list length disagrees with the
                paired completion_token_ids length (defensive: a future
                vLLM/HF mismatch here would otherwise produce mismatched
                tensors inside the per-token IS loss loop and crash later)
            """
            lp_lists = pretok.get("completion_logprobs")
            if not pretok.get("native_path", False) or lp_lists is None:
                return None
            if not all(lp is not None for lp in lp_lists):
                return None
            id_lists = pretok.get("completion_token_ids")
            if id_lists is None or len(id_lists) != len(lp_lists):
                logger.warning(
                    "[old_lp shortcut] completion_token_ids / completion_logprobs "
                    "list-of-lists length mismatch — falling back to HF forward."
                )
                return None
            for i, (lp, ids) in enumerate(zip(lp_lists, id_lists)):
                if ids is None or len(lp) != len(ids):
                    logger.warning(
                        "[old_lp shortcut] per-trajectory logprob/token-id length "
                        f"mismatch at i={i} (lp={len(lp) if lp is not None else None}, "
                        f"ids={len(ids) if ids is not None else None}) — falling "
                        "back to HF forward."
                    )
                    return None
            tensors: list[Any] = []
            for lp in lp_lists:
                # Empty completion (immediate EOS / 0 tokens) → empty tensor.
                # The trainer's per-token loss aggregation (sum / max_len)
                # contributes exactly 0 for empty tensors, which is the
                # correct no-op behaviour. Earlier code used a [0.0] sentinel,
                # but that adds a fake "token" whose IS ratio = exp(new−0) is
                # NOT a no-op and leaks A/max_len into the surrogate loss.
                if len(lp) == 0:
                    tensors.append(torch.empty(0, dtype=torch.float32, device=self.device))
                else:
                    tensors.append(torch.tensor(lp, dtype=torch.float32, device=self.device))
            return tensors

        a1_old_from_vllm = _logprobs_to_tensors(a1_pretok)
        a2_old_from_vllm = _logprobs_to_tensors(a2_pretok) if a2_pretok is not None else None
        fb_old_from_vllm = _logprobs_to_tensors(f1_pretok) if f1_pretok is not None else None

        # Determine whether we can take the vLLM-logprob shortcut for old_lp.
        if single_turn_a1:
            can_skip_old_pass = a1_old_from_vllm is not None
        else:
            can_skip_old_pass = (
                a1_old_from_vllm is not None
                and a2_old_from_vllm is not None
                and fb_old_from_vllm is not None
            )

        with torch.no_grad():
            if can_skip_old_pass:
                old_a1_lps_list = list(a1_old_from_vllm)
                if not single_turn_a1:
                    old_a2_lps_list = list(a2_old_from_vllm)
                    old_fb_lps_list = list(fb_old_from_vllm)
                logger.info(
                    "  old_lp shortcut: using vLLM sample-time logprobs "
                    f"({n_traj} trajectories, skipped HF old-pass)"
                )
            else:
                # Single-adapter: combine a1/a2/f1 into one forward pass
                # per mini-batch (3x fewer GPU ops). Multi-adapter mode
                # groups by adapter inside _routed_forward_multi — turns
                # that share an adapter still fuse into one forward.
                for mb_s in range(0, n_traj, inner_mini_bs):
                    mb_e = min(mb_s + inner_mini_bs, n_traj)
                    if single_turn_a1:
                        (a1_lps,) = self._routed_forward_multi(
                            [("a1", a1_pretok)], unwrapped_model, mb_s, mb_e
                        )
                        old_a1_lps_list += a1_lps
                    else:
                        a1_lps, a2_lps, fb_lps = self._routed_forward_multi(
                            [("a1", a1_pretok), ("a2", a2_pretok), ("f1", f1_pretok)],
                            unwrapped_model,
                            mb_s,
                            mb_e,
                        )
                        old_a1_lps_list += a1_lps
                        old_a2_lps_list += a2_lps
                        old_fb_lps_list += fb_lps

            # Ref log-probs: only needed when kl_coeff > 0.
            # Three cases:
            #   1. separate ref_model provided: use it directly (non-PEFT path).
            #   2. otherwise: disable adapter layers; KL anchors against base model.
            # The adapter state MUST be restored to enabled before leaving this
            # block — downstream code (training forward, vLLM sync, checkpoint
            # save) assumes "default" is active.
            if self.config.kl_coeff > 0:
                if self.ref_model is None:
                    unwrapped_model.disable_adapter_layers()
                    ref_m = unwrapped_model
                else:
                    ref_m = self.ref_model

                try:
                    for mb_s in range(0, n_traj, inner_mini_bs):
                        mb_e = min(mb_s + inner_mini_bs, n_traj)
                        if single_turn_a1:
                            (a1_lps,) = self._forward_from_pretokenized_multi(
                                [a1_pretok], ref_m, mb_s, mb_e
                            )
                            ref_a1_lps_list += a1_lps
                        else:
                            a1_lps, a2_lps, fb_lps = self._forward_from_pretokenized_multi(
                                [a1_pretok, a2_pretok, f1_pretok], ref_m, mb_s, mb_e
                            )
                            ref_a1_lps_list += a1_lps
                            ref_a2_lps_list += a2_lps
                            ref_fb_lps_list += fb_lps
                finally:
                    if self.ref_model is None:
                        unwrapped_model.enable_adapter_layers()

            # Per-token log-probs: keep as lists of variable-length tensors.
            if single_turn_a1:
                # A1-only baseline: only A1 log-probs participate in the loss.
                old_a1_lps = old_a1_lps_list
                old_a2_lps = None
                old_resp_lps = old_a1_lps_list
                old_fb_lps = None
            elif separate_turns:
                # Separate A1 and A2 for per-turn loss
                old_a1_lps = old_a1_lps_list
                old_a2_lps = old_a2_lps_list
                old_resp_lps = [
                    torch.cat([a1, a2]) for a1, a2 in zip(old_a1_lps_list, old_a2_lps_list)
                ]
                old_fb_lps = old_fb_lps_list
            else:
                old_a1_lps = None
                old_a2_lps = None
                old_resp_lps = [
                    torch.cat([a1, a2]) for a1, a2 in zip(old_a1_lps_list, old_a2_lps_list)
                ]
                old_fb_lps = old_fb_lps_list  # already list of per-token tensors

            if self.config.kl_coeff > 0:
                ref_a1_lps = ref_a1_lps_list
                if single_turn_a1:
                    ref_a2_lps = None
                    ref_fb_lps = None
                else:
                    ref_a2_lps = ref_a2_lps_list
                    ref_fb_lps = ref_fb_lps_list
            else:
                ref_a1_lps = None
                ref_a2_lps = None
                ref_fb_lps = None

        # No explicit cache clear -- PyTorch reuses memory automatically.

        clip_range = self.config.clip_range
        # DAPO Clip-Higher (arXiv:2503.14476): asymmetric PPO clipping. Upper
        # bound = 1 + clip_high (when set), lower bound = 1 - clip_range. This
        # lets positive-advantage tokens grow further before the clip cap fires,
        # amplifying the gradient of rare WR (wrong→right) flips relative to
        # the abundant zero-flip cases.
        _ch = float(getattr(self.config, "clip_high", 0.0) or 0.0)
        clip_low = clip_range
        clip_high = _ch if _ch > 0 else clip_range
        num_inner = self.config.num_inner_epochs

        # Gradient accumulation. With ``grad_acc > 1`` we treat N consecutive
        # outer ``train_step`` calls as a single optimization micro-cycle:
        #   * ``optimizer.zero_grad()`` fires at the START of the cycle
        #     (when ``global_step % grad_acc == 0``).
        #   * Each outer step's backward accumulates into the existing grads.
        #   * ``optimizer.step()`` + grad-clip fire at the END of the cycle
        #     (when ``(global_step + 1) % grad_acc == 0``).
        #   * Each loss is multiplied by ``1/grad_acc`` before backward so
        #     the sum of N micro-batches' gradients reproduces the magnitude
        #     of a single equivalent-batch backward (the "mean" convention,
        #     matching HF Trainer).
        #
        # DDP all-reduce still fires every outer step — accumulating across
        # ranks but reducing each rank's micro-step's gradient as it arrives
        # is correct and avoids a giant final all-reduce.
        #
        # Guard: ``num_inner > 1`` is PPO-style multiple-pass updates per
        # outer step. Combining it with grad-acc would require deciding
        # whether to accumulate across inner_epochs or across outer steps;
        # the current trainer only supports outer-step accumulation. We
        # explicitly forbid the combination rather than silently mis-stepping.
        grad_acc = max(1, int(getattr(self.config, "gradient_accumulation_steps", 1) or 1))
        if grad_acc > 1 and num_inner > 1:
            raise NotImplementedError(
                f"gradient_accumulation_steps={grad_acc} is only wired for "
                f"num_inner_epochs=1; got num_inner_epochs={num_inner}. To support "
                "both, decide whether accumulation runs across inner epochs or "
                "outer steps and gate the zero_grad/step calls accordingly."
            )
        is_cycle_start = self.global_step % grad_acc == 0
        is_cycle_end = (self.global_step + 1) % grad_acc == 0
        loss_scale = 1.0 / grad_acc

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
            # zero_grad only at the start of an accumulation cycle (and always
            # when grad_acc=1, which makes is_cycle_start always True).
            if is_cycle_start:
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

                if single_turn_a1:
                    (mb_a1_lps,) = self._routed_forward_multi(
                        [("a1", a1_pretok)],
                        inner_model,
                        mb_start,
                        mb_end,
                        accumulate_entropy=True,
                    )
                    mb_a2_lps = [None] * len(mb_a1_lps)
                    mb_fb_lps = [None] * len(mb_a1_lps)
                else:
                    mb_a1_lps, mb_a2_lps, mb_fb_lps = self._routed_forward_multi(
                        [("a1", a1_pretok), ("a2", a2_pretok), ("f1", f1_pretok)],
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
                # Per-adapter partial losses (only populated in the multi-turn
                # path). See _multi_adapter_backward for why these are tracked
                # separately from mb_loss when routing is enabled.
                mb_response_loss = None
                mb_feedback_loss = None
                for j in range(mb_end - mb_start):
                    ti = mb_start + j

                    fb_new = mb_fb_lps[j]

                    if single_turn_a1:
                        # Single-turn baseline: Dr.GRPO loss on A1 only.
                        # No A2 / F1 contribution — keeps the trainer reduced
                        # to vanilla GRPO so we can isolate algorithm bugs
                        # from multi-turn / two-reward composition issues.
                        a1_new = mb_a1_lps[j]
                        a1_ratio_raw = torch.nan_to_num(
                            a1_new - old_a1_lps[ti].detach(),
                            nan=0.0,
                            posinf=20.0,
                            neginf=-20.0,
                        )
                        a1_ratio = torch.exp(torch.clamp(a1_ratio_raw, -20.0, 20.0))
                        a1_clipped = torch.clamp(a1_ratio, 1 - clip_low, 1 + clip_high)
                        a1_max_len = float(self.config.rollout.a1_max_completion_length or 200)
                        traj_resp_loss = (
                            -torch.min(
                                a1_ratio * resp_advantages[ti], a1_clipped * resp_advantages[ti]
                            ).sum()
                            / a1_max_len
                        )
                        self._resp_total_tokens += a1_ratio.numel()
                        self._resp_clipped_tokens += (a1_ratio != a1_clipped).sum().item()
                        traj_fb_loss = torch.zeros((), device=a1_new.device, dtype=a1_new.dtype)

                        if self.config.kl_coeff > 0 and ref_a1_lps is not None:
                            a1_kl = _kl_term_drgrpo(ref_a1_lps[ti], a1_new, a1_max_len)
                            a1_kl_c = self.config.kl_coeff * getattr(
                                self.config, "a1_kl_coeff", 1.0
                            )
                            traj_kl_loss = a1_kl_c * a1_kl
                        else:
                            traj_kl_loss = torch.zeros((), device=a1_new.device, dtype=a1_new.dtype)

                        traj_loss = (traj_resp_loss + traj_fb_loss + traj_kl_loss) / n_traj_inner
                        mb_loss = traj_loss if mb_loss is None else mb_loss + traj_loss

                        epoch_resp_loss += traj_resp_loss.item() / n_traj_inner
                        epoch_fb_loss += 0.0
                        epoch_kl_loss += traj_kl_loss.item() / n_traj_inner
                        continue
                    elif separate_turns and a1_advantages is not None:
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
                        a1_clipped = torch.clamp(a1_ratio, 1 - clip_low, 1 + clip_high)
                        a2_clipped = torch.clamp(a2_ratio, 1 - clip_low, 1 + clip_high)

                        # Dr.GRPO aggregation (arXiv:2503.20783):
                        # sum over completion tokens / max_completion_length, NOT mean.
                        # This removes length bias — a 10-token answer contributes 10/max_len
                        # while a 200-token answer contributes 200/max_len, so the per-token
                        # gradient magnitude is consistent regardless of sequence length.
                        # TRL reference: (per_token_loss * mask).sum() / (batch * max_comp_len)
                        # Our per-sample call: sum / max_comp_len, then /n_traj below == /batch.
                        a1_max_len = float(self.config.rollout.a1_max_completion_length or 200)
                        a2_max_len = float(self.config.rollout.a2_max_completion_length or 200)
                        a1_loss = (
                            -torch.min(
                                a1_ratio * a1_advantages[ti], a1_clipped * a1_advantages[ti]
                            ).sum()
                            / a1_max_len
                        )
                        a2_loss = (
                            -torch.min(
                                a2_ratio * a2_advantages[ti], a2_clipped * a2_advantages[ti]
                            ).sum()
                            / a2_max_len
                        )

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
                        # Original: joint A1+A2 response loss (Dr.GRPO sum / max_len)
                        resp_new = torch.cat([mb_a1_lps[j], mb_a2_lps[j]])
                        resp_ratio_raw = torch.nan_to_num(
                            resp_new - old_resp_lps[ti].detach(), nan=0.0, posinf=20.0, neginf=-20.0
                        )
                        resp_log_ratio = torch.clamp(resp_ratio_raw, min=-20.0, max=20.0)
                        resp_ratio = torch.exp(resp_log_ratio)
                        resp_clipped_ratio = torch.clamp(resp_ratio, 1 - clip_low, 1 + clip_high)
                        resp_surr1 = resp_ratio * resp_advantages[ti]
                        resp_surr2 = resp_clipped_ratio * resp_advantages[ti]
                        resp_max_len = float(
                            self.config.rollout.a1_max_completion_length or 200
                        ) + float(self.config.rollout.a2_max_completion_length or 200)
                        traj_resp_loss = -torch.min(resp_surr1, resp_surr2).sum() / resp_max_len

                        self._resp_total_tokens += resp_ratio.numel()
                        self._resp_clipped_tokens += (resp_ratio != resp_clipped_ratio).sum().item()

                    # Feedback: per-token clipped surrogate (Dr.GRPO sum / max_len)
                    fb_ratio_raw = torch.nan_to_num(
                        fb_new - old_fb_lps[ti].detach(), nan=0.0, posinf=20.0, neginf=-20.0
                    )
                    fb_log_ratio = torch.clamp(fb_ratio_raw, min=-20.0, max=20.0)
                    fb_ratio = torch.exp(fb_log_ratio)
                    fb_clipped_ratio = torch.clamp(fb_ratio, 1 - clip_low, 1 + clip_high)
                    fb_surr1 = fb_ratio * fb_advantages[ti]
                    fb_surr2 = fb_clipped_ratio * fb_advantages[ti]
                    f1_max_len = float(self.config.rollout.f1_max_completion_length or 512)
                    traj_fb_loss = -torch.min(fb_surr1, fb_surr2).sum() / f1_max_len

                    # Track clip fraction
                    self._fb_total_tokens += fb_ratio.numel()
                    self._fb_clipped_tokens += (fb_ratio != fb_clipped_ratio).sum().item()

                    # KL loss with Dr.GRPO sum/max_len normalization (matches
                    # the policy loss aggregation, removing length bias on the
                    # KL anchor — see _kl_term_drgrpo docstring).
                    # SCoRe insight: strong A1 KL anchors first turn near base
                    # model, preventing "direct solution collapse" where the
                    # model just improves A1 instead of learning self-correction.
                    #
                    # NOTE: KL terms are tracked per-turn (a1_kl_term, a2_kl_term,
                    # f1_kl_term) — not just as a single traj_kl_loss — so that
                    # the multi-adapter backward path below can route each KL
                    # contribution to the adapter that owns its turn. The
                    # single-adapter path still consumes the summed traj_kl_loss.
                    if self.config.kl_coeff > 0 and ref_a1_lps is not None:
                        a1_kl_max_len = float(self.config.rollout.a1_max_completion_length or 200)
                        a2_kl_max_len = float(self.config.rollout.a2_max_completion_length or 200)
                        f1_kl_max_len = float(self.config.rollout.f1_max_completion_length or 512)

                        a1_kl = _kl_term_drgrpo(ref_a1_lps[ti], mb_a1_lps[j], a1_kl_max_len)
                        a2_kl = _kl_term_drgrpo(ref_a2_lps[ti], mb_a2_lps[j], a2_kl_max_len)
                        f1_kl = _kl_term_drgrpo(ref_fb_lps[ti], fb_new, f1_kl_max_len)

                        a1_kl_c = self.config.kl_coeff * getattr(self.config, "a1_kl_coeff", 1.0)
                        a2_kl_c = self.config.kl_coeff * getattr(self.config, "a2_kl_coeff", 1.0)
                        fb_kl_c = self.config.kl_coeff * getattr(self.config, "fb_kl_coeff", 1.0)

                        a1_kl_term = a1_kl_c * a1_kl
                        a2_kl_term = a2_kl_c * a2_kl
                        f1_kl_term = fb_kl_c * f1_kl
                        traj_kl_loss = a1_kl_term + a2_kl_term + f1_kl_term
                    else:
                        zero_kl = torch.tensor(0.0, device=fb_new.device)
                        a1_kl_term = zero_kl
                        a2_kl_term = zero_kl
                        f1_kl_term = zero_kl
                        traj_kl_loss = zero_kl

                    traj_loss = (traj_resp_loss + traj_fb_loss + traj_kl_loss) / n_traj_inner
                    mb_loss = traj_loss if mb_loss is None else mb_loss + traj_loss

                    # Per-adapter partial losses. With multi-adapter routing AND
                    # gradient checkpointing, a SINGLE backward on the combined
                    # ``mb_loss`` mis-attributes the gradient checkpointed text-
                    # decoder gradients: the checkpoint recomputes the forward
                    # at backward time using the model's CURRENT active adapter,
                    # so all text-decoder gradient lands on whichever adapter
                    # happened to be active when ``.backward()`` fires (regardless
                    # of which adapter was active during the original forward).
                    # Splitting the loss per adapter and calling ``.backward()``
                    # separately with the correct active adapter set before each
                    # call routes each turn's gradient to the right adapter.
                    #
                    # Loss decomposition (assumes a1+a2 share an adapter, f1
                    # uses its own — matches every routing configuration we ship):
                    #   response part:  a1_loss + a2_loss + a1_kl_term + a2_kl_term
                    #   feedback part:  f1_loss + f1_kl_term
                    # If a future routing puts a1/a2 on different adapters,
                    # the multi-adapter backward path below would need a 3-way
                    # split — guarded by an assertion in _multi_adapter_backward.
                    traj_response_part = (traj_resp_loss + a1_kl_term + a2_kl_term) / n_traj_inner
                    traj_feedback_part = (traj_fb_loss + f1_kl_term) / n_traj_inner
                    mb_response_loss = (
                        traj_response_part
                        if mb_response_loss is None
                        else mb_response_loss + traj_response_part
                    )
                    mb_feedback_loss = (
                        traj_feedback_part
                        if mb_feedback_loss is None
                        else mb_feedback_loss + traj_feedback_part
                    )

                    epoch_resp_loss += traj_resp_loss.item() / n_traj_inner
                    epoch_fb_loss += traj_fb_loss.item() / n_traj_inner
                    epoch_kl_loss += traj_kl_loss.item() / n_traj_inner

                self._multi_adapter_backward(
                    inner_model=inner_model,
                    mb_loss=mb_loss,
                    mb_response_loss=mb_response_loss,
                    mb_feedback_loss=mb_feedback_loss,
                    loss_scale=loss_scale,
                )

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
            #
            # When ``grad_acc > 1`` we want to clip and log the FULL accumulated
            # gradient — but the clip itself must only fire at the end of the
            # cycle, otherwise we'd clip mid-cycle gradients (already partial)
            # and the cycle's effective contribution would be artificially
            # capped. Compute the norm every micro-step for observability;
            # only clip + step at cycle end.
            grad_norm = torch.nn.utils.clip_grad_norm_(
                inner_model.parameters(),
                1.0 if is_cycle_end else float("inf"),
            ).item()

            # Linear LR warmup: 0 -> learning_rate over lr_warmup_steps. When
            # lr_warmup_steps == 0 (default), compute_warmup_lr returns
            # learning_rate at every step -> behavior bit-for-bit unchanged.
            warmup_steps = getattr(self.config, "lr_warmup_steps", 0)
            if warmup_steps > 0:
                effective_lr = compute_warmup_lr(
                    self.global_step, self.config.learning_rate, warmup_steps
                )
                for g in self.optimizer.param_groups:
                    g["lr"] = effective_lr

            # Skip optimizer step if any gradient contains NaN/inf. A NaN
            # anywhere in the accumulated gradients also poisons subsequent
            # micro-steps in the same cycle — we zero_grad to discard the
            # entire cycle's accumulation, then resume cleanly.
            has_nan_grad = not math.isfinite(grad_norm)
            if has_nan_grad:
                logger.warning(
                    f"  [inner epoch {inner_epoch + 1}] NaN/inf gradient detected -- "
                    "skipping optimizer step and discarding accumulated grads"
                )
                self.optimizer.zero_grad()
            elif is_cycle_end:
                self.optimizer.step()
            # else: mid-cycle micro-step — keep grads, wait for cycle end.

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

        # Reward component breakdown (averaged across trajectories).
        # A2-related components on the PAG path carry placeholder 0.0 for
        # gated trajectories (A2 never ran). Averaging those zeros into the
        # mean would understate the actual A2 component signal by exactly
        # the gating fraction. Detect PAG breakdowns via the ``gated``
        # attribute and skip A2 components for gated trajectories. A1 and
        # F1 components are valid for every trajectory (A1 always runs;
        # F1 always runs) and accumulate unconditionally.
        # Pre-declare the full PAG component key set so every rank has the
        # same keys in rollout_metrics regardless of trajectory mix. Without
        # this, a rank where ALL K trajectories gated would not register the
        # a2_correctness / a2_format / shaping_bonus keys at all, while a
        # rank with at least one non-gated trajectory would — causing the
        # per-rank tensor packed by _reduce_metrics_across_ranks to have
        # different lengths, which deadlocks NCCL ALLREDUCE.
        _resp_component_names = (
            "a1_correctness",
            "a1_format",
            "a2_correctness",
            "a2_format",
            "shaping_bonus",
            "no_regression",
            "wr_bonus",
        )
        _fb_component_names = ("verification", "format", "downstream")
        for name in _resp_component_names:
            rollout_metrics.setdefault(f"sr/resp_{name}_mean", [])
        for name in _fb_component_names:
            rollout_metrics.setdefault(f"sr/fb_{name}_mean", [])

        _a2_component_keys = ("a2_correctness", "a2_format", "shaping_bonus")
        for result in rollout_results:
            if result.response_breakdowns:
                for bd in result.response_breakdowns:
                    is_gated = bool(getattr(bd, "gated", False))
                    for comp_name, comp_val in bd.weighted_components.items():
                        if is_gated and comp_name in _a2_component_keys:
                            # Skip — gated A2 placeholder, would bias the mean.
                            continue
                        key = f"sr/resp_{comp_name}_mean"
                        rollout_metrics.setdefault(key, [])
                        rollout_metrics[key].append(comp_val)
            if result.feedback_breakdowns:
                for bd in result.feedback_breakdowns:
                    for comp_name, comp_val in bd.weighted_components.items():
                        key = f"sr/fb_{comp_name}_mean"
                        rollout_metrics.setdefault(key, [])
                        rollout_metrics[key].append(comp_val)
        # Average the component lists. Empty lists (which occur on ranks
        # where every trajectory was gated for a given A2 component) must
        # collapse to a scalar so the cross-rank reduce sees the same key
        # type on every rank — otherwise the all_reduce tensor sizes differ
        # and NCCL deadlocks at 30 min watchdog timeout.
        for key in list(rollout_metrics.keys()):
            if key.startswith("sr/resp_") and key.endswith("_mean"):
                val = rollout_metrics[key]
                if isinstance(val, list):
                    rollout_metrics[key] = (sum(val) / len(val)) if val else 0.0
            if key.startswith("sr/fb_") and key.endswith("_mean"):
                val = rollout_metrics[key]
                if isinstance(val, list):
                    rollout_metrics[key] = (sum(val) / len(val)) if val else 0.0

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
        # Per-turn means (SCoRe-style), only populated when separate_turn_loss.
        # Under PAG selective revision, gated trajectories carry a 0.0 placeholder
        # in a2_rewards_t (gated → A2 didn't run). Naive mean over all N·K would
        # mechanically depress a2_reward_mean as F1's gate fires more often,
        # disconnecting the metric from actual A2 quality. Compute a2_reward_mean
        # over the active (non-gated) trajectories only, mirroring sr/r_a2_mean
        # in rollout.compute_self_reflection_metrics. (Bug #3 fix.)
        if separate_turns and a1_rewards_t is not None and a2_rewards_t is not None:
            a1_reward_global = a1_rewards_t.mean().item()
            if use_pag:
                a2_sum_local = float(a2_rewards_t[a2_active_mask].sum().item())
                a2_count_local = float(a2_active_mask.sum().item())
            else:
                a2_sum_local = float(a2_rewards_t.sum().item())
                a2_count_local = float(a2_rewards_t.numel())
            a2_reward_global = (
                a2_sum_local / a2_count_local if a2_count_local > 0 else float("nan")
            )
        else:
            a1_reward_global = float("nan")
            a2_reward_global = float("nan")
            a2_sum_local = 0.0
            a2_count_local = 0.0
        if _dist_is_initialized():
            import torch as _torch

            vals = _torch.tensor(
                [
                    resp_reward_global,
                    fb_reward_global,
                    a1_reward_global if not math.isnan(a1_reward_global) else 0.0,
                    a2_sum_local,
                    a2_count_local,
                ],
                device=self.device,
                dtype=_torch.float32,
            )
            _torch.distributed.all_reduce(vals, op=_torch.distributed.ReduceOp.SUM)
            world_size = _torch.distributed.get_world_size()
            resp_reward_global = float(vals[0].item()) / world_size
            fb_reward_global = float(vals[1].item()) / world_size
            if separate_turns:
                a1_reward_global = float(vals[2].item()) / world_size
                # Globally-correct mean = sum_across_ranks / count_across_ranks
                # (NOT the average of per-rank means, which would skew when
                # gate rates differ across ranks).
                a2_sum_global = float(vals[3].item())
                a2_count_global = float(vals[4].item())
                a2_reward_global = (
                    a2_sum_global / a2_count_global
                    if a2_count_global > 0
                    else float("nan")
                )

        return SelfReflectionTrainStepResult(
            loss=(total_resp_loss + total_fb_loss + total_kl_loss) / num_inner,
            response_loss=total_resp_loss / num_inner,
            feedback_loss=total_fb_loss / num_inner,
            kl_loss=total_kl_loss / num_inner,
            response_reward_mean=resp_reward_global,
            feedback_reward_mean=fb_reward_global,
            a1_reward_mean=a1_reward_global,
            a2_reward_mean=a2_reward_global,
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
                adapter_callback=self._build_adapter_callback(gen_model),
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
        completion_token_ids: Optional[list[Optional[list[int]]]] = None,
        completion_logprobs: Optional[list[Optional[list[float]]]] = None,
        sampling_temperature: float = 1.0,
    ) -> dict:
        """Pre-compute chat-template strings and token lengths for a trajectory batch.

        Called once before the inner optimization loop so that
        apply_chat_template + per-sequence tokenizer calls are not repeated on
        every inner epoch mini-batch forward pass.

        When the rollout engine provided ``completion_token_ids`` for every
        trajectory in the batch, the pretok dict is configured for the
        **native-token** path: ``full_lens`` is computed from
        ``prompt_lens + len(vllm_completion_ids)`` rather than from
        retokenizing the full text, and the forward pass assembles
        ``input_ids = prompt_ids ++ vllm_completion_ids`` directly — bypassing
        the lossy ``apply_chat_template + tokenize`` round-trip on the
        completion text (audit Bug 2). This is the only path supported under
        ``--use_vllm``; mixed batches where any trajectory lacks
        ``completion_token_ids`` will not enable the native flag and the
        forward pass will raise.

        Args:
            messages_list: N full message lists (prompt + assistant completion)
            images: N PIL Images (one per sequence, may be None)
            completion_token_ids: N-list of actual completion token ids
                emitted by the rollout engine. Required for the native path.
            completion_logprobs: N-list of per-token sampled logprobs aligned
                with ``completion_token_ids``. Stored on the pretok dict so
                the trainer can use them as ``old_lp`` directly, skipping one
                HF forward pass per step.
            sampling_temperature: Temperature the rollout engine used at
                sample time for these trajectories. vLLM's per-step logprobs
                live on ``log_softmax(logits / T)``, so the HF forward pass
                that computes ``new_lp`` / ``ref_lp`` MUST apply the same
                divisor to keep the PPO ratio ``exp(new_lp - old_lp) = 1`` at
                step 0 (before any policy update). A1/F1 use T=1.0 (no-op
                divisor); A2 typically uses T=0.7 in the active YAMLs and
                this is the value that previously caused a systematic ratio
                bias on A2 tokens.

        Returns:
            dict with keys:
                full_texts: list[str] of N full chat-template strings
                prompt_texts: list[str] of N prompt-only chat-template strings
                prompt_lens: list[int] of N prompt token counts (text-only
                    tokenization, used to locate the prompt/completion seam
                    in input_ids during the forward pass)
                full_lens: list[int] of N full sequence token counts
                images: the same images list (stored for convenience)
                completion_token_ids: same list as the argument (or None)
                completion_logprobs: same list as the argument (or None)
                native_path: bool — True iff every trajectory has a non-None
                    completion_token_ids. The forward pass uses this flag to
                    pick the assembly mode.
                sampling_temperature: float — vLLM sampling temperature for
                    this set, plumbed into the HF forward as a log-softmax
                    divisor so old_lp (vLLM) and new_lp/ref_lp (HF) sit on
                    the same distribution.
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

        # Decide path. Native path requires every trajectory to carry a
        # non-None completion_token_ids list. When any trajectory lacks one
        # (e.g. a mixed HF-fallback batch in tests), the legacy retokenize
        # path is used. Under ``--use_vllm`` the production rollout always
        # emits completion_token_ids end-to-end, so the legacy branch below
        # exists only for test stubs and is not exercised in real training.
        all_have_ids = (
            completion_token_ids is not None
            and len(completion_token_ids) == len(prompt_lens)
            and all(c is not None for c in completion_token_ids)
        )
        native_path = all_have_ids

        if native_path:
            # Defense: vLLM samples from the full vocabulary, including the
            # vision-pad / vision-marker special tokens that the model uses
            # to mark image regions in the prompt. Probability is tiny per
            # step but over thousands of trajectories we eventually see one.
            #
            # If a sampled completion contains an extra ``<|image_pad|>``
            # token (151655 in Qwen2.5-VL), the assembled
            # ``prompt_ids ++ completion_ids`` then has more image-pad
            # tokens than the prompt's ``image_grid_thw`` features can fill,
            # and Qwen2.5-VL's ``get_placeholder_mask`` raises
            # ``ValueError: Image features and image tokens do not match``
            # — taking the entire training step and the whole pod down with
            # it (this surfaced after ~12h on the production
            # ``single-fb-only`` run; the legacy retokenize path didn't
            # have this exposure because text → tokens via
            # ``apply_chat_template`` doesn't preserve image-pad as the
            # special token).
            #
            # Strip vision special tokens from the completion before the
            # length budget is computed; keep ``completion_logprobs``
            # paired so the old_lp shortcut tensor stays aligned with the
            # filtered ids the forward pass will see.
            vision_special_ids = set()
            for tok_name in (
                "<|image_pad|>",
                "<|video_pad|>",
                "<|vision_pad|>",
                "<|vision_start|>",
                "<|vision_end|>",
            ):
                tok_id = self.processor.tokenizer.convert_tokens_to_ids(tok_name)
                if isinstance(tok_id, int) and tok_id != self.processor.tokenizer.unk_token_id:
                    vision_special_ids.add(tok_id)

            if vision_special_ids:
                filtered_ids: list[list[int]] = []
                filtered_lps: list[Optional[list[float]]] = []
                total_dropped = 0
                for i, ids in enumerate(completion_token_ids):
                    lps_i = completion_logprobs[i] if completion_logprobs else None
                    if lps_i is not None and len(lps_i) == len(ids):
                        kept = [(t, lp) for t, lp in zip(ids, lps_i) if t not in vision_special_ids]
                        f_ids = [t for t, _ in kept]
                        f_lps = [lp for _, lp in kept]
                    else:
                        f_ids = [t for t in ids if t not in vision_special_ids]
                        f_lps = lps_i
                    total_dropped += len(ids) - len(f_ids)
                    filtered_ids.append(f_ids)
                    filtered_lps.append(f_lps)
                if total_dropped > 0:
                    logger.warning(
                        f"[native loss] dropped {total_dropped} vision-special "
                        "token(s) sampled by vLLM mid-completion across "
                        f"{len(completion_token_ids)} trajectories — would have "
                        "caused image_grid_thw / image_pad count mismatch in "
                        "the HF forward pass."
                    )
                completion_token_ids = filtered_ids
                if completion_logprobs is not None:
                    completion_logprobs = filtered_lps

            # Length budget comes directly from vLLM. We DO NOT retokenize the
            # full text — the assembled input_ids in the forward pass will be
            # ``prompt_ids ++ vllm_completion_ids`` with len == prompt_len +
            # len(completion_ids).
            full_lens = [pl + len(c) for pl, c in zip(prompt_lens, completion_token_ids)]
        else:
            # Legacy retokenize fallback. Reached only when a trajectory
            # lacks ``completion_token_ids`` (test stubs / future rollout
            # paths that don't emit ids). Production vLLM and HF rollouts
            # always emit token_ids, so the native_path branch above is the
            # only path exercised in real training. ``full_lens`` here
            # reflects what ``apply_chat_template + tokenize`` produces for
            # the full text, which may differ from len(prompt) +
            # len(completion) due to non-bijective BPE round-trips.
            full_enc = self.processor.tokenizer(
                full_texts, padding=False, return_attention_mask=False
            )
            full_lens = [len(ids) for ids in full_enc["input_ids"]]

        return {
            "full_texts": full_texts,
            "prompt_texts": prompt_texts,
            "prompt_lens": prompt_lens,
            "full_lens": full_lens,
            "images": images,
            "completion_token_ids": completion_token_ids,
            "completion_logprobs": completion_logprobs,
            "native_path": native_path,
            "sampling_temperature": float(sampling_temperature),
        }

    def _routed_forward_multi(
        self,
        turn_pretok_pairs: list[tuple[str, dict]],
        model: Any,
        mb_start: int = 0,
        mb_end: Optional[int] = None,
        accumulate_entropy: bool = False,
    ) -> list[list[Any]]:
        """Adapter-routing-aware wrapper around ``_forward_from_pretokenized_multi``.

        In single-adapter mode (``self._routing.enabled == False``), behaves
        exactly like calling the underlying multi-forward on the pretokenized
        sets in the given order — one batched forward pass over all turns.

        In multi-adapter mode, groups the input pairs by their resolved
        adapter (``self._routing.adapter_for_turn(turn)``). For each group:
          1. Activates the adapter via ``_set_active_adapter``.
          2. Runs ``_forward_from_pretokenized_multi`` on the group's
             pretoks (one fused forward per adapter, preserving the
             batching efficiency when several turns share an adapter).
          3. Stores results indexed back to the caller's original turn order.

        This means a typical response/feedback split (A1+A2 share an
        adapter, F1 has its own) collapses to 2 forward passes per
        mini-batch — same per-token gradient semantics, only ~33% more
        forward overhead than the fused single-adapter path.

        After all groups have run, the active adapter is restored to the
        first trainable spec in the routing list so any downstream code
        that assumes a known active adapter (e.g. the next mini-batch)
        sees a stable value.

        Args:
            turn_pretok_pairs: List of (turn_name, pretok_dict). turn_name
                is one of ``"a1"`` / ``"f1"`` / ``"a2"``. Order is preserved
                in the output.
            model: Unwrapped PEFT model to run forward on.
            mb_start: Mini-batch start index (passed through).
            mb_end: Mini-batch end index (passed through).
            accumulate_entropy: Whether the forward should accumulate
                per-token entropy stats (passed through).

        Returns:
            List of len(turn_pretok_pairs) sublists — each sublist holds
            the per-trajectory log-prob tensors for that input set, in
            input order.
        """
        if not self._routing.enabled:
            pretok_list = [pt for _, pt in turn_pretok_pairs]
            return self._forward_from_pretokenized_multi(
                pretok_list, model, mb_start, mb_end, accumulate_entropy
            )

        # Group input indices by resolved adapter, preserving input order
        # within each group.
        adapter_to_indices: dict[str, list[int]] = {}
        for i, (turn, _) in enumerate(turn_pretok_pairs):
            adapter = self._routing.adapter_for_turn(turn)
            adapter_to_indices.setdefault(adapter, []).append(i)

        results: list[Optional[list[Any]]] = [None] * len(turn_pretok_pairs)
        for adapter_name, indices in adapter_to_indices.items():
            self._set_active_adapter(model, adapter_name)
            group_pretoks = [turn_pretok_pairs[i][1] for i in indices]
            group_outs = self._forward_from_pretokenized_multi(
                group_pretoks, model, mb_start, mb_end, accumulate_entropy
            )
            for local_idx, original_idx in enumerate(indices):
                results[original_idx] = group_outs[local_idx]

        # NOTE: we used to "restore active adapter to first trainable" here so
        # the next mini-batch / save / KL pass observed a stable starting
        # point. That cleanup is the trigger for the grad-ckpt mis-attribution
        # bug — see ``_multi_adapter_backward`` for details. We deliberately
        # LEAVE the active adapter at whichever group ran last; the trainer's
        # per-adapter backward path resets it explicitly before each
        # ``.backward()``, and the next mini-batch's first
        # ``_set_active_adapter`` call also resets it.

        # All slots filled by the group loop above; assert defensively.
        assert all(r is not None for r in results), "routed forward lost a turn"
        return results  # type: ignore[return-value]

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
        all_prompt_texts: list[str] = []
        all_prompt_lens: list[int] = []
        all_full_lens: list[int] = []
        all_imgs: list[Any] = []
        all_completion_ids: list[Optional[list[int]]] = []
        set_sizes: list[int] = []
        # Per-trajectory sampling temperature, expanded from each pretok's
        # set-level value. vLLM returns logprobs computed on
        # ``log_softmax(logits / T)``, so the HF forward below MUST divide
        # ``shift_logits`` by the same ``T`` before its own ``log_softmax`` to
        # keep ``old_lp`` (vLLM) and ``new_lp`` / ``ref_lp`` (HF) on the same
        # distribution. A1 and F1 use T=1.0 (divisor is a no-op); A2 typically
        # uses T=0.7. Without this divisor the PPO ratio
        # ``exp(new_lp - old_lp)`` is biased ≠ 1 even at step 0 for A2 tokens
        # — see the bug discussion in the commit that introduced this code.
        per_traj_temperatures: list[float] = []
        # Native path requires every input pretok to be flagged native_path.
        # If any pretok is legacy, we fall back to the legacy retokenize
        # forward for the whole batch (avoids mixing assembly modes inside
        # one forward pass).
        native_path = all(bool(pt.get("native_path", False)) for pt in pretok_list)

        for pretok in pretok_list:
            _end = mb_end if mb_end is not None else len(pretok["full_texts"])
            sl = slice(mb_start, _end)
            size = _end - mb_start
            set_sizes.append(size)
            all_full_texts.extend(pretok["full_texts"][sl])
            all_prompt_texts.extend(pretok.get("prompt_texts", pretok["full_texts"])[sl])
            all_prompt_lens.extend(pretok["prompt_lens"][sl])
            all_full_lens.extend(pretok["full_lens"][sl])
            all_imgs.extend(pretok["images"][sl])
            comp_ids_list = pretok.get("completion_token_ids")
            if comp_ids_list is None:
                all_completion_ids.extend([None] * size)
            else:
                all_completion_ids.extend(comp_ids_list[sl])
            # Default to 1.0 if the pretok was built without the field
            # (legacy test stubs). 1.0 makes the divisor a no-op.
            temp = float(pretok.get("sampling_temperature", 1.0))
            if not (temp > 0.0):
                raise ValueError(
                    f"_forward_from_pretokenized_multi: sampling_temperature must be > 0, got {temp}"
                )
            per_traj_temperatures.extend([temp] * size)

        n = len(all_full_texts)
        has_image = any(img is not None for img in all_imgs)

        orig_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"
        try:
            if native_path:
                # Native path: feed the processor PROMPT text only (so
                # ``input_ids`` covers the prompt with image tokens correctly
                # placed and ``pixel_values`` / ``image_grid_thw`` are aligned
                # with those positions). We then concatenate vLLM's actual
                # sampled completion ids — bypassing the lossy retokenize on
                # completion text (Bug 2 fix).
                proc_text = all_prompt_texts
            else:
                proc_text = all_full_texts
            if has_image:
                batch_inputs = self.processor(
                    text=proc_text,
                    images=all_imgs,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
            else:
                batch_inputs = self.processor(
                    text=proc_text,
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
        finally:
            self.processor.tokenizer.padding_side = orig_side

        if native_path:
            # The processor returned (n, max_prompt_len) input_ids, left-padded.
            # We need to attach vLLM completion tokens to each row and re-pad.
            #
            # Procedure per trajectory i:
            #   - take the unpadded prompt slice from input_ids[i]
            #     (left-pad means real tokens sit at the right edge of the
            #     padded row; the unpadded length equals
            #     attention_mask[i].sum())
            #   - append completion_token_ids[i]
            #   - the resulting full sequence length is
            #     unpadded_prompt_len + len(completion_ids)
            # We then left-pad the batch back to the new max sequence length.
            # ``pixel_values`` and ``image_grid_thw`` are unchanged — they
            # depend on images, not on the text suffix.
            prompt_input_ids = batch_inputs["input_ids"]
            prompt_attn = batch_inputs["attention_mask"]
            # Use the tokenizer's pad token, falling back to EOS only when
            # the tokenizer explicitly aliases them (Qwen2.5-VL does:
            # pad_token_id == eos_token_id == 151645). We never silently fall
            # back to 0 — for Qwen that is the UNK token, and using it as
            # padding would corrupt the embedding lookup at attention-masked
            # positions that downstream layers may still touch (e.g., layernorm
            # statistics over the unmasked-but-padded slots in some kernels).
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.processor.tokenizer.eos_token_id
            assert pad_token_id is not None, (
                "Tokenizer has neither pad_token_id nor eos_token_id set — "
                "cannot construct the native-path padded batch safely. "
                "Configure the tokenizer with an explicit pad token."
            )

            unpadded_prompt_lens = prompt_attn.sum(dim=1).tolist()
            full_unpad_lens = [
                int(unpadded_prompt_lens[i]) + len(all_completion_ids[i]) for i in range(n)
            ]
            max_full = max(full_unpad_lens) if full_unpad_lens else 0
            batched_ids = torch.full(
                (n, max_full),
                fill_value=pad_token_id,
                dtype=prompt_input_ids.dtype,
                device=self.device,
            )
            batched_attn = torch.zeros((n, max_full), dtype=prompt_attn.dtype, device=self.device)
            for i in range(n):
                pl = int(unpadded_prompt_lens[i])
                cl = len(all_completion_ids[i])
                seq_len = pl + cl
                # Slice the unpadded prompt out of the left-padded row.
                padded_prompt_width = prompt_input_ids.shape[1]
                prompt_unpad = prompt_input_ids[i, padded_prompt_width - pl : padded_prompt_width]
                completion_t = torch.tensor(
                    all_completion_ids[i],
                    dtype=prompt_input_ids.dtype,
                    device=self.device,
                )
                seq = torch.cat([prompt_unpad, completion_t], dim=0)
                # Left-pad the assembled sequence into the batch row.
                batched_ids[i, max_full - seq_len :] = seq
                batched_attn[i, max_full - seq_len :] = 1

            forward_kwargs: dict[str, Any] = {
                "input_ids": batched_ids,
                "attention_mask": batched_attn,
                "use_cache": False,
            }
            # Forward Qwen2.5-VL specific image tensors when present.
            if "pixel_values" in batch_inputs:
                forward_kwargs["pixel_values"] = batch_inputs["pixel_values"]
            if "image_grid_thw" in batch_inputs:
                forward_kwargs["image_grid_thw"] = batch_inputs["image_grid_thw"]
            outputs = model(**forward_kwargs)
            logits = outputs.logits  # (n, max_full, vocab_size)
            del outputs

            total_len = max_full
            # Per-trajectory full_len is unpadded_prompt_len + completion_len.
            effective_full_lens = full_unpad_lens
            # Per-trajectory prompt_len is unpadded_prompt_len (image-token-
            # expanded, NOT the text-only prompt_lens recorded in pretok —
            # those are used only by the legacy path).
            effective_prompt_lens = [int(x) for x in unpadded_prompt_lens]
            # Use the assembled batch as the source of shift_labels.
            label_source = batched_ids
        else:
            outputs = model(**batch_inputs, use_cache=False)
            logits = outputs.logits  # (n, padded_seq_len, vocab_size)
            del outputs  # free non-logit activations early
            total_len = batch_inputs["input_ids"].shape[1]
            effective_full_lens = all_full_lens
            effective_prompt_lens = all_prompt_lens
            label_source = batch_inputs["input_ids"]

        all_lps: list[Any] = []
        for i in range(n):
            pad_len = total_len - effective_full_lens[i]
            real_prompt_start = pad_len + effective_prompt_lens[i]
            shift_logits = logits[i, real_prompt_start - 1 : -1, :]
            shift_labels = label_source[i, real_prompt_start:]
            shift_logits = torch.nan_to_num(shift_logits, nan=0.0, posinf=1e4, neginf=-1e4)
            # Apply the per-turn sampling temperature divisor so HF new_lp lives on
            # the same softmax-temperature distribution as vLLM's sample-time old_lp.
            # Without this, the A2 PPO ratio is systematically biased (Bug #1):
            # vLLM returns log softmax(logits / 0.7) at sample time, but HF computes
            # log softmax(logits) here, so exp(new_lp − old_lp) ≠ 1 even at step 0.
            # No-op for A1 / F1 (T=1.0).
            temp_i = per_traj_temperatures[i]
            if temp_i != 1.0:
                shift_logits = shift_logits / temp_i
            lp = F.log_softmax(shift_logits, dim=-1)
            token_lp = lp.gather(1, shift_labels.unsqueeze(-1)).squeeze(-1)
            # Empty completion → empty tensor (NOT a fake [0.0] token). The
            # per-token loss / KL aggregations are `sum / max_len` which
            # cleanly evaluates to 0 for an empty tensor. Earlier code used
            # a [0.0] sentinel here, which paired with the matching sentinel
            # in `_logprobs_to_tensors` to inject an `A/max_len` loss
            # contribution for every empty trajectory. See bug audit Bug-2-b.
            all_lps.append(
                token_lp
                if token_lp.numel() > 0
                else torch.empty(0, dtype=torch.float32, device=self.device)
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

    def _compute_pag_a2_advantages(
        self,
        rewards: Any,
        active_mask: Any,
        k: int,
        loss_type: str = "grpo",
    ) -> Any:
        """Per-segment K-group baseline for the PAG A2 turn.

        Mirrors ``_compute_group_advantages`` but the K-group mean/std are
        computed ONLY over trajectories where ``active_mask=True`` (i.e. the
        selective-revision gate did NOT stop the trajectory at F1). Gated
        trajectories receive advantage = 0 — combined with their empty A2
        completion this contributes zero to the A2 policy loss.

        Args:
            rewards: Tensor of all A2 rewards (shape [N*K]). Entries for
                gated trajectories are placeholders (0.0) and are ignored
                by the baseline computation.
            active_mask: Boolean tensor of shape [N*K], True iff the
                trajectory generated A2 (NOT gated).
            k: K-group size.
            loss_type: ``"grpo"`` for mean+std, ``"dr_grpo"`` for mean only.

        Returns:
            Tensor of advantages, shape [N*K]. Gated positions = 0.
        """
        import torch

        n_total = len(rewards)
        n_groups = n_total // k
        advantages = torch.zeros_like(rewards)

        for i in range(n_groups):
            start = i * k
            end = start + k
            group = rewards[start:end]
            mask = active_mask[start:end]
            n_active = int(mask.sum().item())
            if n_active == 0:
                # Every trajectory in this group gated out at F1. No A2
                # gradient signal — leave advantages at zero.
                continue
            active_vals = group[mask]
            mean = active_vals.mean()
            if loss_type == "dr_grpo":
                # Subtract mean; gated positions stay at zero advantage
                # (their placeholder reward 0.0 would otherwise produce a
                # spurious negative advantage of -mean if we let it through).
                adv = group - mean
            else:
                # Population std over active members. n_active=1 → std=0;
                # mirror the legacy "std<=0 → advantage 0" behaviour.
                std = active_vals.std(correction=0) if n_active > 1 else torch.zeros_like(mean)
                if std.item() > 0:
                    adv = (group - mean) / (std + 1e-8)
                else:
                    adv = torch.zeros_like(group)
            # Force gated positions to 0 regardless of how the active-mean
            # subtraction lands on their placeholder reward.
            advantages[start:end] = torch.where(mask, adv, torch.zeros_like(adv))

        # Handle ragged tail (n_total % k != 0). Same logic.
        remaining = n_total - n_groups * k
        if remaining > 0:
            start = n_groups * k
            group = rewards[start:]
            mask = active_mask[start:]
            n_active = int(mask.sum().item())
            if n_active > 0:
                active_vals = group[mask]
                mean = active_vals.mean()
                if loss_type == "dr_grpo":
                    adv = group - mean
                else:
                    std = active_vals.std(correction=0) if n_active > 1 else torch.zeros_like(mean)
                    if std.item() > 0:
                        adv = (group - mean) / (std + 1e-8)
                    else:
                        adv = torch.zeros_like(group)
                advantages[start:] = torch.where(mask, adv, torch.zeros_like(adv))

        return advantages

    def _compute_gdpo_advantages(
        self,
        components: Any,  # tensor of shape [N*K, n_components]
        weights: Any,  # tensor of shape [n_components]
        k: int,
    ) -> Any:
        """Compute GDPO advantages: per-component K-group normalize, weighted
        sum, then batch-renormalize (Liu et al. 2026, arXiv:2601.05242, Eqs. 4-7).

        For each prompt's K-group of size k:
          1. For each component j: A_j_i = (r_j_i - mean_k) / (std_k + eps)
          2. A_sum_i = sum_j(w_j * A_j_i)
          3. Batch-renormalize across all N*K samples:
             Â_i = (A_sum_i - batch_mean) / (batch_std + eps)

        Components with std=0 in the group contribute 0 to the advantage
        (no information), implemented via std clamp.

        Args:
            components: Tensor of raw per-component rewards, shape [N*K, n_components]
            weights: Per-component weight multipliers, shape [n_components]
            k: K-group size

        Returns:
            Tensor of GDPO advantages, shape [N*K]
        """
        import torch

        eps = 1e-8
        n_total, n_components = components.shape
        n_groups = n_total // k

        # Step 1+2: per-component K-group normalize, then weighted sum.
        # We compute per-group means/stds across the K dimension.
        # NOTE: ``correction=0`` selects the population (ML) std rather than
        # torch's default Bessel-corrected sample std (ddof=1). At our
        # batch_size*K = 24, ddof=1 inflates the denominator ~2-4% vs ddof=0,
        # which slightly dampens advantages. The GDPO paper (Liu 2026,
        # arXiv:2601.05242) writes the per-group normalisation in standard
        # population-variance form, so we match that.
        a_sum = torch.zeros(n_total, dtype=components.dtype, device=components.device)
        # Track per-group non-degeneracy so the batch-renorm step can detect
        # the "most groups dead, one alive" trap that caused the production
        # grad explosion (see comment on the batch-renorm guard below).
        # A group is "alive" if ANY component had non-zero std.
        group_alive = torch.zeros(n_groups, dtype=torch.bool, device=components.device)
        for gi in range(n_groups):
            start = gi * k
            end = start + k
            group = components[start:end]  # [k, n_components]
            mean = group.mean(dim=0, keepdim=True)  # [1, n_components]
            std = group.std(dim=0, keepdim=True, correction=0)  # [1, n_components]
            # std=0 components: clamp the denominator + mask via torch.where
            # rather than materialising `float("inf")` in the graph. The
            # outcome is numerically identical (normalized = 0 for the
            # degenerate component) but stays well-behaved under
            # torch.autograd.set_detect_anomaly debugging.
            std_clamped = torch.clamp(std, min=eps)
            normalized = torch.where(
                std > eps,
                (group - mean) / std_clamped,
                torch.zeros_like(group),
            )
            # Weighted sum across components
            a_sum[start:end] = (normalized * weights.unsqueeze(0)).sum(dim=1)
            group_alive[gi] = (std > eps).any()

        # Handle remainder (tail not divisible by k) — same pattern. No
        # alive-counting needed since the remainder is single-trajectory
        # tail and feeds the same batch-renorm decision.
        remaining = n_total - n_groups * k
        if remaining > 0:
            start = n_groups * k
            group = components[start:]
            mean = group.mean(dim=0, keepdim=True)
            std = group.std(dim=0, keepdim=True, correction=0)
            std_clamped = torch.clamp(std, min=eps)
            normalized = torch.where(
                std > eps,
                (group - mean) / std_clamped,
                torch.zeros_like(group),
            )
            a_sum[start:] = (normalized * weights.unsqueeze(0)).sum(dim=1)

        # Step 3: batch-wide renormalization. Same population-std convention.
        # GUARD against the GDPO grad-explosion trap discovered in production
        # (commit 8ee2b27 run: feedback grad_norm spiked to 50k). When the
        # response head saturates, almost every K-group has std=0 on every
        # component → a_sum is mostly zero → batch_std collapses to the
        # spread of the ONE surviving non-degenerate group divided by
        # ~sqrt(N*K). Dividing the surviving group's a_sum by that tiny
        # batch_std then amplifies its advantage by ~sqrt(n_groups / n_alive),
        # driving grad_norm into the hundreds-to-thousands range.
        #
        # Fix: skip batch-renorm ONLY when (a) at least one group is
        # degenerate AND (b) fewer than 2 groups survived. The healthy
        # paths — all groups alive, or single-group batches — keep the
        # original batch-renorm behaviour bit-for-bit.
        n_alive = int(group_alive.sum().item())
        n_dead = n_groups - n_alive
        batch_mean = a_sum.mean()
        if n_dead > 0 and n_alive < 2:
            return a_sum - batch_mean
        batch_std = a_sum.std(correction=0)
        if batch_std > eps:
            return (a_sum - batch_mean) / (batch_std + eps)
        return a_sum - batch_mean

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

    def _multi_adapter_backward(
        self,
        inner_model: Any,
        mb_loss: Any,
        mb_response_loss: Any,
        mb_feedback_loss: Any,
        loss_scale: float = 1.0,
    ) -> None:
        """Call ``.backward()`` on the mini-batch loss, splitting across
        adapters when routing is enabled.

        Background — the bug this fixes:

        When gradient checkpointing is enabled on the text decoder AND multiple
        LoRA adapters route per turn, a single combined ``mb_loss.backward()``
        does NOT correctly attribute text-decoder gradients to the adapter that
        was active during the original forward.

        The reason: ``torch.utils.checkpoint`` (used by HF
        ``gradient_checkpointing_enable``) recomputes the wrapped function's
        forward during backward. That recompute reads the model's CURRENT
        state — including the active LoRA adapter — instead of the state at
        original-forward time. So every text-decoder gradient lands on
        whichever adapter is currently active, regardless of which adapter
        ran the original forward.

        We verified this empirically: with the response/feedback split, every
        F1 forward (feedback active) was followed by a "restore to first
        trainable = response" cleanup; backward then recomputed text-decoder
        forwards with response active, and feedback's text-side LoRA received
        zero gradient across 250 steps × 4 ranks (optimizer state confirms
        only 400 of 1432 LoRA params ever received a gradient — exactly the
        296 response merger+text plus 4 feedback merger, no feedback text).

        Workaround — split the loss per adapter and backward each separately
        with the correct active adapter set BEFORE the call:

          1. Backward ``mb_response_loss`` with the response-adapter active.
             Its recompute uses response → response's text LoRA gets gradient.
          2. Backward ``mb_feedback_loss`` with the feedback-adapter active.
             Its recompute uses feedback → feedback's text LoRA gets gradient.

        Single-adapter mode falls through to the combined ``mb_loss.backward()``
        (no split needed because there is only one set of LoRA tensors).

        Args:
            inner_model: Unwrapped PEFT model (used for ``_set_active_adapter``).
            mb_loss: The combined trajectory loss summed across turns. Used in
                single-adapter mode (or when all turns route to the same
                adapter, e.g. for the single-turn-A1 baseline).
            mb_response_loss: Sum of A1+A2 policy losses + their KL terms.
                ``None`` when single_turn_a1 (only ``mb_loss`` is populated).
            mb_feedback_loss: F1 policy loss + F1 KL term. ``None`` when
                single_turn_a1.
            loss_scale: Scalar multiplied into every loss before
                ``.backward()``. Used by the grad-accumulation wrapper to
                divide each micro-step's loss by ``gradient_accumulation_steps``
                so that the sum of N accumulated backward()s reproduces the
                same gradient magnitude as a single equivalent-batch backward.
                Default ``1.0`` (no scaling — standard non-accumulating path).
        """
        # Single-adapter or single-turn paths: do the combined backward.
        # ``mb_response_loss is None and mb_feedback_loss is None`` covers the
        # ``single_turn_a1`` baseline, which leaves the per-adapter
        # accumulators untouched and only fills ``mb_loss`` (with A1-only loss).
        if not self._routing.enabled or (mb_response_loss is None and mb_feedback_loss is None):
            if mb_loss is not None:
                (mb_loss * loss_scale).backward()
            return

        a1_adapter = self._routing.adapter_for_turn("a1")
        a2_adapter = self._routing.adapter_for_turn("a2")
        f1_adapter = self._routing.adapter_for_turn("f1")

        # Generalisation guard: this code splits the loss into two adapter
        # buckets — one for A1+A2, one for F1. If a future routing puts A1
        # and A2 on different adapters, we'd need a 3-way split (one backward
        # per adapter). Flag loudly instead of silently mis-attributing.
        if a1_adapter != a2_adapter:
            raise NotImplementedError(
                f"Multi-adapter backward currently assumes A1 and A2 share an "
                f"adapter (got a1={a1_adapter!r}, a2={a2_adapter!r}). To support "
                "asymmetric A1/A2 routing, extend this method to bucket by "
                "{a1, a2, f1} → adapter and call backward once per bucket."
            )

        # When all three turns share an adapter (e.g. single-adapter routing
        # with explicit names), the gradient-checkpoint recompute uses that
        # single adapter regardless — fall back to the combined backward.
        if a1_adapter == f1_adapter:
            if mb_loss is not None:
                (mb_loss * loss_scale).backward()
            return

        # Multi-adapter split. After _routed_forward_multi's cleanup, the
        # active adapter is already the first-trainable, which is typically
        # the response adapter. We re-assert explicitly for clarity (and so
        # this code is correct even if the cleanup is later removed).
        if mb_response_loss is not None and mb_response_loss.requires_grad:
            self._set_active_adapter(inner_model, a1_adapter)
            (mb_response_loss * loss_scale).backward()
        if mb_feedback_loss is not None and mb_feedback_loss.requires_grad:
            self._set_active_adapter(inner_model, f1_adapter)
            (mb_feedback_loss * loss_scale).backward()

    def _set_active_adapter(self, peft_model: Any, name: str) -> None:
        """Set the active adapter, preserving requires_grad on ALL trainable adapters.

        PEFT's ``set_adapter`` flips ``requires_grad=False`` on every adapter
        whose name is not in the active list. That is fine when there is a
        single trainable adapter, but breaks multi-trainable setups: when we
        switch from "response" to "feedback" to run the F1 forward, response
        LoRA params would lose ``requires_grad`` and any pending backward
        edges to them would be silently dropped — gradients on A1+A2 (computed
        with response active) would not reach the response weights.

        We work around this by calling PEFT's ``set_adapter`` and then
        immediately re-enabling ``requires_grad=True`` on every LoRA param
        belonging to an adapter that the routing config marks as trainable.
        Inactive adapters still contribute zero gradient because PEFT's LoRA
        ``forward`` only mixes in adapters listed in ``active_adapters`` —
        ``requires_grad`` controls autograd leaf participation, not forward
        inclusion. The end result: one optimizer can step every trainable
        adapter even though we route per turn.

        Args:
            peft_model: Unwrapped PEFT model.
            name: Target adapter name.
        """
        active = getattr(peft_model, "active_adapters", None)
        if active and len(active) == 1 and active[0] == name:
            return
        peft_model.set_adapter(name)
        if self._routing.enabled:
            self._enforce_trainable_grad_flags(peft_model)

    def _enforce_trainable_grad_flags(self, peft_model: Any) -> None:
        """Re-apply ``requires_grad`` per ``self._routing`` adapter specs.

        Called after every PEFT ``set_adapter`` to undo the side-effect that
        flips requires_grad off on non-active adapters. Frozen adapters stay
        frozen; trainable adapters stay trainable regardless of active state.

        Honors ``self._routing.frozen_lora_patterns``: any LoRA param whose
        name contains a listed pattern (e.g. ``"visual"``) is forced to
        ``requires_grad=False`` regardless of its hosting adapter's
        ``trainable`` flag — used to freeze module-family LoRA inherited
        from a warm-start checkpoint.

        Adapter-name matching is anchored by the LoRA-tensor prefix
        (``lora_A.<name>.`` / ``lora_B.<name>.``) so adapter names that
        share a substring prefix (e.g. ``"response"`` and ``"response_v2"``)
        cannot cross-match and silently corrupt requires_grad on the wrong
        adapter.
        """
        spec_by_name = {a.name: a for a in self._routing.adapters}
        frozen_patterns = tuple(self._routing.frozen_lora_patterns)
        for pname, param in peft_model.named_parameters():
            for adapter_name, spec in spec_by_name.items():
                if f".lora_A.{adapter_name}." in pname or f".lora_B.{adapter_name}." in pname:
                    desired = spec.trainable
                    if desired and frozen_patterns and any(p in pname for p in frozen_patterns):
                        desired = False
                    if param.requires_grad != desired:
                        param.requires_grad = desired
                    break

    def _build_adapter_callback(self, gen_model: Any) -> Any:
        """Return a per-turn rollout callback for multi-adapter routing.

        The callback is invoked by ``generate_self_reflection_rollout`` with
        turn name ``"a1"`` / ``"f1"`` / ``"a2"`` before each batched
        generation. It:

          1. Resolves ``turn → adapter_name`` via ``self._routing.adapter_for_turn``.
          2. No-ops when the target adapter is already active.
          3. When a switch is needed: switches the HF adapter, then if vLLM
             is colocated, sleeps it, wakes for weights, re-syncs the new
             adapter's merged weights, and wakes for generation. The
             sleep-then-resync pattern keeps vLLM's base weights in lock-step
             with whichever adapter is currently routing.

        The dual-merge cost (sleep + wake + resync) only fires when the
        adapter actually changes — F1→A2 with both on the same adapter
        skips the resync entirely.

        Args:
            gen_model: Unwrapped PEFT model used for generation.

        Returns:
            ``Callable[[str], None]`` suitable for the rollout's
            ``adapter_callback`` parameter, or ``None`` when routing is
            disabled.
        """
        if not self._routing.enabled:
            return None

        def callback(turn: str) -> None:
            target = self._routing.adapter_for_turn(turn)
            active = getattr(gen_model, "active_adapters", None)
            if active and len(active) == 1 and active[0] == target:
                return

            self._set_active_adapter(gen_model, target)

            if self.vllm_engine is None:
                return

            self.vllm_engine.sleep()
            self.vllm_engine.wake_up_for_weights()
            self.vllm_engine.update_weights_from_peft(gen_model, accelerator=self.accelerator)
            self.vllm_engine.wake_up_for_generation()

        return callback

    def _save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint + optimizer state for resume.

        Layout:
          * Single-adapter mode: ``<path>/adapter_model.safetensors`` (PEFT
            default for the "default" adapter). Backwards-compatible with
            existing resume paths.
          * Multi-adapter mode: each trainable adapter is saved under
            ``<path>/<adapter_name>/``. Frozen adapters are skipped (their
            weights live in their original ``init_from_checkpoint``).

        Args:
            path: Directory to save to.
        """
        import torch

        path.mkdir(parents=True, exist_ok=True)
        unwrapped = (
            self.accelerator.unwrap_model(self.model)
            if self.accelerator is not None
            else self.model
        )

        if self._routing.enabled:
            trainable = self._routing.trainable_adapter_names()
            # Restore active adapter to the first trainable so any
            # active-adapter-sensitive code paths see a stable value after save.
            unwrapped.save_pretrained(path, selected_adapters=trainable)
            if trainable:
                self._set_active_adapter(unwrapped, trainable[0])
            logger.info(f"Saved checkpoint to {path} (multi-adapter: {trainable})")
        else:
            unwrapped.save_pretrained(path)
            logger.info(f"Saved checkpoint to {path}")
        self.processor.save_pretrained(path)

        # Save optimizer state for resume
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")

        config_path = path / "training_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
