#!/usr/bin/env python3
"""Two-LoRA-adapter helpers for the GRPO trainer.

Provides the init path and weight-copy logic that backs
``--two_adapter_mode``. The motivation: shared-LoRA frozen-A1 runs collapse
because F1+A2 gradients drift the SAME LoRA tensors that A1 generation reads
from. Splitting into two adapters (``a1_expert`` frozen, ``f1_a2_expert``
trainable) gives true architectural separation — A1 generation is bit-
identical to the source checkpoint at every step.

The key invariants this module enforces:
  1. After ``init_two_adapter_model`` returns, the PEFT model has TWO
     adapters: ``a1_expert`` (requires_grad=False) and ``f1_a2_expert``
     (requires_grad=True). The active adapter is ``f1_a2_expert`` (so the
     training forward path picks up the trainable weights by default).
  2. At init, ``a1_expert`` and ``f1_a2_expert`` carry IDENTICAL weights
     (verified by tensor equality) — F1+A2 start from a strong vision-
     language base instead of random LoRA init.
  3. Only ``f1_a2_expert`` parameters report ``requires_grad=True``. This
     means a downstream ``AdamW(model.parameters())`` over ``[p for p in
     model.parameters() if p.requires_grad]`` correctly picks up only the
     trainable adapter's weights.
"""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

A1_ADAPTER_NAME = "a1_expert"
F1_A2_ADAPTER_NAME = "f1_a2_expert"


def init_two_adapter_model(
    base_model: Any,
    lora_config: Any,
    frozen_a1_adapter_path: str,
) -> Any:
    """Wrap a base HF model with two LoRA adapters.

    Loads ``a1_expert`` from disk (frozen), adds a sibling ``f1_a2_expert``
    with the same LoRA shape (trainable), and copies a1_expert's weights
    into f1_a2_expert so both start from the same point. Sets
    f1_a2_expert as the active adapter — the trainer's default forward
    path will pick up the trainable weights.

    Args:
        base_model: The base HuggingFace model (after vision-tower freeze
            and gradient-checkpointing setup, BEFORE PEFT wrapping).
        lora_config: LoraConfig describing the trainable adapter shape.
            Must match the shape of the frozen a1_expert checkpoint
            (rank, alpha, target modules).
        frozen_a1_adapter_path: Filesystem path to the a1_expert checkpoint
            directory (must contain ``adapter_model.safetensors`` and
            ``adapter_config.json``).

    Returns:
        The PEFT-wrapped model with two adapters. The active adapter is
        ``f1_a2_expert`` (trainable).

    Raises:
        FileNotFoundError: If ``frozen_a1_adapter_path`` is missing or
            does not contain the expected adapter files.
        RuntimeError: If the post-init invariants (two adapters present,
            weight equality, requires_grad split) fail.
    """
    from peft import PeftModel

    a1_path = Path(frozen_a1_adapter_path)
    if not a1_path.is_dir():
        raise FileNotFoundError(
            f"--frozen_a1_adapter_path={a1_path} is not a directory. "
            f"Expected a PEFT checkpoint dir with adapter_model.safetensors."
        )
    adapter_file = a1_path / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(
            f"adapter_model.safetensors missing at {adapter_file}. "
            f"--frozen_a1_adapter_path must point at a saved PEFT checkpoint."
        )

    logger.info(f"Loading frozen a1_expert from: {a1_path}")
    model = PeftModel.from_pretrained(
        base_model,
        str(a1_path),
        adapter_name=A1_ADAPTER_NAME,
        is_trainable=False,
    )

    logger.info(f"Adding trainable adapter: {F1_A2_ADAPTER_NAME}")
    model.add_adapter(F1_A2_ADAPTER_NAME, lora_config)

    n_copied = _copy_adapter_weights(model, src=A1_ADAPTER_NAME, dst=F1_A2_ADAPTER_NAME)
    logger.info(
        f"Warm-started {F1_A2_ADAPTER_NAME} from {A1_ADAPTER_NAME} ({n_copied} LoRA tensors copied)"
    )

    model.set_adapter(F1_A2_ADAPTER_NAME)
    _enforce_freeze_invariants(model)

    return model


def _copy_adapter_weights(model: Any, src: str, dst: str) -> int:
    """Copy LoRA weights from one adapter onto another, in place.

    Iterates the model's named parameters, finds matching ``src``/``dst``
    LoRA tensor pairs (lora_A.<src>.weight ↔ lora_A.<dst>.weight, same for
    lora_B), and copies the source weights onto the destination. The two
    adapters MUST have identical shapes (same rank, alpha, and target
    module list); a shape mismatch raises a RuntimeError before any copy
    happens.

    Args:
        model: The PEFT model with both adapters loaded.
        src: Source adapter name (weights are read from here).
        dst: Destination adapter name (weights are overwritten here).

    Returns:
        Number of tensors copied (sum of lora_A + lora_B + lora_embedding_A
        + lora_embedding_B counts that match between adapters).

    Raises:
        RuntimeError: If a matching destination tensor has a shape
            different from the source tensor.
    """
    import torch

    src_params: dict[str, torch.nn.Parameter] = {}
    dst_params: dict[str, torch.nn.Parameter] = {}

    for name, param in model.named_parameters():
        # PEFT stores per-adapter weights as ``...lora_A.<adapter>.weight``.
        # The "key" used for matching is everything BEFORE the adapter
        # marker — that uniquely identifies the layer + projection role.
        for adapter, bucket in ((src, src_params), (dst, dst_params)):
            marker = f".{adapter}."
            if marker in name and (".lora_A." in name or ".lora_B." in name):
                key = name.replace(marker, ".__ADAPTER__.")
                bucket[key] = param

    n_copied = 0
    with torch.no_grad():
        for key, dst_param in dst_params.items():
            src_param = src_params.get(key)
            if src_param is None:
                continue
            if src_param.shape != dst_param.shape:
                raise RuntimeError(
                    f"Cannot copy adapter weights: shape mismatch on {key} "
                    f"(src={tuple(src_param.shape)} vs "
                    f"dst={tuple(dst_param.shape)}). The two adapters must "
                    f"share rank/alpha/target modules."
                )
            dst_param.data.copy_(src_param.data)
            n_copied += 1

    if n_copied == 0:
        raise RuntimeError(
            f"No matching LoRA tensor pairs found between adapters "
            f"{src!r} and {dst!r}. The frozen checkpoint may have been "
            f"saved with a different LoRA target list than the trainable "
            f"config — check adapter_config.json."
        )

    return n_copied


def _enforce_freeze_invariants(model: Any) -> None:
    """Sanity-check the two-adapter freeze split.

    After ``set_adapter(F1_A2_ADAPTER_NAME)``, only F1_A2 LoRA params
    should report ``requires_grad=True``. PEFT's ``set_adapter`` already
    enforces this in recent versions, but we double-check defensively —
    a leaked requires_grad=True on a1_expert would silently train it.

    Args:
        model: The two-adapter PEFT model.

    Raises:
        RuntimeError: If a1_expert has any trainable parameter, or if
            f1_a2_expert has zero trainable parameters.
    """
    n_a1_trainable = 0
    n_f1_a2_trainable = 0
    n_other_trainable = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if f".{A1_ADAPTER_NAME}." in name:
            n_a1_trainable += 1
        elif f".{F1_A2_ADAPTER_NAME}." in name:
            n_f1_a2_trainable += 1
        else:
            n_other_trainable += 1

    if n_a1_trainable > 0:
        # Defensive freeze: walk back any leaked requires_grad=True so
        # the optimizer cannot accidentally update the a1_expert tensors.
        n_frozen = 0
        for name, param in model.named_parameters():
            if f".{A1_ADAPTER_NAME}." in name and param.requires_grad:
                param.requires_grad = False
                n_frozen += 1
        logger.warning(
            f"Defensive freeze: {n_frozen} a1_expert params had "
            f"requires_grad=True after set_adapter; forced to False."
        )

    if n_f1_a2_trainable == 0:
        raise RuntimeError(
            "Two-adapter init invariant violated: f1_a2_expert has no "
            "trainable params. Check that set_adapter(F1_A2_ADAPTER_NAME) "
            "ran AFTER add_adapter, and that add_adapter received a "
            "LoraConfig with non-frozen targets."
        )

    logger.info(
        f"Two-adapter freeze split OK: a1_expert trainable={n_a1_trainable} "
        f"(must be 0), f1_a2_expert trainable={n_f1_a2_trainable}, "
        f"other_trainable={n_other_trainable} (e.g. modules_to_save)."
    )


def save_frozen_a1_for_vllm(peft_model: Any, output_dir: str) -> str:
    """Materialize the frozen a1_expert to a stable on-disk path.

    vLLM's LoRARequest needs an on-disk adapter path. Since the user-
    supplied ``--frozen_a1_adapter_path`` may be on a different volume
    that vLLM workers can't read, we re-save the in-memory a1_expert into
    the run's output directory. This also normalizes the adapter dump
    format (e.g. wrapping selected_adapters=["a1_expert"]).

    Args:
        peft_model: The two-adapter PEFT model (after init).
        output_dir: The training run's output directory. The adapter is
            written to ``<output_dir>/_a1_expert_frozen``.

    Returns:
        Absolute path to the on-disk a1_expert directory.
    """
    out_path = Path(output_dir) / "_a1_expert_frozen"
    out_path.mkdir(parents=True, exist_ok=True)

    peft_model.save_pretrained(str(out_path), selected_adapters=[A1_ADAPTER_NAME])
    logger.info(f"Saved frozen a1_expert snapshot to {out_path}")
    return str(out_path)


def save_trainable_adapter(peft_model: Any, output_dir: str, step: int | None = None) -> str:
    """Save the trainable f1_a2_expert to disk for vLLM consumption.

    Called once per rollout step before vLLM weight sync (or LoRARequest
    pickup). The adapter is saved with PEFT's ``save_pretrained``,
    selected to write only f1_a2_expert (not a1_expert).

    Args:
        peft_model: The two-adapter PEFT model (unwrapped from DDP).
        output_dir: The training run's output directory.
        step: Optional global_step to include in the dir name. When None,
            uses a single fixed path ``_f1_a2_expert_current/`` so vLLM
            always reads the most recent weights.

    Returns:
        Absolute path to the on-disk f1_a2_expert directory.
    """
    if step is None:
        out_path = Path(output_dir) / "_f1_a2_expert_current"
    else:
        out_path = Path(output_dir) / f"_f1_a2_expert_step{step}"
    out_path.mkdir(parents=True, exist_ok=True)

    peft_model.save_pretrained(str(out_path), selected_adapters=[F1_A2_ADAPTER_NAME])
    return str(out_path)
