#!/usr/bin/env python3
"""Multi-LoRA-adapter helpers for the GRPO trainer.

Backs the generic ``--adapter_routing_json`` mode. The trainer can hold N
named LoRA adapters on the same base model and route each turn (A1, F1, A2)
to whichever adapter is responsible for that turn.

Typical configurations:

  * Single adapter (default — no routing config supplied): one adapter named
    "default" handles every turn. Identical behavior to pre-routing code.
  * Response / feedback split: "response" handles A1+A2, "feedback" handles
    F1. Both trainable, both initialised from base. Used when you want F1
    to evolve independently of A1+A2.
  * Frozen-A1 + trainable F1+A2: legacy two-adapter pattern. "a1_expert"
    (frozen, loaded from a prior checkpoint) handles A1; "f1_a2_expert"
    (trainable, warm-started from a1_expert) handles F1 and A2.

The init pipeline:
  1. Wrap the base model with the FIRST adapter. If its spec carries
     ``init_from_checkpoint``, use ``PeftModel.from_pretrained``; otherwise
     use ``get_peft_model`` with the run's LoraConfig.
  2. For each subsequent adapter: either load from disk with
     ``model.load_adapter`` or add fresh with ``model.add_adapter``.
  3. If a spec has ``warm_start_from_adapter``, copy LoRA tensors from the
     referenced sibling so the new adapter starts from a known-good point.
  4. Set ``requires_grad`` per spec, then activate the first trainable
     adapter so the default forward path picks up the trainable weights.
"""

import logging
from pathlib import Path
from typing import Any

from vlm_grpo.config import AdapterRoutingConfig

logger = logging.getLogger(__name__)


DEFAULT_ADAPTER_NAME = "default"


def init_multi_adapter_model(
    base_model: Any,
    lora_config: Any,
    routing: AdapterRoutingConfig,
) -> Any:
    """Wrap a base HF model with the adapters described in ``routing``.

    Args:
        base_model: The base HuggingFace model (after vision-tower freeze
            and gradient-checkpointing setup, BEFORE any PEFT wrapping).
        lora_config: LoraConfig describing trainable-adapter shape. Used
            both for fresh ``add_adapter`` calls and as the shape contract
            for warm-start copies.
        routing: Validated AdapterRoutingConfig. Caller is responsible for
            having called ``routing.validate()`` first.

    Returns:
        The PEFT-wrapped model. The active adapter is the first trainable
        spec in routing.adapters.

    Raises:
        ValueError: When ``routing.adapters`` is empty (use the regular
            ``get_peft_model`` path for single-adapter mode instead — this
            function is for routing.enabled only).
        FileNotFoundError: When an ``init_from_checkpoint`` path is missing.
        RuntimeError: When invariants (warm-start shape match, freeze
            split) fail.
    """
    from peft import PeftModel, get_peft_model

    if not routing.adapters:
        raise ValueError(
            "init_multi_adapter_model called with empty adapter list. "
            "Use get_peft_model(model, lora_config) for single-adapter mode."
        )

    routing.validate()

    first = routing.adapters[0]
    logger.info(f"Initialising multi-adapter model with first adapter: {first.name}")
    if first.init_from_checkpoint:
        ckpt = Path(first.init_from_checkpoint)
        _require_peft_ckpt(ckpt, first.name)
        model = PeftModel.from_pretrained(
            base_model,
            str(ckpt),
            adapter_name=first.name,
            is_trainable=first.trainable,
        )
    else:
        # get_peft_model wraps with adapter_name="default"; rename so the
        # caller-supplied name is what PEFT sees.
        model = get_peft_model(base_model, lora_config, adapter_name=first.name)

    for spec in routing.adapters[1:]:
        logger.info(f"Adding adapter: {spec.name}")
        if spec.init_from_checkpoint:
            ckpt = Path(spec.init_from_checkpoint)
            _require_peft_ckpt(ckpt, spec.name)
            model.load_adapter(
                str(ckpt),
                adapter_name=spec.name,
                is_trainable=spec.trainable,
            )
        else:
            model.add_adapter(spec.name, lora_config)
        if spec.warm_start_from_adapter:
            n_copied = _copy_adapter_weights(
                model,
                src=spec.warm_start_from_adapter,
                dst=spec.name,
            )
            logger.info(
                f"Warm-started {spec.name} from {spec.warm_start_from_adapter} "
                f"({n_copied} LoRA tensors copied)"
            )

    _apply_trainable_flags(model, routing)

    trainable = routing.trainable_adapter_names()
    if trainable:
        # First trainable adapter becomes the training default. The
        # per-turn callback overrides this during rollout/forward when
        # different turns route to different adapters.
        model.set_adapter(trainable[0])
        logger.info(f"Active adapter set to: {trainable[0]}")

    _log_adapter_param_split(model, routing)
    return model


def _require_peft_ckpt(ckpt: Path, adapter_name: str) -> None:
    """Validate that ``ckpt`` is a PEFT-checkpoint directory."""
    if not ckpt.is_dir():
        raise FileNotFoundError(
            f"Adapter {adapter_name!r}: init_from_checkpoint={ckpt} is not "
            "a directory. Expected a PEFT checkpoint dir with "
            "adapter_model.safetensors."
        )
    adapter_file = ckpt / "adapter_model.safetensors"
    if not adapter_file.exists():
        raise FileNotFoundError(
            f"Adapter {adapter_name!r}: adapter_model.safetensors missing at {adapter_file}."
        )


def _copy_adapter_weights(model: Any, src: str, dst: str) -> int:
    """Copy LoRA weights from ``src`` adapter onto ``dst`` in place.

    Both adapters must already be loaded and must share LoRA shape (rank,
    alpha, target modules). Iterates ``named_parameters`` and matches
    pairs by stripping the adapter marker; any unmatched tensor is
    skipped (e.g. ``modules_to_save`` re-projections that exist on one
    adapter but not the other).

    Args:
        model: PEFT model with both adapters loaded.
        src: Source adapter name (read).
        dst: Destination adapter name (overwritten).

    Returns:
        Number of tensors copied.

    Raises:
        RuntimeError: When no matching pairs are found, or when a matched
            pair has incompatible shapes.
    """
    import torch

    src_params: dict[str, torch.nn.Parameter] = {}
    dst_params: dict[str, torch.nn.Parameter] = {}

    for name, param in model.named_parameters():
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
                    f"Shape mismatch copying {src!r}→{dst!r} on {key}: "
                    f"src={tuple(src_param.shape)} dst={tuple(dst_param.shape)}. "
                    "Adapters must share rank / alpha / target modules."
                )
            dst_param.data.copy_(src_param.data)
            n_copied += 1

    if n_copied == 0:
        raise RuntimeError(
            f"No matching LoRA tensor pairs between {src!r} and {dst!r}. "
            "Check adapter_config.json — the LoRA target lists must agree."
        )
    return n_copied


def _apply_trainable_flags(model: Any, routing: AdapterRoutingConfig) -> None:
    """Enforce requires_grad per spec.

    PEFT's ``set_adapter`` already activates the matching adapter's params,
    but a frozen spec must have ``requires_grad=False`` on its tensors
    regardless of which adapter is currently active — otherwise a
    spuriously-trainable frozen adapter could be picked up by the
    optimizer.
    """
    name_to_trainable = {a.name: a.trainable for a in routing.adapters}
    for name, param in model.named_parameters():
        for adapter_name, trainable in name_to_trainable.items():
            marker = f".{adapter_name}."
            if marker not in name:
                continue
            if not trainable and param.requires_grad:
                param.requires_grad = False
            break  # one adapter match per param


def _log_adapter_param_split(model: Any, routing: AdapterRoutingConfig) -> None:
    """Log how many trainable params belong to each adapter (diagnostic).

    Helps catch silently-frozen or silently-trainable adapters at startup.
    """
    counts: dict[str, int] = {a.name: 0 for a in routing.adapters}
    n_other = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        matched = False
        for adapter in counts:
            if f".{adapter}." in name:
                counts[adapter] += 1
                matched = True
                break
        if not matched:
            n_other += 1
    summary = ", ".join(f"{n}={c}" for n, c in counts.items())
    logger.info(
        f"Adapter trainable-param split: {summary}, "
        f"other_trainable={n_other} (e.g. modules_to_save)."
    )

    # Hard error: every adapter the routing labels as trainable must
    # actually have trainable params. A zero count almost always means
    # PEFT silently froze them (e.g. wrong active_adapter at the time of
    # the freeze sweep).
    for spec in routing.adapters:
        if spec.trainable and counts[spec.name] == 0:
            raise RuntimeError(
                f"Adapter {spec.name!r} is marked trainable but has 0 "
                "trainable parameters after init. Check the order of "
                "set_adapter / add_adapter in init_multi_adapter_model "
                "and the adapter_config.json of any loaded checkpoint."
            )


def save_adapter_for_vllm(
    peft_model: Any,
    adapter_name: str,
    output_dir: str,
    step: int | None = None,
) -> str:
    """Materialize one named adapter to disk for vLLM consumption.

    vLLM reads adapter weights from a path, not from in-memory tensors.
    Use this whenever you need to sync a specific adapter's weights to
    a vLLM engine that supports LoRARequest. Single-adapter callers can
    keep using ``update_weights_from_peft`` directly.

    Args:
        peft_model: The (possibly DDP-unwrapped) PEFT model.
        adapter_name: Name of the adapter to dump.
        output_dir: Run output directory; the adapter is written into a
            sub-directory under this path.
        step: Optional global_step suffix for the sub-directory. None →
            stable path (vLLM always reads the most recent weights).

    Returns:
        Absolute path to the on-disk adapter directory.
    """
    suffix = f"_step{step}" if step is not None else "_current"
    out_path = Path(output_dir) / f"_adapter_{adapter_name}{suffix}"
    out_path.mkdir(parents=True, exist_ok=True)

    peft_model.save_pretrained(str(out_path), selected_adapters=[adapter_name])
    return str(out_path)
