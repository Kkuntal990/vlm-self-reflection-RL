#!/usr/bin/env python3
"""
vLLM-accelerated rollout engine for self-reflection GRPO.

Replaces HuggingFace model.generate() with vLLM offline inference
for 3-5x faster rollout generation. Uses sleep mode to share GPU
memory between generation (vLLM) and training (HF model).

The sequential A1 -> F1 -> A2 dependency is handled with three
separate batched vLLM generate() calls per rollout step.

The sleep/wake/sync lifecycle per training step follows the TRL
VLLMGeneration colocate pattern exactly:
    1. wake_up_for_weights()     — allocate weight memory on GPU
    2. update_weights_from_peft()— sync LoRA-merged weights to vLLM
    3. wake_up_for_generation()  — allocate KV cache memory
    4. generate_batch() x3       — A1, F1, A2 inference
    5. sleep()                   — free ALL GPU memory for training

References:
    - TRL VLLMGeneration: trl/generation/vllm_generation.py
    - vLLM sleep mode: https://docs.vllm.ai/en/latest/features/sleep_mode/
    - TRL colocate blog: https://huggingface.co/blog/vllm-colocate
    - vLLM issue #26195: MM cache stale refs after sleep
    - verl issue #1361: disable_mm_preprocessor_cache deprecation
"""

import logging
import os
import sys
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class VLLMRolloutEngine:
    """vLLM-backed rollout engine for self-reflection trajectories.

    Manages a vLLM LLM instance that sleeps during training and wakes
    for rollout generation. Supports Qwen2.5-VL multimodal inputs.

    The sleep/wake cycle ensures GPU memory is shared:
    - During generation: vLLM uses ~20-24 GB for model + KV cache
    - During training: vLLM sleeps (0 GB), HF model uses full GPU budget
    """

    def __init__(
        self,
        model_id: str,
        processor: Any,
        gpu_memory_utilization: float = 0.30,
        max_model_len: int = 2048,
        max_pixels: int = 401408,
        min_pixels: int = 200704,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = True,
        seed: int = 0,
    ) -> None:
        """Initialize the vLLM rollout engine.

        Args:
            model_id: HuggingFace model ID or local checkpoint path
            processor: HuggingFace processor (for chat template formatting)
            gpu_memory_utilization: Fraction of GPU memory for vLLM KV cache
            max_model_len: Maximum total sequence length (prompt + completion)
            max_pixels: Max pixels per image (Qwen2.5-VL dynamic resolution)
            min_pixels: Min pixels per image
            tensor_parallel_size: Number of GPUs for tensor parallelism
            enforce_eager: Disable CUDA graphs to save memory
            seed: Random seed for sampling (use process rank for diversity)
        """
        from vllm import LLM

        # Required for external_launcher mode — prevents vLLM from spawning
        # child processes that conflict with accelerate/DeepSpeed.
        # Reference: TRL VLLMGeneration, vLLM ExecutorWithExternalLauncher
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        self.processor = processor
        self.model_id = model_id

        logger.info(f"Initializing vLLM engine: {model_id}")
        self.llm = LLM(
            model=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
            # Join the existing torch.distributed process group instead of
            # spawning new NCCL workers. Required for compatibility with
            # accelerate/DeepSpeed multi-GPU training.
            # Reference: https://huggingface.co/blog/vllm-colocate
            distributed_executor_backend="external_launcher",
            mm_processor_kwargs={
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            },
            enable_sleep_mode=True,
            dtype="bfloat16",
            seed=seed,
            max_num_batched_tokens=4096,
            # Disable multimodal preprocessor cache. sleep(level=2) frees
            # cached image feature tensors but leaves stale hash references,
            # causing AssertionError on the next generate() call.
            # Use mm_processor_cache_gb (stable API, vLLM ≥0.10.2) instead of
            # the deprecated disable_mm_preprocessor_cache (removed in ≥0.13).
            # Reference: vLLM issue #26195, verl issue #1361
            mm_processor_cache_gb=0,
        )
        logger.info(
            f"vLLM engine ready: gpu_mem={gpu_memory_utilization}, "
            f"max_len={max_model_len}, tp={tensor_parallel_size}"
        )

    def sleep(self) -> None:
        """Put vLLM to sleep, freeing ALL GPU memory for training.

        Level 2 sleep discards both model weights and KV cache,
        freeing all GPU memory. Matches TRL colocate pattern.
        """
        self.llm.sleep(level=2)

    def wake_up_for_weights(self) -> None:
        """Wake up weight memory only (step 1 of 2).

        Frees fragmented training memory, then allocates GPU space for
        model weights. Call update_weights_from_peft() next, then
        wake_up_for_generation() before generating.

        Matches TRL: ``wake_up(tags=["weights"])`` then ``collective_rpc``.
        """
        import torch

        torch.cuda.empty_cache()
        self.llm.wake_up(tags=["weights"])
        # Workaround for vLLM issue #29341: weights may not reload
        # properly after sleep level 2. Since we overwrite weights
        # via load_weights() anyway, this is a safety net.
        try:
            self.llm.collective_rpc("reload_weights")
        except (NotImplementedError, AttributeError):
            pass

    def wake_up_for_generation(self) -> None:
        """Wake up KV cache memory (step 2 of 2).

        Call this AFTER update_weights_from_peft() and BEFORE
        generate_batch(). Allocates the KV cache blocks.
        """
        self.llm.wake_up(tags=["kv_cache"])

    def update_weights_from_peft(
        self,
        peft_model: Any,
        accelerator: Any = None,
    ) -> None:
        """Merge LoRA adapter weights and load into vLLM.

        Temporarily merges the LoRA adapter into the base weights,
        extracts the merged state dict, strips the PEFT parameter name
        prefixes (``base_model.model.``, ``.base_layer``), skips
        adapter-only parameters (``lora_``, ``original_module``), and
        loads the result into vLLM.

        For DeepSpeed ZeRO-3, all sharded parameters are gathered
        before merge/load so that each rank sees the full tensors.

        Follows the same weight-sync logic as TRL ``VLLMGeneration.sync_weights``.

        Args:
            peft_model: PEFT model with LoRA adapter (may be DDP-wrapped)
            accelerator: Optional Accelerate ``Accelerator`` instance. When
                provided, DeepSpeed ZeRO-3 sharded parameters are gathered
                automatically before the merge/load cycle.
        """
        from contextlib import nullcontext

        import torch

        # Unwrap DDP/Accelerate if needed
        unwrapped = peft_model
        if hasattr(peft_model, "module"):
            unwrapped = peft_model.module

        # Detect DeepSpeed ZeRO-3: parameters are sharded across GPUs and
        # must be gathered before we can merge adapters or read full tensors.
        gather_ctx = nullcontext
        if accelerator is not None:
            ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
            if ds_plugin is not None and ds_plugin.zero_stage == 3:
                import deepspeed

                gather_ctx = deepspeed.zero.GatheredParameters

        # Determine the PEFT adapter prefix so we can skip adapter-only params.
        # For LoRA this is "lora_" (e.g. lora_A, lora_B).
        peft_prefix = getattr(unwrapped, "prefix", "lora_")

        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        assert hasattr(llm_model, "load_weights"), (
            f"vLLM internal model path broken: {type(llm_model)} has no load_weights(). "
            "Update the attribute path for your vLLM version."
        )

        n_loaded = 0
        with torch.no_grad(), gather_ctx(list(unwrapped.parameters())):
            unwrapped.merge_adapter()
            try:
                for name, param in unwrapped.named_parameters():
                    # Strip the PEFT wrapper prefixes to recover original HF names.
                    # PEFT wraps: base_model.model.<original_name>[.base_layer]
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")

                    # Skip adapter-only parameters (lora_A, lora_B, etc.) —
                    # they don't exist in the base model / vLLM.
                    if peft_prefix in name:
                        continue

                    # Skip modules_to_save bookkeeping parameters.
                    if "original_module" in name:
                        continue

                    # Also strip any modules_to_save wrapper prefix.
                    name = name.replace("modules_to_save.default.", "")

                    llm_model.load_weights([(name, param.data)])
                    n_loaded += 1
            finally:
                unwrapped.unmerge_adapter()

        # Reset prefix cache — cached KV states from previous weights
        # are invalid after weight sync.
        self.llm.reset_prefix_cache()
        logger.info(f"Updated vLLM weights ({n_loaded} params synced)")

    def generate_batch(
        self,
        prompts: list[str],
        images: list[Any],
        max_new_tokens: int,
        temperature: float,
        top_p: float = 0.9,
    ) -> list[str]:
        """Generate completions for a batch of prompts with images.

        Uses vLLM's continuous batching and PagedAttention for
        high-throughput inference. Supports multimodal inputs.

        Args:
            prompts: List of formatted prompt strings (from apply_chat_template)
            images: List of PIL Images (or None), one per prompt
            max_new_tokens: Maximum tokens per completion
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p sampling parameter

        Returns:
            List of completion text strings
        """
        from vllm import SamplingParams

        do_sample = temperature > 0
        sampling_params = SamplingParams(
            temperature=temperature if do_sample else 0.0,
            top_p=top_p if do_sample else 1.0,
            max_tokens=max_new_tokens,
        )

        # Build vLLM inputs with multimodal data
        vllm_inputs = []
        for prompt, image in zip(prompts, images):
            inp: dict[str, Any] = {"prompt": prompt}
            if image is not None:
                inp["multi_modal_data"] = {"image": image}
            vllm_inputs.append(inp)

        outputs = self.llm.generate(vllm_inputs, sampling_params)

        return [out.outputs[0].text.strip() for out in outputs]
