#!/usr/bin/env python3
"""
vLLM-accelerated rollout engine for self-reflection GRPO.

Replaces HuggingFace model.generate() with vLLM offline inference
for 3-5x faster rollout generation.

**No sleep mode.** vLLM's sleep/wake cycle has an unfixed CPython
refcount leak (issues #20431, #16993, #24879, #33625) that causes
SIGABRT after ~1000 cycles. Every workaround failed:
  - sleep(level=2): crash at ~1000 cycles
  - sleep(level=1): same crash
  - gc.collect(): slows but doesn't prevent
  - Engine restart: "Sleep mode can only be used for one instance"

Instead, vLLM runs with a small KV cache (gpu_memory_utilization=0.05)
that stays resident alongside the training model. Memory budget:
  vLLM model:     14 GiB (always resident)
  vLLM KV cache:   3 GiB (0.05 * ~66 GiB)
  Training model: 14 GiB (DDP)
  Available:     ~47 GiB (enough for gradient checkpointing)

The lifecycle per training step is simply:
    1. update_weights_from_peft() — sync LoRA-merged weights
    2. generate_batch() x3        — A1, F1, A2 inference
    (no sleep, no wake, no crash)

References:
    - vLLM sleep mode bugs: #20431, #16993, #24879, #33625
    - TRL colocate blog: https://huggingface.co/blog/vllm-colocate
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

    vLLM stays resident on GPU with a small KV cache. No sleep/wake
    cycle — avoids the refcount crash that affects all vLLM versions
    since 0.8.0. Weight sync happens via load_weights() each step.
    """

    def __init__(
        self,
        model_id: str,
        processor: Any,
        gpu_memory_utilization: float = 0.05,
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
            gpu_memory_utilization: Fraction of GPU memory for vLLM KV cache.
                Default 0.05 (~3 GiB) — small to leave room for training.
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
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        # Disable usage stats background thread (races with weight sync).
        os.environ["VLLM_NO_USAGE_STATS"] = "1"
        os.environ["DO_NOT_TRACK"] = "1"

        self.processor = processor
        self.model_id = model_id
        self.gpu_memory_utilization = gpu_memory_utilization

        logger.info(f"Initializing vLLM engine (no sleep mode): {model_id}")
        self.llm = LLM(
            model=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            mm_processor_kwargs={
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            },
            # No sleep mode — avoids CPython refcount crash.
            enable_sleep_mode=False,
            dtype="bfloat16",
            seed=seed,
            max_num_batched_tokens=4096,
            mm_processor_cache_gb=0,
        )
        logger.info(
            f"vLLM engine ready: gpu_mem={gpu_memory_utilization}, "
            f"max_len={max_model_len}, tp={tensor_parallel_size}"
        )

    def update_weights_from_peft(
        self,
        peft_model: Any,
        accelerator: Any = None,
    ) -> None:
        """Merge LoRA adapter weights and load into vLLM.

        Temporarily merges the LoRA adapter into the base weights,
        strips PEFT prefixes, and loads into vLLM via load_weights().

        Args:
            peft_model: PEFT model with LoRA adapter (may be DDP-wrapped)
            accelerator: Optional Accelerate Accelerator for ZeRO-3 gather.
        """
        from contextlib import nullcontext

        import torch

        # Unwrap DDP/Accelerate if needed
        unwrapped = peft_model
        if hasattr(peft_model, "module"):
            unwrapped = peft_model.module

        # Detect DeepSpeed ZeRO-3
        gather_ctx = nullcontext
        if accelerator is not None:
            ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
            if ds_plugin is not None and ds_plugin.zero_stage == 3:
                import deepspeed

                gather_ctx = deepspeed.zero.GatheredParameters

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
                    name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    if peft_prefix in name:
                        continue
                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")
                    llm_model.load_weights([(name, param.data)])
                    n_loaded += 1
            finally:
                unwrapped.unmerge_adapter()

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

        Args:
            prompts: List of formatted prompt strings
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

        vllm_inputs = []
        for prompt, image in zip(prompts, images):
            inp: dict[str, Any] = {"prompt": prompt}
            if image is not None:
                inp["multi_modal_data"] = {"image": image}
            vllm_inputs.append(inp)

        outputs = self.llm.generate(vllm_inputs, sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]
