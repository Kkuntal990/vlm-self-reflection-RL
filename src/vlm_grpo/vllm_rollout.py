#!/usr/bin/env python3
"""
vLLM-accelerated rollout engine for self-reflection GRPO.

Replaces HuggingFace model.generate() with vLLM offline inference
for 3-5x faster rollout generation. Uses sleep mode to share GPU
memory between generation (vLLM) and training (HF model).

The sequential A1 -> F1 -> A2 dependency is handled with three
separate batched vLLM generate() calls per rollout step.

References:
    - vLLM sleep mode: https://docs.vllm.ai/en/latest/features/sleep_mode/
    - TRL colocate mode: https://huggingface.co/blog/vllm-colocate
    - vLLM Qwen2.5-VL: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen2.5-VL.html

Usage:
    from vlm_grpo.vllm_rollout import VLLMRolloutEngine

    engine = VLLMRolloutEngine("Qwen/Qwen2.5-VL-7B-Instruct", processor)
    engine.wake_up()
    completions = engine.generate_batch(prompts, images, max_tokens=200, temperature=1.0)
    engine.sleep()
"""

import logging
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
        gpu_memory_utilization: float = 0.40,
        max_model_len: int = 2048,
        max_pixels: int = 401408,
        min_pixels: int = 200704,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = True,
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
        """
        from vllm import LLM

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
            limit_mm_per_prompt={"image": 1},
            mm_processor_kwargs={
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
            },
            enable_sleep_mode=True,
            dtype="bfloat16",
        )
        logger.info(
            f"vLLM engine ready: gpu_mem={gpu_memory_utilization}, "
            f"max_len={max_model_len}, tp={tensor_parallel_size}"
        )

    def sleep(self) -> None:
        """Put vLLM to sleep, freeing GPU memory for training.

        Level 1 sleep offloads model weights to CPU and discards
        the KV cache, freeing most GPU memory.
        """
        self.llm.sleep(level=1)

    def wake_up(self) -> None:
        """Wake vLLM up, reloading weights to GPU."""
        self.llm.wake_up()

    def update_weights_from_peft(self, peft_model: Any) -> None:
        """Merge LoRA adapter weights and load into vLLM.

        Temporarily merges the LoRA adapter into the base weights,
        extracts the merged state dict, loads into vLLM, then
        unmerges to restore the PEFT model for training.

        Args:
            peft_model: PEFT model with LoRA adapter (may be DDP-wrapped)
        """
        import torch

        # Unwrap DDP/Accelerate if needed
        unwrapped = peft_model
        if hasattr(peft_model, "module"):
            unwrapped = peft_model.module

        named_params = []
        with torch.no_grad():
            unwrapped.merge_adapter()
            for name, param in unwrapped.named_parameters():
                named_params.append((name, param.data))
            unwrapped.unmerge_adapter()

        # Load merged weights into vLLM's model
        self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(
            named_params
        )
        logger.info(f"Updated vLLM weights ({len(named_params)} params)")

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
