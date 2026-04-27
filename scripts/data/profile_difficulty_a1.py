#!/usr/bin/env python3
"""Profile per-prompt A1 difficulty on the base Qwen2.5-VL-7B-Instruct policy.

For each prompt in a JSONL dataset, draw K independent A1 completions with
vLLM, score each against the ground truth via `verify_answer()`, and write a
JSONL of per-prompt pass rates.

Inference-only — no LoRA, no training, no F1/A2. Output drives the static
difficulty filter (drop trivial + brick-wall buckets) before training.

Usage:
    uv run python scripts/data/profile_difficulty_a1.py \\
        --dataset /outputs/livr_data/livr_perception_mcq.jsonl \\
        --output /outputs/livr_data/livr_difficulty_a1.jsonl \\
        --model Qwen/Qwen2.5-VL-7B-Instruct \\
        --k 8 \\
        --max_completion_length 200 \\
        --temperature 1.0 \\
        --top_p 0.9 \\
        --batch_size 32

Output schema (one JSON object per prompt):
    {
      "sample_index": int,
      "dataset_name": str,
      "answer_type": str,
      "ground_truth": str,
      "k": int,
      "a1_correct_count": int,
      "a1_pass_rate": float,
      "completions": [str, ...]   # truncated to <= 256 chars each
    }
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Maximum chars of completion text to retain in output (debugging; full text
# is not needed downstream and bloats the JSONL).
_COMPLETION_PREVIEW_CHARS = 256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Input JSONL dataset path")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HF model id or local checkpoint path",
    )
    parser.add_argument(
        "--image_base_dir", default="/outputs/image_base", help="Base for relative image paths"
    )
    parser.add_argument("--k", type=int, default=8, help="Samples per prompt")
    parser.add_argument(
        "--max_completion_length", type=int, default=200, help="Max tokens per A1 completion"
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Prompts per vLLM generate() call (each yields K completions)",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--use_think_answer_tags",
        action="store_true",
        help="Match training A1 prompt by appending think+answer tag instruction",
    )
    parser.add_argument(
        "--use_answer_tag_only",
        action="store_true",
        help="Match training A1 prompt by appending answer-tag-only instruction",
    )
    parser.add_argument(
        "--strict_answer_extraction",
        action="store_true",
        help="Require atomic answer inside <answer> tag (training-faithful when tags enabled)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.85,
        help="vLLM GPU mem fraction. Single-purpose job — can run hot (no sleep).",
    )
    parser.add_argument("--max_model_len", type=int, default=2048, help="vLLM max sequence length")
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=401408,
        help="Qwen2.5-VL dynamic resolution upper bound",
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=200704,
        help="Qwen2.5-VL dynamic resolution lower bound",
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="GPUs for vLLM tensor parallelism"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip prompts already present in --output (by sample_index)",
    )
    return parser.parse_args()


def _load_existing_indices(output_path: str) -> set[int]:
    """Return set of sample_index already written to output (resume support)."""
    if not os.path.exists(output_path):
        return set()
    seen: set[int] = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = row.get("sample_index")
            if isinstance(idx, int):
                seen.add(idx)
    return seen


def _build_a1_prompt_text(
    processor: Any,
    question: str,
    use_think_answer_tags: bool,
    use_answer_tag_only: bool,
) -> str:
    """Format A1 user message via the processor's chat template.

    Returns a single string ready for vLLM (with vision placeholders inserted
    by apply_chat_template).
    """
    from vlm_grpo.prompts import build_initial_answer_prompt

    msgs = build_initial_answer_prompt(
        question=question,
        use_think_answer_tags=use_think_answer_tags,
        use_answer_tag_only=use_answer_tag_only,
    )
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main() -> None:
    args = _parse_args()

    # vLLM env hygiene
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")

    # Deferred imports (vLLM + torch are heavy)
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    from vlm_grpo.data import load_image_safe, load_self_reflection_dataset
    from vlm_grpo.rewards.verifier import verify_answer

    logger.info("Loading dataset: %s", args.dataset)
    samples = load_self_reflection_dataset(
        dataset_path=args.dataset,
        image_base_dir=args.image_base_dir,
        max_samples=args.max_samples,
        max_pixels=args.max_pixels,
    )
    logger.info("Loaded %d samples", len(samples))

    # Resume support — skip already-profiled indices
    seen_indices: set[int] = set()
    if args.resume:
        seen_indices = _load_existing_indices(args.output)
        logger.info("Resume: found %d already-profiled samples", len(seen_indices))
        samples = [s for s in samples if s["sample_index"] not in seen_indices]
        logger.info("Remaining: %d", len(samples))

    if not samples:
        logger.info("Nothing to do. Exiting.")
        return

    logger.info("Loading processor: %s", args.model)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    logger.info("Initializing vLLM (no sleep mode, single-purpose inference)")
    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=args.tensor_parallel_size,
        mm_processor_kwargs={
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
        seed=args.seed,
    )
    logger.info("vLLM ready.")

    sampling_params = SamplingParams(
        n=args.k,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_completion_length,
        seed=args.seed,
    )

    # Open output file in append mode so resume / partial runs accumulate.
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_f = open(args.output, "a", encoding="utf-8")

    n_total = len(samples)
    n_done = 0
    t_start = time.time()

    try:
        for batch_start in range(0, n_total, args.batch_size):
            batch = samples[batch_start : batch_start + args.batch_size]

            # Build vLLM inputs (skip prompts whose image fails to load)
            vllm_inputs: list[dict[str, Any]] = []
            kept_meta: list[dict[str, Any]] = []
            for s in batch:
                image: Image.Image | None = load_image_safe(
                    s["image_path"], max_pixels=args.max_pixels
                )
                if image is None:
                    logger.warning(
                        "Skipping sample_index=%s — image load failed: %s",
                        s["sample_index"],
                        s["image_path"],
                    )
                    continue
                prompt_text = _build_a1_prompt_text(
                    processor=processor,
                    question=s["question"],
                    use_think_answer_tags=args.use_think_answer_tags,
                    use_answer_tag_only=args.use_answer_tag_only,
                )
                vllm_inputs.append(
                    {
                        "prompt": prompt_text,
                        "multi_modal_data": {"image": image},
                    }
                )
                kept_meta.append(s)

            if not vllm_inputs:
                continue

            # Generate K completions per prompt in one call (n=K is set on
            # SamplingParams). vLLM batches across prompts × K internally.
            outputs = llm.generate(vllm_inputs, sampling_params)

            for sample, out in zip(kept_meta, outputs):
                completions = [c.text.strip() for c in out.outputs]
                # Score each completion
                correct = 0
                for c in completions:
                    res = verify_answer(
                        raw_text=c,
                        ground_truth=sample["ground_truth"],
                        answer_type=sample["answer_type"],
                        choices=sample.get("choices", "") or "",
                        strict=args.strict_answer_extraction,
                    )
                    if res.is_correct:
                        correct += 1

                k = len(completions)
                row = {
                    "sample_index": sample["sample_index"],
                    "dataset_name": sample.get("dataset_name", "unknown"),
                    "answer_type": sample["answer_type"],
                    "ground_truth": sample["ground_truth"],
                    "k": k,
                    "a1_correct_count": correct,
                    "a1_pass_rate": correct / k if k > 0 else 0.0,
                    "completions": [c[:_COMPLETION_PREVIEW_CHARS] for c in completions],
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_done += 1

            out_f.flush()

            elapsed = time.time() - t_start
            rate = n_done / elapsed if elapsed > 0 else 0.0
            remaining = (n_total - n_done) / rate if rate > 0 else 0.0
            logger.info(
                "Progress: %d/%d (%.1f%%) | %.2f prompts/s | ETA %.1f min",
                n_done,
                n_total,
                100.0 * n_done / n_total if n_total else 0.0,
                rate,
                remaining / 60.0,
            )

    finally:
        out_f.close()

    logger.info("Done. Wrote %d rows to %s", n_done, args.output)


if __name__ == "__main__":
    main()
