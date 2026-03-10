#!/usr/bin/env python3
"""
LLM-as-judge for open-ended VQA answer equivalence.

Uses Qwen2.5-3B-Instruct on GPU to determine if a predicted answer is
semantically equivalent to a ground truth answer. Replaces embedding
cosine similarity for more accurate semantic matching in the open-ended
verification cascade.

The model is lazy-loaded as a singleton and runs on GPU in float16.
Results are cached via LRU cache to avoid redundant inference.

Activation: Set environment variable VLM_USE_LLM_JUDGE=1 to enable.
When disabled (default), the verifier falls back to embedding cosine sim.

Usage:
    import os
    os.environ["VLM_USE_LLM_JUDGE"] = "1"

    from vlm_grpo.rewards.judge_llm import llm_judge_score

    score = llm_judge_score(
        "The pepper is on the left side",
        "The pepper is on the right of the image",
    )
    assert score < 0.5  # Not equivalent
"""

import functools
import logging
import os
import re
import sys
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Configurable via environment variables
_JUDGE_MODEL_ID = os.environ.get("VLM_JUDGE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")

# Singleton model and tokenizer
_judge_model: Any = None
_judge_tokenizer: Any = None

_JUDGE_PROMPT = """You are an answer equivalence judge for visual question answering.

Two answers should be considered equivalent if they express the same meaning even if wording is different.

Examples of equivalent answers (score 10):
- "3" and "three"
- "There are three slices" and "3 slices"
- "No" and "No, it is not."
- "We are not going" and "We ain't going"
- "a cat" and "a domestic cat sitting on the table"
- "The brand is LG" and "LG"

Examples of NOT equivalent answers (score 0):
- "3" and "4" (different numbers)
- "Yes" and "No" (opposite polarity)
- "left" and "right" (opposite direction)
- "dog" and "cat" (different entities)
- "red" and "blue" (different attributes)
- "The pepper is on the left" and "The pepper is on the right" (contradictory)

Rate how equivalent the predicted answer is to the ground truth on a scale of 0-10:
- 10: Identical or perfect paraphrase (same meaning, same facts)
- 7-9: Mostly equivalent (same core answer, minor differences in detail or wording)
- 4-6: Partially equivalent (some overlap but missing or different key information)
- 1-3: Mostly wrong (different answer, different facts, contradictions)
- 0: Completely wrong or unrelated

Ground Truth: {ground_truth}
Predicted Answer: {predicted}

Respond with only a single integer from 0 to 10."""

# Pattern to extract the score from LLM response
_SCORE_PATTERN = re.compile(r"\b(\d+)\b")


def is_enabled() -> bool:
    """Check if the LLM judge is enabled via environment variable.

    Returns:
        True if VLM_USE_LLM_JUDGE=1 is set
    """
    return os.environ.get("VLM_USE_LLM_JUDGE", "0") == "1"


def _get_judge_model() -> tuple[Any, Any]:
    """Lazy-load the Qwen2.5-3B judge model and tokenizer on GPU.

    Returns:
        Tuple of (model, tokenizer)
    """
    global _judge_model, _judge_tokenizer
    if _judge_model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading LLM judge model: {_JUDGE_MODEL_ID}")
        _judge_tokenizer = AutoTokenizer.from_pretrained(
            _JUDGE_MODEL_ID,
            trust_remote_code=True,
        )
        _judge_model = AutoModelForCausalLM.from_pretrained(
            _JUDGE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        _judge_model.eval()
        logger.info("LLM judge model loaded")
    return _judge_model, _judge_tokenizer


@functools.lru_cache(maxsize=4096)
def llm_judge_score(predicted: str, ground_truth: str) -> float:
    """Score the equivalence of predicted answer vs ground truth using LLM.

    Uses Qwen2.5-3B-Instruct to produce a 0-10 score, normalized to [0, 1].
    Results are cached via LRU cache (maxsize=4096).

    Args:
        predicted: Predicted answer text
        ground_truth: Ground truth answer text

    Returns:
        Equivalence score in [0.0, 1.0] where 1.0 = perfect match
    """
    import torch

    model, tokenizer = _get_judge_model()

    prompt = _JUDGE_PROMPT.format(
        ground_truth=ground_truth.strip(),
        predicted=predicted.strip(),
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    # Decode only the generated tokens (exclude the prompt)
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    # Extract integer score from response
    match = _SCORE_PATTERN.search(generated)
    if match:
        raw_score = int(match.group(1))
        # Clamp to [0, 10] and normalize to [0, 1]
        return min(max(raw_score, 0), 10) / 10.0

    # If parsing fails, treat as uncertain → low score
    logger.warning(f"LLM judge returned unparseable response: '{generated}'")
    return 0.0
