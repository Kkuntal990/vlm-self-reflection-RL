#!/usr/bin/env python3
"""
LLM-as-judge for open-ended VQA answer equivalence.

Uses Qwen2.5-7B-Instruct on GPU to determine if a predicted answer is
semantically equivalent to a ground truth answer. Replaces embedding
cosine similarity for more accurate semantic matching in the open-ended
verification cascade.

The model is lazy-loaded as a singleton and runs on GPU in float16.
Results are cached in a module-level dict to avoid redundant inference.

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
_JUDGE_MODEL_ID = os.environ.get("VLM_JUDGE_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")

# Singleton model and tokenizer
_judge_model: Any = None
_judge_tokenizer: Any = None

# Module-level cache shared by both llm_judge_score and llm_judge_score_batch
_score_cache: dict[tuple[str, str], float] = {}

_JUDGE_PROMPT = """You are an answer correctness judge for visual question answering.

Your task is to compare a predicted answer with the ground truth answer for a \
given question and rate the prediction's correctness.

Evaluation criteria:
- If the ground truth is a definitive answer (a specific word, number, name, \
yes/no), strictly compare the prediction to the ground truth. The prediction \
must match in meaning.
- If the ground truth is open-ended (a description, explanation, or reasoning), \
the prediction is correct if it captures the key facts from the ground truth \
without introducing factual errors. A shorter but factually correct answer \
should still receive a high score.
- Synonyms, paraphrases, and equivalent expressions count as correct.
- Contradictions in key facts (wrong numbers, wrong objects, opposite \
directions) are always incorrect.

Examples of equivalent answers (score 10):
- "3" and "three"
- "There are three slices" and "3 slices"
- "No" and "No, it is not."
- "a cat" and "a domestic cat sitting on the table"
- "taxi" and "cab"

Examples of NOT equivalent answers (score 0):
- "3" and "4" (different numbers)
- "Yes" and "No" (opposite polarity)
- "left" and "right" (opposite direction)
- "dog" and "cat" (different entities)
- "bottles" and "breads" (different objects)

Rate the correctness of the predicted answer on a scale of 0-10:
- 10: Identical or perfect paraphrase (same meaning, same key facts)
- 7-9: Mostly correct (same core answer, minor wording or detail differences)
- 4-6: Partially correct (some overlap but missing or different key information)
- 1-3: Mostly wrong (different answer, different facts, contradictions)
- 0: Completely wrong or unrelated

Question: {question}
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


def _parse_score(generated: str) -> float:
    """Parse integer score from model output and normalize to [0, 1].

    Args:
        generated: Raw model output string.

    Returns:
        Score in [0.0, 1.0], or 0.0 if unparseable.
    """
    match = _SCORE_PATTERN.search(generated)
    if match:
        return min(max(int(match.group(1)), 0), 10) / 10.0
    logger.warning(f"LLM judge returned unparseable response: '{generated}'")
    return 0.0


def llm_judge_score(predicted: str, ground_truth: str, question: str = "") -> float:
    """Score the correctness of predicted answer vs ground truth using LLM.

    Uses Qwen2.5-7B-Instruct to produce a 0-10 score, normalized to [0, 1].
    Results are cached in a module-level dict (maxsize unlimited).

    Args:
        predicted: Predicted answer text
        ground_truth: Ground truth answer text
        question: Original question text (provides context for judgment)

    Returns:
        Correctness score in [0.0, 1.0] where 1.0 = perfect match
    """
    import torch

    key = (predicted.strip(), ground_truth.strip())
    if key in _score_cache:
        return _score_cache[key]

    model, tokenizer = _get_judge_model()

    prompt = _JUDGE_PROMPT.format(
        question=question.strip() if question else "(not provided)",
        ground_truth=key[1],
        predicted=key[0],
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

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    score = _parse_score(generated)
    _score_cache[key] = score
    return score


def llm_judge_score_batch(pairs: list[tuple[str, str]]) -> list[float]:
    """Score a batch of (predicted, ground_truth) pairs in one generate() call.

    Checks the module-level cache first; only uncached pairs go to the GPU.
    All uncached pairs are processed in a single batched generate() call,
    replacing N sequential calls with one.

    Args:
        pairs: List of (predicted, ground_truth) tuples to score.

    Returns:
        List of equivalence scores in [0.0, 1.0], one per input pair.
    """
    import torch

    if not pairs:
        return []

    # Normalize keys and check cache
    keys = [(p.strip(), g.strip()) for p, g in pairs]
    scores: list[float | None] = [_score_cache.get(k) for k in keys]

    # Collect unique uncached pairs (preserve first-occurrence index)
    seen: dict[tuple[str, str], int] = {}  # key → position in inference list
    inference_keys: list[tuple[str, str]] = []
    for i, key in enumerate(keys):
        if scores[i] is None and key not in seen:
            seen[key] = len(inference_keys)
            inference_keys.append(key)

    if inference_keys:
        model, tokenizer = _get_judge_model()

        texts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": _JUDGE_PROMPT.format(ground_truth=gt, predicted=pred),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for pred, gt in inference_keys
        ]

        orig_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        finally:
            tokenizer.padding_side = orig_side

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for pos, key in enumerate(inference_keys):
            generated = tokenizer.decode(
                outputs[pos][prompt_len:], skip_special_tokens=True
            ).strip()
            _score_cache[key] = _parse_score(generated)

    # Fill in all results from cache (now fully populated)
    return [_score_cache[k] for k in keys]


# =============================================================================
# LLM Format Judge
# =============================================================================

_FORMAT_JUDGE_PROMPT = """\
You are a format compliance judge for visual question answering.

Your task: Determine whether the predicted answer has the \
SAME FORMAT as the ground truth answer. \
You are NOT judging correctness — only format similarity.

Format means:
- Same structural type (single letter, letter with text, \
yes/no, number, short phrase, full sentence)
- Similar length (within 2x token count)

Examples of MATCHING format (score 10):
- GT: "B. 24" Pred: "C. 18" (both letter-dot-number)
- GT: "B. Yes" Pred: "A. No" (both letter-dot-word)
- GT: "B" Pred: "C" (both single letters)
- GT: "The fence is in front" Pred: "The cat is behind" (both sentences)
- GT: "15" Pred: "36" (both single numbers)
- GT: "red" Pred: "blue" (both single words)
- GT: "Yes" Pred: "No" (both single yes/no)

Examples of NON-MATCHING format (score 0):
- GT: "B" Pred: "The answer is B because..." (letter vs explanation)
- GT: "Yes" Pred: "I think it might be correct..." (word vs verbose)
- GT: "3" Pred: "There are approximately three items" (number vs sentence)
- GT: "red" Pred: "The dominant color appears to be red" (word vs sentence)

Rate format similarity on a scale of 0-10:
- 10: Identical format
- 7-9: Very similar format (minor differences in length)
- 4-6: Somewhat similar
- 0-3: Different format (e.g., single token vs multi-sentence)

Ground Truth: {ground_truth}
Predicted Answer: {predicted}

Respond with only a single integer from 0 to 10."""

# Separate cache for format judgments (keyed by pred, gt, answer_type)
_format_cache: dict[tuple[str, str, str], float] = {}


def llm_format_judge(
    predicted: str,
    ground_truth: str,
    answer_type: str,
) -> float:
    """Judge whether predicted answer has the same format as ground truth.

    Uses the Qwen2.5-7B-Instruct model to evaluate format similarity,
    NOT semantic correctness. Results are cached.

    Args:
        predicted: Predicted answer text
        ground_truth: Ground truth answer text
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")

    Returns:
        Format similarity score in [0.0, 1.0] where 1.0 = identical format
    """
    import torch

    key = (predicted.strip(), ground_truth.strip(), answer_type)
    if key in _format_cache:
        return _format_cache[key]

    model, tokenizer = _get_judge_model()

    prompt = _FORMAT_JUDGE_PROMPT.format(
        ground_truth=key[1],
        predicted=key[0],
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

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    score = _parse_score(generated)
    _format_cache[key] = score
    return score
