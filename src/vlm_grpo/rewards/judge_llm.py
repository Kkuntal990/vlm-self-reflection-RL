#!/usr/bin/env python3
"""
Optional LLM judge adapter for open-ended answer verification.

Provides an LLM-based judge for evaluating open-ended answers where
deterministic matching is insufficient. Uses hash-based caching to
avoid re-judging identical inputs.

This module is a placeholder for the RW-first phase where MCQ/YesNo
questions dominate. It can be activated later for open-ended evaluation.

Usage:
    from vlm_grpo.rewards.judge_llm import LLMJudgeAdapter

    judge = LLMJudgeAdapter(api_base="http://localhost:8000/v1")
    score = judge.judge_open_ended(
        question="What is the man doing?",
        predicted="cooking pasta",
        ground_truth="making pizza",
    )
"""

import hashlib
import logging
import sys
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Judge prompt template
_JUDGE_PROMPT = """You are evaluating whether a predicted answer is semantically equivalent to a ground truth answer for a visual question answering task.

Question: {question}
Ground Truth: {ground_truth}
Predicted: {predicted}

Is the predicted answer correct? Consider:
- Semantic equivalence (different wording, same meaning)
- Acceptable abbreviations or synonyms
- Minor formatting differences are OK

Respond with ONLY one of: CORRECT, INCORRECT, or UNCERTAIN"""


class LLMJudgeAdapter:
    """Adapter for using an LLM as judge for open-ended answers.

    Only used when answer_type == "open" and deterministic scoring
    cannot resolve correctness.

    Attributes:
        model_id: HuggingFace model ID or API model name
        api_base: Optional API base URL (for vLLM/TGI server)
        cache: Hash-based cache for judge results
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-72B-Instruct",
        api_base: Optional[str] = None,
        max_cache_size: int = 10000,
    ):
        """Initialize the LLM judge.

        Args:
            model_id: Model to use for judging
            api_base: Optional API base URL (for vLLM/TGI server)
            max_cache_size: Maximum number of cached judgments
        """
        self.model_id = model_id
        self.api_base = api_base
        self.max_cache_size = max_cache_size
        self._cache: dict[str, float] = {}
        self._client = None

        logger.info(f"LLMJudgeAdapter initialized (model={model_id}, api_base={api_base})")

    def _get_client(self):
        """Lazy-load the OpenAI-compatible client."""
        if self._client is not None:
            return self._client

        # Lazy import to avoid loading heavy libraries until needed
        from openai import OpenAI

        if self.api_base:
            self._client = OpenAI(base_url=self.api_base, api_key="dummy")
        else:
            self._client = OpenAI()

        return self._client

    def _cache_key(self, question: str, predicted: str, ground_truth: str) -> str:
        """Generate cache key from inputs.

        Args:
            question: Question text
            predicted: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            Cache key string
        """
        content = f"{question}|{predicted}|{ground_truth}"
        return hashlib.sha256(content.encode()).hexdigest()[:20]

    def judge_open_ended(
        self,
        question: str,
        predicted: str,
        ground_truth: str,
    ) -> float:
        """Judge whether an open-ended answer is correct.

        Args:
            question: The original question
            predicted: Model's predicted answer
            ground_truth: Ground truth answer

        Returns:
            1.0 if correct, -1.0 if incorrect, 0.0 if uncertain
        """
        # Check cache
        key = self._cache_key(question, predicted, ground_truth)
        if key in self._cache:
            return self._cache[key]

        # Quick exact match check
        if predicted.strip().lower() == ground_truth.strip().lower():
            self._cache[key] = 1.0
            return 1.0

        try:
            client = self._get_client()
            prompt = _JUDGE_PROMPT.format(
                question=question,
                ground_truth=ground_truth,
                predicted=predicted,
            )

            response = client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0,
            )

            result_text = response.choices[0].message.content.strip().upper()

            if "CORRECT" in result_text and "INCORRECT" not in result_text:
                score = 1.0
            elif "INCORRECT" in result_text:
                score = -1.0
            else:
                score = 0.0

        except Exception as e:
            logger.warning(f"LLM judge failed: {e}")
            score = 0.0

        # Cache result
        if len(self._cache) < self.max_cache_size:
            self._cache[key] = score

        return score
