#!/usr/bin/env python3
"""
Structured answer verification for two-trajectory GRPO rewards.

Wraps match_answer() and extract_answer_from_text() to produce a MatchResult
with an explicit CORRECT/WRONG verdict. There is no "undetermined" -- every
answer gets a definitive verdict.

For deterministic types (MCQ/YesNo/Numeric):
    Extraction failure → WRONG (models cannot escape penalties via garbled output).

For open-ended types:
    Multi-stage cascade: exact match → substring → token-F1 → ANLS → embedding
    cosine similarity (all-mpnet-base-v2). If all stages fail → WRONG.

Usage:
    from vlm_grpo.rewards.verifier import MatchResult, verify_answer

    result = verify_answer("B", "A", "mcq")
    assert result.verdict == "WRONG"

    result = verify_answer("a feline", "cat", "open")
    # Resolved by embedding cosine similarity
    assert result.verdict in ("CORRECT", "WRONG")
"""

import logging
import re
import string
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from vlm_grpo.rewards.deterministic import match_answer
from vlm_grpo.trajectory import extract_answer_from_text
from vlm_grpo.utils import normalized_edit_distance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Answer types where extraction failure = WRONG
DETERMINISTIC_TYPES: frozenset[str] = frozenset({"mcq", "yesno", "numeric"})

# Verdict constants
CORRECT = "CORRECT"
WRONG = "WRONG"

# Open-ended matching thresholds
_TOKEN_F1_THRESHOLD: float = 0.8
_ANLS_THRESHOLD: float = 0.7
_COSINE_SIM_THRESHOLD: float = 0.80

# Embedding model (lazy-loaded singleton)
_EMBED_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
_embed_model: Any = None


@dataclass
class MatchResult:
    """Structured result of answer verification.

    Attributes:
        answer_type: The answer type used for verification
        parse_ok: Whether answer extraction succeeded
        verdict: "CORRECT" or "WRONG"
        extracted: The normalized extracted answer string, or ""
        score: Similarity score for open-ended (cosine sim), None for deterministic
    """

    answer_type: str
    parse_ok: bool
    verdict: str
    extracted: str
    score: float | None

    @property
    def is_correct(self) -> bool:
        """True iff verdict == CORRECT."""
        return self.verdict == CORRECT

    @property
    def is_wrong(self) -> bool:
        """True iff verdict == WRONG."""
        return self.verdict == WRONG

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


# =============================================================================
# Embedding Model
# =============================================================================


def _get_embed_model() -> Any:
    """Lazy-load the sentence-transformer embedding model.

    Returns:
        SentenceTransformer model instance (cached after first call)
    """
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {_EMBED_MODEL_ID}")
        _embed_model = SentenceTransformer(_EMBED_MODEL_ID)
        logger.info("Embedding model loaded")
    return _embed_model


def _compute_cosine_similarity(text1: str, text2: str) -> float:
    """Compute cosine similarity between two texts using sentence embeddings.

    Args:
        text1: First text string
        text2: Second text string

    Returns:
        Cosine similarity in [-1.0, 1.0]
    """
    model = _get_embed_model()
    embeddings = model.encode([text1, text2], convert_to_numpy=True)
    cos_sim = float(
        np.dot(embeddings[0], embeddings[1])
        / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-10)
    )
    return cos_sim


# =============================================================================
# Text Normalization Helpers
# =============================================================================

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Args:
        text: Raw text string

    Returns:
        Normalized text
    """
    text = text.strip().lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> set[str]:
    """Tokenize normalized text into a set of words.

    Args:
        text: Normalized text string

    Returns:
        Set of word tokens
    """
    return set(_normalize_text(text).split())


def _compute_token_f1(pred: str, gt: str) -> float:
    """Compute token-set F1 score between prediction and ground truth.

    Args:
        pred: Predicted answer text
        gt: Ground truth answer text

    Returns:
        F1 score in [0.0, 1.0]
    """
    pred_tokens = _tokenize(pred)
    gt_tokens = _tokenize(gt)

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = pred_tokens & gt_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _check_substring_containment(pred: str, gt: str) -> bool:
    """Check if pred contains gt or gt contains pred (normalized).

    Args:
        pred: Predicted answer text
        gt: Ground truth answer text

    Returns:
        True if one contains the other
    """
    pred_norm = _normalize_text(pred)
    gt_norm = _normalize_text(gt)

    if not pred_norm or not gt_norm:
        return False

    return gt_norm in pred_norm or pred_norm in gt_norm


# =============================================================================
# Main Verification Function
# =============================================================================


def verify_answer(
    raw_text: str,
    ground_truth: str,
    answer_type: str,
    choices: str = "",
    tolerance: float = 0.01,
) -> MatchResult:
    """Verify a raw answer text against ground truth.

    Returns CORRECT or WRONG -- never undetermined. For deterministic types,
    extraction failure is WRONG. For open-ended, a multi-stage cascade
    determines correctness.

    Args:
        raw_text: Raw answer text from the model
        ground_truth: Ground truth answer string
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        choices: MCQ choices string
        tolerance: Numeric comparison tolerance

    Returns:
        MatchResult with verdict "CORRECT" or "WRONG"
    """
    extracted = extract_answer_from_text(raw_text, answer_type, choices)
    is_deterministic = answer_type in DETERMINISTIC_TYPES
    parse_ok = bool(extracted)

    # Extraction failure
    if not extracted:
        return MatchResult(
            answer_type=answer_type,
            parse_ok=False,
            verdict=WRONG,
            extracted="",
            score=None,
        )

    # Deterministic types: delegate to match_answer
    if is_deterministic:
        result = match_answer(extracted, ground_truth, answer_type, tolerance)
        verdict = CORRECT if result is True else WRONG
        return MatchResult(
            answer_type=answer_type,
            parse_ok=parse_ok,
            verdict=verdict,
            extracted=extracted,
            score=None,
        )

    # Open-ended: multi-stage cascade
    return _verify_open_ended(raw_text, extracted, ground_truth, answer_type)


def _verify_open_ended(
    raw_text: str,
    extracted: str,
    ground_truth: str,
    answer_type: str,
) -> MatchResult:
    """Run the open-ended verification cascade.

    Stages (in order, first match wins):
    1. Exact match (case-insensitive)
    2. Substring containment
    3. Token-set F1 >= 0.8
    4. ANLS >= 0.7
    5. Embedding cosine similarity >= 0.80
    6. Otherwise → WRONG

    Args:
        raw_text: Original raw text
        extracted: Extracted answer (stripped text for open type)
        ground_truth: Ground truth answer
        answer_type: Answer type string

    Returns:
        MatchResult with verdict and score
    """
    pred = extracted.strip().lower()
    gt = ground_truth.strip().lower()

    # Stage 1: Exact match
    if pred == gt:
        return MatchResult(
            answer_type=answer_type,
            parse_ok=True,
            verdict=CORRECT,
            extracted=extracted,
            score=1.0,
        )

    # Stage 2: Substring containment
    if _check_substring_containment(extracted, ground_truth):
        return MatchResult(
            answer_type=answer_type,
            parse_ok=True,
            verdict=CORRECT,
            extracted=extracted,
            score=0.95,
        )

    # Stage 3: Token-set F1
    token_f1 = _compute_token_f1(extracted, ground_truth)
    if token_f1 >= _TOKEN_F1_THRESHOLD:
        return MatchResult(
            answer_type=answer_type,
            parse_ok=True,
            verdict=CORRECT,
            extracted=extracted,
            score=token_f1,
        )

    # Stage 4: ANLS (1.0 - normalized_edit_distance)
    anls = 1.0 - normalized_edit_distance(pred, gt)
    if anls >= _ANLS_THRESHOLD:
        return MatchResult(
            answer_type=answer_type,
            parse_ok=True,
            verdict=CORRECT,
            extracted=extracted,
            score=anls,
        )

    # Stage 5: Embedding cosine similarity
    cos_sim = _compute_cosine_similarity(extracted, ground_truth)
    if cos_sim >= _COSINE_SIM_THRESHOLD:
        return MatchResult(
            answer_type=answer_type,
            parse_ok=True,
            verdict=CORRECT,
            extracted=extracted,
            score=cos_sim,
        )

    # All stages failed → WRONG
    return MatchResult(
        answer_type=answer_type,
        parse_ok=True,
        verdict=WRONG,
        extracted=extracted,
        score=cos_sim,
    )
