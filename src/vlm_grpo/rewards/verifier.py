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
from vlm_grpo.trajectory import extract_answer_from_text, extract_mcq_letter_and_text
from vlm_grpo.utils import normalized_edit_distance

# Word-number mapping for counting extraction (Rule 6)
_WORD_TO_NUM: dict[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
_WORD_NUM_PATTERN = re.compile(r"\b(" + "|".join(_WORD_TO_NUM.keys()) + r")\b", re.IGNORECASE)
_NUMBER_IN_TEXT_PATTERN = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

# Pattern for yes/no at start of sentence (Rule 4)
_YESNO_START_PATTERN = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Answer types where extraction failure = WRONG (before fallback checks)
DETERMINISTIC_TYPES: frozenset[str] = frozenset({"mcq", "yesno", "numeric"})

# Verdict constants
CORRECT = "CORRECT"
WRONG = "WRONG"

# Open-ended matching thresholds
_TOKEN_F1_THRESHOLD: float = 0.85
_ANLS_THRESHOLD: float = 0.7
_COSINE_SIM_THRESHOLD: float = 0.80

# Common color words. If both answers mention a color and the colors differ,
# treat the answer as contradictory before any fuzzy semantic matching.
_COLOR_WORDS: frozenset[str] = frozenset(
    {
        "black",
        "blue",
        "brown",
        "gray",
        "green",
        "grey",
        "orange",
        "pink",
        "purple",
        "red",
        "white",
        "yellow",
    }
)

# Antonym pairs for contradiction detection in open-ended matching.
# If pred and GT differ by tokens in one of these pairs, the answer is
# considered WRONG regardless of surface-level similarity scores.
_ANTONYM_PAIRS: list[frozenset[str]] = [
    frozenset({"left", "right"}),
    frozenset({"up", "down"}),
    frozenset({"above", "below"}),
    frozenset({"top", "bottom"}),
    frozenset({"front", "back"}),
    frozenset({"inside", "outside"}),
    frozenset({"before", "after"}),
    frozenset({"yes", "no"}),
    frozenset({"true", "false"}),
    frozenset({"positive", "negative"}),
    frozenset({"higher", "lower"}),
    frozenset({"more", "less"}),
    frozenset({"larger", "smaller"}),
    frozenset({"bigger", "smaller"}),
    frozenset({"increase", "decrease"}),
    frozenset({"north", "south"}),
    frozenset({"east", "west"}),
    frozenset({"horizontal", "vertical"}),
    frozenset({"clockwise", "counterclockwise"}),
    frozenset({"ascending", "descending"}),
]

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

    Returns CORRECT or WRONG -- never undetermined.

    Dispatches by answer_type:
    - mcq: letter comparison, fallback to open if GT has no letter (Rule 3)
    - yesno: sentence-start verdict (Rule 4), fallback to open if no
      yes/no word in GT (Rule 5)
    - counting: number extraction from sentences + word-numbers (Rule 6),
      fallback to open if no number found
    - numeric: standard numeric extraction and comparison
    - open: multi-stage cascade (exact → substring → F1 → ANLS → embedding)

    Args:
        raw_text: Raw answer text from the model
        ground_truth: Ground truth answer string
        answer_type: Answer type ("mcq", "yesno", "numeric", "counting", "open")
        choices: MCQ choices string
        tolerance: Numeric comparison tolerance

    Returns:
        MatchResult with verdict "CORRECT" or "WRONG"
    """
    if answer_type == "mcq":
        return _verify_mcq(raw_text, ground_truth, choices)
    elif answer_type == "yesno":
        return _verify_yesno(raw_text, ground_truth)
    elif answer_type == "counting":
        return _verify_counting(raw_text, ground_truth)
    elif answer_type == "numeric":
        return _verify_numeric(raw_text, ground_truth, tolerance)
    else:
        # Open-ended: multi-stage cascade
        extracted = raw_text.strip()
        if not extracted:
            return MatchResult(
                answer_type=answer_type,
                parse_ok=False,
                verdict=WRONG,
                extracted="",
                score=None,
            )
        return _verify_open_ended(raw_text, extracted, ground_truth, answer_type)


def _verify_mcq(
    raw_text: str,
    ground_truth: str,
    choices: str = "",
) -> MatchResult:
    """MCQ verification with open-ended fallback (Rules 2, 3).

    1. Extract letter from prediction and GT
    2. If both have letters, compare them
    3. If GT has no letter (freeform GT), fallback to open-ended cascade

    Args:
        raw_text: Raw answer text
        ground_truth: Ground truth string
        choices: MCQ choices string

    Returns:
        MatchResult
    """
    pred_extracted = extract_answer_from_text(raw_text, "mcq", choices)
    gt_extracted = extract_answer_from_text(ground_truth, "mcq", choices)

    # Both have letters → compare letters
    if pred_extracted and gt_extracted:
        verdict = CORRECT if pred_extracted == gt_extracted else WRONG
        return MatchResult(
            answer_type="mcq",
            parse_ok=True,
            verdict=verdict,
            extracted=pred_extracted,
            score=None,
        )

    # Pred has letter but GT doesn't → check answer text against GT
    if pred_extracted and not gt_extracted:
        _letter, answer_text = extract_mcq_letter_and_text(raw_text)
        if answer_text:
            gt_norm = ground_truth.strip().lower()
            at_norm = answer_text.strip().lower()
            if gt_norm == at_norm:
                return MatchResult(
                    answer_type="mcq",
                    parse_ok=True,
                    verdict=CORRECT,
                    extracted=raw_text.strip(),
                    score=None,
                )

    # Rule 3: GT has no letter (freeform) → fallback to open-ended
    if not gt_extracted:
        extracted = raw_text.strip()
        if not extracted:
            return MatchResult(
                answer_type="mcq",
                parse_ok=False,
                verdict=WRONG,
                extracted="",
                score=None,
            )
        return _verify_open_ended(raw_text, extracted, ground_truth, "mcq")

    # Pred extraction failed, GT has letter → WRONG
    return MatchResult(
        answer_type="mcq",
        parse_ok=False,
        verdict=WRONG,
        extracted="",
        score=None,
    )


def _verify_yesno(
    raw_text: str,
    ground_truth: str,
) -> MatchResult:
    """Yes/No verification with sentence-start and open-ended fallback (Rules 4, 5).

    1. Check if both pred and GT start with Yes/No → compare verdicts
    2. If GT has no yes/no word → fallback to open-ended cascade
    3. If pred has no yes/no but GT does → WRONG

    Args:
        raw_text: Raw answer text
        ground_truth: Ground truth string

    Returns:
        MatchResult
    """
    from vlm_grpo.trajectory import detect_hedging

    # Anti-hacking: reject hedging predictions
    if detect_hedging(raw_text):
        return MatchResult(
            answer_type="yesno",
            parse_ok=False,
            verdict=WRONG,
            extracted="",
            score=None,
        )

    # Rule 4: Extract yes/no from sentence start
    pred_match = _YESNO_START_PATTERN.match(raw_text)
    gt_match = _YESNO_START_PATTERN.match(ground_truth)

    pred_verdict = pred_match.group(1).lower() if pred_match else None
    gt_verdict = gt_match.group(1).lower() if gt_match else None

    # Both have yes/no at start → compare verdicts
    if pred_verdict and gt_verdict:
        verdict = CORRECT if pred_verdict == gt_verdict else WRONG

        # When verdicts agree, a bare polarity match is not enough — the
        # model may hallucinate the wrong objects (e.g. pred="Yes, bottles"
        # vs GT="Yes, breads").  Ask the LLM judge to check semantic
        # equivalence of the *full* answers.  If the judge is disabled,
        # fall back to the open-ended cascade (token-F1 + embeddings).
        if verdict == CORRECT:
            try:
                from vlm_grpo.rewards.judge_llm import is_enabled, llm_judge_score

                if is_enabled():
                    score = llm_judge_score(raw_text.strip(), ground_truth)
                    verdict = CORRECT if score >= 0.7 else WRONG
                    return MatchResult(
                        answer_type="yesno",
                        parse_ok=True,
                        verdict=verdict,
                        extracted=pred_verdict.capitalize(),
                        score=score,
                    )
            except Exception as e:
                logger.warning(f"LLM judge failed in yesno path, using open-ended: {e}")
                return _verify_open_ended(raw_text, raw_text.strip(), ground_truth, "yesno")

        return MatchResult(
            answer_type="yesno",
            parse_ok=True,
            verdict=verdict,
            extracted=pred_verdict.capitalize(),
            score=None,
        )

    # Rule 5: GT has no yes/no word anywhere → fallback to open-ended
    gt_has_yesno = re.search(r"\b(yes|no)\b", ground_truth, re.IGNORECASE)
    if not gt_has_yesno:
        extracted = raw_text.strip()
        if not extracted:
            return MatchResult(
                answer_type="yesno",
                parse_ok=False,
                verdict=WRONG,
                extracted="",
                score=None,
            )
        return _verify_open_ended(raw_text, extracted, ground_truth, "yesno")

    # GT has yes/no somewhere (maybe not at start) — try broader extraction
    pred_anywhere = re.search(r"\b(yes|no)\b", raw_text, re.IGNORECASE)
    gt_anywhere = re.search(r"\b(yes|no)\b", ground_truth, re.IGNORECASE)
    if pred_anywhere and gt_anywhere:
        p = pred_anywhere.group(1).lower()
        g = gt_anywhere.group(1).lower()
        verdict = CORRECT if p == g else WRONG

        # Same LLM-judge guard as above.
        if verdict == CORRECT:
            try:
                from vlm_grpo.rewards.judge_llm import is_enabled, llm_judge_score

                if is_enabled():
                    score = llm_judge_score(raw_text.strip(), ground_truth)
                    verdict = CORRECT if score >= 0.7 else WRONG
                    return MatchResult(
                        answer_type="yesno",
                        parse_ok=True,
                        verdict=verdict,
                        extracted=p.capitalize(),
                        score=score,
                    )
            except Exception as e:
                logger.warning(f"LLM judge failed in yesno broad path, using open-ended: {e}")
                return _verify_open_ended(raw_text, raw_text.strip(), ground_truth, "yesno")

        return MatchResult(
            answer_type="yesno",
            parse_ok=True,
            verdict=verdict,
            extracted=p.capitalize(),
            score=None,
        )

    # Pred has no yes/no word at all but GT does → WRONG
    return MatchResult(
        answer_type="yesno",
        parse_ok=False,
        verdict=WRONG,
        extracted="",
        score=None,
    )


def _verify_counting(
    raw_text: str,
    ground_truth: str,
) -> MatchResult:
    """Counting verification: extract numbers from sentences (Rule 6).

    1. Extract numbers (digit or word) from both pred and GT
    2. If both yield a number, compare numerically (exact integer match)
    3. If either fails, fallback to open-ended cascade

    Args:
        raw_text: Raw answer text
        ground_truth: Ground truth string

    Returns:
        MatchResult
    """
    pred_num = _extract_number_from_sentence(raw_text)
    gt_num = _extract_number_from_sentence(ground_truth)

    # Both have numbers → compare
    if pred_num is not None and gt_num is not None:
        # Integer comparison for counting
        if isinstance(pred_num, float) or isinstance(gt_num, float):
            match = abs(pred_num - gt_num) < 0.01
        else:
            match = pred_num == gt_num
        verdict = CORRECT if match else WRONG
        return MatchResult(
            answer_type="counting",
            parse_ok=True,
            verdict=verdict,
            extracted=str(pred_num),
            score=None,
        )

    # Fallback to open-ended
    extracted = raw_text.strip()
    if not extracted:
        return MatchResult(
            answer_type="counting",
            parse_ok=False,
            verdict=WRONG,
            extracted="",
            score=None,
        )
    return _verify_open_ended(raw_text, extracted, ground_truth, "counting")


def _verify_numeric(
    raw_text: str,
    ground_truth: str,
    tolerance: float = 0.01,
) -> MatchResult:
    """Standard numeric verification.

    Args:
        raw_text: Raw answer text
        ground_truth: Ground truth string
        tolerance: Numeric comparison tolerance

    Returns:
        MatchResult
    """
    extracted = extract_answer_from_text(raw_text, "numeric")
    if not extracted:
        return MatchResult(
            answer_type="numeric",
            parse_ok=False,
            verdict=WRONG,
            extracted="",
            score=None,
        )

    result = match_answer(extracted, ground_truth, "numeric", tolerance)
    verdict = CORRECT if result is True else WRONG
    return MatchResult(
        answer_type="numeric",
        parse_ok=True,
        verdict=verdict,
        extracted=extracted,
        score=None,
    )


def _extract_number_from_sentence(text: str) -> int | float | None:
    """Extract the key number from a sentence, supporting word-numbers.

    Tries digit numbers first (uses last match), then word-numbers.

    Args:
        text: Text possibly containing a number

    Returns:
        Extracted number or None
    """
    text = text.strip()
    if not text:
        return None

    # Find all digit numbers in text
    digit_matches = list(_NUMBER_IN_TEXT_PATTERN.finditer(text))
    if digit_matches:
        # Use the last number found (the key answer tends to be last)
        num_str = digit_matches[-1].group(0).replace(",", "")
        try:
            val = float(num_str)
            return int(val) if val == int(val) else val
        except ValueError:
            pass

    # Try word-numbers
    word_match = _WORD_NUM_PATTERN.search(text)
    if word_match:
        return _WORD_TO_NUM[word_match.group(1).lower()]

    return None


def _has_antonym_contradiction(pred: str, gt: str) -> bool:
    """Check if pred and GT differ by a critical antonym pair.

    Detects cases where sentences are near-identical but differ in a key
    directional or categorical word (left/right, yes/no, up/down, etc.).
    This prevents the cascade from accepting semantically opposite answers
    that happen to share most tokens.

    Args:
        pred: Predicted answer text
        gt: Ground truth answer text

    Returns:
        True if an antonym contradiction is detected
    """
    pred_tokens = _tokenize(pred)
    gt_tokens = _tokenize(gt)

    pred_only = pred_tokens - gt_tokens
    gt_only = gt_tokens - pred_tokens

    for pair in _ANTONYM_PAIRS:
        if pred_only & pair and gt_only & pair:
            return True
    return False


def _has_color_conflict(pred: str, gt: str) -> bool:
    """Check if pred and GT mention different colors.

    This catches common VQA failures like "red" vs "pink" that can look
    superficially similar under edit-distance scoring because the surrounding
    sentence is nearly identical.
    """
    pred_colors = _tokenize(pred) & _COLOR_WORDS
    gt_colors = _tokenize(gt) & _COLOR_WORDS
    if not pred_colors or not gt_colors:
        return False
    return pred_colors != gt_colors


def _is_atomic_anls_candidate(pred: str, gt: str) -> bool:
    """Whether ANLS is appropriate for this answer pair.

    ANLS is useful for OCR-style typos on short atomic answers ("colur" vs
    "color"), but it is too permissive for sentence-level VQA answers where a
    single wrong token can still yield a high string-similarity score.
    """
    pred_norm = _normalize_text(pred)
    gt_norm = _normalize_text(gt)
    if not pred_norm or not gt_norm:
        return False
    return len(pred_norm.split()) == 1 and len(gt_norm.split()) == 1


def _verify_open_ended(
    raw_text: str,
    extracted: str,
    ground_truth: str,
    answer_type: str,
) -> MatchResult:
    """Run the open-ended verification cascade.

    Stages (in order, first match wins):
    0. Deterministic contradiction checks → WRONG for antonym/color conflicts
    1. Exact match (case-insensitive)
    2. Substring containment
    3. Token-set F1 >= 0.85
    4. ANLS >= 0.7 for atomic answers only
    5. LLM judge (Qwen2.5-3B) → CORRECT/WRONG
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

    # Stage 0: Deterministic contradiction checks
    # If the differing tokens form an antonym pair (left/right, yes/no, etc.),
    # the answer is semantically opposite regardless of surface similarity.
    if _has_antonym_contradiction(extracted, ground_truth):
        return MatchResult(
            answer_type=answer_type,
            parse_ok=True,
            verdict=WRONG,
            extracted=extracted,
            score=None,
        )
    if _has_color_conflict(extracted, ground_truth):
        return MatchResult(
            answer_type=answer_type,
            parse_ok=True,
            verdict=WRONG,
            extracted=extracted,
            score=None,
        )

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
    # Restrict to short atomic answers. On sentence-level outputs ANLS is too
    # forgiving and can accept wrong answers that differ in the key noun/adjective.
    if _is_atomic_anls_candidate(pred, gt):
        anls = 1.0 - normalized_edit_distance(pred, gt)
        if anls >= _ANLS_THRESHOLD:
            return MatchResult(
                answer_type=answer_type,
                parse_ok=True,
                verdict=CORRECT,
                extracted=extracted,
                score=anls,
            )
    else:
        anls = None

    # Stage 5: LLM judge (Qwen2.5-3B-Instruct) or embedding fallback
    # LLM judge is enabled via VLM_USE_LLM_JUDGE=1 environment variable.
    try:
        from vlm_grpo.rewards.judge_llm import is_enabled, llm_judge_score

        if is_enabled():
            score = llm_judge_score(extracted, ground_truth)
            verdict = CORRECT if score >= 0.7 else WRONG
            return MatchResult(
                answer_type=answer_type,
                parse_ok=True,
                verdict=verdict,
                extracted=extracted,
                score=score,
            )
    except Exception as e:
        logger.warning(f"LLM judge failed, falling back to embedding: {e}")

    # Fallback: Embedding cosine similarity (when LLM judge disabled or failed)
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
        score=cos_sim if "cos_sim" in locals() else anls,
    )
