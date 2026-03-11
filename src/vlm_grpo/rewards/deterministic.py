#!/usr/bin/env python3
"""
Deterministic answer matching for MCQ, yes/no, and numeric answer types.

All functions are pure string operations with no GPU dependency.

Usage:
    from vlm_grpo.rewards.deterministic import match_mcq, match_yesno, match_numeric

    assert match_mcq("a", "a") is True
    assert match_yesno("yes", "yes") is True
    assert match_numeric("3.14", "3.14") is True
"""

import re

from vlm_grpo.trajectory import normalize_answer

# Feedback calibration keyword patterns
_POSITIVE_FEEDBACK_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bcorrect\b",
        r"\baccurate\b",
        r"\bright\b",
        r"\bno change needed\b",
        r"\bno changes? (?:are |is )?(?:needed|necessary|required)\b",
        r"\bwell done\b",
        r"\bgood (?:answer|response|job)\b",
        r"\bmatches?\b.*\b(?:image|visual|evidence)\b",
    ]
]

_NEGATIVE_FEEDBACK_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bincorrect\b",
        r"\bwrong\b",
        r"\berror\b",
        r"\bmistake\b",
        r"\bshould be\b",
        r"\bshould have been\b",
        r"\bneeds? (?:to be )?(?:corrected|changed|revised|fixed|updated)\b",
        r"\bnot (?:correct|accurate|right)\b",
        r"\bdoes(?:n't| not) match\b",
        r"\breconsider\b",
        r"\bactually\b.*\bnot\b",
    ]
]


# =============================================================================
# Answer Matching Functions
# =============================================================================


def match_mcq(predicted: str, ground_truth: str) -> bool:
    """MCQ matching on normalized (lowercase) answers.

    Both inputs should already be normalized via normalize_answer().
    Direct string comparison of single lowercase letters.

    Args:
        predicted: Normalized answer (e.g. "a", "b")
        ground_truth: Normalized ground truth (e.g. "a", "b")

    Returns:
        True if the letters match
    """
    return predicted == ground_truth


def match_yesno(predicted: str, ground_truth: str) -> bool:
    """Yes/No matching on normalized (lowercase) answers.

    Both inputs should already be normalized via normalize_answer().
    Direct string comparison of "yes" or "no".

    Args:
        predicted: Normalized answer ("yes" or "no")
        ground_truth: Normalized ground truth ("yes" or "no")

    Returns:
        True if both are the same
    """
    return predicted == ground_truth


def match_numeric(
    predicted: str,
    ground_truth: str,
    tolerance: float = 0.01,
) -> bool:
    """Numeric matching with tolerance.

    Both inputs should already be normalized via normalize_answer().
    Supports integers, floats, and simple fractions (e.g., "1/3").

    Args:
        predicted: Normalized number string
        ground_truth: Normalized ground truth number string
        tolerance: Relative tolerance for comparison

    Returns:
        True if match within tolerance
    """
    pred_val = _parse_number(predicted)
    gt_val = _parse_number(ground_truth)

    if pred_val is None or gt_val is None:
        return False

    if gt_val == 0:
        return abs(pred_val) < tolerance

    return abs(pred_val - gt_val) / max(abs(gt_val), 1e-10) <= tolerance


def match_answer(
    predicted: str,
    ground_truth: str,
    answer_type: str,
    tolerance: float = 0.01,
) -> bool | None:
    """Match predicted answer against ground truth based on answer type.

    Both inputs should already be normalized via normalize_answer().

    Args:
        predicted: Normalized predicted answer
        ground_truth: Normalized ground truth answer
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        tolerance: Numeric tolerance

    Returns:
        True if correct, False if incorrect, None if cannot determine
    """
    if not predicted:
        return None

    if answer_type == "mcq":
        return match_mcq(predicted, ground_truth)
    elif answer_type == "yesno":
        return match_yesno(predicted, ground_truth)
    elif answer_type == "numeric":
        return match_numeric(predicted, ground_truth, tolerance)
    else:
        # Open-ended: direct comparison (both already normalized/lowercase)
        if predicted == ground_truth:
            return True
        # Cannot reliably determine for open-ended
        return None


# =============================================================================
# Private Helpers
# =============================================================================


def _parse_number(text: str) -> float | None:
    """Parse a number from text, supporting fractions.

    Args:
        text: Text containing a number

    Returns:
        Float value or None if parsing failed
    """
    text = text.strip()
    # Handle fractions like "1/3"
    if "/" in text:
        parts = text.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                return None
    # Handle percentage
    text = text.rstrip("%")
    # Handle commas in numbers
    text = text.replace(",", "")
    try:
        return float(text)
    except ValueError:
        # Try extracting first number
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
        return None
