#!/usr/bin/env python3
"""
Deterministic reward components for MCQ, yes/no, numeric, and format scoring.

All reward components are pure functions operating on strings with no GPU dependency.
Each returns a scalar float reward value.

Usage:
    from vlm_grpo.rewards.deterministic import match_mcq, match_yesno, match_numeric

    assert match_mcq("A", "A") is True
    assert match_yesno("Yes", "yes") is True
    assert match_numeric("3.14", "3.14") is True
"""

import re

from vlm_grpo.trajectory import (
    ParsedTrajectory,
    detect_hedging,
    extract_answer_from_text,
)
from vlm_grpo.utils import normalized_edit_distance

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
    """Strict MCQ matching: single letter comparison.

    Both inputs should be single uppercase letters (A-F).
    Handles common variations like "(A)", "A.", "A)".

    Args:
        predicted: Extracted answer letter
        ground_truth: Ground truth letter

    Returns:
        True if the letters match
    """
    pred_clean = _extract_letter(predicted)
    gt_clean = _extract_letter(ground_truth)

    if not pred_clean or not gt_clean:
        return False

    return pred_clean == gt_clean


def match_yesno(predicted: str, ground_truth: str) -> bool:
    """Yes/No matching with normalization.

    Args:
        predicted: Extracted yes/no answer
        ground_truth: Ground truth yes/no

    Returns:
        True if both are the same (yes/yes or no/no)
    """
    pred_norm = predicted.strip().lower()
    gt_norm = ground_truth.strip().lower()

    # Normalize common variations
    pred_val = _normalize_yesno(pred_norm)
    gt_val = _normalize_yesno(gt_norm)

    if pred_val is None or gt_val is None:
        return False

    return pred_val == gt_val


def match_numeric(
    predicted: str,
    ground_truth: str,
    tolerance: float = 0.01,
) -> bool:
    """Numeric matching with tolerance.

    Supports integers, floats, and simple fractions (e.g., "1/3").

    Args:
        predicted: Extracted number string
        ground_truth: Ground truth number string
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
    """Match a predicted answer against ground truth based on answer type.

    Args:
        predicted: Predicted answer (normalized)
        ground_truth: Ground truth answer
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
        # Open-ended: exact match after normalization
        pred_norm = predicted.strip().lower()
        gt_norm = ground_truth.strip().lower()
        if pred_norm == gt_norm:
            return True
        # Cannot reliably determine for open-ended
        return None


# =============================================================================
# Reward Components
# =============================================================================


def compute_format_reward(
    trajectory: ParsedTrajectory,
    final_answer_extracted: str,
    answer_type: str,
) -> float:
    """R_format: Validate completion format and answer type compliance.

    Checks:
    1. Both FEEDBACK: and FINAL_ANSWER: markers present
    2. Final answer matches expected type (MCQ letter, Yes/No, number)
    3. No hedging in yes/no answers
    4. No multiple letters in MCQ answers

    Args:
        trajectory: Parsed trajectory
        final_answer_extracted: Normalized extracted answer
        answer_type: Expected answer type

    Returns:
        +1.0 if valid format, -1.0 if invalid
    """
    if not trajectory.parse_success:
        return -1.0

    if not trajectory.feedback.strip():
        return -1.0

    if not final_answer_extracted:
        return -1.0

    # Additional type-specific validation
    if answer_type == "yesno" and detect_hedging(trajectory.final_answer):
        return -1.0

    return 1.0


def compute_final_correct_reward(
    final_answer_extracted: str,
    ground_truth: str,
    answer_type: str,
    format_valid: bool,
    tolerance: float = 0.01,
) -> float:
    """R_final_correct: Check if final answer matches ground truth.

    Gated by format validity: returns 0.0 if format is invalid.

    Args:
        final_answer_extracted: Normalized extracted answer
        ground_truth: Ground truth answer
        answer_type: Answer type
        format_valid: Whether the format passed validation
        tolerance: Numeric tolerance

    Returns:
        +1.0 correct, -1.0 incorrect, 0.0 gated/undetermined
    """
    if not format_valid:
        return 0.0

    result = match_answer(final_answer_extracted, ground_truth, answer_type, tolerance)

    if result is True:
        return 1.0
    elif result is False:
        return -1.0
    else:
        return 0.0


def compute_no_regression_reward(
    final_answer_extracted: str,
    ground_truth: str,
    answer_type: str,
    format_valid: bool,
    tolerance: float = 0.01,
) -> float:
    """R_no_regression: Penalize RW flips heavily.

    Since all training samples have Answer1=correct:
    - If Answer2 correct: +1.0 (maintained correctness, RR)
    - If Answer2 wrong: -3.0 (regression, RW -- large penalty)
    - If cannot determine: 0.0

    Args:
        final_answer_extracted: Normalized extracted answer
        ground_truth: Ground truth answer
        answer_type: Answer type
        format_valid: Whether format validation passed
        tolerance: Numeric tolerance

    Returns:
        +1.0 for RR, -3.0 for RW, 0.0 for undetermined
    """
    if not format_valid:
        return 0.0

    result = match_answer(final_answer_extracted, ground_truth, answer_type, tolerance)

    if result is True:
        return 1.0
    elif result is False:
        return -3.0
    else:
        return 0.0


def compute_minimal_edit_reward(
    final_answer_extracted: str,
    ground_truth: str,
    answer1: str,
    answer_type: str,
    format_valid: bool,
    tolerance: float = 0.01,
) -> float:
    """R_minimal_edit: Reward minimal edits when both answers correct.

    If Answer1 correct AND Answer2 correct:
        reward = 1.0 - normalized_edit_distance(answer1, answer2)
    Otherwise:
        reward = 0.0

    This encourages the model to not change correct answers unnecessarily.

    Args:
        final_answer_extracted: Normalized extracted answer
        ground_truth: Ground truth answer
        answer1: Precomputed initial answer
        answer_type: Answer type
        format_valid: Whether format validation passed
        tolerance: Numeric tolerance

    Returns:
        [0.0, 1.0] where 1.0 means identical answers, or 0.0 if not applicable
    """
    if not format_valid:
        return 0.0

    # Check if Answer2 is correct
    result = match_answer(final_answer_extracted, ground_truth, answer_type, tolerance)
    if result is not True:
        return 0.0

    # Both correct: reward minimal edit
    # Normalize both for fair comparison
    a1_norm = answer1.strip().lower()
    a2_norm = final_answer_extracted.strip().lower()

    edit_dist = normalized_edit_distance(a1_norm, a2_norm)
    return 1.0 - edit_dist


def compute_feedback_calibration_reward(feedback: str) -> float:
    """R_feedback_calibration: Reward calibrated feedback.

    Since Answer1 is always correct in the RW-first dataset:
    - Feedback saying "correct" / "no change needed": +1.0
    - Feedback suggesting answer is wrong / needs change: -1.0
    - Cannot determine / neutral: 0.0

    Uses keyword detection for efficiency (no GPU needed).

    Args:
        feedback: Extracted feedback text

    Returns:
        +1.0 for calibrated, -1.0 for miscalibrated, 0.0 for neutral
    """
    if not feedback.strip():
        return 0.0

    positive_count = sum(1 for p in _POSITIVE_FEEDBACK_PATTERNS if p.search(feedback))
    negative_count = sum(1 for p in _NEGATIVE_FEEDBACK_PATTERNS if p.search(feedback))

    if positive_count > 0 and negative_count == 0:
        return 1.0
    elif negative_count > 0 and positive_count == 0:
        return -1.0
    elif negative_count > positive_count:
        return -0.5
    elif positive_count > negative_count:
        return 0.5
    else:
        return 0.0


# =============================================================================
# Private Helpers
# =============================================================================


def _extract_letter(text: str) -> str:
    """Extract a single MCQ letter from text.

    Args:
        text: Text possibly containing an MCQ letter

    Returns:
        Uppercase letter (A-F) or empty string
    """
    text = text.strip().upper()
    # Direct single letter
    if len(text) == 1 and text in "ABCDEF":
        return text
    # With parentheses: (A), A)
    match = re.match(r"^\s*\(?([A-F])\)?\s*$", text)
    if match:
        return match.group(1)
    return ""


def _normalize_yesno(text: str) -> bool | None:
    """Normalize yes/no answer to boolean.

    Args:
        text: Lowercased answer text

    Returns:
        True for yes, False for no, None if cannot determine
    """
    text = text.strip().lower()
    if text in ("yes", "y", "true"):
        return True
    if text in ("no", "n", "false"):
        return False
    # Try regex
    match = re.search(r"\b(yes|no)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "yes"
    return None


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
