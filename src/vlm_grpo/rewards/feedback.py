#!/usr/bin/env python3
"""
Feedback reward components for two-trajectory GRPO critic training.

Computes rewards based on feedback quality: calibration (does the feedback
correctly assess A1's correctness?) and downstream-awareness (does the
feedback lead to a correct A2?).

Uses verify_answer() which returns only CORRECT or WRONG (no undetermined).
Includes soft doubt patterns that penalize unnecessary hedging on correct
answers.

Usage:
    from vlm_grpo.rewards.feedback import (
        compute_feedback_calibration_reward,
        compute_downstream_aware_reward,
    )

    r = compute_feedback_calibration_reward(
        "The answer is correct.", a1_is_correct=True,
    )
    assert r == 1.0
"""

import re

from vlm_grpo.rewards.deterministic import (
    _HEDGED_POSITIVE_PATTERNS,
    _NEGATIVE_FEEDBACK_PATTERNS,
    _POSITIVE_FEEDBACK_PATTERNS,
)
from vlm_grpo.rewards.verifier import verify_answer

# Soft doubt patterns: expressions that cast unnecessary doubt
# These trigger a mild penalty when A1 is correct (the model is creating
# uncertainty where none is needed, which can lead to RW flips).
_SOFT_DOUBT_PATTERNS: list[re.Pattern] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bmight be incorrect\b",
        r"\bdouble.?check\b",
        r"\bre.?evaluate\b",
        r"\bnot entirely (?:sure|certain)\b",
        r"\bworth (?:double.?checking|verifying)\b",
    ]
]


def compute_feedback_calibration_reward(
    feedback_text: str,
    a1_is_correct: bool,
) -> float:
    """R_feedback_calibration: Reward feedback that correctly assesses A1.

    When A1 is correct:
        Positive feedback ("correct", "no change needed") → +1.0
        Negative feedback ("incorrect", "should be")      → -1.0
        Mixed (more positive than negative)               → +0.5
        Mixed (more negative than positive)               → -0.5
        Soft doubt only ("double-check", "re-evaluate")   → -0.3
        Neutral / cannot determine                        → 0.0

    When A1 is wrong:
        Negative feedback (correctly identifies error)    → +1.0
        Positive feedback (missed the error)              → -1.0
        Mixed (more negative than positive)               → +0.5
        Mixed (more positive than negative)               → -0.5
        Soft doubt only                                   → +0.3
        Neutral / cannot determine                        → 0.0

    Args:
        feedback_text: Extracted feedback text
        a1_is_correct: Whether A1 was correct

    Returns:
        Calibration reward in [-1.0, +1.0]
    """
    if not feedback_text.strip():
        return 0.0

    # Hedged positives ("partially correct", "on the right track, but...")
    # look positive but are actually corrections. Count them as negative.
    hedged_count = sum(1 for p in _HEDGED_POSITIVE_PATTERNS if p.search(feedback_text))

    positive_count = sum(1 for p in _POSITIVE_FEEDBACK_PATTERNS if p.search(feedback_text))
    negative_count = sum(1 for p in _NEGATIVE_FEEDBACK_PATTERNS if p.search(feedback_text))
    soft_doubt_count = sum(1 for p in _SOFT_DOUBT_PATTERNS if p.search(feedback_text))

    # Hedged positives override: reclassify as negative
    if hedged_count > 0:
        positive_count = max(0, positive_count - hedged_count)
        negative_count += hedged_count

    # Clear positive or negative feedback
    if positive_count > 0 and negative_count == 0:
        return 1.0 if a1_is_correct else -1.0
    if negative_count > 0 and positive_count == 0:
        return 1.0 if not a1_is_correct else -1.0

    # Mixed feedback
    if positive_count > negative_count:
        return 0.5 if a1_is_correct else -0.5
    if negative_count > positive_count:
        return -0.5 if a1_is_correct else 0.5

    # No positive/negative patterns matched -- check soft doubt
    if soft_doubt_count > 0:
        return -0.3 if a1_is_correct else 0.3

    return 0.0




def compute_downstream_aware_reward(
    feedback_text: str,
    a2_extracted: str,
    ground_truth: str,
    answer_type: str,
    a1: str,
    a1_is_correct: bool,
    tolerance: float = 0.01,
) -> float:
    """R_downstream: Judge F1 quality by the quality of A2 it produces.

    The critic is rewarded based on whether its feedback leads to a
    correct refined answer. Uses answer_type-aware values.

    For deterministic types (MCQ, YesNo, Numeric):
        RR: +1.0, RW: -1.5, WR: +3.0, WW: -1.0
        Rationale: small answer space makes correction feasible, so
        we strongly reward corrective feedback (WR=+3.0) and reduce
        the RW penalty to avoid training a "never criticize" policy.

    For open-ended / counting:
        RR: +1.0, RW: -2.0, WR: +2.0, WW: -1.0
        Rationale: large answer space makes correction harder, so
        we keep the heavier RW penalty to protect correct answers.

    Returns 0.0 if feedback is empty.

    Args:
        feedback_text: Extracted feedback text
        a2_extracted: Normalized extracted A2 answer
        ground_truth: Ground truth answer
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        a1: Initial answer text
        a1_is_correct: Whether A1 was correct
        tolerance: Numeric tolerance for comparison

    Returns:
        Downstream-aware reward value
    """
    from vlm_grpo.rewards.verifier import DETERMINISTIC_TYPES

    if not feedback_text.strip():
        return 0.0

    result = verify_answer(a2_extracted, ground_truth, answer_type, tolerance=tolerance)
    a2_correct = result.is_correct

    if answer_type in DETERMINISTIC_TYPES:
        if a1_is_correct:
            return 1.0 if a2_correct else -1.5
        return 3.0 if a2_correct else -1.0

    if a1_is_correct:
        if a2_correct:
            return 1.0  # RR: good feedback maintained correctness
        return -2.0  # RW: bad feedback caused regression
    if a2_correct:
        return 2.0  # WR: feedback fixed the error (2:1 ratio with RR)
    return -1.0  # WW: feedback failed to help
