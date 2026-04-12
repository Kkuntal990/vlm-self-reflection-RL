#!/usr/bin/env python3
"""
Stability reward components for two-trajectory GRPO refiner training.

Computes rewards for maintaining correctness (no-regression) and
encouraging minimal edits when the initial answer is already correct.

Uses verify_answer() which returns only CORRECT or WRONG (no undetermined).
For deterministic types, edit distance is computed on extracted answer tokens.

Usage:
    from vlm_grpo.rewards.stability import (
        compute_no_regression_reward,
        compute_minimal_edit_reward,
    )

    r = compute_no_regression_reward("A", "A", "mcq", True, True)
    assert r == 1.0
"""

from vlm_grpo.rewards.verifier import DETERMINISTIC_TYPES, verify_answer
from vlm_grpo.utils import normalized_edit_distance


def compute_no_regression_reward(
    a2_extracted: str,
    ground_truth: str,
    answer_type: str,
    a1_is_correct: bool,
    tolerance: float = 0.01,
) -> float:
    """R_no_regression: Penalize RW flips, reward WR fixes.

    Uses answer_type-aware values:

    For deterministic types (MCQ, YesNo, Numeric):
        RR: +1.0, RW: -2.0, WR: +3.0, WW: 0.0
        Rationale: changing a letter/word is cheap and the answer space
        is small, so we reduce the RW penalty and increase WR reward
        to encourage the model to attempt corrections via feedback.

    For open-ended / counting:
        RR: +1.0, RW: -3.0, WR: +2.0, WW: 0.0
        Rationale: changing a correct freeform answer is expensive
        (large answer space makes re-generating a correct answer hard),
        so we keep the heavy RW penalty to protect correct answers.

    NOT gated by format: correctness is evaluated independently.

    Args:
        a2_extracted: Normalized extracted A2 answer
        ground_truth: Ground truth answer
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        a1_is_correct: Whether A1 was correct
        tolerance: Numeric tolerance for comparison

    Returns:
        Regression reward value
    """
    result = verify_answer(a2_extracted, ground_truth, answer_type, tolerance=tolerance)
    a2_correct = result.is_correct

    # Deterministic types: encourage correction (lower RW penalty, higher WR reward)
    if answer_type in DETERMINISTIC_TYPES:
        if a1_is_correct:
            return 1.0 if a2_correct else -2.0
        return 3.0 if a2_correct else 0.0

    # Open-ended / counting: protect correct answers (heavy RW penalty)
    if a1_is_correct:
        if a2_correct:
            return 1.0  # RR: maintained correctness
        return -3.0  # RW: regression (heavy penalty)
    if a2_correct:
        return 2.0  # WR: fixed the error
    return 0.0  # WW: neutral


def compute_minimal_edit_reward(
    a1: str,
    a2_extracted: str,
    ground_truth: str,
    answer_type: str,
    lambda_edit: float = 0.5,
    tolerance: float = 0.01,
) -> float:
    """R_minimal_edit: Reward minimal changes when A1 is already correct.

    For deterministic types (MCQ/YesNo/Numeric), edit distance is computed
    on the extracted/normalized token strings (e.g. "A" vs "A").
    For open types, edit distance is computed on raw stripped text.

    NOT gated by format: correctness is evaluated independently.

    Args:
        a1: Initial answer text
        a2_extracted: Normalized extracted A2 answer
        ground_truth: Ground truth answer
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        lambda_edit: Scaling factor for edit distance penalty
        tolerance: Numeric tolerance for comparison

    Returns:
        [0.0, 1.0] where 1.0 means identical answers, 0.0 if not applicable
    """
    a1_result = verify_answer(a1, ground_truth, answer_type, tolerance=tolerance)
    if not a1_result.is_correct:
        return 0.0

    a2_result = verify_answer(a2_extracted, ground_truth, answer_type, tolerance=tolerance)
    if not a2_result.is_correct:
        return 0.0

    # Both correct: compute edit distance on appropriate representation
    if answer_type in DETERMINISTIC_TYPES:
        a1_cmp = a1_result.extracted
        a2_cmp = a2_result.extracted
    else:
        a1_cmp = a1.strip().lower()
        a2_cmp = a2_extracted.strip().lower()

    edit_dist = normalized_edit_distance(a1_cmp, a2_cmp)
    reward = 1.0 - lambda_edit * edit_dist

    return max(reward, 0.0)
