#!/usr/bin/env python3
"""
Correctness reward components for two-trajectory GRPO training.

Computes rewards based on whether A2 (refined answer) matches ground truth,
and whether the A1→A2 transition preserves or improves correctness.

Uses verify_answer() which returns only CORRECT or WRONG (no undetermined).
For deterministic types, parse failure is WRONG. For open-ended, a multi-stage
cascade with embedding similarity determines correctness.

Usage:
    from vlm_grpo.rewards.correctness import (
        compute_a2_correctness_reward,
        compute_downstream_improvement_reward,
    )

    r = compute_a2_correctness_reward("A", "A", "mcq", format_valid=True)
    assert r == 1.0
"""

from vlm_grpo.rewards.verifier import verify_answer


def compute_a2_correctness_reward(
    a2_extracted: str,
    ground_truth: str,
    answer_type: str,
    format_valid: bool,
    tolerance: float = 0.01,
) -> float:
    """R_a2_correct: Check if A2 matches ground truth.

    Gated by format validity: returns 0.0 if format is invalid.

    Args:
        a2_extracted: Normalized extracted A2 answer
        ground_truth: Ground truth answer
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        format_valid: Whether the completion format passed validation
        tolerance: Numeric tolerance for comparison

    Returns:
        +1.0 correct, -1.0 incorrect
    """
    if not format_valid:
        return 0.0

    result = verify_answer(a2_extracted, ground_truth, answer_type, tolerance=tolerance)

    if result.is_correct:
        return 1.0
    return -1.0


def compute_downstream_improvement_reward(
    a1: str,
    a2_extracted: str,
    ground_truth: str,
    answer_type: str,
    a1_is_correct: bool,
    format_valid: bool,
    tolerance: float = 0.01,
) -> float:
    """R_downstream_improvement: Reward based on A1→A2 correctness transition.

    Captures the four possible transitions:
    - RR (Right→Right): A1 correct, A2 correct → +1.0 (maintained)
    - RW (Right→Wrong): A1 correct, A2 wrong   → -2.0 (regression)
    - WR (Wrong→Right): A1 wrong, A2 correct    → +2.0 (fixed it)
    - WW (Wrong→Wrong): A1 wrong, A2 wrong      → -0.5 (failed to fix)

    Gated by format validity: returns 0.0 if format is invalid.

    Args:
        a1: Initial answer text
        a2_extracted: Normalized extracted A2 answer
        ground_truth: Ground truth answer
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        a1_is_correct: Whether A1 was correct
        format_valid: Whether the completion format passed validation
        tolerance: Numeric tolerance for comparison

    Returns:
        Transition reward value
    """
    if not format_valid:
        return 0.0

    result = verify_answer(a2_extracted, ground_truth, answer_type, tolerance=tolerance)
    a2_correct = result.is_correct

    if a1_is_correct:
        if a2_correct:
            return 1.0  # RR: maintained correctness
        return -2.0  # RW: regression
    if a2_correct:
        return 2.0  # WR: fixed the error
    return -0.5  # WW: failed to fix
