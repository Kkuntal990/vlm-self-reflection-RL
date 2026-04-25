#!/usr/bin/env python3
"""
Stability reward components for two-trajectory GRPO refiner training.

Computes the no-regression reward that penalizes RW flips and rewards
WR fixes.

Usage:
    from vlm_grpo.rewards.stability import compute_no_regression_reward

    r = compute_no_regression_reward("A", "A", "mcq", True)
    assert r == 1.0
"""

from vlm_grpo.rewards.verifier import DETERMINISTIC_TYPES, verify_answer


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
        RR: +1.0, RW: -2.0, WR: +3.0, WW: -0.5
        Rationale: changing a letter/word is cheap and the answer space
        is small, so we reduce the RW penalty and increase WR reward
        to encourage the model to attempt corrections via feedback.
        WW carries a small negative to discourage "stable wrong" — a
        K-group of all-WW now produces a non-zero gradient signal via
        this penalty, attacking the dead-K-group failure mode observed
        in v10-base training (52% WW, 0 gradient).

    For open-ended / counting:
        RR: +1.0, RW: -3.0, WR: +2.0, WW: -0.5
        Rationale: changing a correct freeform answer is expensive
        (large answer space makes re-generating a correct answer hard),
        so we keep the heavy RW penalty to protect correct answers.
        Same WW=-0.5 to maintain consistency with the deterministic case.

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

    if answer_type in DETERMINISTIC_TYPES:
        if a1_is_correct:
            return 1.0 if a2_correct else -2.0
        return 3.0 if a2_correct else -0.5

    if a1_is_correct:
        if a2_correct:
            return 1.0
        return -3.0
    if a2_correct:
        return 2.0
    return -0.5
