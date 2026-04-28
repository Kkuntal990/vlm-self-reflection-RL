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
        RR: +1.0, RW: -2.0, WR: +2.35, WW: -0.5
        WR=+2.35 is the exact compensation that ties response-head RR
        and WR after combining with the a1_correctness term
        (0.27·R_a1 contributes ±0.27, requiring R_noreg(WR)−R_noreg(RR)
        = 0.54/0.40 = 1.35). The tie removes the WR>RR gradient bias
        that drove A1 sandbagging in v10-fixes-v2 — the model used to
        prefer "wrong A1 → corrected" over "right A1 → kept right"
        because the former earned +1.16 more in combined reward.
        WW=-0.5 keeps the small negative anchor that breaks dead
        all-WW K-groups (zero-gradient failure mode in v10-base).

    For open-ended / counting:
        RR: +1.0, RW: -3.0, WR: +2.35, WW: -0.5
        Same WR=+2.35 compensation; RW kept at -3.0 since changing a
        correct freeform answer is expensive (large answer space makes
        re-generating a correct answer hard).

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
        return 2.35 if a2_correct else -0.5

    if a1_is_correct:
        if a2_correct:
            return 1.0
        return -3.0
    if a2_correct:
        return 2.35
    return -0.5
