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

    r = compute_a2_correctness_reward("A", "A", "mcq")
    assert r == 1.0
"""

from vlm_grpo.rewards.verifier import verify_answer


def compute_a2_correctness_reward(
    a2_extracted: str,
    ground_truth: str,
    answer_type: str,
    tolerance: float = 0.01,
) -> float:
    """R_a2_correct: Check if A2 matches ground truth.

    For deterministic types (MCQ, YesNo, Numeric): binary +1.0/-1.0.
    For counting: continuous reward based on fuzzy score (CrowdVLM-R1,
    arXiv:2504.03724). Predicting 5 when GT is 6 gets partial credit.
    For open-ended: continuous reward from LLM judge or similarity score
    (RARL arXiv:2506.06600, VisionThink arXiv:2507.13348).

    NOT gated by format: correctness is evaluated independently.
    Format compliance is a separate reward component.

    Args:
        a2_extracted: Normalized extracted A2 answer
        ground_truth: Ground truth answer
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        tolerance: Numeric tolerance for comparison

    Returns:
        Reward in [-1.0, 1.0]. Continuous for counting/open, binary for others.
    """
    result = verify_answer(a2_extracted, ground_truth, answer_type, tolerance=tolerance)

    # Counting and open-ended: use continuous score when available.
    # Maps score [0, 1] to reward [-1, +1] via: reward = 2*score - 1.
    # score=1.0 → +1.0, score=0.5 → 0.0, score=0.0 → -1.0.
    if answer_type in ("counting", "open") and result.score is not None:
        return 2.0 * result.score - 1.0

    # Deterministic types (MCQ, YesNo, Numeric): binary
    if result.is_correct:
        return 1.0
    return -1.0


def compute_a1_correctness_01(
    a1_text: str,
    ground_truth: str,
    answer_type: str,
    tolerance: float = 0.01,
) -> float:
    """R_a1_correct_01: binary {0.0, 1.0} A1 correctness for the single-turn baseline.

    Used by the ``single_turn_a1`` GRPO baseline (config.RolloutConfig.single_turn_a1)
    where the reward range is constrained to [0, 1] to keep all components on a
    common positive scale. Returns 1.0 when ``verify_answer`` says A1 matches the
    ground truth, else 0.0. NOT gated by format — format compliance is a separate
    component (see ``compute_a1_format_01`` in rewards.composition).

    Counting and open-ended types still bottom out to a binary verdict here
    rather than the continuous score path used by ``compute_a2_correctness_reward``,
    so the baseline reward stays strictly in {0, 1}.

    Args:
        a1_text: Raw A1 text (extraction is delegated to verify_answer).
        ground_truth: Ground truth answer.
        answer_type: Answer type ("mcq", "yesno", "numeric", "counting", "open").
        tolerance: Numeric tolerance for comparison.

    Returns:
        1.0 if A1 is correct, 0.0 otherwise.
    """
    result = verify_answer(a1_text, ground_truth, answer_type, tolerance=tolerance)
    return 1.0 if result.is_correct else 0.0


def compute_downstream_improvement_reward(
    a1: str,
    a2_extracted: str,
    ground_truth: str,
    answer_type: str,
    a1_is_correct: bool,
    tolerance: float = 0.01,
) -> float:
    """R_downstream_improvement: Reward based on A1→A2 correctness transition.

    Captures the four possible transitions:
    - RR (Right→Right): A1 correct, A2 correct → +1.0 (maintained)
    - RW (Right→Wrong): A1 correct, A2 wrong   → -2.0 (regression)
    - WR (Wrong→Right): A1 wrong, A2 correct    → +2.0 (fixed it)
    - WW (Wrong→Wrong): A1 wrong, A2 wrong      → -0.5 (failed to fix)

    NOT gated by format: correctness is evaluated independently.

    Args:
        a1: Initial answer text
        a2_extracted: Normalized extracted A2 answer
        ground_truth: Ground truth answer
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        a1_is_correct: Whether A1 was correct
        tolerance: Numeric tolerance for comparison

    Returns:
        Transition reward value
    """
    result = verify_answer(a2_extracted, ground_truth, answer_type, tolerance=tolerance)
    a2_correct = result.is_correct

    if a1_is_correct:
        if a2_correct:
            return 1.0  # RR: maintained correctness
        return -2.0  # RW: regression
    if a2_correct:
        return 2.0  # WR: fixed the error
    return -0.5  # WW: failed to fix
