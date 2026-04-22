#!/usr/bin/env python3
"""
Feedback reward components for two-trajectory GRPO critic training.

Computes the downstream-aware reward that judges F1 quality by the
correctness of the A2 it produces.

Usage:
    from vlm_grpo.rewards.feedback import compute_downstream_aware_reward

    r = compute_downstream_aware_reward(
        feedback_text="INCORRECT. Should be (B).",
        a2_extracted="B",
        ground_truth="B",
        answer_type="mcq",
        a1="A",
        a1_is_correct=False,
    )
"""

from vlm_grpo.rewards.verifier import verify_answer


def compute_downstream_aware_reward(
    feedback_text: str,
    a2_extracted: str,
    ground_truth: str,
    answer_type: str,
    a1: str,
    a1_is_correct: bool,
    tolerance: float = 0.01,
    use_improvement_reward: bool = False,
) -> float:
    """R_downstream: Judge F1 quality by the quality of A2 it produces.

    Two modes controlled by use_improvement_reward:

    **Transition mode** (default, use_improvement_reward=False):
        For deterministic types (MCQ, YesNo, Numeric):
            RR: +1.0, RW: -1.5, WR: +3.0, WW: -1.0
        For open-ended / counting:
            RR: +1.0, RW: -2.0, WR: +2.0, WW: -1.0

    **Improvement mode** (use_improvement_reward=True):
        R_improve = correctness(A2) - correctness(A1)
            WR: +2.0, RW: -2.0, RR: 0.0, WW: 0.0
        Rationale (Critique-GRPO, arXiv:2506.03106): RR and WW both
        give 0, so the K-group mean converges to 0. WR(+2) and RW(-2)
        dominate the advantage.

    Returns 0.0 if feedback is empty.

    Args:
        feedback_text: Feedback text from the critic
        a2_extracted: Normalized extracted A2 answer
        ground_truth: Ground truth answer
        answer_type: Answer type ("mcq", "yesno", "numeric", "open")
        a1: Initial answer text
        a1_is_correct: Whether A1 was correct
        tolerance: Numeric tolerance for comparison
        use_improvement_reward: If True, use R(A2)-R(A1) instead of
            transition-shaped constants

    Returns:
        Downstream-aware reward value
    """
    from vlm_grpo.rewards.verifier import DETERMINISTIC_TYPES

    if not feedback_text.strip():
        return 0.0

    result = verify_answer(a2_extracted, ground_truth, answer_type, tolerance=tolerance)
    a2_correct = result.is_correct

    if use_improvement_reward:
        r_a1 = 1.0 if a1_is_correct else -1.0
        r_a2 = 1.0 if a2_correct else -1.0
        return r_a2 - r_a1

    if answer_type in DETERMINISTIC_TYPES:
        if a1_is_correct:
            return 1.0 if a2_correct else -1.5
        return 3.0 if a2_correct else -1.0

    if a1_is_correct:
        if a2_correct:
            return 1.0
        return -2.0
    if a2_correct:
        return 2.0
    return -1.0
