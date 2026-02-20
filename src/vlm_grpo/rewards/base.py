#!/usr/bin/env python3
"""
Base types for reward functions.

Provides the RewardBreakdown dataclass for detailed per-sample logging
of all reward components.

Usage:
    from vlm_grpo.rewards.base import RewardBreakdown

    breakdown = RewardBreakdown(
        total_reward=2.5,
        components={"format": 1.0, "final_correct": 1.0, "no_regression": 1.0},
        weighted_components={"format": 0.5, "final_correct": 1.0, "no_regression": 2.0},
        format_valid=True,
        parse_success=True,
        final_answer_extracted="A",
    )
"""

from dataclasses import asdict, dataclass


@dataclass
class RewardBreakdown:
    """Detailed breakdown of reward components for a single sample.

    Attributes:
        total_reward: Weighted sum of all components
        components: Dict mapping component name to raw reward value
        weighted_components: Dict mapping component name to weighted value
        format_valid: Whether the completion format was valid
        parse_success: Whether FEEDBACK/FINAL_ANSWER markers were parsed
        final_answer_extracted: The extracted and normalized final answer
    """

    total_reward: float
    components: dict[str, float]
    weighted_components: dict[str, float]
    format_valid: bool
    parse_success: bool
    final_answer_extracted: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
