#!/usr/bin/env python3
"""
Evaluation utilities for RR/RW/WR/WW transition computation.

Computes the confusion matrix of answer transitions between Answer1
and Answer2 to measure self-reflection quality.

Since the RW-first dataset only contains Answer1-correct samples,
only RR (maintained) and RW (regressed) transitions are expected.

Usage:
    from vlm_grpo.eval import compute_transition_metrics

    metrics = compute_transition_metrics(
        completions=["FEEDBACK:\\nOK\\nFINAL_ANSWER:\\nA"],
        ground_truths=["A"],
        answer1s=["A"],
        answer_types=["mcq"],
        choices_list=[""],
    )
    print(f"RW rate: {metrics.rw_rate:.2%}")
"""

import logging
import sys
from dataclasses import asdict, dataclass

from vlm_grpo.rewards.deterministic import match_answer
from vlm_grpo.trajectory import extract_answer_from_text, extract_completion_text, parse_trajectory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class TransitionMetrics:
    """Metrics for answer transition patterns.

    RR = Right-Right (Answer1 correct, Answer2 correct)
    RW = Right-Wrong (Answer1 correct, Answer2 wrong) -- target to minimize
    WR = Wrong-Right (Answer1 wrong, Answer2 correct)
    WW = Wrong-Wrong (Answer1 wrong, Answer2 wrong)

    For Answer1-correct datasets, only RR and RW are possible.

    Attributes:
        total_samples: Total number of evaluated samples
        rr_count: Right-Right count
        rw_count: Right-Wrong count
        wr_count: Wrong-Right count
        ww_count: Wrong-Wrong count
        rr_rate: RR rate (fraction)
        rw_rate: RW rate (fraction) -- primary metric to minimize
        wr_rate: WR rate (fraction)
        ww_rate: WW rate (fraction)
        format_invalid_count: Samples where format parsing failed
        parse_failure_count: Samples where FEEDBACK/FINAL_ANSWER markers missing
        undetermined_count: Samples where correctness couldn't be determined
    """

    total_samples: int
    rr_count: int
    rw_count: int
    wr_count: int
    ww_count: int
    rr_rate: float
    rw_rate: float
    wr_rate: float
    ww_rate: float
    format_invalid_count: int
    parse_failure_count: int
    undetermined_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def compute_transition_metrics(
    completions: list[str],
    ground_truths: list[str],
    answer1s: list[str],
    answer_types: list[str],
    choices_list: list[str],
) -> TransitionMetrics:
    """Compute RR/RW/WR/WW metrics on a batch of completions.

    Args:
        completions: Generated completion texts (raw strings)
        ground_truths: Ground truth answers
        answer1s: Precomputed Answer1s
        answer_types: Answer type classifications
        choices_list: MCQ choices per sample

    Returns:
        TransitionMetrics with counts and rates
    """
    rr = 0
    rw = 0
    wr = 0
    ww = 0
    fmt_invalid = 0
    parse_fail = 0
    undetermined = 0

    total = len(completions)

    for i in range(total):
        comp_text = extract_completion_text(completions[i])
        gt = ground_truths[i]
        a1 = answer1s[i]
        a_type = answer_types[i]
        ch = choices_list[i]

        # Parse trajectory
        trajectory = parse_trajectory(comp_text)
        if not trajectory.parse_success:
            parse_fail += 1
            continue

        # Extract Answer2
        a2 = extract_answer_from_text(trajectory.final_answer, a_type, ch)
        if not a2:
            fmt_invalid += 1
            continue

        # Check Answer1 correctness
        a1_correct = match_answer(
            extract_answer_from_text(a1, a_type, ch) or a1,
            gt,
            a_type,
        )

        # Check Answer2 correctness
        a2_correct = match_answer(a2, gt, a_type)

        if a1_correct is None or a2_correct is None:
            undetermined += 1
            continue

        if a1_correct and a2_correct:
            rr += 1
        elif a1_correct and not a2_correct:
            rw += 1
        elif not a1_correct and a2_correct:
            wr += 1
        else:
            ww += 1

    determined = rr + rw + wr + ww
    denom = max(determined, 1)

    return TransitionMetrics(
        total_samples=total,
        rr_count=rr,
        rw_count=rw,
        wr_count=wr,
        ww_count=ww,
        rr_rate=rr / denom,
        rw_rate=rw / denom,
        wr_rate=wr / denom,
        ww_rate=ww / denom,
        format_invalid_count=fmt_invalid,
        parse_failure_count=parse_fail,
        undetermined_count=undetermined,
    )


def log_transition_metrics(
    metrics: TransitionMetrics,
    step: int,
    prefix: str = "val",
) -> dict[str, float]:
    """Format transition metrics for WandB/TensorBoard logging.

    Args:
        metrics: Computed transition metrics
        step: Current training step
        prefix: Metric prefix ("val" or "train")

    Returns:
        Dict of metric name -> value for logging
    """
    log_dict = {
        f"{prefix}/rr_rate": metrics.rr_rate,
        f"{prefix}/rw_rate": metrics.rw_rate,
        f"{prefix}/wr_rate": metrics.wr_rate,
        f"{prefix}/ww_rate": metrics.ww_rate,
        f"{prefix}/format_invalid_rate": metrics.format_invalid_count / max(metrics.total_samples, 1),
        f"{prefix}/parse_failure_rate": metrics.parse_failure_count / max(metrics.total_samples, 1),
    }

    logger.info(
        f"[Step {step}] {prefix} transitions: "
        f"RR={metrics.rr_rate:.2%} RW={metrics.rw_rate:.2%} "
        f"WR={metrics.wr_rate:.2%} WW={metrics.ww_rate:.2%} "
        f"(fmt_invalid={metrics.format_invalid_count}, parse_fail={metrics.parse_failure_count})"
    )

    return log_dict
