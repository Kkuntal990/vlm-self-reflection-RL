#!/usr/bin/env python3
"""Tests for the single-turn A1-only GRPO baseline reward path.

Covers:
  - compute_a1_correctness_01 returns binary {0.0, 1.0}
  - compute_a1_format_01 enforces both <think> and <answer> tags + clean inner
  - compute_baseline_a1_reward_breakdown applies 0.9 / 0.1 weights correctly
  - RolloutConfig.single_turn_a1 flag plumbing
  - generate_baseline_a1_rollout fills feedback_breakdowns with zero stubs

Integration with the trainer / vLLM is exercised on the cluster, not here.
"""

from unittest.mock import patch

import pytest

from vlm_grpo.config import BaselineA1RewardWeights, RolloutConfig
from vlm_grpo.rewards.composition import (
    BaselineA1RewardBreakdown,
    compute_a1_format_01,
    compute_baseline_a1_reward_breakdown,
)
from vlm_grpo.rewards.correctness import compute_a1_correctness_01
from vlm_grpo.rollout import (
    SelfReflectionRolloutResult,
    generate_baseline_a1_rollout,
)

# =============================================================================
# compute_a1_correctness_01
# =============================================================================


class TestA1Correctness01:
    """Binary {0.0, 1.0} A1 correctness reward."""

    def test_mcq_correct(self) -> None:
        assert compute_a1_correctness_01("A", "A", "mcq") == 1.0

    def test_mcq_incorrect(self) -> None:
        assert compute_a1_correctness_01("B", "A", "mcq") == 0.0

    def test_mcq_with_tags_correct(self) -> None:
        text = "<think>The picture shows option A.</think><answer>(A)</answer>"
        assert compute_a1_correctness_01(text, "A", "mcq") == 1.0

    def test_yesno_correct(self) -> None:
        assert compute_a1_correctness_01("Yes", "yes", "yesno") == 1.0

    def test_yesno_incorrect(self) -> None:
        assert compute_a1_correctness_01("No", "yes", "yesno") == 0.0

    def test_numeric_correct(self) -> None:
        assert compute_a1_correctness_01("3.14", "3.14", "numeric") == 1.0

    def test_numeric_incorrect(self) -> None:
        assert compute_a1_correctness_01("4.00", "3.14", "numeric") == 0.0

    def test_no_extractable_answer(self) -> None:
        # Pure prose with no MCQ letter and no tags → wrong.
        assert compute_a1_correctness_01("I don't know", "A", "mcq") == 0.0

    def test_returns_strict_zero_or_one(self) -> None:
        # Counting / open paths must NOT bleed continuous scores into the
        # baseline reward — the baseline contract is strictly {0, 1}.
        for ans, gt, at in [
            ("3", "5", "counting"),  # close-but-wrong; no partial credit here
            ("hello world", "hello world", "open"),  # exact match
        ]:
            r = compute_a1_correctness_01(ans, gt, at)
            assert r in (0.0, 1.0), f"got {r} for ({ans!r}, {gt!r}, {at})"


# =============================================================================
# compute_a1_format_01
# =============================================================================


class TestA1Format01:
    """Binary {0.0, 1.0} A1 format reward — strict tag + clean inner."""

    def test_valid_mcq_with_parens(self) -> None:
        text = "<think>It looks like option A.</think><answer>(A)</answer>"
        assert compute_a1_format_01(text, "mcq") == 1.0

    def test_valid_mcq_bare_letter(self) -> None:
        text = "<think>...</think><answer>A</answer>"
        assert compute_a1_format_01(text, "mcq") == 1.0

    def test_no_tags(self) -> None:
        assert compute_a1_format_01("(A)", "mcq") == 0.0

    def test_only_answer_tag(self) -> None:
        # Both <think> AND <answer> are required.
        assert compute_a1_format_01("<answer>(A)</answer>", "mcq") == 0.0

    def test_only_think_tag(self) -> None:
        assert compute_a1_format_01("<think>(A)</think>", "mcq") == 0.0

    def test_empty_inner(self) -> None:
        text = "<think>...</think><answer></answer>"
        assert compute_a1_format_01(text, "mcq") == 0.0

    def test_dirty_inner_mcq(self) -> None:
        # MCQ format is strict: descriptor text inside <answer> is rejected.
        text = "<think>...</think><answer>(A) the first option</answer>"
        assert compute_a1_format_01(text, "mcq") == 0.0

    def test_yesno_valid(self) -> None:
        text = "<think>...</think><answer>yes</answer>"
        assert compute_a1_format_01(text, "yesno") == 1.0

    def test_numeric_valid(self) -> None:
        text = "<think>...</think><answer>42</answer>"
        assert compute_a1_format_01(text, "numeric") == 1.0

    def test_open_any_nonempty(self) -> None:
        text = "<think>...</think><answer>blue</answer>"
        assert compute_a1_format_01(text, "open") == 1.0


# =============================================================================
# compute_baseline_a1_reward_breakdown — combined 0.9 + 0.1 weighting
# =============================================================================


class TestBaselineA1RewardBreakdown:
    """End-to-end 0.9 * correctness + 0.1 * format reward composition."""

    @staticmethod
    def _w() -> BaselineA1RewardWeights:
        return BaselineA1RewardWeights(w_a1_correctness=0.9, w_a1_format=0.1)

    def test_correct_and_formatted_is_one(self) -> None:
        text = "<think>I see option A.</think><answer>(A)</answer>"
        bd = compute_baseline_a1_reward_breakdown(text, "A", "mcq", "", self._w())
        assert isinstance(bd, BaselineA1RewardBreakdown)
        assert bd.total_reward == pytest.approx(1.0, abs=1e-9)
        assert bd.components == {"a1_correctness": 1.0, "a1_format": 1.0}
        assert bd.a1_correct is True
        assert bd.a1_format_valid is True
        assert bd.a1_extracted == "A"

    def test_wrong_but_formatted_is_zero_point_one(self) -> None:
        text = "<think>I see option A.</think><answer>(B)</answer>"
        bd = compute_baseline_a1_reward_breakdown(text, "A", "mcq", "", self._w())
        assert bd.total_reward == pytest.approx(0.1, abs=1e-9)
        assert bd.components == {"a1_correctness": 0.0, "a1_format": 1.0}
        assert bd.a1_correct is False
        assert bd.a1_format_valid is True

    def test_correct_but_no_tags_is_zero_point_nine(self) -> None:
        # Strict format extraction requires <answer> tag, so a1_extracted is
        # empty here — but a1_correct uses verify_answer (liberal extraction)
        # so prose-stuffed correct answers still credit the correctness path.
        bd = compute_baseline_a1_reward_breakdown("(A)", "A", "mcq", "", self._w())
        assert bd.total_reward == pytest.approx(0.9, abs=1e-9)
        assert bd.components == {"a1_correctness": 1.0, "a1_format": 0.0}
        assert bd.a1_correct is True
        assert bd.a1_format_valid is False
        assert bd.a1_extracted == ""  # strict extraction yields nothing without tags

    def test_wrong_and_no_tags_is_zero(self) -> None:
        bd = compute_baseline_a1_reward_breakdown("(B)", "A", "mcq", "", self._w())
        assert bd.total_reward == pytest.approx(0.0, abs=1e-9)
        assert bd.components == {"a1_correctness": 0.0, "a1_format": 0.0}
        assert bd.a1_correct is False
        assert bd.a1_format_valid is False

    def test_formatted_but_dirty_inner_is_zero(self) -> None:
        # `<answer>(B) descriptor</answer>` fails the clean-atomic format
        # check (returns 0.0) AND verifies as wrong (correctness 0.0 since
        # GT is A). Under strict extraction the format-rejecting inner
        # produces a1_extracted="" so liberal correctness sees only the
        # surrounding prose; either way, total reward is 0.0.
        text = "<think>...</think><answer>(B) descriptor</answer>"
        bd = compute_baseline_a1_reward_breakdown(text, "A", "mcq", "", self._w())
        assert bd.total_reward == pytest.approx(0.0, abs=1e-9)

    def test_total_reward_in_unit_interval(self) -> None:
        # Both components live in [0, 1] and the convex combination must too.
        weights = self._w()
        for text, gt in [
            ("<think>x</think><answer>(A)</answer>", "A"),
            ("<think>x</think><answer>(A)</answer>", "B"),
            ("(A)", "A"),
            ("(A)", "B"),
            ("garbage", "A"),
        ]:
            bd = compute_baseline_a1_reward_breakdown(text, gt, "mcq", "", weights)
            assert 0.0 <= bd.total_reward <= 1.0


# =============================================================================
# RolloutConfig wiring
# =============================================================================


class TestRolloutConfigSingleTurn:
    def test_default_off(self) -> None:
        cfg = RolloutConfig()
        assert cfg.single_turn_a1 is False

    def test_explicit_on(self) -> None:
        cfg = RolloutConfig(single_turn_a1=True)
        assert cfg.single_turn_a1 is True


# =============================================================================
# generate_baseline_a1_rollout — only A1 generated, F1/A2 stubbed out
# =============================================================================


class _StubVLLM:
    """Records prompts that the rollout sends to vLLM and returns canned A1
    completions in order. Lets the test assert that no F1 / A2 calls happen.
    """

    def __init__(self, completions: list[str]) -> None:
        self._queue = list(completions)
        self.calls = 0

    def generate_batch(
        self,
        prompts,  # type: ignore[no-untyped-def]
        images,
        max_new_tokens: int,
        temperature: float,
        top_p: float = 0.9,
    ) -> list[str]:
        self.calls += 1
        out = self._queue[: len(prompts)]
        self._queue = self._queue[len(prompts) :]
        return out


class TestBaselineA1Rollout:
    def test_skips_f1_and_a2(self) -> None:
        samples = [
            {
                "question": "Which option is correct?",
                "image": None,
                "ground_truth": "A",
                "answer_type": "mcq",
                "choices": "",
                "dataset_name": "test",
            }
        ]
        config = RolloutConfig(
            k_samples=2,
            batch_size=1,
            single_turn_a1=True,
            use_think_answer_tags=True,
            a1_max_completion_length=64,
            temperature=1.0,
            top_p=0.9,
        )
        weights = BaselineA1RewardWeights(w_a1_correctness=0.9, w_a1_format=0.1)
        a1_outputs = [
            "<think>It is A.</think><answer>(A)</answer>",
            "<think>It is B.</think><answer>(B)</answer>",
        ]
        stub = _StubVLLM(a1_outputs)

        # apply_chat_template is exercised by VLLMRolloutEngine.generate_batch in
        # production; the stub takes pre-formatted prompts directly so we just
        # need a no-op processor for the chat-template call inside the rollout.
        class _StubProcessor:
            def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):  # type: ignore[no-untyped-def]
                return "PROMPT"

        results = generate_baseline_a1_rollout(
            model=None,
            processor=_StubProcessor(),
            samples=samples,
            config=config,
            baseline_weights=weights,
            device="cpu",
            model_type="qwen2vl",
            vllm_engine=stub,
        )

        assert len(results) == 1
        r: SelfReflectionRolloutResult = results[0]

        # Exactly K A1 completions, K empty F1/A2 placeholders.
        assert r.answer1s == a1_outputs
        assert r.feedbacks == ["", ""]
        assert r.answer2s == ["", ""]

        # Reward shape: [0, 1] per trajectory, derived from the new 2-component breakdown.
        assert len(r.response_breakdowns) == 2
        assert all(isinstance(bd, BaselineA1RewardBreakdown) for bd in r.response_breakdowns)
        # First is correct and formatted; second is wrong but formatted.
        assert r.response_rewards[0] == pytest.approx(1.0)
        assert r.response_rewards[1] == pytest.approx(0.1)

        # Feedback head: stubbed to zero so the trainer's two-reward plumbing
        # (zero-variance group filter, advantage compute) stays well-defined
        # but contributes no gradient.
        assert r.feedback_rewards == [0.0, 0.0]
        for fb in r.feedback_breakdowns:
            assert fb.total_reward == 0.0

        # Exactly one vLLM call (just the A1 step) — no F1 and no A2 generation.
        assert stub.calls == 1

    def test_requires_baseline_weights(self) -> None:
        config = RolloutConfig(k_samples=1, batch_size=1, single_turn_a1=True)
        with pytest.raises(ValueError, match="baseline_weights"):
            generate_baseline_a1_rollout(
                model=None,
                processor=None,
                samples=[],
                config=config,
                baseline_weights=None,
            )


# =============================================================================
# generate_self_reflection_rollout dispatches to baseline path on the flag
# =============================================================================


class TestRolloutDispatch:
    def test_dispatch_to_baseline_when_flag_set(self) -> None:
        config = RolloutConfig(k_samples=1, batch_size=1, single_turn_a1=True)
        weights = BaselineA1RewardWeights()

        with patch("vlm_grpo.rollout.generate_baseline_a1_rollout") as mock_baseline:
            mock_baseline.return_value = []
            from vlm_grpo.rollout import generate_self_reflection_rollout

            generate_self_reflection_rollout(
                model=None,
                processor=None,
                samples=[],
                config=config,
                response_weights=None,
                feedback_weights=None,
                baseline_weights=weights,
            )

            mock_baseline.assert_called_once()
            assert mock_baseline.call_args.kwargs["baseline_weights"] is weights
