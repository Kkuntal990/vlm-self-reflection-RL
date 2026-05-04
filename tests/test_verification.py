"""Tests for F1 verification reward + Pattern-A prompt builders (v10+)."""

import pytest

from vlm_grpo.config import FeedbackRewardWeights
from vlm_grpo.prompts import (
    ANSWER_TAG_INSTRUCTION,
    F1_VERIFIER_INSTRUCTION,
    THINK_ANSWER_INSTRUCTION,
    build_critic_prompt,
    build_initial_answer_prompt,
    build_prompt_with_completion,
    build_refiner_prompt,
)
from vlm_grpo.rewards.composition import (
    compute_feedback_format_reward,
    compute_feedback_reward_breakdown,
    compute_verification_accuracy_reward,
)
from vlm_grpo.trajectory import extract_from_boxed, has_think_boxed

# =============================================================================
# Verification accuracy reward (keyword-based)
# =============================================================================


class TestVerificationAccuracyReward:
    """Test compute_verification_accuracy_reward."""

    def test_boxed_correct_a1_right(self) -> None:
        assert (
            compute_verification_accuracy_reward(
                "<think>x</think> \\boxed{CORRECT}", a1_is_correct=True
            )
            == 1.0
        )

    def test_boxed_correct_a1_wrong(self) -> None:
        assert (
            compute_verification_accuracy_reward(
                "<think>x</think> \\boxed{CORRECT}", a1_is_correct=False
            )
            == -1.0
        )

    def test_boxed_incorrect_a1_wrong(self) -> None:
        assert (
            compute_verification_accuracy_reward(
                "<think>x</think> \\boxed{INCORRECT}", a1_is_correct=False
            )
            == 1.0
        )

    def test_boxed_incorrect_a1_right(self) -> None:
        assert (
            compute_verification_accuracy_reward(
                "<think>x</think> \\boxed{INCORRECT}", a1_is_correct=True
            )
            == -1.0
        )

    def test_plain_text_no_box_treated_as_wrong(self) -> None:
        """Plain "CORRECT" without \\boxed{} → no extraction → -1 (wrong verdict)."""
        assert (
            compute_verification_accuracy_reward("CORRECT. Well done.", a1_is_correct=True) == -1.0
        )

    def test_embedded_verdict_in_prose_no_box(self) -> None:
        """No keyword fallback any more: prose without \\boxed{} → -1 (wrong)."""
        assert (
            compute_verification_accuracy_reward(
                "The answer appears incorrect", a1_is_correct=False
            )
            == -1.0
        )

    def test_no_verdict_returns_minus_one(self) -> None:
        assert (
            compute_verification_accuracy_reward(
                "The painting shows blue tones", a1_is_correct=True
            )
            == -1.0
        )

    def test_empty_returns_minus_one(self) -> None:
        """Empty F1 → no boxed verdict → treat as wrong verdict."""
        assert compute_verification_accuracy_reward("", a1_is_correct=True) == -1.0

    def test_no_box_with_keywords_still_minus_one(self) -> None:
        """Keyword fallback removed: 'incorrect' in prose without \\boxed{} → -1."""
        assert (
            compute_verification_accuracy_reward(
                "INCORRECT reasoning but the answer is correct", a1_is_correct=False
            )
            == -1.0
        )

    def test_box_lowercase_correct(self) -> None:
        """Boxed verdict is case-insensitive via .upper()."""
        assert (
            compute_verification_accuracy_reward(
                "<think>x</think> \\boxed{Correct}", a1_is_correct=True
            )
            == 1.0
        )

    def test_box_unparseable_verdict_returns_minus_one(self) -> None:
        """\\boxed{(A)} (not CORRECT/INCORRECT) → wrong verdict."""
        assert (
            compute_verification_accuracy_reward(
                "<think>x</think> \\boxed{(A)}", a1_is_correct=True
            )
            == -1.0
        )


# =============================================================================
# Feedback breakdown integration
# =============================================================================


class TestFeedbackBreakdown:
    """Test compute_feedback_reward_breakdown."""

    @pytest.fixture()
    def weights(self) -> FeedbackRewardWeights:
        # Use convex weights that match v10: 0.45 / 0.45 / 0.1
        return FeedbackRewardWeights(w_downstream=0.45, w_verification_accuracy=0.45, w_format=0.1)

    def test_wr_honest_boxed(self, weights: FeedbackRewardWeights) -> None:
        """F1 says <think>...</think>\\boxed{INCORRECT} (honest on WR): full reward."""
        bd = compute_feedback_reward_breakdown(
            feedback_text="<think>The red dot is on the spout, not the handle.</think> \\boxed{INCORRECT}",
            a1_text="ZZZ_WRONG",
            a2_text="A",
            ground_truth="A",
            answer_type="mcq",
            choices="(A) (B) (C) (D)",
            weights=weights,
            reward_shaping_alpha=5.0,
        )
        # downstream (WR, α=5): 1 + 5·2 = 11
        # verification: INCORRECT when A1 wrong → +1
        # format: has <think> and \\boxed{INCORRECT} → +1
        # gate passes (r_verif > 0), downstream not zeroed.
        assert bd.components["downstream"] == pytest.approx(11.0, abs=0.01)
        assert bd.components["verification"] == 1.0
        assert bd.components["format"] == 1.0
        # total = 0.45·11 + 0.45·1 + 0.1·1 = 4.95 + 0.45 + 0.1 = +5.5
        assert bd.total_reward == pytest.approx(5.5, abs=0.01)

    def test_rr_honest_boxed(self, weights: FeedbackRewardWeights) -> None:
        """F1 says <think>...</think>\\boxed{CORRECT} (honest on RR)."""
        bd = compute_feedback_reward_breakdown(
            feedback_text="<think>Image supports (A).</think> \\boxed{CORRECT}",
            a1_text="A",
            a2_text="A",
            ground_truth="A",
            answer_type="mcq",
            choices="(A) (B) (C) (D)",
            weights=weights,
            reward_shaping_alpha=5.0,
        )
        # downstream (RR, α=5): 1 + 5·0 = 1; verif=+1; format=+1
        # total = 0.45·1 + 0.45·1 + 0.1·1 = +1.0
        assert bd.total_reward == pytest.approx(1.0, abs=0.01)

    def test_no_boxed_natural_path(self, weights: FeedbackRewardWeights) -> None:
        """Plain text verdict (no <think>, no \\boxed{}): no short-circuit.

        Missing \\boxed{} → verification=-1 (wrong verdict), downstream
        gated to 0 by verification < 0, format=0 (no bonus). No special
        sentinel, no -2.0 penalty.
        """
        bd = compute_feedback_reward_breakdown(
            feedback_text="INCORRECT. Should be (A).",
            a1_text="ZZZ_WRONG",
            a2_text="A",
            ground_truth="A",
            answer_type="mcq",
            choices="(A) (B) (C) (D)",
            weights=weights,
            reward_shaping_alpha=5.0,
        )
        assert bd.components["verification"] == -1.0
        assert bd.components["downstream"] == 0.0  # gated by verification < 0
        assert bd.components["format"] == 0.0
        # total = 0.45·0 + 0.45·(-1) + 0.1·0 = -0.45
        assert bd.total_reward == pytest.approx(-0.45, abs=0.01)

    def test_partial_format_no_boxed_natural(self, weights: FeedbackRewardWeights) -> None:
        """<think>...</think> without \\boxed{} → verification=-1, format=0."""
        bd = compute_feedback_reward_breakdown(
            feedback_text="<think>reasoning</think> INCORRECT",
            a1_text="ZZZ_WRONG",
            a2_text="A",
            ground_truth="A",
            answer_type="mcq",
            choices="(A) (B) (C) (D)",
            weights=weights,
            reward_shaping_alpha=5.0,
        )
        assert bd.components["verification"] == -1.0
        assert bd.components["downstream"] == 0.0
        assert bd.components["format"] == 0.0
        assert bd.total_reward == pytest.approx(-0.45, abs=0.01)

    def test_boxed_only_no_think(self, weights: FeedbackRewardWeights) -> None:
        """\\boxed{INCORRECT} without <think>: verification credit, no format bonus."""
        bd = compute_feedback_reward_breakdown(
            feedback_text="\\boxed{INCORRECT}",
            a1_text="ZZZ_WRONG",
            a2_text="A",
            ground_truth="A",
            answer_type="mcq",
            choices="(A) (B) (C) (D)",
            weights=weights,
            reward_shaping_alpha=5.0,
        )
        # boxed extraction works → verdict=INCORRECT matches A1 wrong → +1
        # downstream gate passes (verification > 0) → r_downstream = 11 (WR)
        # format = 0 (missing <think>)
        assert bd.components["verification"] == 1.0
        assert bd.components["downstream"] == pytest.approx(11.0, abs=0.01)
        assert bd.components["format"] == 0.0
        # total = 0.45·11 + 0.45·1 + 0.1·0 = 5.4
        assert bd.total_reward == pytest.approx(5.4, abs=0.01)

    def test_ww_miscalibrated_asymmetric_gate(self, weights: FeedbackRewardWeights) -> None:
        """F1 sycophantic CORRECT + WW (A2 still wrong) — negative downstream flows.

        Asymmetric gate: only positive downstream is gated when verdict wrong;
        negative downstream flows through so F1 is penalised for reinforcing
        a wrong A1 that A2 didn't recover from.
        """
        bd = compute_feedback_reward_breakdown(
            feedback_text="<think>Looks fine.</think> \\boxed{CORRECT}",
            a1_text="ZZZ_WRONG",
            a2_text="ZZZ_STILL_WRONG",
            ground_truth="A",
            answer_type="mcq",
            choices="(A) (B) (C) (D)",
            weights=weights,
            reward_shaping_alpha=5.0,
        )
        # WW downstream shaped: r_a2(-1) + 5·(r_a2-r_a1)=0 → -1
        # verification: CORRECT when A1 wrong → -1
        # asymmetric gate: negative flows through → ds = -1
        # total = 0.45·(-1) + 0.45·(-1) + 0.1·(+1) = -0.80
        assert bd.components["downstream"] == -1.0
        assert bd.components["verification"] == -1.0
        assert bd.components["format"] == 1.0
        assert bd.total_reward == pytest.approx(-0.80, abs=0.01)

    def test_wr_sycophantic_positive_downstream_gated(self, weights: FeedbackRewardWeights) -> None:
        """F1 sycophantic CORRECT + WR (A2 variance-flip right) — positive gated."""
        bd = compute_feedback_reward_breakdown(
            feedback_text="<think>(B) looks right.</think> \\boxed{CORRECT}",
            a1_text="ZZZ_WRONG",
            a2_text="A",  # A2 variance-corrected despite F1 sycophancy
            ground_truth="A",
            answer_type="mcq",
            choices="(A) (B) (C) (D)",
            weights=weights,
            reward_shaping_alpha=5.0,
        )
        # Without gate: would be 0.45·11 + 0.45·(-1) + 0.1·1 = +4.65 (sycophancy reward).
        # Asymmetric gate: raw ds=+11 → clamped to 0. Negative half of gate
        # doesn't engage here.
        # total = 0.45·0 + 0.45·(-1) + 0.1·1 = -0.35
        assert bd.components["downstream"] == 0.0
        assert bd.total_reward == pytest.approx(-0.35, abs=0.01)

    def test_rw_wrong_incorrect_causes_harm(self, weights: FeedbackRewardWeights) -> None:
        """F1 wrong INCORRECT on a right A1 → A2 follows bad advice and regresses.

        The worst case: F1 actively caused harm. Asymmetric gate lets the
        full negative downstream (-11) through even though verdict is wrong.
        """
        bd = compute_feedback_reward_breakdown(
            feedback_text="<think>You're wrong, reconsider.</think> \\boxed{INCORRECT}",
            a1_text="A",
            a2_text="B",  # A2 regressed because of F1's bad advice
            ground_truth="A",
            answer_type="mcq",
            choices="(A) (B) (C) (D)",
            weights=weights,
            reward_shaping_alpha=5.0,
        )
        # RW shaped: r_a2(-1) + 5·(r_a2-r_a1) = -1 + 5·(-2) = -11
        # verification: INCORRECT when A1 right → -1
        # asymmetric gate: negative flows through → ds = -11
        # total = 0.45·(-11) + 0.45·(-1) + 0.1·(+1) = -5.30
        assert bd.components["downstream"] == -11.0
        assert bd.components["verification"] == -1.0
        assert bd.total_reward == pytest.approx(-5.30, abs=0.01)


class TestBoxedExtraction:
    """Test \\boxed{} extraction helpers and format reward."""

    def test_extract_boxed_correct(self) -> None:
        assert extract_from_boxed("<think>x</think> \\boxed{CORRECT}") == "CORRECT"

    def test_extract_boxed_incorrect(self) -> None:
        assert extract_from_boxed("<think>x</think> \\boxed{INCORRECT}") == "INCORRECT"

    def test_extract_boxed_case_insensitive(self) -> None:
        assert extract_from_boxed("\\boxed{correct}").upper() == "CORRECT"

    def test_extract_boxed_absent(self) -> None:
        assert extract_from_boxed("INCORRECT plain text") == ""

    def test_has_think_boxed_valid(self) -> None:
        assert has_think_boxed("<think>reasoning</think> \\boxed{CORRECT}") is True

    def test_has_think_boxed_missing_boxed(self) -> None:
        assert has_think_boxed("<think>reasoning</think> INCORRECT") is False

    def test_has_think_boxed_missing_think(self) -> None:
        assert has_think_boxed("\\boxed{CORRECT}") is False

    def test_format_reward_valid(self) -> None:
        assert compute_feedback_format_reward("<think>x</think> \\boxed{CORRECT}") == 1.0

    def test_format_reward_structurally_complete_but_invalid_verdict(self) -> None:
        """Format is purely structural — invalid verdict still gets +1 (verification handles correctness)."""
        assert compute_feedback_format_reward("<think>x</think> \\boxed{maybe}") == 0.0

    def test_format_reward_missing(self) -> None:
        """Binary {0,+1}: missing structure → 0 (no penalty, no bonus)."""
        assert compute_feedback_format_reward("INCORRECT plain text") == 0.0

    def test_verification_only_uses_boxed(self) -> None:
        # F1 text says 'correct' in prose but boxes INCORRECT — boxed is the only signal.
        verdict = "<think>It might seem correct but I disagree.</think> \\boxed{INCORRECT}"
        assert compute_verification_accuracy_reward(verdict, a1_is_correct=False) == 1.0


# =============================================================================
# Pattern A prompt structure tests (v10+)
# =============================================================================


class TestPatternAStructure:
    """Confirm all three builders return single-user-turn, no system, no role-flip."""

    def test_a1_single_user_turn(self) -> None:
        msgs = build_initial_answer_prompt("What color?", use_think_answer_tags=True)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        # No system message
        assert all(m["role"] != "system" for m in msgs)

    def test_a1_embeds_think_answer_instruction_in_user(self) -> None:
        msgs = build_initial_answer_prompt("What color?", use_think_answer_tags=True)
        text = msgs[0]["content"][-1]["text"]
        assert "What color?" in text
        assert THINK_ANSWER_INSTRUCTION in text
        assert "<think>" in text
        assert "<answer>" in text

    def test_a1_answer_tag_only_still_works(self) -> None:
        msgs = build_initial_answer_prompt("What color?", use_answer_tag_only=True)
        text = msgs[0]["content"][-1]["text"]
        assert ANSWER_TAG_INSTRUCTION in text
        assert "<think>" not in text

    def test_a1_image_in_user(self) -> None:
        msgs = build_initial_answer_prompt("What?", use_think_answer_tags=True)
        types = [c["type"] for c in msgs[0]["content"]]
        assert "image" in types
        assert types[0] == "image"  # image before text

    def test_f1_single_user_turn_no_role_flip(self) -> None:
        msgs = build_critic_prompt("What color?", "Blue", model_type="qwen2vl")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        # No assistant turn before user (no role flip)
        assert all(m["role"] != "assistant" for m in msgs)
        assert all(m["role"] != "system" for m in msgs)

    def test_f1_contains_question_candidate_and_instruction(self) -> None:
        msgs = build_critic_prompt("What color?", "Blue", model_type="qwen2vl")
        text = msgs[0]["content"][-1]["text"]
        assert "Question: What color?" in text
        assert "Candidate answer: Blue" in text
        assert F1_VERIFIER_INSTRUCTION in text

    def test_a2_single_user_turn_flat(self) -> None:
        msgs = build_refiner_prompt(
            "What color?", "Blue", "INCORRECT. Brown.", use_think_answer_tags=True
        )
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        # No stacked assistant/user turns — A1 and F1 are embedded as text
        assert all(m["role"] != "assistant" for m in msgs)
        assert all(m["role"] != "system" for m in msgs)

    def test_a2_embeds_prior_context_in_user(self) -> None:
        msgs = build_refiner_prompt(
            "What color?", "Blue", "INCORRECT. Brown.", use_think_answer_tags=True
        )
        text = msgs[0]["content"][-1]["text"]
        assert "Question: What color?" in text
        assert "Your previous answer: Blue" in text
        assert "Feedback on your previous answer: INCORRECT. Brown." in text
        assert THINK_ANSWER_INSTRUCTION in text


class TestCompletionAppend:
    """Test build_prompt_with_completion appends assistant turn correctly."""

    def test_appends_assistant(self) -> None:
        prompt = build_initial_answer_prompt("What?", use_think_answer_tags=True)
        full = build_prompt_with_completion(prompt, "<answer>(A)</answer>")
        assert len(full) == 2
        assert full[-1]["role"] == "assistant"
        assert full[-1]["content"][0]["text"] == "<answer>(A)</answer>"
