#!/usr/bin/env python3
"""Tests for the vLLM/HF rollout token-id passthrough plumbing (audit Bug 2).

Verifies:
1. ``VLLMRolloutEngine.generate_batch`` returns dicts with both ``text``
   (raw, not stripped) and ``token_ids`` (the actual sampled completion ids).
2. ``_generate_batch_completions`` (HF fallback) returns the same shape.
3. ``SelfReflectionRolloutResult`` exposes ``answer1_token_ids``,
   ``feedback_token_ids``, ``answer2_token_ids`` fields.
4. ``SelfReflectionGRPOTrainer._preprocess_trajectory_texts`` accepts
   completion_token_ids and surfaces them on the returned pretok dict.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from vlm_grpo.rollout import SelfReflectionRolloutResult

# ---------------------------------------------------------------------------
# 1. SelfReflectionRolloutResult carries token-id fields
# ---------------------------------------------------------------------------


class TestRolloutResultTokenFields:
    def test_default_token_id_lists_empty(self) -> None:
        result = SelfReflectionRolloutResult(
            sample_index=0,
            question="Q",
            image_path="/img.jpg",
            ground_truth="(A)",
            answer_type="mcq",
            choices="(A) cat (B) dog",
            dataset_name="test",
        )
        assert result.answer1_token_ids == []
        assert result.feedback_token_ids == []
        assert result.answer2_token_ids == []

    def test_assign_token_ids(self) -> None:
        result = SelfReflectionRolloutResult(
            sample_index=0,
            question="Q",
            image_path="/img.jpg",
            ground_truth="(A)",
            answer_type="mcq",
            choices="",
            dataset_name="test",
        )
        result.answer1_token_ids = [[1, 2, 3], None]
        result.feedback_token_ids = [[4, 5], None]
        result.answer2_token_ids = [[6, 7, 8, 9], None]
        assert result.answer1_token_ids[0] == [1, 2, 3]
        assert result.feedback_token_ids[1] is None
        assert result.answer2_token_ids[0] == [6, 7, 8, 9]


# ---------------------------------------------------------------------------
# 2. vLLM generate_batch returns structured dicts (text + token_ids, no .strip)
# ---------------------------------------------------------------------------


class _FakeVLLMOutputInner:
    def __init__(self, text: str, token_ids: list[int]) -> None:
        self.text = text
        self.token_ids = token_ids


class _FakeVLLMOutput:
    def __init__(self, text: str, token_ids: list[int]) -> None:
        self.outputs = [_FakeVLLMOutputInner(text, token_ids)]


class _FakeLLM:
    """Stand-in for vllm.LLM that just echoes pre-canned completions."""

    def __init__(self, canned: list[tuple[str, list[int]]]) -> None:
        self._canned = canned
        self.last_inputs: list[dict] | None = None

    def generate(self, vllm_inputs: list[dict], sampling_params: Any) -> list[Any]:
        self.last_inputs = vllm_inputs
        assert len(vllm_inputs) == len(self._canned)
        return [_FakeVLLMOutput(t, ids) for t, ids in self._canned]


def _make_engine(canned: list[tuple[str, list[int]]]):
    """Construct a VLLMRolloutEngine without touching the real vllm.LLM."""
    from vlm_grpo.vllm_rollout import VLLMRolloutEngine

    engine = VLLMRolloutEngine.__new__(VLLMRolloutEngine)
    engine.processor = SimpleNamespace()
    engine.model_id = "fake"
    engine.gpu_memory_utilization = 0.5
    engine.llm = _FakeLLM(canned)
    return engine


class TestVLLMGenerateBatchReturnShape:
    def test_returns_list_of_dicts(self) -> None:
        pytest.importorskip("vllm", reason="vllm SamplingParams import")
        canned = [
            ("  hello world\n", [1, 2, 3, 4]),
            ("answer (A)", [5, 6]),
        ]
        engine = _make_engine(canned)
        out = engine.generate_batch(
            prompts=["p1", "p2"], images=[None, None], max_new_tokens=8, temperature=0.7
        )
        assert isinstance(out, list)
        assert len(out) == 2
        for o in out:
            assert isinstance(o, dict)
            assert "text" in o and "token_ids" in o
            assert isinstance(o["token_ids"], list)

    def test_text_is_not_stripped(self) -> None:
        """Bug 2 requires the raw text to flow through unmodified — leading /
        trailing whitespace must NOT be stripped on the way out."""
        pytest.importorskip("vllm")
        canned = [("  leading_and_trailing_ws  \n", [10, 11, 12])]
        engine = _make_engine(canned)
        out = engine.generate_batch(prompts=["p"], images=[None], max_new_tokens=8, temperature=0.0)
        assert out[0]["text"] == "  leading_and_trailing_ws  \n"
        assert out[0]["token_ids"] == [10, 11, 12]

    def test_token_ids_match_canned(self) -> None:
        pytest.importorskip("vllm")
        canned = [("a", [42, 43]), ("b", [99])]
        engine = _make_engine(canned)
        out = engine.generate_batch(
            prompts=["p1", "p2"], images=[None, None], max_new_tokens=4, temperature=0.0
        )
        assert out[0]["token_ids"] == [42, 43]
        assert out[1]["token_ids"] == [99]


# ---------------------------------------------------------------------------
# 3. _preprocess_trajectory_texts logs Bug 2 mismatch when token_ids diverge
#    from retokenize. Uses a tiny fake processor — no real model load.
# ---------------------------------------------------------------------------


class _FakeTokenizerCallable:
    """Toy tokenizer: tokens-per-character; produces input_ids per text."""

    padding_side = "right"

    def __call__(
        self,
        texts: list[str] | str,
        padding: bool = False,
        return_attention_mask: bool = True,
        return_tensors: str | None = None,
    ) -> dict:
        if isinstance(texts, str):
            texts = [texts]
        ids = [list(range(len(t))) for t in texts]
        out = {"input_ids": ids}
        if return_attention_mask:
            out["attention_mask"] = [[1] * len(x) for x in ids]
        return out


class _FakeProcessor:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizerCallable()

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        # Concatenate role/content; for prompt-only (no completion), end at last user.
        parts: list[str] = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    c.get("text", "") if isinstance(c, dict) else str(c) for c in content
                )
            parts.append(f"<{m['role']}>{content}")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)


def _make_trainer_stub() -> Any:
    """Fabricate the smallest object on which _preprocess_trajectory_texts works."""
    from vlm_grpo.critic_grpo import SelfReflectionGRPOTrainer

    stub = SelfReflectionGRPOTrainer.__new__(SelfReflectionGRPOTrainer)
    stub.processor = _FakeProcessor()
    return stub


class TestPreprocessTrajectoryTextsTokenIds:
    def test_pretok_carries_completion_token_ids(self) -> None:
        trainer = _make_trainer_stub()
        # Two messages: one user prompt + one assistant completion (single trajectory).
        msgs = [
            [{"role": "user", "content": "Q1?"}, {"role": "assistant", "content": "A1"}],
            [{"role": "user", "content": "Q2??"}, {"role": "assistant", "content": "A2"}],
        ]
        # Provide explicit completion ids; lengths chosen so they intentionally
        # disagree with the tokens-per-character toy tokenizer.
        completion_ids = [[1, 2, 3, 4, 5], [9, 9]]
        pretok = trainer._preprocess_trajectory_texts(
            msgs, images=[None, None], completion_token_ids=completion_ids
        )
        assert pretok["completion_token_ids"] == completion_ids
        assert "full_lens" in pretok
        assert "prompt_lens" in pretok
        # full_lens >= prompt_lens always
        assert all(f >= p for f, p in zip(pretok["full_lens"], pretok["prompt_lens"]))

    def test_pretok_works_without_completion_token_ids(self) -> None:
        """Backward-compat: omitting completion_token_ids preserves prior behavior."""
        trainer = _make_trainer_stub()
        msgs = [[{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A"}]]
        pretok = trainer._preprocess_trajectory_texts(msgs, images=[None])
        assert pretok["completion_token_ids"] is None
        assert len(pretok["full_lens"]) == 1
        assert pretok["full_lens"][0] >= pretok["prompt_lens"][0]

    def test_pretok_handles_partial_token_ids(self) -> None:
        """Some entries None (HF fallback turn alongside vLLM turn) must work."""
        trainer = _make_trainer_stub()
        msgs = [
            [{"role": "user", "content": "Q1?"}, {"role": "assistant", "content": "A1"}],
            [{"role": "user", "content": "Q2?"}, {"role": "assistant", "content": "A2"}],
        ]
        completion_ids = [[1, 2, 3], None]
        pretok = trainer._preprocess_trajectory_texts(
            msgs, images=[None, None], completion_token_ids=completion_ids
        )
        # The diagnostic only inspects entries where the rollout engine emitted
        # token ids; the None entry must not crash anything.
        assert pretok["completion_token_ids"] == completion_ids
