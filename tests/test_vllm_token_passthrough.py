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


class _FakeLogprob:
    """Match vllm.outputs.Logprob — only ``.logprob`` is read by the trainer."""

    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


class _FakeVLLMOutputInner:
    def __init__(
        self,
        text: str,
        token_ids: list[int],
        logprobs: list[float] | None = None,
    ) -> None:
        self.text = text
        self.token_ids = token_ids
        # vLLM shape: list[dict[token_id, Logprob] | None] aligned with token_ids
        if logprobs is None:
            self.logprobs = None
        else:
            assert len(logprobs) == len(token_ids), "logprobs must align with token_ids"
            self.logprobs = [
                {tok: _FakeLogprob(lp)} for tok, lp in zip(token_ids, logprobs)
            ]


class _FakeVLLMOutput:
    def __init__(
        self,
        text: str,
        token_ids: list[int],
        logprobs: list[float] | None = None,
    ) -> None:
        self.outputs = [_FakeVLLMOutputInner(text, token_ids, logprobs)]


class _FakeLLM:
    """Stand-in for vllm.LLM that just echoes pre-canned completions."""

    def __init__(self, canned: list[tuple[str, list[int]]]) -> None:
        self._canned = canned
        self.last_inputs: list[dict] | None = None

    def generate(self, vllm_inputs: list[dict], sampling_params: Any) -> list[Any]:
        self.last_inputs = vllm_inputs
        assert len(vllm_inputs) == len(self._canned)
        # Synthesize per-token logprobs deterministically so tests can assert
        # exact values flowed through. ``-0.1 * (i + 1)`` per token i.
        return [
            _FakeVLLMOutput(t, ids, logprobs=[-0.1 * (i + 1) for i in range(len(ids))])
            for t, ids in self._canned
        ]


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

    def test_logprobs_returned_aligned_with_token_ids(self) -> None:
        """vLLM's sample-time per-token logprobs must flow through 1:1 with
        token_ids so the trainer can use them as ``old_lp`` directly."""
        pytest.importorskip("vllm")
        canned = [("a", [42, 43]), ("b", [99])]
        engine = _make_engine(canned)
        out = engine.generate_batch(
            prompts=["p1", "p2"], images=[None, None], max_new_tokens=4, temperature=0.0
        )
        # _FakeLLM.generate synthesizes ``-0.1 * (i + 1)`` per token i.
        assert "logprobs" in out[0]
        assert "logprobs" in out[1]
        assert len(out[0]["logprobs"]) == len(out[0]["token_ids"])
        assert len(out[1]["logprobs"]) == len(out[1]["token_ids"])
        assert out[0]["logprobs"] == pytest.approx([-0.1, -0.2])
        assert out[1]["logprobs"] == pytest.approx([-0.1])

    def test_logprobs_default_to_zero_when_missing(self) -> None:
        """Defensive: if vLLM returns None/missing logprobs (shouldn't happen
        when SamplingParams(logprobs>=1) is set, but guard against it), the
        trainer's exp(new_lp - old_lp) ratio should treat them as no-op."""
        pytest.importorskip("vllm")
        from vlm_grpo.vllm_rollout import VLLMRolloutEngine

        engine = VLLMRolloutEngine.__new__(VLLMRolloutEngine)
        engine.processor = SimpleNamespace()
        engine.model_id = "fake"
        engine.gpu_memory_utilization = 0.5

        class _NoLogprobsLLM:
            def generate(self, vllm_inputs: list[dict], _params: Any) -> list[Any]:
                # Return outputs with logprobs explicitly set to None.
                return [_FakeVLLMOutput("x", [7, 8], logprobs=None)]

        engine.llm = _NoLogprobsLLM()
        out = engine.generate_batch(
            prompts=["p"], images=[None], max_new_tokens=4, temperature=0.0
        )
        assert out[0]["logprobs"] == [0.0, 0.0]


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


def _make_trainer_stub_with_native(use_vllm_native_loss: bool) -> Any:
    """Trainer stub with config.rollout.use_vllm_native_loss configured."""
    from vlm_grpo.critic_grpo import SelfReflectionGRPOTrainer

    stub = SelfReflectionGRPOTrainer.__new__(SelfReflectionGRPOTrainer)
    stub.processor = _FakeProcessor()
    stub.config = SimpleNamespace(
        rollout=SimpleNamespace(use_vllm_native_loss=use_vllm_native_loss)
    )
    return stub


# ---------------------------------------------------------------------------
# 4. Native path: full_lens computed from vLLM ids (Bug 2 fix).
# ---------------------------------------------------------------------------


class TestPretokNativePath:
    """``use_vllm_native_loss=True`` makes Bug 2 structurally impossible.

    The trainer's forward pass assembles ``input_ids = prompt_ids ++
    vllm_completion_ids`` directly, so ``full_lens`` is computed as
    ``prompt_lens + len(completion_ids)`` rather than from a retokenize that
    can disagree with vLLM's actual sampling boundary.
    """

    def test_native_path_full_lens_match_completion_ids(self) -> None:
        trainer = _make_trainer_stub_with_native(use_vllm_native_loss=True)
        msgs = [
            [{"role": "user", "content": "Q1?"}, {"role": "assistant", "content": "A1"}],
            [{"role": "user", "content": "Q2!"}, {"role": "assistant", "content": "A2"}],
        ]
        # Completion ids carry lengths 5 and 11 — DELIBERATELY different from
        # what the toy tokenizer would produce by retokenizing the assistant
        # text "A1" / "A2" (which would give 2 each). Native path should
        # honor the supplied lengths.
        completion_ids = [[1, 2, 3, 4, 5], list(range(100, 111))]
        completion_logprobs = [[-0.1] * 5, [-0.2] * 11]
        pretok = trainer._preprocess_trajectory_texts(
            msgs,
            images=[None, None],
            completion_token_ids=completion_ids,
            completion_logprobs=completion_logprobs,
        )
        assert pretok["native_path"] is True
        # full_lens[i] - prompt_lens[i] == len(completion_ids[i])  ALWAYS.
        for i, ids in enumerate(completion_ids):
            assert pretok["full_lens"][i] - pretok["prompt_lens"][i] == len(ids)
        assert pretok["completion_logprobs"] == completion_logprobs

    def test_legacy_path_when_flag_off(self) -> None:
        trainer = _make_trainer_stub_with_native(use_vllm_native_loss=False)
        msgs = [
            [{"role": "user", "content": "Q1?"}, {"role": "assistant", "content": "A1"}],
        ]
        completion_ids = [[1, 2, 3, 4, 5]]
        pretok = trainer._preprocess_trajectory_texts(
            msgs, images=[None], completion_token_ids=completion_ids
        )
        assert pretok["native_path"] is False
        # Legacy: full_lens comes from retokenizing the full text via the
        # toy tokenizer; lengths may NOT match completion_ids length. The
        # important property is that the path is unchanged from before.
        assert "full_lens" in pretok and "prompt_lens" in pretok

    def test_legacy_when_any_completion_ids_none(self) -> None:
        """Native path requires ALL trajectories to carry token ids; falling
        back when any are None preserves correctness for mixed batches."""
        trainer = _make_trainer_stub_with_native(use_vllm_native_loss=True)
        msgs = [
            [{"role": "user", "content": "Q1?"}, {"role": "assistant", "content": "A1"}],
            [{"role": "user", "content": "Q2?"}, {"role": "assistant", "content": "A2"}],
        ]
        completion_ids = [[1, 2, 3], None]
        pretok = trainer._preprocess_trajectory_texts(
            msgs, images=[None, None], completion_token_ids=completion_ids
        )
        # Mixed batch must NOT take the native path — the forward pass needs
        # all-or-nothing for the manual assembly to be uniform.
        assert pretok["native_path"] is False

    def test_pretok_carries_completion_logprobs(self) -> None:
        trainer = _make_trainer_stub()
        msgs = [
            [{"role": "user", "content": "Q1?"}, {"role": "assistant", "content": "A1"}],
        ]
        pretok = trainer._preprocess_trajectory_texts(
            msgs,
            images=[None],
            completion_token_ids=[[1, 2]],
            completion_logprobs=[[-0.5, -0.3]],
        )
        assert pretok["completion_logprobs"] == [[-0.5, -0.3]]


# ---------------------------------------------------------------------------
# 5. Native forward pass label alignment.
# ---------------------------------------------------------------------------
#
# The single highest-risk piece of this change is the manual assembly of
# input_ids = prompt_ids ++ vllm_completion_ids inside
# _forward_from_pretokenized_multi. An off-by-one in shift_logits /
# shift_labels would silently corrupt every gradient.
#
# We can exercise the native branch end-to-end without loading Qwen2.5-VL by
# stubbing self.processor (returns deterministic prompt input_ids) and the
# model callable (returns deterministic logits). With a logits tensor whose
# argmax at every position equals the input_id at that position, log_softmax
# is dominated by ~1.0 mass on the gathered token, so the per-token logprob
# of every COMPLETION token recovered from the gather is verifiable.


class _StubTorchTokenizer:
    """Minimal tokenizer that produces deterministic input_ids by mapping
    each character to its Unicode code point. Padding-aware. Returns
    torch tensors when ``return_tensors='pt'``."""

    def __init__(self, pad_token_id: int = 0) -> None:
        self.padding_side = "right"
        self.pad_token_id = pad_token_id
        self.eos_token_id = pad_token_id

    def __call__(
        self,
        texts,
        padding: bool = False,
        return_attention_mask: bool = True,
        return_tensors: str | None = None,
    ):
        if isinstance(texts, str):
            texts = [texts]
        ids = [[ord(c) for c in t] for t in texts]
        if return_tensors == "pt":
            import torch

            max_len = max((len(x) for x in ids), default=0)
            n = len(ids)
            input_ids = torch.full((n, max_len), self.pad_token_id, dtype=torch.long)
            attn = torch.zeros((n, max_len), dtype=torch.long)
            for i, row in enumerate(ids):
                if self.padding_side == "left":
                    input_ids[i, max_len - len(row) :] = torch.tensor(row, dtype=torch.long)
                    attn[i, max_len - len(row) :] = 1
                else:
                    input_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
                    attn[i, : len(row)] = 1
            out = {"input_ids": input_ids, "attention_mask": attn}
            return out
        # Fallback shape used by the legacy path (returns lists, no tensors).
        out = {"input_ids": ids}
        if return_attention_mask:
            out["attention_mask"] = [[1] * len(x) for x in ids]
        return out


class _StubProcessor:
    def __init__(self) -> None:
        self.tokenizer = _StubTorchTokenizer()

    def __call__(
        self,
        text=None,
        images=None,
        return_tensors: str | None = None,
        padding: bool = False,
    ):
        # Minimal SimpleNamespace-style dict mimicking BatchFeature with .to().
        out = self.tokenizer(text, padding=padding, return_tensors=return_tensors)
        # Add a .to() that just returns self (no real device move).
        return _StubBatch(out)

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        parts = []
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


class _StubBatch(dict):
    """dict with .to(device) → self for the test (no device move)."""

    def to(self, _device):
        return self


def _make_native_trainer_stub():
    """Trainer set up to take the native path with a stub processor."""
    from vlm_grpo.critic_grpo import SelfReflectionGRPOTrainer

    stub = SelfReflectionGRPOTrainer.__new__(SelfReflectionGRPOTrainer)
    stub.processor = _StubProcessor()
    stub.config = SimpleNamespace(
        rollout=SimpleNamespace(use_vllm_native_loss=True)
    )
    # _forward_from_pretokenized_multi reads self.device for tensor placement.
    import torch

    stub.device = torch.device("cpu")
    stub._entropy_sum = 0.0
    stub._entropy_tokens = 0
    return stub


class _IdentityLogitsModel:
    """Deterministic stub: logits[i, t, v] = 1.0 iff v == input_ids[i, t],
    else 0.0. After log_softmax, the gathered token at each position
    receives the dominant log-prob (close to 0 for vocab=2, more negative
    for larger vocab — value verified analytically per test)."""

    def __init__(self, vocab_size: int = 1024) -> None:
        self.vocab_size = vocab_size

    def __call__(self, input_ids=None, attention_mask=None, use_cache=None, **_kw):
        import torch

        n, seq_len = input_ids.shape
        logits = torch.zeros(n, seq_len, self.vocab_size)
        # Place a unit at the index equal to the token id at each position.
        idx = input_ids.clamp(min=0, max=self.vocab_size - 1).unsqueeze(-1)
        logits.scatter_(-1, idx, 1.0)
        return SimpleNamespace(logits=logits)


class TestNativeForwardLabelAlignment:
    """Closes the test gap the code-reviewer flagged: native-path manual
    assembly is fully exercised against deterministic logits, so an
    off-by-one in shift_logits / shift_labels would surface here."""

    def test_completion_lps_match_completion_ids_length(self) -> None:
        import torch

        trainer = _make_native_trainer_stub()
        msgs = [
            [{"role": "user", "content": "ABCD"}, {"role": "assistant", "content": "Z"}],
            [{"role": "user", "content": "EFG"}, {"role": "assistant", "content": "Z"}],
        ]
        completion_ids = [[1, 2, 3], [4, 5]]
        completion_logprobs = [[-0.1] * 3, [-0.2] * 2]
        pretok = trainer._preprocess_trajectory_texts(
            msgs,
            images=[None, None],
            completion_token_ids=completion_ids,
            completion_logprobs=completion_logprobs,
        )
        assert pretok["native_path"] is True

        model = _IdentityLogitsModel(vocab_size=1024)
        (lps_a1,) = trainer._forward_from_pretokenized_multi(
            [pretok], model, mb_start=0, mb_end=2
        )
        # One tensor per trajectory; tensor length equals completion length.
        assert len(lps_a1) == 2
        assert lps_a1[0].shape[0] == len(completion_ids[0])
        assert lps_a1[1].shape[0] == len(completion_ids[1])
        # All log-probs are finite (no nan/inf — would indicate the gather
        # picked up a padding row instead of a real completion position).
        assert torch.isfinite(lps_a1[0]).all()
        assert torch.isfinite(lps_a1[1]).all()

    def test_completion_lps_recover_correct_argmax_token(self) -> None:
        """Direct off-by-one test on the GRPO label semantics.

        The stub places a +1.0 logit unit at ``logits[i, t,
        input_ids[i, t]]`` — i.e. the unit at position t favors predicting
        the token AT position t.

        GRPO labels mean: ``logits[t]`` predicts ``input_ids[t+1]``. The
        forward gathers ``log_softmax(logits[real_prompt_start - 1 + t])
        [input_ids[real_prompt_start + t]]`` for t-th completion token.

        With the identity stub:
          - ``logits[real_prompt_start - 1 + t]`` has its +1.0 unit at the
            PRIOR token's id (``input_ids[real_prompt_start - 1 + t]``),
            NOT at ``input_ids[real_prompt_start + t]``.
          - So the gather picks up logit 0.0 → log_softmax = -log(vocab).
          - If alignment were off by 1 (gathering the current token's own
            logit instead), we'd see log_softmax ≈ 1 - log(vocab + e).

        The two values differ by exactly 1 nat. Asserting we land on the
        first value, not the second, is a sharp off-by-one detector.
        """
        import math

        trainer = _make_native_trainer_stub()
        msgs = [
            [{"role": "user", "content": "AB"}, {"role": "assistant", "content": "Z"}],
        ]
        completion_ids = [[7, 8, 9]]
        completion_logprobs = [[0.0, 0.0, 0.0]]
        pretok = trainer._preprocess_trajectory_texts(
            msgs,
            images=[None],
            completion_token_ids=completion_ids,
            completion_logprobs=completion_logprobs,
        )

        vocab = 1024
        model = _IdentityLogitsModel(vocab_size=vocab)
        (lps_set,) = trainer._forward_from_pretokenized_multi(
            [pretok], model, mb_start=0, mb_end=1
        )
        token_lp = lps_set[0]

        # Correct alignment: gathered logit = 0.0
        # log_softmax over [0,...,0,1.0_at_one_index] of an UNRELATED 0.0 entry
        # = -log(vocab - 1 + e). For vocab=1024, e=2.718 → -log(1025.72) ≈ -6.933.
        expected_correct = -math.log(vocab - 1 + math.e)
        # Misaligned (off-by-one in the wrong direction): would gather +1.0
        # logit → log_softmax = 1 - log(vocab - 1 + e) ≈ -5.933.
        expected_misaligned = 1.0 + expected_correct

        for v in token_lp.tolist():
            assert abs(v - expected_correct) < 1e-3, (
                f"native-path label alignment broken: got {v}, expected "
                f"{expected_correct} (correct alignment) or {expected_misaligned} "
                f"(off-by-one). Value matches {expected_misaligned} ⇒ off-by-one bug."
            )
