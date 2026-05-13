#!/usr/bin/env python3
"""Tests for the PAG selective-revision gate extraction logic.

Matches PAG paper §3.1 + released code (``verl/workers/rollout/vllm_rollout/
vllm_pag_rollout_spmd.py``): the gate triggers iff F1's verdict is the
literal "correct" sentinel. We translate "trailing sentence regex" →
"our ``\\boxed{CORRECT|INCORRECT}`` extraction" since that's the verdict
format our F1 instruction asks for.

Gate semantics:
  - boxed == "CORRECT"      → trajectory STOPS (A2 not generated)
  - boxed == "INCORRECT"    → A2 generated (revision)
  - boxed missing / malformed → A2 generated (safe default — never stop
    a trajectory based on an unparseable verdict)
"""

from vlm_grpo.trajectory import extract_from_boxed


def _gate_should_skip_a2(f1_text: str) -> bool:
    """Mirrors the gate logic in ``rollout.py`` — pulled out for testing.

    A2 is SKIPPED iff the boxed extraction is exactly ``CORRECT``
    (case-insensitive, whitespace-tolerant). Anything else, including
    ``INCORRECT``, missing boxed, or arbitrary garbage, proceeds to A2.
    """
    verdict = extract_from_boxed(f1_text).upper().strip()
    return verdict == "CORRECT"


def test_gate_correct_skips_a2():
    """Standard well-formed F1 saying CORRECT → gate fires."""
    f1 = "<think>I checked each step</think>\\boxed{CORRECT}"
    assert _gate_should_skip_a2(f1) is True


def test_gate_incorrect_generates_a2():
    """Standard well-formed F1 saying INCORRECT → A2 generated."""
    f1 = "<think>found an error</think>\\boxed{INCORRECT}"
    assert _gate_should_skip_a2(f1) is False


def test_gate_correct_case_insensitive():
    """Lowercase ``correct`` is still treated as the stop sentinel.

    PAG's regex matches case-insensitively; our ``.upper()`` does the
    same after extraction.
    """
    assert _gate_should_skip_a2("<think>ok</think>\\boxed{correct}") is True
    assert _gate_should_skip_a2("<think>ok</think>\\boxed{Correct}") is True


def test_gate_correct_whitespace_tolerant():
    """Inner whitespace inside boxed should not affect the gate."""
    f1 = "<think>ok</think>\\boxed{   CORRECT   }"
    assert _gate_should_skip_a2(f1) is True


def test_gate_missing_boxed_does_not_skip():
    """No ``\\boxed{}`` in F1 → cannot extract verdict → safe default
    is to generate A2 (don't terminate on an ambiguous F1).

    PAG paper §A.4 + Fig. 13 lesson: never let an unparseable F1
    short-circuit the policy turn — that's how the verifier-collapse
    pathway opens.
    """
    assert _gate_should_skip_a2("<think>no verdict here</think>") is False
    assert _gate_should_skip_a2("just some prose with no tags at all") is False
    assert _gate_should_skip_a2("") is False


def test_gate_garbage_boxed_does_not_skip():
    """``\\boxed{maybe}`` / arbitrary inner content → A2 generated.

    Only the exact ``CORRECT`` sentinel triggers the gate.
    """
    assert _gate_should_skip_a2("<think>x</think>\\boxed{maybe correct}") is False
    assert _gate_should_skip_a2("<think>x</think>\\boxed{42}") is False
    assert _gate_should_skip_a2("<think>x</think>\\boxed{(A)}") is False


def test_gate_partial_sentinel_does_not_skip():
    """``\\boxed{CORRECTish}`` is not ``CORRECT`` — only the exact
    sentinel triggers the gate. Prevents a half-formed verdict from
    short-circuiting an otherwise revisable trajectory.
    """
    assert _gate_should_skip_a2("<think>x</think>\\boxed{CORRECTish}") is False
    assert _gate_should_skip_a2("<think>x</think>\\boxed{INCORRECT}") is False
