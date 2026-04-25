#!/usr/bin/env python3
"""Dry-run: render Qwen chat template for training prompts vs eval-wrapper prompts.

Goal: verify that the eval wrapper produces the EXACT same rendered text the
model sees during training, so eval-time trajectories replicate training-time
trajectories. Compares:
  1. Current training builders (src/vlm_grpo/prompts.py — Pattern A, no
     custom system, single user turn, think+answer tags)
  2. v9b wrapper (old: role-flipped F1, stacked-conv A2, custom system,
     <answer>-only tags) — should DIFFER
  3. Proposed Pattern-A wrapper for eval — should MATCH training byte-for-byte

Usage:
    python3 dryrun_eval_vs_training.py
"""

from __future__ import annotations

import os
import sys

REPO_SRC = "/workspace/repo/src"
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from vlm_grpo.prompts import (  # noqa: E402
    THINK_ANSWER_INSTRUCTION,
    F1_VERIFIER_INSTRUCTION,
    build_initial_answer_prompt,
    build_critic_prompt,
    build_refiner_prompt,
)


SAMPLE = {
    "question": "Which of the two images is more colorful?\n(A) the first image\n(B) the second image",
    "a1_completion": "<think>The first image has saturated reds and yellows.</think><answer>(A)</answer>",
    "f1_completion": "<think>The first image clearly has more vibrant colors.</think>\\boxed{CORRECT}",
}


# ---------------------------------------------------------------------------
# v9b wrapper messages (OLD — what the existing eval YAML produces)
# ---------------------------------------------------------------------------
_V9B_VL_SYSTEM = (
    "You are a visual question answering assistant. "
    "Look at the image carefully and answer the question. "
    "When given feedback on your previous answer, re-examine the image "
    "and either correct your answer or keep it if you believe it is right.\n\n"
    "Put your final answer inside <answer> tags.\n"
    "Example: <answer>(A)</answer>"
)
_V9B_CRITIC_SYSTEM = (
    "You are a visual question answering verifier. "
    "Given an image, a question, and the user's answer, "
    "determine whether the answer is correct or incorrect. "
    "State your verdict as CORRECT or INCORRECT and briefly explain why."
)


def v9b_a1_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": _V9B_VL_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]


def v9b_f1_messages(question: str, a1: str) -> list[dict]:
    # Role-flipped: assistant carries image+question, user carries A1
    return [
        {"role": "system", "content": _V9B_CRITIC_SYSTEM},
        {
            "role": "assistant",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
        {"role": "user", "content": a1},
    ]


def v9b_a2_messages(question: str, a1: str, fb: str) -> list[dict]:
    # Stacked conversation: user=Q, assistant=A1, user=fb
    return [
        {"role": "system", "content": _V9B_VL_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
        {"role": "assistant", "content": a1},
        {"role": "user", "content": fb},
    ]


# ---------------------------------------------------------------------------
# Proposed Pattern-A eval wrapper (NEW — should MATCH training)
# ---------------------------------------------------------------------------
def patternA_a1_messages(question: str) -> list[dict]:
    return build_initial_answer_prompt(question, use_think_answer_tags=True)


def patternA_f1_messages(question: str, a1: str) -> list[dict]:
    return build_critic_prompt(question, a1, model_type="qwen2vl")


def patternA_a2_messages(question: str, a1: str, fb: str) -> list[dict]:
    return build_refiner_prompt(question, a1, fb, use_think_answer_tags=True)


# ---------------------------------------------------------------------------
# Render with Qwen chat template
# ---------------------------------------------------------------------------
def render(processor, msgs: list[dict]) -> str:
    return processor.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def diff_block(a: str, b: str, label_a: str, label_b: str) -> None:
    import difflib

    if a == b:
        print(f"  EXACT MATCH ({len(a)} chars)")
        return
    diff = list(
        difflib.unified_diff(
            a.splitlines(keepends=True),
            b.splitlines(keepends=True),
            fromfile=label_a,
            tofile=label_b,
            n=2,
        )
    )
    print(f"  DIFFERS — {sum(1 for _ in diff)} diff lines")
    sys.stdout.writelines(diff[:60])


def main() -> None:
    from transformers import AutoProcessor

    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
    print(f"Loading processor for {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)

    q = SAMPLE["question"]
    a1 = SAMPLE["a1_completion"]
    fb = SAMPLE["f1_completion"]

    pairs = [
        (
            "A1",
            v9b_a1_messages(q),
            patternA_a1_messages(q),
            build_initial_answer_prompt(q, use_think_answer_tags=True),
        ),
        (
            "F1",
            v9b_f1_messages(q, a1),
            patternA_f1_messages(q, a1),
            build_critic_prompt(q, a1, model_type="qwen2vl"),
        ),
        (
            "A2",
            v9b_a2_messages(q, a1, fb),
            patternA_a2_messages(q, a1, fb),
            build_refiner_prompt(q, a1, fb, use_think_answer_tags=True),
        ),
    ]

    for label, v9b_msgs, evalA_msgs, train_msgs in pairs:
        print("\n" + "=" * 78)
        print(f"TURN {label}")
        print("=" * 78)
        train_text = render(processor, train_msgs)
        eval_text = render(processor, evalA_msgs)
        v9b_text = render(processor, v9b_msgs)

        print(f"\n--- Training rendered ({label}) ---")
        print(train_text)
        print(f"--- end training {label} ---\n")

        print(f"[{label}] training vs proposed Pattern-A eval wrapper:")
        diff_block(train_text, eval_text, "training", "patternA_eval")

        print(f"\n[{label}] training vs v9b OLD eval wrapper:")
        diff_block(train_text, v9b_text, "training", "v9b_eval")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"THINK_ANSWER_INSTRUCTION env: {THINK_ANSWER_INSTRUCTION!r}")
    print(f"F1_VERIFIER_INSTRUCTION env:  {F1_VERIFIER_INSTRUCTION!r}")


if __name__ == "__main__":
    main()
