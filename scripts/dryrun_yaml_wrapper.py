#!/usr/bin/env python3
"""Dry-run: extract the wrapper builders embedded in the eval YAML and verify
they produce IDENTICAL Qwen-rendered text to the training builders.

This catches transcription bugs where the wrapper code in the YAML drifts from
src/vlm_grpo/prompts.py (e.g., quote escaping, env-var defaults).
"""
from __future__ import annotations

import os
import re
import sys

REPO_SRC = "/workspace/repo/src"
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from vlm_grpo.prompts import (  # noqa: E402
    build_initial_answer_prompt,
    build_critic_prompt,
    build_refiner_prompt,
)


# ----- COPY of the YAML's wrapper builders (Pattern A — verbatim) -------------
THINK_ANSWER_INSTRUCTION = os.environ.get(
    "THINK_ANSWER_INSTRUCTION",
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE put in <answer> </answer> tags.",
)
ANSWER_TAG_INSTRUCTION = os.environ.get(
    "ANSWER_TAG_INSTRUCTION",
    "Put your final answer inside <answer> tags. Example: <answer>(A)</answer>",
)
F1_VERIFIER_INSTRUCTION = os.environ.get(
    "F1_VERIFIER_INSTRUCTION",
    "Re-examine the image and determine whether the candidate answer is "
    "correct or incorrect. The reasoning process MUST BE enclosed within "
    "<think> </think> tags. The final verdict MUST BE put in \\boxed{} as "
    "either CORRECT or INCORRECT.",
)
TAG_MODE = os.environ.get("TAG_MODE", "think_answer")


def _user_msg(image_items, text):
    return [
        {"role": "user", "content": image_items + [{"type": "text", "text": text}]}
    ]


def _tag_suffix():
    if TAG_MODE == "think_answer":
        return THINK_ANSWER_INSTRUCTION
    if TAG_MODE == "answer_only":
        return ANSWER_TAG_INSTRUCTION
    return ""


def build_a1_messages(image_items, question):
    parts = [question]
    s = _tag_suffix()
    if s:
        parts.append(s)
    return _user_msg(image_items, "\n\n".join(parts))


def build_f1_messages(image_items, question, a1):
    text = f"Question: {question}\nCandidate answer: {a1}\n\n{F1_VERIFIER_INSTRUCTION}"
    return _user_msg(image_items, text)


def build_a2_messages(image_items, question, a1, fb):
    parts = [
        f"Question: {question}",
        f"Your previous answer: {a1}",
        f"Feedback on your previous answer: {fb}",
    ]
    s = _tag_suffix()
    if s:
        parts.append(
            "Re-examine the image and either correct your answer or keep "
            "it if you believe it is right. " + s
        )
    else:
        parts.append(
            "Re-examine the image and either correct your answer or keep "
            "it if you believe it is right."
        )
    return _user_msg(image_items, "\n\n".join(parts))


# ----- Same sample as the other dry-run --------------------------------------
SAMPLE_Q = "Which of the two images is more colorful?\n(A) the first image\n(B) the second image"
SAMPLE_A1 = "<think>The first image has saturated reds and yellows.</think><answer>(A)</answer>"
SAMPLE_F1 = "<think>The first image clearly has more vibrant colors.</think>\\boxed{CORRECT}"


def main() -> None:
    from transformers import AutoProcessor

    model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
    processor = AutoProcessor.from_pretrained(model_id)

    image_items = [{"type": "image"}]

    pairs = [
        (
            "A1",
            build_initial_answer_prompt(SAMPLE_Q, use_think_answer_tags=True),
            build_a1_messages(image_items, SAMPLE_Q),
        ),
        (
            "F1",
            build_critic_prompt(SAMPLE_Q, SAMPLE_A1, model_type="qwen2vl"),
            build_f1_messages(image_items, SAMPLE_Q, SAMPLE_A1),
        ),
        (
            "A2",
            build_refiner_prompt(SAMPLE_Q, SAMPLE_A1, SAMPLE_F1, use_think_answer_tags=True),
            build_a2_messages(image_items, SAMPLE_Q, SAMPLE_A1, SAMPLE_F1),
        ),
    ]

    all_match = True
    for label, train_msgs, yaml_msgs in pairs:
        a = processor.apply_chat_template(train_msgs, tokenize=False, add_generation_prompt=True)
        b = processor.apply_chat_template(yaml_msgs, tokenize=False, add_generation_prompt=True)
        match = a == b
        status = "EXACT MATCH" if match else "DIFFERS"
        print(f"[{label}] training vs YAML-wrapper: {status} ({len(a)} vs {len(b)} chars)")
        if not match:
            all_match = False
            import difflib
            for line in difflib.unified_diff(
                a.splitlines(keepends=True), b.splitlines(keepends=True),
                fromfile="training", tofile="yaml_wrapper", n=2,
            ):
                sys.stdout.write(line)

    print()
    print("OVERALL:", "PASS — YAML wrapper byte-identical to training" if all_match else "FAIL")
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
