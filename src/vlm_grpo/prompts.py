#!/usr/bin/env python3
"""
Prompt builders for GRPO self-reflection training (v10+: Pattern A).

All three turns (A1, F1, A2) use a SINGLE USER MESSAGE with NO custom system
prompt. Qwen2.5-VL's chat template injects its default
"You are a helpful assistant." automatically. Instructions are embedded in
the user message text.

No role-flipping. The candidate answer and prior feedback are quoted as
text inside the user message rather than sent as assistant/user turns.

This matches the LLaVA-Critic-R1 convention (arXiv:2509.00676,
Qwen2.5-VL-7B base): the vendored EasyR1 training code builds messages as
`[{"role": "user", "content": [image, text]}]` only and relies on the
processor's chat template for the system prompt. See experiments.md for the
broader literature survey (LLaVA-Critic, Volcano, Critic-V, Critique-GRPO,
MT-Bench, CriticGPT, etc.) — none use role-flipping for same-model critique.

Env vars (override instruction text, not system prompts):
    ANSWER_TAG_INSTRUCTION      — A1/A2 answer-tag-only instruction
    THINK_ANSWER_INSTRUCTION    — A1/A2 think+answer tag instruction
    F1_VERIFIER_INSTRUCTION     — F1 verifier instruction

Usage:
    from vlm_grpo.prompts import (
        build_initial_answer_prompt,
        build_critic_prompt,
        build_refiner_prompt,
        build_prompt_with_completion,
    )

    a1 = build_initial_answer_prompt("What color is the bear?",
                                     use_answer_tag_only=True)
    f1 = build_critic_prompt("What color is the bear?", "Gray")
    a2 = build_refiner_prompt("What color is the bear?", "Gray",
                              "INCORRECT. The bear is brown.",
                              use_answer_tag_only=True)
"""

import os


def _prompt_from_env(env_var: str, default: str) -> str:
    """Return env-var value if set and non-empty, else default.

    Args:
        env_var: Name of environment variable to check.
        default: Fallback instruction text.

    Returns:
        Instruction text (env var if set, otherwise default).
    """
    val = os.environ.get(env_var, "").strip()
    return val if val else default


# =============================================================================
# User-message instruction text (v10+)
# =============================================================================
# These are NOT system prompts — they are embedded in the user message
# alongside the question / candidate / feedback. Qwen's chat template
# provides the system prompt ("You are a helpful assistant.") by default.

ANSWER_TAG_INSTRUCTION = _prompt_from_env(
    "ANSWER_TAG_INSTRUCTION",
    "Put your final answer inside <answer> tags. Example: <answer>(A)</answer>",
)

THINK_ANSWER_INSTRUCTION = _prompt_from_env(
    "THINK_ANSWER_INSTRUCTION",
    "The reasoning process MUST BE enclosed within <think> </think> tags. "
    "The final answer MUST BE put in <answer> </answer> tags.",
)

F1_VERIFIER_INSTRUCTION = _prompt_from_env(
    "F1_VERIFIER_INSTRUCTION",
    "Re-examine the image and determine whether the candidate answer is "
    "correct or incorrect. The reasoning process MUST BE enclosed within "
    "<think> </think> tags. The final verdict MUST BE put in \\boxed{} as "
    "either CORRECT or INCORRECT.",
)


# =============================================================================
# Legacy text constants (kept for scripts/preference/*.py data builders)
# =============================================================================
# Offline preference-data scripts reference these as system-message text
# when they build their own conversation structure. Not used by the live
# A1/F1/A2 builders below.

VL_ASSISTANT_SYSTEM_PROMPT = (
    "You are a visual question answering assistant. "
    "Look at the image carefully and answer the question."
)

FEEDBACK_CRITIC_SYSTEM_PROMPT = (
    "You are a visual question answering critic. Given an image, a question, "
    "and the user's answer, provide constructive feedback on the answer "
    "identifying what is correct, what is incorrect, and how to improve. "
    "Ground your feedback in what is visible in the image."
)


# =============================================================================
# Self-Reflection Prompt Builders (Pattern A — single user turn, no system)
# =============================================================================


def _user_message_with_image(text: str) -> list[dict]:
    """Build a single-user-turn message list with image + text.

    Args:
        text: The full user text (question + instructions + any embedded
              prior context).

    Returns:
        One-element message list: [{"role": "user", "content": [image, text]}]
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": text},
            ],
        }
    ]


def build_initial_answer_prompt(
    question: str,
    use_think_answer_tags: bool = False,
    use_answer_tag_only: bool = False,
) -> list[dict]:
    """Build A1 prompt — single user turn with image + question (+ tag instruction).

    Args:
        question: The visual question (cleaned, no <image> tag)
        use_think_answer_tags: If True, append <think>+<answer> tag instruction
        use_answer_tag_only: If True, append <answer>-only tag instruction

    Returns:
        One-element message list containing a single user turn.
    """
    parts = [question]
    if use_think_answer_tags:
        parts.append(THINK_ANSWER_INSTRUCTION)
    elif use_answer_tag_only:
        parts.append(ANSWER_TAG_INSTRUCTION)
    return _user_message_with_image("\n\n".join(parts))


def build_critic_prompt(
    question: str,
    answer1: str,
    answer_type: str = "open",
    choices: str = "",
    model_type: str = "qwen2vl",
) -> list[dict]:
    """Build F1 prompt — single user turn with image + Q + candidate + verifier instruction.

    No role-flipping. The candidate answer is quoted as text inside the
    user message (per LLaVA-Critic-R1, LLaVA-Critic, Volcano, Critic-V,
    Critique-GRPO, CriticGPT convention).

    Args:
        question: The visual question (cleaned, no <image> tag)
        answer1: The initial answer to verify
        answer_type: Expected answer type (unused, kept for API compat)
        choices: Optional MCQ choices (unused, kept for API compat)
        model_type: Model family (unused, kept for API compat — all families
            use the same Pattern A now)

    Returns:
        One-element message list containing a single user turn.
    """
    text = f"Question: {question}\nCandidate answer: {answer1}\n\n{F1_VERIFIER_INSTRUCTION}"
    return _user_message_with_image(text)


def build_refiner_prompt(
    question: str,
    answer1: str,
    feedback1: str,
    answer_type: str = "open",
    choices: str = "",
    use_think_answer_tags: bool = False,
    use_answer_tag_only: bool = False,
) -> list[dict]:
    """Build A2 prompt — single user turn with image + Q + prior A1 + F1 + tag instruction.

    Flattened refinement (Pattern A). A1 and F1 are quoted as text inside the
    same user turn that requests the refined answer. This matches
    Critique-GRPO's refinement template (arXiv:2506.03106) and avoids the
    stacked-conversation tag-leakage pathway.

    Args:
        question: The visual question (cleaned, no <image> tag)
        answer1: The initial answer being refined
        feedback1: Raw feedback text (passed as-is)
        answer_type: Expected answer type (unused, kept for API compat)
        choices: Optional MCQ choices (unused, kept for API compat)
        use_think_answer_tags: If True, append <think>+<answer> tag instruction
        use_answer_tag_only: If True, append <answer>-only tag instruction

    Returns:
        One-element message list containing a single user turn.
    """
    parts = [
        f"Question: {question}",
        f"Your previous answer: {answer1}",
        f"Feedback on your previous answer: {feedback1}",
    ]
    if use_think_answer_tags:
        parts.append(
            "Re-examine the image and either correct your answer or keep "
            "it if you believe it is right. " + THINK_ANSWER_INSTRUCTION
        )
    elif use_answer_tag_only:
        parts.append(
            "Re-examine the image and either correct your answer or keep "
            "it if you believe it is right. " + ANSWER_TAG_INSTRUCTION
        )
    else:
        parts.append(
            "Re-examine the image and either correct your answer or keep "
            "it if you believe it is right."
        )
    return _user_message_with_image("\n\n".join(parts))


def build_prompt_with_completion(
    prompt_messages: list[dict],
    completion: str,
) -> list[dict]:
    """Append assistant completion to a prompt for log-prob computation.

    Generic helper that works with any prompt (A1, F1, or A2).

    Args:
        prompt_messages: The prompt messages (without generation prompt)
        completion: The generated text to score

    Returns:
        Full message list with assistant completion appended.
    """
    return prompt_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": completion}]}
    ]
