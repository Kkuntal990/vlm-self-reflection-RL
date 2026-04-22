#!/usr/bin/env python3
"""
Prompt builders for GRPO self-reflection training.

Provides prompt builders for three conversation flows:
1. Initial answer (A1): VL assistant generates answer to image+question
2. Feedback (F1): Critic generates feedback with role-flipped conversation
3. Refined answer (A2): VL assistant revises given raw feedback

The conversation flow matches the inference script
(self_reflective_inference_v2.py) exactly.

Prompt Override via Environment Variables:
    All system prompts can be overridden via env vars so each experiment
    yaml is self-contained and reproducible. The Python defaults match the
    v9 setup; set env vars in the k8s yaml for any other variant.

    Env vars (each overrides the matching constant):
        VL_ASSISTANT_PROMPT          → VL_ASSISTANT_SYSTEM_PROMPT
        VL_ASSISTANT_PROMPT_TAGS     → VL_ASSISTANT_SYSTEM_PROMPT_WITH_TAGS
        VL_ASSISTANT_PROMPT_ANS_TAG  → VL_ASSISTANT_SYSTEM_PROMPT_WITH_ANSWER_TAG
        FEEDBACK_CRITIC_PROMPT       → FEEDBACK_CRITIC_SYSTEM_PROMPT
        BINARY_VERIFIER_PROMPT       → BINARY_VERIFICATION_SYSTEM_PROMPT
        FEEDBACK_VERIFIER_PROMPT     → FEEDBACK_VERIFIER_SYSTEM_PROMPT

Usage:
    from vlm_grpo.prompts import (
        build_initial_answer_prompt,
        build_critic_prompt,
        build_refiner_prompt,
        build_prompt_with_completion,
    )

    # A1 generation
    a1_prompt = build_initial_answer_prompt("What color is the bear?")

    # F1 generation (role-flipped critic)
    f1_prompt = build_critic_prompt("What color is the bear?", "Gray")

    # A2 generation (raw feedback, no format forcing)
    a2_prompt = build_refiner_prompt("What color is the bear?", "Gray", "Check again.")

    # Log-prob computation (prompt + completion)
    full = build_prompt_with_completion(a1_prompt, "Gray")
"""
import os


def _prompt_from_env(env_var: str, default: str) -> str:
    """Return env-var value if set and non-empty, else default.

    Enables per-experiment prompt overrides via yaml env vars while keeping
    Python defaults as the current stable fallback.

    Args:
        env_var: Name of environment variable to check.
        default: Fallback prompt text.

    Returns:
        Prompt text (env var if set, otherwise default).
    """
    val = os.environ.get(env_var, "").strip()
    return val if val else default


# =============================================================================
# System Prompts (defaults — can be overridden via env vars in k8s yaml)
# =============================================================================

# VL Assistant system prompt for A1 and A2 generation.
VL_ASSISTANT_SYSTEM_PROMPT = _prompt_from_env(
    "VL_ASSISTANT_PROMPT",
    "You are a visual question answering assistant. "
    "Look at the image carefully and answer the question. "
    "When given feedback on your previous answer, re-examine the image "
    "and either correct your answer or keep it if you believe it is right.",
)

# Feedback critic system prompt for F1 generation.
# Matches the fire_feedback and nlf_feedback SFT training prompts.
FEEDBACK_CRITIC_SYSTEM_PROMPT = _prompt_from_env(
    "FEEDBACK_CRITIC_PROMPT",
    "You are a visual question answering critic. Given an image, a question, "
    "and the conversation history of the user's answers and prior feedback, "
    "provide constructive feedback on the user's latest answer identifying "
    "what is correct, what is incorrect, and how to improve. Ground your "
    "feedback in what is visible in the image.",
)

# Binary verification prompt for v8 mode (deprecated — no tags).
# F1 classifies the answer as CORRECT or INCORRECT with brief justification.
BINARY_VERIFICATION_SYSTEM_PROMPT = _prompt_from_env(
    "BINARY_VERIFIER_PROMPT",
    "You are a visual question answering verifier. "
    "Given an image, a question, and the user's answer, "
    "determine whether the answer is correct or incorrect. "
    "State your verdict as CORRECT or INCORRECT and briefly explain why.",
)

# v9 feedback verifier prompt with <feedback> tags for robust extraction.
# The model puts its verdict in tags, then explains. This mirrors how
# <answer> tags work for A1/A2 — deterministic extraction, no regex heuristics.
FEEDBACK_VERIFIER_SYSTEM_PROMPT = _prompt_from_env(
    "FEEDBACK_VERIFIER_PROMPT",
    "You are a visual question answering verifier. "
    "Given an image, a question, and the user's answer, "
    "determine whether the answer is correct or incorrect. "
    "Put your verdict inside <feedback> tags as either CORRECT or INCORRECT, "
    "then briefly explain why.\n"
    "Example: <feedback>INCORRECT</feedback> The answer should be (B) because "
    "the image shows a red car, not a blue one.",
)

# System prompt variant with think/answer tag instructions.
# Used for A1 and A2 generation when use_think_answer_tags=True.
VL_ASSISTANT_SYSTEM_PROMPT_WITH_TAGS = _prompt_from_env(
    "VL_ASSISTANT_PROMPT_TAGS",
    "You are a visual question answering assistant. "
    "Look at the image carefully and answer the question. "
    "When given feedback on your previous answer, re-examine the image "
    "and either correct your answer or keep it if you believe it is right.\n\n"
    "Always structure your response as: first explain your visual reasoning "
    "inside <think> tags, then give your final answer inside <answer> tags.\n"
    "Example: <think>I see a cat in the top-left corner.</think>"
    "<answer>(A)</answer>",
)

# Answer-tag-only variant. No <think> tags required — model can reason
# freely in plain text, but must wrap the final answer in <answer> tags.
# Used for v8+ where we want structured extraction without forcing
# a specific reasoning format.
VL_ASSISTANT_SYSTEM_PROMPT_WITH_ANSWER_TAG = _prompt_from_env(
    "VL_ASSISTANT_PROMPT_ANS_TAG",
    "You are a visual question answering assistant. "
    "Look at the image carefully and answer the question. "
    "When given feedback on your previous answer, re-examine the image "
    "and either correct your answer or keep it if you believe it is right.\n\n"
    "Put your final answer inside <answer> tags.\n"
    "Example: <answer>(A)</answer>",
)


# Backward-compatible aliases
REFINER_SYSTEM_PROMPT = VL_ASSISTANT_SYSTEM_PROMPT
CRITIC_SYSTEM_PROMPT = FEEDBACK_CRITIC_SYSTEM_PROMPT

# =============================================================================
# Self-Reflection Prompt Builders (matching inference script)
# =============================================================================


def build_initial_answer_prompt(
    question: str,
    use_think_answer_tags: bool = False,
    use_answer_tag_only: bool = False,
) -> list[dict]:
    """Build prompt for initial answer (A1) generation.

    Matches inference script Turn 0:
        System: VL assistant prompt
        User: [image] + question [+ tag instruction if enabled]

    Args:
        question: The visual question (cleaned, no <image> tag)
        use_think_answer_tags: If True, use <think>+<answer> tag format
        use_answer_tag_only: If True, use <answer> tag only (no <think>)

    Returns:
        List of message dicts for A1 generation
    """
    if use_think_answer_tags:
        system_prompt = VL_ASSISTANT_SYSTEM_PROMPT_WITH_TAGS
    elif use_answer_tag_only:
        system_prompt = VL_ASSISTANT_SYSTEM_PROMPT_WITH_ANSWER_TAG
    else:
        system_prompt = VL_ASSISTANT_SYSTEM_PROMPT

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]


def build_critic_prompt(
    question: str,
    answer1: str,
    answer_type: str = "open",
    choices: str = "",
    model_type: str = "llava",
    use_binary_verification: bool = False,
) -> list[dict]:
    """Build critic prompt for feedback generation with role-flipped conversation.

    Roles are flipped so the model critiques "someone else's" answer:
    - Assistant presents the question (+ image for Qwen2VL)
    - User provides the answer to be critiqued
    - Model generates feedback as the next assistant turn

    This matches the FIRE fire_feedback SFT training format and the
    eval-time self_reflective_inference_v2.py conversation structure.

    For LLaVA: image is hoisted to the system message because LLaVA's
    apply_chat_template does not reliably preserve <image> in non-user roles.

    For Qwen2.5-VL: image is placed in the assistant message since Qwen's
    processor handles images natively in any role.

    When use_binary_verification=True, the system prompt instructs the
    model to output exactly "CORRECT" or "INCORRECT" instead of open
    feedback.

    Args:
        question: The visual question (cleaned, no <image> tag)
        answer1: The initial answer to critique
        answer_type: Expected answer type (unused, kept for API compat)
        choices: Optional MCQ choices (unused, kept for API compat)
        model_type: Model family ("llava" or "qwen2vl")
        use_binary_verification: If True, use binary verification prompt

    Returns:
        List of message dicts in conversational format
    """
    critic_system = (
        FEEDBACK_VERIFIER_SYSTEM_PROMPT
        if use_binary_verification
        else FEEDBACK_CRITIC_SYSTEM_PROMPT
    )

    if model_type == "qwen2vl":
        # Qwen2.5-VL: role-flipped, image in assistant message (native support)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": critic_system}],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": answer1}],
            },
        ]
    else:
        # LLaVA: role-flipped with image hoisted to system message
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": critic_system},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": question}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": answer1}],
            },
        ]

    return messages


def build_refiner_prompt(
    question: str,
    answer1: str,
    feedback1: str,
    answer_type: str = "open",
    choices: str = "",
    use_think_answer_tags: bool = False,
    use_answer_tag_only: bool = False,
) -> list[dict]:
    """Build refiner prompt for answer refinement (A2).

    Matches inference script Turn 1+ refinement step:
        System: VL assistant prompt
        User: [image] + question
        Assistant: answer1
        User: feedback1 [+ tag instruction if enabled]

    The model generates A2 as the assistant.

    Args:
        question: The visual question (cleaned, no <image> tag)
        answer1: The initial answer being refined
        feedback1: Raw feedback from the critic (passed as-is)
        answer_type: Expected answer type (unused, kept for API compat)
        choices: Optional MCQ choices (unused, kept for API compat)
        use_think_answer_tags: If True, use <think>+<answer> tag format
        use_answer_tag_only: If True, use <answer> tag only (no <think>)

    Returns:
        List of message dicts in conversational format
    """
    if use_think_answer_tags:
        system_prompt = VL_ASSISTANT_SYSTEM_PROMPT_WITH_TAGS
    elif use_answer_tag_only:
        system_prompt = VL_ASSISTANT_SYSTEM_PROMPT_WITH_ANSWER_TAG
    else:
        system_prompt = VL_ASSISTANT_SYSTEM_PROMPT
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer1}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": feedback1}],
        },
    ]

    return messages


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
        Full message list with assistant completion appended
    """
    return prompt_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": completion}]}
    ]
