#!/usr/bin/env python3
"""
Prompt builders for GRPO self-reflection training.

Provides prompt builders for three conversation flows:
1. Initial answer (A1): VL assistant generates answer to image+question
2. Feedback (F1): Critic generates feedback with role-flipped conversation
3. Refined answer (A2): VL assistant revises given raw feedback

The conversation flow matches the inference script
(self_reflective_inference_v2.py) exactly.

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

# =============================================================================
# System Prompts
# =============================================================================

# VL Assistant system prompt for A1 and A2 generation.
VL_ASSISTANT_SYSTEM_PROMPT = (
    "You are a visual question answering assistant. "
    "Look at the image carefully and answer the question. "
    "When given feedback on your previous answer, re-examine the image "
    "and either correct your answer or keep it if you believe it is right."
)

# Feedback critic system prompt for F1 generation.
# Matches the fire_feedback and nlf_feedback SFT training prompts.
FEEDBACK_CRITIC_SYSTEM_PROMPT = (
    "You are a visual question answering critic. Given an image, a question, "
    "and the conversation history of the user's answers and prior feedback, "
    "provide constructive feedback on the user's latest answer identifying "
    "what is correct, what is incorrect, and how to improve. Ground your "
    "feedback in what is visible in the image."
)

# System prompt variant with think/answer tag instructions.
# Used for A1 and A2 generation when use_think_answer_tags=True.
VL_ASSISTANT_SYSTEM_PROMPT_WITH_TAGS = (
    "You are a visual question answering assistant. "
    "Look at the image carefully and answer the question. "
    "When given feedback on your previous answer, re-examine the image "
    "and either correct your answer or keep it if you believe it is right.\n\n"
    "Always structure your response as: first explain your visual reasoning "
    "inside <think> tags, then give your final answer inside <answer> tags.\n"
    "Example: <think>I see a cat in the top-left corner.</think>"
    "<answer>(A)</answer>"
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
) -> list[dict]:
    """Build prompt for initial answer (A1) generation.

    Matches inference script Turn 0:
        System: VL assistant prompt
        User: [image] + question [+ tag instruction if enabled]

    Args:
        question: The visual question (cleaned, no <image> tag)
        use_think_answer_tags: If True, append tag format instruction

    Returns:
        List of message dicts for A1 generation
    """
    system_prompt = (
        VL_ASSISTANT_SYSTEM_PROMPT_WITH_TAGS if use_think_answer_tags
        else VL_ASSISTANT_SYSTEM_PROMPT
    )

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

    Args:
        question: The visual question (cleaned, no <image> tag)
        answer1: The initial answer to critique
        answer_type: Expected answer type (unused, kept for API compat)
        choices: Optional MCQ choices (unused, kept for API compat)
        model_type: Model family ("llava" or "qwen2vl")

    Returns:
        List of message dicts in conversational format
    """
    if model_type == "qwen2vl":
        # Qwen2.5-VL: role-flipped, image in assistant message (native support)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": FEEDBACK_CRITIC_SYSTEM_PROMPT}],
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
                    {"type": "text", "text": FEEDBACK_CRITIC_SYSTEM_PROMPT},
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
        use_think_answer_tags: If True, append tag format instruction to feedback

    Returns:
        List of message dicts in conversational format
    """
    system_prompt = (
        VL_ASSISTANT_SYSTEM_PROMPT_WITH_TAGS if use_think_answer_tags
        else VL_ASSISTANT_SYSTEM_PROMPT
    )
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
