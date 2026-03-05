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
# System Prompts (matching inference script exactly)
# =============================================================================

# VL Assistant system prompt (from fire_messages SFT training)
VL_ASSISTANT_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant. You should produce accurate, "
    "detailed, and grounded answers based on the image and the user's instructions. "
    "When given feedback, critique, or scores, revise your response to improve "
    "correctness, specificity, and completeness."
)

# Feedback critic system prompt (from fire_feedback SFT training)
FEEDBACK_CRITIC_SYSTEM_PROMPT = (
    "You are a vision-language critic that evaluates answers to visual questions "
    "and helps improve them. Use the image, question, and dialogue history to "
    "judge the latest answer by: - correctness and visual grounding (matches "
    "what's visible / implied), - compliance with the requested format (option "
    "letter, units, etc.), - completeness. Be conservative: confirm the answer "
    "as correct if it is consistent with the image/question and follows the "
    'required format. Only say "incorrect" when you can name a specific '
    "contradiction or missing requirement. If you are uncertain, do not "
    "guess\u2014ask to re-check one concrete detail. Write a brief natural "
    "paragraph: start with a clear verdict, give 1\u20132 grounded reasons, and "
    "(if needed) one practical next step. Keep the tone polite and encouraging."
)

# Backward-compatible aliases
REFINER_SYSTEM_PROMPT = VL_ASSISTANT_SYSTEM_PROMPT
CRITIC_SYSTEM_PROMPT = FEEDBACK_CRITIC_SYSTEM_PROMPT

# =============================================================================
# Legacy single-trajectory prompts (kept for backward compatibility)
# =============================================================================

REFLECTION_SYSTEM_PROMPT = (
    "You are a helpful vision-language assistant that can reflect on your answers. "
    "When asked to reflect, evaluate your previous answer against the visual evidence, "
    "provide honest feedback, and give a corrected answer if needed."
)

REFLECTION_INSTRUCTION_TEMPLATE = (
    "Please reflect on your answer above. "
    "First, provide feedback on whether the answer is correct based on the image. "
    "Then provide your final answer.\n\n"
    "You MUST use this exact format:\n"
    "FEEDBACK:\n<your feedback here>\n"
    "FINAL_ANSWER:\n<your final answer here>\n\n"
    "Rules for FINAL_ANSWER:\n{answer_type_hint}"
)

MCQ_HINT = (
    "- Respond with ONLY the option letter (e.g., A, B, C, D). "
    "Do not include parentheses or explanation."
)
YESNO_HINT = "- Respond with ONLY 'Yes' or 'No'. No hedging or qualification."
NUMERIC_HINT = "- Respond with ONLY the number. No units or explanation."
OPEN_HINT = "- Give a brief, direct answer."


def build_answer_type_hint(answer_type: str, choices: str = "") -> str:
    """Build answer type hint for the reflection instruction.

    Args:
        answer_type: Expected answer type ("mcq", "yesno", "numeric", "open")
        choices: Optional comma-separated choices for MCQ

    Returns:
        Hint string about expected answer format
    """
    if answer_type == "mcq":
        hint = MCQ_HINT
        if choices:
            hint += f"\n- Available choices: {choices}"
        return hint
    elif answer_type == "yesno":
        return YESNO_HINT
    elif answer_type == "numeric":
        return NUMERIC_HINT
    else:
        return OPEN_HINT


def build_reflection_prompt(
    question: str,
    answer1: str,
    answer_type: str = "open",
    choices: str = "",
) -> list[dict]:
    """Build the legacy reflection prompt (FEEDBACK/FINAL_ANSWER format).

    Kept for backward compatibility with train_grpo_rw.py.

    Args:
        question: The visual question (cleaned, no <image> tag)
        answer1: The precomputed initial answer (verified correct)
        answer_type: Expected answer type ("mcq", "yesno", "numeric", "open")
        choices: Optional comma-separated choices for MCQ

    Returns:
        List of message dicts in TRL conversational format.
    """
    answer_type_hint = build_answer_type_hint(answer_type, choices)
    reflection_instruction = REFLECTION_INSTRUCTION_TEMPLATE.format(
        answer_type_hint=answer_type_hint
    )

    messages = [
        {
            "role": "system",
            "content": REFLECTION_SYSTEM_PROMPT,
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
            "content": answer1,
        },
        {
            "role": "user",
            "content": reflection_instruction,
        },
    ]

    return messages


# =============================================================================
# Self-Reflection Prompt Builders (matching inference script)
# =============================================================================


def build_initial_answer_prompt(question: str) -> list[dict]:
    """Build prompt for initial answer (A1) generation.

    Matches inference script Turn 0:
        System: VL assistant prompt
        User: [image] + question

    Args:
        question: The visual question (cleaned, no <image> tag)

    Returns:
        List of message dicts for A1 generation
    """
    return [
        {
            "role": "system",
            "content": VL_ASSISTANT_SYSTEM_PROMPT,
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
) -> list[dict]:
    """Build role-flipped critic prompt for feedback generation.

    Matches inference script Turn 1+ feedback step:
        System: critic prompt
        Assistant: [image] + question   (role-flipped!)
        User: answer1

    The model generates feedback as the assistant.

    Args:
        question: The visual question (cleaned, no <image> tag)
        answer1: The initial answer to critique
        answer_type: Expected answer type (unused, kept for API compat)
        choices: Optional MCQ choices (unused, kept for API compat)

    Returns:
        List of message dicts in conversational format
    """
    messages = [
        {
            "role": "system",
            "content": FEEDBACK_CRITIC_SYSTEM_PROMPT,
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
            "content": answer1,
        },
    ]

    return messages


def build_refiner_prompt(
    question: str,
    answer1: str,
    feedback1: str,
    answer_type: str = "open",
    choices: str = "",
) -> list[dict]:
    """Build refiner prompt for answer refinement (A2).

    Matches inference script Turn 1+ refinement step:
        System: VL assistant prompt
        User: [image] + question
        Assistant: answer1
        User: feedback1  (raw, no format-forcing instructions)

    The model generates A2 as the assistant.

    Args:
        question: The visual question (cleaned, no <image> tag)
        answer1: The initial answer being refined
        feedback1: Raw feedback from the critic (passed as-is)
        answer_type: Expected answer type (unused, kept for API compat)
        choices: Optional MCQ choices (unused, kept for API compat)

    Returns:
        List of message dicts in conversational format
    """
    messages = [
        {
            "role": "system",
            "content": VL_ASSISTANT_SYSTEM_PROMPT,
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
            "content": answer1,
        },
        {
            "role": "user",
            "content": feedback1,
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
    return prompt_messages + [{"role": "assistant", "content": completion}]
