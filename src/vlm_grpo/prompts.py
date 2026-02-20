#!/usr/bin/env python3
"""
Prompt builders for GRPO self-reflection training.

Constructs prompts in conversational format for TRL's GRPOTrainer.
The model receives image + question + Answer1 and generates a single
completion containing both free-form FEEDBACK and FINAL_ANSWER.

Usage:
    from vlm_grpo.prompts import build_reflection_prompt

    messages = build_reflection_prompt(
        question="What color is the bear?",
        answer1="The bear is gray.",
        answer_type="open",
    )
"""

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

MCQ_HINT = "- Respond with ONLY the option letter (e.g., A, B, C, D). Do not include parentheses or explanation."
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
    """Build the reflection prompt in conversational format for VLM.

    Creates a two-turn conversation:
    1. User asks question (with image placeholder)
    2. Assistant provides Answer1
    3. User asks for reflection with structured output format

    The model then generates one completion containing FEEDBACK + FINAL_ANSWER.

    Args:
        question: The visual question (cleaned, no <image> tag)
        answer1: The precomputed initial answer (verified correct)
        answer_type: Expected answer type ("mcq", "yesno", "numeric", "open")
        choices: Optional comma-separated choices for MCQ

    Returns:
        List of message dicts in TRL conversational format.
        The image content is included as {"type": "image"} placeholder
        (TRL pairs this with the images column from the dataset).
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
