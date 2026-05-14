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
    THINK_ANSWER_INSTRUCTION    — A1/A2 think+answer tag instruction
    F1_VERIFIER_INSTRUCTION     — F1 verifier instruction

Usage:
    from vlm_grpo.prompts import (
        build_initial_answer_prompt,
        build_critic_prompt,
        build_refiner_prompt,
        build_prompt_with_completion,
    )

    a1 = build_initial_answer_prompt("What color is the bear?")
    f1 = build_critic_prompt("What color is the bear?", "Gray")
    a2 = build_refiner_prompt("What color is the bear?", "Gray",
                              "INCORRECT. The bear is brown.")
"""

import os
import re


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

# F1 role prefix prepended to the critic user message. Empty default keeps
# legacy YAMLs unchanged; new YAMLs set this env var explicitly to engage
# the verifier-role framing (literature: PAG arXiv:2506.10406 "model
# collapse" mitigation via role separation; Critic-V arXiv:2411.18203).
F1_ROLE_PREFIX = _prompt_from_env("F1_ROLE_PREFIX", "")

# A2 role prefix prepended to the refiner user message. Same role-isolation
# rationale as F1_ROLE_PREFIX. Empty default = legacy behaviour.
A2_ROLE_PREFIX = _prompt_from_env("A2_ROLE_PREFIX", "")

# A2 reviser instruction text (appears AFTER the verdict-hoisted feedback
# block and BEFORE THINK_ANSWER_INSTRUCTION). Defaults to the legacy
# permissive wording; new YAMLs override with deterministic "re-examine
# independently" wording for both selective-revision and no-gate variants.
A2_REVISER_INSTRUCTION = _prompt_from_env(
    "A2_REVISER_INSTRUCTION",
    "Re-examine the image and either correct your answer or keep it if you believe it is right.",
)


# =============================================================================
# F1 verdict extraction (used by A2 prompt builder to hoist verdict to top)
# =============================================================================

_BOXED_VERDICT_RE = re.compile(r"\\boxed\{\s*(CORRECT|INCORRECT)\s*\}", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_f1_verdict_and_reasoning(feedback_text: str) -> tuple[str, str]:
    """Extract the F1 verdict and reasoning from raw F1 text.

    The F1 model emits free-form text that ends with `\\boxed{CORRECT}` or
    `\\boxed{INCORRECT}` after a `<think>...</think>` block. To make the
    verdict salient to the A2 refiner (literature: Huang 2023 arXiv:2310.01798,
    Kamoi TACL 2024 arXiv:2406.01297 identify feedback salience as the
    refiner-side bottleneck), we extract both pieces and hoist them.

    Args:
        feedback_text: Raw F1 text as emitted by the model.

    Returns:
        ``(verdict, reasoning)`` where ``verdict`` is one of
        ``"CORRECT"`` / ``"INCORRECT"`` / ``"MALFORMED"`` (case preserved
        as uppercase) and ``reasoning`` is the contents of the first
        ``<think>...</think>`` block, or the full feedback text if no
        ``<think>`` block is found.
    """
    m = _BOXED_VERDICT_RE.search(feedback_text)
    verdict = m.group(1).upper() if m else "MALFORMED"
    t = _THINK_BLOCK_RE.search(feedback_text)
    reasoning = t.group(1).strip() if t else feedback_text.strip()
    return verdict, reasoning


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


def build_initial_answer_prompt(question: str) -> list[dict]:
    """Build A1 prompt — single user turn with image + question + think/answer tag instruction.

    Args:
        question: The visual question (cleaned, no <image> tag)

    Returns:
        One-element message list containing a single user turn.
    """
    return _user_message_with_image(f"{question}\n\n{THINK_ANSWER_INSTRUCTION}")


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

    When ``F1_ROLE_PREFIX`` env var is set, the role-prefix string is
    prepended to the user message to combat the same-weight in-loop role
    confusion documented in PAG (arXiv:2506.10406 "model collapse").

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
    parts: list[str] = []
    if F1_ROLE_PREFIX:
        parts.append(F1_ROLE_PREFIX)
    parts.extend(
        [
            f"Question: {question}",
            f"Candidate answer: {answer1}",
            F1_VERIFIER_INSTRUCTION,
        ]
    )
    return _user_message_with_image("\n\n".join(parts))


def build_refiner_prompt(
    question: str,
    answer1: str,
    feedback1: str,
    answer_type: str = "open",
    choices: str = "",
) -> list[dict]:
    """Build A2 prompt — single user turn with image + Q + prior A1 + F1 + tag instruction.

    Flattened refinement (Pattern A). A1 and F1 are quoted as text inside the
    same user turn that requests the refined answer. This matches
    Critique-GRPO's refinement template (arXiv:2506.03106) and avoids the
    stacked-conversation tag-leakage pathway.

    The F1 verdict (``\\boxed{CORRECT|INCORRECT}``) and reasoning are
    extracted from ``feedback1`` and hoisted to the top of the feedback
    block. This makes the verdict salient to the refiner (literature: Huang
    2023 arXiv:2310.01798, Kamoi TACL 2024 arXiv:2406.01297). When the
    F1 text is malformed (no ``\\boxed{}`` parseable), the full raw F1
    text is shown as the reasoning (legacy fallback).

    When ``A2_ROLE_PREFIX`` env var is set, the role-prefix string is
    prepended (role-isolation per PAG arXiv:2506.10406).
    ``A2_REVISER_INSTRUCTION`` overrides the default "either correct or
    keep" wording — new variants set it to a deterministic "independently
    determine" wording that works for both selective-revision (gate ON)
    and no-gate setups.

    Args:
        question: The visual question (cleaned, no <image> tag)
        answer1: The initial answer being refined
        feedback1: Raw F1 text (verdict + reasoning extracted internally)
        answer_type: Expected answer type (unused, kept for API compat)
        choices: Optional MCQ choices (unused, kept for API compat)

    Returns:
        One-element message list containing a single user turn.
    """
    verdict, reasoning = extract_f1_verdict_and_reasoning(feedback1)
    parts: list[str] = []
    if A2_ROLE_PREFIX:
        parts.append(A2_ROLE_PREFIX)
    parts.append(f"Question: {question}")
    parts.append(f"Verifier verdict: \\boxed{{{verdict}}}")
    parts.append(f"Verifier reasoning: {reasoning}")
    parts.append(f"Your previous answer: {answer1}")
    parts.append(A2_REVISER_INSTRUCTION + " " + THINK_ANSWER_INSTRUCTION)
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
