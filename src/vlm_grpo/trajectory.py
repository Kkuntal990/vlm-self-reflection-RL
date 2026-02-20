#!/usr/bin/env python3
"""
Parse GRPO completion trajectories with FEEDBACK/FINAL_ANSWER markers.

This module handles extraction of structured output from model completions,
including answer normalization and anti-reward-hacking validation.

Usage:
    from vlm_grpo.trajectory import parse_trajectory, extract_answer_from_text

    traj = parse_trajectory("FEEDBACK:\\nLooks correct.\\nFINAL_ANSWER:\\nA")
    assert traj.parse_success
    assert traj.feedback == "Looks correct."
    assert traj.final_answer == "A"

    answer = extract_answer_from_text("(A) Yes", "mcq")
    assert answer == "A"
"""

import re
from dataclasses import asdict, dataclass

# Marker patterns (case-insensitive, with optional whitespace)
_FEEDBACK_PATTERN = re.compile(r"FEEDBACK\s*:\s*\n?", re.IGNORECASE)
_FINAL_ANSWER_PATTERN = re.compile(r"FINAL_ANSWER\s*:\s*\n?", re.IGNORECASE)

# Answer extraction patterns
_MCQ_LETTER_PATTERN = re.compile(r"[A-F]")
_MCQ_STRICT_PATTERN = re.compile(r"^\s*\(?([A-F])\)?\s*$")
_YESNO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_NUMERIC_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")

# Hedging detection patterns for yes/no answers
_HEDGING_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bmaybe\b",
        r"\bpossibly\b",
        r"\bperhaps\b",
        r"\bit could be\b",
        r"\bit might be\b",
        r"\bI think\b",
        r"\bI believe\b",
        r"\bit seems\b",
        r"\bprobably\b",
        r"\bnot sure\b",
        r"\bnot certain\b",
        r"\blikely\b",
        r"\bunlikely\b",
    ]
]


@dataclass
class ParsedTrajectory:
    """A parsed completion trajectory.

    Attributes:
        feedback: Extracted feedback text (stripped)
        final_answer: Extracted final answer text (stripped)
        raw_completion: Original completion text
        has_feedback_marker: Whether FEEDBACK: marker was found
        has_final_answer_marker: Whether FINAL_ANSWER: marker was found
        parse_success: Whether both markers were found and parsed
    """

    feedback: str
    final_answer: str
    raw_completion: str
    has_feedback_marker: bool
    has_final_answer_marker: bool
    parse_success: bool

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def parse_trajectory(completion: str) -> ParsedTrajectory:
    """Parse a completion string to extract FEEDBACK and FINAL_ANSWER.

    Handles various edge cases:
    - Missing markers (returns empty strings, parse_success=False)
    - Extra whitespace around markers
    - Multiple FINAL_ANSWER markers (uses last one)
    - FEEDBACK without FINAL_ANSWER and vice versa

    Args:
        completion: Raw completion text from the model

    Returns:
        ParsedTrajectory with extracted components
    """
    feedback = ""
    final_answer = ""
    has_feedback = False
    has_final_answer = False

    # Find all marker positions
    feedback_matches = list(_FEEDBACK_PATTERN.finditer(completion))
    final_answer_matches = list(_FINAL_ANSWER_PATTERN.finditer(completion))

    has_feedback = len(feedback_matches) > 0
    has_final_answer = len(final_answer_matches) > 0

    if has_feedback and has_final_answer:
        # Use first FEEDBACK marker and last FINAL_ANSWER marker
        fb_match = feedback_matches[0]
        fa_match = final_answer_matches[-1]

        fb_start = fb_match.end()
        fa_start = fa_match.end()

        # Feedback is between FEEDBACK: and FINAL_ANSWER:
        if fb_start < fa_match.start():
            feedback = completion[fb_start : fa_match.start()].strip()
        else:
            # FINAL_ANSWER comes before FEEDBACK (malformed but handle gracefully)
            feedback = completion[fb_start:].strip()

        # Final answer is everything after the last FINAL_ANSWER:
        final_answer = completion[fa_start:].strip()

    elif has_feedback and not has_final_answer:
        fb_match = feedback_matches[0]
        feedback = completion[fb_match.end() :].strip()

    elif has_final_answer and not has_feedback:
        fa_match = final_answer_matches[-1]
        final_answer = completion[fa_match.end() :].strip()

    parse_success = has_feedback and has_final_answer

    return ParsedTrajectory(
        feedback=feedback,
        final_answer=final_answer,
        raw_completion=completion,
        has_feedback_marker=has_feedback,
        has_final_answer_marker=has_final_answer,
        parse_success=parse_success,
    )


def extract_completion_text(completion: list[dict] | str) -> str:
    """Extract plain text from TRL completion format.

    TRL passes completions as list[dict] for conversational format
    (e.g., [{"role": "assistant", "content": "..."}]) or str for
    standard format.

    Args:
        completion: Completion in TRL format

    Returns:
        Plain text string
    """
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        # Conversational format: extract assistant content
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                # Handle structured content (list of text/image dicts)
                if isinstance(content, list):
                    texts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            texts.append(item.get("text", ""))
                        elif isinstance(item, str):
                            texts.append(item)
                    return " ".join(texts)
        # Fallback: join all content
        parts = []
        for msg in completion:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
        return " ".join(parts)

    return str(completion)


def extract_answer_from_text(
    text: str,
    answer_type: str,
    choices: str = "",
) -> str:
    """Extract a normalized answer from text given the expected type.

    For MCQ: extracts single letter (A-F), rejects multiple different letters.
    For YesNo: extracts "Yes" or "No", rejects hedging.
    For Numeric: extracts first number.
    For Open: returns text as-is (stripped).

    Args:
        text: Raw answer text
        answer_type: Expected answer type ("mcq", "yesno", "numeric", "open")
        choices: Optional comma-separated MCQ choices

    Returns:
        Normalized answer string, or empty string if extraction failed
    """
    text = text.strip()
    if not text:
        return ""

    if answer_type == "mcq":
        return _extract_mcq_answer(text)
    elif answer_type == "yesno":
        return _extract_yesno_answer(text)
    elif answer_type == "numeric":
        return _extract_numeric_answer(text)
    else:
        return text


def _extract_mcq_answer(text: str) -> str:
    """Extract MCQ answer letter from text.

    Anti-hacking: rejects if multiple different letters are found,
    which could indicate the model is hedging across options.

    Args:
        text: Raw answer text

    Returns:
        Single uppercase letter (A-F) or empty string if extraction failed
    """
    # Try strict match first (just the letter, optionally with parens)
    strict_match = _MCQ_STRICT_PATTERN.match(text)
    if strict_match:
        return strict_match.group(1).upper()

    # Find all MCQ letters in the text
    letters = _MCQ_LETTER_PATTERN.findall(text.upper())
    if not letters:
        return ""

    # Anti-hacking: reject if multiple DIFFERENT letters found
    unique_letters = set(letters)
    if len(unique_letters) > 1:
        return ""

    return letters[0]


def _extract_yesno_answer(text: str) -> str:
    """Extract yes/no answer from text.

    Anti-hacking: rejects if hedging language is detected.

    Args:
        text: Raw answer text

    Returns:
        "Yes" or "No", or empty string if extraction failed or hedging detected
    """
    # Check for hedging first
    if detect_hedging(text):
        return ""

    match = _YESNO_PATTERN.search(text)
    if not match:
        return ""

    answer = match.group(1).lower()
    return "Yes" if answer == "yes" else "No"


def _extract_numeric_answer(text: str) -> str:
    """Extract numeric answer from text.

    Args:
        text: Raw answer text

    Returns:
        Number string, or empty string if extraction failed
    """
    match = _NUMERIC_PATTERN.search(text)
    if not match:
        return ""
    return match.group(0)


def detect_hedging(text: str) -> bool:
    """Detect hedging language in answers.

    Checks for patterns like "maybe", "possibly", "I think",
    "it seems" that indicate uncertainty.

    Args:
        text: Answer text to check

    Returns:
        True if hedging detected
    """
    for pattern in _HEDGING_PATTERNS:
        if pattern.search(text):
            return True
    return False
