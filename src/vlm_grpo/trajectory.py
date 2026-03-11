#!/usr/bin/env python3
"""
Answer extraction and normalization for VLM completions.

Handles extraction of answers from model completions, answer normalization,
and anti-reward-hacking validation.

Usage:
    from vlm_grpo.trajectory import extract_answer_from_text, normalize_answer

    answer = extract_answer_from_text("(A) Yes", "mcq")
    assert answer == "A"

    normalized = normalize_answer("(A)")
    assert normalized == "a"
"""

import re

# Answer extraction patterns
_MCQ_LETTER_PATTERN = re.compile(r"[A-F]")
_MCQ_STRICT_PATTERN = re.compile(r"^\s*(?:\(([A-F])\)|([A-F])\.)\s*$")
# Matches "(A)", "(B)" or "A.", "B." at word boundary
_MCQ_OPTION_PATTERN = re.compile(r"(?:\(([A-F])\)|([A-F])\.)")
# Captures "(A) Yes" or "A. Yes" → letter + answer_text
_MCQ_LETTER_AND_TEXT_PATTERN = re.compile(r"(?:\(([A-F])\)\s*|([A-F])\.\s*)(.*)", re.DOTALL)
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


def normalize_answer(text: str) -> str:
    """Minimal surface-level normalization of raw answer text.

    Only cleans surface noise — does NOT search for answers within text.

    Normalization steps:
        1. Strip leading/trailing whitespace
        2. Lowercase
        3. Remove trailing punctuation (. , ; :)
        4. Strip wrapping parentheses: "(a)" -> "a"
        5. Strip trailing parenthesis: "a)" -> "a"

    Does NOT do:
        - Search for letters/numbers within prose ("the answer is b" stays as-is)
        - Convert number words ("three" stays "three")
        - Remove filler phrases

    Args:
        text: Raw answer text

    Returns:
        Surface-cleaned lowercase text
    """
    text = text.strip()
    text = text.lower()
    # Remove trailing punctuation
    text = text.rstrip(".,;:")
    text = text.strip()
    # Unwrap parentheses: "(a)" -> "a", "a)" -> "a"
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    elif text.endswith(")"):
        text = text[:-1].strip()
    return text


def extract_answer_from_text(
    text: str,
    answer_type: str,
    choices: str = "",
) -> str:
    """Liberal extraction of answer from text for correctness scoring.

    Searches within text to find an answer — used by correctness reward,
    NOT by format reward. This intentionally recovers answers from prose
    like "The answer is B" so correctness can still be evaluated even
    when format is wrong.

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
    # Try strict match first (just the letter: "A", "(A)", "A.")
    strict_match = _MCQ_STRICT_PATTERN.match(text)
    if strict_match:
        letter = strict_match.group(1) or strict_match.group(2)
        return letter.upper()

    # Try option pattern: "(A)" / "(B)" or "A." / "B." — handles "(A) Yes", "B. 24"
    option_match = _MCQ_OPTION_PATTERN.search(text)
    if option_match:
        letter = option_match.group(1) or option_match.group(2)
        return letter.upper()

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


def extract_mcq_letter_and_text(text: str) -> tuple[str, str]:
    """Extract both the option letter and the answer text from MCQ response.

    For "(A) Yes" returns ("A", "Yes"). For "B" returns ("B", "").
    Used by verify_answer to check GT against both the letter and text.

    Args:
        text: Raw answer text

    Returns:
        Tuple of (letter, answer_text). Either may be empty string.
    """
    text = text.strip()
    if not text:
        return ("", "")

    match = _MCQ_LETTER_AND_TEXT_PATTERN.search(text)
    if match:
        letter = (match.group(1) or match.group(2)).upper()
        answer_text = match.group(3).strip()
        return (letter, answer_text)

    # Fall back to just extracting the letter
    letter = _extract_mcq_answer(text)
    return (letter, "")


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
