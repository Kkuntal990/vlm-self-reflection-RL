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

# Tag extraction patterns for <think>...</think><answer>...</answer> format
_ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_THINK_TAG_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)

# Answer extraction patterns (case-insensitive for a-f / A-F)
_MCQ_LETTER_PATTERN = re.compile(r"[A-Fa-f]")
_MCQ_STRICT_PATTERN = re.compile(r"^\s*(?:\(([A-Fa-f])\)|([A-Fa-f])[.\s]?)\s*$")
# Matches "(A)", "(b)" or standalone "A.", "b." (not inside words like "presented.")
_MCQ_OPTION_PATTERN = re.compile(r"(?:\(([A-Fa-f])\)|(?<!\w)([A-Fa-f])\.)")
# Captures "(A) Yes" or "a. Yes" → letter + answer_text
_MCQ_LETTER_AND_TEXT_PATTERN = re.compile(r"(?:\(([A-Fa-f])\)\s*|([A-Fa-f])\.\s*)(.*)", re.DOTALL)
# Matches "The answer is X" / "answer: X" / "Answer is X" / "answer:X" patterns
_MCQ_ANSWER_IS_PATTERN = re.compile(
    r"(?:the\s+)?answer\s*(?:is|:)\s*\(?([A-Fa-f])\)?", re.IGNORECASE
)
_YESNO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)
_NUMERIC_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
_NUMBER_WORDS: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15",
}
_NUMBER_WORD_PATTERN = re.compile(
    r"\b(" + "|".join(_NUMBER_WORDS.keys()) + r")\b", re.IGNORECASE
)

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


def extract_from_answer_tags(text: str) -> str:
    """Extract content from <answer>...</answer> tags.

    If <answer> tags are found, returns the inner content.
    If no tags found, returns the original text unchanged (fallback).

    This enables a clean fallback: when use_think_answer_tags=False,
    calling this function is a no-op since raw answers don't contain tags.

    Args:
        text: Raw model output, possibly containing <think> and <answer> tags.

    Returns:
        Content inside <answer> tags, or original text if no tags found.
    """
    match = _ANSWER_TAG_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def has_think_answer_tags(text: str) -> bool:
    """Check whether text contains both <think> and <answer> tags.

    Used by the format reward to verify the model followed the tag format.

    Args:
        text: Raw model output.

    Returns:
        True if both <think> and <answer> tags are present.
    """
    return bool(_THINK_TAG_PATTERN.search(text) and _ANSWER_TAG_PATTERN.search(text))


def extract_answer_from_text(
    text: str,
    answer_type: str,
    choices: str = "",
    require_answer_tag: bool = False,
) -> str:
    """Liberal extraction of answer from text for correctness scoring.

    Searches within text to find an answer — used by correctness reward,
    NOT by format reward. This intentionally recovers answers from prose
    like "The answer is B" so correctness can still be evaluated even
    when format is wrong.

    When require_answer_tag=True (answer-tag-only mode), extraction only
    works from <answer> tag content. If no tag is found, returns "" (failed)
    to prevent false positives from stray letters in prose like "A hen".

    For MCQ: extracts single letter (A-F), rejects multiple different letters.
    For YesNo: extracts "Yes" or "No", rejects hedging.
    For Numeric: extracts first number.
    For Open: returns text as-is (stripped).

    Args:
        text: Raw answer text
        answer_type: Expected answer type ("mcq", "yesno", "numeric", "open")
        choices: Optional comma-separated MCQ choices
        require_answer_tag: If True, return "" when <answer> tags are missing

    Returns:
        Normalized answer string, or empty string if extraction failed
    """
    # When answer tags required, only extract from tag content
    if require_answer_tag and not _ANSWER_TAG_PATTERN.search(text):
        return ""

    # Extract from <answer> tags first to avoid matching stray letters
    # in <think> sections (e.g., "reflectance." → false 'E' match)
    text = extract_from_answer_tags(text)
    if not text:
        return ""

    if answer_type == "mcq":
        return _extract_mcq_answer(text)
    elif answer_type == "yesno":
        return _extract_yesno_answer(text)
    elif answer_type in ("numeric", "counting"):
        return _extract_numeric_answer(text, allow_words=answer_type == "counting")
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

    # Try "The answer is X" / "answer: X" pattern
    answer_is_match = _MCQ_ANSWER_IS_PATTERN.search(text)
    if answer_is_match:
        return answer_is_match.group(1).upper()

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


def _extract_numeric_answer(text: str, allow_words: bool = False) -> str:
    """Extract numeric answer from text.

    Args:
        text: Raw answer text
        allow_words: If True, also match number words (e.g., "six" -> "6").
            Used for counting tasks where models may spell out numbers.

    Returns:
        Number string, or empty string if extraction failed
    """
    match = _NUMERIC_PATTERN.search(text)
    if match:
        return match.group(0)

    if allow_words:
        word_match = _NUMBER_WORD_PATTERN.search(text)
        if word_match:
            return _NUMBER_WORDS[word_match.group(1).lower()]

    return ""


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
