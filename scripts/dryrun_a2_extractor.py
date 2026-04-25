#!/usr/bin/env python3
"""Verify the YAML's _extract_mcq_letter matches training's
extract_answer_from_text(text, "mcq", strict=True) on realistic A2 outputs."""
from __future__ import annotations

import re
import sys

REPO_SRC = "/workspace/repo/src"
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from vlm_grpo.trajectory import extract_answer_from_text  # noqa: E402


# === Eval-YAML extractor (verbatim copy from the YAML) =====================
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
_MCQ_STRICT_RE = re.compile(r"^\s*(?:\(([A-Fa-f])\)|([A-Fa-f])[.\s]?)\s*$")
_MCQ_OPTION_RE = re.compile(r"(?:\(([A-Fa-f])\)|(?<!\w)([A-Fa-f])\.)")
_MCQ_ANSWER_IS_RE = re.compile(
    r"(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-Fa-f])\)?", re.IGNORECASE
)
_MCQ_LETTER_RE = re.compile(r"[A-Fa-f]")


def _extract_mcq_letter(response: str) -> str:
    m = _ANSWER_TAG_RE.search(response)
    if not m:
        return ""
    payload = m.group(1).strip()
    if not payload:
        return ""
    sm = _MCQ_STRICT_RE.match(payload)
    if sm:
        return (sm.group(1) or sm.group(2)).upper()
    om = _MCQ_OPTION_RE.search(payload)
    if om:
        return (om.group(1) or om.group(2)).upper()
    aim = _MCQ_ANSWER_IS_RE.search(payload)
    if aim:
        return aim.group(1).upper()
    letters = _MCQ_LETTER_RE.findall(payload.upper())
    if not letters:
        return ""
    if len(set(letters)) > 1:
        return ""
    return letters[0]


# === Test cases — realistic A2 outputs =====================================
CASES = [
    # (label, A2_output, expected_letter)
    ("clean A in tags", "<think>The image shows 3 dogs.</think><answer>(A)</answer>", "A"),
    ("bare letter in tags", "<answer>B</answer>", "B"),
    ("letter w/ period", "<answer>C.</answer>", "C"),
    ("verbose payload", "<answer>(B) three blue floats</answer>", "B"),
    ("F option (LIVR uses A-F)", "<answer>(F)</answer>", "F"),
    ("answer-is phrasing", "<answer>The answer is (A)</answer>", "A"),
    ("hedging — should reject", "<answer>(A) or (B)</answer>", ""),
    ("hedging same letter twice — accept", "<answer>(A) clearly A</answer>", "A"),
    ("no <answer> tag — format fail", "I think the answer is (B)", ""),
    ("empty answer tag", "<answer></answer>", ""),
    ("only think, no answer", "<think>Looks like (C).</think>", ""),
    ("stray letter in think + valid answer",
     "<think>option E was wrong, (A) is right</think><answer>A</answer>", "A"),
    ("answer tag with garbage", "<answer>not sure</answer>", ""),
    ("lowercase letter", "<answer>(b)</answer>", "B"),
]


def main() -> None:
    print(f"{'label':<48} {'eval':<6} {'training':<10} {'expected':<10} {'match'}")
    print("-" * 90)
    pass_n = 0
    for label, text, expected in CASES:
        got_eval = _extract_mcq_letter(text)
        got_train = extract_answer_from_text(text, "mcq", strict=True)
        match = (got_eval == got_train) and (got_eval == expected)
        if match:
            pass_n += 1
        print(
            f"{label:<48} {got_eval!r:<6} {got_train!r:<10} {expected!r:<10} "
            f"{'PASS' if match else 'FAIL'}"
        )
    print()
    print(f"OVERALL: {pass_n}/{len(CASES)} cases pass — eval ≡ training ≡ expected")
    sys.exit(0 if pass_n == len(CASES) else 1)


if __name__ == "__main__":
    main()
