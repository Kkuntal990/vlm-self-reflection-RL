#!/usr/bin/env python3
"""Verify VLMEvalKit's can_infer scores raw A2 outputs from BOTH base and
our trained model identically — proves the wrapper is fair to base."""
from __future__ import annotations

import sys

sys.path.insert(0, "/tmp/VLMEvalKit")

# Import can_infer module directly to avoid vlmeval package __init__ side-effects
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "matching_util", "/tmp/VLMEvalKit/vlmeval/utils/matching_util.py"
)
_mod = importlib.util.module_from_spec(_spec)
# Stub out vlmeval.smp.log to avoid importing the heavy package
import types  # noqa: E402

_smp = types.ModuleType("vlmeval.smp.log")
_smp.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules["vlmeval"] = types.ModuleType("vlmeval")
sys.modules["vlmeval.smp"] = types.ModuleType("vlmeval.smp")
sys.modules["vlmeval.smp.log"] = _smp
_spec.loader.exec_module(_mod)
can_infer = _mod.can_infer

# A 4-option MCQ (BLINK style)
CHOICES = {"A": "0", "B": "3", "C": "2", "D": "1"}
GT = "B"

# Match the YAML wrapper's _prediction_for_can_infer
import re as _re
_ANSWER_TAG_RE = _re.compile(r"<answer>\s*(.*?)\s*</answer>", _re.IGNORECASE | _re.DOTALL)


def _prediction_for_can_infer(response):
    m = _ANSWER_TAG_RE.search(response)
    if m:
        payload = m.group(1).strip()
        if payload:
            return payload
    return response

# (label, raw_response, expected_can_infer_result)
CASES = [
    # --- Outputs the BASE model might produce ---
    ("base: bare letter", "B", "B"),
    ("base: 'The answer is (B)'", "The answer is (B)", "B"),
    ("base: '(B) three blue floats'", "(B) three blue floats", "B"),
    ("base: verbose with letter at end", "Looking at the image, I count three blue objects, so the answer is B.", "B"),
    ("base: hedged", "It's either A or B", False),  # 2 letters → can_infer fails

    # --- Outputs OUR trained model produces ---
    ("ours: clean tags", "<think>Three floats visible.</think><answer>(B)</answer>", "B"),
    ("ours: bare letter in tags", "<think>...</think><answer>B</answer>", "B"),
    ("ours: verbose payload", "<think>...</think><answer>(B) three blue floats</answer>", "B"),
    ("ours: 'B' inside think only — no tags fallthru",
     "<think>I think the answer is B</think>", "B"),  # last-5-tokens may include 'B'
    ("ours: contradictory think vs answer",
     "<think>Looks like A</think><answer>(B)</answer>", "B"),  # Both letters, can_infer may fail
]


def main() -> None:
    print(f"{'label':<55} {'pre-extract':<25} {'can_infer':<12} {'expected':<12} {'match'}")
    print("-" * 120)
    pass_n = 0
    for label, response, expected in CASES:
        prepared = _prediction_for_can_infer(response)
        got = can_infer(prepared, CHOICES)
        match = got == expected
        if match:
            pass_n += 1
        prep_short = (prepared[:23] + "..") if len(prepared) > 25 else prepared
        print(f"{label:<55} {prep_short:<25} {str(got):<12} {str(expected):<12} {'PASS' if match else 'FAIL'}")
    print()
    print(f"OVERALL: {pass_n}/{len(CASES)} pass")
    sys.exit(0 if pass_n == len(CASES) else 1)


if __name__ == "__main__":
    main()
