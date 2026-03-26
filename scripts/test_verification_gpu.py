#!/usr/bin/env python3
"""
Comprehensive verification pipeline test WITH LLM judge on GPU.

Tests all answer types (MCQ, YesNo, Counting, Short, Open) with
realistic examples from our training data. Requires GPU and
VLM_USE_LLM_JUDGE=1 to exercise the full pipeline.

Usage:
    VLM_USE_LLM_JUDGE=1 python3 /tmp/test_verification_gpu.py
"""

import os
import sys
import time

# Add the test package to path
sys.path.insert(0, "/tmp/vlm_grpo_test")

os.environ["VLM_USE_LLM_JUDGE"] = "1"

from vlm_grpo.rewards.verifier import verify_answer, CORRECT, WRONG  # noqa: E402
from vlm_grpo.rewards.correctness import compute_a2_correctness_reward  # noqa: E402

# =============================================================================
# Test Cases
# =============================================================================

TESTS = [
    # =========================================================================
    # MCQ — Letter GT
    # =========================================================================
    ("MCQ", "mcq", "A", "A", True, "Exact letter match"),
    ("MCQ", "mcq", "(B)", "B", True, "Parenthesized letter"),
    ("MCQ", "mcq", "B", "A", False, "Wrong letter"),
    ("MCQ", "mcq", "C. 72 degrees", "C", True, "Letter with answer text"),
    ("MCQ", "mcq", "The answer is C", "C", True, "The answer is X pattern"),
    ("MCQ", "mcq", "answer: B", "B", True, "answer: X pattern"),
    ("MCQ", "mcq", "I believe the answer is D based on the image", "D", True,
     "Letter embedded in explanation"),
    ("MCQ", "mcq", "", "A", False, "Empty answer"),
    ("MCQ", "mcq", "I think it could be A or B", "A", False,
     "Hedging multiple letters (anti-hack)"),
    ("MCQ", "mcq", "The image shows a beautiful sunset", "B", False,
     "No letter at all in response"),
    ("MCQ", "mcq", "(d) This image contains a bowl.", "D", True,
     "Lowercase letter with text"),

    # =========================================================================
    # MCQ — Text GT (aokvqa style: GT is option text, not letter)
    # =========================================================================
    ("MCQ-text", "mcq", "Cab", "Cab", True, "Text GT exact match"),
    ("MCQ-text", "mcq", "cab", "Cab", True, "Text GT case insensitive"),
    ("MCQ-text", "mcq", "A cab", "Cab", True, "Text GT with article"),
    ("MCQ-text", "mcq", "Farmer", "Farmer.", True, "Text GT trailing period"),
    ("MCQ-text", "mcq", "taxi", "Cab", True,
     "Text GT synonym (needs LLM judge)"),
    ("MCQ-text", "mcq", "bus", "Cab", False, "Text GT completely wrong"),
    ("MCQ-text", "mcq", "office", "office", True, "Text GT office exact"),
    ("MCQ-text", "mcq", "an office setting", "office", True,
     "Text GT with elaboration"),
    ("MCQ-text", "mcq", "restaurant", "office", False, "Text GT wrong place"),

    # =========================================================================
    # Yes/No — Bare and with explanation
    # =========================================================================
    ("YesNo", "yesno", "Yes", "Yes", True, "Bare yes match"),
    ("YesNo", "yesno", "No", "Yes", False, "Wrong polarity"),
    ("YesNo", "yesno", "No", "No", True, "Bare no match"),
    ("YesNo", "yesno", "", "Yes", False, "Empty answer"),
    ("YesNo", "yesno", "Maybe yes", "Yes", False, "Hedging rejected"),
    ("YesNo", "yesno", "I think so, probably yes.", "Yes", False,
     "Hedging with yes"),

    # Yes/No with full sentence GT (our actual data format)
    ("YN-sent", "yesno",
     "Yes, there is a television set visible in the living room.",
     "Yes, there is a television set visible in the living room.",
     True, "Full sentence exact match"),
    ("YN-sent", "yesno",
     "Yes, there is a TV in the room.",
     "Yes, there is a television set visible in the living room.",
     True, "Paraphrased TV → television (same meaning)"),
    ("YN-sent", "yesno",
     "Yes, there are bottles on the table.",
     "Yes, there are breads on the table.",
     False, "Same polarity but WRONG objects (bottles vs breads)"),
    ("YN-sent", "yesno",
     "No, the van is gray but large.",
     "No, there is no snow in this image.",
     False, "Same polarity but UNRELATED topics"),
    ("YN-sent", "yesno",
     "Yes, there are two dogs playing in the park.",
     "Yes, there are two dogs playing in the park.",
     True, "Full match dogs"),
    ("YN-sent", "yesno",
     "Yes, there are two cats playing in the park.",
     "Yes, there are two dogs playing in the park.",
     False, "Wrong animals (cats vs dogs)"),
    ("YN-sent", "yesno",
     "No, the building does not appear to be old.",
     "No, the building does not look old or deteriorated.",
     True, "Same meaning rephrased"),
    ("YN-sent", "yesno",
     "Yes, the man is wearing a hat.",
     "No, the man is not wearing a hat.",
     False, "Opposite polarity with explanation"),

    # =========================================================================
    # Counting — Fuzzy reward
    # =========================================================================
    ("Count", "counting", "3", "3", True, "Exact match"),
    ("Count", "counting", "0", "0", True, "Zero count exact"),
    ("Count", "counting", "5", "6", True, "Off by 1 from GT=6 (score~0.83)"),
    ("Count", "counting", "3", "6", True, "Off by 3 from GT=6 (score~0.50)"),
    ("Count", "counting", "12", "6", False, "Double the GT (score=0)"),
    ("Count", "counting", "1", "0", True,
     "Off by 1 from GT=0 (fixed: score~0.67)"),
    ("Count", "counting", "2", "1", True,
     "Off by 1 from GT=1 (fixed: score~0.67)"),
    ("Count", "counting", "0", "1", True,
     "Off by 1 from GT=1 other dir (score~0.67)"),
    ("Count", "counting", "There are three cats in the image", "3", True,
     "Word number in sentence"),
    ("Count", "counting", "two", "2", True, "Word number 'two'"),
    ("Count", "counting", "", "3", False, "Empty answer"),
    ("Count", "counting", "many", "5", False, "Non-numeric word"),

    # =========================================================================
    # Short answer — deterministic + embedding
    # =========================================================================
    ("Short", "open", "red", "red", True, "Exact 1-word"),
    ("Short", "open", "Red", "red", True, "Case insensitive"),
    ("Short", "open", "a dog", "dog", True, "Article prefix (substring)"),
    ("Short", "open", "the color is red", "red", True,
     "Verbose contains GT (substring)"),
    ("Short", "open", "blue", "red", False, "Wrong color"),
    ("Short", "open", "December", "December", True, "Month exact"),
    ("Short", "open", "fire truck", "fire truk", True,
     "Typo handled by ANLS"),
    ("Short", "open", "kitchen table", "dining table", None,
     "Near-synonym (may pass with LLM judge)"),
    ("Short", "open", "baseball", "tennis", False, "Different sport"),
    ("Short", "open", "cat", "dog", False, "Different animal"),
    ("Short", "open", "automobile", "car", True, "Synonym (embedding)"),
    ("Short", "open", "NYC", "New York City", None,
     "Abbreviation (may need LLM judge)"),

    # =========================================================================
    # Open-ended / Descriptive — Full pipeline
    # =========================================================================
    ("Open", "open",
     "The man is making pizza by spreading marinara sauce on the dough.",
     "The man is spreading marinara sauce on pizza dough with a spoon.",
     True, "Paraphrased cooking description"),
    ("Open", "open",
     "The bathroom has a toilet and a urinal side by side, which is unusual for most homes.",
     "This bathroom has both a toilet and a urinal, which is unusual.",
     True, "Same observation different wording"),
    ("Open", "open",
     "The image shows a sunny beach with colorful umbrellas.",
     "The city has tall buildings and busy streets.",
     False, "Completely different scene"),
    ("Open", "open",
     "The cat is sitting on the left side of the couch.",
     "The cat is sitting on the right side of the couch.",
     False, "Antonym contradiction (left vs right)"),
    ("Open", "open",
     "The trend shows an increase over the past decade.",
     "The trend shows a decrease over the past decade.",
     False, "Antonym (increase vs decrease)"),
    ("Open", "open",
     "There are two elephants walking near the road with trees in the background.",
     "Yes, there is an elephant present in the image. It is walking in the background near a road.",
     True, "Different structure same content (needs LLM judge)"),
    ("Open", "open",
     "The woman is talking on her cell phone while crossing a busy street.",
     "The woman is looking at her phone while in traffic, which is dangerous.",
     None, "Similar scene slightly different (borderline)"),
    ("Open", "open",
     "Workers should wear hard hats, safety shoes, and reflective vests near trains.",
     "Workers should wear specific protective gear such as hard hats, safety shoes, and reflective vests when working near railway tracks.",
     True, "Same safety advice, GT is more detailed"),
    ("Open", "open",
     "Playing video games can be fun and help with hand-eye coordination.",
     "Playing video games can be fun and entertaining, and it can also help improve hand-eye coordination and problem-solving skills.",
     True, "Subset of GT content"),
    ("Open", "open",
     "The image shows a cat sleeping on a windowsill in bright sunlight.",
     "The photograph depicts a group of children playing soccer in a park.",
     False, "Completely unrelated scenes"),

    # =========================================================================
    # Thought/Answer extraction
    # =========================================================================
    ("Extract", "mcq",
     "Thought: Looking at the options carefully, I think B is correct because the image shows a farm. Answer: B",
     "B", True, "Thought/Answer MCQ"),
    ("Extract", "open",
     "Thought: The image shows a red double-decker bus. Answer: The bus is red and appears to be a London-style double-decker bus.",
     "The bus is red.", True, "Thought/Answer open"),
    ("Extract", "counting",
     "Thought: Let me count carefully. I see 1, 2, 3 dogs. Answer: 3",
     "3", True, "Thought/Answer counting"),
    ("Extract", "yesno",
     "Thought: The image clearly shows a cat on the sofa. Answer: Yes, there is a cat on the sofa.",
     "Yes, there is a cat sitting on the couch.",
     True, "Thought/Answer yesno with paraphrase"),

    # =========================================================================
    # Edge cases
    # =========================================================================
    ("Edge", "mcq", "A or B or C", "A", False, "Multiple letters rejected"),
    ("Edge", "open", "I cannot determine the answer from this image.",
     "The image shows a red car.", False, "Refusal vs actual answer"),
    ("Edge", "open", "N/A", "The building is a church.", False,
     "N/A response"),
    ("Edge", "counting", "-1", "3", False, "Negative number for counting"),
    ("Edge", "open",
     "The image shows the same thing as described.",
     "A brown dog is lying on a green couch.",
     False, "Vague non-answer"),
]


def main() -> None:
    """Run all tests and report results."""
    print("=" * 90)
    print("COMPREHENSIVE VERIFICATION PIPELINE TEST (LLM Judge ENABLED)")
    print("=" * 90)

    # Verify LLM judge is enabled
    from vlm_grpo.rewards.judge_llm import is_enabled
    print(f"LLM Judge enabled: {is_enabled()}")
    print()

    # Warm up the LLM judge model
    print("Loading LLM judge model (Qwen2.5-3B)...")
    t0 = time.time()
    from vlm_grpo.rewards.judge_llm import llm_judge_score
    _ = llm_judge_score("test", "test")
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print()

    pass_count = 0
    fail_count = 0
    unexpected_count = 0
    results_by_category = {}

    for category, atype, pred, gt, expected, description in TESTS:
        r = verify_answer(pred, gt, atype)
        actual = r.is_correct
        score_str = f"score={r.score:.2f}" if r.score is not None else "score=None"

        if expected is None:
            # Borderline case — just report
            status = "BORDERLINE"
            icon = "~"
            unexpected_count += 1
        elif actual == expected:
            status = "PASS"
            icon = "✓"
            pass_count += 1
        else:
            status = "FAIL"
            icon = "✗"
            fail_count += 1

        # Track per category
        if category not in results_by_category:
            results_by_category[category] = {"pass": 0, "fail": 0, "borderline": 0}
        if status == "PASS":
            results_by_category[category]["pass"] += 1
        elif status == "FAIL":
            results_by_category[category]["fail"] += 1
        else:
            results_by_category[category]["borderline"] += 1

        # Print
        verdict_str = "CORRECT" if actual else "WRONG  "
        expected_str = "CORRECT" if expected else ("WRONG" if expected is not None else "???")
        print(f"  {icon} [{category:<10}] {verdict_str} | {score_str:<12} | {description}")
        if status == "FAIL":
            print(f"    EXPECTED: {expected_str}, GOT: {verdict_str}")
            print(f"    pred: \"{pred[:70]}\"")
            print(f"    gt:   \"{gt[:70]}\"")
        elif status == "BORDERLINE":
            print(f"    BORDERLINE (no expected): {verdict_str}, {score_str}")

    # Summary
    total = pass_count + fail_count + unexpected_count
    print()
    print("=" * 90)
    print(f"RESULTS: {pass_count}/{total} PASS, {fail_count} FAIL, {unexpected_count} BORDERLINE")
    print("=" * 90)
    print()
    print("Per-category breakdown:")
    for cat, counts in sorted(results_by_category.items()):
        cat_total = counts["pass"] + counts["fail"] + counts["borderline"]
        print(f"  {cat:<12}: {counts['pass']}/{cat_total} pass"
              f" ({counts['fail']} fail, {counts['borderline']} borderline)")

    # Also test continuous reward for counting and open
    print()
    print("=" * 90)
    print("CONTINUOUS REWARD VALUES (counting & open)")
    print("=" * 90)
    reward_tests = [
        ("counting", "3", "3", "Exact count"),
        ("counting", "5", "6", "Off by 1 (GT=6)"),
        ("counting", "1", "0", "Off by 1 (GT=0)"),
        ("counting", "3", "6", "Off by 3 (GT=6)"),
        ("counting", "12", "6", "Double GT"),
        ("open", "The man is making pizza.", "The man is spreading marinara sauce on pizza dough.", "Paraphrase"),
        ("open", "red", "red", "Exact short"),
        ("open", "blue", "red", "Wrong color"),
        ("open", "automobile", "car", "Synonym"),
    ]
    for atype, pred, gt, desc in reward_tests:
        reward = compute_a2_correctness_reward(pred, gt, atype, format_valid=True)
        print(f"  [{atype:<10}] reward={reward:+.3f} | {desc}")
        print(f"              pred=\"{pred}\"  gt=\"{gt}\"")

    print()
    if fail_count > 0:
        print(f"⚠ {fail_count} test(s) FAILED — review above")
    else:
        print("All expected tests passed!")


if __name__ == "__main__":
    main()
