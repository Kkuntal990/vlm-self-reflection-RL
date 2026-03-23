#!/usr/bin/env python3
"""Tests for scripts/label_answer_types.py."""

import json

# Import directly from script (self-contained, no src/ deps)
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from label_answer_types import (
    _extract_ground_truth,
    _extract_question,
    _is_numeric,
    detect_answer_type,
    process_jsonl,
)

# =============================================================================
# detect_answer_type
# =============================================================================


class TestDetectAnswerType:
    """Tests for GT-based answer type detection."""

    def test_mcq_with_choices(self) -> None:
        assert detect_answer_type("What color?", "A", choices="(A) Red (B) Blue") == "mcq"

    def test_mcq_pattern_in_question(self) -> None:
        q = "Which is correct? (A) Yes (B) No (C) Maybe"
        assert detect_answer_type(q, "A") == "mcq"

    def test_mcq_dot_pattern(self) -> None:
        q = "Choose one:\nA. Red\nB. Blue"
        assert detect_answer_type(q, "A") == "mcq"

    def test_mcq_single_letter_gt(self) -> None:
        assert detect_answer_type("What is shown?", "B") == "mcq"

    def test_yesno_yes(self) -> None:
        assert detect_answer_type("Is the sky blue?", "Yes") == "yesno"

    def test_yesno_no(self) -> None:
        assert detect_answer_type("Is it raining?", "no") == "yesno"

    def test_yesno_true(self) -> None:
        assert detect_answer_type("Is this correct?", "true") == "yesno"

    def test_numeric_integer(self) -> None:
        assert detect_answer_type("What is the value?", "42") == "numeric"

    def test_numeric_float(self) -> None:
        assert detect_answer_type("What is the ratio?", "3.14") == "numeric"

    def test_numeric_with_comma(self) -> None:
        assert detect_answer_type("How much?", "1,000") == "numeric"

    def test_numeric_percentage(self) -> None:
        assert detect_answer_type("What percent?", "35%") == "numeric"

    def test_numeric_fraction(self) -> None:
        assert detect_answer_type("What fraction?", "3/4") == "numeric"

    def test_counting_how_many(self) -> None:
        assert detect_answer_type("How many cats are there?", "3") == "counting"

    def test_counting_number_of(self) -> None:
        assert detect_answer_type("What is the number of items?", "5") == "counting"

    def test_counting_non_numeric_gt_is_open(self) -> None:
        """Counting question but non-numeric GT → open."""
        assert detect_answer_type("How many are there?", "several") == "open"

    def test_open_freeform(self) -> None:
        assert detect_answer_type("What is shown?", "A red car") == "open"

    def test_open_sentence(self) -> None:
        assert detect_answer_type("Describe it", "The dog is behind the fence") == "open"

    # --- Misclassification fixes: these were the actual bugs ---

    def test_chart_numeric_gt_not_open(self) -> None:
        """Chart question with numeric GT should be numeric, not open."""
        assert detect_answer_type("What is the ratio shown in the pie chart?", "35") == "numeric"

    def test_descriptive_numeric_gt_not_open(self) -> None:
        """Descriptive question with numeric GT should be numeric, not open."""
        assert detect_answer_type("What number is shown on the sign?", "24") == "numeric"

    def test_is_question_with_sentence_gt_is_open(self) -> None:
        """'Is the dog...' with sentence GT should be open, not yesno."""
        assert (
            detect_answer_type(
                "Is the dog behind or in front of the fence?",
                "behind the fence",
            )
            == "open"
        )

    def test_counting_question_numeric_gt(self) -> None:
        """'How many slices...' with GT='3' should be counting."""
        assert detect_answer_type("How many slices of pizza?", "3") == "counting"

    def test_counting_question_percentage_gt(self) -> None:
        """'How many percent...' with GT='35%' should be counting."""
        assert detect_answer_type("How many percent voted yes?", "35%") == "counting"


# =============================================================================
# _is_numeric
# =============================================================================


class TestIsNumeric:
    """Tests for numeric detection helper."""

    def test_integer(self) -> None:
        assert _is_numeric("42") is True

    def test_negative(self) -> None:
        assert _is_numeric("-3") is True

    def test_float(self) -> None:
        assert _is_numeric("3.14") is True

    def test_comma_separated(self) -> None:
        assert _is_numeric("1,000") is True

    def test_percentage(self) -> None:
        assert _is_numeric("35%") is True

    def test_fraction(self) -> None:
        assert _is_numeric("3/4") is True

    def test_word(self) -> None:
        assert _is_numeric("three") is False

    def test_sentence(self) -> None:
        assert _is_numeric("the answer is 3") is False

    def test_empty(self) -> None:
        assert _is_numeric("") is False


# =============================================================================
# Field Extraction
# =============================================================================


class TestFieldExtraction:
    """Tests for question/GT extraction from different formats."""

    def test_flat_format_question(self) -> None:
        sample = {"question": "What is this?", "ground_truth": "A cat"}
        assert _extract_question(sample) == "What is this?"

    def test_flat_format_gt(self) -> None:
        sample = {"question": "What is this?", "ground_truth": "A cat"}
        assert _extract_ground_truth(sample) == "A cat"

    def test_messages_format_question(self) -> None:
        sample = {
            "messages": [
                {"role": "user", "content": "What color?"},
                {"role": "assistant", "content": "Red"},
            ]
        }
        assert _extract_question(sample) == "What color?"

    def test_messages_format_gt(self) -> None:
        sample = {
            "messages": [
                {"role": "user", "content": "What color?"},
                {"role": "assistant", "content": "Red"},
            ]
        }
        assert _extract_ground_truth(sample) == "Red"

    def test_inference_results_gt(self) -> None:
        sample = {"gt_final_answer": "42", "question": "How many?"}
        assert _extract_ground_truth(sample) == "42"


# =============================================================================
# End-to-End JSONL Processing
# =============================================================================


class TestProcessJSONL:
    """Tests for full JSONL processing pipeline."""

    def _make_jsonl(self, samples: list[dict]) -> str:
        """Write samples to a temp JSONL file and return path."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for s in samples:
            tmp.write(json.dumps(s) + "\n")
        tmp.close()
        return tmp.name

    def test_relabels_open_to_numeric(self) -> None:
        """Sample with GT='24' and answer_type='open' should become 'numeric'."""
        samples = [
            {
                "question": "What number is on the sign?",
                "ground_truth": "24",
                "answer_type": "open",
            }
        ]
        input_path = self._make_jsonl(samples)
        output_path = input_path + ".out"

        process_jsonl(input_path, output_path)

        with open(output_path) as f:
            result = json.loads(f.readline())

        assert result["answer_type"] == "numeric"

    def test_relabels_counting(self) -> None:
        """'How many' question with numeric GT → counting."""
        samples = [
            {
                "question": "How many cats are in the image?",
                "ground_truth": "3",
                "answer_type": "open",
            }
        ]
        input_path = self._make_jsonl(samples)
        output_path = input_path + ".out"

        process_jsonl(input_path, output_path)

        with open(output_path) as f:
            result = json.loads(f.readline())

        assert result["answer_type"] == "counting"

    def test_preserves_correct_labels(self) -> None:
        """Already correct labels should not change."""
        samples = [
            {
                "question": "Is the sky blue?",
                "ground_truth": "Yes",
                "answer_type": "yesno",
            },
            {
                "question": "Choose: (A) Red (B) Blue",
                "ground_truth": "A",
                "answer_type": "mcq",
            },
        ]
        input_path = self._make_jsonl(samples)
        output_path = input_path + ".out"

        process_jsonl(input_path, output_path)

        with open(output_path) as f:
            results = [json.loads(line) for line in f]

        assert results[0]["answer_type"] == "yesno"
        assert results[1]["answer_type"] == "mcq"

    def test_messages_format(self) -> None:
        """Messages-format samples should be handled correctly."""
        samples = [
            {
                "messages": [
                    {"role": "user", "content": "What is the value?"},
                    {"role": "assistant", "content": "42"},
                ],
                "category": "descriptive_vqa",
            }
        ]
        input_path = self._make_jsonl(samples)
        output_path = input_path + ".out"

        process_jsonl(input_path, output_path)

        with open(output_path) as f:
            result = json.loads(f.readline())

        assert result["answer_type"] == "numeric"

    def test_dry_run_no_output(self) -> None:
        """Dry run should not create output file."""
        samples = [{"question": "Q?", "ground_truth": "42", "answer_type": "open"}]
        input_path = self._make_jsonl(samples)
        output_path = input_path + ".dry"

        process_jsonl(input_path, output_path, dry_run=True)

        assert not Path(output_path).exists()

    def test_multiple_samples_mixed(self) -> None:
        """Mixed sample types should all be correctly labeled."""
        samples = [
            {"question": "Is it red?", "ground_truth": "Yes", "answer_type": "open"},
            {
                "question": "What is the ratio in the chart?",
                "ground_truth": "35",
                "answer_type": "open",
            },
            {"question": "How many dogs?", "ground_truth": "2", "answer_type": "open"},
            {"question": "Describe it", "ground_truth": "A red car", "answer_type": "open"},
            {"question": "Choose: (A) Yes (B) No", "ground_truth": "A", "answer_type": "open"},
        ]
        input_path = self._make_jsonl(samples)
        output_path = input_path + ".out"

        process_jsonl(input_path, output_path)

        with open(output_path) as f:
            results = [json.loads(line) for line in f]

        assert results[0]["answer_type"] == "yesno"
        assert results[1]["answer_type"] == "numeric"
        assert results[2]["answer_type"] == "counting"
        assert results[3]["answer_type"] == "open"
        assert results[4]["answer_type"] == "mcq"

    def test_inplace_overwrite(self) -> None:
        """Writing to same file as input should work."""
        samples = [{"question": "What number?", "ground_truth": "7", "answer_type": "open"}]
        path = self._make_jsonl(samples)

        process_jsonl(path, path)

        with open(path) as f:
            result = json.loads(f.readline())

        assert result["answer_type"] == "numeric"
