"""Filter FIRE training data into 5 balanced categories (5000 each)."""

import json
import random
import re


def get_question(sample: dict) -> str:
    """Extract the user question from a sample's messages."""
    for m in sample["messages"]:
        if m["role"] == "user":
            return m["content"]
    return ""


def is_mcq(q: str) -> bool:
    """Check if question is multiple choice."""
    return bool(
        re.search(r"\([A-D]\)", q)
        or re.search(r"\b[A-D]\)", q)
        or re.search(r"\bA\.\s|\bB\.\s|\bC\.\s|\bD\.\s", q)
    )


def is_yesno(q: str) -> bool:
    """Check if question is yes/no."""
    q = q.lower().strip()
    if "yes or no" in q:
        return True
    if q.startswith(
        (
            "is ",
            "are ",
            "does ",
            "do ",
            "did ",
            "can ",
            "could ",
            "should ",
            "would ",
            "was ",
            "were ",
            "has ",
            "have ",
            "had ",
        )
    ):
        return True
    return False


def is_counting(q: str) -> bool:
    """Check if question is a counting question."""
    q = q.lower()
    return "how many" in q or "number of" in q


def is_chart(q: str) -> bool:
    """Check if question involves chart/graph reasoning."""
    q = q.lower()
    keywords = [
        "chart",
        "graph",
        "plot",
        "ratio",
        "percentage",
        "percent",
        "trend",
        "bar",
        "line",
        "pie",
    ]
    return any(k in q for k in keywords)


def main() -> None:
    input_path = "/outputs/fire_preprocessed_v3/fire_messages_train_last_loss_only.jsonl"
    output_path = "/outputs/fire_preprocessed_v3/fire_balanced_5categories_5000_each.jsonl"

    categories: dict[str, list] = {
        "mcq": [],
        "yes_no": [],
        "counting": [],
        "chart_reasoning": [],
        "descriptive_vqa": [],
    }

    with open(input_path) as f:
        for idx, line in enumerate(f):
            sample = json.loads(line)
            q = get_question(sample)

            if is_mcq(q):
                categories["mcq"].append((idx, sample))
            elif is_yesno(q):
                categories["yes_no"].append((idx, sample))
            elif is_counting(q):
                categories["counting"].append((idx, sample))
            elif is_chart(q):
                categories["chart_reasoning"].append((idx, sample))
            else:
                categories["descriptive_vqa"].append((idx, sample))

    print("Category counts:")
    for cat, pool in categories.items():
        print(f"  {cat}: {len(pool)}")

    random.seed(42)
    final = []

    for cat, pool in categories.items():
        if len(pool) >= 5000:
            subset = random.sample(pool, 5000)
        else:
            subset = random.choices(pool, k=5000)  # duplicate if needed

        for idx, s in subset:
            final.append(
                {
                    "category": cat,
                    "source_index": idx,
                    "messages": s["messages"],
                    "images": s.get("images", []),
                }
            )

    random.shuffle(final)

    with open(output_path, "w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")

    print(f"Dataset written to: {output_path}")
    print(f"Total samples: {len(final)}")


if __name__ == "__main__":
    main()
