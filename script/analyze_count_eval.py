import argparse
import json
import re
from collections import Counter
from typing import Dict, List, Tuple

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    normalized = text.strip().lower()
    normalized = normalized.replace(".", "")
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in NUMBER_WORDS:
        return NUMBER_WORDS[normalized]
    return normalized


def extract_count_answer(text: str) -> str:
    if text is None:
        return ""
    normalized = text.strip().lower()
    number_match = re.search(r"\d+", normalized)
    if number_match:
        return number_match.group(0)
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", normalized):
            return value
    return normalize_answer(text)


def choose_gold_answer(answers: List[str]) -> str:
    if not answers:
        return ""
    normalized = [normalize_answer(a) for a in answers if a is not None]
    normalized = [a for a in normalized if a]
    if not normalized:
        return ""
    counts = Counter(normalized)
    return counts.most_common(1)[0][0]


def sort_key(item: Tuple[str, Dict[str, float]]) -> Tuple[int, str]:
    key = item[0]
    if key.isdigit():
        return (0, f"{int(key):06d}")
    return (1, key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-number accuracy from count evaluation results."
    )
    parser.add_argument("--input", default="count_eval_origin.json")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.input)
    results = data.get("results", []) if isinstance(data, dict) else []

    per_number: Dict[str, Dict[str, float]] = {}
    total_seen = 0
    total_correct = 0

    for item in results:
        answers = item.get("answers", [])
        gold = choose_gold_answer(answers)
        if not gold:
            continue
        pred_raw = item.get("prediction", "")
        pred = extract_count_answer(pred_raw)

        stats = per_number.setdefault(gold, {"total": 0, "correct": 0})
        stats["total"] += 1
        if pred == gold:
            stats["correct"] += 1
        total_seen += 1
        total_correct += 1 if pred == gold else 0

    summary = []
    for key, stats in sorted(per_number.items(), key=sort_key):
        total = int(stats["total"])
        correct = int(stats["correct"])
        acc = (correct / total) if total else 0.0
        summary.append(
            {"number": key, "total": total, "correct": correct, "accuracy": acc}
        )

    overall_acc = (total_correct / total_seen) if total_seen else 0.0

    print("Per-number accuracy:")
    for row in summary:
        print(
            f"{row['number']:>4}  total={row['total']:>5}  "
            f"correct={row['correct']:>5}  acc={row['accuracy']:.4f}"
        )
    print(f"Overall: total={total_seen} correct={total_correct} acc={overall_acc:.4f}")

    if args.output:
        output = {
            "input": args.input,
            "overall": {
                "total": total_seen,
                "correct": total_correct,
                "accuracy": overall_acc,
            },
            "per_number": summary,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved analysis to {args.output}.")


if __name__ == "__main__":
    main()
