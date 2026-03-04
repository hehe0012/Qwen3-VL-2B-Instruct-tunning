import argparse
import base64
import json
import os
import random
import re
from typing import Dict, List, Optional

from openai import OpenAI

'''
python script/evaluate_count_questions.py \
    --count-file count_finetune_splits/test.json \
    --output count_eval_tunning.json \
    --max-tokens 200 \
    --sample 10
python script/evaluate_count_questions.py --sample 10
'''


DEFAULT_COUNT_FILE = "count_finetune_splits/test.json"
DEFAULT_OUTPUT = "count_eval_tunning2.json"

COUNT_PATTERNS = [
    r"\bhow many\b",
    r"\bnumber of\b",
    r"\bcount\b",
    r"\bamount of\b",
    r"\btotal number\b",
    r"\bquantity\b",
    r"\u6570\u91cf",
    r"\u591a\u5c11",
]

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

NUMBER_WORDS_REVERSE = {v: k for k, v in NUMBER_WORDS.items()}


def is_count_question(text: str) -> bool:
    if not text:
        return False
    normalized = text.strip().lower()
    return any(re.search(pattern, normalized) for pattern in COUNT_PATTERNS)


def build_image_path(image_dir: str, image_id: int) -> str:
    filename = f"COCO_val2014_{image_id:012d}.jpg"
    return os.path.join(image_dir, filename)


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
    return normalized


def vqa_soft_accuracy(prediction: str, answers: List[str]) -> float:
    if not answers:
        return 0.0
    pred = normalize_answer(prediction)
    if not pred:
        return 0.0

    normalized_answers = [normalize_answer(a) for a in answers]
    if len(normalized_answers) == 1:
        return 1.0 if pred == normalized_answers[0] else 0.0
    total = 0.0
    for idx in range(len(normalized_answers)):
        other_answers = [a for j, a in enumerate(normalized_answers) if j != idx]
        match_count = sum(1 for a in other_answers if a == pred)
        total += min(1.0, match_count / 3.0)
    return total / len(normalized_answers)


def build_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def read_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_text_content(message_content) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return " ".join(parts).strip()
    return str(message_content)


def query_model(
    client: OpenAI,
    model: str,
    image_path: str,
    question: str,
    max_tokens: int,
) -> str:
    base64_image = read_image_base64(image_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                    {"type": "text", "text": question},
                ],
            }
        ],
        max_tokens=max_tokens,
    )
    return extract_text_content(response.choices[0].message.content).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a filtered VQA-v2 count-question dataset with VQA soft accuracy."
    )
    parser.add_argument("--count-file", default=DEFAULT_COUNT_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--model", default="/model")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count_data = load_json(args.count_file)
    if isinstance(count_data, list):
        candidates = count_data
    else:
        candidates = count_data.get("questions", [])
    if not candidates:
        raise ValueError(
            f"No questions found in count file: {args.count_file}. "
            "Please re-run filter_count_questions.py to generate it."
        )

    if args.sample is not None:
        random.seed(args.seed)
        if args.sample < len(candidates):
            candidates = random.sample(candidates, args.sample)

    if args.limit is not None:
        candidates = candidates[: args.limit]

    client = build_client(args.base_url, args.api_key)

    results = []
    total_accuracy = 0.0
    correct_count = 0
    for idx, item in enumerate(candidates, start=1):
        if "conversations" in item:
            image_path = item.get("image")
            question = ""
            answers = []
            for turn in item.get("conversations", []):
                if turn.get("from") == "human" and not question:
                    question = turn.get("value", "")
                elif turn.get("from") == "gpt":
                    answers.append(turn.get("value", ""))
            question = question.replace("<image>", "").strip()
            question_id = item.get("question_id")
            image_id = item.get("image_id")
        else:
            image_path = item.get("image_path")
            if not image_path:
                image_id = item.get("image_id")
                image_path = build_image_path("val2014/val2014", image_id)
            question = item.get("question", "")
            answers = item.get("answers", [])
            question_id = item.get("question_id")
            image_id = item.get("image_id")
        if not os.path.exists(image_path):
            continue
        raw_prediction = query_model(
            client,
            model=args.model,
            image_path=image_path,
            question=question,
            max_tokens=args.max_tokens,
        )
        prediction = extract_count_answer(raw_prediction)
        accuracy = vqa_soft_accuracy(prediction, answers)
        total_accuracy += accuracy
        if accuracy >= 1.0:
            correct_count += 1
        results.append(
            {
                "question_id": question_id,
                "image_id": image_id,
                "question": question,
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "accuracy": accuracy,
                "answers": answers,
                "image_path": image_path,
            }
        )
        print(
            f"[{idx}/{len(candidates)}] acc={accuracy:.3f} qid={question_id}"
        )

    mean_accuracy = total_accuracy / len(results) if results else 0.0
    output = {
        "count_patterns": COUNT_PATTERNS,
        "total": len(results),
        "correct_count": correct_count,
        "mean_accuracy": mean_accuracy,
        "results": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved results to {args.output}. Mean accuracy: {mean_accuracy:.4f}")


if __name__ == "__main__":
    main()
