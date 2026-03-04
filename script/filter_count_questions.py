import argparse
import json
import os
import re
from typing import Dict, List


DEFAULT_QUESTION_FILES = [
    os.path.join("v2_Questions_Val_mscoco", "v2_OpenEnded_mscoco_val2014_questions.json")
]
DEFAULT_ANNOTATION_FILE = os.path.join(
    "v2_Annotations_Val_mscoco", "v2_mscoco_val2014_annotations.json"
)
DEFAULT_IMAGE_DIR = "val2014/val2014"
DEFAULT_OUTPUT = "count_questions_val2014.json"

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


def is_count_question(text: str) -> bool:
    if not text:
        return False
    normalized = text.strip().lower()
    return any(re.search(pattern, normalized) for pattern in COUNT_PATTERNS)


def build_image_path(image_dir: str, image_id: int) -> str:
    filename = f"COCO_val2014_{image_id:012d}.jpg"
    return os.path.join(image_dir, filename)


def load_questions(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_questions(
    question_data: Dict,
    annotation_map: Dict[int, Dict],
    image_dir: str,
    source_file: str,
) -> List[Dict]:
    filtered = []
    for item in question_data.get("questions", []):
        question_text = item.get("question", "")
        if not is_count_question(question_text):
            continue
        image_id = item.get("image_id")
        question_id = item.get("question_id")
        annotation = annotation_map.get(question_id)
        if not annotation:
            continue
        image_path = build_image_path(image_dir, image_id)
        image_exists = os.path.exists(image_path)
        enriched = dict(item)
        enriched["is_count"] = True
        enriched["image_path"] = image_path
        enriched["image_exists"] = image_exists
        enriched["answers"] = [a["answer"] for a in annotation.get("answers", [])]
        enriched["source_file"] = source_file
        filtered.append(enriched)
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter count questions from VQA-v2 val2014 question files."
    )
    parser.add_argument(
        "--question-files",
        nargs="+",
        default=DEFAULT_QUESTION_FILES,
        help="Paths to VQA-v2 question JSON files.",
    )
    parser.add_argument(
        "--image-dir",
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing val2014 images.",
    )
    parser.add_argument(
        "--annotation-file",
        default=DEFAULT_ANNOTATION_FILE,
        help="Path to VQA-v2 val2014 annotation JSON file.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations_data = load_questions(args.annotation_file)
    annotation_map = {
        ann["question_id"]: ann for ann in annotations_data.get("annotations", [])
    }
    all_filtered: List[Dict] = []
    for file_path in args.question_files:
        data = load_questions(file_path)
        filtered = filter_questions(
            data,
            annotation_map=annotation_map,
            image_dir=args.image_dir,
            source_file=os.path.basename(file_path),
        )
        all_filtered.extend(filtered)

    output_data = {
        "count_patterns": COUNT_PATTERNS,
        "total_count": len(all_filtered),
        "questions": all_filtered,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_filtered)} count questions to {args.output}")


if __name__ == "__main__":
    main()
