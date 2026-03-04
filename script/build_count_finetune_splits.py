import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

'''
python script/build_count_finetune_splits.py \
  --min-agreement 0.9 --min-numeric-answers 10 \
  --train-size 500 --val-size 100 --test-size 10000 \
--image-prefix val2014/val2014
'''


DEFAULT_INPUT = "count_questions_val2014.json"
DEFAULT_OUTPUT_DIR = "count_finetune_splits"

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
}


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_count_label(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    normalized = text.strip().lower()
    normalized = normalized.replace(".", "")
    normalized = re.sub(r"\s+", " ", normalized)
    number_match = re.search(r"\d+", normalized)
    if number_match:
        return number_match.group(0)
    if normalized in NUMBER_WORDS:
        return NUMBER_WORDS[normalized]
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", normalized):
            return value
    return None


def consensus_label(
    answers: List[str],
    min_agreement: float,
    min_numeric: int,
) -> Optional[Tuple[str, float, int]]:
    if not answers:
        return None
    numeric = [normalize_count_label(a) for a in answers]
    numeric = [n for n in numeric if n is not None]
    if len(numeric) < min_numeric:
        return None
    counts = Counter(numeric)
    label, count = counts.most_common(1)[0]
    agreement = count / len(answers)
    if agreement < min_agreement:
        return None
    return label, agreement, len(numeric)


def split_by_image(
    items: List[Dict],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    include_test: bool = True,
) -> Dict[str, List[Dict]]:
    groups: Dict[int, List[Dict]] = defaultdict(list)
    for item in items:
        groups[item["image_id"]].append(item)

    image_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    total_items = sum(len(groups[iid]) for iid in image_ids)
    target_train = int(total_items * train_ratio)
    target_val = int(total_items * val_ratio)
    target_test = total_items - target_train - target_val

    if not include_test:
        target_test = 0

    remaining = {
        "train": target_train,
        "val": target_val,
    }
    splits = {"train": [], "val": []}
    if include_test:
        remaining["test"] = target_test
        splits["test"] = []

    # Greedy assignment: place larger groups first to reduce imbalance.
    image_ids.sort(key=lambda iid: len(groups[iid]), reverse=True)
    for image_id in image_ids:
        group = groups[image_id]
        group_size = len(group)
        candidates = sorted(
            remaining.items(), key=lambda kv: (kv[1] - group_size < 0, kv[1])
        )
        chosen = candidates[0][0]
        splits[chosen].extend(group)
        remaining[chosen] = max(0, remaining[chosen] - group_size)

    return splits


def split_by_image_targets(
    items: List[Dict],
    seed: int,
    targets: Dict[str, int],
) -> Dict[str, List[Dict]]:
    groups: Dict[int, List[Dict]] = defaultdict(list)
    for item in items:
        groups[item["image_id"]].append(item)

    image_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    remaining = {
        "train": max(0, targets.get("train", 0)),
        "val": max(0, targets.get("val", 0)),
        "test": max(0, targets.get("test", 0)),
    }
    splits = {"train": [], "val": [], "test": []}

    image_ids.sort(key=lambda iid: len(groups[iid]), reverse=True)
    for image_id in image_ids:
        group = groups[image_id]
        group_size = len(group)
        candidates = sorted(
            remaining.items(), key=lambda kv: (kv[1] - group_size < 0, kv[1])
        )
        chosen = candidates[0][0]
        splits[chosen].extend(group)
        remaining[chosen] = max(0, remaining[chosen] - group_size)

    return splits


def bucket_label(label: str) -> str:
    try:
        value = int(label)
    except (TypeError, ValueError):
        return label
    if 9 <= value <= 20:
        return "9-20"
    return str(value)


def bucket_sort_key(label: str) -> Tuple[int, str]:
    if label == "9-20":
        return (9, label)
    try:
        return (int(label), label)
    except (TypeError, ValueError):
        return (10**9, label)


def bucket_counts(items: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for item in items:
        counts[bucket_label(item["label"])] += 1
    return dict(sorted(counts.items(), key=lambda kv: bucket_sort_key(kv[0])))


def compute_split_targets(
    total: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, int]:
    train = int(total * train_ratio)
    val = int(total * val_ratio)
    test = max(0, total - train - val)
    return {"train": train, "val": val, "test": test}


def balance_split(
    items: List[Dict],
    target_total: int,
    seed: int,
) -> List[Dict]:
    if target_total <= 0:
        return []
    if target_total >= len(items):
        return list(items)

    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for item in items:
        buckets[item["label"]].append(item)

    rng = random.Random(seed)
    for bucket_items in buckets.values():
        rng.shuffle(bucket_items)

    labels = sorted(buckets.keys(), key=bucket_sort_key)
    base = target_total // len(labels)
    assigned: Dict[str, int] = {}
    for label in labels:
        assigned[label] = min(base, len(buckets[label]))

    remaining = target_total - sum(assigned.values())
    while remaining > 0:
        progressed = False
        for label in labels:
            if remaining <= 0:
                break
            if assigned[label] < len(buckets[label]):
                assigned[label] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    selected: List[Dict] = []
    for label in labels:
        selected.extend(buckets[label][: assigned[label]])
    return selected


def take_balanced(
    items: List[Dict],
    target_total: int,
    seed: int,
) -> Tuple[List[Dict], List[Dict]]:
    selected = balance_split(items, target_total, seed)
    selected_keys = {
        (item.get("question_id"), item.get("image_id")) for item in selected
    }
    remaining = [
        item
        for item in items
        if (item.get("question_id"), item.get("image_id")) not in selected_keys
    ]
    return selected, remaining


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def build_image_reference(image_path: Optional[str], image_prefix: str) -> str:
    if not image_path:
        return ""
    filename = os.path.basename(image_path.replace("\\", "/"))
    prefix = image_prefix.rstrip("/")
    return f"{prefix}/{filename}"


def to_conversation_item(item: Dict, image_prefix: str) -> Dict:
    image_ref = build_image_reference(item.get("image_path"), image_prefix)
    question = item.get("question", "")
    question = question.replace("<image>", "").strip()
    return {
        "image": image_ref,
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n" + question,
            },
            {
                "from": "gpt",
                "value": str(item.get("label", "")),
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build filtered, bucketed train/val/test splits for count VQA fine-tuning."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-agreement", type=float, default=0.6)
    parser.add_argument("--min-numeric-answers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.85)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--val-size", type=int, default=None)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument(
        "--total-size",
        type=int,
        default=None,
        help="Total size for train+val+test after per-split bucket balancing.",
    )
    parser.add_argument(
        "--image-prefix",
        default="images/counting",
        help="Image path prefix used in the output dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.input)
    if isinstance(data, dict) and "questions" in data:
        results = data.get("questions", [])
    else:
        results = data.get("results", [])

    filtered: List[Dict] = []
    for item in results:
        answers = item.get("answers", [])
        consensus = consensus_label(
            answers,
            min_agreement=args.min_agreement,
            min_numeric=args.min_numeric_answers,
        )
        if not consensus:
            continue
        label, agreement, numeric_count = consensus
        if int(label) >= 7:
            continue
        if int(label) > 20:
            continue
        label = bucket_label(label)
        filtered.append(
            {
                "question_id": item.get("question_id"),
                "image_id": item.get("image_id"),
                "question": item.get("question", ""),
                "image_path": item.get("image_path"),
                "label": label,
                "agreement": agreement,
                "numeric_answer_count": numeric_count,
                "answers": answers,
            }
        )

    has_explicit_sizes = any(
        size is not None for size in (args.train_size, args.val_size, args.test_size)
    )
    if has_explicit_sizes:
        if None in (args.train_size, args.val_size, args.test_size):
            raise ValueError("--train-size/--val-size/--test-size must all be set.")
        targets = {
            "train": args.train_size,
            "val": args.val_size,
            "test": args.test_size,
        }
        pool = list(filtered)
        train_split, pool = take_balanced(pool, targets["train"], args.seed)
        val_split, pool = take_balanced(pool, targets["val"], args.seed + 1)
        test_split, pool = take_balanced(pool, targets["test"], args.seed + 2)
        splits = {
            "train": train_split,
            "val": val_split,
            "test": test_split,
        }
    else:
        splits = split_by_image(
            filtered,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            include_test=True,
        )

    balanced_splits = splits
    split_targets = None
    if has_explicit_sizes:
        split_targets = targets
    elif args.total_size is not None:
        split_targets = compute_split_targets(
            args.total_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )

    if split_targets is not None and not has_explicit_sizes:
        balanced_splits = {
            "train": balance_split(splits["train"], split_targets["train"], args.seed),
            "val": balance_split(splits["val"], split_targets["val"], args.seed + 1),
            "test": balance_split(splits["test"], split_targets["test"], args.seed + 2),
        }

    os.makedirs(args.output_dir, exist_ok=True)
    write_json(
        os.path.join(args.output_dir, "train.json"),
        [to_conversation_item(item, args.image_prefix) for item in balanced_splits["train"]],
    )
    write_json(
        os.path.join(args.output_dir, "val.json"),
        [to_conversation_item(item, args.image_prefix) for item in balanced_splits["val"]],
    )
    write_json(
        os.path.join(args.output_dir, "test.json"),
        [to_conversation_item(item, args.image_prefix) for item in balanced_splits["test"]],
    )

    stats = {
        "source": args.input,
        "total_results": len(results),
        "filtered_items": len(filtered),
        "min_agreement": args.min_agreement,
        "min_numeric_answers": args.min_numeric_answers,
        "seed": args.seed,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": round(1.0 - args.train_ratio - args.val_ratio, 4),
        },
        "initial_splits": {
            "pool": {
                "items": len(filtered),
                "images": len({i["image_id"] for i in filtered}),
                "buckets": bucket_counts(filtered),
            },
        }
        if has_explicit_sizes
        else {
            "train": {
                "items": len(splits["train"]),
                "images": len({i["image_id"] for i in splits["train"]}),
                "buckets": bucket_counts(splits["train"]),
            },
            "val": {
                "items": len(splits["val"]),
                "images": len({i["image_id"] for i in splits["val"]}),
                "buckets": bucket_counts(splits["val"]),
            },
            "test": {
                "items": len(splits["test"]),
                "images": len({i["image_id"] for i in splits["test"]}),
                "buckets": bucket_counts(splits["test"]),
            },
        },
        "final_splits": {
            "total_size": args.total_size,
            "targets": split_targets,
            "train": {
                "items": len(balanced_splits["train"]),
                "images": len({i["image_id"] for i in balanced_splits["train"]}),
                "buckets": bucket_counts(balanced_splits["train"]),
            },
            "val": {
                "items": len(balanced_splits["val"]),
                "images": len({i["image_id"] for i in balanced_splits["val"]}),
                "buckets": bucket_counts(balanced_splits["val"]),
            },
            "test": {
                "items": len(balanced_splits["test"]),
                "images": len({i["image_id"] for i in balanced_splits["test"]}),
                "buckets": bucket_counts(balanced_splits["test"]),
            },
        },
    }
    write_json(os.path.join(args.output_dir, "stats.json"), stats)

    print(
        "Saved splits to {} (train/val/test) and stats.json".format(args.output_dir)
    )


if __name__ == "__main__":
    main()
