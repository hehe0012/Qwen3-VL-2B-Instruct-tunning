import argparse
import json
from pathlib import Path


def load_json(path: Path) -> list:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: list) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def tag_split(items: list, split: str) -> list:
    tagged = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("Each item must be a dict.")
        tagged_item = dict(item)
        tagged_item["split"] = split
        tagged.append(tagged_item)
    return tagged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge train/val JSON and add split tags.")
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("count_finetune_splits/train.json"),
        help="Path to train JSON (default: count_finetune_splits/train.json)",
    )
    parser.add_argument(
        "--val",
        type=Path,
        default=Path("count_finetune_splits/val.json"),
        help="Path to val JSON (default: count_finetune_splits/val.json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("count_finetune_splits/train_val.json"),
        help="Output JSON path (default: count_finetune_splits/train_val.json)",
    )
    args = parser.parse_args()

    train_items = load_json(args.train)
    val_items = load_json(args.val)

    merged = tag_split(train_items, "train") + tag_split(val_items, "val")
    write_json(args.output, merged)

    print(f"Merged {len(train_items)} train and {len(val_items)} val items -> {args.output}")


if __name__ == "__main__":
    main()
