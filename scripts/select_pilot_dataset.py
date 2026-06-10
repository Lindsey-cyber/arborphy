#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.jsonl import read_jsonl, write_jsonl


GROUP_RULES = [
    ("trillium", ("Trillium",)),
    ("taraxacum", ("Taraxacum",)),
    ("acer", ("Acer",)),
    ("quercus", ("Quercus",)),
    ("pteridium", ("Pteridium",)),
    ("rosa", ("Rosa",)),
    ("poaceae", ("Poa", "Lolium")),
    ("lilium", ("Lilium",)),
    ("solidago", ("Solidago",))
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a balanced pilot dataset from curated candidates.")
    parser.add_argument("--input", default="curated/images_v0.1.candidates.jsonl")
    parser.add_argument("--out", default="curated/images_v0.1.jsonl")
    parser.add_argument("--dataset-version", default="dataset_v0.1-pilot")
    parser.add_argument("--split", default="pilot")
    parser.add_argument("--max-per-group", type=int, default=3)
    return parser.parse_args()


def group_for(label: str) -> str | None:
    for group, prefixes in GROUP_RULES:
        if any(label.startswith(prefix) for prefix in prefixes):
            return group
    return None


def main() -> int:
    args = parse_args()
    counts = {group: 0 for group, _ in GROUP_RULES}
    rows = []
    seen_images = set()

    for row in read_jsonl(args.input):
        image_id = row.get("image_id")
        if image_id in seen_images:
            continue
        label = str(row.get("reference_species_label") or "")
        group = group_for(label)
        if group is None:
            continue
        if counts[group] >= args.max_per_group:
            continue
        row["pilot_group"] = group
        row["dataset_version"] = args.dataset_version
        row["dataset_split"] = args.split
        rows.append(row)
        seen_images.add(image_id)
        counts[group] += 1

    count = write_jsonl(args.out, rows)
    print(f"Wrote {count} balanced pilot records to {args.out}")
    for group, _ in GROUP_RULES:
        print(f"{group}: {counts[group]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
