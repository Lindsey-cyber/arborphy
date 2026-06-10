#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.jsonl import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download curated benchmark images for local review.")
    parser.add_argument("--curated", default="curated/images_v0.1.jsonl")
    parser.add_argument("--out-dir", default="data/images/dataset_v0.1-pilot")
    parser.add_argument("--manifest-out", default="curated/images_v0.1.local.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep", type=float, default=0.2)
    return parser.parse_args()


def extension_from_url(url: str) -> str:
    lower = url.lower().split("?", 1)[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if lower.endswith(ext):
            return ext
    return ".jpg"


def download(url: str, path: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "botanic-codex-benchmark/0.1"}
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        path.write_bytes(response.read())


def main() -> int:
    args = parse_args()
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    downloaded_at = datetime.now(timezone.utc).isoformat()

    for index, row in enumerate(read_jsonl(args.curated), start=1):
        if args.limit and index > args.limit:
            break
        image_url = row.get("image_url")
        image_id = row.get("image_id")
        if not image_url or not image_id:
            row["local_image_path"] = None
            row["download_status"] = "missing_url_or_id"
            rows.append(row)
            continue
        ext = extension_from_url(str(image_url))
        local_path = output_dir / f"{image_id}{ext}"
        try:
            if not local_path.exists():
                download(str(image_url), local_path)
                time.sleep(args.sleep)
            row["local_image_path"] = str(local_path)
            row["download_status"] = "downloaded"
            row["image_downloaded_at"] = downloaded_at
        except Exception as exc:
            row["local_image_path"] = None
            row["download_status"] = f"error: {exc}"
        rows.append(row)

    count = write_jsonl(args.manifest_out, rows)
    print(f"Wrote {count} local image manifest records to {args.manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
