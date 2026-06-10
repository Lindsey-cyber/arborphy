#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.jsonl import write_jsonl


API_URL = "https://api.inaturalist.org/v1/observations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download public iNaturalist observation records to JSONL.")
    parser.add_argument("--taxon-name", help="Taxon name query, for example Trillium.")
    parser.add_argument("--taxon-id", help="iNaturalist taxon ID.")
    parser.add_argument("--place-id", help="iNaturalist place ID.")
    parser.add_argument("--quality-grade", default="research", help="iNaturalist quality grade.")
    parser.add_argument("--per-page", type=int, default=30)
    parser.add_argument("--pages", type=int, default=1)
    parser.add_argument("--out", default="raw/inaturalist_observations_v0.1.jsonl")
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Extra API parameter as key=value. Can be repeated."
    )
    return parser.parse_args()


def build_params(args: argparse.Namespace, page: int) -> dict[str, str | int]:
    params: dict[str, str | int] = {
        "photos": "true",
        "quality_grade": args.quality_grade,
        "per_page": args.per_page,
        "page": page,
        "order": "desc",
        "order_by": "created_at"
    }
    if args.taxon_name:
        params["taxon_name"] = args.taxon_name
    if args.taxon_id:
        params["taxon_id"] = args.taxon_id
    if args.place_id:
        params["place_id"] = args.place_id
    for item in args.param:
        if "=" not in item:
            raise SystemExit(f"--param must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        params[key] = value
    return params


def fetch_page(params: dict[str, str | int]) -> dict:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{API_URL}?{query}",
        headers={"User-Agent": "botanic-codex-benchmark/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    args = parse_args()
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for page in range(1, args.pages + 1):
        params = build_params(args, page)
        payload = fetch_page(params)
        for observation in payload.get("results", []):
            observation["_botanic_codex"] = {
                "downloaded_at": fetched_at,
                "api_url": API_URL,
                "query_params": params
            }
            rows.append(observation)
        time.sleep(0.2)

    count = write_jsonl(args.out, rows)
    print(f"Wrote {count} observations to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

