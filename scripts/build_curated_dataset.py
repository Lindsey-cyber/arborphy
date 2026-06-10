#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.jsonl import read_jsonl, write_jsonl


DEFAULT_ALLOWED_LICENSES = {
    "cc0",
    "cc-by",
    "cc-by-sa",
    "cc-by-nc",
    "cc-by-nc-sa"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a curated image manifest from raw iNaturalist observations.")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        help="Raw iNaturalist JSONL input. Can be repeated."
    )
    parser.add_argument("--out", default="curated/images_v0.1.jsonl")
    parser.add_argument("--dataset-version", default="dataset_v0.1-pilot")
    parser.add_argument("--split", default="pilot")
    parser.add_argument(
        "--allowed-license",
        action="append",
        default=[],
        help="Allowed photo license code. Repeat to override defaults."
    )
    parser.add_argument("--max-images", type=int)
    return parser.parse_args()


def best_photo_url(photo: dict[str, Any]) -> str | None:
    for key in ("original_url", "large_url", "medium_url", "url"):
        value = photo.get(key)
        if value:
            return str(value)
    return None


def taxon_label(observation: dict[str, Any]) -> str:
    taxon = observation.get("taxon") or {}
    if taxon.get("name"):
        return str(taxon["name"])
    if observation.get("species_guess"):
        return str(observation["species_guess"])
    return "unknown"


def observation_url(observation: dict[str, Any]) -> str:
    if observation.get("uri"):
        return str(observation["uri"])
    return f"https://www.inaturalist.org/observations/{observation.get('id')}"


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    allowed = set(args.allowed_license) if args.allowed_license else DEFAULT_ALLOWED_LICENSES
    curated_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []

    input_paths = args.input or ["raw/inaturalist_observations_v0.1.jsonl"]
    for input_path in input_paths:
      for observation in read_jsonl(input_path):
        photos = observation.get("photos") or []
        for photo in photos:
            license_code = (photo.get("license_code") or observation.get("license_code") or "").lower()
            if license_code and license_code not in allowed:
                continue
            photo_id = photo.get("id") or len(rows) + 1
            image_url = best_photo_url(photo)
            if not image_url:
                continue
            obs_id = observation.get("id")
            rows.append(
                {
                    "image_id": f"inat_{obs_id}_photo_{photo_id}",
                    "image_source": "iNaturalist",
                    "observation_id": obs_id,
                    "observation_url": observation_url(observation),
                    "observer": (observation.get("user") or {}).get("login"),
                    "observed_on": observation.get("observed_on"),
                    "location_public": observation.get("location"),
                    "taxon_id": (observation.get("taxon") or {}).get("id"),
                    "reference_species_label": taxon_label(observation),
                    "identification_status": observation.get("quality_grade"),
                    "image_url": image_url,
                    "image_license": license_code or None,
                    "dataset_version": args.dataset_version,
                    "dataset_split": args.split,
                    "curated_at": curated_at,
                    "provenance_notes": "Derived from public iNaturalist observation metadata; verify license before publication."
                }
            )
            if args.max_images and len(rows) >= args.max_images:
                return rows
    return rows


def main() -> int:
    args = parse_args()
    rows = build_rows(args)
    count = write_jsonl(args.out, rows)
    print(f"Wrote {count} curated image records to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
