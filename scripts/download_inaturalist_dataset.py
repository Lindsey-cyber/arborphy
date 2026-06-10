#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


API_BASE = "https://api.inaturalist.org/v1"
USER_AGENT = "arborphy-inaturalist-dataset/0.2"

PHOTO_SIZES = ("original", "large", "medium", "small", "thumb", "square")
DEFAULT_LICENSES = ("cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa")
ALL_CC_LICENSES = DEFAULT_LICENSES + ("cc-by-nd", "cc-by-nc-nd")
PHOTO_SIZE_RE = re.compile(
    r"/(?:square|thumb|small|medium|large|original)\.([A-Za-z0-9]+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced iNaturalist image dataset in one step: "
            "discover popular species, fetch observations, and download photos."
        )
    )
    parser.add_argument("--out-dir", default="data/inaturalist_150x100")
    parser.add_argument("--dataset-version", default="inat_150x100_v0.1")
    parser.add_argument("--target-species", type=int, default=150)
    parser.add_argument("--images-per-species", type=int, default=100)
    parser.add_argument(
        "--iconic-taxa",
        default="Plantae",
        help="Broad iNaturalist group to search, e.g. Plantae, Aves, Insecta.",
    )
    parser.add_argument(
        "--all-iconic-taxa",
        action="store_true",
        help="Search all iNaturalist iconic taxa instead of limiting to --iconic-taxa.",
    )
    parser.add_argument(
        "--taxon-id",
        help="Optional broad iNaturalist taxon ID to limit species discovery.",
    )
    parser.add_argument("--place-id", help="Optional iNaturalist place ID.")
    parser.add_argument(
        "--quality-grade",
        default="research",
        choices=("research", "needs_id", "casual"),
    )
    parser.add_argument(
        "--licenses",
        default=",".join(DEFAULT_LICENSES),
        help=(
            "Comma-separated photo licenses. Use 'any' for any non-null photo "
            "license. Defaults avoid no-derivatives licenses."
        ),
    )
    parser.add_argument(
        "--include-nd-licenses",
        action="store_true",
        help="Include CC no-derivatives photo licenses in the default license set.",
    )
    parser.add_argument("--image-size", choices=PHOTO_SIZES, default="original")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--photos-per-observation", type=int, default=1)
    parser.add_argument("--species-candidate-pages", type=int, default=1)
    parser.add_argument("--max-observation-pages", type=int, default=5)
    parser.add_argument("--observations-per-page", type=int, default=200)
    parser.add_argument("--api-sleep", type=float, default=1.0)
    parser.add_argument("--image-sleep", type=float, default=0.1)
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Extra iNaturalist observation filter as key=value. Can be repeated.",
    )
    return parser.parse_args()


def csv_values(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def selected_licenses(args: argparse.Namespace) -> set[str] | None:
    if args.licenses.strip().lower() == "any":
        return None
    if args.include_nd_licenses and args.licenses == ",".join(DEFAULT_LICENSES):
        return set(ALL_CC_LICENSES)
    return {license_code.lower() for license_code in csv_values(args.licenses)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def api_get(
    endpoint: str,
    params: dict[str, str | int],
    api_sleep: float,
    retries: int = 5,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{API_BASE}{endpoint}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if api_sleep > 0:
                time.sleep(api_sleep)
            return payload
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {429, 500, 502, 503, 504}
            if not retryable or attempt == retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else float(2**attempt)
            print(f"API returned {exc.code}; waiting {wait:.1f}s", file=sys.stderr)
            time.sleep(wait)
        except urllib.error.URLError:
            if attempt == retries:
                raise
            time.sleep(float(2**attempt))
    raise RuntimeError("unreachable")


def extra_params(args: argparse.Namespace) -> dict[str, str]:
    params: dict[str, str] = {}
    for item in args.param:
        if "=" not in item:
            raise SystemExit(f"--param must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        params[key] = value
    return params


def base_filters(
    args: argparse.Namespace,
    licenses: set[str] | None,
) -> dict[str, str | int]:
    params: dict[str, str | int] = {
        "photos": "true",
        "photo_licensed": "true",
        "quality_grade": args.quality_grade,
    }
    if licenses:
        params["photo_license"] = ",".join(sorted(licenses))
    if args.place_id:
        params["place_id"] = args.place_id
    if args.taxon_id:
        params["taxon_id"] = args.taxon_id
    if args.iconic_taxa and not args.all_iconic_taxa:
        params["iconic_taxa"] = args.iconic_taxa
    params.update(extra_params(args))
    return params


def discover_species(
    args: argparse.Namespace,
    licenses: set[str] | None,
) -> list[dict[str, Any]]:
    species: list[dict[str, Any]] = []
    seen_taxa: set[int] = set()
    for page in range(1, args.species_candidate_pages + 1):
        params = base_filters(args, licenses)
        params.update({"rank": "species", "per_page": 500, "page": page})
        payload = api_get("/observations/species_counts", params, args.api_sleep)
        for item in payload.get("results", []):
            taxon = item.get("taxon") or {}
            taxon_id = taxon.get("id")
            count = int(item.get("count") or 0)
            if not taxon_id or taxon_id in seen_taxa:
                continue
            if count < args.images_per_species:
                continue
            seen_taxa.add(int(taxon_id))
            species.append(
                {
                    "taxon_id": int(taxon_id),
                    "taxon_name": taxon.get("name"),
                    "preferred_common_name": taxon.get("preferred_common_name"),
                    "rank": taxon.get("rank"),
                    "iconic_taxon_name": taxon.get("iconic_taxon_name"),
                    "observations_count": count,
                    "taxon": taxon,
                }
            )
    return species


def resize_photo_url(url: str, size: str) -> str:
    parts = urllib.parse.urlsplit(url)
    path = PHOTO_SIZE_RE.sub(f"/{size}.\\1", parts.path)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def best_photo_url(photo: dict[str, Any], size: str) -> str | None:
    keys = (f"{size}_url", "original_url", "large_url", "medium_url", "url")
    for key in keys:
        value = photo.get(key)
        if value:
            return resize_photo_url(str(value), size)
    return None


def extension_from_url(url: str) -> str:
    lower = urllib.parse.urlsplit(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if lower.endswith(ext):
            return ext
    return ".jpg"


def slugify(value: str | None, fallback: str) -> str:
    source = value or fallback
    slug = re.sub(r"[^A-Za-z0-9]+", "_", source).strip("_").lower()
    return slug or fallback


def photo_license(photo: dict[str, Any]) -> str | None:
    value = photo.get("license_code")
    return str(value).lower() if value else None


def license_allowed(
    photo: dict[str, Any],
    licenses: set[str] | None,
) -> bool:
    license_code = photo_license(photo)
    if not license_code:
        return False
    return licenses is None or license_code in licenses


def taxon_label(observation: dict[str, Any], species: dict[str, Any]) -> str:
    taxon = observation.get("taxon") or {}
    return str(taxon.get("name") or species.get("taxon_name") or "unknown")


def observation_url(observation: dict[str, Any]) -> str:
    if observation.get("uri"):
        return str(observation["uri"])
    return f"https://www.inaturalist.org/observations/{observation.get('id')}"


def download_photo(
    url: str,
    path: Path,
    overwrite: bool,
    image_sleep: float,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return "already_exists"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp_path = path.with_name(f"{path.name}.part")
    with urllib.request.urlopen(request, timeout=120) as response:
        tmp_path.write_bytes(response.read())
    tmp_path.replace(path)
    if image_sleep > 0:
        time.sleep(image_sleep)
    return "downloaded"


def image_record(
    args: argparse.Namespace,
    species: dict[str, Any],
    observation: dict[str, Any],
    photo: dict[str, Any],
    image_url: str,
    local_path: Path | None,
    download_status: str,
    image_index_for_species: int,
    run_started_at: str,
) -> dict[str, Any]:
    taxon = observation.get("taxon") or {}
    obs_id = observation.get("id")
    photo_id = photo.get("id") or image_index_for_species
    observation_taxon_name = taxon_label(observation, species)
    return {
        "dataset_version": args.dataset_version,
        "dataset_split": "train",
        "image_id": f"inat_{obs_id}_photo_{photo_id}",
        "image_source": "iNaturalist",
        "image_url": image_url,
        "image_size_requested": args.image_size,
        "local_image_path": str(local_path) if local_path else None,
        "download_status": download_status,
        "image_license": photo_license(photo),
        "photo_id": photo_id,
        "photo_attribution": photo.get("attribution"),
        "photo_index_for_species": image_index_for_species,
        "observation_id": obs_id,
        "observation_url": observation_url(observation),
        "observer": (observation.get("user") or {}).get("login"),
        "observed_on": observation.get("observed_on"),
        "created_at": observation.get("created_at"),
        "location_public": observation.get("location"),
        "quality_grade": observation.get("quality_grade"),
        "taxon_id": species.get("taxon_id"),
        "taxon_name": species.get("taxon_name"),
        "reference_species_label": species.get("taxon_name"),
        "target_taxon_id": species.get("taxon_id"),
        "target_taxon_name": species.get("taxon_name"),
        "target_common_name": species.get("preferred_common_name"),
        "observation_taxon_id": taxon.get("id"),
        "observation_taxon_name": observation_taxon_name,
        "observation_taxon_rank": taxon.get("rank"),
        "preferred_common_name": (
            taxon.get("preferred_common_name")
            or species.get("preferred_common_name")
        ),
        "taxon_rank": taxon.get("rank") or species.get("rank"),
        "iconic_taxon_name": (
            taxon.get("iconic_taxon_name")
            or species.get("iconic_taxon_name")
        ),
        "taxon_ancestor_ids": taxon.get("ancestor_ids"),
        "species_observation_count": species.get("observations_count"),
        "curated_at": run_started_at,
        "provenance_notes": (
            "Selected with iNaturalist API observation/species_counts and "
            "observation search; keep license and attribution with every image."
        ),
    }


def collect_species_images(
    args: argparse.Namespace,
    species: dict[str, Any],
    licenses: set[str] | None,
    out_dir: Path,
    run_started_at: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    raw_observations: list[dict[str, Any]] = []
    seen_photos: set[str] = set()
    seen_observations: set[int] = set()
    species_slug = slugify(str(species.get("taxon_name")), f"taxon_{species['taxon_id']}")

    for page in range(1, args.max_observation_pages + 1):
        if len(records) >= args.images_per_species:
            break
        params = base_filters(args, licenses)
        params.update(
            {
                "taxon_id": species["taxon_id"],
                "per_page": min(args.observations_per_page, 200),
                "page": page,
                "order_by": "votes",
                "order": "desc",
            }
        )
        payload = api_get("/observations", params, args.api_sleep)
        observations = payload.get("results", [])
        if not observations:
            break

        for observation in observations:
            if len(records) >= args.images_per_species:
                break
            obs_id = observation.get("id")
            photos_used_for_observation = 0
            for photo in observation.get("photos") or []:
                if photos_used_for_observation >= args.photos_per_observation:
                    break
                if not license_allowed(photo, licenses):
                    continue
                photo_id = str(photo.get("id") or "")
                image_key = f"{obs_id}:{photo_id}"
                if image_key in seen_photos:
                    continue
                image_url = best_photo_url(photo, args.image_size)
                if not image_url:
                    continue

                ext = extension_from_url(image_url)
                image_id = f"inat_{obs_id}_photo_{photo_id or len(records) + 1}"
                local_path = out_dir / "images" / species_slug / f"{image_id}{ext}"
                if args.metadata_only:
                    status = "metadata_only"
                    saved_path = None
                else:
                    try:
                        status = download_photo(
                            image_url,
                            local_path,
                            args.overwrite,
                            args.image_sleep,
                        )
                        saved_path = local_path
                    except Exception as exc:
                        status = f"error: {exc}"
                        saved_path = None

                records.append(
                    image_record(
                        args,
                        species,
                        observation,
                        photo,
                        image_url,
                        saved_path,
                        status,
                        len(records) + 1,
                        run_started_at,
                    )
                )
                seen_photos.add(image_key)
                photos_used_for_observation += 1
                if obs_id is not None and obs_id not in seen_observations:
                    raw_observations.append(observation)
                    seen_observations.add(int(obs_id))
    return records, raw_observations


def main() -> int:
    args = parse_args()
    if args.target_species < 1:
        raise SystemExit("--target-species must be at least 1")
    if args.images_per_species < 1:
        raise SystemExit("--images-per-species must be at least 1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = datetime.now(timezone.utc).isoformat()
    licenses = selected_licenses(args)

    run_config = {
        "api_base": API_BASE,
        "user_agent": USER_AGENT,
        "run_started_at": run_started_at,
        "dataset_version": args.dataset_version,
        "target_species": args.target_species,
        "images_per_species": args.images_per_species,
        "iconic_taxa": None if args.all_iconic_taxa else args.iconic_taxa,
        "taxon_id": args.taxon_id,
        "place_id": args.place_id,
        "quality_grade": args.quality_grade,
        "licenses": "any" if licenses is None else sorted(licenses),
        "image_size": args.image_size,
        "metadata_only": args.metadata_only,
    }
    write_json(out_dir / "run_config.json", run_config)

    print("Discovering popular species from iNaturalist...")
    candidates = discover_species(args, licenses)
    if not candidates:
        raise SystemExit("No species candidates found. Try broader filters.")
    print(f"Found {len(candidates)} candidate species with enough observations.")

    if args.dry_run:
        selected = candidates[: args.target_species]
        write_jsonl(out_dir / "species.jsonl", selected)
        for index, species in enumerate(selected, start=1):
            common = species.get("preferred_common_name") or ""
            print(
                f"{index:03d}. {species['taxon_name']} "
                f"{common} ({species['observations_count']} observations)"
            )
        print(f"Dry run wrote species list to {out_dir / 'species.jsonl'}")
        return 0

    selected_species: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    raw_observation_rows: list[dict[str, Any]] = []

    for species in candidates:
        if len(selected_species) >= args.target_species:
            break
        taxon_name = species.get("taxon_name")
        print(
            f"[{len(selected_species) + 1}/{args.target_species}] "
            f"{taxon_name}: collecting up to {args.images_per_species} images"
        )
        image_rows, observation_rows = collect_species_images(
            args,
            species,
            licenses,
            out_dir,
            run_started_at,
        )
        if len(image_rows) < args.images_per_species:
            print(
                f"  skipped: only found {len(image_rows)} usable images",
                file=sys.stderr,
            )
            continue
        selected_species.append(species)
        manifest_rows.extend(image_rows[: args.images_per_species])
        raw_observation_rows.extend(observation_rows)
        print(f"  kept {len(image_rows[: args.images_per_species])} images")

    write_jsonl(out_dir / "species.jsonl", selected_species)
    write_jsonl(out_dir / "manifest.jsonl", manifest_rows)
    write_jsonl(out_dir / "raw_observations.jsonl", raw_observation_rows)

    print(f"Wrote species to {out_dir / 'species.jsonl'}")
    print(f"Wrote manifest to {out_dir / 'manifest.jsonl'}")
    print(f"Wrote raw observations to {out_dir / 'raw_observations.jsonl'}")
    print(
        f"Finished {len(selected_species)} species and {len(manifest_rows)} images."
    )
    if len(selected_species) < args.target_species:
        print(
            "Warning: target species count was not reached. Increase "
            "--species-candidate-pages or --max-observation-pages.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
