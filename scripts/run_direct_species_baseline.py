from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import re
import sys
import threading
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "newcomb_wildflower_guide" / "experiment_repro"
EXPERIMENT_OUTPUT_DIR = EXPERIMENT_DIR / "output"
ARTIFACT_DIR = ROOT / "trials" / "artifacts"
DEFAULT_IMAGE_SET = "sample.csv"
DEFAULT_MODELS = [
    "openai/gpt-4o-mini",
    "openai/gpt-5-mini",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-haiku-4-5",
]
REQUIRED_SAMPLE_COLUMNS = [
    "newcomb_species_name",
    "species_inat",
    "taxon_id",
    "observation_id",
    "photo_id",
    "photo_url",
]

sys.path.insert(0, str(EXPERIMENT_DIR))
sys.path.insert(0, str(ROOT / "scripts" / "adapters"))
from model_adapter import call_model  # noqa: E402
from openrouter_models import model_pricing_snapshot  # noqa: E402
from openrouter_usage import (  # noqa: E402
    DEFAULT_MODEL_COST_WARN_USD,
    DEFAULT_REQUEST_COST_WARN_USD,
    DEFAULT_TOKEN_WARN,
    DEFAULT_TOTAL_COST_WARN_USD,
    UsageLogMonitor,
    aggregate_usage_log_full,
    format_money,
    format_tokens,
)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return result[:120] or "none"


def split_models(model_arg: str) -> list[str]:
    models = [model.strip() for model in model_arg.split(",") if model.strip()]
    if not models:
        raise SystemExit("--model must include at least one model id")
    return models


def normalize_sample_limit(sample_limit: str) -> int | str:
    sample_limit = sample_limit.strip()
    if sample_limit.lower() == "all":
        return "all"
    try:
        limit = int(sample_limit)
    except ValueError as exc:
        raise SystemExit("--sample-limit must be a positive integer or 'all'") from exc
    if limit < 1:
        raise SystemExit("--sample-limit must be a positive integer or 'all'")
    return limit


def resolve_image_set(image_set: str) -> Path:
    image_set = image_set.strip()
    image_path = Path(image_set)
    if not image_set or image_path.name != image_set:
        raise SystemExit(f"--image-set must be a CSV filename under {EXPERIMENT_OUTPUT_DIR}")
    if image_path.suffix.lower() != ".csv":
        raise SystemExit("--image-set must be a CSV filename")
    path = EXPERIMENT_OUTPUT_DIR / image_set
    if not path.exists():
        raise SystemExit(f"image_set CSV does not exist: {path}")
    return path


def read_sample(path: Path, sample_limit: int | str) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [column for column in REQUIRED_SAMPLE_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"image_set CSV is missing required columns: {', '.join(missing)}")
        rows = list(reader)
    if sample_limit == "all":
        return rows
    return rows[:sample_limit]


def species_candidates(rows: list[dict[str, str]]) -> list[str]:
    species = sorted({row["species_inat"].strip() for row in rows if row.get("species_inat", "").strip()})
    if not species:
        raise SystemExit("image_set CSV has no species_inat values")
    return species


def direct_species_parts(photo_url: str, candidates: list[str]) -> list[Any]:
    option_lines = "\n".join(f"{index}. {species}" for index, species in enumerate(candidates, 1))
    return [
        (
            "Identify the plant species in the image using only the candidate list below.\n"
            "Return exactly one candidate species name, or INCONCLUSIVE if the image is not sufficient.\n\n"
            f"Candidate species:\n{option_lines}"
        ),
        {"image": photo_url},
    ]


def parse_species(raw: str, candidates: list[str]) -> str:
    text = raw.strip()
    lowered = text.casefold()
    if not text:
        return "INCONCLUSIVE"
    if "inconclusive" in lowered or "cannot determine" in lowered or "not sufficient" in lowered:
        return "INCONCLUSIVE"

    for species in candidates:
        if text.casefold() == species.casefold():
            return species
    for species in candidates:
        if species.casefold() in lowered:
            return species

    match = re.match(r"^\s*(\d+)", text)
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(candidates):
            return candidates[index]

    return text.splitlines()[0][:200]


def output_paths(trial_id: str, out_file: str) -> tuple[Path, Path, Path]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    if out_file:
        raw_path = Path(out_file).expanduser()
        if not raw_path.is_absolute():
            raw_path = ARTIFACT_DIR / raw_path
    else:
        raw_path = ARTIFACT_DIR / f"{trial_id}.direct_species.csv"
    summary_path = raw_path.with_name(f"{raw_path.stem}.summary.csv")
    metadata_path = raw_path.with_name(f"{raw_path.stem}.metadata.json")
    return raw_path, summary_path, metadata_path


def existing_done(raw_path: Path) -> set[tuple[str, str, str, int]]:
    if not raw_path.exists():
        return set()
    with raw_path.open(newline="") as f:
        return {
            (row["trial_id"], row["model"], str(row["observation_id"]), int(row["run_index"]))
            for row in csv.DictReader(f)
        }


def append_row(path: Path, row: dict[str, object], fieldnames: list[str]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_summary(raw_path: Path, summary_path: Path, models: list[str], runs: int) -> None:
    with raw_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    summary_rows = []
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        n_predictions = len(model_rows)
        correct_count = sum(row["correct"].lower() == "true" for row in model_rows)
        by_obs: dict[str, list[str]] = {}
        for row in model_rows:
            by_obs.setdefault(str(row["observation_id"]), []).append(row["parsed_species"])
        complete_obs = {obs: vals for obs, vals in by_obs.items() if len(vals) == runs}
        consistent_count = sum(len(set(vals)) == 1 for vals in complete_obs.values())
        summary_rows.append(
            {
                "model": model,
                "n_predictions": n_predictions,
                "n_images": len(by_obs),
                "runs": runs,
                "correct_count": correct_count,
                "accuracy": correct_count / n_predictions if n_predictions else "",
                "consistency_count": consistent_count,
                "consistency_denominator": len(complete_obs),
                "consistency": consistent_count / len(complete_obs) if complete_obs else "",
            }
        )

    with summary_path.open("w", newline="") as f:
        fieldnames = [
            "model",
            "n_predictions",
            "n_images",
            "runs",
            "correct_count",
            "accuracy",
            "consistency_count",
            "consistency_denominator",
            "consistency",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def write_metadata(
    metadata_path: Path,
    *,
    trial_id: str,
    args: argparse.Namespace,
    models: list[str],
    sample_path: Path,
    raw_path: Path,
    summary_path: Path,
    candidate_species: list[str],
    sample_row_count: int,
    usage_log_file: Path,
    price_snapshot: list[dict[str, object]],
) -> None:
    usage = aggregate_usage_log_full(usage_log_file)
    metadata = {
        "trial_id": trial_id,
        "experiment": "direct_species_baseline",
        "image_set": args.image_set,
        "sample_limit": args.sample_limit,
        "runs": args.runs,
        "models": models,
        "mode": args.mode,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "workers": args.workers,
        "sample_file": str(sample_path),
        "output_file": str(raw_path),
        "summary_file": str(summary_path),
        "candidate_species": candidate_species,
        "budget": {
            "sample_rows": sample_row_count,
            "model_count": len(models),
            "runs": args.runs,
            "expected_model_calls": sample_row_count * len(models) * args.runs,
            "openrouter_usage_log": str(usage_log_file),
            "openrouter_usage": usage["overall"],
            "openrouter_usage_by_model": usage["by_model"],
            "openrouter_price_snapshot": price_snapshot,
            "openrouter_budget_alerts": {
                "enabled": args.mode == "openrouter" and not args.no_budget_alerts,
                "total_cost_interval_usd": args.budget_warn_usd,
                "model_cost_interval_usd": args.model_budget_warn_usd,
                "total_token_interval": args.token_warn,
                "model_token_interval": args.model_token_warn,
                "request_cost_warn_usd": args.request_cost_warn_usd,
            },
            "cost_note": (
                "OpenRouter usage.cost is the credits charged to the account, treated here as "
                "USD-equivalent for experiment budgeting. Response usage is authoritative."
            ),
        },
        "python": sys.version.split()[0],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run John-style direct species identification baseline.")
    parser.add_argument("--model", default=",".join(DEFAULT_MODELS), help="Comma-separated model ids.")
    parser.add_argument("--image-set", default=DEFAULT_IMAGE_SET, help="CSV filename under experiment_repro/output.")
    parser.add_argument("--sample-limit", default="all", help="Number of image rows to run, or 'all'.")
    parser.add_argument("--runs", type=int, default=3, help="Independent runs per image.")
    parser.add_argument("--mode", choices=["openrouter", "mock"], default="openrouter", help="Model adapter mode.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel model calls.")
    parser.add_argument("--timeout", default="120", help="OpenRouter timeout seconds.")
    parser.add_argument("--temperature", type=float, default=0.0, help="OpenRouter temperature.")
    parser.add_argument("--max-tokens", type=int, default=120, help="OpenRouter max_tokens.")
    parser.add_argument(
        "--budget-warn-usd",
        type=float,
        default=env_float("OPENROUTER_BUDGET_WARN_USD", DEFAULT_TOTAL_COST_WARN_USD),
        help="Print a running alert each time total OpenRouter cost crosses this many USD-equivalent credits.",
    )
    parser.add_argument(
        "--model-budget-warn-usd",
        type=float,
        default=env_float("OPENROUTER_MODEL_BUDGET_WARN_USD", DEFAULT_MODEL_COST_WARN_USD),
        help="Print a running alert each time any single model crosses this many USD-equivalent credits.",
    )
    parser.add_argument(
        "--token-warn",
        type=int,
        default=env_int("OPENROUTER_TOKEN_WARN", DEFAULT_TOKEN_WARN),
        help="Print a running alert each time total OpenRouter tokens cross this interval.",
    )
    parser.add_argument(
        "--model-token-warn",
        type=int,
        default=env_int("OPENROUTER_MODEL_TOKEN_WARN", DEFAULT_TOKEN_WARN),
        help="Print a running alert each time any single model crosses this token interval.",
    )
    parser.add_argument(
        "--request-cost-warn-usd",
        type=float,
        default=env_float("OPENROUTER_REQUEST_COST_WARN_USD", DEFAULT_REQUEST_COST_WARN_USD),
        help="Print a running alert for any single OpenRouter call at or above this USD-equivalent cost.",
    )
    parser.add_argument(
        "--no-budget-alerts",
        action="store_true",
        help="Disable running OpenRouter budget/token alerts while keeping usage logging enabled.",
    )
    parser.add_argument("--trial-id", default="", help="Stable trial id. Defaults to timestamp.")
    parser.add_argument("--out-file", default="", help="Optional raw output CSV filename under trials/artifacts.")
    args = parser.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be positive")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    for attr in ("budget_warn_usd", "model_budget_warn_usd", "request_cost_warn_usd"):
        if getattr(args, attr) < 0:
            raise SystemExit(f"--{attr.replace('_', '-')} must be zero or positive")
    for attr in ("token_warn", "model_token_warn"):
        if getattr(args, attr) < 0:
            raise SystemExit(f"--{attr.replace('_', '-')} must be zero or positive")

    os.environ["EXPERIMENT_MODEL_MODE"] = args.mode
    os.environ["OPENROUTER_TIMEOUT"] = str(args.timeout)
    os.environ["OPENROUTER_TEMPERATURE"] = str(args.temperature)
    os.environ["OPENROUTER_MAX_TOKENS"] = str(args.max_tokens)

    models = split_models(args.model)
    sample_limit = normalize_sample_limit(args.sample_limit)
    sample_path = resolve_image_set(args.image_set)
    all_sample_rows = read_sample(sample_path, "all")
    sample_rows = all_sample_rows if sample_limit == "all" else all_sample_rows[:sample_limit]
    candidates = species_candidates(all_sample_rows)
    trial_id = args.trial_id or f"direct-species-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    raw_path, summary_path, metadata_path = output_paths(trial_id, args.out_file)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    usage_log_file = raw_path.with_name(f"{raw_path.stem}.openrouter_usage.jsonl")
    os.environ["OPENROUTER_USAGE_LOG"] = str(usage_log_file)
    price_snapshot = model_pricing_snapshot(models) if args.mode == "openrouter" else []
    done = existing_done(raw_path)

    fieldnames = [
        "trial_id",
        "model",
        "run_index",
        "image_set",
        "sample_limit",
        "observation_id",
        "photo_id",
        "photo_url",
        "taxon_id",
        "species_inat",
        "newcomb_species_name",
        "candidate_species_json",
        "prompt_parts_json",
        "raw_response",
        "parsed_species",
        "parse_rule",
        "correct",
    ]

    tasks = []
    for model in models:
        for run_index in range(1, args.runs + 1):
            for row in sample_rows:
                key = (trial_id, model, str(row["observation_id"]), run_index)
                if key not in done:
                    tasks.append((model, run_index, row))

    print("Running direct species baseline")
    print(f"  trial id: {trial_id}")
    print(f"  image set: {args.image_set} ({sample_path})")
    print(f"  sample rows: {len(sample_rows)}")
    print(f"  candidate species: {len(candidates)}")
    print(f"  models: {', '.join(models)}")
    print(f"  runs per image: {args.runs}")
    print(f"  pending calls: {len(tasks)}")
    if args.mode == "openrouter":
        print(
            "  cost alerts: "
            f"total every {format_money(args.budget_warn_usd) or '$0'}, "
            f"per model every {format_money(args.model_budget_warn_usd) or '$0'}, "
            f"tokens every {format_tokens(args.token_warn) or '0'}, "
            f"single call at {format_money(args.request_cost_warn_usd) or '$0'}"
        )
        print("  OpenRouter prices:")
        for row in price_snapshot:
            if row.get("pricing_error"):
                print(f"    {row.get('model')}: {row.get('pricing_error')}")
            else:
                print(
                    f"    {row.get('model')}: "
                    f"{format_money(row.get('prompt_usd_per_million')) or 'unknown'}/1M prompt, "
                    f"{format_money(row.get('completion_usd_per_million')) or 'unknown'}/1M completion "
                    f"({row.get('name') or row.get('openrouter_id') or row.get('model')})"
                )
    print(f"  output: {raw_path}")
    print(f"  summary: {summary_path}")
    sys.stdout.flush()

    write_lock = threading.Lock()

    def process_task(model: str, run_index: int, row: dict[str, str]) -> None:
        parts = direct_species_parts(row["photo_url"], candidates)
        raw = call_model(model, parts)
        parsed = parse_species(raw, candidates)
        output_row = {
            "trial_id": trial_id,
            "model": model,
            "run_index": run_index,
            "image_set": args.image_set,
            "sample_limit": args.sample_limit,
            "observation_id": row["observation_id"],
            "photo_id": row["photo_id"],
            "photo_url": row["photo_url"],
            "taxon_id": row["taxon_id"],
            "species_inat": row["species_inat"],
            "newcomb_species_name": row["newcomb_species_name"],
            "candidate_species_json": json.dumps(candidates),
            "prompt_parts_json": json.dumps(parts, ensure_ascii=False),
            "raw_response": raw,
            "parsed_species": parsed,
            "parse_rule": "exact candidate; substring candidate; leading option number; INCONCLUSIVE keywords; else first line",
            "correct": parsed.casefold() == row["species_inat"].strip().casefold(),
        }
        with write_lock:
            append_row(raw_path, output_row, fieldnames)

    error_count = 0
    monitor = None
    if args.mode == "openrouter" and not args.no_budget_alerts:
        monitor = UsageLogMonitor(
            usage_log_file,
            total_cost_interval_usd=args.budget_warn_usd,
            model_cost_interval_usd=args.model_budget_warn_usd,
            total_token_interval=args.token_warn,
            model_token_interval=args.model_token_warn,
            request_cost_warn_usd=args.request_cost_warn_usd,
            emit=lambda message: print(message, flush=True),
        )
        monitor.start()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_task, model, run_index, row) for model, run_index, row in tasks]
            for index, future in enumerate(as_completed(futures), 1):
                exc = future.exception()
                if exc:
                    error_count += 1
                    print(f"  [ERROR] {exc}", flush=True)
                elif index % 10 == 0 or index == len(futures):
                    print(f"  completed {index}/{len(futures)}", flush=True)
    finally:
        if monitor is not None:
            monitor.stop()

    if raw_path.exists():
        write_summary(raw_path, summary_path, models, args.runs)
    write_metadata(
        metadata_path,
        trial_id=trial_id,
        args=args,
        models=models,
        sample_path=sample_path,
        raw_path=raw_path,
        summary_path=summary_path,
        candidate_species=candidates,
        sample_row_count=len(sample_rows),
        usage_log_file=usage_log_file,
        price_snapshot=price_snapshot,
    )

    if error_count:
        print(f"Done with {error_count} errors.")
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
