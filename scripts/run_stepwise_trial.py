from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "newcomb_wildflower_guide" / "experiment_repro"
EXPERIMENT_OUTPUT_DIR = EXPERIMENT_DIR / "output"
ARTIFACT_DIR = ROOT / "trials" / "artifacts"
PROMPT_SET_DIR = EXPERIMENT_DIR / "prompt_sets"
DEFAULT_PROMPT_SET_ID = "stepwise-v1"
DEFAULT_IMAGE_SET = "sample.csv"
REQUIRED_SAMPLE_COLUMNS = [
    "newcomb_species_name",
    "species_inat",
    "taxon_id",
    "observation_id",
    "photo_id",
    "photo_url",
]
PRIMARY_FEATURES = ["key_flower_type", "key_plant_type", "key_leaf_type"]
FEATURE_ALIASES = {
    "primary": PRIMARY_FEATURES,
}

sys.path.insert(0, str(ROOT / "scripts" / "adapters"))
from openrouter_models import resolve_model_alias  # noqa: E402


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


def slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return result[:120] or "none"


def split_features(features: str) -> list[str]:
    feature_list: list[str] = []
    for feature in features.split(","):
        feature = feature.strip()
        if not feature:
            continue
        feature_list.extend(FEATURE_ALIASES.get(feature, [feature]))
    return feature_list


def available_prompt_sets() -> list[str]:
    if not PROMPT_SET_DIR.exists():
        return []
    return sorted(path.stem for path in PROMPT_SET_DIR.glob("*.json"))


def normalize_prompt_set(prompt_set: str) -> str:
    prompt_set = prompt_set.strip()
    if prompt_set.endswith(".json"):
        prompt_set = prompt_set.removesuffix(".json")
    if not prompt_set or Path(prompt_set).name != prompt_set:
        raise SystemExit(f"Invalid --prompt-set {prompt_set!r}. Use a prompt JSON name from {relative_or_absolute(PROMPT_SET_DIR)}.")
    available = available_prompt_sets()
    if prompt_set not in available:
        choices = ", ".join(available) or "(none)"
        raise SystemExit(f"Unknown --prompt-set {prompt_set!r}. Available prompt sets: {choices}")
    return prompt_set


def available_image_sets() -> list[str]:
    if not EXPERIMENT_OUTPUT_DIR.exists():
        return []
    available = []
    for path in sorted(EXPERIMENT_OUTPUT_DIR.glob("*.csv")):
        try:
            with path.open(newline="") as f:
                header = next(csv.reader(f))
        except (OSError, StopIteration):
            continue
        if all(column in header for column in REQUIRED_SAMPLE_COLUMNS):
            available.append(path.name)
    return available


def joined_features(features: str) -> str:
    return ",".join(split_features(features))


def normalize_image_set(image_set: str) -> str:
    image_set = image_set.strip()
    image_set_path = Path(image_set)
    if not image_set or image_set_path.name != image_set:
        raise SystemExit(
            f"Invalid --image-set {image_set!r}. Use a CSV filename from "
            f"{relative_or_absolute(EXPERIMENT_OUTPUT_DIR)}."
        )
    if image_set_path.suffix.lower() != ".csv":
        raise SystemExit(f"Invalid --image-set {image_set!r}. image_set must be a CSV filename.")
    return image_set


def image_set_path(image_set: str) -> Path:
    return EXPERIMENT_OUTPUT_DIR / normalize_image_set(image_set)


def validate_image_set_csv(image_set: str) -> None:
    path = image_set_path(image_set)
    if not path.exists():
        available = ", ".join(available_image_sets()) or "(none)"
        raise SystemExit(f"image_set CSV does not exist: {relative_or_absolute(path)}. Available CSVs: {available}")

    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise SystemExit(f"image_set CSV is empty: {relative_or_absolute(path)}") from exc

    missing = [column for column in REQUIRED_SAMPLE_COLUMNS if column not in header]
    if missing:
        raise SystemExit(
            f"image_set CSV {relative_or_absolute(path)} is missing required columns: {', '.join(missing)}"
        )


def normalize_sample_limit(sample_limit: str) -> str:
    sample_limit = sample_limit.strip()
    if sample_limit.lower() == "all":
        return "all"
    try:
        limit = int(sample_limit)
    except ValueError as exc:
        raise SystemExit("--sample-limit must be a positive integer or 'all'") from exc
    if limit < 1:
        raise SystemExit("--sample-limit must be a positive integer or 'all'")
    return str(limit)


def metadata_sample_limit(sample_limit: str) -> int | str:
    if sample_limit.lower() == "all":
        return "all"
    try:
        return int(sample_limit)
    except ValueError:
        return sample_limit


def relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def uv_executable() -> str:
    found = shutil.which("uv")
    if found:
        return found
    local = Path.home() / ".local" / "bin" / "uv"
    if local.exists():
        return str(local)
    return "uv"


def command_text(raw_args: list[str]) -> str:
    quoted_args = " ".join(shlex.quote(arg) for arg in raw_args)
    return f"uv run python scripts/run_stepwise_trial.py {quoted_args}".rstrip()


def default_trial_id(args: argparse.Namespace, model: str, sample_limit: str) -> str:
    if args.trial_id:
        return args.trial_id

    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    sample_slug = "all" if sample_limit.lower() == "all" else f"n{sample_limit}"
    feature_slug = slug("-".join(split_features(args.features)))
    return f"stepwise-{timestamp}-{slug(model)}-{slug(args.image_set)}-{sample_slug}-{feature_slug}"


def output_path(args: argparse.Namespace, trial_id: str) -> Path:
    if args.out_file:
        path = Path(args.out_file).expanduser()
        return path if path.is_absolute() else ARTIFACT_DIR / path
    return ARTIFACT_DIR / f"{slug(trial_id)}.csv"


def metadata_path_for(trial_id: str) -> Path:
    return ARTIFACT_DIR / f"{slug(trial_id)}.metadata.json"


def write_metadata(
    *,
    metadata_file: Path,
    trial_id: str,
    raw_args: list[str],
    model: str,
    args: argparse.Namespace,
    output_file: Path,
    sample_limit: str,
) -> None:
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "trial_id": trial_id,
        "command": command_text(raw_args),
        "image_set": args.image_set,
        "data_split": args.data_split,
        "prompt_set": args.prompt_set,
        "run_id": args.run_id,
        "model": model,
        "sample_limit": metadata_sample_limit(sample_limit),
        "features": split_features(args.features),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "mode": args.mode,
        "output_file": relative_or_absolute(output_file),
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "package_manager": "uv",
    }
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> int:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(description="Run a JM stepwise trial with concise options.")
    parser.add_argument("--model", default="openrouter/free", help="OpenRouter model id or local alias openrouter/free.")
    parser.add_argument("--sample-limit", default="1", help="Number of sample rows to run, or 'all'.")
    parser.add_argument(
        "--image-set",
        default=DEFAULT_IMAGE_SET,
        help="CSV filename under newcomb_wildflower_guide/experiment_repro/output, for example sample.csv.",
    )
    parser.add_argument(
        "--data-split",
        choices=["all"],
        default="all",
        help="Data split to run. Only 'all' is currently implemented until split manifests exist.",
    )
    parser.add_argument(
        "--features",
        default="key_flower_type",
        help="Comma-separated feature list, or alias 'primary' for the three Newcomb primary features.",
    )
    parser.add_argument(
        "--prompt-set",
        default=DEFAULT_PROMPT_SET_ID,
        help="Prompt JSON version from newcomb_wildflower_guide/experiment_repro/prompt_sets. Accepts names with or without .json.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Model sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=200, help="OpenRouter max_tokens value.")
    parser.add_argument("--run-id", default="run-001", help="Independent run label for repeated-run experiments.")
    parser.add_argument("--workers", default="1", help="Thread worker count.")
    parser.add_argument("--timeout", default="75", help="Seconds before one model call is treated as timed out.")
    parser.add_argument("--mode", choices=["command", "mock"], default="command", help="Model adapter mode.")
    parser.add_argument(
        "--out-file",
        default="",
        help="Optional output CSV filename. Relative paths are written under trials/artifacts.",
    )
    parser.add_argument(
        "--trial-id",
        default="",
        help="Optional trial id. Defaults to an auto-generated id based on time, model, sample limit, and features.",
    )
    raw_args = sys.argv[1:]
    if any(arg.strip() == "" for arg in raw_args):
        parser.error(
            "received a whitespace-only argument. If you used a multiline shell command, "
            "make sure each line-continuation backslash is the final character on its line."
        )
    args = parser.parse_args(raw_args)
    args.features = joined_features(args.features)
    args.image_set = normalize_image_set(args.image_set)
    args.prompt_set = normalize_prompt_set(args.prompt_set)
    sample_limit = normalize_sample_limit(args.sample_limit)
    validate_image_set_csv(args.image_set)

    model = args.model
    if args.mode == "command":
        model = resolve_model_alias(model, require_image=True)
    trial_id = default_trial_id(args, model, sample_limit)
    out_file = output_path(args, trial_id)
    metadata_file = metadata_path_for(trial_id)

    env = os.environ.copy()
    env["EXPERIMENT_MODEL_MODE"] = args.mode
    env["EXPERIMENT_MODELS"] = model
    env["EXPERIMENT_SAMPLE_LIMIT"] = sample_limit
    env["EXPERIMENT_FEATURES"] = args.features
    env["EXPERIMENT_IMAGE_SET"] = args.image_set
    env["EXPERIMENT_DATA_SPLIT"] = args.data_split
    env["EXPERIMENT_PROMPT_SET"] = args.prompt_set
    env["EXPERIMENT_RUN_ID"] = args.run_id
    env["EXPERIMENT_NUM_WORKERS"] = args.workers
    env["EXPERIMENT_MODEL_COMMAND_TIMEOUT"] = args.timeout
    env["EXPERIMENT_ARTIFACT_DIR"] = str(ARTIFACT_DIR)
    env["EXPERIMENT_OUT_FILE"] = str(out_file)
    env["EXPERIMENT_TRIAL_ID"] = trial_id
    env["OPENROUTER_TIMEOUT"] = args.timeout
    env["OPENROUTER_TEMPERATURE"] = str(args.temperature)
    env["OPENROUTER_MAX_TOKENS"] = str(args.max_tokens)

    uv = uv_executable()
    if args.mode == "command":
        env["EXPERIMENT_MODEL_COMMAND"] = (
            f"{shlex.quote(uv)} run python ../../scripts/adapters/openrouter_command_adapter.py"
        )

    print("Running JM stepwise trial")
    print(f"  trial id: {trial_id}")
    print(f"  image set: {args.image_set}")
    print(f"  data split: {args.data_split}")
    print(f"  prompt set: {args.prompt_set}")
    print(f"  run id: {args.run_id}")
    print(f"  model: {model}")
    print(f"  sample limit: {sample_limit}")
    print(f"  features: {args.features}")
    print(f"  temperature: {args.temperature}")
    print(f"  max tokens: {args.max_tokens}")
    print(f"  workers: {args.workers}")
    print(f"  timeout: {args.timeout}s per model call")
    print(f"  output: {relative_or_absolute(out_file)}")
    print(f"  metadata: {relative_or_absolute(metadata_file)}")
    sys.stdout.flush()

    result = subprocess.run(
        [uv, "run", "python", "run_stepwise_local.py"],
        cwd=EXPERIMENT_DIR,
        env=env,
        check=False,
    )
    write_metadata(
        metadata_file=metadata_file,
        trial_id=trial_id,
        raw_args=raw_args,
        model=model,
        args=args,
        output_file=out_file,
        sample_limit=sample_limit,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
