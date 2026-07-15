from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "trials" / "analysis" / "calibration"
REQUIRED_COLUMNS = ["model", "feature", "feature_value", "prompt_type", "expected", "correct"]
MODEL_PREFIXES = ("openai/", "google/", "anthropic/")


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else ROOT / path


def source_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def canonical_model(model: str) -> str:
    model = str(model)
    for prefix in MODEL_PREFIXES:
        if model.startswith(prefix):
            return model.removeprefix(prefix)
    return model


def read_results(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"{source_label(path)} is missing columns: {', '.join(missing)}")
        df["source_file"] = source_label(path)
        frames.append(df)
    if not frames:
        raise ValueError("At least one input CSV is required")
    result = pd.concat(frames, ignore_index=True)
    result["model_raw"] = result["model"]
    result["model"] = result["model"].map(canonical_model)
    result["correct_bool"] = result["correct"].astype(str).str.lower().isin({"true", "1", "yes"})
    return result


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("correct_bool", "size"),
            correct_count=("correct_bool", "sum"),
            accuracy=("correct_bool", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values(group_cols).reset_index(drop=True)


def blind_mc_delta(summary: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    base_cols = [*group_cols, "prompt_type", "accuracy", "n", "correct_count"]
    pivot_source = summary.loc[
        summary["prompt_type"].isin(["blind_mc", "blind_mc_no_reference_photos"]),
        base_cols,
    ].copy()
    if pivot_source.empty:
        return pd.DataFrame(columns=[*group_cols, "blind_mc_accuracy", "no_reference_photos_accuracy", "accuracy_delta"])

    pivot = pivot_source.pivot_table(
        index=group_cols,
        columns="prompt_type",
        values="accuracy",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    if "blind_mc" not in pivot:
        pivot["blind_mc"] = pd.NA
    if "blind_mc_no_reference_photos" not in pivot:
        pivot["blind_mc_no_reference_photos"] = pd.NA
    pivot = pivot.rename(
        columns={
            "blind_mc": "blind_mc_accuracy",
            "blind_mc_no_reference_photos": "no_reference_photos_accuracy",
        }
    )
    pivot["accuracy_delta"] = pivot["no_reference_photos_accuracy"] - pivot["blind_mc_accuracy"]
    return pivot[[*group_cols, "blind_mc_accuracy", "no_reference_photos_accuracy", "accuracy_delta"]]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize calibration result CSVs and blind-MC ablations.")
    parser.add_argument("--input", required=True, nargs="+", help="Path to one or more calibration result CSVs.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for summary CSVs.")
    args = parser.parse_args()

    input_paths = [resolve_path(path_text) for path_text in args.input]
    out_dir = resolve_path(args.out_dir)
    df = read_results(input_paths)

    by_prompt = summarize(df, ["prompt_type"])
    by_model = summarize(df, ["model"])
    by_model_prompt = summarize(df, ["model", "prompt_type"])
    by_model_prompt_feature = summarize(df, ["model", "prompt_type", "feature"])
    model_delta = blind_mc_delta(by_model_prompt, ["model"])
    model_feature_delta = blind_mc_delta(by_model_prompt_feature, ["model", "feature"])

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "summary_by_prompt.csv": by_prompt,
        "summary_by_model.csv": by_model,
        "summary_by_model_prompt.csv": by_model_prompt,
        "summary_by_model_prompt_feature.csv": by_model_prompt_feature,
        "blind_mc_no_reference_photos_delta_by_model.csv": model_delta,
        "blind_mc_no_reference_photos_delta_by_model_feature.csv": model_feature_delta,
    }
    for filename, frame in outputs.items():
        frame.to_csv(out_dir / filename, index=False)

    print(f"Read {len(df)} rows from {len(input_paths)} input file(s)")
    for filename in outputs:
        print(f"Wrote {out_dir / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
