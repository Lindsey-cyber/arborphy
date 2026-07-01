from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "trials" / "analysis"

sys.path.insert(0, str(ROOT / "scripts"))
from stepwise_metrics import (  # noqa: E402
    BASE_GROUP_COLUMNS,
    FEATURE_GROUP_COLUMNS,
    annotate_results,
    metric_definitions,
    outcome_by_true_value,
    outcome_pairs,
    require_columns,
    summarize,
    whole_experiment_dashboard,
)


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else ROOT / path


def source_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def read_results(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        require_columns(df)
        df["source_file"] = source_label(path)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build stepwise per-trial rows, outcome summaries, and dashboard CSVs."
    )
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="Path to one or more stepwise result CSVs.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for analysis CSVs. Defaults to trials/analysis.",
    )
    args = parser.parse_args()

    input_paths = [resolve_path(path_text) for path_text in args.input]
    out_dir = resolve_path(args.out_dir)
    df = read_results(input_paths)
    annotated = annotate_results(df)
    overall = summarize(annotated, BASE_GROUP_COLUMNS)
    by_model_feature = summarize(annotated, FEATURE_GROUP_COLUMNS)
    by_true_value = outcome_by_true_value(annotated)
    dashboard = whole_experiment_dashboard(annotated)
    pairs = outcome_pairs(annotated)
    definitions = metric_definitions()

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "per_trial_rows.csv": annotated,
        "summary_overall.csv": overall,
        "summary_by_model_feature.csv": by_model_feature,
        "outcome_by_true_value.csv": by_true_value,
        "dashboard_whole_experiment.csv": dashboard,
        "outcome_pairs.csv": pairs,
        "metric_definitions.csv": definitions,
    }
    for filename, frame in outputs.items():
        frame.to_csv(out_dir / filename, index=False)

    print(f"Read {len(df)} rows from {len(input_paths)} input file(s)")
    for filename in outputs:
        print(f"Wrote {out_dir / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
