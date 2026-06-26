from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "trials" / "analysis"

BASE_GROUP_COLUMNS = ["trial_id", "model"]
FEATURE_GROUP_COLUMNS = ["trial_id", "model", "feature"]


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def normalize_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return normalize_text(series).str.lower().isin({"true", "1", "yes", "y"})


def rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise SystemExit(f"Input CSV is missing required columns: {', '.join(missing)}")


def summarize_group(group: pd.DataFrame, source_file: str) -> dict[str, object]:
    p1 = normalize_text(group["p1_parsed"]).str.upper()
    p2 = normalize_text(group["p2_parsed"])
    true_value = normalize_text(group["true_value"])
    committed = normalize_bool(group["committed"])

    n_rows = len(group)
    n_observations = group["observation_id"].astype(str).nunique() if "observation_id" in group.columns else None

    sees_feature = p1.eq("YES")
    inconclusive = p2.str.upper().eq("INCONCLUSIVE")
    correct_value = p2.str.casefold().eq(true_value.str.casefold()) & ~inconclusive
    wrong = committed & ~correct_value

    sees_feature_count = int(sees_feature.sum())
    committed_count = int(committed.sum())
    inc_count = int(inconclusive.sum())
    correct_count = int(correct_value.sum())
    wrong_count = int(wrong.sum())

    return {
        "source_file": source_file,
        "n_rows": n_rows,
        "n_observations": n_observations,
        "sees_feature_count": sees_feature_count,
        "sees_feature_rate": rate(sees_feature_count, n_rows),
        "committed_count": committed_count,
        "commitment_rate": rate(committed_count, n_rows),
        "inc_count": inc_count,
        "inc_rate": rate(inc_count, n_rows),
        "correct_count": correct_count,
        "overall_correct_rate": rate(correct_count, n_rows),
        "wrong_count": wrong_count,
        "wrong_rate": rate(wrong_count, n_rows),
        "committed_accuracy": rate(correct_count, committed_count),
    }


def summarize(df: pd.DataFrame, group_columns: list[str], source_file: str) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(summarize_group(group, source_file))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize JM/Newcomb stepwise result CSVs.")
    parser.add_argument("--input", required=True, help="Path to one stepwise result CSV.")
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for summary CSVs. Defaults to trials/analysis.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    df = pd.read_csv(input_path)
    require_columns(
        df,
        [
            "trial_id",
            "model",
            "feature",
            "p1_parsed",
            "p2_parsed",
            "true_value",
            "committed",
        ],
    )

    source_file = input_path.resolve().relative_to(ROOT).as_posix() if input_path.resolve().is_relative_to(ROOT) else str(input_path)
    overall = summarize(df, BASE_GROUP_COLUMNS, source_file)
    by_model_feature = summarize(df, FEATURE_GROUP_COLUMNS, source_file)

    out_dir.mkdir(parents=True, exist_ok=True)
    overall_path = out_dir / "summary_overall.csv"
    by_model_feature_path = out_dir / "summary_by_model_feature.csv"
    overall.to_csv(overall_path, index=False)
    by_model_feature.to_csv(by_model_feature_path, index=False)

    print(f"Read {len(df)} rows from {source_file}")
    print(f"Wrote {overall_path}")
    print(f"Wrote {by_model_feature_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
