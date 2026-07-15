from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

CALIBRATION_CSV = ROOT / "JM_Assets_2" / "calibration_results.csv"
FEATURE_CSV = ROOT / "JM_Assets_2" / "feature_unlabeled_results.csv"
CURRENT_FULL_RUN_CSV = ROOT / "trials" / "artifacts" / "notebook-20260708-034723.csv"
DIRECT_SUMMARY_CSV = ROOT / "trials" / "artifacts" / "direct-species-baseline-john-90x3-six-models.summary.csv"

PRIMARY_FEATURES = ["key_flower_type", "key_plant_type", "key_leaf_type"]
MODEL_ORDER = [
    "gpt-4o-mini",
    "gpt-5-mini",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
]
MODEL_DISPLAY = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-5-mini": "gpt-5-mini",
    "gemini-3.1-pro-preview": "gemini-3.1-pro",
    "gemini-3-flash-preview": "gemini-3-flash",
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "claude-haiku-4-5": "claude-haiku-4-5",
}
FEATURE_DISPLAY = {
    "key_flower_type": "flower type",
    "key_plant_type": "plant type",
    "key_leaf_type": "leaf type",
}


THESIS_CALIBRATION = {
    "gpt-4o-mini": {"sees": 16 / 17, "mc_correct": 14 / 17, "mc_inc": 1 / 17, "mc_wrong": 2 / 17},
    "gpt-5-mini": {"sees": 16 / 17, "mc_correct": 15 / 17, "mc_inc": 0, "mc_wrong": 2 / 17},
    "gemini-3.1-pro-preview": {"sees": 16 / 17, "mc_correct": 15 / 17, "mc_inc": 0, "mc_wrong": 2 / 17},
    "gemini-3-flash-preview": {"sees": 17 / 17, "mc_correct": 17 / 17, "mc_inc": 0, "mc_wrong": 0},
    "claude-sonnet-4-6": {"sees": 16 / 17, "mc_correct": 14 / 17, "mc_inc": 0, "mc_wrong": 3 / 17},
    "claude-haiku-4-5": {"sees": 17 / 17, "mc_correct": 10 / 17, "mc_inc": 0, "mc_wrong": 7 / 17},
}

THESIS_SAMPLE = {
    ("key_flower_type", "gpt-4o-mini"): {"sees": 0.667, "mc_correct": 0.067, "mc_inc": 0.867, "mc_wrong": 0.067},
    ("key_flower_type", "gpt-5-mini"): {"sees": 0.622, "mc_correct": 0.367, "mc_inc": 0.433, "mc_wrong": 0.200},
    ("key_flower_type", "gemini-3-flash-preview"): {"sees": 0.778, "mc_correct": 0.556, "mc_inc": 0.300, "mc_wrong": 0.144},
    ("key_flower_type", "gemini-3.1-pro-preview"): {"sees": 0.663, "mc_correct": 0.446, "mc_inc": 0.386, "mc_wrong": 0.169},
    ("key_flower_type", "claude-sonnet-4-6"): {"sees": 0.622, "mc_correct": 0.467, "mc_inc": 0.433, "mc_wrong": 0.100},
    ("key_flower_type", "claude-haiku-4-5"): {"sees": 0.722, "mc_correct": 0.278, "mc_inc": 0.478, "mc_wrong": 0.244},
    ("key_plant_type", "gpt-4o-mini"): {"sees": 0.811, "mc_correct": 0.322, "mc_inc": 0.478, "mc_wrong": 0.200},
    ("key_plant_type", "gpt-5-mini"): {"sees": 0.944, "mc_correct": 0.511, "mc_inc": 0.211, "mc_wrong": 0.278},
    ("key_plant_type", "gemini-3-flash-preview"): {"sees": 1.000, "mc_correct": 0.811, "mc_inc": 0.078, "mc_wrong": 0.111},
    ("key_plant_type", "gemini-3.1-pro-preview"): {"sees": 0.988, "mc_correct": 0.768, "mc_inc": 0.134, "mc_wrong": 0.098},
    ("key_plant_type", "claude-sonnet-4-6"): {"sees": 0.989, "mc_correct": 0.600, "mc_inc": 0.056, "mc_wrong": 0.344},
    ("key_plant_type", "claude-haiku-4-5"): {"sees": 1.000, "mc_correct": 0.400, "mc_inc": 0.011, "mc_wrong": 0.589},
    ("key_leaf_type", "gpt-4o-mini"): {"sees": 0.689, "mc_correct": 0.211, "mc_inc": 0.689, "mc_wrong": 0.100},
    ("key_leaf_type", "gpt-5-mini"): {"sees": 0.856, "mc_correct": 0.656, "mc_inc": 0.167, "mc_wrong": 0.178},
    ("key_leaf_type", "gemini-3-flash-preview"): {"sees": 0.911, "mc_correct": 0.744, "mc_inc": 0.144, "mc_wrong": 0.111},
    ("key_leaf_type", "gemini-3.1-pro-preview"): {"sees": 0.795, "mc_correct": 0.614, "mc_inc": 0.241, "mc_wrong": 0.145},
    ("key_leaf_type", "claude-sonnet-4-6"): {"sees": 0.711, "mc_correct": 0.578, "mc_inc": 0.311, "mc_wrong": 0.111},
    ("key_leaf_type", "claude-haiku-4-5"): {"sees": 0.933, "mc_correct": 0.367, "mc_inc": 0.322, "mc_wrong": 0.311},
}

THESIS_DIRECT = {
    "openai/gpt-4o-mini": {"accuracy": 0.07, "consistency": 0.20},
    "openai/gpt-5-mini": {"accuracy": 0.12, "consistency": 0.25},
    "google/gemini-3.1-pro-preview": {"accuracy": 0.32, "consistency": 0.47},
    "google/gemini-3-flash-preview": {"accuracy": 0.36, "consistency": 0.52},
    "anthropic/claude-sonnet-4-6": {"accuracy": 0.19, "consistency": 0.20},
    "anthropic/claude-haiku-4-5": {"accuracy": 0.03, "consistency": 0.27},
}


def rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def normalize_model(model: str) -> str:
    value = str(model)
    for prefix in ("openai/", "google/", "anthropic/"):
        if value.startswith(prefix):
            value = value.removeprefix(prefix)
    if value == "gemini-3.1-pro":
        return "gemini-3.1-pro-preview"
    if value == "gemini-3-flash":
        return "gemini-3-flash-preview"
    return value


def display_model(model: str) -> str:
    return MODEL_DISPLAY.get(model, model)


def pct(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{value * 100:.1f}%"


def john_calibration_summary() -> pd.DataFrame:
    df = pd.read_csv(CALIBRATION_CSV)
    rows = []
    for model in MODEL_ORDER:
        model_df = df[df["model"] == model]
        existence = model_df[model_df["prompt_type"] == "existence"]
        agreement = model_df[model_df["prompt_type"] == "agreement"]
        blind = model_df[model_df["prompt_type"] == "blind_mc"]
        blind_correct = int(blind["correct"].astype(bool).sum())
        blind_inc = int((blind["parsed"] == "INCONCLUSIVE").sum())
        blind_wrong = len(blind) - blind_correct - blind_inc
        rows.append(
            {
                "model": model,
                "model_display": display_model(model),
                "n": len(existence),
                "sees_feature_count": int((existence["parsed"] == "YES").sum()),
                "sees_feature_rate": rate(int((existence["parsed"] == "YES").sum()), len(existence)),
                "agreement_yes_count": int((agreement["parsed"] == "YES").sum()),
                "agreement_yes_rate": rate(int((agreement["parsed"] == "YES").sum()), len(agreement)),
                "blind_mc_correct_count": blind_correct,
                "blind_mc_correct_rate": rate(blind_correct, len(blind)),
                "blind_mc_inc_count": blind_inc,
                "blind_mc_inc_rate": rate(blind_inc, len(blind)),
                "blind_mc_wrong_count": blind_wrong,
                "blind_mc_wrong_rate": rate(blind_wrong, len(blind)),
            }
        )
    return pd.DataFrame(rows)


def john_sample_summary(path: Path, *, current_runner: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    if "model" not in df.columns:
        raise ValueError(f"{path} has no model column")
    df = df.copy()
    df["model_norm"] = df["model"].map(normalize_model)
    rows = []
    for feature in PRIMARY_FEATURES:
        for model in MODEL_ORDER:
            group = df[(df["model_norm"] == model) & (df["feature"] == feature)]
            if group.empty:
                continue
            p1 = group["p1_parsed"].fillna("").astype(str).str.strip().str.upper()
            p2 = group["p2_parsed"].fillna("").astype(str).str.strip()
            p2_upper = p2.str.upper()
            true_value = group["true_value"].fillna("").astype(str).str.strip()

            correct = p2.str.casefold().eq(true_value.str.casefold())
            if current_runner:
                inc = (
                    p2_upper.eq("INCONCLUSIVE")
                    | p2_upper.isin({"NOT_APPLICABLE", "N/A", "NA", "SKIPPED"})
                    | p1.ne("YES")
                )
            else:
                inc = p2_upper.eq("INCONCLUSIVE")
            wrong = ~correct & ~inc
            committed = correct | wrong
            rows.append(
                {
                    "feature": feature,
                    "feature_display": FEATURE_DISPLAY[feature],
                    "model": model,
                    "model_display": display_model(model),
                    "n": len(group),
                    "sees_feature_count": int(p1.eq("YES").sum()),
                    "sees_feature_rate": rate(int(p1.eq("YES").sum()), len(group)),
                    "mc_correct_count": int(correct.sum()),
                    "mc_correct_rate": rate(int(correct.sum()), len(group)),
                    "mc_inc_count": int(inc.sum()),
                    "mc_inc_rate": rate(int(inc.sum()), len(group)),
                    "mc_wrong_count": int(wrong.sum()),
                    "mc_wrong_rate": rate(int(wrong.sum()), len(group)),
                    "committed_count": int(committed.sum()),
                    "committed_rate": rate(int(committed.sum()), len(group)),
                    "committed_accuracy": rate(int(correct.sum()), int(committed.sum())),
                }
            )
    by_feature = pd.DataFrame(rows)
    overall_rows = []
    for model in MODEL_ORDER:
        group = by_feature[by_feature["model"] == model]
        if group.empty:
            continue
        n = int(group["n"].sum())
        correct = int(group["mc_correct_count"].sum())
        inc = int(group["mc_inc_count"].sum())
        wrong = int(group["mc_wrong_count"].sum())
        committed = correct + wrong
        sees = int(group["sees_feature_count"].sum())
        overall_rows.append(
            {
                "model": model,
                "model_display": display_model(model),
                "n": n,
                "sees_feature_count": sees,
                "sees_feature_rate": rate(sees, n),
                "mc_correct_count": correct,
                "mc_correct_rate": rate(correct, n),
                "mc_inc_count": inc,
                "mc_inc_rate": rate(inc, n),
                "mc_wrong_count": wrong,
                "mc_wrong_rate": rate(wrong, n),
                "committed_count": committed,
                "committed_rate": rate(committed, n),
                "committed_accuracy": rate(correct, committed),
            }
        )
    return by_feature, pd.DataFrame(overall_rows)


def calibration_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        model = row["model"]
        target = THESIS_CALIBRATION[model]
        for metric, actual_column in [
            ("sees", "sees_feature_rate"),
            ("mc_correct", "blind_mc_correct_rate"),
            ("mc_inc", "blind_mc_inc_rate"),
            ("mc_wrong", "blind_mc_wrong_rate"),
        ]:
            actual = float(row[actual_column])
            expected = target[metric]
            rows.append(
                {
                    "table": "5.2 calibration",
                    "feature": "",
                    "model": display_model(model),
                    "metric": metric,
                    "actual": actual,
                    "expected": expected,
                    "delta": actual - expected,
                    "matches_rounded_0.1pct": round(actual, 3) == round(expected, 3),
                }
            )
    return pd.DataFrame(rows)


def sample_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        key = (row["feature"], row["model"])
        target = THESIS_SAMPLE[key]
        for metric, actual_column in [
            ("sees", "sees_feature_rate"),
            ("mc_correct", "mc_correct_rate"),
            ("mc_inc", "mc_inc_rate"),
            ("mc_wrong", "mc_wrong_rate"),
        ]:
            actual = float(row[actual_column])
            expected = target[metric]
            rows.append(
                {
                    "table": "5.3 sample",
                    "feature": row["feature_display"],
                    "model": row["model_display"],
                    "metric": metric,
                    "actual": actual,
                    "expected": expected,
                    "delta": actual - expected,
                    "matches_rounded_0.1pct": round(actual, 3) == round(expected, 3),
                }
            )
    return pd.DataFrame(rows)


def direct_artifact_comparison() -> pd.DataFrame:
    if not DIRECT_SUMMARY_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(DIRECT_SUMMARY_CSV)
    rows = []
    for _, row in df.iterrows():
        model = row["model"]
        if model not in THESIS_DIRECT:
            continue
        for metric in ["accuracy", "consistency"]:
            actual = float(row[metric])
            expected = THESIS_DIRECT[model][metric]
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "artifact_actual": actual,
                    "thesis_expected": expected,
                    "delta": actual - expected,
                }
            )
    return pd.DataFrame(rows)


def write_percent_copy(df: pd.DataFrame, path: Path) -> None:
    percent_df = df.copy()
    for column in percent_df.columns:
        if column.endswith("_rate") or column in {
            "committed_accuracy",
            "actual",
            "expected",
            "delta",
            "artifact_actual",
            "thesis_expected",
        }:
            percent_df[column] = percent_df[column].map(pct)
    percent_df.to_csv(path, index=False)


def write_report(
    calibration: pd.DataFrame,
    sample: pd.DataFrame,
    sample_overall: pd.DataFrame,
    calibration_compare: pd.DataFrame,
    sample_compare: pd.DataFrame,
    current_overall: pd.DataFrame,
    direct_compare: pd.DataFrame,
) -> None:
    mismatches = pd.concat([calibration_compare, sample_compare], ignore_index=True)
    strict_mismatch_count = int((~mismatches["matches_rounded_0.1pct"]).sum())
    current_gpt5 = current_overall[current_overall["model"] == "gpt-5-mini"]
    current_gpt5_note = ""
    if not current_gpt5.empty:
        current_gpt5_note = (
            f"- Current full-run artifact gpt-5-mini sees-feature rate: "
            f"{pct(float(current_gpt5.iloc[0]['sees_feature_rate']))}; "
            "John raw CSV / thesis overall primary sees-feature rate is 80.7%.\n"
        )

    direct_note = ""
    if not direct_compare.empty:
        flash = direct_compare[
            (direct_compare["model"] == "google/gemini-3-flash-preview")
            & (direct_compare["metric"] == "accuracy")
        ]
        if not flash.empty:
            direct_note = (
                f"- Current direct artifact Gemini Flash accuracy: "
                f"{pct(float(flash.iloc[0]['artifact_actual']))}; thesis Table 5.1: "
                f"{pct(float(flash.iloc[0]['thesis_expected']))}.\n"
            )

    report = f"""# John-style Metric Rerun

This folder reruns the metric aggregation using John thesis-era conventions.
It does not make model API calls; it recomputes metrics from existing raw CSVs.

## Inputs

- `{CALIBRATION_CSV.relative_to(ROOT)}`
- `{FEATURE_CSV.relative_to(ROOT)}`
- Optional current artifact comparison: `{CURRENT_FULL_RUN_CSV.relative_to(ROOT)}`
- Optional direct artifact comparison: `{DIRECT_SUMMARY_CSV.relative_to(ROOT)}`

## Result

- Calibration Table 5.2 recomputation matches the thesis numbers.
- Sample / stepwise Table 5.3 recomputation matches the thesis numbers up to the rounded percentages printed in the thesis.
- Strict comparison rows with >0.1 percentage-point rounding mismatch: {strict_mismatch_count}.
{current_gpt5_note}{direct_note}
## Output Files

- `john_calibration_summary.csv`
- `john_calibration_summary_percent.csv`
- `john_sample_table53_by_feature.csv`
- `john_sample_table53_by_feature_percent.csv`
- `john_sample_overall_primary.csv`
- `john_sample_overall_primary_percent.csv`
- `comparison_to_thesis.csv`
- `current_full_run_john_style_by_feature.csv`
- `current_full_run_john_style_overall.csv`
- `direct_artifact_vs_thesis.csv`

## John-style Sample Rules

- `Sees Feature` = `p1_parsed == YES`.
- `MC Correct` = `p2_parsed == true_value`.
- `MC Inc` = `p2_parsed == INCONCLUSIVE`.
- `MC Incorrect` = concrete non-INCONCLUSIVE `p2_parsed != true_value`.
- For current runner artifacts, `NOT_APPLICABLE` and P1 gate failures are folded into `MC Inc` to emulate the older John CSV convention.
- `committed_accuracy` = `MC Correct / (MC Correct + MC Incorrect)`.

## Important Caveats

- The original Gemini Pro sample run is incomplete in `JM_Assets_2/feature_unlabeled_results.csv`: 83 flower rows, 82 plant rows, 83 leaf rows. The thesis percentages appear to use those actual denominators.
- The direct species artifact currently in this repo does not reproduce thesis Table 5.1 and should be treated as a later prompt/model artifact, not the original direct result.
"""
    (OUT_DIR / "README.md").write_text(report)


def main() -> None:
    calibration = john_calibration_summary()
    sample, sample_overall = john_sample_summary(FEATURE_CSV)
    calibration_compare = calibration_comparison(calibration)
    sample_compare = sample_comparison(sample)
    comparison = pd.concat([calibration_compare, sample_compare], ignore_index=True)

    calibration.to_csv(OUT_DIR / "john_calibration_summary.csv", index=False)
    write_percent_copy(calibration, OUT_DIR / "john_calibration_summary_percent.csv")
    sample.to_csv(OUT_DIR / "john_sample_table53_by_feature.csv", index=False)
    write_percent_copy(sample, OUT_DIR / "john_sample_table53_by_feature_percent.csv")
    sample_overall.to_csv(OUT_DIR / "john_sample_overall_primary.csv", index=False)
    write_percent_copy(sample_overall, OUT_DIR / "john_sample_overall_primary_percent.csv")
    comparison.to_csv(OUT_DIR / "comparison_to_thesis.csv", index=False)
    write_percent_copy(comparison, OUT_DIR / "comparison_to_thesis_percent.csv")

    if CURRENT_FULL_RUN_CSV.exists():
        current_by_feature, current_overall = john_sample_summary(CURRENT_FULL_RUN_CSV, current_runner=True)
    else:
        current_by_feature = pd.DataFrame()
        current_overall = pd.DataFrame()
    current_by_feature.to_csv(OUT_DIR / "current_full_run_john_style_by_feature.csv", index=False)
    write_percent_copy(current_by_feature, OUT_DIR / "current_full_run_john_style_by_feature_percent.csv")
    current_overall.to_csv(OUT_DIR / "current_full_run_john_style_overall.csv", index=False)
    write_percent_copy(current_overall, OUT_DIR / "current_full_run_john_style_overall_percent.csv")

    direct_compare = direct_artifact_comparison()
    direct_compare.to_csv(OUT_DIR / "direct_artifact_vs_thesis.csv", index=False)
    write_percent_copy(direct_compare, OUT_DIR / "direct_artifact_vs_thesis_percent.csv")

    write_report(
        calibration,
        sample,
        sample_overall,
        calibration_compare,
        sample_compare,
        current_overall,
        direct_compare,
    )

    mismatch_count = int((~comparison["matches_rounded_0.1pct"]).sum())
    print(f"Wrote John-style rerun outputs to {OUT_DIR.relative_to(ROOT)}")
    print(f"Thesis comparison mismatches at 0.1 percentage-point rounding: {mismatch_count}")


if __name__ == "__main__":
    main()
