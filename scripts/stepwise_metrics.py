from __future__ import annotations

from typing import Iterable

import pandas as pd


BASE_GROUP_COLUMNS = ["trial_id", "run_id", "model", "prompt_set"]
FEATURE_GROUP_COLUMNS = ["trial_id", "run_id", "model", "prompt_set", "feature"]
TRUE_VALUE_GROUP_COLUMNS = ["model", "feature", "true_value"]
OUTCOME_PAIR_GROUP_COLUMNS = ["model", "feature"]
REQUIRED_RESULT_COLUMNS = [
    "trial_id",
    "model",
    "feature",
    "p1_parsed",
    "p2_parsed",
    "true_value",
    "committed",
]
OUTCOME_CORRECT = "CORRECT"
OUTCOME_WRONG = "WRONG"
OUTCOME_INCONCLUSIVE = "INCONCLUSIVE"
OUTCOME_NOT_APPLICABLE = "NOT_APPLICABLE"
NOT_APPLICABLE_VALUES = {"NA", "N/A", "SKIPPED", OUTCOME_NOT_APPLICABLE}


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


def require_columns(df: pd.DataFrame, columns: Iterable[str] = REQUIRED_RESULT_COLUMNS) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")


def available_group_columns(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    return [column for column in candidates if column in df.columns]


def annotate_results(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df)
    annotated = df.copy()
    p1 = normalize_text(annotated["p1_parsed"]).str.upper()
    p2 = normalize_text(annotated["p2_parsed"])
    p2_upper = p2.str.upper()
    true_value = normalize_text(annotated["true_value"])
    committed = normalize_bool(annotated["committed"])
    p1_yes = p1.eq("YES")
    p1_no = p1.eq("NO")
    p1_inconclusive = p1.eq("INCONCLUSIVE")
    p1_other = ~(p1_yes | p1_no | p1_inconclusive)
    not_applicable = ~p1_yes | p2_upper.isin(NOT_APPLICABLE_VALUES)
    inconclusive = p1_yes & (p2.eq("") | p2_upper.eq(OUTCOME_INCONCLUSIVE))
    concrete = p1_yes & p2.ne("") & ~inconclusive & ~not_applicable
    correct = concrete & p2.str.casefold().eq(true_value.str.casefold())
    wrong = concrete & ~correct

    annotated["feature_count_unit"] = 1
    annotated["p1_sees_feature"] = p1_yes
    annotated["p1_does_not_see_feature"] = p1_no
    annotated["p1_visibility_inconclusive"] = p1_inconclusive | p1_other
    annotated["p2_applicable"] = p1_yes
    annotated["p2_not_applicable"] = not_applicable
    annotated["p2_inconclusive"] = inconclusive
    annotated["committed_bool"] = committed
    annotated["correct_value"] = correct
    annotated["wrong_value"] = wrong
    annotated["predicted_value"] = p2
    annotated.loc[inconclusive, "predicted_value"] = OUTCOME_INCONCLUSIVE
    annotated.loc[not_applicable, "predicted_value"] = OUTCOME_NOT_APPLICABLE
    annotated["outcome"] = OUTCOME_NOT_APPLICABLE
    annotated.loc[correct, "outcome"] = OUTCOME_CORRECT
    annotated.loc[wrong, "outcome"] = OUTCOME_WRONG
    annotated.loc[inconclusive, "outcome"] = OUTCOME_INCONCLUSIVE
    return annotated


def most_common_wrong_prediction(group: pd.DataFrame) -> str:
    wrong_predictions = normalize_text(group.loc[group["outcome"] == OUTCOME_WRONG, "predicted_value"])
    if wrong_predictions.empty:
        return ""
    counts = wrong_predictions.value_counts().rename_axis("predicted_value").reset_index(name="count")
    counts = counts.sort_values(["count", "predicted_value"], ascending=[False, True])
    return str(counts.iloc[0]["predicted_value"])


def summarize_group(group: pd.DataFrame, source_file: str | None = None) -> dict[str, object]:
    annotated = annotate_results(group)
    feature_count = len(annotated)
    n_observations = (
        annotated["observation_id"].astype(str).nunique() if "observation_id" in annotated.columns else None
    )

    features_seen = int(annotated["p1_sees_feature"].sum())
    features_not_seen = int(annotated["p1_does_not_see_feature"].sum())
    visibility_inconclusive_count = int(annotated["p1_visibility_inconclusive"].sum())
    p2_applicable_count = int(annotated["p2_applicable"].sum())
    correct_count = int((annotated["outcome"] == OUTCOME_CORRECT).sum())
    wrong_count = int((annotated["outcome"] == OUTCOME_WRONG).sum())
    inconclusive_count = int((annotated["outcome"] == OUTCOME_INCONCLUSIVE).sum())
    not_applicable_count = int((annotated["outcome"] == OUTCOME_NOT_APPLICABLE).sum())
    committed_count = correct_count + wrong_count

    summary = {
        "feature_count": feature_count,
        "n_observations": n_observations,
        "features_seen": features_seen,
        "features_seen_rate": rate(features_seen, feature_count),
        "features_not_seen": features_not_seen,
        "features_not_seen_rate": rate(features_not_seen, feature_count),
        "visibility_inconclusive_count": visibility_inconclusive_count,
        "visibility_inconclusive_rate": rate(visibility_inconclusive_count, feature_count),
        "p2_applicable_count": p2_applicable_count,
        "committed_count": committed_count,
        "commitment_rate": rate(committed_count, feature_count),
        "p2_commitment_rate": rate(committed_count, p2_applicable_count),
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "inconclusive_count": inconclusive_count,
        "not_applicable_count": not_applicable_count,
        "correct_rate": rate(correct_count, feature_count),
        "wrong_rate": rate(wrong_count, feature_count),
        "inconclusive_rate": rate(inconclusive_count, feature_count),
        "not_applicable_rate": rate(not_applicable_count, feature_count),
        "p2_correct_rate": rate(correct_count, p2_applicable_count),
        "p2_wrong_rate": rate(wrong_count, p2_applicable_count),
        "p2_inconclusive_rate": rate(inconclusive_count, p2_applicable_count),
        "committed_accuracy": rate(correct_count, committed_count),
    }
    if source_file is not None:
        summary = {"source_file": source_file, **summary}
    return summary


def summarize(df: pd.DataFrame, group_columns: list[str], source_file: str | None = None) -> pd.DataFrame:
    group_columns = available_group_columns(df, group_columns)
    rows = []
    for keys, group in df.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(summarize_group(group, source_file))
        rows.append(row)
    return pd.DataFrame(rows)


def whole_experiment_dashboard(df: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_group(df)
    summary["trial_count"] = df["trial_id"].astype(str).nunique() if "trial_id" in df.columns else None
    summary["model_count"] = df["model"].astype(str).nunique() if "model" in df.columns else None
    summary["feature_name_count"] = df["feature"].astype(str).nunique() if "feature" in df.columns else None
    return pd.DataFrame([summary])


def outcome_by_true_value(df: pd.DataFrame) -> pd.DataFrame:
    annotated = annotate_results(df)
    group_columns = available_group_columns(annotated, TRUE_VALUE_GROUP_COLUMNS)
    rows = []
    for keys, group in annotated.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "feature_count": len(group),
                "features_seen": int(group["p1_sees_feature"].sum()),
                "correct_count": int((group["outcome"] == OUTCOME_CORRECT).sum()),
                "wrong_count": int((group["outcome"] == OUTCOME_WRONG).sum()),
                "inconclusive_count": int((group["outcome"] == OUTCOME_INCONCLUSIVE).sum()),
                "not_applicable_count": int((group["outcome"] == OUTCOME_NOT_APPLICABLE).sum()),
            }
        )
        row["features_seen_rate"] = rate(row["features_seen"], row["feature_count"])
        row["correct_rate"] = rate(row["correct_count"], row["feature_count"])
        row["wrong_rate"] = rate(row["wrong_count"], row["feature_count"])
        row["inconclusive_rate"] = rate(row["inconclusive_count"], row["feature_count"])
        row["not_applicable_rate"] = rate(row["not_applicable_count"], row["feature_count"])
        row["most_common_wrong_prediction"] = most_common_wrong_prediction(group)
        rows.append(row)
    return pd.DataFrame(rows)


def outcome_pairs(df: pd.DataFrame) -> pd.DataFrame:
    annotated = annotate_results(df)
    group_columns = available_group_columns(annotated, OUTCOME_PAIR_GROUP_COLUMNS)
    pair_columns = group_columns + ["true_value", "predicted_value", "outcome"]
    return (
        annotated.groupby(pair_columns, dropna=False, sort=True)
        .size()
        .reset_index(name="count")
    )


def metric_definitions() -> pd.DataFrame:
    rows = [
        {
            "metric": "feature_count",
            "numerator": "count(rows)",
            "denominator": "",
            "csv_formula": "one row = one observation_id x feature task",
            "useful_signal": "Experiment size and denominator; not a performance metric by itself.",
        },
        {
            "metric": "features_seen",
            "numerator": "count(p1_parsed == 'YES')",
            "denominator": "feature_count",
            "csv_formula": "p1_parsed",
            "useful_signal": "Visibility-gate or model-willingness signal; not accuracy without human visibility labels.",
        },
        {
            "metric": "features_not_seen",
            "numerator": "count(p1_parsed == 'NO')",
            "denominator": "feature_count",
            "csv_formula": "p1_parsed",
            "useful_signal": "P1 gate says the feature is not visible; P2 should be NOT_APPLICABLE.",
        },
        {
            "metric": "visibility_inconclusive_count",
            "numerator": "count(p1_parsed == 'INCONCLUSIVE' or other unparsable P1)",
            "denominator": "feature_count",
            "csv_formula": "p1_parsed",
            "useful_signal": "P1 gate uncertainty; separate from P2 classification uncertainty.",
        },
        {
            "metric": "not_applicable_count",
            "numerator": "count(p1_parsed != 'YES')",
            "denominator": "feature_count",
            "csv_formula": "p1_parsed, p2_parsed",
            "useful_signal": "P2 was not asked because the visibility gate did not pass.",
        },
        {
            "metric": "correct_count",
            "numerator": "count(p1_parsed == 'YES' and p2_parsed == true_value)",
            "denominator": "",
            "csv_formula": "p2_parsed, true_value",
            "useful_signal": "Direct count of applicable P2 tasks solved correctly.",
        },
        {
            "metric": "wrong_count",
            "numerator": "count(p1_parsed == 'YES' and p2_parsed not in {'', 'INCONCLUSIVE', 'NOT_APPLICABLE'} and p2_parsed != true_value)",
            "denominator": "",
            "csv_formula": "p2_parsed, true_value",
            "useful_signal": "Direct count of concrete but wrong answers.",
        },
        {
            "metric": "inconclusive_count",
            "numerator": "count(p1_parsed == 'YES' and p2_parsed == 'INCONCLUSIVE')",
            "denominator": "",
            "csv_formula": "p2_parsed",
            "useful_signal": "P2 classification uncertainty after P1 said the feature was visible.",
        },
        {
            "metric": "commitment_rate",
            "numerator": "correct_count + wrong_count",
            "denominator": "feature_count",
            "csv_formula": "outcome in {'CORRECT', 'WRONG'}",
            "useful_signal": "End-to-end concrete-answer rate across all feature tasks.",
        },
        {
            "metric": "p2_commitment_rate",
            "numerator": "correct_count + wrong_count",
            "denominator": "p2_applicable_count",
            "csv_formula": "outcome in {'CORRECT', 'WRONG'} where p1_parsed == 'YES'",
            "useful_signal": "Concrete-answer rate only among rows where P2 was actually asked.",
        },
        {
            "metric": "correct_rate",
            "numerator": "correct_count",
            "denominator": "feature_count",
            "csv_formula": "outcome == 'CORRECT'",
            "useful_signal": "Best single end-to-end success rate when INCONCLUSIVE should count as not solved.",
        },
        {
            "metric": "wrong_rate",
            "numerator": "wrong_count",
            "denominator": "feature_count",
            "csv_formula": "outcome == 'WRONG'",
            "useful_signal": "Risk signal: how often the model gives a concrete wrong value.",
        },
        {
            "metric": "inconclusive_rate",
            "numerator": "inconclusive_count",
            "denominator": "feature_count",
            "csv_formula": "outcome == 'INCONCLUSIVE'",
            "useful_signal": "End-to-end P2 uncertainty rate; excludes rows where P2 was not applicable.",
        },
        {
            "metric": "not_applicable_rate",
            "numerator": "not_applicable_count",
            "denominator": "feature_count",
            "csv_formula": "outcome == 'NOT_APPLICABLE'",
            "useful_signal": "How often P2 was skipped because P1 did not say YES.",
        },
        {
            "metric": "p2_correct_rate",
            "numerator": "correct_count",
            "denominator": "p2_applicable_count",
            "csv_formula": "outcome == 'CORRECT' where p1_parsed == 'YES'",
            "useful_signal": "P2 accuracy denominator after the visibility gate passes.",
        },
        {
            "metric": "p2_wrong_rate",
            "numerator": "wrong_count",
            "denominator": "p2_applicable_count",
            "csv_formula": "outcome == 'WRONG' where p1_parsed == 'YES'",
            "useful_signal": "P2 concrete error rate after the visibility gate passes.",
        },
        {
            "metric": "p2_inconclusive_rate",
            "numerator": "inconclusive_count",
            "denominator": "p2_applicable_count",
            "csv_formula": "outcome == 'INCONCLUSIVE' where p1_parsed == 'YES'",
            "useful_signal": "P2 abstention rate after the visibility gate passes.",
        },
        {
            "metric": "committed_accuracy",
            "numerator": "correct_count",
            "denominator": "correct_count + wrong_count",
            "csv_formula": "outcome == 'CORRECT' among concrete outcomes",
            "useful_signal": "Quality score among concrete answers; pair with commitment_rate to avoid rewarding excessive abstention.",
        },
        {
            "metric": "most_common_wrong_prediction",
            "numerator": "mode(predicted_value where outcome == 'WRONG')",
            "denominator": "",
            "csv_formula": "predicted_value, outcome",
            "useful_signal": "Shows the most common concrete wrong prediction for each true value.",
        },
    ]
    return pd.DataFrame(rows)
