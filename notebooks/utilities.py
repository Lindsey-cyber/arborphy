from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import pandas as pd

try:
    from IPython.display import Markdown, display
except ImportError:
    Markdown = str

    def display(value: object) -> None:
        print(value)


def find_repo_root() -> Path:
    path = Path.cwd().resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "scripts" / "run_stepwise_trial.py").exists():
            return candidate
    notebooks_parent = Path(__file__).resolve().parents[1]
    if (notebooks_parent / "scripts" / "run_stepwise_trial.py").exists():
        return notebooks_parent
    raise FileNotFoundError("Could not find repo root containing scripts/run_stepwise_trial.py")


ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stepwise_trial import metadata_path_for, uv_executable  # noqa: E402
from scripts.stepwise_metrics import (  # noqa: E402
    FEATURE_GROUP_COLUMNS,
    annotate_results,
    metric_definitions,
    outcome_by_true_value,
    outcome_pairs,
    summarize,
    whole_experiment_dashboard,
)


DISPLAY_ROW_LIMIT = 50


def feature_arg(features: str | Iterable[str]) -> str:
    if isinstance(features, str):
        return features
    return ",".join(features)


def int_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def percent_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.1%}"


def ratio_text(count: object, denominator: object, rate: object) -> str:
    count_part = int_text(count)
    denominator_part = int_text(denominator)
    rate_part = percent_text(rate)
    if count_part and denominator_part and rate_part:
        return f"{count_part}/{denominator_part} ({rate_part})"
    if count_part and denominator_part:
        return f"{count_part}/{denominator_part}"
    return count_part or rate_part


def compact_dashboard_view(dashboard: pd.DataFrame) -> pd.DataFrame:
    if dashboard.empty:
        return dashboard
    row = dashboard.iloc[0]
    feature_count = row.get("feature_count")
    committed_count = row.get("committed_count")
    records = [
        {"metric": "feature_tasks", "value": int_text(feature_count)},
        {"metric": "unique_observations", "value": int_text(row.get("n_observations"))},
        {"metric": "trials", "value": int_text(row.get("trial_count"))},
        {"metric": "models", "value": int_text(row.get("model_count"))},
        {"metric": "features", "value": int_text(row.get("feature_name_count"))},
        {
            "metric": "features_seen",
            "value": ratio_text(row.get("features_seen"), feature_count, row.get("features_seen_rate")),
        },
        {
            "metric": "committed",
            "value": ratio_text(committed_count, feature_count, row.get("commitment_rate")),
        },
        {
            "metric": "correct",
            "value": ratio_text(row.get("correct_count"), feature_count, row.get("correct_rate")),
        },
        {
            "metric": "wrong",
            "value": ratio_text(row.get("wrong_count"), feature_count, row.get("wrong_rate")),
        },
        {
            "metric": "inconclusive",
            "value": ratio_text(row.get("inconclusive_count"), feature_count, row.get("inconclusive_rate")),
        },
        {
            "metric": "accuracy_when_committed",
            "value": ratio_text(row.get("correct_count"), committed_count, row.get("committed_accuracy")),
        },
    ]
    return pd.DataFrame(records)


def compact_summary_view(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    identity_columns = [
        column
        for column in ["trial_id", "run_id", "model", "prompt_set", "feature"]
        if column in summary.columns
    ]
    view = summary[identity_columns].copy()
    view["n"] = summary["feature_count"].map(int_text)
    if "n_observations" in summary.columns:
        view["observations"] = summary["n_observations"].map(int_text)
    view["seen"] = [
        ratio_text(row.get("features_seen"), row.get("feature_count"), row.get("features_seen_rate"))
        for _, row in summary.iterrows()
    ]
    view["committed"] = [
        ratio_text(row.get("committed_count"), row.get("feature_count"), row.get("commitment_rate"))
        for _, row in summary.iterrows()
    ]
    view["correct"] = [
        ratio_text(row.get("correct_count"), row.get("feature_count"), row.get("correct_rate"))
        for _, row in summary.iterrows()
    ]
    view["wrong"] = [
        ratio_text(row.get("wrong_count"), row.get("feature_count"), row.get("wrong_rate"))
        for _, row in summary.iterrows()
    ]
    view["inconclusive"] = [
        ratio_text(row.get("inconclusive_count"), row.get("feature_count"), row.get("inconclusive_rate"))
        for _, row in summary.iterrows()
    ]
    view["accuracy_when_committed"] = [
        ratio_text(row.get("correct_count"), row.get("committed_count"), row.get("committed_accuracy"))
        for _, row in summary.iterrows()
    ]
    return view


def compact_outcome_by_true_value_view(by_true_value: pd.DataFrame) -> pd.DataFrame:
    if by_true_value.empty:
        return by_true_value
    identity_columns = [
        column
        for column in ["model", "feature", "true_value"]
        if column in by_true_value.columns
    ]
    view = by_true_value[identity_columns].copy()
    view["n"] = by_true_value["feature_count"].map(int_text)
    view["correct"] = [
        ratio_text(row.get("correct_count"), row.get("feature_count"), row.get("correct_rate"))
        for _, row in by_true_value.iterrows()
    ]
    view["wrong"] = [
        ratio_text(row.get("wrong_count"), row.get("feature_count"), row.get("wrong_rate"))
        for _, row in by_true_value.iterrows()
    ]
    view["inconclusive"] = [
        ratio_text(row.get("inconclusive_count"), row.get("feature_count"), row.get("inconclusive_rate"))
        for _, row in by_true_value.iterrows()
    ]
    view["most_common_wrong_prediction"] = by_true_value["most_common_wrong_prediction"]
    return view


def compact_rows_view(rows: pd.DataFrame, limit: int = DISPLAY_ROW_LIMIT) -> pd.DataFrame:
    columns = [
        column
        for column in [
            "observation_id",
            "model",
            "prompt_set",
            "feature",
            "true_value",
            "p1_parsed",
            "p2_parsed",
            "outcome",
        ]
        if column in rows.columns
    ]
    return rows[columns].head(limit).copy()


def run_stepwise_notebook_trial(
    *,
    model: str,
    mode: str,
    image_set: str,
    data_split: str,
    sample_limit: int | str,
    features: str | Iterable[str],
    prompt_set: str,
    temperature: float,
    max_tokens: int,
    workers: int,
    timeout: int,
    run_id: str,
    trial_id: str,
    out_file: str = "",
) -> dict[str, object]:
    trial_id = trial_id or f"notebook-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"
    cmd = [
        uv_executable(),
        "run",
        "python",
        "scripts/run_stepwise_trial.py",
        "--model",
        model,
        "--mode",
        mode,
        "--image-set",
        image_set,
        "--data-split",
        data_split,
        "--sample-limit",
        str(sample_limit),
        "--features",
        feature_arg(features),
        "--prompt-set",
        prompt_set,
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(max_tokens),
        "--workers",
        str(workers),
        "--timeout",
        str(timeout),
        "--run-id",
        run_id,
        "--trial-id",
        trial_id,
    ]
    if out_file:
        cmd.extend(["--out-file", out_file])

    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Trial failed with exit code {result.returncode}")

    metadata_file = metadata_path_for(trial_id)
    metadata = json.loads(metadata_file.read_text())
    output_file = Path(metadata["output_file"])
    if not output_file.is_absolute():
        output_file = ROOT / output_file

    raw = pd.read_csv(output_file)
    rows = annotate_results(raw)
    dashboard = whole_experiment_dashboard(rows)
    summary = summarize(rows, FEATURE_GROUP_COLUMNS)
    by_true_value = outcome_by_true_value(rows)
    pairs = outcome_pairs(rows)
    definitions = metric_definitions()
    dashboard_view = compact_dashboard_view(dashboard)
    summary_view = compact_summary_view(summary)
    by_true_value_view = compact_outcome_by_true_value_view(by_true_value)
    rows_view = compact_rows_view(rows)

    display(Markdown(f"### Trial saved to `{output_file.relative_to(ROOT)}`"))
    display(pd.DataFrame([metadata]))
    display(Markdown("### Whole-Experiment Dashboard"))
    display(dashboard_view)
    display(Markdown("### Per-Feature / Model Summary (Compact)"))
    display(summary_view)
    display(Markdown("### Outcome by True Value"))
    display(by_true_value_view)
    display(Markdown("### Outcome Pairs: True Value x Predicted Value"))
    display(pairs)
    display(Markdown(f"### Per-Trial Rows (First {DISPLAY_ROW_LIMIT})"))
    display(rows_view)
    display(Markdown("### Metric Definitions"))
    display(definitions)
    return {
        "rows": rows,
        "summary": summary,
        "dashboard": dashboard,
        "outcome_by_true_value": by_true_value,
        "outcome_pairs": pairs,
        "metric_definitions": definitions,
        "rows_view": rows_view,
        "summary_view": summary_view,
        "dashboard_view": dashboard_view,
        "outcome_by_true_value_view": by_true_value_view,
        "metadata": metadata,
        "output_file": output_file,
    }
