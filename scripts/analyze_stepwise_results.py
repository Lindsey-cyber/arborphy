from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "trials" / "analysis"
DEFAULT_AUDIT = ROOT / "manual_audit" / "sample_10_manual_audit.csv"

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


def normalize_join_id(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def attach_audit(results: pd.DataFrame, audit_path: Path) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "manual_audit"))
    from audit_schema import load_audit, validate_audit

    audit = load_audit(audit_path)
    validation = validate_audit(audit)
    if validation.errors:
        raise ValueError("Invalid audit CSV:\n" + "\n".join(validation.errors))
    for warning in validation.warnings:
        print(f"Audit warning: {warning}", file=sys.stderr)

    results = results.copy()
    results["photo_id"] = normalize_join_id(results["photo_id"])
    audit["photo_id"] = normalize_join_id(audit["photo_id"])
    audit_columns = [
        "photo_id",
        "feature",
        "species_level_true_value",
        "human_visible",
        "human_can_assign_value",
        "human_value_if_assignable",
        "matches_species_level_true_value",
        "difficulty",
        "review_status",
        "exclusion_reason",
        "reviewer",
        "reviewed_at",
        "notes",
    ]
    audit = audit[audit_columns].rename(
        columns={
            "species_level_true_value": "audit_species_level_true_value",
            "notes": "audit_notes",
        }
    )
    audit_payload_columns = [
        column for column in audit.columns if column not in {"photo_id", "feature"}
    ]
    results = results.drop(
        columns=[column for column in audit_payload_columns if column in results.columns],
        errors="ignore",
    )
    return results.merge(audit, on=["photo_id", "feature"], how="left", validate="many_to_one")


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
        "--audit-csv",
        default="",
        help=(
            "Optional photo-level human audit CSV. When supplied, canonical P1/P2 gold metrics are added. "
            f"The current pilot is {DEFAULT_AUDIT.relative_to(ROOT)}."
        ),
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
    if args.audit_csv:
        df = attach_audit(df, resolve_path(args.audit_csv))
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
    if "gold_p1_evaluable" in annotated.columns:
        outputs["gold_evaluation_rows.csv"] = annotated[
            annotated["gold_p1_evaluable"] | annotated["gold_p2_evaluable"]
        ].copy()
    for filename, frame in outputs.items():
        frame.to_csv(out_dir / filename, index=False)

    print(f"Read {len(df)} rows from {len(input_paths)} input file(s)")
    for filename in outputs:
        print(f"Wrote {out_dir / filename}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
