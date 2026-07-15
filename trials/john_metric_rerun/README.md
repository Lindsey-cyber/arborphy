# John-style Metric Rerun

This folder reruns the metric aggregation using John thesis-era conventions.
It does not make model API calls; it recomputes metrics from existing raw CSVs.

## Inputs

- `JM_Assets_2/calibration_results.csv`
- `JM_Assets_2/feature_unlabeled_results.csv`
- Optional current artifact comparison: `trials/artifacts/notebook-20260708-034723.csv`
- Optional direct artifact comparison: `trials/artifacts/direct-species-baseline-john-90x3-six-models.summary.csv`

## Result

- Calibration Table 5.2 recomputation matches the thesis numbers.
- Sample / stepwise Table 5.3 recomputation matches the thesis numbers up to the rounded percentages printed in the thesis.
- Strict comparison rows with >0.1 percentage-point rounding mismatch: 0.
- Current full-run artifact gpt-5-mini sees-feature rate: 27.4%; John raw CSV / thesis overall primary sees-feature rate is 80.7%.
- Current direct artifact Gemini Flash accuracy: 95.2%; thesis Table 5.1: 36.0%.

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
