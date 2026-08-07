# Trials

This folder stores stepwise trial result CSVs and metadata JSON files under
`trials/artifacts/`.

The trial wrapper reads `.env` from the repo root before it starts the model
runner, so `OPENROUTER_API_KEY=...` can live there. To load the same `.env` into
your current terminal session, run:

```bash
set -a
source .env
set +a
```

Run the normal smoke test from the repo root:

```bash
uv run python scripts/run_stepwise_trial.py \
  --model 'openai/gpt-4o-mini' \
  --sample-limit 1 \
  --features key_flower_type
```

Full option version:

```bash
uv run python scripts/run_stepwise_trial.py \
  --model openrouter/free \
  --image-set sample.csv \
  --data-split all \
  --sample-limit 20 \
  --features key_flower_type,key_plant_type,key_leaf_type \
  --prompt-set stepwise-v1 \
  --workers 4 \
  --timeout 120 \
  --mode command \
  --out-file stepwise_trial_result.csv \
  --trial-id trial_001
```

Supported feature names are currently `key_flower_type`, `key_plant_type`, and
`key_leaf_type`.

Prompt versions live as JSON files in
`newcomb_wildflower_guide/experiment_repro/prompt_sets/`; `--prompt-set`
selects one of those files and changes the actual P1/P2 prompt text sent to the
model. `--image-set` is a CSV filename under
`newcomb_wildflower_guide/experiment_repro/output/` or
`manual_audit/generated/`; the default `sample.csv` resolves to the former.
`--sample-limit` controls how many rows are taken from that CSV: use a positive
integer for the first N rows or `all` for the full CSV. `--data-split`
supports `all`, `easy`, `core`, and `challenging`; tiers require a reviewed
manifest with a `benchmark_tier` column.

If `--out-file` is omitted, the wrapper creates an auto-generated `trial_id`
using the run time, model, sample limit, and feature list, then writes matching
CSV and metadata files:

```text
trials/artifacts/stepwise-20260624-143000-openai-gpt-4o-mini-n1-key_flower_type.csv
trials/artifacts/stepwise-20260624-143000-openai-gpt-4o-mini-n1-key_flower_type.metadata.json
```

If `--out-file` is a relative filename, it is also written under
`trials/artifacts/`. Use an absolute path only when you intentionally want to
write somewhere else. The metadata filename is always based on `trial_id`.

If `--trial-id jm-baseline-001` is provided and `--out-file` is omitted, the
wrapper writes:

```text
trials/artifacts/jm-baseline-001.csv
trials/artifacts/jm-baseline-001.metadata.json
```

Each generated CSV includes a `trial_id` column and does not include a
`model_command` column.

Use `scripts/analyze_stepwise_results.py` to build analysis views from one or
more raw trial CSVs. It writes:

- `per_trial_rows.csv`
- `summary_overall.csv`
- `summary_by_model_feature.csv`
- `dashboard_whole_experiment.csv`
- `outcome_by_true_value.csv`
- `outcome_pairs.csv`
- `metric_definitions.csv`

Attach human photo-level labels for canonical P1/P2 metrics:

```bash
uv run python scripts/analyze_stepwise_results.py \
  --input trials/artifacts/example.csv \
  --audit-csv manual_audit/sample_10_manual_audit.csv \
  --out-dir trials/analysis/example
```

Without approved audit rows, `true_value` accuracy is species-level exploratory
analysis, not a formal photo-level benchmark result.

Each row is assigned one primary outcome: `CORRECT`, `WRONG`, `INCONCLUSIVE`,
or `NOT_APPLICABLE`. `NOT_APPLICABLE` means P1 did not parse as `YES`, so P2
was skipped instead of being treated as a P2 inconclusive answer.
`outcome_by_true_value.csv` groups by model, feature, and true value to show P1
visibility counts plus `correct_count`, `wrong_count`, `inconclusive_count`,
`not_applicable_count`, their rates, and `most_common_wrong_prediction`.

## Local CLIP baseline

Install once with `uv sync --extra local-cv`, then run:

```bash
uv run python scripts/run_stepwise_trial.py \
  --mode local-clip \
  --model openai/clip-vit-base-patch32 \
  --image-set sample.csv \
  --sample-limit 1 \
  --features key_leaf_type \
  --timeout 600 \
  --trial-id local-clip-smoke
```

The first run downloads model weights. Inference is local after that, although
remote image URLs still need to be fetched. This is a zero-shot screening
baseline, not a botanical specialist model.

Each metadata JSON records the trial setup:

```json
{
  "trial_id": "jm-baseline-001",
  "command": "uv run python scripts/run_stepwise_trial.py --model openai/gpt-4o-mini --sample-limit 20 --features key_flower_type,key_plant_type,key_leaf_type --trial-id jm-baseline-001",
  "image_set": "sample.csv",
  "data_split": "all",
  "prompt_set": "stepwise-v1",
  "model": "openai/gpt-4o-mini",
  "sample_limit": 20,
  "features": ["key_flower_type", "key_plant_type", "key_leaf_type"],
  "mode": "command",
  "output_file": "trials/artifacts/jm-baseline-001.csv",
  "git_commit": "...",
  "python": "...",
  "package_manager": "uv"
}
```

## Recorded Trials

| Trial | Record | CSV | Metadata |
| --- | --- | --- | --- |
| Real OpenRouter smoke test, `sample-limit 1`, `key_flower_type` | `trials/stepwise-20260624-034808-openai-gpt-4o-mini-n1-key_flower_type.md` | `trials/artifacts/stepwise-20260624-034808-openai-gpt-4o-mini-n1-key_flower_type.csv` | `trials/artifacts/stepwise-20260624-034808-openai-gpt-4o-mini-n1-key_flower_type.metadata.json` |
