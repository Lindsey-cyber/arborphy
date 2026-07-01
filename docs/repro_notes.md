# Reproduction Notes

These notes describe the current environment and commands for reproducing the
Newcomb/JM stepwise experiment pipeline.

## Environment

- Repository root: `/Users/lindseyma/Documents/GitHub/arborphy`
- Package manager: `uv`
- Observed `uv` version: `uv 0.9.0`
- Project Python requirement: `>=3.12`
- Python recorded by the real smoke-test metadata: `3.12.10`
- System `python3` observed outside `uv`: `3.14.0`

Use `uv run ...` for experiment commands so the recorded Python environment is
the one used by the runner.

## Install / Sync

From the repo root:

```bash
uv sync
```

The project dependencies are declared in `pyproject.toml`:

- `duckdb`
- `pandas`

## Environment Variables

The wrapper `scripts/run_stepwise_trial.py` reads `.env` from the repo root
before starting the model runner.

Required for real OpenRouter calls:

```bash
OPENROUTER_API_KEY=...
```

Optional OpenRouter settings:

```bash
OPENROUTER_FREE_MODEL=...
OPENROUTER_MAX_TOKENS=200
OPENROUTER_TEMPERATURE=0
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_REFERER=https://local.codex
OPENROUTER_TITLE=JM reproduction
```

To load `.env` into the current terminal session:

```bash
set -a
source .env
set +a
```

This is optional for the main trial wrapper because it loads `.env`
automatically.

## Data Paths

Generated JM-style runner inputs:

- `newcomb_wildflower_guide/experiment_repro/output/feature_value_pairs.csv`
- `newcomb_wildflower_guide/experiment_repro/output/references.csv`
- `newcomb_wildflower_guide/experiment_repro/output/newcomb_preprocessed.csv`
- `newcomb_wildflower_guide/experiment_repro/output/sample.csv`
- `newcomb_wildflower_guide/experiment_repro/output/illustration_paths.json`

Source extraction tables:

- `newcomb_wildflower_guide/extracted_vocabulary/parquet/`

Observation source:

- `observations_pound_ridge/ward_pound_ridge_species.csv`

Trial outputs:

- `trials/artifacts/*.csv`
- `trials/artifacts/*.metadata.json`

## Build Inputs

The JM-style inputs can be rebuilt from the Newcomb parquet files and
observation CSV:

```bash
cd newcomb_wildflower_guide/experiment_repro
uv run python build_inputs.py
```

Current generated input summary:

- `feature_value_pairs.csv`: 76 rows
- `references.csv`: 76 rows
- `newcomb_preprocessed.csv`: 781 rows
- `sample.csv`: 288 rows
- matched illustration paths: 12

## Smoke Test Command

Run from the repo root:

```bash
uv run python scripts/run_stepwise_trial.py \
  --model 'openai/gpt-4o-mini' \
  --sample-limit 1 \
  --features key_flower_type
```

The current real API smoke test was run with:

```bash
uv run python scripts/run_stepwise_trial.py --model openai/gpt-4o-mini --sample-limit 1 --features key_flower_type
```

It produced:

- `trials/artifacts/stepwise-20260624-034808-openai-gpt-4o-mini-n1-key_flower_type.csv`
- `trials/artifacts/stepwise-20260624-034808-openai-gpt-4o-mini-n1-key_flower_type.metadata.json`

## Full Trial Command Shape

Do not run this at large sample sizes until OpenRouter access and cost are
approved.

```bash
uv run python scripts/run_stepwise_trial.py \
  --model openrouter/free \
  --image-set sample.csv \
  --sample-limit 20 \
  --features key_flower_type,key_plant_type,key_leaf_type \
  --workers 4 \
  --timeout 120 \
  --mode command \
  --out-file stepwise_trial_result.csv \
  --trial-id trial_001
```

Supported features in the current runner:

- `key_flower_type`
- `key_plant_type`
- `key_leaf_type`

Prompt sets are real JSON inputs under
`newcomb_wildflower_guide/experiment_repro/prompt_sets/`. Choose one with
`--prompt-set`; the selected version changes the P1/P2 prompt text and is
recorded in both the result CSV and metadata. Current prompt sets:

- `stepwise-v1`
- `stepwise-v1-concise`
- `stepwise-v1-strict`

`--image-set` is a CSV filename under
`newcomb_wildflower_guide/experiment_repro/output/`. The default is
`sample.csv`, so this argument currently resolves to
`newcomb_wildflower_guide/experiment_repro/output/sample.csv`. Future image
sets can be added by placing another CSV in that output directory.

`--sample-limit` controls how many rows are taken from that CSV. Use a positive
integer for the first N rows, or `all` to run the full CSV. The image-set CSV
must include these columns: `newcomb_species_name`, `species_inat`, `taxon_id`,
`observation_id`, `photo_id`, and `photo_url`.

`--data-split` currently accepts only `all`; train/validation/test split
manifests have not been implemented.

## Model Adapter

Wrapper:

- `scripts/run_stepwise_trial.py`

Local runner:

- `newcomb_wildflower_guide/experiment_repro/run_stepwise_local.py`

Command adapter:

- `scripts/adapters/openrouter_command_adapter.py`

Adapter contract:

- Runner sends JSON on stdin:
  `{"model": "...", "parts": [...]}`
- `parts` can contain text strings and image dictionaries like
  `{"image": "https://..."}`.
- Adapter converts this to OpenRouter chat-completions message content.
- Adapter prints the model's text response to stdout.

OpenRouter request settings:

- `temperature`: defaults to `0`
- `max_tokens`: defaults to `200`
- timeout: controlled by `--timeout` / `OPENROUTER_TIMEOUT`
- seed: not currently set

For each observation-feature pair, the stepwise runner makes one P1 visibility
call. If P1 parses as `YES`, it makes a second P2 multiple-choice call.

## Analysis Views

Build the analysis CSVs from one or more trial result files:

```bash
uv run python scripts/analyze_stepwise_results.py \
  --input trials/artifacts/example.csv \
  --out-dir trials/analysis/example
```

The analysis step writes three layers of views:

- per-trial rows: `per_trial_rows.csv`, the raw CSV plus derived fields such as
  `p1_sees_feature`, `p2_inconclusive`, `correct_value`, `wrong_value`, and
  `predicted_value`, and `outcome`
- per-feature/model summary: `summary_by_model_feature.csv`,
  `outcome_by_true_value.csv`, and `outcome_pairs.csv`
- whole-experiment dashboard: `dashboard_whole_experiment.csv` and
  `summary_overall.csv`

Each row has exactly one primary outcome:

- `CORRECT`: `p2_parsed == true_value`
- `WRONG`: `p2_parsed` is a concrete value and does not equal `true_value`
- `INCONCLUSIVE`: `p2_parsed == INCONCLUSIVE`

`outcome_by_true_value.csv` groups by `model`, `feature`, and `true_value`, then
reports `correct_count`, `wrong_count`, `inconclusive_count`, `correct_rate`,
`wrong_rate`, `inconclusive_rate`, and `most_common_wrong_prediction`.

`metric_definitions.csv` records each metric's numerator, denominator, CSV
columns, and signal caveat. In particular, `features_seen` is based on
`p1_parsed == YES`; it is useful as a visibility-gate signal, but it is not
accuracy unless a human visibility label is added.

## Evaluation Fields

The output CSV includes:

- `true_value`: expected Newcomb key value from `newcomb_preprocessed.csv`
- `p1_raw`, `p1_parsed`: raw and parsed visibility answer
- `p2_raw`, `p2_parsed`: raw and parsed multiple-choice answer
- `feature_correct`: `True` when the parsed answer matches the expected value
- `committed`: `True` when the model committed to a concrete P2 value
- `trial_id`: trial identifier used to link CSV rows to metadata
- `run_id`: repeated-run label used by resume logic, so the same
  model/observation/feature can be run again under a different run id

## Common Errors

### Multiline Command Breaks at `--features`

Symptom:

```text
run_stepwise_trial.py: error: unrecognized arguments:
zsh: command not found: --features
```

Cause: a trailing space after a line-continuation backslash.

Fix: make sure each `\` is the final character on its line, or use a one-line
command.

### Missing API Key

Symptom:

```text
OpenRouter adapter error: OPENROUTER_API_KEY is required
```

Fix: add `OPENROUTER_API_KEY=...` to `.env` at the repo root, or export it in
the shell.

### Unsupported Feature

Symptom:

```text
Unsupported EXPERIMENT_FEATURES: ...
```

Fix: use one or more currently supported features:
`key_flower_type,key_plant_type,key_leaf_type`.
