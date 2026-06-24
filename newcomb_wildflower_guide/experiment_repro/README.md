# Newcomb Local Experiment Reproduction

This folder recreates John Matter's Spring-2026 Newcomb experiment inputs from
the newer `descriptive_vocab_assets` source data.

It is rooted in `descriptive_vocab_assets/newcomb_wildflower_guide/` rather
than `JM_Assets/`, but it temporarily reuses the 11 JM Newcomb illustration
PNGs because the canonical Newcomb illustration extraction has not yet been
completed.

## What this generates

Running `build_inputs.py` creates JM-style experiment inputs under `output/`:

- `feature_value_pairs.csv`
- `references.csv`
- `newcomb_preprocessed.csv`
- `sample.csv`
- `illustration_paths.json`
- `references_with_paths.csv`

## Local runners

This folder also includes local runner scripts that mirror the JM experiment
shape without depending on John's missing `models.py`.

- `run_calibration_local.py`
- `run_stepwise_local.py`
- `model_adapter.py`

The runners use a simple adapter contract:

- `EXPERIMENT_MODEL_MODE=mock`
  - default
  - returns deterministic placeholder responses so the pipeline can be tested
- `EXPERIMENT_MODEL_MODE=command`
  - shells out to `EXPERIMENT_MODEL_COMMAND`
  - sends `{"model": ..., "parts": [...]}` JSON on stdin
  - expects plain text response on stdout

## Data sources

- Newcomb parquets from `../extracted_vocabulary/parquet/`
- Site observations from `../../observations_pound_ridge/`
- JM illustration PNGs from `../../../JM_Assets/data/illustrations/`

## Notes

- `feature_value_pairs.csv` is generated from the Newcomb route-step model.
- `newcomb_preprocessed.csv` is reconstructed by expanding taxon-to-route links
  into a flattened JM-style path table.
- `references.csv` is a best-effort bridge table. Because the canonical Newcomb
  illustration extraction is still pending, only values that can be matched to
  JM's hand-curated illustrations receive local illustration paths.
- `sample.csv` is rebuilt from the current Pound Ridge observation snapshot,
  joined by scientific name to Newcomb taxa.

## Run

```bash
uv run python build_inputs.py
```

Dry-run the local runners with the mock adapter:

```bash
uv run python run_calibration_local.py
uv run python run_stepwise_local.py
```

Use a real adapter command:

```bash
EXPERIMENT_MODEL_MODE=command \
EXPERIMENT_MODEL_COMMAND='uv run python ../../scripts/adapters/openrouter_command_adapter.py' \
EXPERIMENT_MODELS='gpt-5-mini,claude-sonnet-4-6' \
uv run python run_stepwise_local.py
```

Run the smallest real OpenRouter stepwise smoke test:

```bash
cd /Users/lindseyma/Downloads/JM_Assets/descriptive_vocab_assets
uv run python scripts/run_stepwise_trial.py \
  --model openrouter/free \
  --sample-limit 10 \
  --features key_flower_type
```

The stepwise output records the image URL, prompt parts JSON, raw model answers,
parsed answers, and parse rules. For each trial, set a distinct
`EXPERIMENT_OUT_FILE` if you want a fixed name; otherwise the runner creates an
auto-named CSV using the timestamp, model, sample limit, and feature list.

`openrouter/free` is a local convenience alias. It resolves to a current free
OpenRouter model that supports image input. To force a specific free model, set
`OPENROUTER_FREE_MODEL` to a concrete model id ending in `:free`.

For OpenRouter, set `OPENROUTER_API_KEY` first. The included
`../../scripts/adapters/openrouter_command_adapter.py` reads the runner payload from stdin, calls
OpenRouter chat completions, and prints the model response to stdout.

Small text-only adapter smoke test:

```bash
export OPENROUTER_API_KEY='...'
printf '{"model":"openai/gpt-4o-mini","parts":["Reply with OK only."]}' \
  | uv run python ../../scripts/adapters/openrouter_command_adapter.py
```
