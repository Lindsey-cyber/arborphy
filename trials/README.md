# Trials

This folder stores stepwise trial result CSVs under `trials/artifacts/`.

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
  --sample-limit 20 \
  --features key_flower_type,key_plant_type,key_leaf_type \
  --workers 4 \
  --timeout 120 \
  --mode command \
  --out-file stepwise_trial_result.csv \
  --trial-id trial_001
```

Supported feature names are currently `key_flower_type`, `key_plant_type`, and
`key_leaf_type`.

If `--out-file` is omitted, the wrapper writes an auto-named CSV to
`trials/artifacts/` using the run time, model, sample limit, and feature list,
for example:

```text
trials/artifacts/stepwise-20260624-143000-openai-gpt-4o-mini-n1-key_flower_type.csv
```

If `--out-file` is a relative filename, it is also written under
`trials/artifacts/`. Use an absolute path only when you intentionally want to
write somewhere else.

The generated CSV does not include `trial_id` or `model_command` columns.
