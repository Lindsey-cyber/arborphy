# Trial Log: JM Reproduction

## Trial ID
jm-repro-001

## Trial lifecycle

| Step | Status | Notes |
|---|---|---|
| `git checkout X` | Partial | Work started on `main` before the formal trial branch process was written down. |
| Work on experiment setup | Done | Fixed Python/UV setup, added OpenRouter command adapter, added stepwise smoke-test controls. |
| `git checkout X.x` | Not done | Use a dedicated branch/checkpoint for the next trial. |
| Run experiment | Done | Ran 1 sample, 1 feature, 1 OpenRouter model. |
| Gather artifacts | Done | See artifact section below. |
| Log results | Done | This file records the trial. |
| Commit | Pending | Commit only intended setup/log files; exclude unrelated local changes. |

## Goal
Reproduce the JM/John stepwise pipeline on current asset data with one real
OpenRouter model call path.

## Environment
- Package manager: UV
- Model gateway: OpenRouter
- Python version: 3.12.10
- API key env var: OPENROUTER_API_KEY
- Data path: `newcomb_wildflower_guide/experiment_repro/output`

## Setup commands
- `uv sync`
- `printf '{"model":"openai/gpt-4o-mini","parts":["Reply with OK only."]}' | uv run python ../../scripts/adapters/openrouter_command_adapter.py`

## Experiment command

```bash
cd /Users/lindseyma/Downloads/JM_Assets/descriptive_vocab_assets/newcomb_wildflower_guide/experiment_repro

EXPERIMENT_MODEL_MODE=command \
EXPERIMENT_MODEL_COMMAND='uv run python ../../scripts/adapters/openrouter_command_adapter.py' \
EXPERIMENT_MODELS='openai/gpt-4o-mini' \
EXPERIMENT_SAMPLE_LIMIT=1 \
EXPERIMENT_FEATURES='key_flower_type' \
EXPERIMENT_NUM_WORKERS=1 \
EXPERIMENT_OUT_FILE='jm-repro-001-openrouter-smoke.csv' \
uv run python run_stepwise_local.py
```

## Output artifact
- Path: `newcomb_wildflower_guide/experiment_repro/output/stepwise_results_local.csv`
- Note: this file currently contains prior mock/local rows plus this real OpenRouter row.
- Next clean artifact path should be: `newcomb_wildflower_guide/experiment_repro/output/jm-repro-001-openrouter-smoke.csv`
- Committed clean artifact copy: `trials/artifacts/jm-repro-001-openrouter-smoke.csv`

## Artifact details

| Field | Value |
|---|---|
| Model | `openai/gpt-4o-mini` |
| Observation ID | `7089628` |
| Photo URL | `https://inaturalist-open-data.s3.amazonaws.com/photos/9108809/medium.jpeg` |
| Taxon ID | `85320` |
| Species | `Monarda fistulosa` |
| Newcomb species name | `Monarda fistulosa` |
| Feature | `key_flower_type` |
| Expected / true value | `Irregular Flowers` |

## Model answers and parsing

| Prompt step | Raw model answer | Parsed answer | Parse rule |
|---|---|---|---|
| P1 existence check | `YES` | `YES` | `parse_ync`: response starts with `YES`, so parsed as `YES`. |
| P2 blind multiple choice | `Cannot determine from this image.` | `INCONCLUSIVE` | `parse_mc`: response contains `cannot determine`, so parsed as `INCONCLUSIVE`. |

## Prompt artifacts

Future clean rows for this trial should include:

- `p1_prompt_parts_json`
- `p2_prompt_parts_json`

These fields store the exact prompt parts sent to the model, including the image
reference and prompt text/options. The first real smoke-test row was generated
before these prompt columns were added, so rerun with the clean artifact command
above.

## Evaluation result
- `feature_correct=True`
- `committed=False`
- Plain interpretation: the model said the flower type was visible, but did not commit to a specific flower type. This is counted as non-pruning, not as a wrong committed answer.

## Artifacts gathered
- `uv.lock`
- `scripts/adapters/openrouter_command_adapter.py`
- `newcomb_wildflower_guide/experiment_repro/output/stepwise_results_local.csv`
- `trials/artifacts/jm-repro-001-openrouter-smoke.csv`
- `trials/artifacts/jm-repro-002-openrouter-free-n10-key_flower_type.csv`
- `repro_notes.md` in the outer asset folder

## Results
- UV environment works with Python 3.12.10.
- OpenRouter adapter text smoke test works.
- Real OpenRouter stepwise smoke test works for 1 sample and 1 feature.
- Runner now supports `EXPERIMENT_SAMPLE_LIMIT`, `EXPERIMENT_FEATURES`, and `EXPERIMENT_OUT_FILE`.
- New stepwise rows now record `photo_url`, prompt parts JSON, raw model answers, parsed answers, and parse rules.

## Blockers
- Full JM reproduction has not been run yet.
- This trial's first real output was written into a mixed output file with mock rows.

## Next step
- Rerun with the concise wrapper and `--model openrouter/free` to create an auto-named clean per-trial artifact:
  `uv run python scripts/run_stepwise_trial.py --model openrouter/free --sample-limit 10 --features key_flower_type`
