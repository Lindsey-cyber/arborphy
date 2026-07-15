# Tian Report Summary: John/Newcomb Benchmark Reproduction

## 1. Current Goal

Over the past few weeks, the main goal has been to reproduce John’s Newcomb/iNaturalist benchmark and turn the original experiment into a more reproducible, configurable, and analyzable framework.

The core task is to evaluate six vision models on three Newcomb primary features:

| Primary feature | Options |
|---|---|
| `key_flower_type` | `3 Regular Parts`; `4 Regular Parts`; `5 Regular Parts`; `6 Regular Parts`; `7 or More Regular Parts`; `Irregular Flowers`; `Parts Indistinguishable` |
| `key_plant_type` | `Shrubs`; `Vines`; `Wildflowers - Alternate Leaves`; `Wildflowers - Basal Leaves Only`; `Wildflowers - No Apparent Leaves`; `Wildflowers - Opposite or Whorled Leaves` |
| `key_leaf_type` | `Leaves Divided`; `Leaves Entire`; `Leaves Toothed or Lobed`; `No Apparent Leaves` |

The highest-priority open items are:

- the formal benchmark definition
- the final metrics
- the dashboard / visualization design

## 2. Main Engineering Progress

The original terminal-only stepwise experiment has been reorganized into a notebook workflow. The main execution logic has been moved into `utilities.py` and `scripts/run_stepwise_trial.py`, so the notebook itself is cleaner.

The first notebook cell has two types of parameters:

| Parameter Type | Parameter | Purpose |
|---|---|---|
| Main experiment setting | `MODEL` / multiple models | Select model(s) to test |
| Main experiment setting | `IMAGE_SET` | Select the input image CSV, such as `sample.csv` |
| Main experiment setting | `SAMPLE_LIMIT` | Control how many images to run; number or `all` |
| Main experiment setting | `FEATURES` | Select Newcomb features to evaluate |
| Main experiment setting | `PROMPT_SET` | Select prompt version |
| Main experiment setting | `RUN_ID` / `TRIAL_ID` | Label an experiment for tracking and comparison |
| Helper/runtime setting | `TEMPERATURE`, `MAX_TOKENS` | Control model output behavior |
| Helper/runtime setting | `WORKERS`, `TIMEOUT` | Control concurrency and timeout |
| Helper/runtime setting | `DATA_SPLIT`, `MODE`, `OUT_FILE` | Record split, run mode, and output path |

Each trial saves a result CSV and metadata JSON, including:

- prompt
- raw output
- parsed output
- true value
- committed / outcome
- cost / budget metadata

The current framework supports:

- running multiple models in one experiment
- live progress with completed / pending calls / budget alerts
- automatic saving under `trials/artifacts`
- interactive HTML result pages
- dashboard views for experiment scale, single-row artifact examples, P1/P2 red-yellow-green plots, per-model summary, per-feature summary, and whole-experiment summary

## 3. Dataset Setup

### `references.csv`

`references.csv` currently has 67 reference entries:

- 17 primary feature-value reference examples
- 50 subgroup reference entries for later fine-grained feature/KG work

The 17 primary examples cover:

- flower type: 7
- plant type: 6
- leaf type: 4

These reference images were manually curated by John as representative images for each feature value. They are best suited for calibration: testing whether models understand our feature/value definitions on clear reference examples.

### `sample.csv`

`sample.csv` contains 90 iNaturalist full photos:

- 30 species
- 3 images per species
- each image has species-level Newcomb values

Important limitation: `sample.csv` is not a photo-level visibility label dataset. It does not tell us whether a given feature is actually visible in a specific photo.

### Key Difference

| Dataset | Image | Label | Best Use | Main Limitation |
|---|---|---|---|---|
| `references.csv` | reference image + illustration | feature / feature_value | calibration | not a sample-photo visibility label |
| `sample.csv` | iNaturalist full photo | species-level Newcomb values | full-photo benchmark | no photo-level human visibility ground truth |

## 4. John Benchmark Reproduction and Ablation

### Calibration Reproduction

John’s `references.csv` has been integrated into the pipeline, and calibration reproduction has been run.

Purpose of calibration: test whether models understand standard Newcomb feature labels on clear reference examples.

Calibration uses 17 primary reference examples. Each example is evaluated with three tasks:

- `Existence`: can the model detect the feature?
- `Agreement`: does the model agree with John’s label?
- `Blind MC`: can the model select the correct feature value from options without being directly given the answer?

Current calibration result scale:

- 6 models
- 17 primary values
- 3 prompt types
- 306 rows total

### Direct Species Baseline

The direct species baseline provides a direct image-to-species comparison point against the KG / stepwise pipeline.

Current direct baseline scale:

- 90 images
- 3 runs per image
- 6 models
- 1620 predictions total

It answers the question: does the KG / stepwise pipeline add value beyond direct image-to-species recognition?

### No-reference-photo Blind MC Ablation

Because calibration `blind_mc` gives the model the test image together with option reference material, we need to separate feature/value understanding from reference-image matching.

The first ablation removed option reference photos while keeping option text / description / illustration.

| Prompt | Correct | Accuracy |
|---|---:|---:|
| original `blind_mc` | 85/102 | 83.3% |
| `blind_mc_no_reference_photos` | 58/102 | 56.9% |

After removing option reference photos, accuracy dropped from 83.3% to 56.9%, a 26.5 percentage-point decrease.

By model:

| Model | Original | No Ref Photos | Drop |
|---|---:|---:|---:|
| Gemini Flash | 100.0% | 76.5% | -23.5 |
| Claude Sonnet | 82.4% | 64.7% | -17.6 |
| Gemini 3.1 Pro | 88.2% | 58.8% | -29.4 |
| Claude Haiku | 58.8% | 52.9% | -5.9 |
| GPT-4o mini | 82.4% | 41.2% | -41.2 |
| GPT-5 mini | 88.2% | 47.1% | -41.2 |

By feature:

| Feature | Original | No Ref Photos | Drop |
|---|---:|---:|---:|
| `key_flower_type` | 83.3% | 52.4% | -31.0 |
| `key_plant_type` | 83.3% | 52.8% | -30.6 |
| `key_leaf_type` | 83.3% | 70.8% | -12.5 |

More precise interpretation: this experiment mainly shows that **identical reference photos in calibration are very helpful**. In the original `blind_mc`, the correct option reference photo may be the same reference exemplar as the test image, so the model may partly be doing identical-image matching.

This does not mean all reference photos are unhelpful, and it does not test whether non-identical reference exemplars would help. The next step should be to add more reference exemplars, especially multiple non-identical representative images per feature value, and then run held-out / non-identical reference ablations.

## 5. Metrics and Human Visible Tags

Earlier TP / FP / FN / TN metrics were removed because they did not fit this task well.

Each row is now categorized into one of four outcomes:

- `CORRECT`: P1 = `YES`, and `p2_parsed == true_value`
- `WRONG`: P1 = `YES`, and P2 gives a concrete value that does not equal `true_value`
- `INCONCLUSIVE`: P1 = `YES`, but `p2_parsed == INCONCLUSIVE`
- `N/A`: P1 did not pass the visibility gate, so P2 was skipped. In the code, this is stored as `NOT_APPLICABLE`

P1 / P2 semantics:

- P1 is the visibility gate: `YES` / `NO` / `INCONCLUSIVE`
- P2 only classifies feature value when P1 = `YES`
- If P1 is not `YES`, P2 is skipped / N/A, not a normal inconclusive classification

Current core calculations:

| Metric | Formula |
|---|---|
| `feature_count` | total rows in the current group |
| `p2_applicable_count` | `count(p1_parsed == 'YES')` |
| `correct_count` | `count(p1_parsed == 'YES' and p2_parsed == true_value)` |
| `wrong_count` | `count(p1_parsed == 'YES' and p2_parsed is concrete and p2_parsed != true_value)` |
| `inconclusive_count` | `count(p1_parsed == 'YES' and p2_parsed == 'INCONCLUSIVE')` |
| `not_applicable_count` | `count(p1_parsed != 'YES')`, or rows where `p2_parsed` is `NA` / `N/A` / `SKIPPED` / `NOT_APPLICABLE` |

Main aggregated metrics:

- `correct_rate`
- `wrong_rate`
- `inconclusive_rate`
- `not_applicable_rate`
- `most_common_wrong_prediction`
- `committed_accuracy`

Human visible tags are used to evaluate whether the P1 visibility gate is effective. With human visible tags, we can separate:

- feature is visible, but the model refuses to continue
- feature is not visible, but the model still guesses
- feature is visible and the model continues, but P2 classifies it incorrectly

A small manual audit pilot has been created:

- 10 images selected from `sample.csv`
- 3 primary features per image
- 30 audit items total
- CSV: `manual_audit/sample_10_manual_audit.csv`
- HTML review: `manual_audit/sample_10_review.html`
- local write-back tool: `manual_audit/review_server.py`

The goal of this pilot is not to immediately build a full dataset, but to test whether human visibility labels can help evaluate P1.

## 6. Open Issues

- `sample.csv` does not have photo-level visibility ground truth
- Evaluating P1 YES / NO / INCONCLUSIVE accuracy requires human visible tags
- A cleaner feature-recognition benchmark may require human-labeled bounding boxes or crop images
- The reference exemplar set is still too small; the next step is to add multiple non-identical representative images per feature value

## 7. Future Direction Setup

The next benchmark design should include separable experiments to avoid mixing direct model recognition ability with KG reasoning ability.

Suggested experiment routes:

1. Direct image + AI -> species
2. Direct image + AI -> feature values
3. Predicted feature values -> KG -> species
4. Human feature values -> KG -> species
5. Image + KG candidates -> species
6. Image + random/wrong KG candidates -> species

These ablations can help determine:

- whether the KG itself is useful
- whether the bottleneck is visual feature extraction or KG mapping
- whether the model actually uses KG candidates
- the upper bound of the KG pipeline when given human-labeled feature values
