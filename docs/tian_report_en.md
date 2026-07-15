# Tian Report Summary: John/Newcomb Vision Benchmark Reproduction

## 1. Current Goal

Over the past few weeks, the main goal has been to reproduce John’s Newcomb/iNaturalist vision benchmark and turn it into a more reproducible, configurable, and analyzable experiment framework.

The core task is to evaluate vision models on Newcomb primary features in plant images:

- `key_flower_type`
- `key_plant_type`
- `key_leaf_type`

The remaining work is to clarify:

- the formal benchmark definition
- the final metrics
- the dashboard / visualization design

## 2. Main Engineering Progress

The original terminal-only stepwise experiment has been reorganized into a cleaner notebook workflow.

The first notebook cell now controls the main experiment parameters:

- `model` / multiple models
- `image_set`
- `sample_limit`
- `features`
- `prompt_set`
- `temperature`
- `workers`
- `timeout`
- `trial_id` / `run_id`

The main execution logic has been moved into `utilities.py` and `stepwise_trial.py`, making the notebook much cleaner.

Each trial saves:

- result CSV
- metadata JSON
- prompt
- raw output
- parsed output
- true value
- committed / outcome
- cost / budget metadata

The current framework supports:

- running multiple models in one experiment
- live progress with completed / pending calls
- automatic artifact saving under `trials/artifacts`
- interactive HTML result pages
- dashboard views for experiment scale, single-row artifact examples, P1/P2 red-yellow-green plots, per-model summary, per-feature summary, and whole-experiment summary

## 3. Dataset Setup

### `references.csv`

`references.csv` currently contains 67 rows of reference material. Each row contains:

- `feature`
- `feature_value`
- `reference_image_link`
- `reference_description`

Among these, the primary features contain 17 feature-value reference examples:

- flower type: 7
- plant type: 6
- leaf type: 4

The remaining 50 rows are subgroup reference entries.

These reference images were manually curated by John as representative images for each feature value, and they are intended to be clear visual examples.

`references.csv` is best suited for calibration: testing whether models understand Newcomb feature/value labels on clear reference examples.

### `sample.csv`

`sample.csv` contains 90 iNaturalist full photos:

- 30 species
- 3 images per species
- each image has species-level Newcomb values

Important limitation: `sample.csv` is not a photo-level visibility label dataset. It does not tell us whether a given feature is actually visible in each photo.

Therefore, `sample.csv` can test full-photo behavior, but it cannot directly evaluate the accuracy of P1 visibility judgments.

### Key Difference

| Dataset | Image | Label | Best Use | Main Limitation |
|---|---|---|---|---|
| `references.csv` | reference image + illustration | feature / feature_value | calibration | not a sample-photo visibility label |
| `sample.csv` | iNaturalist full photo | species-level Newcomb values | full-photo benchmark | no photo-level human visibility ground truth |

## 4. Prompt / Input Ablation

Because `references.csv` contains reference images, illustrations, and text descriptions, it functions as both visual reference material and label/context material.

The next step is to run prompt/input ablations to test whether the benchmark is sensitive to input format, for example:

- text only
- text + reference image
- text + illustration
- text + reference image + illustration

The goal is not only to compare accuracy, but also to test whether models are learning the abstract visual pattern rather than depending on one specific reference image or prompt wording.

## 5. John Benchmark Reproduction

### Calibration Reproduction

John’s `references.csv` has been integrated into the pipeline, and calibration reproduction has been run.

Calibration means:

> testing whether models can understand Newcomb feature/value labels on clear reference examples.

It is not a full-photo benchmark. It evaluates whether models understand the standard feature labels.

Calibration uses 17 primary reference examples. Each example is evaluated with three tasks:

- Existence: can the model detect the feature?
- Agreement: does the model agree with John’s label?
- Blind MC: can the model select the correct feature value from options without being given the answer?

Current calibration result scale:

- 6 models
- 17 primary values
- 3 prompt types
- 306 rows total

### Direct Species Baseline

The direct species baseline provides a direct image-to-species comparison point against the KG / stepwise pipeline.

Current result scale:

- 90 images
- 3 runs per image
- 6 models
- 1620 predictions total

This baseline is important because it tests:

> whether the KG / stepwise pipeline adds value beyond direct image-to-species recognition.

## 6. Metrics

Earlier TP / FP / FN / TN metrics were removed because they did not fit this task well.

Each row is now categorized into one of four main outcomes:

- `CORRECT`: `p2_parsed == true_value`
- `WRONG`: `p2_parsed` is a concrete value but does not equal `true_value`
- `INCONCLUSIVE`: `p2_parsed == INCONCLUSIVE`
- `N/A`: P1 did not pass the visibility gate, so P2 was skipped. In the code, this is stored as `NOT_APPLICABLE`

Based on the current code, the calculations are:

- `feature_count`: total rows in the current group
- `p2_applicable_count`: `count(p1_parsed == 'YES')`
- `correct_count`: `count(p1_parsed == 'YES' and p2_parsed == true_value)`
- `wrong_count`: `count(p1_parsed == 'YES' and p2_parsed is concrete and p2_parsed != true_value)`
- `inconclusive_count`: `count(p1_parsed == 'YES' and p2_parsed == 'INCONCLUSIVE')`
- `not_applicable_count`: `count(p1_parsed != 'YES')`, or rows where `p2_parsed` is `NA` / `N/A` / `SKIPPED` / `NOT_APPLICABLE`

Aggregated metrics include:

- `correct_count`
- `wrong_count`
- `inconclusive_count`
- `not_applicable_count`
- `correct_rate`
- `wrong_rate`
- `inconclusive_rate`
- `not_applicable_rate`
- `most_common_wrong_prediction`
- `committed_accuracy`

Main rate denominators:

- `correct_rate = correct_count / feature_count`
- `wrong_rate = wrong_count / feature_count`
- `inconclusive_rate = inconclusive_count / feature_count`
- `not_applicable_rate = not_applicable_count / feature_count`
- `p2_correct_rate = correct_count / p2_applicable_count`
- `p2_wrong_rate = wrong_count / p2_applicable_count`
- `p2_inconclusive_rate = inconclusive_count / p2_applicable_count`
- `committed_accuracy = correct_count / (correct_count + wrong_count)`

### P1 / P2 Semantics

P1 is the visibility gate:

- `YES`
- `NO`
- `INCONCLUSIVE`

P2 should only classify the feature value when P1 = `YES`.

If P1 is not `YES`, P2 should be treated as skipped / N/A rather than as a normal inconclusive classification.

## 7. Human Visible Tags and Manual Audit Pilot

The main purpose of human visible tags is to evaluate whether the P1 visibility gate is effective.

With human visible tags, we can separate:

- cases where a feature is visible but the model refuses to classify it
- cases where a feature is not visible but the model guesses anyway
- cases where a feature is visible and the model proceeds, but P2 classifies it incorrectly

A small manual audit pilot has been created:

- 10 images selected from `sample.csv`
- 3 primary features per image
- 30 audit items total

Related files:

- CSV: `manual_audit/sample_10_manual_audit.csv`
- fillable HTML: `manual_audit/sample_10_review.html`
- local server write-back tool: `manual_audit/review_server.py`

The goal of this pilot is not to immediately build a full dataset, but to test whether human visibility labels can help evaluate P1.

## 8. Current Assessment

There are two separate questions:

1. Is the experiment runner stable, reproducible, and analyzable?
2. Are the current data and labels sufficient to define a reliable benchmark?

The first question is mostly in good shape: the runner, notebook, artifacts, and dashboard are now largely built.

The second question is not fully resolved:

- `sample.csv` does not have photo-level visibility ground truth
- `references.csv` is calibration reference material, not human visibility labels for `sample.csv`
- evaluating P1 YES / NO / INCONCLUSIVE accuracy requires human visible tags
- a cleaner feature-recognition benchmark may require human-labeled bounding boxes or crop images

## 9. Future Direction Setup

The next benchmark design should include several separable experiments to avoid mixing direct model recognition ability with KG reasoning ability.

Suggested experiment routes:

1. Direct image -> species
2. Direct image -> feature values
3. Predicted feature values -> KG -> species
4. Human feature values -> KG -> species
5. Image + KG candidates -> species
6. Image + random/wrong KG candidates -> species

These ablations can help determine:

- whether the KG itself is useful
- whether the bottleneck is visual feature extraction or KG mapping
- whether the model actually uses KG candidates
- the upper bound of the KG pipeline when given human-labeled feature values

## 10. One-Sentence Summary for Tian

I have reproduced John/Newcomb’s stepwise vision benchmark as a cleaner notebook-based experiment framework that supports multi-model runs, artifact saving, metadata/cost tracking, and dashboard visualization. The main issue has shifted from whether the experiment can run to how the benchmark should be defined: `references.csv` is useful for calibration, while `sample.csv` supports full-photo behavior testing but lacks photo-level visibility ground truth. The next step is to finalize metrics and visualization, validate P1 with manual human-visible tags or crop labels, and run ablations comparing direct recognition with KG-based pipelines.
