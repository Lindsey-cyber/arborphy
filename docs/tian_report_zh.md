# Tian 汇报简版：John/Newcomb Vision Benchmark 复现与整理

## 1. 当前目标

这几周的主要目标是复现 John 的 Newcomb/iNaturalist vision benchmark，并把原始实验整理成一个更可复现、可调参、可分析的实验框架。

当前核心任务是测试 vision models 对植物图像中 Newcomb primary features 的识别能力：

- `key_flower_type`
- `key_plant_type`
- `key_leaf_type`

接下来仍需要继续明确三件事：

- benchmark 的正式定义
- metrics 的最终定义
- dashboard / visualization 的展示设计

## 2. 已完成的工程进展

原来只能在 terminal 里运行的 stepwise experiment 已经整理成 notebook workflow。

Notebook 现在可以在第一个 cell 控制主要实验参数：

- `model` / 多模型
- `image_set`
- `sample_limit`
- `features`
- `prompt_set`
- `temperature`
- `workers`
- `timeout`
- `trial_id` / `run_id`

主要运行逻辑已经抽到 `utilities.py` 和 run_`stepwise_trial.py`，notebook 本身更干净。

每次 trial 会保存：

- result CSV
- metadata JSON
- prompt
- raw output
- parsed output
- true value
- committed / outcome
- cost / budget metadata

当前实验框架已经支持：

- 一次运行多个模型
- live progress，显示 completed / pending calls
- 自动保存到 `trials/artifacts`
- 生成 HTML 形式的交互结果页面
- dashboard 展示实验规模、单行 artifact example、P1/P2 红黄绿图、per-model summary、per-feature summary、whole-experiment summary

## 3. Dataset Setup

### `references.csv`

`references.csv` 当前有 67 行 reference material。每行包含：

- `feature`
- `feature_value`
- `reference_image_link`
- `reference_description`

其中 primary features 有 17 个 feature-value reference examples：

- flower type: 7
- plant type: 6
- leaf type: 4

另外还有 50 个 subgroup reference entries。

这些 reference images 是 John manually curated 的 representative images，用来作为每个 feature value 的清楚参考例子。

`references.csv` 更适合用于 calibration：测试模型在清楚 reference examples 上能不能理解 Newcomb feature/value。

### `sample.csv`

`sample.csv` 包含 90 张 iNaturalist full photos：

- 30 个 species
- 每个 species 3 张图
- 每张图有 species-level Newcomb values

重要限制：`sample.csv` 不是 photo-level visibility label。它不能告诉我们“这张照片里某个 feature 是否真的看得见”。

因此，`sample.csv` 可以测试 full-photo behavior，但不能直接评估 P1 visibility judgment 的准确性。

### 关键区别

| Dataset | Image | Label | 适合用途 | 主要限制 |
|---|---|---|---|---|
| `references.csv` | reference image + illustration | feature / feature_value | calibration | 不是 sample photo visibility label |
| `sample.csv` | iNaturalist full photo | species-level Newcomb values | full-photo benchmark | 没有 photo-level human visibility ground truth |

## 4. Prompt / Input Ablation

因为 `references.csv` 同时包含 reference image、illustration 和 text description，它既是 visual reference，又是 label/context material。

因此现在需要做 prompt/input ablation，测试当前 benchmark 对输入形式是否敏感，例如：

- 只给 text
- 给 text + reference image
- 给 text + illustration
- 给 text + reference image + illustration

目的不是只看 accuracy，而是判断模型是否真的理解 abstract visual pattern，而不是只依赖某张 reference image 或 prompt wording。

## 5. John Benchmark Reproduction

### Calibration Reproduction

已将 John 的 `references.csv` 接入 pipeline，并跑了 calibration reproduction。

Calibration 的意义是：

> 在清楚 reference examples 上，测试模型能不能理解 Newcomb feature/value。

它不是 full-photo benchmark，而是测试模型是否理解标准 feature labels。

Calibration 使用 17 个 primary reference examples，每个 example 跑三个任务：

- Existence: 模型是否能看到该 feature
- Agreement: 模型是否同意 John 的 label
- Blind MC: 不直接给答案，让模型从选项中选正确 feature value

John 的 calibration result 当前规模是：

- 6 models
- 17 primary values
- 3 prompt types
- 共 306 rows

### Direct Species Baseline

Direct species baseline 的意义是提供一个直接识别物种的 baseline，用来和 KG / stepwise pipeline 对比。

当前结果规模：

- 90 images
- 3 runs per image
- 6 models
- 共 1620 predictions

这个 baseline 很重要，因为它可以回答：

> KG / stepwise pipeline 是否真的比 direct image-to-species recognition 更有价值？

## 6. Metrics

早期尝试过 TP / FP / FN / TN，但后来发现不适合当前任务，所以已经去掉。

现在每行结果主要分为四类：

- `CORRECT`: `p2_parsed == true_value`
- `WRONG`: `p2_parsed` 是具体值，但不等于 `true_value`
- `INCONCLUSIVE`: `p2_parsed == INCONCLUSIVE`
- `N/A`: P1 没有通过 visibility gate，所以 P2 没有继续判断。代码里对应 `NOT_APPLICABLE`

按照现有代码，计算方式是：

- `feature_count`: 当前 group 里的总行数
- `p2_applicable_count`: `count(p1_parsed == 'YES')`
- `correct_count`: `count(p1_parsed == 'YES' and p2_parsed == true_value)`
- `wrong_count`: `count(p1_parsed == 'YES' and p2_parsed is concrete and p2_parsed != true_value)`
- `inconclusive_count`: `count(p1_parsed == 'YES' and p2_parsed == 'INCONCLUSIVE')`
- `not_applicable_count`: `count(p1_parsed != 'YES')`，或 `p2_parsed` 是 `NA` / `N/A` / `SKIPPED` / `NOT_APPLICABLE`

聚合指标包括：

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

主要 rate 的分母：

- `correct_rate = correct_count / feature_count`
- `wrong_rate = wrong_count / feature_count`
- `inconclusive_rate = inconclusive_count / feature_count`
- `not_applicable_rate = not_applicable_count / feature_count`
- `p2_correct_rate = correct_count / p2_applicable_count`
- `p2_wrong_rate = wrong_count / p2_applicable_count`
- `p2_inconclusive_rate = inconclusive_count / p2_applicable_count`
- `committed_accuracy = correct_count / (correct_count + wrong_count)`

### P1 / P2 语义

P1 是 visibility gate：

- `YES`
- `NO`
- `INCONCLUSIVE`

只有 P1 = `YES` 时，P2 才应该继续判断 feature value。

如果 P1 不是 `YES`，P2 应理解为 skipped / N/A，而不是普通 inconclusive。

## 7. Human Visible Tag 与 Manual Audit Pilot

Human visible tag 的主要意义是判断 P1 visibility gate 是否有效。

如果有 human visible tag，就可以区分：

- feature 明明可见，但模型没有继续判断
- feature 不可见，但模型仍然强行猜
- feature 可见，模型也判断了，但 P2 识别错

为此已经做了一个 manual audit pilot：

- 从 `sample.csv` 选了 10 张图
- 每张图标 3 个 primary features
- 共 30 行 audit items

相关文件：

- CSV: `manual_audit/sample_10_manual_audit.csv`
- 可填写 HTML: `manual_audit/sample_10_review.html`
- 本地 server 写回工具: `manual_audit/review_server.py`

这个 pilot 的目的不是立刻扩成完整数据集，而是验证人工标注 visibility 是否能帮助评估 P1。

## 8. 当前判断

目前需要分开看两个问题：

1. 实验机器是否跑得稳定、可复现、可分析？
2. 当前数据和标签是否足够定义一个可靠 benchmark？

第一个问题目前进展较好：runner、notebook、artifacts、dashboard 都已经基本搭起来。

第二个问题还没有完全解决：

- `sample.csv` 没有 photo-level visibility ground truth
- `references.csv` 是 calibration reference material，不是 sample 的 human visibility label
- 如果要评估 P1 的 YES / NO / INCONCLUSIVE 是否准确，需要 human visible tags
- 如果要更干净地评估 feature recognition，可能需要人工标注 bounding box / crop dataset

## 9. Future Direction Setup

下一步建议把 benchmark 设计成几组可区分的实验，避免把模型裸眼识别能力和 KG 推理能力混在一起。

建议实验路线：

1. Direct image -> species
2. Direct image -> feature values
3. Predicted feature values -> KG -> species
4. Human feature values -> KG -> species
5. Image + KG candidates -> species
6. Image + random/wrong KG candidates -> species

这些 ablations 可以帮助判断：

- KG 本身是否有用
- bottleneck 是 visual feature extraction 还是 KG mapping
- 模型是否真的使用 KG candidates
- human-labeled feature values 能给 KG pipeline 提供多高的上限

## 10. 给 Tian 的一句话总结

当前我已经把 John/Newcomb 的 stepwise vision benchmark 复现成了一个更清楚的 notebook-based experiment framework，可以多模型运行、保存 artifacts、记录 metadata/cost，并生成 dashboard。现在最重要的问题已经从“能不能跑”转向“benchmark 到底如何定义”：`references.csv` 适合 calibration，`sample.csv` 适合 full-photo behavior test，但它没有 photo-level visibility ground truth。下一步需要明确 metrics 和 visualization，同时通过 manual audit / human visible tags 或 crop dataset 来验证 P1 visibility gate，并设计 direct baseline 与 KG pipeline 的 ablation 对比。
