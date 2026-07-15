# JM Thesis / Slides / Source Result Alignment

本报告对照了：

- `/Users/lindseyma/Downloads/JM_Final_Thesis.pdf`
- `/Users/lindseyma/Downloads/JM_Thesis_Prezo_May26 (1).pdf`
- `JM_Assets_2/run_calibration.py`
- `JM_Assets_2/run_feature_unlabeled.py`
- `JM_Assets_2/calibration_results.csv`
- `JM_Assets_2/feature_unlabeled_results.csv`
- 相关复现实验 artifact：`trials/artifacts/*.csv`

结论先说：论文 Table 5.2 和 Table 5.3 基本可以由 `JM_Assets_2` 里的两个结果 CSV 重算出来；但当前 repo 的整理版 runner / notebook artifact 已经改变了若干实验口径，因此不能直接逐列和论文表比较。

## 1. 三类实验的定义

| 实验 | 数据 | 目标 | 论文位置 | 对应代码 / 结果 |
|---|---:|---|---|---|
| Direct species identification | 90 张 iNaturalist 图片，30 个候选 species，3 runs/model | 不走 KG / feature pipeline，直接识别物种 | Table 5.1 | 论文/slides 有结果；`JM_Assets_2` 没有 direct 源码 |
| Calibration on curated reference set | 17 个 primary feature-value reference exemplars | 在人工确认表达的 reference image 上测试模型是否对齐 Newcomb feature vocabulary | Table 5.2, Figure 5.1/5.2 | `run_calibration.py`, `calibration_results.csv` |
| Stepwise/sample on expression-unlabeled subset | 90 张 sample full photos，3 primary features/model | P1 判断 feature 是否可见，P2 做 blind multiple choice | Table 5.3 | `run_feature_unlabeled.py`, `feature_unlabeled_results.csv` |

## 2. Direct 识别结果

论文和 slides 给出的 direct baseline 是：

| Provider | Model | Accuracy | 3-run consistency |
|---|---|---:|---:|
| OpenAI | gpt-4o-mini | 7% | 20% |
| OpenAI | gpt-5-mini | 12% | 25% |
| Google | gemini-3.1-pro | 32% | 47% |
| Google | gemini-3-flash | 36% | 52% |
| Anthropic | claude-sonnet-4-6 | 19% | 20% |
| Anthropic | claude-haiku-4-5 | 3% | 27% |

重要：当前 repo 里的 `trials/artifacts/direct-species-baseline-john-90x3-six-models.summary.csv` 不是这组论文数字。例如它显示 `google/gemini-3-flash-preview` accuracy 为 95.2%，`anthropic/claude-sonnet-4-6` 为 63.3%。所以 direct 这部分如果要复现论文，不能直接拿当前 artifact 当原始论文结果。

可能原因：

- `JM_Assets_2` 没有 John 原始 direct 脚本；当前 repo 的 `scripts/run_direct_species_baseline.py` 是后来的复现入口。
- 当前 direct prompt 明确给出 30 个候选 species 列表，并要求从候选里返回一个；原论文只描述为 direct species identification，未在正文给出完整 prompt。
- 模型版本、OpenRouter 路由、prompt/parser 都可能已经不同。

## 3. Calibration 结果

John 的 calibration 脚本实际跑了 3 类 prompt：

- `existence`: feature 是否 visually discernible；期望 `YES`
- `agreement`: 给出 true value，让模型判断是否一致；期望 `YES`
- `blind_mc`: 不给答案，从 value options 里选；期望选中 true value

论文 Table 5.2 只主要报告 `existence` 和 `blind_mc`；`agreement` 在 Figure 5.2 里作为诊断出现。

### 3.1 按模型聚合

| Model | Existence YES | Agreement YES | Blind MC correct | Blind MC inc | Blind MC wrong |
|---|---:|---:|---:|---:|---:|
| gpt-4o-mini | 16/17 (94.1%) | 11/17 (64.7%) | 14/17 (82.4%) | 1/17 | 2/17 |
| gpt-5-mini | 16/17 (94.1%) | 10/17 (58.8%) | 15/17 (88.2%) | 0/17 | 2/17 |
| gemini-3.1-pro | 16/17 (94.1%) | 15/17 (88.2%) | 15/17 (88.2%) | 0/17 | 2/17 |
| gemini-3-flash | 17/17 (100%) | 15/17 (88.2%) | 17/17 (100%) | 0/17 | 0/17 |
| claude-sonnet-4-6 | 16/17 (94.1%) | 8/17 (47.1%) | 14/17 (82.4%) | 0/17 | 3/17 |
| claude-haiku-4-5 | 17/17 (100%) | 12/17 (70.6%) | 10/17 (58.8%) | 0/17 | 7/17 |

这和论文 Table 5.2 的 `Sees Feature` / `Multiple Choice` 数字一致。

### 3.2 按 feature 粗看

| Feature | Reference values | 总体观察 |
|---|---:|---|
| key_flower_type | 7 | Flash 和 Gemini Pro 在 blind MC 上 7/7；Haiku 只有 4/7；gpt-4o-mini 有 1 个 inc 和 1 个 wrong |
| key_plant_type | 6 | 最大争议集中在 `Vines`，多个模型把它选成 `Wildflowers - Alternate Leaves` |
| key_leaf_type | 4 | Flash 4/4；Haiku 2/4；常见错法是把 toothed/lobed 或 divided 归到 entire |

典型案例和论文 Figure 5.2 一致：

- `Xyris caroliniana`, flower type = `3 Regular Parts`: 6 个模型 P1/Agreement/MC 基本全成功。
- `Vicia angustifolia`, plant type = `Vines`: 6 个模型都确认 P1 可见，但 4 个模型在 blind MC 里选成 `Wildflowers - Alternate Leaves`。

## 4. Sample / Stepwise 结果

论文 Table 5.3 的口径是：每个 feature/model 下，报告 90 张 sample image 上的 `Sees Feature`、`MC Correct`、`MC Inc`、`MC Incorrect`。这里的 `Inc` 在 John 旧 CSV 里包含了 P1 不是 YES 时被跳过的情况，因为旧脚本会把 P2 写成 `INCONCLUSIVE`。

### 4.1 论文 Table 5.3 可由 `feature_unlabeled_results.csv` 重算

| Feature | Model | n | Sees Feature | MC Correct | MC Inc | MC Incorrect |
|---|---|---:|---:|---:|---:|---:|
| flower | gpt-4o-mini | 90 | 66.7% | 6.7% | 86.7% | 6.7% |
| flower | gpt-5-mini | 90 | 62.2% | 36.7% | 43.3% | 20.0% |
| flower | gemini-3-flash | 90 | 77.8% | 55.6% | 30.0% | 14.4% |
| flower | gemini-3.1-pro | 83 | 66.3% | 44.6% | 38.6% | 16.9% |
| flower | claude-sonnet-4-6 | 90 | 62.2% | 46.7% | 43.3% | 10.0% |
| flower | claude-haiku-4-5 | 90 | 72.2% | 27.8% | 47.8% | 24.4% |
| plant | gpt-4o-mini | 90 | 81.1% | 32.2% | 47.8% | 20.0% |
| plant | gpt-5-mini | 90 | 94.4% | 51.1% | 21.1% | 27.8% |
| plant | gemini-3-flash | 90 | 100.0% | 81.1% | 7.8% | 11.1% |
| plant | gemini-3.1-pro | 82 | 98.8% | 76.8% | 13.4% | 9.8% |
| plant | claude-sonnet-4-6 | 90 | 98.9% | 60.0% | 5.6% | 34.4% |
| plant | claude-haiku-4-5 | 90 | 100.0% | 40.0% | 1.1% | 58.9% |
| leaf | gpt-4o-mini | 90 | 68.9% | 21.1% | 68.9% | 10.0% |
| leaf | gpt-5-mini | 90 | 85.6% | 65.6% | 16.7% | 17.8% |
| leaf | gemini-3-flash | 90 | 91.1% | 74.4% | 14.4% | 11.1% |
| leaf | gemini-3.1-pro | 83 | 79.5% | 61.4% | 24.1% | 14.5% |
| leaf | claude-sonnet-4-6 | 90 | 71.1% | 57.8% | 31.1% | 11.1% |
| leaf | claude-haiku-4-5 | 90 | 93.3% | 36.7% | 32.2% | 31.1% |

注意：`gemini-3.1-pro-preview` 在 `feature_unlabeled_results.csv` 里 primary 只有 248 行，不是完整 270 行：

- flower: 83
- plant: 82
- leaf: 83

论文表里 Gemini Pro 的百分比看起来是按这些实际行数算的，而不是固定用 90 作为分母。所以如果现在复跑补齐到 90/90/90，数字会自然不一样。

### 4.2 Overall primary 指标

如果把三个 primary features 合在一起，并按论文 Table 5.3 的 observation-level 口径算：

| Model | n | Sees Feature | MC Correct | MC Inc | MC Incorrect | Committed Acc |
|---|---:|---:|---:|---:|---:|---:|
| gpt-4o-mini | 270 | 72.2% | 20.0% | 67.8% | 12.2% | 62.1% |
| gpt-5-mini | 270 | 80.7% | 51.1% | 27.0% | 21.9% | 70.1% |
| gemini-3.1-pro | 248 | 81.5% | 60.9% | 25.4% | 13.7% | 81.6% |
| gemini-3-flash | 270 | 89.6% | 70.4% | 17.4% | 12.2% | 85.2% |
| claude-sonnet-4-6 | 270 | 77.4% | 54.8% | 26.7% | 18.5% | 74.7% |
| claude-haiku-4-5 | 270 | 88.5% | 34.8% | 27.0% | 38.1% | 47.7% |

这里 `Committed Acc` 是 `correct / (correct + wrong)`，对应论文后面 Table 5.4 里 Flash run 0 的 85.2%。它不是 Table 5.3 里的 `MC Correct` 列。

## 5. 源码和结果不一致的关键点

### 5.1 `run_feature_unlabeled.py` 当前脚本不是生成现有 CSV 的完整版本

文件头写的是 `Stepwise Feature Classification experiment (§5.3c)`，但当前内容和 `feature_unlabeled_results.csv` 不完全一致：

- 当前脚本 `MODELS` 只启用了 `gemini-3.1-pro-preview`，其他模型都注释掉了。
- 当前脚本 `ALL_FEATURES` 只启用了 3 个 primary features。
- 当前脚本输出 `stepwise_results.csv`。
- 现有结果文件叫 `feature_unlabeled_results.csv`，包含 6 个模型，并且还包含部分 `key_subgroup_1` / `key_subgroup_2` 行。

因此，这个脚本是某个后期/中间状态，不是生成现有 CSV 的完全快照。

### 5.2 旧 John 脚本和当前 repo runner 的 P1/P2 口径不同

John 旧 sample 脚本：

- P1 不是 `YES` 时，不跑 P2。
- 但输出里把 `p2_parsed` 写成 `INCONCLUSIVE`。
- `feature_correct=True` 用于“non-pruning”，所以 inconclusive 也可能是 True；这不能当作 accuracy。

当前整理版 runner：

- P1 不是 `YES` 时，把 `p2_parsed` 写成 `NOT_APPLICABLE`。
- `scripts/stepwise_metrics.py` 会把 `NOT_APPLICABLE` 和真正的 P2 `INCONCLUSIVE` 分开。

所以如果要复现论文 Table 5.3，需要把当前结果里的 `NOT_APPLICABLE` 和 `INCONCLUSIVE` 合并成论文里的 `Inc`。

### 5.3 当前 full six-model notebook artifact 和论文 sample 数字不同

`trials/artifacts/notebook-20260708-034723.csv` 是当前 repo 的 full six-model primary run。按论文式口径粗算：

| Model | n | Sees | MC Correct | Inc folded | Wrong | Committed Acc |
|---|---:|---:|---:|---:|---:|---:|
| gpt-4o-mini | 270 | 67.4% | 22.6% | 65.2% | 12.2% | 64.9% |
| gpt-5-mini | 270 | 27.4% | 17.0% | 78.9% | 4.1% | 80.7% |
| gemini-3.1-pro | 270 | 83.0% | 54.8% | 35.6% | 9.6% | 85.1% |
| gemini-3-flash | 270 | 92.6% | 70.7% | 15.2% | 14.1% | 83.4% |
| claude-sonnet-4-6 | 270 | 83.7% | 60.4% | 18.5% | 21.1% | 74.1% |
| claude-haiku-4-5 | 270 | 91.1% | 40.0% | 21.9% | 38.1% | 51.2% |

最大的异常是 `gpt-5-mini`：论文/JM CSV 里 primary overall `Sees Feature` 是 80.7%，当前 full run 是 27.4%。这不是普通随机波动，优先查模型别名、OpenRouter 路由、prompt set、以及 P1 parser/response style。

### 5.4 Calibration reproduction artifact 也不是论文原始口径

`trials/artifacts/calibration-reproduction-john-reference-no-illustrations.summary.csv` 的名字说明它是 `no-illustrations` 条件；论文 Table 5.2 / John `run_calibration.py` 的 blind MC 是带 reference photo 和 botanical illustration 的。

例如这个 artifact 中：

- `google/gemini-3-flash-preview` blind MC = 15/17，而论文/JM CSV 是 17/17。
- `openai/gpt-5-mini` existence = 6/17，而论文/JM CSV 是 16/17。

所以它更适合作为 ablation，而不是论文结果复现。

## 6. 排查 checklist

如果你的复跑数字和论文对不上，按这个顺序查：

1. 确认数据文件：是否用的是 90 行 `sample.csv` 和 17 个 primary references；`JM_Assets_2/data/sample.csv` 与 `newcomb_wildflower_guide/experiment_repro/output/sample.csv` 当前一致。
2. 确认 feature 范围：论文 Table 5.3 只算 `key_flower_type`, `key_plant_type`, `key_leaf_type`，不算 subgroup。
3. 确认分母：Gemini Pro 原始 sample CSV 没跑满 270 primary rows；不要强行用 90/feature 去复算论文中它的百分比。
4. 确认 Inc 口径：论文 Table 5.3 把 P1 gate fail 后的 skipped P2 也折进 `Inc`；当前 repo 会把它们标成 `NOT_APPLICABLE`。
5. 不要用 `feature_correct` 直接算 accuracy：John 旧脚本里 inconclusive 也会被标成 `feature_correct=True`，表示 non-pruning，不表示答对。
6. 确认 reference material：论文 calibration blind MC 带 reference photo + illustration；`no-reference` 或 `no-illustrations` ablation 不能和 Table 5.2 直接比。
7. 确认模型 id / provider 路由：`gpt-5-mini`, `gemini-3.1-pro-preview` 等模型名在论文、John 脚本、OpenRouter alias、当前 runner 之间可能不是同一个实际后端。
8. 确认 prompt set：当前 repo 支持 `stepwise-v1`, `stepwise-v1-concise`, `stepwise-v1-strict`；论文/JM CSV 对应的是旧 prompt 近似 `stepwise-v1`。
9. Direct baseline 单独处理：当前 repo direct artifact 与论文 Table 5.1 差异巨大，不能作为论文 direct 原始结果。

