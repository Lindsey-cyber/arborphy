# arborphy

这个仓库现在只有一个主入口：`scripts/download_inaturalist_dataset.py`。

它做一件事：从 iNaturalist 自动做一个图片数据集。

流程很简单：

1. 先问 iNaturalist：哪些物种的 research-grade、有授权照片的观察记录最多。
2. 自动选前面的热门物种，不再手写 9 个类别。
3. 对每个物种抓 observation 元数据。
4. 从 observation 里挑有 license 的照片。
5. 把图片 URL 改成指定尺寸，默认 `original`。
6. 下载图片，并写出给 KG/LLM 用的 manifest。

## 最常用命令

默认是植物，因为项目现在看起来是 botany / arborphy 方向：

```bash
python3 scripts/download_inaturalist_dataset.py
```

默认目标：

- 150 个热门植物物种
- 每个物种 100 张图片
- research-grade observations
- 有照片 license 的图片
- 图片尺寸请求 `original`
- 输出到 `data/inaturalist_150x100/`

生成结果：

```text
data/inaturalist_150x100/
  run_config.json
  species.jsonl
  manifest.jsonl
  raw_observations.jsonl
  images/
```

## 只先看看会选哪些物种

这个命令只生成物种列表，不下载图片：

```bash
python3 scripts/download_inaturalist_dataset.py --dry-run
```

## 调整图片数量

比如先做一个小测试：10 个物种，每个 20 张：

```bash
python3 scripts/download_inaturalist_dataset.py \
  --target-species 10 \
  --images-per-species 20 \
  --out-dir data/test_10x20
```

正式目标 150 x 100：

```bash
python3 scripts/download_inaturalist_dataset.py \
  --target-species 150 \
  --images-per-species 100 \
  --out-dir data/inaturalist_150x100
```

## 图片清晰度

默认：

```bash
--image-size original
```

iNaturalist 的 `original` 是最高尺寸变体，通常最大边 2048px。你也可以改成：

```bash
--image-size large
```

`large` 通常最大边 1024px，下载更快，也更不容易撞到媒体下载限制。

官方 API 文档说明：iNaturalist 图片 URL 可以只替换尺寸词来取其他尺寸，例如把 `medium.jpg` 改成 `original.jpg`。可用尺寸包括 `original`、`large`、`medium`、`small`、`thumb`、`square`。

## 不只抓植物

默认只抓 `Plantae`：

```bash
--iconic-taxa Plantae
```

改成鸟类：

```bash
python3 scripts/download_inaturalist_dataset.py --iconic-taxa Aves
```

改成昆虫：

```bash
python3 scripts/download_inaturalist_dataset.py --iconic-taxa Insecta
```

完全不限制大类：

```bash
python3 scripts/download_inaturalist_dataset.py --all-iconic-taxa
```

## 输出文件是做什么的

`species.jsonl`：一个物种一行。保存被选中的热门物种、taxon_id、学名、常用名、iNaturalist observation 数量和原始 taxon 信息。

`manifest.jsonl`：一个图片一行。KG/LLM 最常用这个文件。里面有图片路径、图片 URL、license、observation_id、observer、observed_on、taxon_id、taxon_name、ancestor_ids 等字段。`taxon_id` / `taxon_name` 是这张图归属的目标物种；如果 iNaturalist 的 observation 识别到了亚种或变种，会另外保存在 `observation_taxon_id` / `observation_taxon_name` / `observation_taxon_rank`。

`raw_observations.jsonl`：保存选中图片对应的原始 iNaturalist observation。这个文件字段多，但不是冗余；后面做 knowledge graph、溯源、重新抽字段时会有用。

`run_config.json`：这次运行用了什么参数，方便以后复现。

## 下载速度

iNaturalist 官方建议 API 请求保持在 60 requests/minute 或更低，并且媒体下载超过 5GB/hour 或 24GB/day 可能被封。默认参数已经加了 `--api-sleep 1.0` 和 `--image-sleep 0.1`。如果跑 150 x 100 的 `original` 图片，建议分批跑或者保持默认速度。

## taxon-name 是什么

旧脚本里的：

```bash
--taxon-name Trillium
```

意思是只搜索名字匹配 `Trillium` 的分类群，所以当然会限制抓取范围。现在新脚本不需要这样做；它默认会自动找热门物种。

## 旧的 pilot 是什么

旧流程里的“候选集 -> 平衡 pilot 集”只是为了做一个很小的人工测试集：先生成很多候选图片，再从写死的 9 个植物组里每组抽几张。

现在目标是 150 个物种 x 每个 100 张，所以不再需要那个 pilot 逻辑。
