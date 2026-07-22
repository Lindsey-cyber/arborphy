# Project Update for Tian — Concise English Version

## 0. Summary and current limitations

The current pipeline first localizes the flower, leaf, or plant; extracts local shape evidence; matches Newcomb reference images; and finally uses the knowledge graph to filter candidate taxa. The pipeline, human benchmark, metrics, and step-by-step visualizations are operational.

Two limitations should be stated at the beginning of the presentation:

1. **Organ localization still produces false positives.** In some photos where no judgeable flower is present, Grounding DINO labels leaves, green plant structures, or background regions as flowers. It can also label a flower as a leaf.
2. **The reference photos are not yet sufficiently representative.** Each primary feature value currently has roughly one reference photo, and some references contain distracting backgrounds, multiple organs, or weak examples of the target feature. The calibration query can also be identical to the correct option reference, so the current reference set is not yet a strong formal benchmark.

The pilot contains only 10 photos. It is suitable for pipeline validation and error diagnosis, not a final model-performance claim.

## 1. Research objective

The current task is separated into:

- P1: is the requested feature visible and judgeable in the photo?
- P2: when a human can judge it, which Newcomb feature value describes the localized region?
- Graph: use predicted flower, plant, and leaf values to filter 387 Newcomb taxa.

## 2. Current data

| Data | Current size | Purpose |
| --- | ---: | --- |
| Newcomb references | 17 primary-feature references | calibration and image matching |
| iNaturalist `sample.csv` | 90 photos, 30 species | full-photo candidate pool |
| Current formal pilot | 10 photos; 30 P1 items; 17 evaluable P2 items | current local-pipeline evaluation |
| Newcomb graph | 387 taxa | deterministic candidate filtering |

The revised 10-photo pilot is fully reviewed: 30 approved P1 items, 17 assignable P2 items, and zero label conflicts.

## 3. Model review

| Priority | Hugging Face model | Size | Role in the project |
| --- | --- | ---: | --- |
| 1 | [Grounding DINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | approximately 0.2B | zero-shot localization of flowers, leaves, stems, and vines; current primary detector |
| 1 | [DINOv2 Small](https://huggingface.co/facebook/dinov2-small) | 22.1M | visual features from localized flower/leaf crops; suitable for frozen embeddings and kNN/prototypes |
| 1 | [SAM 2.1 Hiera Tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) | 39M | flower/leaf contours for margins, division, and symmetry; not a classifier |
| 2 | [BioCLIP](https://huggingface.co/imageomics/bioclip) | ViT-B/16 | biology-domain embedding and zero-shot control; closer to biological photos than generic CLIP |
| 2 | [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) | ViT-L/14 | medium-scale domain-model ceiling; stronger biological representations but more expensive locally |
| 2 | [OWLv2 Base](https://huggingface.co/google/owlv2-base-patch16-ensemble) | approximately 0.2B | detector control for separating localization failures from classification failures |
| 3 | [SigLIP 2 Base](https://huggingface.co/google/siglip2-base-patch16-224) | 0.4B | general zero-shot semantic ceiling; not the preferred lightweight route |
| 3 | [Florence-2 Base](https://huggingface.co/microsoft/Florence-2-base) | 0.23B | detection, region captioning, and grounding; useful as a diagnostic model |

## 4. Current local pipeline and API use

`field photo → Grounding DINO Tiny organ box → localized crop → SAM2.1 Tiny mask/edge/shape measurements → DINOv2-small reference matching → Newcomb graph filtering`

- Grounding DINO localizes flower, leaf, or whole-plant context.
- SAM2 stores contours, area, circularity, solidity, aspect ratio, color, and related explicit evidence.
- DINOv2 ranks localized crops against curated reference images using cosine similarity.
- The graph applies exact Newcomb-value filters without an LLM.

The **17-row local calibration run and the revised 10-photo local run reported here did not call the OpenRouter API**. The 10-photo metadata records `mode=local-parts` and OpenRouter `captured_calls=0`; all 60 P1/P2 model calls ran locally. Earlier six-hosted-model ablations and the GPT baseline did use OpenRouter, but those are historical comparisons rather than the current local-pipeline results.

## 5. How graph filtering works and why there is no final single-species prediction yet

The Newcomb graph is a table of 387 taxa. Each taxon has corresponding `key_flower_type`, `key_plant_type`, and `key_leaf_type` values.

For each photo, the program collects concrete P2 predictions and applies a strict set intersection:

```text
candidates = all 387 taxa
candidates = candidates where key_flower_type == predicted flower value
candidates = candidates where key_plant_type == predicted plant value
candidates = candidates where key_leaf_type == predicted leaf value
```

- An `INCONCLUSIVE` feature or a feature without a concrete value is not used for filtering.
- The graph does not calculate probabilities or call a model; it performs exact-match filtering only.
- Mutually inconsistent predictions can reduce the candidate set to zero.
- `original species retained` means that the labeled species remains somewhere in the filtered set; it is not Top-1 accuracy.

The pipeline currently does not output one final species **not because this requires an online API**, but because the experiment intentionally separates visual-feature accuracy from graph candidate reduction. If the product only needs a small Newcomb candidate list, the pipeline can stop after graph filtering with no online model. If a single species is required, a local image-to-candidate ranker can rank the remaining taxa. An online VLM is an optional baseline, not a necessary step.

## 6. Core results

### Calibration reference set

| Metric | Result |
| --- | ---: |
| P1 feature visibility | 17/17 = 100% |
| P2 feature-value match | 16/17 = 94.1% |

This result verifies pipeline wiring and vocabulary alignment only. The calibration query and the correct-option reference photo are identical, so DINOv2 may perform identity-image matching. The 94.1% result should not be interpreted as unseen-photo generalization.

The earlier six-hosted-model ablation showed the same concern: removing option reference photos reduced blind-MC accuracy from 83.3% to 56.9%. Reference images are important, but fair evaluation requires non-identical held-out references.

### Revised 10-photo pilot

Formal P2 rates use the 17 items with `human_visible=YES` and `human_can_assign_value=YES` as the denominator:

| Metric | Result | Meaning |
| --- | ---: | --- |
| P1 visibility accuracy | 24/30 = 80.0% | all six errors are false-visible; the model returns YES for all 30 items |
| P1 inconclusive rate | 0/30 = 0.0% | P1 never returns inconclusive |
| P2 coverage | 14/17 = 82.4% | 14 concrete predictions among 17 evaluable items |
| P2 correct rate / overall accuracy | 6/17 = 35.3% | correct predictions among all evaluable P2 items |
| P2 wrong rate | 8/17 = 47.1% | concrete but incorrect predictions among all evaluable P2 items |
| P2 inconclusive rate | 3/17 = 17.6% | P2 ran but returned no concrete value |
| P2 selective accuracy | 6/14 = 42.9% | accuracy among the 14 concrete predictions only |
| Median graph candidates | 387 → 19.5 | approximately 95.0% median candidate reduction |
| Labeled species retained | 2/10 | the original species remains in the model candidate set for two photos; this is not Top-1 accuracy |

The decomposition is complete: `6 correct + 8 wrong + 3 inconclusive = 17` formal P2 items.

## 7. Current conclusions

1. Graph filtering sharply reduces the candidate set, but it also strictly propagates upstream errors: one incorrect feature can eliminate the correct species.
2. The main bottleneck is organ localization and visual feature extraction. Grounding DINO can label a non-flower region as a flower, a flower as a leaf, or select a background organ.
3. The representativeness and independence of the reference set are insufficient, which affects both DINOv2 matching and benchmark validity.
4. SAM2 makes the errors visible and measurable, but its geometric measurements have not yet been trained into Newcomb-specific petal-count, leaf-margin, or texture classifiers.

## 8. Next steps

1. Add multiple clear, single-organ, non-identical reference photos per feature value and create held-out reference evaluation to remove calibration identity leakage.
2. Annotate focal plants, focal flower/leaf boxes, and organ ownership; split P1 into presence, focal-plant ownership, and assignability.
3. Re-run Grounding DINO, OWLv2, and other detector/prompt variants on the revised benchmark.
4. If the project requires a single species output, evaluate a local candidate reranker first; use an online VLM only as a performance comparison.

