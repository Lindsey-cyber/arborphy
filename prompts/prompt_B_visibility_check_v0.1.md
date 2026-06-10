# Prompt B: Visibility Check

Prompt version: `prompt_B_visibility_check_v0.1`

Purpose: decide whether the image provides enough visible evidence for a target feature family.

## Template

You are analyzing a natural-history image.

Target feature family:

`{{feature_family}}`

Task:

Decide whether the target feature family is sufficiently visible in the image for structured classification.

Rules:

- Use only visible evidence in the image.
- Do not infer visibility from a species name or reference knowledge.
- If the relevant plant part is absent, hidden, too small, too blurry, out of frame, or belongs to an unclear background organism, answer `INCONCLUSIVE` or `NO`.
- "Not visible" does not mean the organism lacks the feature.
- Output exactly one label and no other text.

Allowed output:

`YES`

`INCONCLUSIVE`

`NO`

