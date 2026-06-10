# Prompt C: Structured Feature Classification

Prompt version: `prompt_C_structured_feature_classification_v0.1`

Purpose: classify a visible feature using only admissible schema values.

## Template

You are analyzing a natural-history image.

Target feature family:

`{{feature_family}}`

Admissible values:

`{{admissible_values}}`

Task:

The target feature family has already been judged visible. Select the single best feature value from the admissible values.

Rules:

- Use only visible evidence in the image.
- Do not infer from species identity alone.
- Select exactly one admissible value.
- If the visible evidence does not fit the schema, select `unknown_visible` if available.
- Include brief visual evidence.
- Include an uncertainty note.
- Confidence must be a number from 0.0 to 1.0.

Output format:

```json
{
  "feature_family": "{{feature_family}}",
  "feature_value": "",
  "confidence": 0.0,
  "visual_evidence": "",
  "uncertainty_note": ""
}
```

