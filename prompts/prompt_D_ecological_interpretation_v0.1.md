# Prompt D: Ecological Interpretation

Prompt version: `prompt_D_ecological_interpretation_v0.1`

Purpose: connect confirmed feature-level observations to possible ecological meaning while preserving uncertainty.

## Template

You are interpreting confirmed feature-level observations from a natural-history image.

Confirmed features:

`{{confirmed_features}}`

Reference notes:

`{{reference_notes}}`

Task:

Explain possible ecological meaning of these confirmed visible features.

Rules:

- Clearly separate direct observation from interpretation.
- Mark speculative claims as speculative.
- Do not claim an ecological relationship is true unless the evidence supports it.
- "No record" does not mean a relationship is false.
- Include follow-up observations that would reduce uncertainty.

Output format:

```json
{
  "observed_features": [],
  "possible_ecological_meaning": "",
  "evidence_level": "direct_image_evidence|reference_supported|speculative|unknown",
  "uncertainty": "",
  "required_follow_up_observations": []
}
```

