# Prompt A: Direct Species Baseline

Prompt version: `prompt_A_direct_species_baseline_v0.1`

Purpose: unstructured baseline for species identification. This is not the product goal and must not be used as ground truth.

## Template

You are analyzing a natural-history image.

Task:

Identify the species shown in the image if possible.

Rules:

- If species-level identification is not visually supported, say so.
- Do not invent details that are not visible.
- Include a brief note about uncertainty.
- Do not make feature claims unless they are visible in the image.

Output format:

```json
{
  "species_guess": "",
  "taxonomic_level": "species|genus|family|unknown",
  "confidence": 0.0,
  "visual_evidence": "",
  "uncertainty_note": ""
}
```

