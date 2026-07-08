"""Calibration experiment (§5.1).

Runs three prompt types on the 17 primary feature exemplars across all 6 models:
  existence  – Is [feature] visually discernible?           → YES / NO / INC
  agreement  – Is [feature] = [true_value] in this image?  → YES / NO / INC
  blind_mc   – Which value does this image show?            → selected value

Each row in the output CSV corresponds to one (model, feature_value, prompt_type) trial.
Primary features only: key_flower_type, key_plant_type, key_leaf_type.
"""

import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import call_model

# ── Paths ─────────────────────────────────────────────────────────────────────
REFERENCES_CSV     = "/Users/John/Newcomb/references/references.csv"
IMAGE_LINKS_JSON   = "/Users/John/Newcomb/references/image_links.json"
REF_IMAGES_DIR     = "/Users/John/Newcomb/final_thesis/reference_images"
ILLUSTRATION_PATHS = "/Users/John/Newcomb/references/illustration_paths.json"
OUT_FILE           = os.path.join(os.path.dirname(__file__), "calibration_results.csv")

MODELS = [
    "gpt-4o-mini",
    "gpt-5-mini",
    # "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
]

PRIMARY_FEATURES = {"key_flower_type", "key_plant_type", "key_leaf_type"}

FEATURE_DISPLAY = {
    "key_flower_type": "flower type",
    "key_plant_type":  "plant type",
    "key_leaf_type":   "leaf type",
    "key_subgroup_1":  "subsidiary feature (subgroup 1)",
    "key_subgroup_2":  "subsidiary feature (subgroup 2)",
    "key_subgroup_3":  "subsidiary feature (subgroup 3)",
}

# ── Load data ─────────────────────────────────────────────────────────────────
refs        = pd.read_csv(REFERENCES_CSV)
illust      = json.load(open(ILLUSTRATION_PATHS))
image_links = json.load(open(IMAGE_LINKS_JSON))

# Build photo_id → local filepath mapping from image_links.json
# image_links format: {"feature:value": {"Species Name": "https://.../photos/PHOTO_ID"}}
def _find_local(species_name: str, photo_id: str) -> str | None:
    fname_base = species_name.lower().replace(" ", "_")
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(REF_IMAGES_DIR, f"{fname_base}{ext}")
        if os.path.exists(p):
            return p
    # Fallback: try by photo ID
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(REF_IMAGES_DIR, f"{photo_id}{ext}")
        if os.path.exists(p):
            return p
    return None

PHOTO_MAP: dict[str, str] = {}  # photo_id → local path
for fv_key, species_dict in image_links.items():
    for species_name, url in species_dict.items():
        if not url:
            continue
        pid = url.rstrip("/").split("/")[-1]
        local = _find_local(species_name, pid)
        if local:
            PHOTO_MAP[pid] = local

# Primary-features-only reference rows (calibration is on primary exemplars only)
primary_refs = refs[
    refs["feature"].isin(PRIMARY_FEATURES) & refs["reference_image_link"].notna()
].copy()
print(f"Primary reference exemplars: {len(primary_refs)}")


def ref_img_path(link: str) -> str | None:
    if not link:
        return None
    pid = link.rstrip("/").split("/")[-1]
    return PHOTO_MAP.get(pid)


# Build lookup: (feature, value) → option dict (for blind_mc options)
def build_options(feature_col: str) -> list[dict]:
    opts = []
    for _, row in refs[refs["feature"] == feature_col].iterrows():
        link = row.get("reference_image_link", "") or ""
        pid  = link.rstrip("/").split("/")[-1] if link else None
        img  = PHOTO_MAP.get(pid) if pid else None
        ik   = f"{feature_col}:{row['feature_value']}"
        ip   = illust.get(ik)
        opts.append({
            "value":       row["feature_value"],
            "img_path":    img,
            "illust_path": ip if (ip and os.path.exists(str(ip))) else None,
            "description": row.get("reference_description", "") or "",
        })
    return opts

options_by_feature = {f: build_options(f) for f in refs["feature"].dropna().unique()}


# ── Prompt builders ───────────────────────────────────────────────────────────

def existence_parts(feature_col: str, ref_img: str, true_value: str = "") -> list:
    fname = FEATURE_DISPLAY.get(feature_col, feature_col)
    extra = ""
    if feature_col == "key_leaf_type" and "no apparent" in true_value.lower():
        extra = (" Note: the absence of visible leaves IS itself a discernible leaf-type"
                 " feature — if the stem has no apparent leaves, answer YES.")
    return [
        {"image": ref_img},
        (f"Your task is to identify if the visual feature '{fname}' is discernible "
         f"in the following image. Answer only: YES, NO, or INCONCLUSIVE. "
         f"Answer YES if and only if the feature is clearly discernible in the image.{extra}"),
    ]


def agreement_parts(feature_col: str, true_value: str, description: str, ref_img: str) -> list:
    fname = FEATURE_DISPLAY.get(feature_col, feature_col)
    desc_clause = f" — {description}" if description else ""
    return [
        {"image": ref_img},
        (f"A botanical expert has classified the '{fname}' of the plant in this image "
         f"as '{true_value}'{desc_clause}.\n"
         f"Is this classification consistent with what you can observe in the image?\n"
         f"Reply with exactly one word: YES, NO, or INCONCLUSIVE."),
    ]


def blind_mc_parts(feature_col: str, options: list[dict], ref_img: str) -> list:
    """Test image first, then each option with its reference photo and illustration."""
    fname = FEATURE_DISPLAY.get(feature_col, feature_col)
    parts = [
        {"image": ref_img},
        (f"Which of the following options best describes the '{fname}' of the plant in this image?\n"
         f"If the feature is not clearly visible, choose 'Cannot determine from this image'.\n\n"
         f"Options:"),
    ]
    for i, opt in enumerate(options, 1):
        desc = f" — {opt['description']}" if opt["description"] else ""
        parts.append(f"\n{i}. **{opt['value']}**{desc}")
        if opt["img_path"]:
            parts.append(f"Reference photo for option {i}:")
            parts.append({"image": opt["img_path"]})
        if opt["illust_path"]:
            parts.append(f"Botanical illustration for option {i}:")
            parts.append({"image": opt["illust_path"]})
    n = len(options)
    parts.append(f"\n{n + 1}. **Cannot determine from this image**")
    parts.append(
        f"\nRespond with ONLY the exact option text (e.g., \"{options[0]['value']}\") "
        f"or \"Cannot determine from this image\"."
    )
    return parts


# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_ync(response: str) -> str:
    r = response.upper()
    if r.startswith("YES"):   return "YES"
    if r.startswith("NO"):    return "NO"
    if "INC" in r:            return "INCONCLUSIVE"
    for kw in ("YES", "NO", "INCONCLUSIVE"):
        if kw in r:
            return kw
    return response[:80]


def parse_mc(response: str, options: list[dict]) -> str | None:
    r = response.strip()
    for opt in options:
        if r.lower() == opt["value"].lower():
            return opt["value"]
    for opt in options:
        if opt["value"].lower() in r.lower():
            return opt["value"]
    if "cannot determine" in r.lower():
        return "INCONCLUSIVE"
    m = re.match(r"^(\d+)", r)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(options):
            return options[idx]["value"]
        if idx == len(options):
            return "INCONCLUSIVE"
    return None


# ── Resume ────────────────────────────────────────────────────────────────────
if os.path.exists(OUT_FILE):
    existing = pd.read_csv(OUT_FILE)
    done = set(zip(existing["model"], existing["feature_value"], existing["prompt_type"]))
    print(f"Resuming: {len(done)} rows already done.")
else:
    done = set()

# ── Main loop ─────────────────────────────────────────────────────────────────
for model in MODELS:
    print(f"\n=== {model} ===")
    for _, ref_row in primary_refs.iterrows():
        feature_col = ref_row["feature"]
        true_value  = ref_row["feature_value"]
        description = ref_row.get("reference_description", "") or ""
        ref_img     = ref_img_path(ref_row.get("reference_image_link", "") or "")

        if ref_img is None:
            continue

        options = options_by_feature.get(feature_col, [])

        def _save(row: dict) -> None:
            pd.DataFrame([row]).to_csv(
                OUT_FILE, mode="a", header=not os.path.exists(OUT_FILE), index=False
            )

        # ── Existence check ──────────────────────────────────────────────────
        key = (model, true_value, "existence")
        if key not in done:
            raw    = call_model(model, existence_parts(feature_col, ref_img, true_value))
            parsed = parse_ync(raw)
            _save({"model": model, "feature": feature_col,
                   "feature_value": true_value, "prompt_type": "existence",
                   "raw_response": raw, "parsed": parsed,
                   "expected": "YES", "correct": parsed == "YES"})
            done.add(key)
            print(f"  exist  {true_value[:45]:<45} → {parsed}")

        # ── Agreement check ──────────────────────────────────────────────────
        key = (model, true_value, "agreement")
        if key not in done:
            raw    = call_model(model, agreement_parts(feature_col, true_value, description, ref_img))
            parsed = parse_ync(raw)
            _save({"model": model, "feature": feature_col,
                   "feature_value": true_value, "prompt_type": "agreement",
                   "raw_response": raw, "parsed": parsed,
                   "expected": "YES", "correct": parsed == "YES"})
            done.add(key)
            print(f"  agree  {true_value[:45]:<45} → {parsed}")

        # ── Blind multiple choice ────────────────────────────────────────────
        key = (model, true_value, "blind_mc")
        if key not in done:
            raw        = call_model(model, blind_mc_parts(feature_col, options, ref_img))
            parsed_val = parse_mc(raw, options)
            _save({"model": model, "feature": feature_col,
                   "feature_value": true_value, "prompt_type": "blind_mc",
                   "raw_response": raw, "parsed": parsed_val,
                   "expected": true_value, "correct": parsed_val == true_value})
            done.add(key)
            print(f"  blind  {true_value[:45]:<45} → {parsed_val}  ({'✓' if parsed_val == true_value else '✗'})")

print(f"\nDone. Raw results → {OUT_FILE}")
print("Run analyze.py to compute metrics.")
