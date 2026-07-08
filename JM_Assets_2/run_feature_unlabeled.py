"""Stepwise Feature Classification experiment (§5.3c) — primary baseline.

For each test image and each feature in the species' Newcomb path:
  P1  Existence check — fresh session, test image only
        "Is [feature] visible?" → YES / NO / INCONCLUSIVE
  P2  Blind MC        — fresh session, test image + reference material per option
        "Which value?" (only asked if P1 = YES)
        Each option shown with its reference photo, botanical illustration
        (primary features only), and textual description — interleaved.

One fresh model call per (image × feature × prompt), no cross-feature context.

Output: one row per (model, observation_id, feature).
"""

import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models import call_model

# ── Paths ─────────────────────────────────────────────────────────────────────
SAMPLE_FILE        = "/Users/John/Newcomb/implementation/2_baseline/sample.csv"
KG_FILE            = "/Users/John/Newcomb/newcomb_preprocessed.csv"
FEATURE_VALUES_CSV = "/Users/John/Newcomb/feature_value_pairs.csv"
REFERENCES_CSV     = "/Users/John/Newcomb/references/references.csv"
IMAGE_LINKS_JSON   = "/Users/John/Newcomb/references/image_links.json"
REF_IMAGES_DIR     = "/Users/John/Newcomb/final_thesis/reference_images"
ILLUSTRATION_PATHS = "/Users/John/Newcomb/references/illustration_paths.json"
OUT_FILE           = os.path.join(os.path.dirname(__file__), "stepwise_results.csv")

MODELS = [
    # "gpt-4o-mini",
    # "gpt-5-mini",
    "gemini-3.1-pro-preview",
    # "gemini-3-flash-preview",
    # "claude-sonnet-4-6",
    # "claude-haiku-4-5",
]

ALL_FEATURES = [
    "key_flower_type", "key_plant_type", "key_leaf_type",
    # "key_subgroup_1",  "key_subgroup_2", "key_subgroup_3",
]
PRIMARY = set(ALL_FEATURES[:3])

FEATURE_DISPLAY = {
    "key_flower_type": "flower type",
    "key_plant_type":  "plant type",
    "key_leaf_type":   "leaf type",
    "key_subgroup_1":  "subsidiary feature (subgroup 1)",
    "key_subgroup_2":  "subsidiary feature (subgroup 2)",
    "key_subgroup_3":  "subsidiary feature (subgroup 3)",
}

# ── Load data ─────────────────────────────────────────────────────────────────
sample      = pd.read_csv(SAMPLE_FILE)
kg          = pd.read_csv(KG_FILE)
fv          = pd.read_csv(FEATURE_VALUES_CSV)
refs        = pd.read_csv(REFERENCES_CSV)
illust      = json.load(open(ILLUSTRATION_PATHS))
image_links = json.load(open(IMAGE_LINKS_JSON))

kg_indexed = kg.set_index("species_inat")

# Build photo_id → local filepath mapping from image_links.json
def _find_local(species_name: str, photo_id: str) -> str | None:
    fname_base = species_name.lower().replace(" ", "_")
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(REF_IMAGES_DIR, f"{fname_base}{ext}")
        if os.path.exists(p):
            return p
    for ext in (".jpg", ".jpeg", ".png"):
        p = os.path.join(REF_IMAGES_DIR, f"{photo_id}{ext}")
        if os.path.exists(p):
            return p
    return None

PHOTO_MAP: dict[str, str] = {}
for fv_key, species_dict in image_links.items():
    for species_name, url in species_dict.items():
        if not url:
            continue
        pid = url.rstrip("/").split("/")[-1]
        local = _find_local(species_name, pid)
        if local:
            PHOTO_MAP[pid] = local

# All admissible values per feature level
values_by_feature: dict[str, list[str]] = {
    feat: sorted(fv[fv["feature"] == feat]["value"].tolist())
    for feat in ALL_FEATURES
}

# Reference material per (feature, value)
ref_mat: dict[tuple, dict] = {}
for _, r in refs.iterrows():
    feat = r["feature"]
    val  = r["feature_value"]
    link = r.get("reference_image_link", "") or ""
    pid  = link.rstrip("/").split("/")[-1] if link else None
    img  = PHOTO_MAP.get(pid) if pid else None
    ik   = f"{feat}:{val}"
    ip   = illust.get(ik)
    ref_mat[(feat, val)] = {
        "img_path":    img,
        "illust_path": ip if (ip and os.path.exists(str(ip))) else None,
        "description": r.get("reference_description", "") or "",
    }


def get_true_path(species_inat: str) -> dict[str, str]:
    if species_inat not in kg_indexed.index:
        return {}
    row = kg_indexed.loc[species_inat]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return {f: str(row[f]).strip() for f in ALL_FEATURES
            if f in row.index and isinstance(row[f], str) and row[f].strip()}


# ── Prompt builders ───────────────────────────────────────────────────────────

def p1_parts(feature_col: str, photo_url: str) -> list:
    fname = FEATURE_DISPLAY.get(feature_col, feature_col)
    return [
        {"image": photo_url},
        (f"Your task is to identify if the visual feature '{fname}' is discernible "
         f"in the following image. Answer only: YES, NO, or INCONCLUSIVE. "
         f"Answer YES if and only if the feature is clearly discernible in the image."),
    ]


def p2_parts(feature_col: str, values: list[str], photo_url: str) -> list:
    """Test image first, then each option interleaved with its reference material."""
    fname = FEATURE_DISPLAY.get(feature_col, feature_col)
    parts = [
        {"image": photo_url},
        (f"Which of the following options best describes the '{fname}' of the plant "
         f"in this image? If the feature is not clearly visible, choose "
         f"'Cannot determine from this image'.\n\nOptions:"),
    ]
    for i, val in enumerate(values, 1):
        mat  = ref_mat.get((feature_col, val), {})
        desc = mat.get("description", "")
        parts.append(f"\n{i}. **{val}**" + (f" — {desc}" if desc else ""))
        if mat.get("img_path"):
            parts.append(f"Reference photo for option {i}:")
            parts.append({"image": mat["img_path"]})
        if mat.get("illust_path"):
            parts.append(f"Botanical illustration for option {i}:")
            parts.append({"image": mat["illust_path"]})
    parts.append(f"\n{len(values) + 1}. **Cannot determine from this image**")
    parts.append(
        f"\nRespond with ONLY the exact option text (e.g., \"{values[0]}\") "
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
    return "INCONCLUSIVE"


def parse_mc(response: str, values: list[str]) -> str:
    r = response.strip()
    for v in values:
        if v.lower() == r.lower():      return v
    for v in values:
        if v.lower() in r.lower():      return v
    if "cannot determine" in r.lower(): return "INCONCLUSIVE"
    m = re.match(r"^(\d+)", r)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(values):     return values[idx]
        if idx == len(values):         return "INCONCLUSIVE"
    return "INCONCLUSIVE"


# ── Resume ────────────────────────────────────────────────────────────────────
if os.path.exists(OUT_FILE):
    existing = pd.read_csv(OUT_FILE)
    done = set(zip(
        existing["model"],
        existing["observation_id"].astype(str),
        existing["feature"],
    ))
    print(f"Resuming: {len(done)} (model, obs, feature) triples done.")
else:
    done = set()

# ── Main loop ─────────────────────────────────────────────────────────────────
_csv_lock  = threading.Lock()
_done_lock = threading.Lock()

NUM_WORKERS = 3


def process_task(model, row, feature_col):
    obs_id    = str(row["observation_id"])
    photo_url = row["photo_url"]
    species   = row["species_inat"]
    true_path = get_true_path(species)

    with _done_lock:
        if (model, obs_id, feature_col) in done:
            return

    true_value = true_path.get(feature_col)
    if true_value is None:
        return   # not in this species' Newcomb path

    values = values_by_feature.get(feature_col, [])

    # ── P1: existence check (test image only) ────────────────────────
    p1_raw    = call_model(model, p1_parts(feature_col, photo_url))
    p1_parsed = parse_ync(p1_raw)

    # ── P2: blind MC (test image + reference material per option) ────
    if p1_parsed == "YES":
        p2_raw    = call_model(model, p2_parts(feature_col, values, photo_url))
        p2_parsed = parse_mc(p2_raw, values)
    else:
        p2_raw    = ""
        p2_parsed = "INCONCLUSIVE"

    # ── Correctness ──────────────────────────────────────────────────
    if p2_parsed == true_value:
        feature_correct, committed = True,  True
    elif p2_parsed == "INCONCLUSIVE":
        feature_correct, committed = True,  False   # non-pruning
    else:
        feature_correct, committed = False, True    # wrong → path fails

    with _csv_lock:
        pd.DataFrame([{
            "model":                model,
            "observation_id":       obs_id,
            "photo_id":             row["photo_id"],
            "taxon_id":             row["taxon_id"],
            "species_inat":         species,
            "newcomb_species_name": row["newcomb_species_name"],
            "feature":              feature_col,
            "true_value":           true_value,
            "p1_raw":               p1_raw,
            "p1_parsed":            p1_parsed,
            "p2_raw":               p2_raw,
            "p2_parsed":            p2_parsed,
            "feature_correct":      feature_correct,
            "committed":            committed,
        }]).to_csv(OUT_FILE, mode="a", header=not os.path.exists(OUT_FILE), index=False)
        done.add((model, obs_id, feature_col))

    status = "✓" if feature_correct else "✗"
    print(f"  obs={obs_id} {feature_col} P1={p1_parsed} P2={p2_parsed} [{status}]")


for model in MODELS:
    print(f"\n=== {model} ===")
    tasks = [
        (model, row, feature_col)
        for _, row in sample.iterrows()
        for feature_col in ALL_FEATURES
    ]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_task, m, r, f) for m, r, f in tasks]
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                print(f"  [ERROR] {exc}")

print(f"\nDone. Raw results → {OUT_FILE}")
print("Run analyze.py to compute metrics.")
