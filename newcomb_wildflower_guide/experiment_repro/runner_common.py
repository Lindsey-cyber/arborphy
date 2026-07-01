from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
PROMPT_SET_DIR = ROOT / "prompt_sets"
DEFAULT_PROMPT_SET_ID = "stepwise-v1"
DEFAULT_IMAGE_SET = "sample.csv"
REQUIRED_SAMPLE_COLUMNS = [
    "newcomb_species_name",
    "species_inat",
    "taxon_id",
    "observation_id",
    "photo_id",
    "photo_url",
]

FEATURE_DISPLAY = {
    "key_flower_type": "flower type",
    "key_plant_type": "plant type",
    "key_leaf_type": "leaf type",
    "key_subgroup_1": "subsidiary feature (subgroup 1)",
    "key_subgroup_2": "subsidiary feature (subgroup 2)",
    "key_subgroup_3": "subsidiary feature (subgroup 3)",
}

PRIMARY_FEATURES = ["key_flower_type", "key_plant_type", "key_leaf_type"]


def available_prompt_sets() -> list[str]:
    if not PROMPT_SET_DIR.exists():
        return []
    return sorted(path.stem for path in PROMPT_SET_DIR.glob("*.json"))


def load_prompt_set(prompt_set_id: str | None = None) -> dict[str, str]:
    prompt_set_id = (prompt_set_id or DEFAULT_PROMPT_SET_ID).strip()
    if prompt_set_id.endswith(".json"):
        prompt_set_id = prompt_set_id.removesuffix(".json")
    if not prompt_set_id or Path(prompt_set_id).name != prompt_set_id:
        raise ValueError(f"Invalid prompt set id: {prompt_set_id!r}")
    prompt_path = PROMPT_SET_DIR / f"{prompt_set_id}.json"
    if not prompt_path.exists():
        available = ", ".join(available_prompt_sets()) or "(none)"
        raise ValueError(f"Unknown prompt set '{prompt_set_id}'. Available prompt sets: {available}")

    data = json.loads(prompt_path.read_text())
    if data.get("id", prompt_set_id) != prompt_set_id:
        raise ValueError(f"Prompt set id mismatch in {prompt_path}: expected '{prompt_set_id}'")

    required = [
        "id",
        "p1_visibility",
        "p2_intro",
        "p2_option_line",
        "p2_reference_photo_label",
        "p2_reference_illustration_label",
        "p2_uncertain_option",
        "p2_response_instruction",
        "leaf_absence_note",
    ]
    missing = [key for key in required if not isinstance(data.get(key), str)]
    if missing:
        raise ValueError(f"Prompt set '{prompt_set_id}' is missing string fields: {', '.join(missing)}")
    return data


def resolve_image_set_path(image_set: str | None = None) -> Path:
    image_set = (image_set or DEFAULT_IMAGE_SET).strip()
    image_set_path = Path(image_set)
    if not image_set or image_set_path.name != image_set:
        raise ValueError(f"Invalid image_set {image_set!r}. Use a CSV filename from {OUTPUT_DIR}.")
    if image_set_path.suffix.lower() != ".csv":
        raise ValueError(f"Invalid image_set {image_set!r}. image_set must be a CSV filename.")

    sample_path = OUTPUT_DIR / image_set
    if not sample_path.exists():
        raise FileNotFoundError(f"image_set CSV does not exist: {sample_path}")
    return sample_path


def validate_sample_columns(sample: pd.DataFrame, sample_path: Path) -> None:
    missing = [column for column in REQUIRED_SAMPLE_COLUMNS if column not in sample.columns]
    if missing:
        raise ValueError(
            f"image_set CSV {sample_path} is missing required columns: {', '.join(missing)}"
        )


def load_inputs(image_set: str | None = None) -> dict[str, object]:
    refs = pd.read_csv(OUTPUT_DIR / "references.csv")
    fv = pd.read_csv(OUTPUT_DIR / "feature_value_pairs.csv")
    kg = pd.read_csv(OUTPUT_DIR / "newcomb_preprocessed.csv")
    sample_path = resolve_image_set_path(image_set)
    sample = pd.read_csv(sample_path)
    validate_sample_columns(sample, sample_path)
    illustration_paths = json.loads((OUTPUT_DIR / "illustration_paths.json").read_text())

    values_by_feature = {
        feat: fv.loc[fv["feature"] == feat, "value"].dropna().tolist()
        for feat in fv["feature"].dropna().unique()
    }

    ref_mat = {}
    for _, row in refs.iterrows():
        feat = row["feature"]
        val = row["feature_value"]
        illust_path = row.get("reference_illustration_path", "")
        if not isinstance(illust_path, str):
            illust_path = ""
        ref_mat[(feat, val)] = {
            "img_path": None,
            "illust_path": illust_path if illust_path and Path(illust_path).exists() else None,
            "description": row.get("reference_description", "") or "",
        }

    return {
        "refs": refs,
        "fv": fv,
        "kg": kg,
        "sample": sample,
        "sample_path": sample_path,
        "illustration_paths": illustration_paths,
        "values_by_feature": values_by_feature,
        "ref_mat": ref_mat,
    }


def get_true_path(path_table: pd.DataFrame, species_inat: str) -> dict[str, str]:
    matches = path_table.loc[path_table["species_inat"] == species_inat]
    if matches.empty:
        return {}
    row = matches.iloc[0]
    return {
        feature: str(row[feature]).strip()
        for feature in FEATURE_DISPLAY
        if feature in row.index and isinstance(row[feature], str) and row[feature].strip()
    }


def existence_parts(
    feature_col: str,
    image_path_or_url: str,
    true_value: str = "",
    prompt_set: dict[str, str] | None = None,
) -> list:
    prompt_set = prompt_set or load_prompt_set()
    fname = FEATURE_DISPLAY.get(feature_col, feature_col)
    extra = ""
    if feature_col == "key_leaf_type" and "no apparent" in true_value.lower():
        extra = prompt_set["leaf_absence_note"]
    return [
        {"image": image_path_or_url},
        prompt_set["p1_visibility"].format(feature_display=fname, leaf_absence_note=extra),
    ]


def agreement_parts(feature_col: str, true_value: str, description: str, image_path_or_url: str) -> list:
    fname = FEATURE_DISPLAY.get(feature_col, feature_col)
    desc_clause = f" — {description}" if description else ""
    return [
        {"image": image_path_or_url},
        (
            f"A botanical expert has classified the '{fname}' of the plant in this image "
            f"as '{true_value}'{desc_clause}.\n"
            f"Is this classification consistent with what you can observe in the image?\n"
            f"Reply with exactly one word: YES, NO, or INCONCLUSIVE."
        ),
    ]


def blind_mc_parts(
    feature_col: str,
    options: list[dict],
    image_path_or_url: str,
    prompt_set: dict[str, str] | None = None,
) -> list:
    prompt_set = prompt_set or load_prompt_set()
    fname = FEATURE_DISPLAY.get(feature_col, feature_col)
    parts = [
        {"image": image_path_or_url},
        prompt_set["p2_intro"].format(feature_display=fname),
    ]
    for i, opt in enumerate(options, 1):
        desc = f" — {opt['description']}" if opt["description"] else ""
        parts.append(
            prompt_set["p2_option_line"].format(
                index=i,
                value=opt["value"],
                description_clause=desc,
            )
        )
        if opt.get("img_path"):
            parts.append(prompt_set["p2_reference_photo_label"].format(index=i))
            parts.append({"image": opt["img_path"]})
        if opt.get("illust_path"):
            parts.append(prompt_set["p2_reference_illustration_label"].format(index=i))
            parts.append({"image": opt["illust_path"]})
    parts.append(prompt_set["p2_uncertain_option"].format(index=len(options) + 1))
    example_value = options[0]["value"] if options else "one listed option"
    parts.append(
        prompt_set["p2_response_instruction"].format(
            example_value=example_value,
            uncertain_option="Cannot determine from this image",
        )
    )
    return parts


def parse_ync(response: str) -> str:
    r = response.upper()
    if r.startswith("YES"):
        return "YES"
    if r.startswith("NO"):
        return "NO"
    if "INC" in r:
        return "INCONCLUSIVE"
    for kw in ("YES", "NO", "INCONCLUSIVE"):
        if kw in r:
            return kw
    return "INCONCLUSIVE"


def parse_mc(response: str, values: list[str]) -> str:
    r = response.strip()
    for value in values:
        if value.lower() == r.lower():
            return value
    for value in values:
        if value.lower() in r.lower():
            return value
    if "cannot determine" in r.lower():
        return "INCONCLUSIVE"
    match = re.match(r"^(\d+)", r)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(values):
            return values[idx]
        if idx == len(values):
            return "INCONCLUSIVE"
    return "INCONCLUSIVE"


def build_options(refs: pd.DataFrame, ref_mat: dict, feature_col: str) -> list[dict]:
    options = []
    for _, row in refs.loc[refs["feature"] == feature_col].iterrows():
        mat = ref_mat.get((feature_col, row["feature_value"]), {})
        options.append(
            {
                "value": row["feature_value"],
                "img_path": mat.get("img_path"),
                "illust_path": mat.get("illust_path"),
                "description": mat.get("description", ""),
            }
        )
    return options
