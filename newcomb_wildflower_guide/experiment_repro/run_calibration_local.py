from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import threading

import pandas as pd

from model_adapter import call_model
from runner_common import (
    OUTPUT_DIR,
    PRIMARY_FEATURES,
    agreement_parts,
    blind_mc_parts,
    build_options,
    existence_parts,
    load_inputs,
    parse_mc,
    parse_ync,
)


MODELS = [m.strip() for m in os.environ.get("EXPERIMENT_MODELS", "mock-local").split(",") if m.strip()]
NUM_WORKERS = int(os.environ.get("EXPERIMENT_NUM_WORKERS", "1"))
OUT_FILE = Path(os.environ.get("EXPERIMENT_OUT_FILE", OUTPUT_DIR / "calibration_results_local.csv")).expanduser()
if not OUT_FILE.is_absolute():
    OUT_FILE = OUTPUT_DIR / OUT_FILE


def main() -> None:
    inputs = load_inputs()
    refs = inputs["refs"]
    ref_mat = inputs["ref_mat"]
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    primary_refs = refs[
        refs["feature"].isin(PRIMARY_FEATURES) & refs["reference_image_link"].fillna("").ne("")
    ].copy()
    options_by_feature = {feature: build_options(refs, ref_mat, feature) for feature in refs["feature"].dropna().unique()}

    done = set()
    if OUT_FILE.exists():
        existing = pd.read_csv(OUT_FILE)
        done = set(zip(existing["model"], existing["feature"], existing["feature_value"], existing["prompt_type"]))

    tasks = []
    for model in MODELS:
        print(f"\n=== {model} ===")
        for _, ref_row in primary_refs.iterrows():
            feature_col = ref_row["feature"]
            true_value = ref_row["feature_value"]
            description = ref_row.get("reference_description", "") or ""
            material = ref_mat.get((feature_col, true_value), {})
            ref_img = material.get("img_path") or material.get("illust_path") or ""
            if not ref_img:
                print(f"  skip {feature_col}={true_value}: no reference image")
                continue
            options = options_by_feature.get(feature_col, [])

            tasks.append(
                (
                    model,
                    true_value,
                    "existence",
                    feature_col,
                    "YES",
                    existence_parts(feature_col, ref_img, true_value),
                    "ync",
                    [],
                )
            )
            tasks.append(
                (
                    model,
                    true_value,
                    "agreement",
                    feature_col,
                    "YES",
                    agreement_parts(feature_col, true_value, description, ref_img),
                    "ync",
                    [],
                )
            )
            tasks.append(
                (
                    model,
                    true_value,
                    "blind_mc",
                    feature_col,
                    true_value,
                    blind_mc_parts(feature_col, options, ref_img),
                    "mc",
                    [o["value"] for o in options],
                )
            )

    pending = [task for task in tasks if (task[0], task[3], task[1], task[2]) not in done]
    print(f"\nPrimary reference exemplars: {len(primary_refs)}")
    print(f"Total prompt tasks: {len(tasks)}")
    print(f"Pending prompt tasks: {len(pending)}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Output: {OUT_FILE}")

    csv_lock = threading.Lock()
    done_lock = threading.Lock()
    error_count = 0

    def process_task(task: tuple) -> None:
        model, feature_value, prompt_type, feature_col, expected, parts, parser_kind, values = task
        key = (model, feature_col, feature_value, prompt_type)
        with done_lock:
            if key in done:
                return

        raw = call_model(model, parts)
        parsed = parse_ync(raw) if parser_kind == "ync" else parse_mc(raw, values)

        with csv_lock:
            if key in done:
                return
            save_row(done, model, feature_value, prompt_type, feature_col, expected, raw, parsed)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_task, task) for task in pending]
        for index, future in enumerate(as_completed(futures), 1):
            exc = future.exception()
            if exc:
                error_count += 1
                print(f"  [ERROR] {exc}")
            elif index % 10 == 0 or index == len(futures):
                print(f"  completed {index}/{len(futures)}")

    print(f"\nDone. Raw results -> {OUT_FILE}")
    if error_count:
        raise SystemExit(1)


def save_row(done: set, model: str, feature_value: str, prompt_type: str, feature_col: str, expected: str, raw: str, parsed: str) -> None:
    key = (model, feature_col, feature_value, prompt_type)
    pd.DataFrame([
        {
            "model": model,
            "feature": feature_col,
            "feature_value": feature_value,
            "prompt_type": prompt_type,
            "raw_response": raw,
            "parsed": parsed,
            "expected": expected,
            "correct": parsed == expected,
        }
    ]).to_csv(OUT_FILE, mode="a", header=not OUT_FILE.exists(), index=False)
    done.add(key)


if __name__ == "__main__":
    main()
