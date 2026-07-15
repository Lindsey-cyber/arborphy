from __future__ import annotations

import base64
from html import escape
from io import BytesIO
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable

import pandas as pd

try:
    from IPython.display import HTML, Markdown, display
except ImportError:
    HTML = str
    Markdown = str

    def display(value: object) -> None:
        print(value)


def find_repo_root() -> Path:
    path = Path.cwd().resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "scripts" / "run_stepwise_trial.py").exists():
            return candidate
    notebooks_parent = Path(__file__).resolve().parents[1]
    if (notebooks_parent / "scripts" / "run_stepwise_trial.py").exists():
        return notebooks_parent
    raise FileNotFoundError("Could not find repo root containing scripts/run_stepwise_trial.py")


ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stepwise_trial import metadata_path_for, uv_executable  # noqa: E402
from scripts.stepwise_metrics import (  # noqa: E402
    FEATURE_GROUP_COLUMNS,
    annotate_results,
    metric_definitions,
    outcome_by_true_value,
    outcome_pairs,
    summarize,
    whole_experiment_dashboard,
)


DISPLAY_ROW_LIMIT = 50
FEATURE_TITLES = {
    "key_flower_type": "Flower Type",
    "key_plant_type": "Plant Type",
    "key_leaf_type": "Leaf Type",
}
HEATMAP_COLORS = {
    "green": "#08b745",
    "yellow": "#ffc400",
    "red": "#ff2d2d",
    "gray": "#d7dde5",
    "blank": "#f1f5f9",
}


def feature_arg(features: str | Iterable[str]) -> str:
    if isinstance(features, str):
        return features
    return ",".join(features)


def model_arg(model: str | Iterable[str]) -> str:
    if isinstance(model, str):
        return model
    return ",".join(model)


def run_trial_command(cmd: list[str]) -> None:
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    output_lines: list[str] = []
    progress_started = False

    def show_progress(message: str) -> None:
        nonlocal progress_started
        if not progress_started:
            display(Markdown("### Run Progress"))
            progress_started = True
        print(message)

    assert process.stdout is not None
    for line in process.stdout:
        output_lines.append(line)
        text = line.strip().lstrip("\a")
        if not text:
            continue

        if text.startswith("pending feature tasks:"):
            show_progress(text)
        elif text.startswith(
            (
                "expected model calls:",
                "expected completed rows:",
                "model call budget:",
                "cost alerts:",
                "OpenRouter prices:",
                "already completed tasks:",
                "tasks without true value:",
            )
        ):
            show_progress(text)
        elif (
            text.startswith(("[BUDGET ALERT]", "[TOKEN ALERT]"))
            or "/1M prompt" in text
            or "OpenRouter models" in text
            or "model not found in OpenRouter" in text
        ):
            show_progress(text)
        elif text.startswith("note:"):
            show_progress(text)
        elif text.startswith("completed feature tasks:"):
            show_progress(text)
        elif "[ERROR]" in text or text.startswith("Failed:"):
            show_progress(text)

    returncode = process.wait()
    if returncode != 0:
        tail = "".join(output_lines[-200:])
        if tail:
            print(tail, file=sys.stderr)
        raise RuntimeError(f"Trial failed with exit code {returncode}")


def int_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:g}"


def percent_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.1%}"


def money_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)
    if amount == 0:
        return "$0"
    if amount < 0.01:
        return f"${amount:.4f}"
    return f"${amount:.2f}"


def ratio_text(count: object, denominator: object, rate: object) -> str:
    count_part = int_text(count)
    denominator_part = int_text(denominator)
    rate_part = percent_text(rate)
    if denominator_part == "0" and not rate_part:
        return f"NA ({count_part or '0'}/0)"
    if count_part and denominator_part and rate_part:
        return f"{count_part}/{denominator_part} ({rate_part})"
    if count_part and denominator_part:
        return f"{count_part}/{denominator_part}"
    return count_part or rate_part


def unique_count(rows: pd.DataFrame, column: str) -> int:
    if column not in rows.columns:
        return 0
    return int(rows[column].dropna().astype(str).nunique())


def path_text(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def parse_prompt_parts(value: object) -> list[object]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    return parsed if isinstance(parsed, list) else [parsed]


def compact_text_html(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value)
    return (
        '<pre style="white-space:pre-wrap;margin:0;font:12px/1.45 ui-monospace,'
        f'SFMono-Regular,Menlo,monospace;">{escape(text)}</pre>'
    )


def feature_title(feature: str) -> str:
    return FEATURE_TITLES.get(feature, feature.replace("_", " ").title())


def compact_model_name(model: object) -> str:
    text = str(model or "").strip()
    if not text:
        return "(unknown)"
    text = text.split("/")[-1].replace(":free", "")
    replacements = {
        "gpt-4o-mini": "GPT-4o mini",
        "gpt-5-mini": "GPT-5 mini",
        "gemini-3-flash-preview": "Gemini 3 Flash",
        "gemini-3.1-pro": "Gemini 3.1 Pro",
        "claude-sonnet": "Claude Sonnet",
        "claude-haiku": "Claude Haiku",
    }
    return replacements.get(text.lower(), text.replace("-", " ").title())


def prompt_part_html(part: object, index: int, input_image_url: str = "") -> str:
    label = f"Part {index}"
    if isinstance(part, dict) and "image" in part:
        image_url = str(part.get("image", ""))
        same_image = image_url and image_url == input_image_url
        image_label = "input image" if same_image else "image"
        thumb = (
            f'<img src="{escape(image_url)}" style="max-width:180px;max-height:140px;object-fit:contain;'
            'border:1px solid #e2e8f0;margin-top:6px;">'
            if image_url
            else ""
        )
        return f"""
        <div style="margin:8px 0;">
            <div style="font-size:12px;font-weight:700;color:#475569;">{label}: {image_label}</div>
            <a href="{escape(image_url)}" target="_blank" style="font-size:12px;word-break:break-all;">{escape(image_url)}</a>
            <div>{thumb}</div>
        </div>
        """
    text = json.dumps(part, ensure_ascii=False, indent=2) if isinstance(part, (dict, list)) else str(part)
    return f"""
    <div style="margin:8px 0;">
        <div style="font-size:12px;font-weight:700;color:#475569;">{label}</div>
        <pre style="white-space:pre-wrap;margin:4px 0 0 0;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;">{escape(text)}</pre>
    </div>
    """


def prompt_parts_html(parts: list[object], input_image_url: str = "") -> str:
    if not parts:
        return '<span style="color:#64748b;">SKIPPED / NOT_APPLICABLE</span>'
    return "".join(prompt_part_html(part, index, input_image_url) for index, part in enumerate(parts, 1))


def pick_trace_row(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        raise ValueError("Cannot pick a trace row from an empty dataframe")
    if "p2_prompt_parts_json" in rows.columns:
        p2_parts = rows["p2_prompt_parts_json"].map(parse_prompt_parts)
        candidates = rows.loc[p2_parts.map(bool)]
        if not candidates.empty:
            return candidates.iloc[0]
    return rows.iloc[0]


def artifact_value_html(column: str, value: object, row: pd.Series) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return '<span style="color:#64748b;">empty</span>'
    text = str(value)
    if text == "":
        return '<span style="color:#64748b;">empty</span>'

    if column == "photo_url":
        return (
            f'<a href="{escape(text)}" target="_blank" style="word-break:break-all;">{escape(text)}</a>'
            f'<div><img src="{escape(text)}" style="max-width:220px;max-height:170px;object-fit:contain;'
            'border:1px solid #e2e8f0;margin-top:6px;"></div>'
        )

    if column in {"p1_prompt_parts_json", "p2_prompt_parts_json"}:
        parts = parse_prompt_parts(value)
        if not parts:
            return '<span style="color:#64748b;">empty / skipped</span>'
        input_image_url = str(row.get("photo_url", ""))
        return f"""
        <details>
            <summary style="cursor:pointer;font-weight:700;">{len(parts)} prompt parts</summary>
            <div style="max-height:360px;overflow:auto;border:1px solid #e2e8f0;padding:8px;margin-top:6px;">
                {prompt_parts_html(parts, input_image_url)}
            </div>
        </details>
        """

    if column in {"p1_raw", "p2_raw", "p1_parse_rule", "p2_parse_rule"}:
        return compact_text_html(value)

    if len(text) > 120:
        return compact_text_html(value)
    return escape(text)


def display_artifact_row_example(raw: pd.DataFrame) -> None:
    if raw.empty:
        return
    display(HTML(artifact_row_example_html(raw)))


def artifact_row_example_html(raw: pd.DataFrame) -> str:
    if raw.empty:
        return ""
    row = pick_trace_row(raw)
    body = "".join(
        f"""
        <tr>
            <th style="vertical-align:top;text-align:left;width:210px;padding:9px 10px;border-bottom:1px solid #e2e8f0;background:#f8fafc;">
                <code>{escape(column)}</code>
            </th>
            <td style="vertical-align:top;padding:9px 10px;border-bottom:1px solid #e2e8f0;">
                {artifact_value_html(column, row.get(column, ""), row)}
            </td>
        </tr>
        """
        for column in raw.columns
    )
    return f"""
    <table class="vertical-table" style="border-collapse:collapse;width:100%;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px;">
        <tbody>{body}</tbody>
    </table>
    """


def heatmap_color(kind: str, row: pd.Series) -> str:
    if kind == "see":
        p1 = str(row.get("p1_parsed", "")).strip().upper()
        if p1 == "YES":
            return HEATMAP_COLORS["green"]
        if p1 == "NO":
            return HEATMAP_COLORS["red"]
        return HEATMAP_COLORS["yellow"]

    outcome = str(row.get("outcome", "")).strip().upper()
    if outcome == "CORRECT":
        return HEATMAP_COLORS["green"]
    if outcome == "WRONG":
        return HEATMAP_COLORS["red"]
    if outcome == "NOT_APPLICABLE" or str(row.get("p2_parsed", "")).strip().upper() in {"NA", "N/A", "SKIPPED", "NOT_APPLICABLE"}:
        return HEATMAP_COLORS["gray"]
    if outcome == "INCONCLUSIVE":
        return HEATMAP_COLORS["yellow"]
    return HEATMAP_COLORS["blank"]


def heatmap_score(kind: str, group: pd.DataFrame) -> float | None:
    if group.empty:
        return None
    if kind == "see":
        return float(group["p1_parsed"].fillna("").astype(str).str.upper().eq("YES").mean())
    return float(group["outcome"].fillna("").astype(str).str.upper().eq("CORRECT").mean())


def heatmap_features(rows: pd.DataFrame) -> list[str]:
    features = [feature for feature in FEATURE_TITLES if feature in set(rows["feature"].astype(str))]
    features.extend(
        feature
        for feature in rows["feature"].dropna().astype(str).unique()
        if feature not in features
    )
    return features


def heatmap_models(rows: pd.DataFrame) -> list[str]:
    return list(rows["model"].dropna().astype(str).drop_duplicates())


def heatmap_category(kind: str, row: pd.Series) -> str:
    if kind == "see":
        p1 = str(row.get("p1_parsed", "")).strip().upper()
        if p1 == "YES":
            return "yes"
        if p1 == "NO":
            return "no"
        return "inconclusive"

    outcome = str(row.get("outcome", "")).strip().upper()
    p2 = str(row.get("p2_parsed", "")).strip().upper()
    if outcome == "CORRECT":
        return "correct"
    if outcome == "WRONG":
        return "wrong"
    if outcome == "INCONCLUSIVE":
        return "inconclusive"
    if outcome == "NOT_APPLICABLE" or p2 in {"NA", "N/A", "SKIPPED", "NOT_APPLICABLE"}:
        return "na"
    return "na"


def heatmap_category_specs(kind: str) -> list[tuple[str, str, str]]:
    if kind == "see":
        return [
            ("yes", "YES", HEATMAP_COLORS["green"]),
            ("inconclusive", "INCONCLUSIVE", HEATMAP_COLORS["yellow"]),
            ("no", "NO", HEATMAP_COLORS["red"]),
        ]
    return [
        ("correct", "CORRECT", HEATMAP_COLORS["green"]),
        ("inconclusive", "INCONCLUSIVE", HEATMAP_COLORS["yellow"]),
        ("wrong", "WRONG", HEATMAP_COLORS["red"]),
        ("na", "N/A", HEATMAP_COLORS["gray"]),
    ]


def matplotlib_font(preferred: str, fallback: str = "DejaVu Sans") -> str:
    try:
        from matplotlib import font_manager

        font_manager.findfont(preferred, fallback_to_default=False)
    except Exception:
        return fallback
    return preferred


def ensure_matplotlib_config_dir() -> None:
    config_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/arborphy-matplotlib"))
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", "/tmp/arborphy-cache"))
    config_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def heatmap_matrix(kind: str, feature_rows: pd.DataFrame, model_order: list[str]) -> tuple[list[list[int]], int]:
    specs = heatmap_category_specs(kind)
    category_index = {name: index for index, (name, _, _) in enumerate(specs)}
    blank_index = len(specs)
    n_cells = max(1, feature_rows.groupby("model", dropna=False).size().max())
    matrix = [[blank_index for _ in range(n_cells)] for _ in model_order]

    sort_columns = [column for column in ["observation_id", "trial_id", "run_id"] if column in feature_rows.columns]
    if sort_columns:
        feature_rows = feature_rows.sort_values(sort_columns, kind="stable")

    for row_index, model in enumerate(model_order):
        model_rows = feature_rows.loc[feature_rows["model"].astype(str) == str(model)]
        for col_index, (_, row) in enumerate(model_rows.iterrows()):
            matrix[row_index][col_index] = category_index.get(heatmap_category(kind, row), blank_index)
    return matrix, n_cells


def render_benchmark_heatmap_png(rows: pd.DataFrame, kind: str) -> str:
    ensure_matplotlib_config_dir()
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render the benchmark heatmap") from exc

    features = heatmap_features(rows)
    model_order = heatmap_models(rows)
    if not features or not model_order:
        return ""

    specs = heatmap_category_specs(kind)
    colors = [color for _, _, color in specs] + [HEATMAP_COLORS["blank"]]
    cmap = ListedColormap(colors)
    title_font = "DejaVu Sans"
    label_font = matplotlib_font("Arial")
    panel_count = len(features)
    model_count = len(model_order)
    fig_width = 7.6
    fig_height = max(3.4, 1.15 + panel_count * (0.72 + 0.23 * model_count))
    fig, axes = plt.subplots(panel_count, 1, figsize=(fig_width, fig_height), dpi=180)
    if panel_count == 1:
        axes = [axes]

    fig.patch.set_facecolor("white")
    fig.suptitle(
        "See Feature" if kind == "see" else "MC Feature Value",
        fontfamily=title_font,
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )

    for ax, feature in zip(axes, features):
        feature_rows = rows.loc[rows["feature"] == feature].copy()
        matrix, n_cells = heatmap_matrix(kind, feature_rows, model_order)
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=len(colors) - 1, aspect="auto", interpolation="nearest")
        ax.set_title(feature_title(feature), loc="left", fontfamily=label_font, fontsize=10, pad=4)
        ax.set_xticks([])
        ax.set_yticks(range(model_count))
        ax.set_yticklabels([compact_model_name(model) for model in model_order], fontfamily=label_font, fontsize=7)
        ax.tick_params(axis="both", length=0, pad=2)
        if n_cells <= 150:
            ax.set_xticks([x - 0.5 for x in range(1, n_cells)], minor=True)
            ax.grid(which="minor", axis="x", color="white", linewidth=0.18)
        for spine in ax.spines.values():
            spine.set_linewidth(0.45)
            spine.set_color("#1f2937")

        scores = [
            score
            for model in model_order
            if (score := heatmap_score(kind, feature_rows.loc[feature_rows["model"].astype(str) == str(model)])) is not None
        ]
        best = max(scores) if scores else None
        worst = min(scores) if scores else None
        ax.text(
            0.38,
            -0.08,
            f"best: {best:.0%}" if best is not None else "best: NA",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            fontfamily=label_font,
            fontweight="bold",
            color="#008c2f",
        )
        ax.text(
            0.62,
            -0.08,
            f"worst: {worst:.0%}" if worst is not None else "worst: NA",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            fontfamily=label_font,
            fontweight="bold",
            color="#e60019",
        )

    handles = [Patch(facecolor=color, edgecolor="none", label=label) for _, label, color in specs]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
        prop={"family": label_font, "size": 7},
        handlelength=1.8,
        handleheight=0.8,
        columnspacing=1.4,
    )
    fig.subplots_adjust(left=0.19, right=0.985, top=0.89, bottom=0.20, hspace=1.05)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def benchmark_heatmap_images(rows: pd.DataFrame) -> dict[str, str]:
    if rows.empty:
        return {}
    return {
        "see": render_benchmark_heatmap_png(rows, "see"),
        "mc": render_benchmark_heatmap_png(rows, "mc"),
    }


def benchmark_heatmap_html(images: dict[str, str]) -> str:
    if not images:
        return ""
    return f"""
    <div class="chart-grid" style="display:grid;grid-template-columns:repeat(2,minmax(360px,1fr));gap:22px;align-items:start;overflow-x:auto;">
        <figure class="chart-card" style="margin:0;background:#fff;border:1px solid #e5e7eb;padding:10px;">
            <img src="{images.get('see', '')}" alt="See Feature heatmap" style="display:block;width:100%;height:auto;">
        </figure>
        <figure class="chart-card" style="margin:0;background:#fff;border:1px solid #e5e7eb;padding:10px;">
            <img src="{images.get('mc', '')}" alt="MC Feature Value heatmap" style="display:block;width:100%;height:auto;">
        </figure>
    </div>
    """


def display_benchmark_heatmap(rows: pd.DataFrame, images: dict[str, str] | None = None) -> None:
    images = images if images is not None else benchmark_heatmap_images(rows)
    display(HTML(f"<div>{benchmark_heatmap_html(images)}</div>"))


def focus_key_columns(rows: pd.DataFrame) -> list[str]:
    return [
        column
        for column in ["trial_id", "run_id", "prompt_set", "observation_id"]
        if column in rows.columns
    ] or ["__index__"]


def focus_key(row: pd.Series, key_columns: list[str]) -> tuple[str, ...]:
    if key_columns == ["__index__"]:
        return (str(row.name),)
    return tuple(str(row.get(column, "")) for column in key_columns)


def focus_columns(model_rows: pd.DataFrame, key_columns: list[str]) -> list[tuple[str, ...]]:
    if key_columns == ["__index__"]:
        return [(str(index),) for index in model_rows.index]
    key_frame = model_rows[key_columns].fillna("").astype(str).drop_duplicates()
    return [tuple(values) for values in key_frame.sort_values(key_columns, kind="stable").to_numpy()]


def focus_heatmap_matrix(
    rows: pd.DataFrame,
    *,
    model: str,
    kind: str,
) -> tuple[list[list[int]], list[str], list[tuple[str, ...]]]:
    model_rows = rows.loc[rows["model"].astype(str) == str(model)].copy()
    features = heatmap_features(model_rows)
    key_columns = focus_key_columns(model_rows)
    task_columns = focus_columns(model_rows, key_columns)
    specs = heatmap_category_specs(kind)
    category_index = {name: index for index, (name, _, _) in enumerate(specs)}
    blank_index = len(specs)
    matrix = [[blank_index for _ in task_columns] for _ in features]
    feature_index = {feature: index for index, feature in enumerate(features)}
    task_index = {key: index for index, key in enumerate(task_columns)}

    sort_columns = [column for column in key_columns if column in model_rows.columns]
    if sort_columns:
        model_rows = model_rows.sort_values(sort_columns + ["feature"], kind="stable")

    for _, row in model_rows.iterrows():
        feature = str(row.get("feature", ""))
        key = focus_key(row, key_columns)
        if feature not in feature_index or key not in task_index:
            continue
        matrix[feature_index[feature]][task_index[key]] = category_index.get(heatmap_category(kind, row), blank_index)
    return matrix, features, task_columns


def render_model_focus_heatmap_png(rows: pd.DataFrame, model: str, kind: str) -> str:
    ensure_matplotlib_config_dir()
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render the model focus heatmap") from exc

    matrix, features, task_columns = focus_heatmap_matrix(rows, model=model, kind=kind)
    if not features or not task_columns:
        return ""

    specs = heatmap_category_specs(kind)
    colors = [color for _, _, color in specs] + [HEATMAP_COLORS["blank"]]
    cmap = ListedColormap(colors)
    title_font = "DejaVu Sans"
    label_font = matplotlib_font("Arial")
    fig_width = max(7.4, min(16, 2.4 + 0.055 * len(task_columns)))
    fig_height = max(2.6, 1.55 + 0.42 * len(features))
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=180)
    fig.patch.set_facecolor("white")
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=len(colors) - 1, aspect="auto", interpolation="nearest")
    ax.set_title(
        f"{compact_model_name(model)} - {'See Feature' if kind == 'see' else 'MC Feature Value'}",
        fontfamily=title_font,
        fontsize=13,
        fontweight="bold",
        pad=8,
    )
    ax.set_xticks([])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([feature_title(feature) for feature in features], fontfamily=label_font, fontsize=8)
    ax.tick_params(axis="both", length=0, pad=3)
    if len(task_columns) <= 180:
        ax.set_xticks([x - 0.5 for x in range(1, len(task_columns))], minor=True)
        ax.grid(which="minor", axis="x", color="white", linewidth=0.18)
    if len(features) <= 60:
        ax.set_yticks([y - 0.5 for y in range(1, len(features))], minor=True)
        ax.grid(which="minor", axis="y", color="white", linewidth=0.35)
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_color("#1f2937")

    handles = [Patch(facecolor=color, edgecolor="none", label=label) for _, label, color in specs]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        prop={"family": label_font, "size": 6.5},
        handlelength=1.4,
        handleheight=0.75,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.18, right=0.99, top=0.78, bottom=0.28)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=180, facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def model_focus_heatmap_images(rows: pd.DataFrame) -> dict[str, dict[str, str]]:
    if rows.empty:
        return {}
    return {
        model: {
            "see": render_model_focus_heatmap_png(rows, model, "see"),
            "mc": render_model_focus_heatmap_png(rows, model, "mc"),
        }
        for model in heatmap_models(rows)
    }


def model_focus_summary_view(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    rows = []
    group_columns = [column for column in ["model"] if column in summary.columns]
    if not group_columns:
        return pd.DataFrame()
    for model, group in summary.groupby("model", dropna=False, sort=False):
        feature_count = int(group["feature_count"].sum())
        p2_applicable = int(group["p2_applicable_count"].sum())
        committed = int(group["committed_count"].sum())
        correct = int(group["correct_count"].sum())
        wrong = int(group["wrong_count"].sum())
        inconclusive = int(group["inconclusive_count"].sum())
        not_applicable = int(group["not_applicable_count"].sum())
        features_seen = int(group["features_seen"].sum())
        rows.append(
            {
                "model": model,
                "features": int_text(group["feature"].nunique()) if "feature" in group.columns else "",
                "n": int_text(feature_count),
                "p1_seen": ratio_text(features_seen, feature_count, features_seen / feature_count if feature_count else None),
                "p2_correct": ratio_text(correct, p2_applicable, correct / p2_applicable if p2_applicable else None),
                "p2_wrong": ratio_text(wrong, p2_applicable, wrong / p2_applicable if p2_applicable else None),
                "p2_inconclusive": ratio_text(
                    inconclusive,
                    p2_applicable,
                    inconclusive / p2_applicable if p2_applicable else None,
                ),
                "p2_na": ratio_text(
                    not_applicable,
                    feature_count,
                    not_applicable / feature_count if feature_count else None,
                ),
                "accuracy_when_committed": ratio_text(
                    correct,
                    committed,
                    correct / committed if committed else None,
                ),
            }
        )
    return pd.DataFrame(rows)


def model_focus_heatmap_html(focus_images: dict[str, dict[str, str]]) -> str:
    if not focus_images:
        return ""
    sections = []
    for index, (model, images) in enumerate(focus_images.items()):
        open_attr = " open" if index == 0 else ""
        sections.append(
            f"""
            <details class="subpanel"{open_attr}>
                <summary>{escape(compact_model_name(model))} <span>{escape(model)}</span></summary>
                <div class="chart-grid focus-chart-grid">
                    <figure class="chart-card">
                        <img src="{images.get('see', '')}" alt="{escape(model)} see feature focus heatmap">
                    </figure>
                    <figure class="chart-card">
                        <img src="{images.get('mc', '')}" alt="{escape(model)} MC feature value focus heatmap">
                    </figure>
                </div>
            </details>
            """
        )
    return "".join(sections)


def display_model_focus_heatmaps(rows: pd.DataFrame, focus_images: dict[str, dict[str, str]] | None = None) -> None:
    focus_images = focus_images if focus_images is not None else model_focus_heatmap_images(rows)
    display(HTML(model_focus_heatmap_html(focus_images)))


def experiment_scale_view(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    completed_rows = len(rows)
    image_count = unique_count(rows, "observation_id")
    feature_count = unique_count(rows, "feature")
    model_count = unique_count(rows, "model")
    trial_count = unique_count(rows, "trial_id")
    run_count = unique_count(rows, "run_id") if "run_id" in rows.columns else 0
    config_columns = [
        column
        for column in ["trial_id", "run_id", "prompt_set", "model"]
        if column in rows.columns
    ]
    config_count = (
        int(rows[config_columns].fillna("").astype(str).drop_duplicates().shape[0])
        if config_columns
        else max(model_count, 1)
    )
    image_feature_tasks = image_count * feature_count
    expected_rows = image_feature_tasks * max(config_count, 1)
    p2_call_count = int(rows["p2_applicable"].sum()) if "p2_applicable" in rows.columns else 0
    estimated_model_calls = completed_rows + p2_call_count
    max_model_calls = expected_rows * 2
    completion_rate = completed_rows / expected_rows if expected_rows else None
    call_budget_rate = estimated_model_calls / max_model_calls if max_model_calls else None

    records = [
        {
            "metric": "images",
            "value": int_text(image_count),
            "meaning": "unique observation_id values",
        },
        {
            "metric": "features",
            "value": int_text(feature_count),
            "meaning": "unique feature names",
        },
        {
            "metric": "image_feature_tasks",
            "value": int_text(image_feature_tasks),
            "meaning": "images x features, before multiplying by models/runs",
        },
        {
            "metric": "models",
            "value": int_text(model_count),
            "meaning": "unique model values",
        },
        {
            "metric": "runs",
            "value": int_text(run_count) if run_count else "",
            "meaning": "unique run_id values when present",
        },
        {
            "metric": "trials",
            "value": int_text(trial_count),
            "meaning": "unique trial_id values",
        },
        {
            "metric": "model_run_configs",
            "value": int_text(config_count),
            "meaning": "unique trial_id x run_id x prompt_set x model combinations",
        },
        {
            "metric": "expected_completed_rows_if_full_grid",
            "value": int_text(expected_rows),
            "meaning": "images x features x model_run_configs",
        },
        {
            "metric": "completed_rows",
            "value": ratio_text(completed_rows, expected_rows, completion_rate),
            "meaning": "actual CSV rows / expected full-grid rows",
        },
        {
            "metric": "p1_model_calls_completed",
            "value": int_text(completed_rows),
            "meaning": "one P1 call per completed row",
        },
        {
            "metric": "p2_model_calls_completed",
            "value": int_text(p2_call_count),
            "meaning": "one P2 call only where P1 parsed YES",
        },
        {
            "metric": "estimated_model_calls_completed",
            "value": int_text(estimated_model_calls),
            "meaning": "P1 calls + P2 calls represented by completed rows",
        },
        {
            "metric": "max_model_calls_if_all_p2",
            "value": int_text(max_model_calls),
            "meaning": "upper bound if every full-grid row asks both P1 and P2",
        },
        {
            "metric": "model_call_budget_used",
            "value": ratio_text(estimated_model_calls, max_model_calls, call_budget_rate),
            "meaning": "completed model calls / upper-bound full-grid model calls",
        },
    ]
    return pd.DataFrame(records)


def scale_metric_value(scale_view: pd.DataFrame, metric: str) -> str:
    if scale_view.empty:
        return ""
    matches = scale_view.loc[scale_view["metric"] == metric, "value"]
    if matches.empty:
        return ""
    return str(matches.iloc[0])


def metadata_budget(metadata: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(metadata, dict):
        return {}
    budget = metadata.get("budget")
    return budget if isinstance(budget, dict) else {}


def nested_dict(metadata: dict[str, object], key: str) -> dict[str, object]:
    value = metadata.get(key)
    return value if isinstance(value, dict) else {}


def response_model_text(row: dict[str, object]) -> str:
    response_models = row.get("response_models")
    if not isinstance(response_models, list):
        return ""
    parts = []
    for item in response_models:
        if not isinstance(item, dict):
            continue
        model = str(item.get("model", "")).strip()
        calls = int_text(item.get("calls"))
        if model:
            parts.append(f"{model} ({calls})" if calls else model)
    return ", ".join(parts)


def openrouter_usage_by_model_view(metadata: dict[str, object] | None) -> pd.DataFrame:
    budget = metadata_budget(metadata)
    rows = budget.get("openrouter_usage_by_model")
    if not isinstance(rows, list):
        return pd.DataFrame()
    records = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        records.append(
            {
                "model": row.get("model", ""),
                "response_models": response_model_text(row),
                "calls": int_text(row.get("captured_calls")),
                "prompt_tokens": int_text(row.get("prompt_tokens")),
                "completion_tokens": int_text(row.get("completion_tokens")),
                "reasoning_tokens": int_text(row.get("reasoning_tokens")),
                "cached_tokens": int_text(row.get("cached_tokens")),
                "cache_write_tokens": int_text(row.get("cache_write_tokens")),
                "total_tokens": int_text(row.get("total_tokens")),
                "cost": money_text(row.get("cost_usd")),
                "avg_cost_call": money_text(row.get("avg_cost_per_call_usd")),
                "cost_per_1k_tokens": money_text(row.get("cost_per_1k_tokens_usd")),
            }
        )
    return pd.DataFrame(records)


def openrouter_price_snapshot_view(metadata: dict[str, object] | None) -> pd.DataFrame:
    budget = metadata_budget(metadata)
    rows = budget.get("openrouter_price_snapshot")
    if not isinstance(rows, list):
        return pd.DataFrame()
    records = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        records.append(
            {
                "model": row.get("model", ""),
                "openrouter_id": row.get("openrouter_id", ""),
                "name": row.get("name", ""),
                "prompt_per_1m": money_text(row.get("prompt_usd_per_million")),
                "completion_per_1m": money_text(row.get("completion_usd_per_million")),
                "reasoning_per_1m": money_text(row.get("reasoning_usd_per_million")),
                "cache_read_per_1m": money_text(row.get("cache_read_usd_per_million")),
                "context_length": int_text(row.get("context_length")),
                "pricing_error": row.get("pricing_error", ""),
            }
        )
    return pd.DataFrame(records)


def compact_dashboard_view(
    dashboard: pd.DataFrame,
    scale_view: pd.DataFrame | None = None,
    metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    if dashboard.empty:
        return dashboard
    scale_view = scale_view if scale_view is not None else pd.DataFrame()
    budget = metadata_budget(metadata)
    usage = nested_dict(budget, "openrouter_usage")
    row = dashboard.iloc[0]
    feature_count = row.get("feature_count")
    p2_applicable_count = row.get("p2_applicable_count")
    committed_count = row.get("committed_count")
    completed_calls = budget.get("completed_model_calls") or scale_metric_value(scale_view, "estimated_model_calls_completed")
    max_calls = budget.get("max_model_calls_full_grid") or scale_metric_value(scale_view, "max_model_calls_if_all_p2")
    try:
        call_budget_rate = float(completed_calls) / float(max_calls) if completed_calls and max_calls else None
    except (TypeError, ValueError):
        call_budget_rate = None
    records = [
        {"metric": "images", "value": scale_metric_value(scale_view, "images") or int_text(row.get("n_observations"))},
        {"metric": "features", "value": scale_metric_value(scale_view, "features") or int_text(row.get("feature_name_count"))},
        {"metric": "image_feature_tasks", "value": scale_metric_value(scale_view, "image_feature_tasks")},
        {"metric": "models", "value": scale_metric_value(scale_view, "models") or int_text(row.get("model_count"))},
        {"metric": "runs", "value": scale_metric_value(scale_view, "runs")},
        {"metric": "trials", "value": scale_metric_value(scale_view, "trials") or int_text(row.get("trial_count"))},
        {"metric": "model_run_configs", "value": scale_metric_value(scale_view, "model_run_configs")},
        {
            "metric": "expected_completed_rows_if_full_grid",
            "value": scale_metric_value(scale_view, "expected_completed_rows_if_full_grid"),
        },
        {"metric": "completed_rows", "value": scale_metric_value(scale_view, "completed_rows") or int_text(feature_count)},
        {
            "metric": "estimated_model_calls_completed",
            "value": int_text(completed_calls) or scale_metric_value(scale_view, "estimated_model_calls_completed"),
        },
        {
            "metric": "model_call_budget",
            "value": ratio_text(completed_calls, max_calls, call_budget_rate),
        },
        {
            "metric": "captured_openrouter_calls",
            "value": int_text(usage.get("captured_calls")) if usage.get("captured_calls") is not None else "not captured",
        },
        {
            "metric": "captured_tokens",
            "value": int_text(usage.get("total_tokens")) if usage.get("total_tokens") is not None else "not captured",
        },
        {
            "metric": "captured_cost_usd",
            "value": money_text(usage.get("cost_usd")) or "not captured",
        },
        {
            "metric": "features_seen",
            "value": ratio_text(row.get("features_seen"), feature_count, row.get("features_seen_rate")),
        },
        {
            "metric": "features_not_seen",
            "value": ratio_text(row.get("features_not_seen"), feature_count, row.get("features_not_seen_rate")),
        },
        {
            "metric": "p1_inconclusive",
            "value": ratio_text(
                row.get("visibility_inconclusive_count"),
                feature_count,
                row.get("visibility_inconclusive_rate"),
            ),
        },
        {
            "metric": "p2_not_applicable",
            "value": ratio_text(
                row.get("not_applicable_count"),
                feature_count,
                row.get("not_applicable_rate"),
            ),
        },
        {
            "metric": "p2_committed",
            "value": ratio_text(committed_count, p2_applicable_count, row.get("p2_commitment_rate")),
        },
        {
            "metric": "p2_correct",
            "value": ratio_text(row.get("correct_count"), p2_applicable_count, row.get("p2_correct_rate")),
        },
        {
            "metric": "p2_wrong",
            "value": ratio_text(row.get("wrong_count"), p2_applicable_count, row.get("p2_wrong_rate")),
        },
        {
            "metric": "p2_inconclusive",
            "value": ratio_text(
                row.get("inconclusive_count"),
                p2_applicable_count,
                row.get("p2_inconclusive_rate"),
            ),
        },
        {
            "metric": "accuracy_when_committed",
            "value": ratio_text(row.get("correct_count"), committed_count, row.get("committed_accuracy")),
        },
    ]
    return pd.DataFrame(records)


def compact_summary_view(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    identity_columns = [
        column
        for column in ["trial_id", "run_id", "model", "prompt_set", "feature"]
        if column in summary.columns
    ]
    view = summary[identity_columns].copy()
    view["n"] = summary["feature_count"].map(int_text)
    if "n_observations" in summary.columns:
        view["observations"] = summary["n_observations"].map(int_text)
    view["p1_seen"] = [
        ratio_text(row.get("features_seen"), row.get("feature_count"), row.get("features_seen_rate"))
        for _, row in summary.iterrows()
    ]
    view["p1_no"] = [
        ratio_text(row.get("features_not_seen"), row.get("feature_count"), row.get("features_not_seen_rate"))
        for _, row in summary.iterrows()
    ]
    view["p1_unclear"] = [
        ratio_text(
            row.get("visibility_inconclusive_count"),
            row.get("feature_count"),
            row.get("visibility_inconclusive_rate"),
        )
        for _, row in summary.iterrows()
    ]
    view["p2_na"] = [
        ratio_text(row.get("not_applicable_count"), row.get("feature_count"), row.get("not_applicable_rate"))
        for _, row in summary.iterrows()
    ]
    view["p2_committed"] = [
        ratio_text(row.get("committed_count"), row.get("p2_applicable_count"), row.get("p2_commitment_rate"))
        for _, row in summary.iterrows()
    ]
    view["p2_correct"] = [
        ratio_text(row.get("correct_count"), row.get("p2_applicable_count"), row.get("p2_correct_rate"))
        for _, row in summary.iterrows()
    ]
    view["p2_wrong"] = [
        ratio_text(row.get("wrong_count"), row.get("p2_applicable_count"), row.get("p2_wrong_rate"))
        for _, row in summary.iterrows()
    ]
    view["p2_inconclusive"] = [
        ratio_text(row.get("inconclusive_count"), row.get("p2_applicable_count"), row.get("p2_inconclusive_rate"))
        for _, row in summary.iterrows()
    ]
    view["accuracy_when_committed"] = [
        ratio_text(row.get("correct_count"), row.get("committed_count"), row.get("committed_accuracy"))
        for _, row in summary.iterrows()
    ]
    return view


def compact_outcome_by_true_value_view(by_true_value: pd.DataFrame) -> pd.DataFrame:
    if by_true_value.empty:
        return by_true_value
    identity_columns = [
        column
        for column in ["model", "feature", "true_value"]
        if column in by_true_value.columns
    ]
    view = by_true_value[identity_columns].copy()
    view["n"] = by_true_value["feature_count"].map(int_text)
    view["p1_seen"] = [
        ratio_text(row.get("features_seen"), row.get("feature_count"), row.get("features_seen_rate"))
        for _, row in by_true_value.iterrows()
    ]
    view["correct"] = [
        ratio_text(row.get("correct_count"), row.get("feature_count"), row.get("correct_rate"))
        for _, row in by_true_value.iterrows()
    ]
    view["wrong"] = [
        ratio_text(row.get("wrong_count"), row.get("feature_count"), row.get("wrong_rate"))
        for _, row in by_true_value.iterrows()
    ]
    view["inconclusive"] = [
        ratio_text(row.get("inconclusive_count"), row.get("feature_count"), row.get("inconclusive_rate"))
        for _, row in by_true_value.iterrows()
    ]
    view["not_applicable"] = [
        ratio_text(row.get("not_applicable_count"), row.get("feature_count"), row.get("not_applicable_rate"))
        for _, row in by_true_value.iterrows()
    ]
    view["most_common_wrong_prediction"] = by_true_value["most_common_wrong_prediction"]
    return view


def compact_rows_view(rows: pd.DataFrame, limit: int = DISPLAY_ROW_LIMIT) -> pd.DataFrame:
    columns = [
        column
        for column in [
            "observation_id",
            "model",
            "prompt_set",
            "feature",
            "true_value",
            "p1_parsed",
            "p2_parsed",
            "outcome",
        ]
        if column in rows.columns
    ]
    return rows[columns].head(limit).copy()


def safe_html_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower() or "table"


def dataframe_table_html(df: pd.DataFrame, table_id: str, max_rows: int | None = None) -> str:
    if df.empty:
        return '<p class="empty-note">No rows.</p>'
    view = df.head(max_rows).copy() if max_rows is not None else df.copy()
    return view.to_html(index=False, escape=True, classes="data-table", table_id=table_id, border=0)


def table_section_html(title: str, df: pd.DataFrame, table_id: str, max_rows: int | None = None) -> str:
    count_note = f"{min(len(df), max_rows)}/{len(df)} rows" if max_rows is not None and len(df) > max_rows else f"{len(df)} rows"
    return f"""
    <details class="panel">
        <summary>{escape(title)} <span>{escape(count_note)}</span></summary>
        <div class="table-tools">
            <input type="search" placeholder="Search this table" data-search-target="{escape(table_id)}">
        </div>
        <div class="table-wrap">
            {dataframe_table_html(df, table_id, max_rows)}
        </div>
    </details>
    """


def dashboard_cards_html(dashboard_view: pd.DataFrame) -> str:
    if dashboard_view.empty:
        return ""
    selected = [
        ("completed_rows", "Rows"),
        ("features_seen", "P1 seen"),
        ("p2_correct", "P2 correct"),
        ("p2_wrong", "P2 wrong"),
        ("accuracy_when_committed", "Committed accuracy"),
        ("captured_cost_usd", "Cost"),
    ]
    labels = dict(selected)
    metric_order = {metric: index for index, (metric, _) in enumerate(selected)}
    dashboard_view = dashboard_view.loc[dashboard_view["metric"].isin(labels)].copy()
    if dashboard_view.empty:
        return ""
    dashboard_view["__order"] = dashboard_view["metric"].map(metric_order)
    dashboard_view = dashboard_view.sort_values("__order", kind="stable")
    cards = []
    for _, row in dashboard_view.iterrows():
        metric = str(row.get("metric", ""))
        cards.append(
            f"""
            <div class="metric-card">
                <div class="metric-name">{escape(labels.get(metric, metric))}</div>
                <div class="metric-value">{escape(str(row.get("value", "")))}</div>
            </div>
            """
        )
    return f'<div class="metric-grid">{"".join(cards)}</div>'


def metadata_html(metadata: dict[str, object], output_file: Path) -> str:
    if not metadata:
        metadata = {}
    budget = metadata_budget(metadata)
    usage = nested_dict(budget, "openrouter_usage")
    usage_by_model = openrouter_usage_by_model_view(metadata)
    price_snapshot = openrouter_price_snapshot_view(metadata)
    rows = {
        "trial_id": metadata.get("trial_id", output_file.stem),
        "output_csv": output_file.name,
        "model": metadata.get("model", ""),
        "models": ", ".join(metadata.get("models", [])) if isinstance(metadata.get("models"), list) else "",
        "image_set": metadata.get("image_set", ""),
        "sample_limit": metadata.get("sample_limit", ""),
        "features": ", ".join(metadata.get("features", [])) if isinstance(metadata.get("features"), list) else "",
        "prompt_set": metadata.get("prompt_set", ""),
        "run_id": metadata.get("run_id", ""),
        "expected_completed_rows": budget.get("expected_completed_rows", ""),
        "model_call_budget": (
            f"{budget.get('min_model_calls_full_grid')} to {budget.get('max_model_calls_full_grid')}"
            if budget.get("min_model_calls_full_grid") or budget.get("max_model_calls_full_grid")
            else ""
        ),
        "completed_model_calls": budget.get("completed_model_calls", ""),
        "captured_openrouter_calls": usage.get("captured_calls", ""),
        "captured_prompt_tokens": usage.get("prompt_tokens", ""),
        "captured_completion_tokens": usage.get("completion_tokens", ""),
        "captured_reasoning_tokens": usage.get("reasoning_tokens", ""),
        "captured_cached_tokens": usage.get("cached_tokens", ""),
        "captured_tokens": usage.get("total_tokens", ""),
        "captured_cost_usd": money_text(usage.get("cost_usd")) or "",
        "avg_cost_per_call": money_text(usage.get("avg_cost_per_call_usd")) or "",
        "cost_per_1k_tokens": money_text(usage.get("cost_per_1k_tokens_usd")) or "",
    }
    body = "".join(
        f"<tr><th>{escape(key)}</th><td>{escape(str(value))}</td></tr>"
        for key, value in rows.items()
        if str(value)
    )
    usage_table = ""
    if not usage_by_model.empty:
        usage_table = f"""
        <h3>OpenRouter Usage by Model</h3>
        <div class="table-wrap">{dataframe_table_html(usage_by_model, "metadata-usage-by-model")}</div>
        """
    price_table = ""
    if not price_snapshot.empty:
        price_table = f"""
        <h3>OpenRouter Price Snapshot</h3>
        <div class="table-wrap">{dataframe_table_html(price_snapshot, "metadata-price-snapshot")}</div>
        """
    return f"""
    <details class="panel">
        <summary>Run Metadata</summary>
        <table class="metadata-table"><tbody>{body}</tbody></table>
        {usage_table}
        {price_table}
    </details>
    """


def dashboard_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def dashboard_bool(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def dashboard_json(data: object) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=True, allow_nan=False).replace("</", "<\\/")


def interactive_dashboard_payload(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {"models": [], "features": [], "tasks": []}

    models = [
        {"id": model, "label": compact_model_name(model)}
        for model in heatmap_models(rows)
    ]
    features = [
        {"id": feature, "label": feature_title(feature)}
        for feature in heatmap_features(rows)
    ]
    sort_columns = [
        column
        for column in ["feature", "model", "observation_id", "trial_id", "run_id"]
        if column in rows.columns
    ]
    ordered = rows.sort_values(sort_columns, kind="stable") if sort_columns else rows.copy()
    tasks = []
    for index, (_, row) in enumerate(ordered.iterrows()):
        feature = dashboard_text(row.get("feature"))
        model = dashboard_text(row.get("model"))
        species = dashboard_text(row.get("newcomb_species_name")) or dashboard_text(row.get("species_inat"))
        outcome = dashboard_text(row.get("outcome")).upper() or "NOT_APPLICABLE"
        p1 = dashboard_text(row.get("p1_parsed")).upper() or "INCONCLUSIVE"
        tasks.append(
            {
                "id": index,
                "model": model,
                "modelLabel": compact_model_name(model),
                "feature": feature,
                "featureLabel": feature_title(feature),
                "observationId": dashboard_text(row.get("observation_id")),
                "photoId": dashboard_text(row.get("photo_id")),
                "photoUrl": dashboard_text(row.get("photo_url")),
                "species": species,
                "trueValue": dashboard_text(row.get("true_value")),
                "predictedValue": dashboard_text(row.get("predicted_value")) or dashboard_text(row.get("p2_parsed")),
                "p1": p1,
                "p2": dashboard_text(row.get("p2_parsed")),
                "outcome": outcome,
                "committed": dashboard_bool(row.get("committed_bool", row.get("committed"))),
            }
        )
    return {"models": models, "features": features, "tasks": tasks}


def interactive_dashboard_html(rows: pd.DataFrame) -> str:
    payload = dashboard_json(interactive_dashboard_payload(rows))
    return f"""
    <section class="interactive-dashboard" aria-label="Interactive result explorer">
        <div class="control-bar">
            <label>Feature
                <select id="dash-feature-filter"></select>
            </label>
            <label>Model
                <select id="dash-model-filter"></select>
            </label>
            <div class="mode-switch" role="group" aria-label="Cell color mode">
                <button type="button" class="mode-button is-active" data-mode="outcome" aria-pressed="true">P2 outcome</button>
                <button type="button" class="mode-button" data-mode="visibility" aria-pressed="false">P1 visibility</button>
            </div>
        </div>
        <div id="dash-live-stats" class="live-stats"></div>
        <div class="viz-layout">
            <section class="matrix-panel" aria-labelledby="dash-matrix-title">
                <div class="section-heading">
                    <h2 id="dash-matrix-title">Task matrix</h2>
                    <div id="dash-legend" class="legend"></div>
                </div>
                <div id="dash-matrix" class="task-matrix"></div>
            </section>
            <aside class="selected-panel" aria-live="polite">
                <h2>Selected task</h2>
                <div id="dash-selected"></div>
            </aside>
        </div>
        <div class="chart-pair">
            <section class="mini-chart" aria-labelledby="dash-model-bars-title">
                <h2 id="dash-model-bars-title">Model comparison</h2>
                <div id="dash-model-bars"></div>
            </section>
            <section class="mini-chart" aria-labelledby="dash-true-bars-title">
                <h2 id="dash-true-bars-title">True value breakdown</h2>
                <div id="dash-true-bars"></div>
            </section>
        </div>
        <script id="dash-data" type="application/json">{payload}</script>
    </section>
    """


def dashboard_css() -> str:
    return """
    :root {
        color-scheme: light;
        --bg: #f5f7fa;
        --surface: #ffffff;
        --surface-soft: #eef2f6;
        --border: #d8dee7;
        --text: #15202b;
        --muted: #5f6f82;
        --accent: #2f6f9f;
        --correct: #248a45;
        --wrong: #bf3d3a;
        --inconclusive: #c88a1d;
        --na: #9aa4b2;
        --shadow: 0 14px 40px rgba(21, 32, 43, 0.08);
    }
    * { box-sizing: border-box; }
    body {
        margin: 0;
        padding: 24px;
        background: var(--bg);
        color: var(--text);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, Helvetica, sans-serif;
        font-size: 14px;
        line-height: 1.45;
    }
    main { max-width: 1360px; margin: 0 auto; }
    header {
        display: flex;
        justify-content: space-between;
        gap: 18px;
        align-items: center;
        margin-bottom: 16px;
        padding: 0 2px;
    }
    header > div {
        min-width: 0;
    }
    h1 {
        margin: 0;
        font-size: 25px;
        font-weight: 500;
    }
    .subtitle { color: var(--muted); margin-top: 6px; }
    .open-csv {
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text);
        border-radius: 8px;
        padding: 8px 12px;
        text-decoration: none;
        white-space: nowrap;
        font-weight: 500;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
        margin: 16px 0;
    }
    .metric-card {
        border: 1px solid var(--border);
        background: var(--surface);
        border-radius: 8px;
        padding: 12px;
    }
    .metric-name {
        color: var(--muted);
        font-size: 12px;
        text-transform: uppercase;
    }
    .metric-value { font-size: 19px; font-weight: 500; margin-top: 4px; }
    .interactive-dashboard {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        box-shadow: var(--shadow);
        padding: 14px;
        margin: 16px 0;
    }
    .control-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: end;
        gap: 10px;
        margin-bottom: 12px;
    }
    .control-bar label {
        display: grid;
        gap: 5px;
        min-width: 190px;
        color: var(--muted);
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
    }
    select,
    input[type="search"] {
        width: 100%;
        border: 1px solid var(--border);
        background: var(--surface);
        color: var(--text);
        border-radius: 8px;
        padding: 8px 10px;
        font: inherit;
    }
    .mode-switch {
        display: inline-flex;
        border: 1px solid var(--border);
        background: var(--surface-soft);
        border-radius: 8px;
        padding: 3px;
        gap: 3px;
    }
    .mode-button {
        appearance: none;
        border: 0;
        border-radius: 6px;
        background: transparent;
        color: var(--muted);
        padding: 8px 10px;
        font: inherit;
        cursor: pointer;
    }
    .mode-button.is-active {
        background: var(--surface);
        color: var(--text);
        box-shadow: 0 1px 4px rgba(21, 32, 43, 0.12);
    }
    .live-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 8px;
        margin: 10px 0 14px 0;
    }
    .live-stat {
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 10px;
        background: var(--surface-soft);
    }
    .live-stat span {
        display: block;
        color: var(--muted);
        font-size: 12px;
    }
    .live-stat strong {
        display: block;
        margin-top: 3px;
        font-size: 17px;
        font-weight: 500;
    }
    .viz-layout {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(260px, 340px);
        gap: 14px;
        align-items: start;
    }
    .matrix-panel,
    .selected-panel,
    .mini-chart {
        border: 1px solid var(--border);
        border-radius: 8px;
        background: var(--surface);
        padding: 12px;
    }
    .section-heading {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 10px;
    }
    h2 {
        margin: 0;
        font-size: 15px;
        font-weight: 500;
    }
    .legend {
        display: flex;
        flex-wrap: wrap;
        gap: 8px 12px;
        color: var(--muted);
        font-size: 12px;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        white-space: nowrap;
    }
    .swatch {
        width: 10px;
        height: 10px;
        border-radius: 2px;
        display: inline-block;
        border: 1px solid rgba(21, 32, 43, 0.16);
    }
    .feature-block + .feature-block {
        margin-top: 16px;
        padding-top: 14px;
        border-top: 1px solid var(--border);
    }
    .feature-title {
        color: var(--muted);
        font-size: 12px;
        font-weight: 500;
        margin-bottom: 8px;
        text-transform: uppercase;
    }
    .matrix-row {
        display: grid;
        grid-template-columns: minmax(120px, 170px) minmax(0, 1fr);
        gap: 10px;
        align-items: start;
        margin: 7px 0;
    }
    .matrix-model {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        color: var(--text);
    }
    .matrix-model small {
        display: block;
        color: var(--muted);
        font-size: 11px;
    }
    .cell-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 3px;
        min-height: 16px;
    }
    .task-cell {
        appearance: none;
        width: 13px;
        height: 13px;
        min-width: 13px;
        border: 1px solid rgba(21, 32, 43, 0.12);
        border-radius: 3px;
        padding: 0;
        cursor: pointer;
    }
    .task-cell.is-selected {
        outline: 2px solid var(--accent);
        outline-offset: 1px;
    }
    .status-correct,
    .status-yes { background: var(--correct); }
    .status-wrong,
    .status-no { background: var(--wrong); }
    .status-inconclusive { background: var(--inconclusive); }
    .status-na { background: var(--na); }
    .status-empty { background: var(--surface-soft); }
    .selected-empty {
        color: var(--muted);
        margin: 0;
    }
    .selected-photo {
        display: block;
        width: 100%;
        max-height: 210px;
        object-fit: contain;
        background: var(--surface-soft);
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .selected-list {
        display: grid;
        gap: 8px;
        margin: 0;
    }
    .selected-list div {
        display: grid;
        gap: 2px;
    }
    .selected-list dt {
        color: var(--muted);
        font-size: 12px;
    }
    .selected-list dd {
        margin: 0;
        word-break: break-word;
    }
    .chart-pair {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
        margin-top: 14px;
    }
    .bar-list {
        display: grid;
        gap: 10px;
        margin-top: 10px;
    }
    .bar-row {
        display: grid;
        grid-template-columns: minmax(120px, 180px) minmax(0, 1fr) auto;
        gap: 10px;
        align-items: center;
    }
    .bar-label {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .bar-track {
        display: flex;
        height: 16px;
        overflow: hidden;
        border-radius: 999px;
        background: var(--surface-soft);
        border: 1px solid var(--border);
    }
    .bar-segment {
        min-width: 0;
    }
    .bar-value {
        color: var(--muted);
        font-size: 12px;
        white-space: nowrap;
    }
    .panel {
        border: 1px solid var(--border);
        background: var(--surface);
        border-radius: 8px;
        margin: 12px 0;
        overflow: hidden;
    }
    .panel > summary {
        cursor: pointer;
        padding: 11px 13px;
        font-weight: 500;
        background: var(--surface);
    }
    .panel[open] > summary {
        border-bottom: 1px solid var(--border);
        background: var(--surface-soft);
    }
    .panel > summary span {
        color: var(--muted);
        font-weight: 400;
        margin-left: 8px;
        font-size: 12px;
    }
    .table-tools { padding: 10px 12px 0 12px; }
    .table-wrap {
        overflow: auto;
        max-height: 620px;
        padding: 10px 12px 12px 12px;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        font-size: 12px;
    }
    th, td {
        border-bottom: 1px solid var(--border);
        padding: 7px 8px;
        text-align: left;
        vertical-align: top;
    }
    thead th {
        position: sticky;
        top: 0;
        background: var(--surface-soft);
        z-index: 1;
    }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .vertical-table th { width: 230px; background: var(--surface-soft); }
    .metadata-table th { width: 170px; background: var(--surface-soft); }
    .empty-note { padding: 12px; color: var(--muted); }
    @media (prefers-color-scheme: dark) {
        :root {
            color-scheme: dark;
            --bg: #151a1f;
            --surface: #20262d;
            --surface-soft: #2a323b;
            --border: #39434f;
            --text: #edf2f7;
            --muted: #a9b4c0;
            --accent: #7cb7df;
            --correct: #49a866;
            --wrong: #d1605a;
            --inconclusive: #d4a037;
            --na: #788392;
            --shadow: none;
        }
    }
    @media (max-width: 1080px) {
        .viz-layout,
        .chart-pair {
            grid-template-columns: 1fr;
        }
    }
    @media (max-width: 900px) {
        body { padding: 16px; }
        header { display: block; }
        .open-csv {
            display: inline-block;
            margin-top: 12px;
        }
        .matrix-row,
        .bar-row {
            grid-template-columns: 1fr;
        }
        .matrix-model,
        .bar-label {
            white-space: normal;
        }
        .control-bar label {
            min-width: min(100%, 260px);
        }
    }
    """


def dashboard_script() -> str:
    return """
    document.querySelectorAll("[data-search-target]").forEach((input) => {
        input.addEventListener("input", () => {
            const table = document.getElementById(input.dataset.searchTarget);
            if (!table) return;
            const query = input.value.trim().toLowerCase();
            table.querySelectorAll("tbody tr").forEach((row) => {
                row.style.display = row.textContent.toLowerCase().includes(query) ? "" : "none";
            });
        });
    });

    (() => {
        const dataElement = document.getElementById("dash-data");
        if (!dataElement) return;
        const data = JSON.parse(dataElement.textContent || "{}");
        const tasks = Array.isArray(data.tasks) ? data.tasks : [];
        const models = Array.isArray(data.models) ? data.models : [];
        const features = Array.isArray(data.features) ? data.features : [];
        const taskById = new Map(tasks.map((task) => [task.id, task]));
        const state = {
            feature: "__all__",
            model: "__all__",
            mode: "outcome",
            selectedId: tasks.length ? tasks[0].id : null,
        };
        const els = {
            featureFilter: document.getElementById("dash-feature-filter"),
            modelFilter: document.getElementById("dash-model-filter"),
            stats: document.getElementById("dash-live-stats"),
            legend: document.getElementById("dash-legend"),
            matrix: document.getElementById("dash-matrix"),
            selected: document.getElementById("dash-selected"),
            modelBars: document.getElementById("dash-model-bars"),
            trueBars: document.getElementById("dash-true-bars"),
        };

        function addOption(select, value, label) {
            if (!select) return;
            const option = document.createElement("option");
            option.value = value;
            option.textContent = label;
            select.appendChild(option);
        }

        addOption(els.featureFilter, "__all__", "All features");
        features.forEach((feature) => addOption(els.featureFilter, feature.id, feature.label));
        addOption(els.modelFilter, "__all__", "All models");
        models.forEach((model) => addOption(els.modelFilter, model.id, model.label));

        function filteredTasks(options = {}) {
            const feature = options.feature === undefined ? state.feature : options.feature;
            const model = options.model === undefined ? state.model : options.model;
            return tasks.filter((task) => {
                const featureOk = feature === "__all__" || task.feature === feature;
                const modelOk = model === "__all__" || task.model === model;
                return featureOk && modelOk;
            });
        }

        function countsFor(items) {
            const counts = {
                n: items.length,
                yes: 0,
                no: 0,
                visibilityInconclusive: 0,
                correct: 0,
                wrong: 0,
                inconclusive: 0,
                na: 0,
            };
            items.forEach((task) => {
                if (task.p1 === "YES") counts.yes += 1;
                else if (task.p1 === "NO") counts.no += 1;
                else counts.visibilityInconclusive += 1;

                if (task.outcome === "CORRECT") counts.correct += 1;
                else if (task.outcome === "WRONG") counts.wrong += 1;
                else if (task.outcome === "INCONCLUSIVE") counts.inconclusive += 1;
                else counts.na += 1;
            });
            counts.applicable = counts.correct + counts.wrong + counts.inconclusive;
            counts.committed = counts.correct + counts.wrong;
            return counts;
        }

        function pct(count, total) {
            if (!total) return "NA";
            return `${Math.round((count / total) * 100)}%`;
        }

        function ratio(count, total) {
            if (!total) return "NA";
            return `${count}/${total} (${pct(count, total)})`;
        }

        function empty(container, text) {
            if (!container) return;
            container.textContent = "";
            const p = document.createElement("p");
            p.className = "selected-empty";
            p.textContent = text;
            container.appendChild(p);
        }

        function statusClass(task) {
            if (state.mode === "visibility") {
                if (task.p1 === "YES") return "status-yes";
                if (task.p1 === "NO") return "status-no";
                return "status-inconclusive";
            }
            if (task.outcome === "CORRECT") return "status-correct";
            if (task.outcome === "WRONG") return "status-wrong";
            if (task.outcome === "INCONCLUSIVE") return "status-inconclusive";
            return "status-na";
        }

        function statusLabel(task) {
            return state.mode === "visibility" ? task.p1 : task.outcome;
        }

        function legendItems() {
            if (state.mode === "visibility") {
                return [
                    ["status-yes", "YES"],
                    ["status-no", "NO"],
                    ["status-inconclusive", "INCONCLUSIVE"],
                ];
            }
            return [
                ["status-correct", "CORRECT"],
                ["status-wrong", "WRONG"],
                ["status-inconclusive", "INCONCLUSIVE"],
                ["status-na", "N/A"],
            ];
        }

        function renderLegend() {
            if (!els.legend) return;
            els.legend.textContent = "";
            legendItems().forEach(([klass, label]) => {
                const item = document.createElement("span");
                item.className = "legend-item";
                const swatch = document.createElement("span");
                swatch.className = `swatch ${klass}`;
                item.appendChild(swatch);
                item.appendChild(document.createTextNode(label));
                els.legend.appendChild(item);
            });
        }

        function renderStats() {
            if (!els.stats) return;
            const visible = filteredTasks();
            const counts = countsFor(visible);
            const statRows = [
                ["Rows", String(counts.n)],
                ["P1 YES", ratio(counts.yes, counts.n)],
                ["P2 correct", ratio(counts.correct, counts.applicable)],
                ["Wrong", ratio(counts.wrong, counts.applicable)],
                ["Committed accuracy", ratio(counts.correct, counts.committed)],
            ];
            els.stats.textContent = "";
            statRows.forEach(([label, value]) => {
                const card = document.createElement("div");
                card.className = "live-stat";
                const name = document.createElement("span");
                name.textContent = label;
                const metric = document.createElement("strong");
                metric.textContent = value;
                card.append(name, metric);
                els.stats.appendChild(card);
            });
        }

        function renderMatrix() {
            if (!els.matrix) return;
            els.matrix.textContent = "";
            const visible = filteredTasks();
            if (!visible.length) {
                empty(els.matrix, "No tasks match the current filters.");
                return;
            }
            const featureList = features.filter((feature) => {
                return (state.feature === "__all__" || state.feature === feature.id)
                    && visible.some((task) => task.feature === feature.id);
            });
            const modelList = models.filter((model) => {
                return (state.model === "__all__" || state.model === model.id)
                    && visible.some((task) => task.model === model.id);
            });
            featureList.forEach((feature) => {
                const block = document.createElement("section");
                block.className = "feature-block";
                const title = document.createElement("div");
                title.className = "feature-title";
                title.textContent = feature.label;
                block.appendChild(title);
                modelList.forEach((model) => {
                    const rowTasks = visible.filter((task) => task.feature === feature.id && task.model === model.id);
                    if (!rowTasks.length) return;
                    const row = document.createElement("div");
                    row.className = "matrix-row";
                    const label = document.createElement("div");
                    label.className = "matrix-model";
                    label.textContent = model.label;
                    const small = document.createElement("small");
                    small.textContent = `${rowTasks.length} tasks`;
                    label.appendChild(small);
                    const strip = document.createElement("div");
                    strip.className = "cell-strip";
                    rowTasks.forEach((task) => {
                        const button = document.createElement("button");
                        button.type = "button";
                        button.className = `task-cell ${statusClass(task)}`;
                        if (task.id === state.selectedId) button.classList.add("is-selected");
                        button.title = `${model.label} / ${feature.label}: ${statusLabel(task)}`;
                        button.setAttribute(
                            "aria-label",
                            `${model.label}, ${feature.label}, observation ${task.observationId || "unknown"}, ${statusLabel(task)}`
                        );
                        button.addEventListener("click", () => {
                            state.selectedId = task.id;
                            refresh();
                        });
                        strip.appendChild(button);
                    });
                    row.append(label, strip);
                    block.appendChild(row);
                });
                els.matrix.appendChild(block);
            });
        }

        function segmentDefinitions(counts) {
            if (state.mode === "visibility") {
                return [
                    ["yes", counts.yes, "status-yes", "YES"],
                    ["no", counts.no, "status-no", "NO"],
                    ["visibilityInconclusive", counts.visibilityInconclusive, "status-inconclusive", "INCONCLUSIVE"],
                ];
            }
            return [
                ["correct", counts.correct, "status-correct", "CORRECT"],
                ["wrong", counts.wrong, "status-wrong", "WRONG"],
                ["inconclusive", counts.inconclusive, "status-inconclusive", "INCONCLUSIVE"],
                ["na", counts.na, "status-na", "N/A"],
            ];
        }

        function appendStackedBar(container, labelText, items, valueText) {
            const counts = countsFor(items);
            const row = document.createElement("div");
            row.className = "bar-row";
            const label = document.createElement("div");
            label.className = "bar-label";
            label.title = labelText;
            label.textContent = labelText;
            const track = document.createElement("div");
            track.className = "bar-track";
            segmentDefinitions(counts).forEach(([, count, klass, segmentLabel]) => {
                if (!count || !counts.n) return;
                const segment = document.createElement("span");
                segment.className = `bar-segment ${klass}`;
                segment.style.width = `${(count / counts.n) * 100}%`;
                segment.title = `${segmentLabel}: ${count}/${counts.n}`;
                track.appendChild(segment);
            });
            const value = document.createElement("div");
            value.className = "bar-value";
            value.textContent = valueText(counts);
            row.append(label, track, value);
            container.appendChild(row);
        }

        function renderModelBars() {
            if (!els.modelBars) return;
            els.modelBars.textContent = "";
            const list = document.createElement("div");
            list.className = "bar-list";
            const modelList = models.filter((model) => state.model === "__all__" || state.model === model.id);
            const rows = modelList
                .map((model) => ({
                    model,
                    items: filteredTasks({model: model.id}),
                }))
                .filter((row) => row.items.length)
                .sort((a, b) => {
                    const ac = countsFor(a.items);
                    const bc = countsFor(b.items);
                    const av = state.mode === "visibility" ? ac.yes / Math.max(ac.n, 1) : ac.correct / Math.max(ac.applicable, 1);
                    const bv = state.mode === "visibility" ? bc.yes / Math.max(bc.n, 1) : bc.correct / Math.max(bc.applicable, 1);
                    return bv - av;
                });
            if (!rows.length) {
                empty(els.modelBars, "No model rows to compare.");
                return;
            }
            rows.forEach(({model, items}) => {
                appendStackedBar(
                    list,
                    model.label,
                    items,
                    (counts) => state.mode === "visibility"
                        ? `YES ${pct(counts.yes, counts.n)}`
                        : `P2 ${pct(counts.correct, counts.applicable)}`
                );
            });
            els.modelBars.appendChild(list);
        }

        function renderTrueBars() {
            if (!els.trueBars) return;
            els.trueBars.textContent = "";
            const visible = filteredTasks().filter((task) => task.trueValue);
            if (!visible.length) {
                empty(els.trueBars, "No true-value rows match the current filters.");
                return;
            }
            const groups = new Map();
            visible.forEach((task) => {
                if (!groups.has(task.trueValue)) groups.set(task.trueValue, []);
                groups.get(task.trueValue).push(task);
            });
            const rows = Array.from(groups.entries())
                .sort((a, b) => b[1].length - a[1].length || a[0].localeCompare(b[0]))
                .slice(0, 12);
            const list = document.createElement("div");
            list.className = "bar-list";
            rows.forEach(([trueValue, items]) => {
                appendStackedBar(
                    list,
                    trueValue,
                    items,
                    (counts) => state.mode === "visibility"
                        ? `${counts.n} rows`
                        : `${counts.correct}/${Math.max(counts.applicable, 0)} correct`
                );
            });
            els.trueBars.appendChild(list);
        }

        function renderSelected() {
            if (!els.selected) return;
            let selected = taskById.get(state.selectedId);
            const visible = filteredTasks();
            if (!selected || !visible.some((task) => task.id === selected.id)) {
                selected = visible[0] || null;
                state.selectedId = selected ? selected.id : null;
            }
            els.selected.textContent = "";
            if (!selected) {
                empty(els.selected, "Select a task cell to inspect it.");
                return;
            }
            if (selected.photoUrl) {
                const link = document.createElement("a");
                link.href = selected.photoUrl;
                link.target = "_blank";
                link.rel = "noreferrer";
                const img = document.createElement("img");
                img.className = "selected-photo";
                img.src = selected.photoUrl;
                img.alt = selected.species || "Observation photo";
                link.appendChild(img);
                els.selected.appendChild(link);
            }
            const dl = document.createElement("dl");
            dl.className = "selected-list";
            [
                ["Model", selected.modelLabel],
                ["Feature", selected.featureLabel],
                ["Species", selected.species || "unknown"],
                ["Observation", selected.observationId || "unknown"],
                ["True value", selected.trueValue || "empty"],
                ["Prediction", selected.predictedValue || "empty"],
                ["P1", selected.p1],
                ["Outcome", selected.outcome],
            ].forEach(([term, value]) => {
                const group = document.createElement("div");
                const dt = document.createElement("dt");
                dt.textContent = term;
                const dd = document.createElement("dd");
                dd.textContent = value;
                group.append(dt, dd);
                dl.appendChild(group);
            });
            els.selected.appendChild(dl);
        }

        function refresh() {
            renderLegend();
            renderStats();
            renderMatrix();
            renderSelected();
            renderModelBars();
            renderTrueBars();
        }

        if (els.featureFilter) {
            els.featureFilter.addEventListener("change", () => {
                state.feature = els.featureFilter.value;
                refresh();
            });
        }
        if (els.modelFilter) {
            els.modelFilter.addEventListener("change", () => {
                state.model = els.modelFilter.value;
                refresh();
            });
        }
        document.querySelectorAll(".mode-button").forEach((button) => {
            button.addEventListener("click", () => {
                state.mode = button.dataset.mode || "outcome";
                document.querySelectorAll(".mode-button").forEach((item) => {
                    const active = item === button;
                    item.classList.toggle("is-active", active);
                    item.setAttribute("aria-pressed", active ? "true" : "false");
                });
                refresh();
            });
        });
        refresh();
    })();
    """


def dashboard_html_document(
    *,
    raw: pd.DataFrame,
    dashboard_view: pd.DataFrame,
    focus_summary_view: pd.DataFrame,
    summary_view: pd.DataFrame,
    by_true_value_view: pd.DataFrame,
    pairs: pd.DataFrame,
    rows_view: pd.DataFrame,
    definitions: pd.DataFrame,
    metadata: dict[str, object],
    output_file: Path,
    heatmap_images: dict[str, str] | None = None,
    focus_images: dict[str, dict[str, str]] | None = None,
) -> str:
    trial_id = str(metadata.get("trial_id") or output_file.stem)
    csv_href = escape(output_file.name)
    rows = annotate_results(raw)
    interactive = interactive_dashboard_html(rows)
    sections = [
        metadata_html(metadata, output_file),
        f"""
        <details class="panel">
            <summary>Artifact CSV Row Example</summary>
            <div class="table-wrap">{artifact_row_example_html(raw)}</div>
        </details>
        """,
        table_section_html("Model Summary", focus_summary_view, "focus-summary-table"),
        table_section_html("Per-Feature / Model Summary", summary_view, "summary-table"),
        table_section_html("Outcome by True Value", by_true_value_view, "true-value-table"),
        table_section_html("Outcome Pairs: True Value x Predicted Value", pairs, "pairs-table"),
        table_section_html(f"Per-Trial Rows (First {DISPLAY_ROW_LIMIT})", rows_view, "rows-table"),
        table_section_html("Metric Definitions", definitions, "definitions-table"),
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Stepwise Trial Dashboard - {escape(trial_id)}</title>
    <style>{dashboard_css()}</style>
</head>
<body>
<main>
    <header>
        <div>
            <h1>Stepwise Trial Dashboard</h1>
            <div class="subtitle">{escape(trial_id)}</div>
        </div>
        <a class="open-csv" href="{csv_href}">Open CSV</a>
    </header>
    {dashboard_cards_html(dashboard_view)}
    {interactive}
    {''.join(sections)}
</main>
<script>{dashboard_script()}</script>
</body>
</html>
"""


def dashboard_html_path(output_file: Path) -> Path:
    return output_file.with_name(f"{output_file.stem}.dashboard.html")


def write_dashboard_html(
    *,
    raw: pd.DataFrame,
    dashboard_view: pd.DataFrame,
    focus_summary_view: pd.DataFrame,
    summary_view: pd.DataFrame,
    by_true_value_view: pd.DataFrame,
    pairs: pd.DataFrame,
    rows_view: pd.DataFrame,
    definitions: pd.DataFrame,
    metadata: dict[str, object],
    output_file: Path,
    heatmap_images: dict[str, str] | None = None,
    focus_images: dict[str, dict[str, str]] | None = None,
) -> Path:
    html_file = dashboard_html_path(output_file)
    html_file.write_text(
        dashboard_html_document(
            raw=raw,
            dashboard_view=dashboard_view,
            focus_summary_view=focus_summary_view,
            summary_view=summary_view,
            by_true_value_view=by_true_value_view,
            pairs=pairs,
            rows_view=rows_view,
            definitions=definitions,
            metadata=metadata,
            output_file=output_file,
            heatmap_images=heatmap_images,
            focus_images=focus_images,
        ),
        encoding="utf-8",
    )
    return html_file


def dashboard_from_csv(csv_path: str | Path) -> Path:
    output_file = Path(csv_path).expanduser()
    if not output_file.is_absolute():
        output_file = ROOT / output_file
    raw = pd.read_csv(output_file)
    rows = annotate_results(raw)
    dashboard = whole_experiment_dashboard(rows)
    summary = summarize(rows, FEATURE_GROUP_COLUMNS)
    by_true_value = outcome_by_true_value(rows)
    pairs = outcome_pairs(rows)
    definitions = metric_definitions()
    scale_view = experiment_scale_view(rows)
    metadata_file = output_file.with_suffix(".metadata.json")
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {"trial_id": output_file.stem}
    dashboard_view = compact_dashboard_view(dashboard, scale_view, metadata)
    focus_summary_view = model_focus_summary_view(summary)
    summary_view = compact_summary_view(summary)
    by_true_value_view = compact_outcome_by_true_value_view(by_true_value)
    rows_view = compact_rows_view(rows)
    return write_dashboard_html(
        raw=raw,
        dashboard_view=dashboard_view,
        focus_summary_view=focus_summary_view,
        summary_view=summary_view,
        by_true_value_view=by_true_value_view,
        pairs=pairs,
        rows_view=rows_view,
        definitions=definitions,
        metadata=metadata,
        output_file=output_file,
    )


def run_stepwise_notebook_trial(
    *,
    model: str | Iterable[str],
    mode: str,
    image_set: str,
    data_split: str,
    sample_limit: int | str,
    features: str | Iterable[str],
    prompt_set: str,
    temperature: float,
    max_tokens: int,
    workers: int,
    timeout: int,
    run_id: str,
    trial_id: str,
    out_file: str = "",
) -> dict[str, object]:
    trial_id = trial_id or f"notebook-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"
    cmd = [
        uv_executable(),
        "run",
        "python",
        "scripts/run_stepwise_trial.py",
        "--model",
        model_arg(model),
        "--mode",
        mode,
        "--image-set",
        image_set,
        "--data-split",
        data_split,
        "--sample-limit",
        str(sample_limit),
        "--features",
        feature_arg(features),
        "--prompt-set",
        prompt_set,
        "--temperature",
        str(temperature),
        "--max-tokens",
        str(max_tokens),
        "--workers",
        str(workers),
        "--timeout",
        str(timeout),
        "--run-id",
        run_id,
        "--trial-id",
        trial_id,
    ]
    if out_file:
        cmd.extend(["--out-file", out_file])

    run_trial_command(cmd)

    metadata_file = metadata_path_for(trial_id)
    metadata = json.loads(metadata_file.read_text())
    output_file = Path(metadata["output_file"])
    if not output_file.is_absolute():
        output_file = ROOT / output_file

    raw = pd.read_csv(output_file)
    rows = annotate_results(raw)
    dashboard = whole_experiment_dashboard(rows)
    summary = summarize(rows, FEATURE_GROUP_COLUMNS)
    by_true_value = outcome_by_true_value(rows)
    pairs = outcome_pairs(rows)
    definitions = metric_definitions()
    scale_view = experiment_scale_view(rows)
    dashboard_view = compact_dashboard_view(dashboard, scale_view, metadata)
    focus_summary_view = model_focus_summary_view(summary)
    summary_view = compact_summary_view(summary)
    by_true_value_view = compact_outcome_by_true_value_view(by_true_value)
    rows_view = compact_rows_view(rows)
    usage_model_view = openrouter_usage_by_model_view(metadata)
    heatmap_images = benchmark_heatmap_images(rows)
    focus_images = model_focus_heatmap_images(rows)
    html_file = write_dashboard_html(
        raw=raw,
        dashboard_view=dashboard_view,
        focus_summary_view=focus_summary_view,
        summary_view=summary_view,
        by_true_value_view=by_true_value_view,
        pairs=pairs,
        rows_view=rows_view,
        definitions=definitions,
        metadata=metadata,
        output_file=output_file,
        heatmap_images=heatmap_images,
        focus_images=focus_images,
    )

    display(Markdown(f"### Trial saved to `{path_text(output_file)}`"))
    display(Markdown(f"### Interactive HTML dashboard saved to `{path_text(html_file)}`"))
    display(HTML(f'<p><a href="{html_file.as_uri()}" target="_blank">Open interactive HTML dashboard</a></p>'))
    display(Markdown("### Artifact CSV Row Example"))
    display_artifact_row_example(raw)
    display(Markdown("### Whole-Experiment Dashboard"))
    display_benchmark_heatmap(rows, heatmap_images)
    display(dashboard_view)
    if not usage_model_view.empty:
        display(Markdown("### OpenRouter Usage by Model"))
        display(usage_model_view)
    display(Markdown("### Model Focus View"))
    display(focus_summary_view)
    display_model_focus_heatmaps(rows, focus_images)
    display(Markdown("### Per-Feature / Model Summary (Compact)"))
    display(summary_view)
    display(Markdown("### Outcome by True Value"))
    display(by_true_value_view)
    display(Markdown("### Outcome Pairs: True Value x Predicted Value"))
    display(pairs)
    display(Markdown(f"### Per-Trial Rows (First {DISPLAY_ROW_LIMIT})"))
    display(rows_view)
    display(Markdown("### Metric Definitions"))
    display(definitions)
    return {
        "rows": rows,
        "summary": summary,
        "dashboard": dashboard,
        "outcome_by_true_value": by_true_value,
        "outcome_pairs": pairs,
        "metric_definitions": definitions,
        "scale_view": scale_view,
        "rows_view": rows_view,
        "usage_model_view": usage_model_view,
        "focus_summary_view": focus_summary_view,
        "summary_view": summary_view,
        "dashboard_view": dashboard_view,
        "outcome_by_true_value_view": by_true_value_view,
        "metadata": metadata,
        "output_file": output_file,
        "html_file": html_file,
    }
