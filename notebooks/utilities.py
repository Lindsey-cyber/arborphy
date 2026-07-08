from __future__ import annotations

import base64
from html import escape
from io import BytesIO
import json
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
        text = line.strip()
        if not text:
            continue

        if text.startswith("pending feature tasks:"):
            show_progress(text)
        elif text.startswith(("expected model calls:", "already completed tasks:", "tasks without true value:")):
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
    completion_rate = completed_rows / expected_rows if expected_rows else None

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
    ]
    return pd.DataFrame(records)


def scale_metric_value(scale_view: pd.DataFrame, metric: str) -> str:
    if scale_view.empty:
        return ""
    matches = scale_view.loc[scale_view["metric"] == metric, "value"]
    if matches.empty:
        return ""
    return str(matches.iloc[0])


def compact_dashboard_view(dashboard: pd.DataFrame, scale_view: pd.DataFrame | None = None) -> pd.DataFrame:
    if dashboard.empty:
        return dashboard
    scale_view = scale_view if scale_view is not None else pd.DataFrame()
    row = dashboard.iloc[0]
    feature_count = row.get("feature_count")
    p2_applicable_count = row.get("p2_applicable_count")
    committed_count = row.get("committed_count")
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
            "value": scale_metric_value(scale_view, "estimated_model_calls_completed"),
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
    <details class="panel" open>
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
    cards = []
    for _, row in dashboard_view.iterrows():
        cards.append(
            f"""
            <div class="metric-card">
                <div class="metric-name">{escape(str(row.get("metric", "")))}</div>
                <div class="metric-value">{escape(str(row.get("value", "")))}</div>
            </div>
            """
        )
    return f'<div class="metric-grid">{"".join(cards)}</div>'


def metadata_html(metadata: dict[str, object], output_file: Path) -> str:
    if not metadata:
        metadata = {}
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
    }
    body = "".join(
        f"<tr><th>{escape(key)}</th><td>{escape(str(value))}</td></tr>"
        for key, value in rows.items()
        if str(value)
    )
    return f"""
    <details class="panel" open>
        <summary>Run Metadata</summary>
        <table class="metadata-table"><tbody>{body}</tbody></table>
    </details>
    """


def dashboard_css() -> str:
    return """
    :root {
        color-scheme: light;
        --border: #d9dee7;
        --soft: #f7f9fc;
        --text: #111827;
        --muted: #64748b;
        --accent: #2563eb;
    }
    * { box-sizing: border-box; }
    body {
        margin: 0;
        padding: 28px;
        background: #ffffff;
        color: var(--text);
        font-family: Arial, Helvetica, sans-serif;
        font-size: 14px;
    }
    main { max-width: 1480px; margin: 0 auto; }
    header {
        display: flex;
        justify-content: space-between;
        gap: 18px;
        align-items: flex-end;
        border-bottom: 1px solid var(--border);
        padding-bottom: 14px;
        margin-bottom: 18px;
    }
    h1 {
        margin: 0;
        font: 800 28px/1.15 "DejaVu Sans", Arial, sans-serif;
        letter-spacing: 0;
    }
    .subtitle { color: var(--muted); margin-top: 6px; }
    .open-csv {
        color: var(--accent);
        text-decoration: none;
        white-space: nowrap;
        font-weight: 700;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 10px;
        margin: 16px 0;
    }
    .metric-card {
        border: 1px solid var(--border);
        background: var(--soft);
        padding: 10px 12px;
    }
    .metric-name {
        color: var(--muted);
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: .04em;
    }
    .metric-value { font-size: 18px; font-weight: 800; margin-top: 4px; }
    .chart-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(420px, 1fr));
        gap: 22px;
        align-items: start;
        overflow-x: auto;
        margin: 18px 0;
    }
    .chart-card {
        margin: 0;
        background: #fff;
        border: 1px solid var(--border);
        padding: 10px;
    }
    .chart-card img { display: block; width: 100%; height: auto; }
    .panel {
        border: 1px solid var(--border);
        background: #fff;
        margin: 14px 0;
    }
    .panel > summary {
        cursor: pointer;
        padding: 11px 13px;
        font-weight: 800;
        background: var(--soft);
        border-bottom: 1px solid var(--border);
    }
    .panel > summary span {
        color: var(--muted);
        font-weight: 500;
        margin-left: 8px;
        font-size: 12px;
    }
    .table-tools { padding: 10px 12px 0 12px; }
    input[type="search"] {
        width: min(420px, 100%);
        border: 1px solid var(--border);
        padding: 7px 9px;
        font: 13px Arial, Helvetica, sans-serif;
    }
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
        border-bottom: 1px solid #e5e7eb;
        padding: 7px 8px;
        text-align: left;
        vertical-align: top;
    }
    thead th {
        position: sticky;
        top: 0;
        background: #eef2f7;
        z-index: 1;
    }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .vertical-table th { width: 230px; background: var(--soft); }
    .metadata-table th { width: 170px; background: var(--soft); }
    .empty-note { padding: 12px; color: var(--muted); }
    @media (max-width: 900px) {
        body { padding: 16px; }
        header { display: block; }
        .chart-grid { grid-template-columns: 1fr; }
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
    """


def dashboard_html_document(
    *,
    raw: pd.DataFrame,
    dashboard_view: pd.DataFrame,
    summary_view: pd.DataFrame,
    by_true_value_view: pd.DataFrame,
    pairs: pd.DataFrame,
    rows_view: pd.DataFrame,
    definitions: pd.DataFrame,
    metadata: dict[str, object],
    output_file: Path,
    heatmap_images: dict[str, str],
) -> str:
    trial_id = str(metadata.get("trial_id") or output_file.stem)
    csv_href = escape(output_file.name)
    heatmap = benchmark_heatmap_html(heatmap_images)
    sections = [
        metadata_html(metadata, output_file),
        f"""
        <details class="panel" open>
            <summary>Artifact CSV Row Example</summary>
            <div class="table-wrap">{artifact_row_example_html(raw)}</div>
        </details>
        """,
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
    {heatmap}
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
    summary_view: pd.DataFrame,
    by_true_value_view: pd.DataFrame,
    pairs: pd.DataFrame,
    rows_view: pd.DataFrame,
    definitions: pd.DataFrame,
    metadata: dict[str, object],
    output_file: Path,
    heatmap_images: dict[str, str],
) -> Path:
    html_file = dashboard_html_path(output_file)
    html_file.write_text(
        dashboard_html_document(
            raw=raw,
            dashboard_view=dashboard_view,
            summary_view=summary_view,
            by_true_value_view=by_true_value_view,
            pairs=pairs,
            rows_view=rows_view,
            definitions=definitions,
            metadata=metadata,
            output_file=output_file,
            heatmap_images=heatmap_images,
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
    dashboard_view = compact_dashboard_view(dashboard, scale_view)
    summary_view = compact_summary_view(summary)
    by_true_value_view = compact_outcome_by_true_value_view(by_true_value)
    rows_view = compact_rows_view(rows)
    metadata_file = output_file.with_suffix(".metadata.json")
    metadata = json.loads(metadata_file.read_text()) if metadata_file.exists() else {"trial_id": output_file.stem}
    heatmap_images = benchmark_heatmap_images(rows)
    return write_dashboard_html(
        raw=raw,
        dashboard_view=dashboard_view,
        summary_view=summary_view,
        by_true_value_view=by_true_value_view,
        pairs=pairs,
        rows_view=rows_view,
        definitions=definitions,
        metadata=metadata,
        output_file=output_file,
        heatmap_images=heatmap_images,
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
    dashboard_view = compact_dashboard_view(dashboard, scale_view)
    summary_view = compact_summary_view(summary)
    by_true_value_view = compact_outcome_by_true_value_view(by_true_value)
    rows_view = compact_rows_view(rows)
    heatmap_images = benchmark_heatmap_images(rows)
    html_file = write_dashboard_html(
        raw=raw,
        dashboard_view=dashboard_view,
        summary_view=summary_view,
        by_true_value_view=by_true_value_view,
        pairs=pairs,
        rows_view=rows_view,
        definitions=definitions,
        metadata=metadata,
        output_file=output_file,
        heatmap_images=heatmap_images,
    )

    display(Markdown(f"### Trial saved to `{path_text(output_file)}`"))
    display(Markdown(f"### Interactive HTML dashboard saved to `{path_text(html_file)}`"))
    display(HTML(f'<p><a href="{html_file.as_uri()}" target="_blank">Open interactive HTML dashboard</a></p>'))
    display(Markdown("### Artifact CSV Row Example"))
    display_artifact_row_example(raw)
    display(Markdown("### Whole-Experiment Dashboard"))
    display_benchmark_heatmap(rows, heatmap_images)
    display(dashboard_view)
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
        "summary_view": summary_view,
        "dashboard_view": dashboard_view,
        "outcome_by_true_value_view": by_true_value_view,
        "metadata": metadata,
        "output_file": output_file,
        "html_file": html_file,
    }
