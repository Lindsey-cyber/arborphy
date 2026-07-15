from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import math
import threading
from typing import Any, Callable, Iterable


DEFAULT_TOTAL_COST_WARN_USD = 1.0
DEFAULT_MODEL_COST_WARN_USD = 1.0
DEFAULT_TOKEN_WARN = 1_000_000
DEFAULT_REQUEST_COST_WARN_USD = 0.05

TOKEN_FIELDS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "audio_tokens",
    "video_tokens",
    "image_tokens",
}
SUMMARY_FIELDS = [
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "cache_write_tokens",
    "audio_tokens",
    "video_tokens",
    "image_tokens",
    "cost_usd",
    "upstream_inference_cost_usd",
    "prompt_cost_usd",
    "completion_cost_usd",
]


def number_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: object) -> int | None:
    numeric = number_or_none(value)
    if numeric is None:
        return None
    return int(numeric)


def nested_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def first_number(*values: object) -> float | None:
    for value in values:
        numeric = number_or_none(value)
        if numeric is not None:
            return numeric
    return None


def format_money(value: object) -> str:
    amount = number_or_none(value)
    if amount is None:
        return ""
    if amount == 0:
        return "$0"
    if amount < 0.01:
        return f"${amount:.4f}"
    return f"${amount:.2f}"


def format_tokens(value: object) -> str:
    numeric = number_or_none(value)
    if numeric is None:
        return ""
    return f"{int(numeric):,}"


def usage_summary_from_record(record: dict[str, Any]) -> dict[str, Any]:
    usage = nested_dict(record.get("usage"))
    prompt_details = nested_dict(usage.get("prompt_tokens_details"))
    completion_details = nested_dict(usage.get("completion_tokens_details"))
    cost_details = nested_dict(usage.get("cost_details"))

    summary = {
        "requested_model": record.get("requested_model"),
        "response_model": record.get("response_model"),
        "prompt_tokens": int_or_none(usage.get("prompt_tokens")),
        "completion_tokens": int_or_none(usage.get("completion_tokens")),
        "total_tokens": int_or_none(usage.get("total_tokens")),
        "reasoning_tokens": int_or_none(
            first_number(
                completion_details.get("reasoning_tokens"),
                usage.get("reasoning_tokens"),
                record.get("reasoning_tokens"),
            )
        ),
        "cached_tokens": int_or_none(
            first_number(prompt_details.get("cached_tokens"), usage.get("cached_tokens"))
        ),
        "cache_write_tokens": int_or_none(
            first_number(prompt_details.get("cache_write_tokens"), usage.get("cache_write_tokens"))
        ),
        "audio_tokens": int_or_none(
            first_number(
                prompt_details.get("audio_tokens"),
                completion_details.get("audio_tokens"),
                usage.get("audio_tokens"),
            )
        ),
        "video_tokens": int_or_none(
            first_number(prompt_details.get("video_tokens"), usage.get("video_tokens"))
        ),
        "image_tokens": int_or_none(
            first_number(
                completion_details.get("image_tokens"),
                prompt_details.get("image_tokens"),
                usage.get("image_tokens"),
            )
        ),
        "cost_usd": first_number(
            usage.get("cost"),
            usage.get("cost_usd"),
            usage.get("total_cost"),
            usage.get("total_cost_usd"),
            record.get("total_cost"),
            record.get("usage") if isinstance(record.get("usage"), (int, float, str)) else None,
        ),
        "upstream_inference_cost_usd": first_number(
            cost_details.get("upstream_inference_cost"),
            usage.get("upstream_inference_cost"),
        ),
        "prompt_cost_usd": first_number(
            cost_details.get("upstream_inference_prompt_cost"),
            cost_details.get("prompt_cost"),
        ),
        "completion_cost_usd": first_number(
            cost_details.get("upstream_inference_completions_cost"),
            cost_details.get("upstream_inference_completion_cost"),
            cost_details.get("completion_cost"),
        ),
    }
    return summary


def _record_key(record: dict[str, Any], index: int) -> str:
    record_id = record.get("id")
    if record_id:
        return str(record_id)
    return "|".join(
        [
            str(index),
            str(record.get("created_at", "")),
            str(record.get("requested_model", "")),
            str(record.get("response_model", "")),
        ]
    )


def read_usage_log_records(usage_log_file: Path) -> list[dict[str, Any]]:
    if not usage_log_file.exists():
        return []

    records: list[dict[str, Any]] = []
    with usage_log_file.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


class UsageAccumulator:
    def __init__(self, *, model: str | None = None) -> None:
        self.model = model
        self.calls = 0
        self.sums: defaultdict[str, float] = defaultdict(float)
        self.seen_fields: set[str] = set()
        self.response_models: Counter[str] = Counter()

    def add(self, record: dict[str, Any]) -> None:
        summary = usage_summary_from_record(record)
        self.calls += 1
        response_model = summary.get("response_model")
        if response_model:
            self.response_models[str(response_model)] += 1
        for field in SUMMARY_FIELDS:
            value = number_or_none(summary.get(field))
            if value is None:
                continue
            self.sums[field] += value
            self.seen_fields.add(field)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"captured_calls": self.calls}
        if self.model is not None:
            result["model"] = self.model
        if self.response_models:
            result["response_models"] = [
                {"model": model, "calls": calls}
                for model, calls in self.response_models.most_common()
            ]
        for field in SUMMARY_FIELDS:
            if field not in self.seen_fields:
                result[field] = None
            elif field in TOKEN_FIELDS:
                result[field] = int(self.sums[field])
            else:
                result[field] = self.sums[field]

        cost = number_or_none(result.get("cost_usd"))
        total_tokens = number_or_none(result.get("total_tokens"))
        if cost is not None and self.calls:
            result["avg_cost_per_call_usd"] = cost / self.calls
        else:
            result["avg_cost_per_call_usd"] = None
        if cost is not None and total_tokens:
            result["cost_per_1k_tokens_usd"] = cost / total_tokens * 1000
        else:
            result["cost_per_1k_tokens_usd"] = None
        if total_tokens and self.calls:
            result["avg_tokens_per_call"] = total_tokens / self.calls
        else:
            result["avg_tokens_per_call"] = None
        return result


def aggregate_usage_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    overall = UsageAccumulator()
    by_model: dict[str, UsageAccumulator] = {}

    for record in records:
        overall.add(record)
        requested_model = str(record.get("requested_model") or record.get("response_model") or "unknown")
        if requested_model not in by_model:
            by_model[requested_model] = UsageAccumulator(model=requested_model)
        by_model[requested_model].add(record)

    by_model_rows = [acc.to_dict() for acc in by_model.values()]
    by_model_rows.sort(
        key=lambda row: (
            number_or_none(row.get("cost_usd")) or 0.0,
            number_or_none(row.get("total_tokens")) or 0.0,
        ),
        reverse=True,
    )
    return {"overall": overall.to_dict(), "by_model": by_model_rows}


def aggregate_usage_log_full(usage_log_file: Path) -> dict[str, Any]:
    return aggregate_usage_records(read_usage_log_records(usage_log_file))


def aggregate_usage_log(usage_log_file: Path) -> dict[str, Any]:
    return aggregate_usage_log_full(usage_log_file)["overall"]


def aggregate_usage_log_by_model(usage_log_file: Path) -> list[dict[str, Any]]:
    return aggregate_usage_log_full(usage_log_file)["by_model"]


def _next_threshold(value: object, interval: float) -> float | None:
    if interval <= 0:
        return None
    current = number_or_none(value) or 0.0
    return (math.floor(current / interval) + 1) * interval


class UsageLogMonitor:
    def __init__(
        self,
        usage_log_file: Path,
        *,
        total_cost_interval_usd: float = DEFAULT_TOTAL_COST_WARN_USD,
        model_cost_interval_usd: float = DEFAULT_MODEL_COST_WARN_USD,
        total_token_interval: int = DEFAULT_TOKEN_WARN,
        model_token_interval: int = DEFAULT_TOKEN_WARN,
        request_cost_warn_usd: float = DEFAULT_REQUEST_COST_WARN_USD,
        poll_seconds: float = 2.0,
        emit: Callable[[str], None] | None = None,
    ) -> None:
        self.usage_log_file = usage_log_file
        self.total_cost_interval_usd = total_cost_interval_usd
        self.model_cost_interval_usd = model_cost_interval_usd
        self.total_token_interval = total_token_interval
        self.model_token_interval = model_token_interval
        self.request_cost_warn_usd = request_cost_warn_usd
        self.poll_seconds = poll_seconds
        self.emit = emit or print
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen_records: set[str] = set()
        self._next_total_cost: float | None = None
        self._next_total_tokens: float | None = None
        self._next_model_cost: dict[str, float | None] = {}
        self._next_model_tokens: dict[str, float | None] = {}

    def start(self) -> None:
        self._prime()
        self._thread = threading.Thread(target=self._run, name="openrouter-usage-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.poll()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.poll_seconds * 2, 1.0))
        self.poll()

    def _prime(self) -> None:
        records = read_usage_log_records(self.usage_log_file)
        self._seen_records = {_record_key(record, index) for index, record in enumerate(records)}
        aggregate = aggregate_usage_records(records)
        overall = aggregate["overall"]
        self._next_total_cost = _next_threshold(overall.get("cost_usd"), self.total_cost_interval_usd)
        self._next_total_tokens = _next_threshold(overall.get("total_tokens"), float(self.total_token_interval))
        for row in aggregate["by_model"]:
            model = str(row.get("model") or "unknown")
            self._next_model_cost[model] = _next_threshold(row.get("cost_usd"), self.model_cost_interval_usd)
            self._next_model_tokens[model] = _next_threshold(row.get("total_tokens"), float(self.model_token_interval))

    def _run(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            self.poll()

    def poll(self) -> None:
        records = read_usage_log_records(self.usage_log_file)
        if not records:
            return

        new_records = []
        for index, record in enumerate(records):
            key = _record_key(record, index)
            if key in self._seen_records:
                continue
            self._seen_records.add(key)
            new_records.append(record)

        for record in new_records:
            summary = usage_summary_from_record(record)
            cost = number_or_none(summary.get("cost_usd"))
            if cost is not None and self.request_cost_warn_usd > 0 and cost >= self.request_cost_warn_usd:
                model = summary.get("requested_model") or summary.get("response_model") or "unknown"
                self.emit(
                    "[BUDGET ALERT] "
                    f"single OpenRouter call on {model} cost {format_money(cost)} "
                    f"({format_tokens(summary.get('prompt_tokens'))} prompt, "
                    f"{format_tokens(summary.get('completion_tokens'))} completion tokens)."
                    "\a"
                )

        aggregate = aggregate_usage_records(records)
        overall = aggregate["overall"]
        self._emit_cost_threshold(
            scope="total OpenRouter",
            current=overall.get("cost_usd"),
            next_attr="_next_total_cost",
            interval=self.total_cost_interval_usd,
            calls=overall.get("captured_calls"),
            tokens=overall.get("total_tokens"),
        )
        self._emit_token_threshold(
            scope="total OpenRouter",
            current=overall.get("total_tokens"),
            next_attr="_next_total_tokens",
            interval=float(self.total_token_interval),
            calls=overall.get("captured_calls"),
            cost=overall.get("cost_usd"),
        )
        for row in aggregate["by_model"]:
            model = str(row.get("model") or "unknown")
            if model not in self._next_model_cost:
                self._next_model_cost[model] = _next_threshold(0, self.model_cost_interval_usd)
            if model not in self._next_model_tokens:
                self._next_model_tokens[model] = _next_threshold(0, float(self.model_token_interval))
            self._emit_cost_threshold(
                scope=model,
                current=row.get("cost_usd"),
                next_mapping=self._next_model_cost,
                mapping_key=model,
                interval=self.model_cost_interval_usd,
                calls=row.get("captured_calls"),
                tokens=row.get("total_tokens"),
            )
            self._emit_token_threshold(
                scope=model,
                current=row.get("total_tokens"),
                next_mapping=self._next_model_tokens,
                mapping_key=model,
                interval=float(self.model_token_interval),
                calls=row.get("captured_calls"),
                cost=row.get("cost_usd"),
            )

    def _emit_cost_threshold(
        self,
        *,
        scope: str,
        current: object,
        interval: float,
        calls: object,
        tokens: object,
        next_attr: str | None = None,
        next_mapping: dict[str, float | None] | None = None,
        mapping_key: str | None = None,
    ) -> None:
        if interval <= 0:
            return
        current_value = number_or_none(current)
        if current_value is None:
            return
        if next_attr is not None:
            next_value = getattr(self, next_attr)
        elif next_mapping is not None and mapping_key is not None:
            next_value = next_mapping.get(mapping_key)
        else:
            return
        while next_value is not None and current_value >= next_value:
            self.emit(
                "[BUDGET ALERT] "
                f"{scope} cost reached {format_money(current_value)} "
                f"after {format_tokens(calls)} calls / {format_tokens(tokens)} tokens."
                "\a"
            )
            next_value += interval
        if next_attr:
            setattr(self, next_attr, next_value)
        elif next_mapping is not None and mapping_key is not None:
            next_mapping[mapping_key] = next_value

    def _emit_token_threshold(
        self,
        *,
        scope: str,
        current: object,
        interval: float,
        calls: object,
        cost: object,
        next_attr: str | None = None,
        next_mapping: dict[str, float | None] | None = None,
        mapping_key: str | None = None,
    ) -> None:
        if interval <= 0:
            return
        current_value = number_or_none(current)
        if current_value is None:
            return
        if next_attr is not None:
            next_value = getattr(self, next_attr)
        elif next_mapping is not None and mapping_key is not None:
            next_value = next_mapping.get(mapping_key)
        else:
            return
        while next_value is not None and current_value >= next_value:
            self.emit(
                "[TOKEN ALERT] "
                f"{scope} usage reached {format_tokens(current_value)} tokens "
                f"after {format_tokens(calls)} calls; cost so far {format_money(cost) or 'not captured'}."
                "\a"
            )
            next_value += interval
        if next_attr:
            setattr(self, next_attr, next_value)
        elif next_mapping is not None and mapping_key is not None:
            next_mapping[mapping_key] = next_value
