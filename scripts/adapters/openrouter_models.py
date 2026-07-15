from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any


MODELS_URL = "https://openrouter.ai/api/v1/models"
FREE_MODEL_ALIAS = "openrouter/free"
FREE_VISION_MODEL_ALIAS = "openrouter/free-vision"


class ModelResolutionError(RuntimeError):
    pass


def resolve_model_alias(model: str, *, require_image: bool) -> str:
    model = model.strip()
    if model not in {FREE_MODEL_ALIAS, FREE_VISION_MODEL_ALIAS}:
        return model

    override = os.environ.get("OPENROUTER_FREE_MODEL", "").strip()
    if override:
        return override

    data = fetch_models()
    for candidate in data.get("data", []):
        if is_usable_free_model(candidate, require_image=require_image):
            return str(candidate["id"])

    requirement = "free image-capable" if require_image else "free"
    raise ModelResolutionError(
        f"Could not find a {requirement} OpenRouter model. "
        "Set OPENROUTER_FREE_MODEL to a concrete model id ending in ':free'."
    )


def fetch_models() -> dict[str, Any]:
    req = urllib.request.Request(MODELS_URL, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise ModelResolutionError(f"Could not fetch OpenRouter models: {exc}") from exc


def is_usable_free_model(model: dict[str, Any], *, require_image: bool) -> bool:
    model_id = str(model.get("id", ""))
    name = str(model.get("name", ""))
    text = f"{model_id} {name}".lower()
    if any(term in text for term in ("safety", "moderation", "guardrail", "filter", "embedding")):
        return False

    pricing = model.get("pricing") or {}
    if not (is_zero_price(pricing.get("prompt")) and is_zero_price(pricing.get("completion"))):
        return False

    architecture = model.get("architecture") or {}
    input_modalities = set(architecture.get("input_modalities") or [])
    output_modalities = set(architecture.get("output_modalities") or [])
    if "text" not in input_modalities or "text" not in output_modalities:
        return False
    if require_image and "image" not in input_modalities:
        return False
    return True


def is_zero_price(value: Any) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def model_pricing_snapshot(model_ids: list[str]) -> list[dict[str, Any]]:
    try:
        data = fetch_models()
    except ModelResolutionError as exc:
        return [{"model": model_id, "pricing_error": str(exc)} for model_id in model_ids]

    index: dict[str, dict[str, Any]] = {}
    for model in data.get("data", []):
        if not isinstance(model, dict):
            continue
        for key in model_lookup_keys(model):
            index.setdefault(key, model)

    snapshots = []
    for model_id in model_ids:
        match = index.get(normalize_model_key(model_id))
        if match is None:
            snapshots.append({"model": model_id, "pricing_error": "model not found in OpenRouter /models"})
            continue
        pricing = match.get("pricing") if isinstance(match.get("pricing"), dict) else {}
        top_provider = match.get("top_provider") if isinstance(match.get("top_provider"), dict) else {}
        snapshots.append(
            {
                "model": model_id,
                "openrouter_id": match.get("id"),
                "canonical_slug": match.get("canonical_slug"),
                "name": match.get("name"),
                "prompt_usd_per_million": price_per_million(pricing.get("prompt")),
                "completion_usd_per_million": price_per_million(pricing.get("completion")),
                "reasoning_usd_per_million": price_per_million(pricing.get("internal_reasoning")),
                "cache_read_usd_per_million": price_per_million(pricing.get("input_cache_read")),
                "cache_write_usd_per_million": price_per_million(pricing.get("input_cache_write")),
                "context_length": match.get("context_length"),
                "max_completion_tokens": top_provider.get("max_completion_tokens"),
            }
        )
    return snapshots


def model_lookup_keys(model: dict[str, Any]) -> set[str]:
    keys = set()
    for field in ("id", "canonical_slug"):
        value = model.get(field)
        if isinstance(value, str) and value.strip():
            keys.add(normalize_model_key(value))
    return keys


def normalize_model_key(model_id: str) -> str:
    return model_id.strip().casefold().replace(".", "-")


def price_per_million(value: Any) -> float | None:
    try:
        return float(value) * 1_000_000
    except (TypeError, ValueError):
        return None
