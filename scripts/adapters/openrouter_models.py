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
