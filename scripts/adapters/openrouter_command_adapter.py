from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
import sys
from typing import Any
import urllib.error
import urllib.request

from openrouter_models import resolve_model_alias


class AdapterError(RuntimeError):
    pass


def main() -> None:
    try:
        request = json.load(sys.stdin)
        response = call_openrouter(request)
    except Exception as exc:
        print(f"OpenRouter adapter error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(response)


def call_openrouter(request: dict[str, Any]) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise AdapterError("OPENROUTER_API_KEY is required")

    model = str(request.get("model") or os.environ.get("OPENROUTER_MODEL") or "openai/gpt-4o-mini").strip()
    if not model or model == "mock-local":
        raise AdapterError(
            "Set EXPERIMENT_MODELS to an OpenRouter model id, for example openai/gpt-4o-mini"
        )

    parts = request.get("parts")
    if not isinstance(parts, list):
        raise AdapterError("Input JSON must contain a list field named 'parts'")
    model = resolve_model_alias(model, require_image=parts_has_image(parts))

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": parts_to_openrouter_content(parts),
            }
        ],
        "max_tokens": int(os.environ.get("OPENROUTER_MAX_TOKENS", "200")),
        "temperature": float(os.environ.get("OPENROUTER_TEMPERATURE", "0")),
    }

    url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    req = urllib.request.Request(
        f"{url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://local.codex"),
            "X-Title": os.environ.get("OPENROUTER_TITLE", "JM reproduction"),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=float(os.environ.get("OPENROUTER_TIMEOUT", "90"))) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise AdapterError(f"HTTP {exc.code}: {body[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise AdapterError(f"Network error: {exc}") from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AdapterError(f"Unexpected OpenRouter response: {json.dumps(data)[:2000]}") from exc

    return normalize_content(content)


def parts_to_openrouter_content(parts: list[Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, str):
            content.append({"type": "text", "text": part})
        elif isinstance(part, dict) and "image" in part:
            content.append({"type": "image_url", "image_url": {"url": image_to_url(str(part["image"]))}})
        else:
            content.append({"type": "text", "text": json.dumps(part, ensure_ascii=False)})
    return content


def parts_has_image(parts: list[Any]) -> bool:
    return any(isinstance(part, dict) and "image" in part for part in parts)


def image_to_url(image_ref: str) -> str:
    if image_ref.startswith(("http://", "https://", "data:")):
        return image_ref

    path = Path(image_ref).expanduser()
    if not path.is_absolute():
        cwd_candidate = Path.cwd() / path
        script_candidate = Path(__file__).resolve().parent / path
        path = cwd_candidate if cwd_candidate.exists() else script_candidate

    if not path.exists():
        raise AdapterError(f"Image path does not exist: {image_ref}")

    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                texts.append(item["text"])
            elif isinstance(item, str):
                texts.append(item)
        return "\n".join(texts).strip()
    return str(content).strip()


if __name__ == "__main__":
    main()
