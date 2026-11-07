"""Gemini access with a model fallback chain (vision labelling only).

Free-tier keys have small per-model daily quotas and models get overloaded (503) or
retired (404). generate() tries each model in order and moves on for those errors.

Env (repo-root .env is loaded first, set variables win):
    GEMINI_API_KEY          required
    VALMET_GEMINI_MODELS    optional comma-separated model list; default DEFAULT_MODELS

``google-genai`` is imported lazily so the rest of the package works without it.
"""

from __future__ import annotations

import os
import time
from typing import Any

from valmet.diagnostics.llm_client import load_env_file

API_KEY_ENV = "GEMINI_API_KEY"
MODELS_ENV = "VALMET_GEMINI_MODELS"
DEFAULT_MODELS = (
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemma-4-31b-it",
)
TEMPERATURE = 0.2
DEAD_MODEL_SKIP_S = 600.0  # skip a quota-exhausted/retired model for this long
SKIP_CODES = ("429", "503", "404", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "NOT_FOUND")
OVERLOAD_CODES = ("503", "UNAVAILABLE")

_dead: dict[str, float] = {}


class GeminiUnavailableError(RuntimeError):
    """No key, or every model in the chain failed."""


def models() -> list[str]:
    raw = os.environ.get(MODELS_ENV, "").strip()
    return [m.strip() for m in raw.split(",") if m.strip()] or list(DEFAULT_MODELS)


def _client() -> Any:
    from google import genai

    load_env_file()
    key = os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise GeminiUnavailableError(f"{API_KEY_ENV} not set")
    return genai.Client(api_key=key)


def generate(
    contents: Any,
    json_mode: bool = True,
    temperature: float | None = None,
) -> tuple[str, str]:
    """Returns (text, model_used)."""
    from google.genai import types

    client = _client()
    errors = []
    for model in models():
        if time.time() - _dead.get(model, 0) < DEAD_MODEL_SKIP_S:
            continue
        kw: dict[str, Any] = {"temperature": TEMPERATURE if temperature is None else temperature}
        if json_mode:
            kw["response_mime_type"] = "application/json"
        try:
            r = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**kw),
            )
        except Exception as e:  # noqa: BLE001 - classify and fall through to the next model
            s = str(e)
            errors.append(f"{model}: {s[:80]}")
            if not any(c in s for c in SKIP_CODES):
                raise
            if not any(c in s for c in OVERLOAD_CODES):
                _dead[model] = time.time()
            continue
        if r.text:
            return r.text, model
        errors.append(f"{model}: empty response")
    raise GeminiUnavailableError("all models failed: " + " | ".join(errors))
