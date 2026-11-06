"""OpenAI-compatible chat client for the LLM diagnosis (provider-agnostic).

Two model roles:

- **live** (``diagnose_fault(use_llm=True)``): ``VALMET_LLM_MODEL``, default
  Qwen/Qwen3.8-Flash-Next, 60 s limit. Fast enough for the viewer.
- **cache** (``python -m valmet.diagnostics.llm_diagnose``, which builds the cached answers):
  ``VALMET_LLM_CACHE_MODEL``, default moonshotai/Kimi-K3, 300 s limit. Slower, best answers.

Each model's sampling settings come from ``MODEL_PROFILES`` below (taken from the model
cards and the obj_0001 comparison). Unknown models get ``GENERIC_PROFILE``. Any setting
can be overridden by env; overrides apply to whichever role is running. The repo-root
``.env`` is loaded first and never overrides variables that are already set. Empty values
mean "use the default".

    FEATHERLESS_API_KEY          API key. Required for the LLM path; never logged or stored.
    VALMET_LLM_BASE_URL          default https://api.featherless.ai/v1
    VALMET_LLM_MODEL             live model, default Qwen/Qwen3.8-Flash-Next
    VALMET_LLM_CACHE_MODEL       cache-building model, default moonshotai/Kimi-K3
    VALMET_LLM_TEMPERATURE       default: the model's profile
    VALMET_LLM_TOP_P             default: the model's profile
    VALMET_LLM_ENABLE_THINKING   false | true | omit; default: the model's profile
    VALMET_LLM_MAX_TOKENS        default: the model's profile
    VALMET_LLM_TIMEOUT_S         default 60 (live) / 300 (cache); hard wall-clock limit per call

``enable_thinking`` is sent as ``chat_template_kwargs`` (Qwen-style). ``omit`` sends
nothing, for models whose thinking can't be switched off that way (Kimi-K3 always thinks).
Never use temperature 0 with Qwen3: on Featherless it degenerates ("!!!!").

The real safeguard is a hard wall-clock deadline per call: the request runs in a daemon
thread and is abandoned when the deadline passes. The SDK's own ``timeout`` alone is not
enough, because it limits each network wait, not the whole request (a real call ran for
252 s with timeout=60). SDK retries are disabled. ``max_tokens`` is only a soft cap: some
providers don't enforce it. Thinking models count their reasoning against it, so they need
more (32768), or they stop with ``finish_reason=length`` before writing the answer.

The ``openai`` and ``python-dotenv`` packages are imported lazily, so the rest of the
module works without them (the LLM path then falls back to the rule-based result).
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from valmet.common.artifacts import REPO_ROOT

API_KEY_ENV = "FEATHERLESS_API_KEY"
BASE_URL_ENV = "VALMET_LLM_BASE_URL"
MODEL_ENV = "VALMET_LLM_MODEL"
CACHE_MODEL_ENV = "VALMET_LLM_CACHE_MODEL"
TEMPERATURE_ENV = "VALMET_LLM_TEMPERATURE"
TOP_P_ENV = "VALMET_LLM_TOP_P"
THINKING_ENV = "VALMET_LLM_ENABLE_THINKING"
TIMEOUT_ENV = "VALMET_LLM_TIMEOUT_S"
MAX_TOKENS_ENV = "VALMET_LLM_MAX_TOKENS"
SETTING_ENVS = (
    BASE_URL_ENV,
    MODEL_ENV,
    CACHE_MODEL_ENV,
    TEMPERATURE_ENV,
    TOP_P_ENV,
    THINKING_ENV,
    TIMEOUT_ENV,
    MAX_TOKENS_ENV,
)

DEFAULT_BASE_URL = "https://api.featherless.ai/v1"
LIVE, CACHE = "live", "cache"
ROLES = {
    # role: (model env, default model, default timeout in s)
    LIVE: (MODEL_ENV, "Qwen/Qwen3.8-Flash-Next", 60.0),
    CACHE: (CACHE_MODEL_ENV, "moonshotai/Kimi-K3", 300.0),
}
DEFAULT_MODEL = ROLES[LIVE][1]
DEFAULT_CACHE_MODEL = ROLES[CACHE][1]


@dataclass(frozen=True)
class ModelProfile:
    temperature: float
    top_p: float
    enable_thinking: bool | None  # None: don't send chat_template_kwargs at all
    max_tokens: int


# From the model cards and the obj_0001 comparison (Featherless, 2026-11).
MODEL_PROFILES: dict[str, ModelProfile] = {
    # Card: non-thinking (instruct) settings. 43 s, citations ok: best speed/quality.
    "Qwen/Qwen3.8-Flash-Next": ModelProfile(0.7, 0.8, False, 4096),
    # Always thinks; card's single-step settings. 80 s, citations ok: best answers.
    "moonshotai/Kimi-K3": ModelProfile(1.0, 0.95, None, 32768),
    # Card: temperature 1.0, top_p 0.95; no thinking switch. 28 s, citations ok, verbose.
    "deepseek-ai/DeepSeek-V4.1-Flash": ModelProfile(1.0, 0.95, None, 32768),
    # Qwen non-thinking settings (temperature 0 degenerates). 152 s, mostly restates rules.
    "Qwen/Qwen3-32B": ModelProfile(0.7, 0.8, False, 4096),
    # Reasoning effort defaults to max and can't be switched off: no answer within 32768.
    "zai-org/GLM-5.3": ModelProfile(1.0, 0.95, None, 32768),
}
GENERIC_PROFILE = ModelProfile(0.7, 0.8, None, 4096)

_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}
_OMIT = {"omit", "none"}


class LLMError(RuntimeError):
    """The LLM call could not be made or failed. ``reason`` is short and key-free."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class LLMConfigError(LLMError):
    """Bad or incomplete settings (e.g. a non-numeric temperature, openai not installed)."""


class LLMTimeoutError(LLMError):
    """The request exceeded the wall-clock deadline (LLMSettings.timeout_s)."""


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    model: str
    temperature: float
    top_p: float
    enable_thinking: bool | None  # None: don't send chat_template_kwargs at all
    timeout_s: float = ROLES[LIVE][2]
    max_tokens: int = GENERIC_PROFILE.max_tokens
    role: str = LIVE
    api_key: str = field(default="", repr=False)

    def record(self) -> dict[str, Any]:
        """The settings to store with a result. Never includes the key."""
        return {
            "role": self.role,
            "base_url": self.base_url,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "enable_thinking": self.enable_thinking,
            "timeout_s": self.timeout_s,
            "max_tokens": self.max_tokens,
        }

    def request_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.enable_thinking is not None:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
            }
        return kwargs


@dataclass(frozen=True)
class ChatReply:
    content: str
    model: str | None
    finish_reason: str | None
    elapsed_s: float


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def load_env_file() -> None:
    """Load the repo-root .env (if python-dotenv is installed). Set variables win."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env", override=False)


def _number(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise LLMConfigError(f"{name}={raw!r} is not a number") from None


def _thinking(default: bool | None) -> bool | None:
    raw = _env(THINKING_ENV).lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    if raw in _OMIT:
        return None
    raise LLMConfigError(f"{THINKING_ENV}={raw!r}: use true, false or omit")


def profile_for(model: str) -> ModelProfile:
    return MODEL_PROFILES.get(model, GENERIC_PROFILE)


def load_settings(role: str = LIVE) -> LLMSettings:
    """Settings for a role ("live" or "cache"): the model's profile plus env overrides."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; use {sorted(ROLES)}")
    load_env_file()
    model_env, default_model, default_timeout = ROLES[role]
    model = _env(model_env) or default_model
    profile = profile_for(model)
    return LLMSettings(
        base_url=_env(BASE_URL_ENV) or DEFAULT_BASE_URL,
        model=model,
        temperature=_number(TEMPERATURE_ENV, profile.temperature),
        top_p=_number(TOP_P_ENV, profile.top_p),
        enable_thinking=_thinking(profile.enable_thinking),
        timeout_s=_timeout(default_timeout),
        max_tokens=_max_tokens(profile.max_tokens),
        role=role,
        api_key=_env(API_KEY_ENV),
    )


def _max_tokens(default: int) -> int:
    value = _number(MAX_TOKENS_ENV, default)
    if value != int(value) or not 1 <= value <= 262144:
        raise LLMConfigError(f"{MAX_TOKENS_ENV}={value:g}: use a whole number in [1, 262144]")
    return int(value)


def _timeout(default: float) -> float:
    value = _number(TIMEOUT_ENV, default)
    if not 0 < value <= 3600:
        raise LLMConfigError(f"{TIMEOUT_ENV}={value:g}: use a number of seconds in (0, 3600]")
    return value


def make_client(settings: LLMSettings) -> Any:
    """An OpenAI client for the configured endpoint (tests replace this function)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise LLMConfigError("openai package not installed") from None
    return OpenAI(
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.timeout_s,  # per network wait; chat() enforces the total
        max_retries=0,  # retries would multiply the time per call
    )


def _is_timeout(err: BaseException) -> bool:
    return isinstance(err, TimeoutError) or "timeout" in type(err).__name__.lower()


def _with_deadline(fn: Callable[[], Any], timeout_s: float) -> Any:
    """Run fn() in a daemon thread; raise TimeoutError if it is not done in timeout_s.

    The abandoned call keeps running in the background until the SDK gives up, but the
    caller (and the viewer) is free again after timeout_s, and a daemon thread never
    blocks interpreter exit.
    """
    box: dict[str, Any] = {}
    done = threading.Event()

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as err:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = err
        finally:
            done.set()

    threading.Thread(target=run, name="valmet-llm-call", daemon=True).start()
    if not done.wait(timeout_s):
        raise TimeoutError(f"no reply within {timeout_s:g} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def chat(client: Any, settings: LLMSettings, messages: list[dict[str, str]]) -> ChatReply:
    """One chat completion. Raises LLMError / LLMTimeoutError, never a provider exception."""
    start = time.perf_counter()
    try:
        resp = _with_deadline(
            lambda: client.chat.completions.create(messages=messages, **settings.request_kwargs()),
            settings.timeout_s,
        )
    except Exception as err:  # noqa: BLE001 - any provider failure becomes a fallback
        if _is_timeout(err):
            raise LLMTimeoutError(f"timeout after {settings.timeout_s:g} s") from None
        status = getattr(err, "status_code", None)
        detail = f" (HTTP {status})" if status else ""
        raise LLMError(f"API error: {type(err).__name__}{detail}") from None
    elapsed = time.perf_counter() - start
    choices = getattr(resp, "choices", None) or []
    if not choices:
        raise LLMError("API returned no choices")
    message = getattr(choices[0], "message", None)
    return ChatReply(
        content=(getattr(message, "content", None) or ""),
        model=getattr(resp, "model", None),
        finish_reason=getattr(choices[0], "finish_reason", None),
        elapsed_s=round(elapsed, 2),
    )
