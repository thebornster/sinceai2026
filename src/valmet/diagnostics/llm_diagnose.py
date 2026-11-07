"""LLM diagnosis: rule-based result + Excel evidence -> grounded, citation-checked JSON.

``diagnose_with_llm`` sends the rule-based result and the object's evidence bundle to an
OpenAI-compatible model (see ``llm_client``) and parses the reply into the rule-based
schema with ``method: "llm"``. The LLM result adds an ``explanation`` and richer
``next_steps`` (priority, look_for, confirms, rules_out, basis); every rule-based key is
still there.

Code-side filters run on every reply, and what they change is recorded in
``llm.filters``:
- "General background:" sentences that contain a number are dropped;
- safety precautions are kept only if taken from a similar case's safe_evidence_workflow or
  the knowledge doc (the rule-based workflow is added if missing), plus the LOTO line;
- a step with ``basis: "evidence"`` becomes ``"hypothesis"`` unless every meaningful word
  (lower-case, no filler words or punctuation) of its cited row's recommended_next_check /
  safe_evidence_workflow / content text appears in the step.

Every citation is then checked (``citations.check_citations``). If the check fails, the
model is re-asked once with the exact failures (``llm.citation_repair``) and the answer
with fewer failures is kept. Evidence items that still fail are dropped, and the report
is attached as ``citation_check``.

Anything that goes wrong (no key, API error, timeout, empty reply, invalid JSON after one
retry) returns the rule-based result with ``method: "rule_based_fallback"`` and a short
``fallback_reason``, so the viewer never crashes.

Results are cached in ``data/fixtures/metadata/<scene_id>/llm_diagnoses/<object_id>.json``
and reused unless ``refresh=True`` or an input changed (rules, evidence, prompt), whichever
model made them: the cache is built with the cache model (Kimi-K3, ``python -m
valmet.diagnostics.llm_diagnose``) and served to live calls that default to a faster
model. Offline mode only reads the cache.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from valmet.diagnostics import citations, llm_client
from valmet.diagnostics.evidence import ROUTINE_EVENT_CODES, ROUTINE_EVENT_TYPES
from valmet.diagnostics.llm_client import LLMError
from valmet.metadata.mapping import METADATA_FIXTURES_DIR

CACHE_ROOT = METADATA_FIXTURES_DIR  # tests point this at a temp folder

# Which model role a new LLM call uses: "live" (diagnose_fault) or "cache" (building the
# cache with `python -m valmet.diagnostics.llm_diagnose`). See llm_client.ROLES.
_ROLE: ContextVar[str] = ContextVar("valmet_llm_role", default=llm_client.LIVE)


@contextmanager
def use_role(role: str) -> Iterator[None]:
    """Make LLM calls inside this block use the given role's model and limits."""
    if role not in llm_client.ROLES:
        raise ValueError(f"unknown role {role!r}")
    token = _ROLE.set(role)
    try:
        yield
    finally:
        _ROLE.reset(token)


# False inside the cache-build CLI: there, only main() writes, after checking the answer.
_WRITE_CACHE: ContextVar[bool] = ContextVar("valmet_llm_write_cache", default=True)


@contextmanager
def _no_cache_write() -> Iterator[None]:
    token = _WRITE_CACHE.set(False)
    try:
        yield
    finally:
        _WRITE_CACHE.reset(token)


CACHE_SUBDIR = "llm_diagnoses"
IDENTITY_FIELDS = ("object_id", "device_id", "asset_id", "tag", "is_target")
LOTO = "Follow site lock-out/tag-out procedures before any physical inspection."
MAX_ERROR_CHARS = 1500

SYSTEM_PROMPT = (
    """\
You are a field-service diagnostic assistant for industrial process equipment. A service \
technician is about to inspect one asset. Tell them what the data shows, what it probably \
means and how to check it on site. Add value beyond the rule-based result in the user \
message: explain and plan, do not restate it.

SOURCES AND NUMBERS
1. Everything you say about this asset must come from the user message: the rule-based \
result and the Excel evidence bundle.
2. Every number taken from the data is followed by its source in parentheses, exactly as \
given, e.g. "health score 45 (Current_State!R15)". Sources look like Sheet!R<row>.
3. Derived numbers (means, z-scores, counts, differences, ratios) are allowed only when \
labelled "derived from <sources>", e.g. "z = +15.4, derived from Telemetry_Sample_3d!R2017 \
and healthy-peer telemetry".
4. Never invent values, thresholds, limits, part numbers or procedures. You may use general \
engineering knowledge to explain how a failure mode works or what a technician would \
observe, but only in sentences that start with "General background:", contain no numbers, \
and never present it as data about this asset.

EVENTS
5. Fault alarms are only events whose event_type is not one of """
    + ", ".join(ROUTINE_EVENT_TYPES)
    + """ and whose event_code is not one of """
    + ", ".join(ROUTINE_EVENT_CODES)
    + """. All other events are routine records: call them "events" or "records", never \
"alarms", and never treat them as evidence of a fault.

CONTENT
6. "findings" = EVIDENCE: observed facts only, each with its _source. Check the telemetry \
trend yourself (first, last, change).
7. "likely_cause" = HYPOTHESES: each text starts with "Hypothesis:" and cites the similar \
case(s), knowledge document or rows behind it.
8. "explanation": 3 to 6 sentences in plain technician language. Explain why the observed \
combination of signals (for example vibration rising while pressure and flow fall) points \
to the hypothesised failure mode, citing the knowledge document and similar cases. Say \
which signals in the data support it and which expected signals are missing or not \
measured. Mark general engineering background as rule 4 requires.
9. "safety_precautions" come before any inspection step; include the cases' \
safe_evidence_workflow.
10. "next_steps": a prioritised inspection sequence, priority 1 first, 3 to 6 steps. Each \
step has: "text" (what to do), "look_for" (what to observe or measure), "confirms" (what \
result would confirm the hypothesis), "rules_out" (what result would rule it out), "basis" \
("evidence" if it comes from recommended_next_check, safe_evidence_workflow or the \
knowledge document, else "general background") and "_source" (null for general \
background). Inspection and observation only: never autonomous control actions (starting \
or stopping equipment, changing setpoints, operating valves); those are for authorised \
staff following site procedures.
11. "evidence" lists every Excel value you used as {"source", "field", "value"}, with the \
value copied exactly from the evidence bundle. Every Sheet!R<row> you mention anywhere must \
also appear in "evidence".

Reply with one JSON object only: no markdown, no commentary, no reasoning. Schema:
{
  "object_id": string, "device_id": string, "asset_id": string, "tag": string,
  "is_target": boolean,
  "summary": string,
  "findings": [{"type": string, "field": string or null, "text": string, \
"_source": "Sheet!R<row>" or null}],
  "likely_cause": [{"text": "Hypothesis: ...", "case_id": string or null, \
"_source": string or null}],
  "explanation": string,
  "safety_precautions": [{"text": string, "case_id": string or null, \
"_source": string or null}],
  "next_steps": [{"priority": integer, "text": string, "look_for": string, \
"confirms": string, "rules_out": string, "basis": "evidence" or "general background", \
"_source": string or null}],
  "evidence": [{"source": "Sheet!R<row>", "field": string, "value": string or number}],
  "confidence": number or null,
  "method": "llm"
}"""
)


# --------------------------------------------------------------------------- #
# Reply schema
# --------------------------------------------------------------------------- #
class _Item(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class SourcedText(_Item):
    text: str = Field(min_length=1)
    case_id: str | None = None
    source: str | None = Field(default=None, alias="_source")


class InspectionStep(_Item):
    priority: int = Field(ge=1)
    text: str = Field(min_length=1)
    look_for: str = Field(min_length=1)
    confirms: str = Field(min_length=1)
    rules_out: str = Field(min_length=1)
    basis: str = Field(min_length=1)
    source: str | None = Field(default=None, alias="_source")


class Finding(_Item):
    type: str = Field(min_length=1)
    field: str | None = None
    text: str = Field(min_length=1)
    source: str | None = Field(default=None, alias="_source")


class EvidenceItem(_Item):
    source: str = Field(min_length=1)
    field: str = Field(min_length=1)
    value: str | int | float | bool | None


class LLMDiagnosis(_Item):
    """What we keep from the reply. Identity fields and confidence come from the rules."""

    summary: str = Field(min_length=1)
    findings: list[Finding]
    likely_cause: list[SourcedText]
    explanation: str = Field(min_length=1)
    safety_precautions: list[SourcedText] = Field(min_length=1)
    next_steps: list[InspectionStep] = Field(min_length=1)
    evidence: list[EvidenceItem] = Field(min_length=1)


class ReplyError(ValueError):
    """The reply could not be turned into a valid LLMDiagnosis."""


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)  # never closed
_FENCE = re.compile(r"```[A-Za-z0-9_-]*")


def parse_reply(text: str) -> dict[str, Any]:
    """Strip <think> blocks and fences, decode the first JSON object, validate it."""
    if not text or not text.strip():
        raise ReplyError("empty reply")
    cleaned = _THINK_OPEN.sub("", _THINK_BLOCK.sub("", text))
    cleaned = _FENCE.sub("", cleaned)
    start = cleaned.find("{")
    if start < 0:
        raise ReplyError("no JSON object in the reply")
    try:
        obj, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError as err:
        raise ReplyError(f"invalid JSON: {err}") from None
    try:
        model = LLMDiagnosis.model_validate(obj)
    except ValidationError as err:
        raise ReplyError(f"JSON does not match the schema: {err}"[:MAX_ERROR_CHARS]) from None
    out = model.model_dump(by_alias=True)
    out["next_steps"].sort(key=lambda s: s["priority"])  # stable: ties keep reply order
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _base(rules: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rules.items() if k not in ("method", "fallback_reason")}


def fallback(rules: dict[str, Any], reason: str) -> dict[str, Any]:
    out = copy.deepcopy(_base(rules))
    out["method"] = "rule_based_fallback"
    out["fallback_reason"] = reason
    return out


def cache_path(scene_id: str, object_id: str) -> Path:
    return Path(CACHE_ROOT) / scene_id / CACHE_SUBDIR / f"{object_id}.json"


def input_sha256(rules: dict[str, Any], evidence: dict[str, Any]) -> str:
    """Fingerprint of the inputs (rules, evidence, prompt); a change makes the cache stale.

    The model is deliberately not part of it: a cached answer built by the cache model
    (e.g. Kimi-K3) is reused even when the live default is a different model.
    """
    payload = json.dumps(
        {"rules": _base(rules), "evidence": evidence, "prompt": SYSTEM_PROMPT},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_cache(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("method") == "llm" else None


def _write_cache(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_messages(
    object_id: str, rules: dict[str, Any], evidence: dict[str, Any]
) -> list[dict[str, str]]:
    user = (
        f"Diagnose object {object_id}.\n\n"
        "Rule-based result (background only: explain and plan beyond it, and add no fact "
        "that is not in the evidence):\n"
        f"{json.dumps(_base(rules), ensure_ascii=False)}\n\n"
        "Excel evidence bundle for this object:\n"
        f"{json.dumps(evidence, ensure_ascii=False)}\n\n"
        "Reply with the JSON object only."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _ask(
    client: Any, settings: llm_client.LLMSettings, messages: list[dict[str, str]]
) -> tuple[dict[str, Any], llm_client.ChatReply, dict[str, Any]]:
    """(parsed reply, last raw reply, run info). One retry on an unusable reply."""
    reply = llm_client.chat(client, settings, messages)
    info: dict[str, Any] = {"attempts": 1, "retry_reason": None, "elapsed_s": reply.elapsed_s}
    if not reply.content.strip():
        raise LLMError(f"empty reply (finish_reason={reply.finish_reason})")
    try:
        return parse_reply(reply.content), reply, info
    except ReplyError as err:
        info["retry_reason"] = str(err)[:300]
        retry = messages + [
            {"role": "assistant", "content": reply.content[:4000]},
            {
                "role": "user",
                "content": f"Your reply could not be used: {err}\n"
                "Reply again with only the corrected JSON object, same schema, no other text.",
            },
        ]
    reply = llm_client.chat(client, settings, retry)
    info["attempts"] = 2
    info["elapsed_s"] = round(info["elapsed_s"] + reply.elapsed_s, 2)
    if not reply.content.strip():
        raise LLMError(f"empty reply on retry (finish_reason={reply.finish_reason})")
    return parse_reply(reply.content), reply, info  # a second ReplyError propagates


# --------------------------------------------------------------------------- #
# Code-side filters (applied to every parsed reply; what they change is recorded)
# --------------------------------------------------------------------------- #
# Text columns a step or a safety item can legitimately be taken from.
SUPPORT_FIELDS = ("recommended_next_check", "safe_evidence_workflow", "content")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_DIGIT = re.compile(r"\d")


def _norm(text: Any) -> str:
    """Lower-case words only: no citations, brackets or punctuation."""
    text = citations.REF_RE.sub(" ", str(text or ""))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _part_of(text: Any, source_text: Any) -> bool:
    """True if `text` is `source_text`, or a contiguous part of it of 3+ words."""
    a, b = _norm(text), _norm(source_text)
    return bool(a and b) and (a == b or (len(a.split()) >= 3 and f" {a} " in f" {b} "))


# Words that carry no instruction ("inspect THE process conditions AT the pump").
FILLER_WORDS = frozenset(
    {
        *("a", "an", "the", "this", "that", "these", "those", "it", "its"),
        *("of", "to", "in", "on", "at", "by", "for", "from", "with", "as", "and", "or"),
        *("then", "any", "all", "some", "their", "your", "our"),
        *("is", "are", "be", "been", "being", "was", "were"),
        *("please", "also", "carefully", "first", "next", "around", "near"),
    }
)


def _singular(word: str) -> str:
    """Fold simple English plurals: conditions -> condition, processes -> process."""
    if len(word) > 4 and word.endswith("es") and word[:-2].endswith(("ss", "sh", "ch", "x")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def meaningful_words(text: Any) -> set[str]:
    """Lower-case content words: no filler, punctuation or citations; plurals folded."""
    return {_singular(w) for w in _norm(text).split() if w not in FILLER_WORDS}


def supports(row_text: Any, step_text: Any) -> bool:
    """True if every meaningful word of the cited row's text appears in the step."""
    need = meaningful_words(row_text)
    return bool(need) and need <= meaningful_words(step_text)


def drop_background_numbers(parsed: dict[str, Any]) -> list[dict[str, str]]:
    """Remove "General background:" sentences that contain a number (in place)."""
    dropped: list[dict[str, str]] = []

    def clean(where: str, text: str) -> str:
        keep = []
        for sentence in _SENTENCE_END.split(text):
            bare = citations.REF_RE.sub("", sentence)
            if sentence.strip().lower().startswith("general background") and _DIGIT.search(bare):
                dropped.append({"where": where, "sentence": sentence.strip()})
            else:
                keep.append(sentence)
        return " ".join(keep).strip()

    for key in ("summary", "explanation"):
        parsed[key] = clean(key, parsed[key])
    for section, fields in (
        ("findings", ("text",)),
        ("likely_cause", ("text",)),
        ("next_steps", ("text", "look_for", "confirms", "rules_out")),
    ):
        for i, item in enumerate(parsed[section]):
            for f in fields:
                item[f] = clean(f"{section}[{i}].{f}", item[f])
    return dropped


def filter_safety(
    parsed: dict[str, Any], rules: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, list[str]]:
    """Keep only safety items taken from a case workflow or the knowledge doc, plus LOTO.

    Kept items get the matching row as _source (and that row's text in evidence). The
    rule-based workflow items are added if the model left them out. LOTO is always last.
    """
    allowed = [
        (c[citations.SOURCE_COL], "safe_evidence_workflow", c["safe_evidence_workflow"], c)
        for c in evidence.get("similar_cases") or []
        if c.get("safe_evidence_workflow")
    ]
    doc = evidence.get("knowledge") or {}
    if doc.get("content"):
        allowed.append((doc[citations.SOURCE_COL], "content", doc["content"], {}))

    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    for item in parsed["safety_precautions"]:
        if _norm(item["text"]) == _norm(LOTO):
            continue  # re-added below, in its canonical wording
        match = next((a for a in allowed if _part_of(item["text"], a[2])), None)
        if match is None:
            dropped.append(item["text"])
            continue
        src, field, value, case = match
        kept.append({"text": item["text"], "case_id": case.get("case_id"), "_source": src})
        _ensure_evidence(parsed, src, field, value)

    added: list[str] = []
    for item in rules["safety_precautions"]:
        if item["_source"] and not any(_part_of(k["text"], item["text"]) for k in kept):
            kept.append(dict(item))
            _ensure_evidence(parsed, item["_source"], "safe_evidence_workflow", item["text"])
            added.append(item["text"])
    kept.append({"text": LOTO, "case_id": None, "_source": None})
    parsed["safety_precautions"] = kept
    return {"dropped": dropped, "added": added}


def _ensure_evidence(parsed: dict[str, Any], source: str, field: str, value: Any) -> None:
    if not any(e["source"] == source and e["field"] == field for e in parsed["evidence"]):
        parsed["evidence"].append({"source": source, "field": field, "value": value})


def downgrade_unsupported_steps(
    parsed: dict[str, Any], rules: dict[str, Any], evidence: dict[str, Any]
) -> list[dict[str, Any]]:
    """basis "evidence" -> "hypothesis" when the cited row has no text the step relies on."""
    known = citations.reference_values(evidence, rules)
    texts: dict[str, list[str]] = {}
    for (src, field), value in known.items():
        if field in SUPPORT_FIELDS and isinstance(value, str):
            texts.setdefault(src, []).append(value)

    downgraded = []
    for step in parsed["next_steps"]:
        if step["basis"].strip().lower() != "evidence":
            continue
        src = step["_source"]
        step_text = f"{step['text']} {step['look_for']}"
        if not src:
            reason = "no _source"
        elif src not in texts:
            reason = f"{src} has no {'/'.join(SUPPORT_FIELDS)} text"
        elif not any(supports(t, step_text) for t in texts[src]):
            reason = f"{src} does not contain the step's instruction"
        else:
            continue
        step["basis"] = "hypothesis"
        downgraded.append({"priority": step["priority"], "text": step["text"], "reason": reason})
    return downgraded


def _assemble(
    rules: dict[str, Any], parsed: dict[str, Any], evidence: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(result without the run record, what the filters changed). Mutates a copy only."""
    parsed = copy.deepcopy(parsed)
    filters = {"background_sentences_dropped": drop_background_numbers(parsed)}
    safety = filter_safety(parsed, rules, evidence)
    filters["safety_dropped"] = safety["dropped"]
    filters["safety_added"] = safety["added"]
    filters["basis_downgraded"] = downgrade_unsupported_steps(parsed, rules, evidence)
    result = {
        **{k: rules[k] for k in IDENTITY_FIELDS},  # never trust the model for identity
        "summary": parsed["summary"],
        "findings": parsed["findings"],
        "likely_cause": parsed["likely_cause"],
        "explanation": parsed["explanation"],
        "safety_precautions": parsed["safety_precautions"],
        "next_steps": parsed["next_steps"],
        "evidence": parsed["evidence"],
        "confidence": rules["confidence"],  # the Excel's model_confidence, not the LLM's
        "method": "llm",
    }
    return result, filters


def _repair_message(failed: list[dict[str, Any]]) -> str:
    return (
        "Your answer failed the citation check. Exact failures:\n"
        f"{json.dumps(failed, ensure_ascii=False)}\n"
        'Fix every one: list each cited Sheet!R<row> in "evidence" with the exact field and '
        "value from the evidence bundle, correct wrong values, or remove the citation. Keep "
        "everything else. Reply with the full corrected JSON object only."
    )


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #
def diagnose_with_llm(
    object_id: str,
    rules: dict[str, Any],
    evidence: dict[str, Any] | None,
    *,
    scene_id: str,
    mode: str,
    refresh: bool = False,
) -> dict[str, Any]:
    """LLM diagnosis, or the rule-based fallback.

    mode: "live" (citations checked against the Excel), "snapshot" (against evidence.json;
    the workbook is missing) or "offline" (no API call: cache or fallback only).
    """
    if mode not in ("live", "snapshot", "offline"):
        raise ValueError(f"unknown mode {mode!r}")
    path = cache_path(scene_id, object_id)
    digest = input_sha256(rules, evidence) if evidence is not None else None

    if not refresh:  # any model's answer counts, as long as the inputs are unchanged
        cached = _read_cache(path)
        if cached is not None:
            if digest is None or cached.get("llm", {}).get("input_sha256") == digest:
                return cached
            if mode == "offline":
                return fallback(rules, "cached LLM diagnosis is stale (inputs changed)")
    if mode == "offline":
        return fallback(rules, "offline and no cached LLM diagnosis")
    if evidence is None:
        return fallback(rules, "no evidence for this object (snapshot evidence.json missing)")
    try:
        settings = llm_client.load_settings(_ROLE.get())
    except LLMError as err:
        return fallback(rules, err.reason)
    if not settings.api_key:
        return fallback(rules, f"no API key ({llm_client.API_KEY_ENV} is empty)")

    messages = build_messages(object_id, rules, evidence)
    try:
        client = llm_client.make_client(settings)
        parsed, reply, info = _ask(client, settings, messages)
    except LLMError as err:
        return fallback(rules, err.reason)
    except ReplyError as err:
        return fallback(rules, f"invalid reply after retry: {err}"[:300])

    def check(res: dict[str, Any]) -> dict[str, Any]:
        return citations.check_citations(
            res, offline=(mode == "snapshot"), evidence=evidence, rules=rules
        )

    result, filters = _assemble(rules, parsed, evidence)
    report = check(result)
    info["citation_repair"] = None
    if not report["ok"]:
        result, report, filters, reply = _repair_citations(
            client,
            settings,
            messages,
            parsed,
            rules,
            evidence,
            info,
            check,
            first=(result, report, filters, reply),
        )

    citations.drop_failed_evidence(result, report)
    result["citation_check"] = report
    result["llm"] = {
        **settings.record(),
        "response_model": reply.model,
        "finish_reason": reply.finish_reason,
        **info,
        "filters": filters,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "input_sha256": digest,
    }
    if _WRITE_CACHE.get():
        _write_cache(path, result)
    return result


def _repair_citations(
    client: Any,
    settings: llm_client.LLMSettings,
    messages: list[dict[str, str]],
    parsed: dict[str, Any],
    rules: dict[str, Any],
    evidence: dict[str, Any],
    info: dict[str, Any],
    check: Any,
    *,
    first: tuple[dict[str, Any], dict[str, Any], dict[str, Any], llm_client.ChatReply],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], llm_client.ChatReply]:
    """Re-ask once with the exact citation failures; keep whichever answer fails less."""
    result, report, filters, reply = first
    repair: dict[str, Any] = {"failed_before": report["failed"], "used": False}
    info["citation_repair"] = repair
    retry = messages + [
        {"role": "assistant", "content": json.dumps(parsed, ensure_ascii=False)},
        {"role": "user", "content": _repair_message(report["failed"])},
    ]
    info["attempts"] += 1
    try:
        reply2 = llm_client.chat(client, settings, retry)
        info["elapsed_s"] = round(info["elapsed_s"] + reply2.elapsed_s, 2)
        if not reply2.content.strip():
            raise LLMError(f"empty reply (finish_reason={reply2.finish_reason})")
        result2, filters2 = _assemble(rules, parse_reply(reply2.content), evidence)
    except (LLMError, ReplyError) as err:
        repair["error"] = (err.reason if isinstance(err, LLMError) else str(err))[:300]
        return first
    report2 = check(result2)
    repair["failed_after"] = report2["failed"]
    if len(report2["failed"]) <= len(report["failed"]):
        repair["used"] = True
        return result2, report2, filters2, reply2
    return first


# --------------------------------------------------------------------------- #
# Cache-build CLI
# --------------------------------------------------------------------------- #
REBUILD_HINT = "python -m valmet.diagnostics.llm_diagnose --refresh obj_0001"
MIN_CACHE_TIMEOUT_S = llm_client.ROLES[llm_client.CACHE][2]  # 300 s


def cache_build_problem(result: dict[str, Any], settings: llm_client.LLMSettings) -> str | None:
    """None if `result` may be written to the cache, else the reason it may not.

    Only a real LLM answer made by the cache model, in the cache role, with at least the
    cache timeout qualifies. Fallbacks (incl. timeouts) never do.
    """
    if result.get("method") != "llm":
        return f"not an LLM answer ({result.get('method')}: {result.get('fallback_reason')})"
    rec = result.get("llm") or {}
    if rec.get("role") != llm_client.CACHE:
        return f"made in role {rec.get('role')!r}, not {llm_client.CACHE!r}"
    if rec.get("model") != settings.model:
        return f"made by {rec.get('model')!r}, not the cache model {settings.model!r}"
    if (rec.get("timeout_s") or 0) < MIN_CACHE_TIMEOUT_S:
        return (
            f"timeout {rec.get('timeout_s')} s is below the cache minimum {MIN_CACHE_TIMEOUT_S:g} s"
        )
    return None


def main(argv: list[str] | None = None) -> int:
    """Build cached LLM diagnoses with the cache model (VALMET_LLM_CACHE_MODEL).

    Usage: [--refresh] [--out DIR] [object_id ...]   (default obj_0001)

    Always asks the model again (--refresh is accepted for clarity). Writes
    <object_id>.json to the scene's llm_diagnoses folder, or to DIR with --out, but only if
    the answer passes cache_build_problem(); otherwise nothing is written. Exit code: 0 all
    written, 1 something refused, 2 bad settings (nothing called).
    """
    import sys

    from valmet import api

    args = list(sys.argv[1:] if argv is None else argv)
    out_dir = None
    if "--out" in args:
        i = args.index("--out")
        if i + 1 >= len(args):
            print("--out needs a directory")
            return 2
        out_dir = Path(args.pop(i + 1))
        args.pop(i)
    object_ids = [a for a in args if a != "--refresh"] or ["obj_0001"]

    try:
        settings = llm_client.load_settings(llm_client.CACHE)
    except LLMError as err:
        print(f"refused: {err.reason}")
        return 2
    print(
        f"cache build: model={settings.model} role={settings.role} "
        f"timeout={settings.timeout_s:g} s max_tokens={settings.max_tokens}",
        flush=True,
    )
    if settings.timeout_s < MIN_CACHE_TIMEOUT_S:
        print(
            f"refused: cache timeout {settings.timeout_s:g} s is below "
            f"{MIN_CACHE_TIMEOUT_S:g} s (unset {llm_client.TIMEOUT_ENV} for cache builds)"
        )
        return 2

    status = 0
    for object_id in object_ids:
        with use_role(llm_client.CACHE), _no_cache_write():
            d = api.diagnose_fault(object_id, use_llm=True, refresh=True)
        scene_id = api.get_asset(object_id)["scene_id"]
        dest = out_dir / f"{object_id}.json" if out_dir else cache_path(scene_id, object_id)
        problem = cache_build_problem(d, settings)
        if problem:
            print(f"{object_id}: REFUSED, nothing written to {dest}: {problem}")
            status = 1
            continue
        _write_cache(dest, d)
        rec, cc = d["llm"], d["citation_check"]
        print(
            f"{object_id}: wrote {dest} (model={rec['model']} role={rec['role']} "
            f"timeout={rec['timeout_s']:g} s attempts={rec['attempts']} "
            f"elapsed={rec['elapsed_s']} s citation ok={cc['ok']})"
        )
    return status


if __name__ == "__main__":
    # `python -m` runs this file as a second module named __main__. Call main() on the
    # shared module that valmet.api uses, so use_role() sets the role diagnose_fault sees.
    from valmet.diagnostics import llm_diagnose as _shared

    raise SystemExit(_shared.main())
