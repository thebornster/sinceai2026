"""Citation checker for diagnoses: every cited Excel value must be real.

``check_citations(result)`` verifies a diagnose_fault result:

1. Each ``evidence`` item ``{source, field, value}``:
   - live: ``source`` is a real workbook row, is one of the rows this object was given
     (its evidence bundle or its rule-based evidence), ``field`` is a column of that row,
     and ``value`` equals the cell;
   - offline: the same against the snapshot (``evidence.json`` + ``diagnoses.json``)
     instead of the Excel.
2. Every ``Sheet!R<row>`` mentioned anywhere else in the result (text and ``_source``
   fields) appears among the evidence items that passed step 1.

Returns ``{"ok": bool, "checked": int, "failed": [...]}``. Each failed evidence item
carries its ``index`` in ``result["evidence"]`` so the caller can drop it. The result
itself is not modified.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from valmet.diagnostics.evidence import _json, get_evidence
from valmet.metadata import snapshot
from valmet.metadata.excel_loader import SOURCE_COL, load_workbook, workbook_path
from valmet.metadata.mapping import metadata_dir

REF_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*!R\d+\b")
# Keys whose contents are not claims: the evidence list itself, our own report, run metadata.
SKIP_KEYS = frozenset({"evidence", "citation_check", "llm"})


# --------------------------------------------------------------------------- #
# Reference data
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2)
def _rows_by_source(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for df in load_workbook(path).values():
        for row in df.to_dict("records"):
            out[row[SOURCE_COL]] = row
    return out


def bundle_values(bundle: dict[str, Any]) -> dict[tuple[str, str], Any]:
    """{(source, field): value} for every sourced value in an evidence bundle."""
    out: dict[tuple[str, str], Any] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            src = node.get(SOURCE_COL)
            if isinstance(src, str):
                for k, v in node.items():
                    if k != SOURCE_COL and not isinstance(v, (dict, list)):
                        out.setdefault((src, k), v)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    # Telemetry summaries: {column: {first, last, min, max, _sources: {stat: source}}}.
    for column, summary in (bundle.get("telemetry") or {}).items():
        for stat, src in (summary.get("_sources") or {}).items():
            out.setdefault((src, column), summary.get(stat))
    walk({k: v for k, v in bundle.items() if k != "telemetry"})
    return out


def _load_bundle(object_id: str, offline: bool) -> dict[str, Any] | None:
    if offline:
        return snapshot.snapshot_evidence(metadata_dir(), object_id)
    return get_evidence(object_id)


def _load_rules(object_id: str, offline: bool) -> dict[str, Any] | None:
    if offline:
        return snapshot.snapshot_record(metadata_dir(), "diagnoses", object_id)
    from valmet.diagnostics.diagnose import _rule_based  # diagnose imports this module

    return _rule_based(object_id)


def reference_values(
    bundle: dict[str, Any], rules: dict[str, Any] | None
) -> dict[tuple[str, str], Any]:
    """Everything the object was given: its evidence bundle plus the rule-based evidence.

    The rule-based result can cite rows outside the bundle (e.g. the newest high-severity
    event, picked from all of the device's alarms, not only the 10 most recent).
    """
    known = bundle_values(bundle)
    for e in (rules or {}).get("evidence") or []:
        if isinstance(e.get("source"), str) and isinstance(e.get("field"), str):
            known.setdefault((e["source"], e["field"]), e.get("value"))
    return known


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def _as_number(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            return None
    return None


def same_value(given: Any, expected: Any) -> bool:
    """Equal as numbers if both are numeric, else equal as whitespace-normalised text."""
    if given is None or expected is None:
        return given is None and expected is None
    a, b = _as_number(given), _as_number(expected)
    if a is not None and b is not None:
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return " ".join(str(given).split()) == " ".join(str(expected).split())


def _check_item(
    src: Any,
    fld: Any,
    val: Any,
    known: dict[tuple[str, str], Any],
    bundle_sources: set[str],
    rows: dict[str, dict[str, Any]] | None,
) -> tuple[str, Any] | None:
    """None if the item is verified, else (kind, expected value or None)."""
    if not isinstance(src, str) or not isinstance(fld, str):
        return ("malformed_item", None)
    if rows is not None:  # live: the Excel is the reference
        if src not in rows:
            return ("missing_source", None)
        if src not in bundle_sources:
            return ("not_in_object_evidence", None)
        if fld not in rows[src]:
            return ("unknown_field", None)
        expected = _json(rows[src][fld])
    else:  # offline: the snapshot is the reference
        if src not in bundle_sources:
            return ("missing_source", None)
        if (src, fld) not in known:
            return ("unknown_field", None)
        expected = known[(src, fld)]
    return None if same_value(val, expected) else ("value_mismatch", expected)


def iter_refs(node: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """(path, "Sheet!R<row>") for every reference in strings outside SKIP_KEYS."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k not in SKIP_KEYS:
                yield from iter_refs(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from iter_refs(v, f"{path}[{i}]")
    elif isinstance(node, str):
        for ref in REF_RE.findall(node):
            yield path, ref


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #
def check_citations(
    result: dict[str, Any],
    *,
    offline: bool = False,
    evidence: dict[str, Any] | None = None,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify a diagnosis' citations.

    ``evidence`` (the object's bundle) and ``rules`` (its rule-based result) are what the
    object was given; both are loaded for ``result["object_id"]`` when omitted. Values are
    always compared with the Excel (live) or the snapshot (offline), never with ``result``.
    """
    object_id = result["object_id"]
    bundle = evidence if evidence is not None else _load_bundle(object_id, offline)
    if bundle is None:
        return {
            "ok": False,
            "checked": 0,
            "failed": [{"kind": "no_reference_data", "detail": "no evidence for this object"}],
        }
    if rules is None:
        rules = _load_rules(object_id, offline)
    known = reference_values(bundle, rules)
    bundle_sources = {src for src, _ in known}
    rows = None if offline else _rows_by_source(workbook_path())

    failed: list[dict[str, Any]] = []
    verified_sources: set[str] = set()
    checked = 0
    for i, item in enumerate(result.get("evidence") or []):
        checked += 1
        item = item if isinstance(item, dict) else {}
        src, fld, val = item.get("source"), item.get("field"), item.get("value")
        problem = _check_item(src, fld, val, known, bundle_sources, rows)
        if problem is None:
            verified_sources.add(src)
            continue
        entry = {"kind": problem[0], "index": i, "source": src, "field": fld, "value": val}
        if problem[0] == "value_mismatch":
            entry["expected"] = problem[1]
        failed.append(entry)

    for where, ref in dict.fromkeys(iter_refs(result)):  # dedupe, keep order
        checked += 1
        if ref not in verified_sources:
            failed.append({"kind": "uncited_reference", "source": ref, "where": where})

    return {"ok": not failed, "checked": checked, "failed": failed}


def drop_failed_evidence(result: dict[str, Any], report: dict[str, Any]) -> None:
    """Remove the evidence items the report marked as failed (in place)."""
    bad = {f["index"] for f in report["failed"] if "index" in f}
    result["evidence"] = [e for i, e in enumerate(result.get("evidence") or []) if i not in bad]
