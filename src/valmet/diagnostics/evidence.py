"""Collect all Excel evidence for one scene object, with source refs.

Every value read from the workbook carries ``_source`` ("Sheet!R<row>"), so
the diagnosis can cite it and the citation checker can verify it.
Run: python -m valmet.diagnostics.evidence [object_id]
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from valmet.metadata.excel_loader import SOURCE_COL, get_sheet
from valmet.metadata.mapping import DETAILS_FILE, metadata_dir
from valmet.metadata.registry import AssetRecord, SourcedRow, get_registry

N_RECENT_ALARMS = 10
N_RECENT_WORK_ORDERS = 5
SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}

DEVICE_FIELDS = (
    "tag",
    "asset_class",
    "area_name",
    "criticality",
    "last_service_date",
    "service_interval_days",
)
CONDITION_FIELDS = (
    "health_score",
    "condition_state",
    "issue_code",
    "issue_summary",
    "urgency",
    "recommended_next_check",
    "model_confidence",
)
ALARM_FIELDS = ("timestamp", "event_type", "severity", "event_code", "description", "status")
MAINTENANCE_FIELDS = ("opened_at", "action", "maintenance_type", "result")
CASE_FIELDS = (
    "case_id",
    "asset_class",
    "historical_issue_code",
    "historical_interpretation",
    "recommended_next_check",
    "safe_evidence_workflow",
    "severity",
)


class UnknownObjectError(KeyError):
    """Raised when an object_id is not in asset_mapping.json."""

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _json(value: Any) -> Any:
    """Make a value JSON-serialisable (datetimes -> ISO strings, NaN -> None)."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):  # numpy scalar
        return _json(value.item())
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    return value


def _pick(row: SourcedRow, fields: tuple[str, ...]) -> dict[str, Any]:
    data = row.model_dump()
    out = {f: _json(data.get(f)) for f in fields}
    out[SOURCE_COL] = row.source_ref
    return out


def _details_path() -> Path:
    return metadata_dir() / DETAILS_FILE


def _mapping_by_object() -> dict[str, dict]:
    return _load_details(_details_path())


@lru_cache(maxsize=4)
def _load_details(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m valmet.metadata.mapping` first, "
            "or set VALMET_METADATA_DIR to the scene's metadata folder."
        )
    entries = json.loads(path.read_text(encoding="utf-8"))
    return {e["object_id"]: e for e in entries}


@lru_cache(maxsize=1)
def _numeric_telemetry_columns() -> tuple[str, ...]:
    df = get_sheet("Telemetry_Sample_3d")
    return tuple(c for c in df.columns if df[c].dtype.kind in "fi")


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def _telemetry_summary(rec: AssetRecord) -> dict[str, Any]:
    rows = [(t.timestamp, t.source_ref, t.model_dump()) for t in rec.telemetry]  # oldest first
    out: dict[str, Any] = {}
    for col in _numeric_telemetry_columns():
        points = [
            (ts, src, d[col])
            for ts, src, d in rows
            if d.get(col) is not None and not (isinstance(d[col], float) and math.isnan(d[col]))
        ]
        if not points:
            continue  # column not measured for this device class
        first, last = points[0], points[-1]
        lo = min(points, key=lambda p: p[2])  # first occurrence on ties
        hi = max(points, key=lambda p: p[2])
        values = [p[2] for p in points]
        out[col] = {
            "n": len(values),
            "first": first[2],
            "last": last[2],
            "min": lo[2],
            "max": hi[2],
            "mean": round(sum(values) / len(values), 4),
            "change": round(last[2] - first[2], 6),
            "first_at": first[0].isoformat(),
            "last_at": last[0].isoformat(),
            "min_at": lo[0].isoformat(),
            "max_at": hi[0].isoformat(),
            "_sources": {"first": first[1], "last": last[1], "min": lo[1], "max": hi[1]},
        }
    return out


def _alarms(rec: AssetRecord) -> dict[str, Any]:
    by_severity = Counter(a.model_dump().get("severity") for a in rec.alarms)
    return {
        "total": len(rec.alarms),
        "by_severity": dict(
            sorted(by_severity.items(), key=lambda kv: (SEVERITY_RANK.get(kv[0], 99), str(kv[0])))
        ),
        "recent": [_pick(a, ALARM_FIELDS) for a in rec.alarms[:N_RECENT_ALARMS]],  # newest first
    }


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #
def resolve(object_id: str) -> tuple[dict[str, Any], AssetRecord]:
    """Return (mapping entry, registry record); clones resolve to their source device."""
    entry = _mapping_by_object().get(object_id) if isinstance(object_id, str) else None
    if entry is None:
        raise UnknownObjectError(
            f"Unknown object_id {object_id!r}: not in {_details_path()}. "
            "Expected an id like 'obj_0001'."
        )
    return _json(entry), get_registry().get(entry["source_asset_id"])


def device_section(rec: AssetRecord) -> dict[str, Any]:
    return {"asset_id": rec.asset_id, **_pick(rec.identity, DEVICE_FIELDS)}


def condition_section(rec: AssetRecord) -> dict[str, Any]:
    return _pick(rec.condition, CONDITION_FIELDS)


def mapped_object_ids() -> list[str]:
    """All object ids in the active scene's asset_mapping_details.json, in file order."""
    return list(_mapping_by_object())


def asset_view(object_id: str) -> dict[str, Any]:
    """Mapping entry plus device identity and condition (what api.get_asset returns)."""
    entry, rec = resolve(object_id)
    return {**entry, "device": device_section(rec), "condition": condition_section(rec)}


def get_evidence(object_id: str) -> dict[str, Any]:
    entry, rec = resolve(object_id)

    knowledge = None
    if rec.knowledge_doc is not None:
        knowledge = _pick(rec.knowledge_doc, ("doc_id", "title", "content"))

    return {
        "object": entry,
        "device": device_section(rec),
        "condition": condition_section(rec),
        "telemetry": _telemetry_summary(rec),
        "alarms": _alarms(rec),
        "maintenance": [
            _pick(m, MAINTENANCE_FIELDS) for m in rec.maintenance[:N_RECENT_WORK_ORDERS]
        ],
        "similar_cases": [_pick(c, CASE_FIELDS) for c in rec.similar_cases],
        "knowledge": knowledge,
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(get_evidence(sys.argv[1] if len(sys.argv) > 1 else "obj_0001"), indent=2))
