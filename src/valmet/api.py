"""Public interface of the metadata/diagnostics module.

Teammates call only these three functions:

    get_asset(object_id, offline=False)
        -> mapping entry + device + condition
    get_nearby_assets(object_ids, offline=False)
        -> list of get_asset results + status_line
    diagnose_fault(object_id, use_llm=False, offline=False, refresh=False)
        -> diagnosis with Excel citations (use_llm: LLM wording, citation-checked, cached)

All outputs are plain JSON-serialisable types. Every Excel value carries
``_source`` as "Sheet!R<row>".

Two modes, same results:
- live: the workbook exists (data/raw/equipment.xlsx or $VALMET_EQUIPMENT_XLSX),
  everything is computed from it.
- snapshot: the workbook is missing, or ``offline=True``; results are read from
  the snapshot files written by ``python -m valmet.metadata.snapshot``.

Which scene: $VALMET_METADATA_DIR (folder with asset_mapping_details.json and the
snapshot), default data/fixtures/metadata/mock_equipment_room/.

Run: python -m valmet.api
"""

from __future__ import annotations

from collections.abc import Iterable as _Iterable
from typing import Any as _Any

from valmet.diagnostics import diagnose as _diagnose
from valmet.diagnostics import evidence as _evidence
from valmet.diagnostics import llm_diagnose as _llm
from valmet.metadata import snapshot as _snapshot
from valmet.metadata.excel_loader import workbook_path as _workbook_path
from valmet.metadata.mapping import metadata_dir as _metadata_dir

__all__ = ["get_asset", "get_nearby_assets", "diagnose_fault"]


def _use_snapshot(offline: bool) -> bool:
    """True -> read the snapshot; False -> compute live. Raises if neither is possible."""
    snap_dir = _metadata_dir()
    missing_snap = _snapshot.missing_snapshot_files(snap_dir)
    if offline:
        if missing_snap:
            raise _snapshot.MetadataUnavailableError(
                "offline=True but the snapshot is incomplete: missing "
                + ", ".join(_snapshot._rel(p) for p in missing_snap)
                + ". "
                + _snapshot.how_to_create(snap_dir.name)
            )
        return True
    wb = _workbook_path()
    if wb.exists():
        return False
    if not missing_snap:
        return True
    raise _snapshot.MetadataUnavailableError(
        f"No equipment data: workbook {_snapshot._rel(wb)} is missing (live mode) and the "
        "snapshot is incomplete: missing "
        + ", ".join(_snapshot._rel(p) for p in missing_snap)
        + ". Either put the workbook at data/raw/equipment.xlsx (or set $VALMET_EQUIPMENT_XLSX), "
        "or create the snapshot: " + _snapshot.how_to_create(snap_dir.name)
    )


def _from_snapshot(kind: str, object_id: str) -> dict[str, _Any]:
    record = _snapshot.snapshot_record(_metadata_dir(), kind, object_id)
    if record is None:
        raise _evidence.UnknownObjectError(
            f"Unknown object_id {object_id!r}: not in "
            f"{_snapshot._rel(_metadata_dir() / _snapshot.ASSETS_FILE)}. "
            "Expected an id like 'obj_0001'."
        )
    return record


def get_asset(object_id: str, offline: bool = False) -> dict[str, _Any]:
    """Mapping entry plus device identity and condition, each with _source.

    Raises UnknownObjectError (a KeyError) if the object has no mapping.
    """
    if _use_snapshot(offline):
        return _from_snapshot("assets", object_id)
    return _evidence.asset_view(object_id)


def get_nearby_assets(object_ids: _Iterable[str], offline: bool = False) -> list[dict[str, _Any]]:
    """get_asset for each id, in input order, plus a one-line status.

    Unknown ids are skipped, since the viewer may pass unmapped objects.
    """
    out = []
    for object_id in object_ids:
        try:
            asset = get_asset(object_id, offline=offline)
        except _evidence.UnknownObjectError:
            continue
        dev, cond = asset["device"], asset["condition"]
        asset["status_line"] = (
            f"{dev['tag']} {dev['asset_class'].replace('_', ' ').lower()} — "
            f"{cond['condition_state'].replace('_', ' ')}, health {cond['health_score']} "
            f"({cond['_source']})"
        )
        out.append(asset)
    return out


def diagnose_fault(
    object_id: str, use_llm: bool = False, offline: bool = False, refresh: bool = False
) -> dict[str, _Any]:
    """Diagnosis for one object, grounded in Excel rows.

    use_llm=True: an LLM explanation in the same schema (``method: "llm"``), with every
    citation checked (``citation_check``) and the result cached. On any failure it returns
    the rule-based result with ``method: "rule_based_fallback"`` and ``fallback_reason``.
    With offline=True the LLM is never called: cached result, else the fallback.
    refresh=True ignores the cache and asks the LLM again.

    Raises UnknownObjectError (a KeyError) if the object has no mapping.
    """
    if not _use_snapshot(offline):
        return _diagnose.diagnose_fault(object_id, use_llm=use_llm, refresh=refresh)
    rules = _from_snapshot("diagnoses", object_id)
    if not use_llm:
        return rules
    snap_dir = _metadata_dir()
    return _llm.diagnose_with_llm(
        object_id,
        rules,
        _snapshot.snapshot_evidence(snap_dir, object_id),
        scene_id=_from_snapshot("assets", object_id)["scene_id"],
        mode="offline" if offline else "snapshot",
        refresh=refresh,
    )


if __name__ == "__main__":
    import json

    print(json.dumps(diagnose_fault("obj_0001"), indent=2, ensure_ascii=False))
