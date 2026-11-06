"""Offline snapshot of everything valmet.api returns, so it works without the Excel.

``export_snapshot(scene_id)`` writes to ``data/fixtures/metadata/<scene_id>/``:

    assets.json         {object_id: get_asset output}
    evidence.json       {object_id: get_evidence output}
    diagnoses.json      {object_id: rule-based diagnose_fault output}
    snapshot_meta.json  excel_file, excel_sha256, created_at, schema_version

If the scene's mapping lives elsewhere (``data/processed/<scene_id>/``), the
mapping files are copied next to the snapshot so the folder is self-contained.

Run: python -m valmet.metadata.snapshot [scene_id]
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from valmet.common.artifacts import ASSET_MAPPING_FILE, REPO_ROOT
from valmet.metadata.excel_loader import WORKBOOK_ENV, workbook_path
from valmet.metadata.mapping import (
    DETAILS_FILE,
    METADATA_FIXTURES_DIR,
    PROCESSED_DIR,
    use_metadata_dir,
)

SNAPSHOT_SCHEMA_VERSION = "1.0"
ASSETS_FILE = "assets.json"
EVIDENCE_FILE = "evidence.json"
DIAGNOSES_FILE = "diagnoses.json"
META_FILE = "snapshot_meta.json"
SNAPSHOT_FILES = (ASSETS_FILE, EVIDENCE_FILE, DIAGNOSES_FILE, META_FILE)
# The files api.py needs in snapshot mode (evidence.json is for consumers/debugging).
REQUIRED_FOR_API = (ASSETS_FILE, DIAGNOSES_FILE, META_FILE)


class MetadataUnavailableError(FileNotFoundError):
    """Neither the Excel (live mode) nor a usable snapshot is available."""


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def how_to_create(scene_id: str = "mock_equipment_room") -> str:
    return (
        f"Create it with `python -m valmet.metadata.snapshot {scene_id}` on a machine that has "
        f"the workbook (data/raw/equipment.xlsx or ${WORKBOOK_ENV})."
    )


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_snapshot(
    scene_id: str,
    out_dir: Path | None = None,
    source_dir: Path | None = None,
) -> Path:
    """Compute every API result live and write the snapshot. Needs the workbook."""
    # Imported here: diagnostics imports metadata, so keep module import order acyclic.
    from valmet.diagnostics.diagnose import diagnose_fault
    from valmet.diagnostics.evidence import asset_view, get_evidence, mapped_object_ids

    out_dir = Path(out_dir) if out_dir else METADATA_FIXTURES_DIR / scene_id
    if source_dir is None:
        source_dir = out_dir if (out_dir / DETAILS_FILE).exists() else PROCESSED_DIR / scene_id
    source_dir = Path(source_dir)
    if not (source_dir / DETAILS_FILE).exists():
        raise FileNotFoundError(
            f"{_rel(source_dir / DETAILS_FILE)} not found. Run "
            "`python -m valmet.metadata.mapping` for this scene first."
        )
    wb = workbook_path()
    if not wb.exists():
        raise MetadataUnavailableError(
            f"Cannot export a snapshot: workbook {_rel(wb)} is missing. "
            f"Put it at data/raw/equipment.xlsx or set ${WORKBOOK_ENV}."
        )

    with use_metadata_dir(source_dir):
        ids = mapped_object_ids()
        assets = {oid: asset_view(oid) for oid in ids}
        evidence = {oid: get_evidence(oid) for oid in ids}
        diagnoses = {oid: diagnose_fault(oid) for oid in ids}

    wrong_scene = sorted({a["scene_id"] for a in assets.values()} - {scene_id})
    if wrong_scene:
        raise ValueError(f"Mapping in {_rel(source_dir)} is for scene(s) {wrong_scene}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if source_dir.resolve() != out_dir.resolve():
        for name in (ASSET_MAPPING_FILE, DETAILS_FILE):
            if (source_dir / name).exists():
                shutil.copyfile(source_dir / name, out_dir / name)
    _write_json(out_dir / ASSETS_FILE, assets)
    _write_json(out_dir / EVIDENCE_FILE, evidence)
    _write_json(out_dir / DIAGNOSES_FILE, diagnoses)
    _write_json(
        out_dir / META_FILE,
        {
            "excel_file": wb.name,
            "excel_sha256": _sha256(wb),
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
        },
    )
    return out_dir


# --------------------------------------------------------------------------- #
# Read
# --------------------------------------------------------------------------- #
def missing_snapshot_files(snapshot_dir: Path) -> list[Path]:
    return [
        Path(snapshot_dir) / n for n in REQUIRED_FOR_API if not (Path(snapshot_dir) / n).exists()
    ]


@lru_cache(maxsize=8)
def _load(snapshot_dir: Path, stamp: tuple[int, ...]) -> dict[str, Any]:
    data = {
        "assets": json.loads((snapshot_dir / ASSETS_FILE).read_text(encoding="utf-8")),
        "diagnoses": json.loads((snapshot_dir / DIAGNOSES_FILE).read_text(encoding="utf-8")),
        "meta": json.loads((snapshot_dir / META_FILE).read_text(encoding="utf-8")),
    }
    version = data["meta"].get("schema_version")
    if version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"{_rel(snapshot_dir / META_FILE)} has schema_version {version!r}, "
            f"expected {SNAPSHOT_SCHEMA_VERSION!r}. Re-export the snapshot."
        )
    return data


def load_snapshot(snapshot_dir: Path) -> dict[str, Any]:
    """Snapshot contents (cached until a file changes). Callers must not mutate it."""
    snapshot_dir = Path(snapshot_dir)
    missing = missing_snapshot_files(snapshot_dir)
    if missing:
        raise MetadataUnavailableError(
            f"Snapshot incomplete: missing {', '.join(_rel(p) for p in missing)}. "
            + how_to_create(snapshot_dir.name)
        )
    stamp = tuple((snapshot_dir / n).stat().st_mtime_ns for n in REQUIRED_FOR_API)
    return _load(snapshot_dir, stamp)


def snapshot_record(snapshot_dir: Path, kind: str, object_id: str) -> dict[str, Any] | None:
    """A deep copy of one stored result (kind: "assets" or "diagnoses"), or None."""
    record = (
        load_snapshot(snapshot_dir)[kind].get(object_id) if isinstance(object_id, str) else None
    )
    return copy.deepcopy(record) if record is not None else None


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    scene_id = args[0] if args else "mock_equipment_room"
    out = export_snapshot(scene_id)
    meta = json.loads((out / META_FILE).read_text(encoding="utf-8"))
    n = len(json.loads((out / ASSETS_FILE).read_text(encoding="utf-8")))
    print(f"Wrote snapshot for {n} objects to {_rel(out)}/")
    for name in SNAPSHOT_FILES:
        print(f"  {name:<20} {(out / name).stat().st_size:>8} bytes")
    print(f"  excel: {meta['excel_file']} sha256={meta['excel_sha256'][:16]}…")


if __name__ == "__main__":
    main()
