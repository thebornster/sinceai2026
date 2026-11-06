"""Shared helpers for the metadata/diagnostics tests (Dev 3). Not a test module."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from valmet.common.artifacts import OBJECTS_FILE, save_list
from valmet.common.contracts import DetectedObject, OrientedBox
from valmet.metadata.excel_loader import workbook_path
from valmet.metadata.mapping import DETAILS_FILE, METADATA_FIXTURES_DIR

MOCK_METADATA_DIR = METADATA_FIXTURES_DIR / "mock_equipment_room"
MOCK_DETAILS = json.loads((MOCK_METADATA_DIR / DETAILS_FILE).read_text(encoding="utf-8"))
MOCK_OBJECT_IDS = [d["object_id"] for d in MOCK_DETAILS]

requires_workbook = pytest.mark.skipif(
    not workbook_path().exists(),
    reason="equipment workbook not present (data/raw/equipment.xlsx or VALMET_EQUIPMENT_XLSX)",
)

# 25 objects: enough to exhaust healthy pumps/motors and force clones.
BIG_SCENE_LABELS = (
    ["pump"] * 4
    + ["motor"] * 4
    + ["valve"] * 6
    + ["pressure_transmitter"] * 2
    + ["vibration_sensor"] * 2
    + ["pipe"] * 3
    + ["tank"] * 2
    + [None] * 2
)


def make_objects(labels: list[str | None], scene_id: str = "test_scene") -> list[DetectedObject]:
    rng = random.Random(0)
    labels = list(labels)
    rng.shuffle(labels)
    out = []
    for i, label in enumerate(labels, start=1):
        center = (rng.uniform(0, 10), rng.uniform(0, 8), rng.uniform(0.3, 2.0))
        out.append(
            DetectedObject(
                object_id=f"obj_{i:04d}",
                scene_id=scene_id,
                label=label,
                confidence=None if label is None else 0.8,
                center=center,
                bbox=OrientedBox(center=center, extent=(0.5, 0.5, 0.5)),
            )
        )
    return out


def write_scene(dir_: Path, objects: list[DetectedObject]) -> Path:
    save_list(Path(dir_) / OBJECTS_FILE, objects)
    return Path(dir_)
