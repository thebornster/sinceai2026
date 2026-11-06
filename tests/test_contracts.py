from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from valmet.common import (
    AssetMapping,
    CoordinateSystem,
    DetectedObject,
    NavigationMapMetadata,
    NavigationPath,
    OrientedBox,
    RobotClearance,
    SceneManifest,
)
from valmet.common.artifacts import (
    MOCK_SCENE_DIR,
    OBJECTS_FILE,
    SceneArtifacts,
    load_list,
    load_model,
    save_list,
    save_model,
)

BOX = OrientedBox(center=(0, 0, 0), extent=(1, 1, 1))
NAV = dict(
    scene_id="s",
    origin=(0, 0, 0),
    resolution=0.1,
    shape=(10, 20),
    clearance=RobotClearance(radius=0.3, height=1.8),
)


def obj(**kw) -> DetectedObject:
    base = dict(object_id="o1", scene_id="s", center=(0, 0, 0), bbox=BOX)
    return DetectedObject(**{**base, **kw})


# ----------------------------------------------------------------- valid / defaults


def test_manifest_defaults_are_explicitly_unknown():
    m = SceneManifest(scene_id="equipment_room", source_filename="Equipment Room.ply")
    cs = m.coordinate_system
    assert cs.up_axis is None and cs.units == "unknown" and cs.meters_per_unit is None
    assert not cs.verified
    assert np.allclose(m.source_to_world, np.eye(4))
    assert m.schema_version == "1.0"


def test_unknown_label_object():
    o = obj()
    assert o.label is None and o.confidence is None and o.gaussians is None


# ----------------------------------------------------------------- round trips


@pytest.mark.parametrize(
    "model",
    [
        SceneManifest(scene_id="s", source_filename="x.ply", source_sha256="a" * 64),
        obj(
            label="pump",
            confidence=0.5,
            bbox=OrientedBox(
                center=(1, 2, 3),
                extent=(1, 2, 3),
                rotation=[[math.cos(1), -math.sin(1), 0], [math.sin(1), math.cos(1), 0], [0, 0, 1]],
            ),
        ),
        NavigationMapMetadata(**NAV),
        NavigationPath(
            scene_id="s",
            start=(0, 0, 0),
            target_object_id="o1",
            inspection_position=(1, 1, 0),
            waypoints=[(0, 0, 0), (1, 1, 0)],
        ),
        AssetMapping(scene_id="s", scene_object_id="o1", equipment_id="PMP-1", method="matched"),
    ],
)
def test_json_roundtrip(model, tmp_path: Path):
    assert type(model).model_validate_json(model.model_dump_json()) == model
    save_model(tmp_path / "m.json", model)
    assert load_model(tmp_path / "m.json", type(model)) == model


def test_list_roundtrip(tmp_path: Path):
    items = [obj(object_id=f"o{i}") for i in range(3)]
    save_list(tmp_path / "objs.json", items)
    assert load_list(tmp_path / "objs.json", DetectedObject) == items
    save_list(tmp_path / "empty.json", [])
    assert load_list(tmp_path / "empty.json", DetectedObject) == []


# ----------------------------------------------------------------- rejection


@pytest.mark.parametrize(
    "build",
    [
        lambda: obj(confidence=1.5, label="pump"),
        lambda: obj(confidence=0.5),  # confidence without label
        lambda: obj(center=(0, 0)),
        lambda: obj(center=(float("nan"), 0, 0)),
        lambda: obj(object_id="has space"),
        lambda: obj(object_id=""),
        lambda: obj(colour="red"),  # unknown field
        lambda: obj(schema_version="2.0"),
        lambda: OrientedBox(center=(0, 0, 0), extent=(-1, 1, 1)),
        lambda: OrientedBox(
            center=(0, 0, 0), extent=(1, 1, 1), rotation=[[2, 0, 0], [0, 1, 0], [0, 0, 1]]
        ),
        lambda: OrientedBox(
            center=(0, 0, 0), extent=(1, 1, 1), rotation=np.diag([1, 1, -1]).tolist()
        ),
        lambda: SceneManifest(
            scene_id="s", source_filename="x", source_to_world=np.eye(3).tolist()
        ),
        lambda: CoordinateSystem(verified=True),
        lambda: CoordinateSystem(units="meters", meters_per_unit=0.5),
        lambda: NavigationMapMetadata(**{**NAV, "resolution": 0}),
        lambda: NavigationMapMetadata(**{**NAV, "shape": (10, 0)}),
        lambda: NavigationMapMetadata(
            **NAV, cell_values={"free": 0, "occupied": 0, "unknown": 255}
        ),
        lambda: NavigationPath(
            scene_id="s",
            start=(0, 0, 0),
            target_object_id="o1",
            inspection_position=(1, 1, 0),
            waypoints=[(0, 0, 0), (2, 2, 0)],
        ),
        lambda: NavigationPath(
            scene_id="s",
            start=(0, 0, 0),
            target_object_id="o1",
            inspection_position=(0, 0, 0),
            waypoints=[],
        ),
        lambda: AssetMapping(scene_id="s", scene_object_id="o1", equipment_id="E", method="guess"),
    ],
)
def test_rejects_malformed(build):
    with pytest.raises(ValidationError):
        build()


def test_models_are_frozen():
    with pytest.raises(ValidationError):
        obj().label = "pump"


# ----------------------------------------------------------------- grid conventions


def test_navmap_cell_world_conversions():
    nav = NavigationMapMetadata(**{**NAV, "origin": (-1.0, 2.0, 0.0)})
    assert nav.world_to_cell([[-1.0, 2.0, 0.0]]).tolist() == [[0, 0]]
    assert nav.world_to_cell([[-0.95, 2.31]]).tolist() == [[0, 3]]
    centers = nav.cell_to_world([[0, 0], [9, 19]])
    assert np.allclose(centers, [[-0.95, 2.05], [-0.05, 3.95]])
    assert (nav.world_to_cell(centers) == [[0, 0], [9, 19]]).all()


# ----------------------------------------------------------------- mock artifacts


def test_mock_scene_is_consistent():
    arts = SceneArtifacts.load(MOCK_SCENE_DIR)
    assert arts.check_consistency() == []
    assert arts.objects and arts.navmap and arts.paths and arts.asset_mappings
    grid = arts.load_occupancy()
    v = arts.navmap.cell_values
    assert set(np.unique(grid)) <= {v.free, v.occupied, v.unknown}
    for path in arts.paths:  # mock waypoints sit in free cells
        cells = arts.navmap.world_to_cell(np.asarray(path.waypoints))
        assert (grid[cells[:, 0], cells[:, 1]] == v.free).all()


def test_mock_routes_stay_in_free_cells_between_waypoints():
    arts = SceneArtifacts.load(MOCK_SCENE_DIR)
    grid = arts.load_occupancy()
    for path in arts.paths:
        waypoints = np.asarray(path.waypoints)
        for start, end in zip(waypoints[:-1], waypoints[1:], strict=True):
            line = start + np.linspace(0.0, 1.0, 200)[:, None] * (end - start)
            cells = arts.navmap.world_to_cell(line)
            assert (grid[cells[:, 0], cells[:, 1]] == arts.navmap.cell_values.free).all()


def test_each_mock_file_loads_independently():
    """A consumer can read just the artifact it needs with plain JSON or the model."""
    raw = json.loads((MOCK_SCENE_DIR / OBJECTS_FILE).read_text())
    assert all(DetectedObject.model_validate(o) for o in raw)
    for name, cls in [
        ("scene_manifest.json", SceneManifest),
        ("navmap.json", NavigationMapMetadata),
    ]:
        load_model(MOCK_SCENE_DIR / name, cls)
    load_list(MOCK_SCENE_DIR / "paths.json", NavigationPath)
    load_list(MOCK_SCENE_DIR / "asset_mapping.json", AssetMapping)


def test_consistency_catches_cross_artifact_errors(tmp_path: Path):
    arts = SceneArtifacts.load(MOCK_SCENE_DIR)
    arts.root = tmp_path  # occupancy grid missing here
    arts.manifest = arts.manifest.model_copy(update={"coordinate_system": CoordinateSystem()})
    arts.paths = [arts.paths[0].model_copy(update={"target_object_id": "ghost"})]
    arts.asset_mappings = [arts.asset_mappings[0].model_copy(update={"scene_id": "other"})]
    problems = "\n".join(arts.check_consistency())
    for expected in ["up_axis", "grid file", "ghost", "other"]:
        assert expected in problems
