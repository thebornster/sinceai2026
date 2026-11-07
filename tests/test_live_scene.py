import shutil

import numpy as np
import pytest

from valmet.common import NavigationPath
from valmet.common.artifacts import (
    CROPS_DIR,
    MANIFEST_FILE,
    SEGMENT_POINTS_FILE,
    MOCK_SCENE_DIR,
    OBJECTS_FILE,
    PATHS_FILE,
    load_list,
    load_model,
    save_list,
    save_model,
)
from valmet.common.contracts import CoordinateSystem, DetectedObject, SceneManifest
from valmet.integration.live_scene import (
    VIEWER_SPLATS_FILE,
    _shortest_routes,
    load_live_scene,
)
from valmet.metadata.mapping import METADATA_DIR_ENV
from valmet.viewer.panels import format_diagnosis, format_metadata


@pytest.fixture(autouse=True)
def _mock_metadata(monkeypatch):
    monkeypatch.delenv(METADATA_DIR_ENV, raising=False)


@pytest.fixture(scope="module")
def scene():
    return load_live_scene(offline=True)


def by_id(scene, object_id):
    return next(obj for obj in scene.objects if obj.object_id == object_id)


def copy_mock(tmp_path):
    root = tmp_path / "scene"
    shutil.copytree(MOCK_SCENE_DIR, root)
    return root


def test_joins_every_object_with_its_equipment(scene):
    assert [obj.object_id for obj in scene.objects] == ["obj_0001", "obj_0002", "obj_0003"]
    pump = by_id(scene, "obj_0001")
    assert pump.asset["device"]["tag"] == "P-002"
    assert "CAVITATION_RISK" in pump.diagnosis["summary"]


def test_faulty_means_not_healthy_in_the_equipment_data(scene):
    assert [obj.object_id for obj in scene.faulty] == ["obj_0001"]


def test_routes_come_from_paths_json(scene):
    for obj in scene.objects:
        assert obj.route.target_object_id == obj.object_id


def test_object_missing_from_paths_json_has_no_route(tmp_path):
    root = copy_mock(tmp_path)
    paths = load_list(root / PATHS_FILE, NavigationPath)
    save_list(root / PATHS_FILE, [p for p in paths if p.target_object_id != "obj_0002"])
    scene = load_live_scene(root, offline=True)
    assert by_id(scene, "obj_0002").route is None
    assert by_id(scene, "obj_0001").route is not None


def test_shortest_route_wins():
    def path(*waypoints):
        return NavigationPath(
            scene_id="s",
            start=waypoints[0],
            target_object_id="a",
            inspection_position=waypoints[-1],
            waypoints=list(waypoints),
        )

    long = path((0, 0, 0), (5, 0, 0))
    short = path((0, 0, 0), (1, 0, 0))
    assert _shortest_routes([long, short])["a"] is short
    assert _shortest_routes(None) == {}


def test_names_are_unique_and_handle_missing_labels(scene):
    assert by_id(scene, "obj_0001").name == "P-002 pump (obj_0001)"
    assert by_id(scene, "obj_0003").name == "CV-001 unclassified (obj_0003)"
    assert len({obj.name for obj in scene.objects}) == len(scene.objects)


def test_object_without_a_record_stays_in_the_scene(tmp_path):
    root = copy_mock(tmp_path)
    objects = load_list(root / OBJECTS_FILE, DetectedObject)
    extra = objects[0].model_copy(update={"object_id": "obj_9999"})
    save_list(root / OBJECTS_FILE, [*objects, extra])

    obj = by_id(load_live_scene(root, offline=True), "obj_9999")
    assert obj.asset is None and obj.diagnosis is None and not obj.is_faulty
    assert obj.name == "pump (obj_9999)"
    assert "no record" in format_metadata(obj)
    assert "No diagnosis" in format_diagnosis(obj)


def test_metadata_from_another_scene_is_not_shown(tmp_path):
    root = copy_mock(tmp_path)
    for name, cls in ((OBJECTS_FILE, DetectedObject), (PATHS_FILE, NavigationPath)):
        items = load_list(root / name, cls)
        save_list(root / name, [item.model_copy(update={"scene_id": "other"}) for item in items])
    (root / "navmap.json").unlink()
    (root / "asset_mapping.json").unlink()
    manifest = load_model(root / MANIFEST_FILE, SceneManifest)
    save_model(root / MANIFEST_FILE, manifest.model_copy(update={"scene_id": "other"}))

    assert all(obj.asset is None for obj in load_live_scene(root, offline=True).objects)


def test_inconsistent_artifacts_are_rejected(tmp_path):
    root = copy_mock(tmp_path)
    paths = load_list(root / PATHS_FILE, NavigationPath)
    ghost = paths[0].model_copy(update={"target_object_id": "ghost"})
    save_list(root / PATHS_FILE, [ghost])
    with pytest.raises(ValueError, match="ghost"):
        load_live_scene(root, offline=True)


def test_scene_without_objects_is_rejected(tmp_path):
    root = copy_mock(tmp_path)
    for name in (OBJECTS_FILE, PATHS_FILE, "asset_mapping.json"):
        (root / name).unlink()
    with pytest.raises(ValueError, match="no objects"):
        load_live_scene(root, offline=True)


def test_non_z_up_world_is_rejected(tmp_path):
    root = copy_mock(tmp_path)
    (root / "navmap.json").unlink()
    manifest = load_model(root / MANIFEST_FILE, SceneManifest)
    frame = CoordinateSystem(up_axis="+Y", units="meters", meters_per_unit=1.0, verified=True)
    save_model(root / MANIFEST_FILE, manifest.model_copy(update={"coordinate_system": frame}))
    with pytest.raises(ValueError, match="Z-up"):
        load_live_scene(root, offline=True)


def test_panels_cite_excel_rows(scene):
    pump = by_id(scene, "obj_0001")
    metadata = format_metadata(pump)
    assert "P-002 (ASSET-014)" in metadata
    assert "attention required, health 45/100" in metadata
    assert "Current_State!R15" in metadata

    diagnosis = format_diagnosis(pump)
    assert "CAVITATION_RISK" in diagnosis
    assert "**Next steps**" in diagnosis
    assert "[Current_State!R15]" in diagnosis


def test_panels_handle_a_missing_confidence_and_a_healthy_object(scene):
    unlabeled = by_id(scene, "obj_0003")
    assert "n/a" in format_metadata(unlabeled)
    assert "unclassified" in format_metadata(unlabeled)
    assert "**Likely cause**" not in format_diagnosis(by_id(scene, "obj_0002"))


def test_mock_scene_has_no_splats_or_crops(scene):
    assert scene.splats is None
    assert all(obj.crop is None for obj in scene.objects)


def test_splats_and_crops_are_loaded_when_present(tmp_path):
    root = copy_mock(tmp_path)
    np.savez(
        root / VIEWER_SPLATS_FILE,
        centers=np.zeros((2, 3), np.float32),
        covariances=np.zeros((2, 3, 3), np.float32),
        rgbs=np.ones((2, 3), np.float32),
        opacities=np.ones((2, 1), np.float32),
    )
    (root / CROPS_DIR).mkdir(parents=True)
    (root / CROPS_DIR / "obj_0001.png").write_bytes(b"png")

    scene = load_live_scene(root, offline=True)
    assert scene.splats["covariances"].shape == (2, 3, 3)
    assert by_id(scene, "obj_0001").crop == root / CROPS_DIR / "obj_0001.png"
    assert by_id(scene, "obj_0002").crop is None
    assert load_live_scene(root, offline=True, load_splats=False).splats is None


def test_pick_points_keep_only_object_points(tmp_path):
    root = copy_mock(tmp_path)
    (root / SEGMENT_POINTS_FILE).parent.mkdir(parents=True)
    np.savez(
        root / SEGMENT_POINTS_FILE,
        xyz=np.arange(15, dtype=np.float32).reshape(5, 3),
        labels=np.array([0, -1, 2, -2, 7]),  # 7: stale, beyond the 3 objects
    )
    scene = load_live_scene(root, offline=True)
    np.testing.assert_array_equal(scene.pick_objects, [0, 2])
    assert scene.pick_points.shape == (2, 3)
