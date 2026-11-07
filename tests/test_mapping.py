import re

import pytest

from metadata_testutils import (
    BIG_SCENE_LABELS,
    MOCK_METADATA_DIR,
    make_objects,
    requires_workbook,
    write_scene,
)
from valmet.common.artifacts import (
    ASSET_MAPPING_FILE,
    MOCK_SCENE_DIR,
    OBJECTS_FILE,
    SceneArtifacts,
    load_list,
)
from valmet.common.contracts import AssetMapping, DetectedObject
from valmet.metadata.mapping import (
    DETAILS_FILE,
    METADATA_FIXTURES_DIR,
    PROCESSED_DIR,
    build_mapping,
    check_against_scene,
    default_output_dir,
    label_asset_class,
    resolve_target,
    run,
    to_contract,
)
from valmet.metadata.registry import AssetRegistry

pytestmark = requires_workbook


@pytest.fixture(scope="module")
def registry():
    return AssetRegistry()


@pytest.fixture(scope="module")
def mock_objects():
    return load_list(MOCK_SCENE_DIR / OBJECTS_FILE, DetectedObject)


@pytest.fixture(scope="module")
def big_objects():
    return make_objects(BIG_SCENE_LABELS)


@pytest.fixture(scope="module")
def big(big_objects, registry):
    return build_mapping(big_objects, registry, target_object=_first_pump(big_objects))


def _first_pump(objects):
    return sorted(o.object_id for o in objects if o.label == "pump")[0]


# --------------------------------------------------------------------------- #
# Mock scene: committed outputs
# --------------------------------------------------------------------------- #
def test_committed_mock_outputs_are_current(tmp_path, registry):
    details, out = run(MOCK_SCENE_DIR, out_dir=tmp_path, registry=registry)
    for name in (ASSET_MAPPING_FILE, DETAILS_FILE):
        assert (tmp_path / name).read_bytes() == (MOCK_METADATA_DIR / name).read_bytes(), name


def test_mock_contract_file_validates_and_is_consistent():
    mappings = load_list(MOCK_METADATA_DIR / ASSET_MAPPING_FILE, AssetMapping)
    arts = SceneArtifacts.load(MOCK_SCENE_DIR)
    arts.asset_mappings = mappings  # ours replaces their placeholder mapping
    assert arts.check_consistency() == []
    assert {m.scene_object_id for m in mappings} == {o.object_id for o in arts.objects}


def test_mock_target_is_obj_0001_asset_014(mock_objects, registry):
    details = build_mapping(mock_objects, registry, target_object="obj_0001")
    targets = [d for d in details if d["is_target"]]
    assert len(targets) == 1
    t = targets[0]
    assert (t["object_id"], t["device_id"], t["source_asset_id"]) == (
        "obj_0001",
        "ASSET-014",
        "ASSET-014",
    )
    assert (t["tag"], t["mapping"], t["method"]) == ("P-002", "target", "procedural")


def test_mock_unlabelled_object_gets_random_healthy_device(mock_objects, registry):
    details = {d["object_id"]: d for d in build_mapping(mock_objects, registry, "obj_0001")}
    d = details["obj_0003"]
    assert d["object_class"] is None
    assert d["method"] == "random"
    assert d["condition_state"] == "healthy"


# --------------------------------------------------------------------------- #
# Target resolution
# --------------------------------------------------------------------------- #
def test_target_defaults_to_path_target(mock_objects):
    assert resolve_target(mock_objects, scene_dir=MOCK_SCENE_DIR) == "obj_0001"


def test_explicit_target_wins_over_path(mock_objects, registry):
    assert resolve_target(mock_objects, "obj_0002", MOCK_SCENE_DIR) == "obj_0002"
    details = build_mapping(mock_objects, registry, target_object="obj_0002")
    assert [d["object_id"] for d in details if d["is_target"]] == ["obj_0002"]


def test_explicit_unknown_target_raises(mock_objects):
    with pytest.raises(ValueError, match="obj_9999"):
        resolve_target(mock_objects, "obj_9999", MOCK_SCENE_DIR)


def test_target_falls_back_to_first_pump_without_path(tmp_path, big_objects):
    write_scene(tmp_path, big_objects)  # no paths.json
    assert resolve_target(big_objects, scene_dir=tmp_path) == _first_pump(big_objects)
    assert resolve_target(big_objects) == _first_pump(big_objects)


def test_no_target_possible_raises():
    with pytest.raises(ValueError, match="pump"):
        resolve_target(make_objects(["valve", None]))


def test_run_with_explicit_target_and_no_path(tmp_path, big_objects, registry):
    scene = write_scene(tmp_path / "scene", big_objects)
    target = sorted(o.object_id for o in big_objects if o.label == "pump")[-1]
    details, out = run(scene, target_object=target, out_dir=tmp_path / "out", registry=registry)
    assert [d["object_id"] for d in details if d["is_target"]] == [target]
    assert (out / ASSET_MAPPING_FILE).exists() and (out / DETAILS_FILE).exists()


# --------------------------------------------------------------------------- #
# Labels and output folders
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "label, expected",
    [
        ("pump", "PROCESS_PUMP"),
        ("Centrifugal_Pump", "PROCESS_PUMP"),
        ("  PUMP ", "PROCESS_PUMP"),
        ("electric-motor", "ELECTRIC_MOTOR"),
        ("pressure_transmitter", "PRESSURE_TRANSMITTER"),
        ("Vibration Sensor", "VIBRATION_SENSOR"),
        ("pressure transmitter/gauge", "PRESSURE_TRANSMITTER"),
        ("Gauge", "PRESSURE_TRANSMITTER"),
        ("pipe", None),
        ("", None),
        (None, None),
    ],
)
def test_label_normalisation(label, expected):
    assert label_asset_class(label) == expected


def test_default_output_dirs():
    assert default_output_dir(MOCK_SCENE_DIR, "mock_equipment_room") == (
        METADATA_FIXTURES_DIR / "mock_equipment_room"
    )
    real = PROCESSED_DIR / "equipment_room"
    assert default_output_dir(real, "equipment_room") == real


# --------------------------------------------------------------------------- #
# Big synthetic scene: the join rules
# --------------------------------------------------------------------------- #
def test_every_object_mapped_exactly_once(big, big_objects):
    ids = [d["object_id"] for d in big]
    assert len(ids) == len(set(ids)) == len(big_objects)
    assert set(ids) == {o.object_id for o in big_objects}
    labels = {o.object_id: o.label for o in big_objects}
    assert all(d["object_class"] == labels[d["object_id"]] for d in big)


def test_non_targets_are_healthy(big, registry):
    assert sum(d["is_target"] for d in big) == 1
    for d in big:
        if not d["is_target"]:
            assert d["condition_state"] == "healthy"
            assert registry.get(d["source_asset_id"]).is_healthy


def test_fields_match_registry(big, registry):
    for d in big:
        rec = registry.get(d["source_asset_id"])
        assert (d["tag"], d["asset_class"], d["health_score"]) == (
            rec.identity.tag,
            rec.asset_class,
            rec.condition.health_score,
        )
        assert d["excel_refs"] == {
            "Asset_Master": rec.identity.source_ref,
            "Current_State": rec.condition.source_ref,
        }


def test_typed_objects_keep_their_class(big):
    for d in big:
        cls = label_asset_class(d["object_class"])
        if cls:
            assert d["asset_class"] == cls, d


def test_class_match_is_maximal(big, registry):
    used = {d["source_asset_id"] for d in big}
    for d in big:
        cls = label_asset_class(d["object_class"])
        if cls and d["mapping"] in ("clone", "synthetic"):
            assert all(r.asset_id in used for r in registry.by_class(cls) if r.is_healthy)


def test_device_ids_unique_and_clones_well_formed(big):
    assert len({d["device_id"] for d in big}) == len(big)
    assert any(d["mapping"] == "clone" for d in big)
    for d in big:
        if d["mapping"] == "clone":
            assert re.fullmatch(rf"{d['source_asset_id']}-c\d+", d["device_id"])
        else:
            assert d["device_id"] == d["source_asset_id"]


def test_contract_method_rules(big):
    for d in big:
        if d["is_target"]:
            assert d["method"] == "procedural"
        elif label_asset_class(d["object_class"]) == d["asset_class"]:
            assert d["method"] == "matched"
        else:
            assert d["method"] == "random"
    assert {d["method"] for d in big if d["object_class"] is None} == {"random"}


def test_contract_view_is_valid_and_consistent(big, big_objects, tmp_path):
    contract = to_contract(big)
    assert all(isinstance(m, AssetMapping) for m in contract)
    assert [m.equipment_id for m in contract] == [d["source_asset_id"] for d in big]
    assert check_against_scene(tmp_path, big_objects, contract) == []


def test_reproducible_and_order_independent(big, big_objects, registry):
    target = _first_pump(big_objects)
    again = build_mapping(list(reversed(big_objects)), AssetRegistry(), target_object=target)
    assert again == big
    other_seed = build_mapping(big_objects, registry, target_object=target, seed=7)
    assert other_seed != big  # the seed really matters
