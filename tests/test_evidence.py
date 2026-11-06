import json

import pytest

from metadata_testutils import (
    BIG_SCENE_LABELS,
    MOCK_OBJECT_IDS,
    make_objects,
    requires_workbook,
    write_scene,
)
from valmet.diagnostics.evidence import UnknownObjectError, get_evidence
from valmet.metadata.excel_loader import load_workbook
from valmet.metadata.mapping import METADATA_DIR_ENV, run

pytestmark = requires_workbook
ALL_OBJECTS = MOCK_OBJECT_IDS


@pytest.fixture(autouse=True)
def _mock_scene(monkeypatch):
    monkeypatch.delenv(METADATA_DIR_ENV, raising=False)  # always test against the mock


@pytest.fixture(scope="module")
def big_scene(tmp_path_factory):
    """25-object scene with clones; returns (metadata dir, details)."""
    root = tmp_path_factory.mktemp("big")
    objects = make_objects(BIG_SCENE_LABELS)
    target = sorted(o.object_id for o in objects if o.label == "pump")[0]
    scene = write_scene(root / "scene", objects)
    details, out = run(scene, target_object=target, out_dir=root / "meta")
    return out, details


@pytest.fixture(scope="module")
def rows_by_source():
    """'Sheet!R<n>' -> the full row dict, for every loaded sheet."""
    out = {}
    for df in load_workbook().values():
        for row in df.to_dict("records"):
            out[row["_source"]] = row
    return out


def _sources(obj):
    """Yield every Sheet!R<row> string anywhere in the evidence."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("_source", "_sources", "excel_refs"):
                yield from (v.values() if isinstance(v, dict) else [v])
            else:
                yield from _sources(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _sources(v)


def test_obj_0001_is_p002():
    ev = get_evidence("obj_0001")
    assert ev["device"]["tag"] == "P-002"
    assert ev["device"]["asset_id"] == "ASSET-014"
    assert ev["condition"]["issue_code"] == "CAVITATION_RISK"
    assert ev["condition"]["_source"] == "Current_State!R15"


@pytest.mark.parametrize("object_id", ALL_OBJECTS)
def test_json_serialisable(object_id):
    json.dumps(get_evidence(object_id))


@pytest.mark.parametrize("object_id", ALL_OBJECTS)
def test_every_source_exists_and_belongs_to_device(object_id, rows_by_source):
    ev = get_evidence(object_id)
    asset_id = ev["device"]["asset_id"]
    sources = list(_sources(ev))
    assert sources
    for src in sources:
        assert src in rows_by_source, src
        row = rows_by_source[src]
        if "asset_id" in row:
            assert row["asset_id"] == asset_id, src
    # Sheets without asset_id are checked via their own keys.
    if ev["knowledge"]:
        assert rows_by_source[ev["knowledge"]["_source"]]["doc_id"] == ev["knowledge"]["doc_id"]
    for case in ev["similar_cases"]:
        row = rows_by_source[case["_source"]]
        assert row["case_id"] == case["case_id"]
        assert row["asset_class"] == ev["device"]["asset_class"]


@pytest.mark.parametrize("object_id", ALL_OBJECTS)
def test_cited_values_match_excel(object_id, rows_by_source):
    ev = get_evidence(object_id)
    for section in ("device", "condition"):
        row = rows_by_source[ev[section]["_source"]]
        for k, v in ev[section].items():
            if k in row and not hasattr(row[k], "isoformat"):
                assert v == row[k], (section, k)
    for col, s in ev["telemetry"].items():
        for which in ("first", "last", "min", "max"):
            assert rows_by_source[s["_sources"][which]][col] == s[which], (col, which)
    for a in ev["alarms"]["recent"]:
        row = rows_by_source[a["_source"]]
        assert (a["severity"], a["event_code"]) == (row["severity"], row["event_code"])


@pytest.mark.parametrize("object_id", ALL_OBJECTS)
def test_telemetry_stats_consistent(object_id):
    for col, s in get_evidence(object_id)["telemetry"].items():
        assert s["min"] <= s["mean"] <= s["max"], col
        assert s["min"] <= s["first"] <= s["max"]
        assert s["min"] <= s["last"] <= s["max"]
        assert s["change"] == pytest.approx(s["last"] - s["first"])
        assert s["first_at"] <= s["last_at"]


def test_pump_telemetry_columns():
    assert set(get_evidence("obj_0001")["telemetry"]) == {
        "process_pressure_bar",
        "process_flow_pct",
        "vibration_mm_s",
        "bearing_temp_c",
        "motor_current_a",
        "speed_rpm",
        "signal_quality_pct",
    }


def test_alarms_and_maintenance_shape():
    ev = get_evidence("obj_0001")
    assert sum(ev["alarms"]["by_severity"].values()) == ev["alarms"]["total"] == 39
    recent = ev["alarms"]["recent"]
    assert len(recent) == 10
    assert [a["timestamp"] for a in recent] == sorted(
        (a["timestamp"] for a in recent), reverse=True
    )
    mh = ev["maintenance"]
    assert len(mh) == 5
    assert [m["opened_at"] for m in mh] == sorted((m["opened_at"] for m in mh), reverse=True)


def test_clone_returns_source_device_evidence(big_scene, monkeypatch):
    meta_dir, details = big_scene
    monkeypatch.setenv(METADATA_DIR_ENV, str(meta_dir))
    clone = next(m for m in details if m["mapping"] == "clone")
    original = next(m for m in details if m["device_id"] == clone["source_asset_id"])
    ev_c, ev_o = get_evidence(clone["object_id"]), get_evidence(original["object_id"])
    assert ev_c["device"]["asset_id"] == clone["source_asset_id"]
    assert ev_c["object"]["device_id"] == clone["device_id"]  # keeps its own derived id
    for section in (
        "device",
        "condition",
        "telemetry",
        "alarms",
        "maintenance",
        "similar_cases",
        "knowledge",
    ):
        assert ev_c[section] == ev_o[section], section


def test_unknown_object_raises_clear_error():
    with pytest.raises(UnknownObjectError, match="obj_9999"):
        get_evidence("obj_9999")
    with pytest.raises(UnknownObjectError, match="obj_014"):  # old 3-digit id format
        get_evidence("obj_014")
    with pytest.raises(KeyError):  # still a KeyError for generic callers
        get_evidence("not-an-id")


def test_big_scene_every_source_belongs_to_device(big_scene, monkeypatch, rows_by_source):
    meta_dir, details = big_scene
    monkeypatch.setenv(METADATA_DIR_ENV, str(meta_dir))
    for d in details:
        ev = get_evidence(d["object_id"])
        assert ev["device"]["asset_id"] == d["source_asset_id"]
        for src in _sources(ev):
            row = rows_by_source[src]
            if "asset_id" in row:
                assert row["asset_id"] == d["source_asset_id"], src


def test_missing_metadata_dir_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv(METADATA_DIR_ENV, str(tmp_path / "nothing"))
    with pytest.raises(FileNotFoundError, match="valmet.metadata.mapping"):
        get_evidence("obj_0001")
