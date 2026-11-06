import json
import shutil

import pytest

from metadata_testutils import MOCK_METADATA_DIR, MOCK_OBJECT_IDS, requires_workbook
from valmet import api
from valmet.common.artifacts import ASSET_MAPPING_FILE
from valmet.diagnostics import diagnose as diagnose_mod
from valmet.diagnostics import evidence as evidence_mod
from valmet.diagnostics.evidence import UnknownObjectError, get_evidence
from valmet.metadata.excel_loader import WORKBOOK_ENV
from valmet.metadata.mapping import DETAILS_FILE, METADATA_DIR_ENV, use_metadata_dir
from valmet.metadata.snapshot import (
    ASSETS_FILE,
    DIAGNOSES_FILE,
    EVIDENCE_FILE,
    META_FILE,
    SNAPSHOT_FILES,
    MetadataUnavailableError,
    export_snapshot,
)

pytestmark = requires_workbook
SCENE = "mock_equipment_room"
NEARBY_IDS = ["obj_0003", "obj_9999", "obj_0001", None, "obj_014", "obj_0002"]


@pytest.fixture(autouse=True)
def _mock_scene(monkeypatch):
    monkeypatch.delenv(METADATA_DIR_ENV, raising=False)


@pytest.fixture(scope="module")
def live():
    """Live results (workbook present), computed once."""
    assert not api._use_snapshot(offline=False)
    return {
        "assets": {o: api.get_asset(o) for o in MOCK_OBJECT_IDS},
        "diagnoses": {o: api.diagnose_fault(o) for o in MOCK_OBJECT_IDS},
        "llm": {o: api.diagnose_fault(o, use_llm=True) for o in MOCK_OBJECT_IDS},
        "nearby": api.get_nearby_assets(NEARBY_IDS),
    }


@pytest.fixture
def no_live(monkeypatch):
    """Make any live computation fail loudly, to prove results come from the snapshot."""

    def boom(*_a, **_k):
        raise AssertionError("live computation used in snapshot mode")

    monkeypatch.setattr(evidence_mod, "asset_view", boom)
    monkeypatch.setattr(diagnose_mod, "diagnose_fault", boom)


@pytest.fixture
def excel_missing(monkeypatch, tmp_path):
    monkeypatch.setenv(WORKBOOK_ENV, str(tmp_path / "missing.xlsx"))


def _snapshot_results(**kw):
    return {
        "assets": {o: api.get_asset(o, **kw) for o in MOCK_OBJECT_IDS},
        "diagnoses": {o: api.diagnose_fault(o, **kw) for o in MOCK_OBJECT_IDS},
        "llm": {o: api.diagnose_fault(o, use_llm=True, **kw) for o in MOCK_OBJECT_IDS},
        "nearby": api.get_nearby_assets(NEARBY_IDS, **kw),
    }


# --------------------------------------------------------------------------- #
# Same output in both modes
# --------------------------------------------------------------------------- #
def test_excel_missing_uses_snapshot_with_identical_output(live, excel_missing, no_live):
    assert api._use_snapshot(offline=False)
    assert _snapshot_results() == live


def test_offline_true_forces_snapshot_even_with_excel(live, no_live):
    assert _snapshot_results(offline=True) == live


def test_snapshot_mode_methods_and_order(live, excel_missing, no_live):
    assert api.diagnose_fault("obj_0001")["method"] == "rule_based"
    assert api.diagnose_fault("obj_0001", use_llm=True)["method"] == "rule_based_fallback"
    out = api.get_nearby_assets(NEARBY_IDS)
    assert [a["object_id"] for a in out] == ["obj_0003", "obj_0001", "obj_0002"]
    assert out[1]["status_line"] == (
        "P-002 process pump — attention required, health 45 (Current_State!R15)"
    )


def test_snapshot_unknown_ids(excel_missing, no_live):
    with pytest.raises(UnknownObjectError, match="obj_9999"):
        api.get_asset("obj_9999")
    with pytest.raises(KeyError, match="obj_9999"):
        api.diagnose_fault("obj_9999")
    assert api.get_nearby_assets(["obj_9999", None]) == []


def test_snapshot_results_are_copies(excel_missing, no_live):
    a = api.get_asset("obj_0001")
    a["device"]["tag"] = "CHANGED"
    d = api.diagnose_fault("obj_0001", use_llm=True)
    assert api.get_asset("obj_0001")["device"]["tag"] == "P-002"
    assert api.diagnose_fault("obj_0001")["method"] == "rule_based"
    assert d["method"] == "rule_based_fallback"


def test_all_snapshot_outputs_json_serialisable(excel_missing, no_live):
    json.dumps(_snapshot_results())


# --------------------------------------------------------------------------- #
# Snapshot files
# --------------------------------------------------------------------------- #
def _read_all(dir_):
    out = {}
    for name in SNAPSHOT_FILES:
        data = json.loads((dir_ / name).read_text(encoding="utf-8"))
        if name == META_FILE:
            data.pop("created_at")
        out[name] = data
    return out


def test_snapshot_is_reproducible(tmp_path):
    a = export_snapshot(SCENE, out_dir=tmp_path / "a", source_dir=MOCK_METADATA_DIR)
    b = export_snapshot(SCENE, out_dir=tmp_path / "b", source_dir=MOCK_METADATA_DIR)
    assert _read_all(a) == _read_all(b)
    for name in (ASSETS_FILE, EVIDENCE_FILE, DIAGNOSES_FILE):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_committed_snapshot_is_current(tmp_path):
    fresh = export_snapshot(SCENE, out_dir=tmp_path / "fresh", source_dir=MOCK_METADATA_DIR)
    assert _read_all(fresh) == _read_all(MOCK_METADATA_DIR)


def test_snapshot_meta_contents(tmp_path):
    import hashlib

    from valmet.metadata.excel_loader import workbook_path

    out = export_snapshot(SCENE, out_dir=tmp_path / "m", source_dir=MOCK_METADATA_DIR)
    meta = json.loads((out / META_FILE).read_text(encoding="utf-8"))
    assert set(meta) == {"excel_file", "excel_sha256", "created_at", "schema_version"}
    assert meta["excel_file"] == workbook_path().name
    assert meta["excel_sha256"] == hashlib.sha256(workbook_path().read_bytes()).hexdigest()
    assert meta["schema_version"] == "1.0"
    assert meta["created_at"].endswith("+00:00")


def test_snapshot_contents_match_live_per_object(tmp_path):
    out = export_snapshot(SCENE, out_dir=tmp_path / "c", source_dir=MOCK_METADATA_DIR)
    assets = json.loads((out / ASSETS_FILE).read_text(encoding="utf-8"))
    evidence = json.loads((out / EVIDENCE_FILE).read_text(encoding="utf-8"))
    assert list(assets) == MOCK_OBJECT_IDS == list(evidence)
    with use_metadata_dir(MOCK_METADATA_DIR):
        for oid in MOCK_OBJECT_IDS:
            assert evidence[oid] == get_evidence(oid)


def test_export_copies_mapping_when_source_is_elsewhere(tmp_path):
    src = tmp_path / "processed"
    src.mkdir()
    for name in (ASSET_MAPPING_FILE, DETAILS_FILE):
        shutil.copyfile(MOCK_METADATA_DIR / name, src / name)
    out = export_snapshot(SCENE, out_dir=tmp_path / "fixtures", source_dir=src)
    for name in (ASSET_MAPPING_FILE, DETAILS_FILE):
        assert (out / name).read_bytes() == (src / name).read_bytes()


def test_export_rejects_wrong_scene(tmp_path):
    with pytest.raises(ValueError, match="mock_equipment_room"):
        export_snapshot("other_scene", out_dir=tmp_path / "x", source_dir=MOCK_METADATA_DIR)


def test_export_needs_the_workbook(tmp_path, excel_missing):
    with pytest.raises(MetadataUnavailableError, match="missing.xlsx"):
        export_snapshot(SCENE, out_dir=tmp_path / "x", source_dir=MOCK_METADATA_DIR)


# --------------------------------------------------------------------------- #
# Clear errors
# --------------------------------------------------------------------------- #
def test_no_excel_and_no_snapshot_gives_clear_error(tmp_path, monkeypatch, excel_missing):
    empty = tmp_path / "empty_scene"
    empty.mkdir()
    monkeypatch.setenv(METADATA_DIR_ENV, str(empty))
    for call in (
        lambda: api.get_asset("obj_0001"),
        lambda: api.get_nearby_assets(["obj_0001"]),
        lambda: api.diagnose_fault("obj_0001"),
    ):
        with pytest.raises(MetadataUnavailableError) as err:
            call()
        msg = str(err.value)
        assert "missing.xlsx" in msg  # which workbook
        assert ASSETS_FILE in msg and DIAGNOSES_FILE in msg and META_FILE in msg  # which files
        assert "python -m valmet.metadata.snapshot empty_scene" in msg  # how to create


def test_partial_snapshot_names_the_missing_file(tmp_path, monkeypatch, excel_missing):
    part = tmp_path / SCENE
    shutil.copytree(MOCK_METADATA_DIR, part)
    (part / DIAGNOSES_FILE).unlink()
    monkeypatch.setenv(METADATA_DIR_ENV, str(part))
    with pytest.raises(MetadataUnavailableError) as err:
        api.diagnose_fault("obj_0001")
    msg = str(err.value)
    assert DIAGNOSES_FILE in msg
    assert ASSETS_FILE not in msg and META_FILE not in msg


def test_offline_without_snapshot_gives_clear_error(tmp_path, monkeypatch):
    empty = tmp_path / "nothing"
    empty.mkdir()
    monkeypatch.setenv(METADATA_DIR_ENV, str(empty))
    with pytest.raises(MetadataUnavailableError, match="offline=True") as err:
        api.get_asset("obj_0001", offline=True)
    assert "python -m valmet.metadata.snapshot" in str(err.value)
