import json
from datetime import date, datetime

import pytest

from metadata_testutils import MOCK_OBJECT_IDS, requires_workbook
from valmet import api
from valmet.metadata.excel_loader import load_workbook
from valmet.metadata.mapping import METADATA_DIR_ENV

pytestmark = requires_workbook
ALL_OBJECTS = MOCK_OBJECT_IDS


@pytest.fixture(autouse=True)
def _mock_scene(monkeypatch):
    monkeypatch.delenv(METADATA_DIR_ENV, raising=False)  # always test against the mock


@pytest.fixture(scope="module")
def rows_by_source():
    out = {}
    for df in load_workbook().values():
        for row in df.to_dict("records"):
            out[row["_source"]] = row
    return out


def _norm(v):
    if hasattr(v, "to_pydatetime"):
        v = v.to_pydatetime()
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if hasattr(v, "item"):
        return v.item()
    return v


# --------------------------------------------------------------------------- #
# diagnose_fault
# --------------------------------------------------------------------------- #
def test_diagnose_obj_0001_mentions_p002_and_cavitation():
    d = api.diagnose_fault("obj_0001")
    assert "P-002" in d["summary"]
    assert "CAVITATION_RISK" in d["summary"]
    assert d["tag"] == "P-002" and d["asset_id"] == "ASSET-014" and d["is_target"]
    assert d["method"] == "rule_based"
    assert d["confidence"] == 0.68


def test_diagnose_obj_0001_findings_include_vibration():
    d = api.diagnose_fault("obj_0001")
    tele = [f for f in d["findings"] if f["type"] == "telemetry_deviation"]
    assert 1 <= len(tele) <= 3
    assert "vibration_mm_s" in [f["field"] for f in tele]
    vib = next(f for f in tele if f["field"] == "vibration_mm_s")
    assert vib["direction"] == "higher"
    # Ranked by |z| descending.
    zs = [abs(f["derived"]["z_score"]) for f in tele]
    assert zs == sorted(zs, reverse=True)
    types = [f["type"] for f in d["findings"]]
    assert "alarm_summary" in types and "latest_high_alarm" in types


@pytest.mark.parametrize("object_id", ALL_OBJECTS)
def test_every_evidence_source_exists_and_value_matches(object_id, rows_by_source):
    d = api.diagnose_fault(object_id)
    assert d["evidence"]
    for e in d["evidence"]:
        assert e["source"] in rows_by_source, e
        assert _norm(rows_by_source[e["source"]][e["field"]]) == e["value"], e


@pytest.mark.parametrize("object_id", ALL_OBJECTS)
def test_every_cited_source_in_output_is_in_evidence(object_id):
    d = api.diagnose_fault(object_id)
    ev_sources = {e["source"] for e in d["evidence"]}
    for section in ("findings", "likely_cause", "safety_precautions", "next_steps"):
        for item in d[section]:
            if item.get("_source"):
                assert item["_source"] in ev_sources, (section, item)


def test_safety_includes_loto_and_next_steps_deduped():
    d = api.diagnose_fault("obj_0001")
    assert any("lock-out/tag-out" in s["text"] for s in d["safety_precautions"])
    texts = [n["text"].lower().strip() for n in d["next_steps"]]
    assert len(texts) == len(set(texts))
    assert d["next_steps"][0]["_source"] == "Current_State!R15"
    assert d["likely_cause"]


@pytest.mark.parametrize("object_id", ["obj_0001", "obj_0002"])
def test_use_llm_falls_back_to_rules(object_id):
    rules = api.diagnose_fault(object_id)
    llm = api.diagnose_fault(object_id, use_llm=True)
    assert llm["method"] == "rule_based_fallback"
    assert rules["method"] == "rule_based"
    assert {k: v for k, v in llm.items() if k != "method"} == {
        k: v for k, v in rules.items() if k != "method"
    }
    json.dumps(llm)


def test_use_llm_unknown_object_still_raises():
    with pytest.raises(KeyError, match="obj_9999"):
        api.diagnose_fault("obj_9999", use_llm=True)


# --------------------------------------------------------------------------- #
# get_asset / get_nearby_assets
# --------------------------------------------------------------------------- #
def test_get_asset_has_sources():
    a = api.get_asset("obj_0001")
    assert a["object_id"] == "obj_0001" and a["device_id"] == "ASSET-014"
    assert a["device"]["_source"] == "Asset_Master!R15"
    assert a["condition"]["_source"] == "Current_State!R15"
    assert a["condition"]["issue_code"] == "CAVITATION_RISK"


def test_nearby_keeps_order_and_skips_unknown():
    ids = ["obj_0003", "obj_9999", "obj_0001", None, "obj_014", "obj_0002"]
    out = api.get_nearby_assets(ids)
    assert [a["object_id"] for a in out] == ["obj_0003", "obj_0001", "obj_0002"]
    assert (
        out[1]["status_line"]
        == "P-002 process pump — attention required, health 45 (Current_State!R15)"
    )
    for a in out:
        assert a["status_line"].startswith(a["device"]["tag"])
        assert a["condition"]["_source"] in a["status_line"]


def test_nearby_empty():
    assert api.get_nearby_assets([]) == []


def test_unknown_object_raises_for_single_lookups():
    with pytest.raises(KeyError, match="obj_9999"):
        api.get_asset("obj_9999")
    with pytest.raises(KeyError, match="obj_9999"):
        api.diagnose_fault("obj_9999")


@pytest.mark.parametrize("object_id", ALL_OBJECTS)
def test_all_outputs_json_serialisable(object_id):
    json.dumps(api.get_asset(object_id))
    json.dumps(api.diagnose_fault(object_id))
    json.dumps(api.get_nearby_assets([object_id, "obj_9999"]))


def test_api_exposes_exactly_three_functions():
    public = {n for n in dir(api) if not n.startswith("_") and callable(getattr(api, n))}
    assert public == {"get_asset", "get_nearby_assets", "diagnose_fault"}
    assert set(api.__all__) == public
