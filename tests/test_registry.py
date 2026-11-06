import pytest

from metadata_testutils import requires_workbook
from valmet.metadata.excel_loader import load_workbook
from valmet.metadata.registry import AssetRegistry

pytestmark = requires_workbook


@pytest.fixture(scope="module")
def wb():
    return load_workbook()


@pytest.fixture(scope="module")
def registry(wb):
    return AssetRegistry(wb)


def test_thirty_records(registry):
    assert len(registry.all()) == 30


def test_every_record_has_identity_and_condition(registry):
    for rec in registry.all():
        assert rec.identity.asset_id == rec.asset_id
        assert rec.condition.asset_id == rec.asset_id
        assert rec.identity.source_ref.startswith("Asset_Master!R")
        assert rec.condition.source_ref.startswith("Current_State!R")


@pytest.mark.parametrize(
    "attr, sheet",
    [
        ("alarms", "Alarms_Events"),
        ("maintenance", "Maintenance_History"),
        ("telemetry", "Telemetry_Sample_3d"),
    ],
)
def test_no_rows_lost_in_join(registry, wb, attr, sheet):
    rows = [row for rec in registry.all() for row in getattr(rec, attr)]
    assert len(rows) == len(wb[sheet])
    # Each Excel row appears exactly once, under the right asset.
    assert len({r.source_ref for r in rows}) == len(wb[sheet])
    for rec in registry.all():
        assert all(r.asset_id == rec.asset_id for r in getattr(rec, attr))


def test_unhealthy_count(registry):
    unhealthy = registry.unhealthy()
    assert len(unhealthy) == 13
    assert all(r.condition.condition_state != "healthy" for r in unhealthy)


def test_asset_014_is_cavitating_pump(registry):
    rec = registry.get("ASSET-014")
    assert rec.asset_class == "PROCESS_PUMP"
    assert rec.condition.issue_code == "CAVITATION_RISK"
    assert rec.condition.source_ref == "Current_State!R15"


def test_ordering(registry):
    for rec in registry.all():
        ts = [t.timestamp for t in rec.telemetry]
        assert ts == sorted(ts)
        al = [a.timestamp for a in rec.alarms]
        assert al == sorted(al, reverse=True)
        mh = [m.opened_at for m in rec.maintenance]
        assert mh == sorted(mh, reverse=True)


def test_knowledge_doc_and_similar_cases_join(registry):
    for rec in registry.all():
        assert rec.knowledge_doc is not None
        assert rec.knowledge_doc.doc_id == rec.condition.knowledge_doc_id
        for case in rec.similar_cases:
            assert case.asset_class == rec.asset_class
            assert case.historical_issue_code == rec.condition.issue_code
    assert len(registry.get("ASSET-014").similar_cases) == 3


def test_by_class_and_unknown_id(registry):
    assert len(registry.by_class("PROCESS_PUMP")) == 6
    with pytest.raises(KeyError):
        registry.get("ASSET-999")


def test_source_ref_does_not_shadow_sheet_source_column(registry, wb):
    # Alarms_Events has its own "source" column; it must survive next to _source.
    alarm = registry.get("ASSET-014").alarms[0]
    dumped = alarm.model_dump()
    assert dumped["_source"] == alarm.source_ref
    assert dumped["_source"].startswith("Alarms_Events!R")
    row = wb["Alarms_Events"].set_index("_source").loc[alarm.source_ref]
    assert dumped["source"] == row["source"]
