"""Asset registry: one AssetRecord per device, joined from the workbook sheets.

Every row model keeps all of its sheet's columns plus ``_source`` (e.g.
``"Current_State!R15"``), so any value can be traced back to the Excel row.
"""

from __future__ import annotations

import math
from datetime import datetime
from functools import lru_cache
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from valmet.metadata.excel_loader import SOURCE_COL, load_workbook


# --------------------------------------------------------------------------- #
# Row models
# --------------------------------------------------------------------------- #
class SourcedRow(BaseModel):
    """One Excel row. Key columns are typed; all other columns are kept as extras."""

    model_config = ConfigDict(extra="allow", serialize_by_alias=True)

    # Named source_ref (not "source") because Alarms_Events has its own "source" column.
    source_ref: str = Field(alias=SOURCE_COL)


class IdentityRow(SourcedRow):  # Asset_Master
    asset_id: str
    tag: str
    asset_class: str


class ConditionRow(SourcedRow):  # Current_State
    asset_id: str
    condition_state: str
    issue_code: str
    health_score: int
    knowledge_doc_id: str


class TelemetryRow(SourcedRow):  # Telemetry_Sample_3d
    asset_id: str
    timestamp: datetime


class AlarmRow(SourcedRow):  # Alarms_Events
    event_id: str
    asset_id: str
    timestamp: datetime


class MaintenanceRow(SourcedRow):  # Maintenance_History
    work_order_id: str
    asset_id: str
    opened_at: datetime


class KnowledgeDocRow(SourcedRow):  # Knowledge_Docs
    doc_id: str
    title: str
    content: str


class ServiceCaseRow(SourcedRow):  # Historical_Service_Cases
    case_id: str
    asset_class: str
    historical_issue_code: str


class AssetRecord(BaseModel):
    asset_id: str
    identity: IdentityRow
    condition: ConditionRow
    telemetry: list[TelemetryRow]  # oldest first
    alarms: list[AlarmRow]  # newest first
    maintenance: list[MaintenanceRow]  # newest first
    knowledge_doc: KnowledgeDocRow | None
    similar_cases: list[ServiceCaseRow]

    @property
    def asset_class(self) -> str:
        return self.identity.asset_class

    @property
    def is_healthy(self) -> bool:
        return self.condition.condition_state == "healthy"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _native(value: Any) -> Any:
    """Convert pandas/numpy scalars to plain Python values (NaN/NaT -> None)."""
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: _native(v) for k, v in row.items()} for row in df.to_dict("records")]


def _group(
    df: pd.DataFrame, model: type[SourcedRow], sort_col: str, newest_first: bool
) -> dict[str, list]:
    # Stable sort: ties keep their Excel order.
    df = df.sort_values(sort_col, ascending=not newest_first, kind="mergesort")
    out: dict[str, list] = {}
    for row in _records(df):
        out.setdefault(row["asset_id"], []).append(model.model_validate(row))
    return out


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
class AssetRegistry:
    def __init__(self, workbook: dict[str, pd.DataFrame] | None = None) -> None:
        wb = workbook if workbook is not None else load_workbook()

        identities = {
            r["asset_id"]: IdentityRow.model_validate(r) for r in _records(wb["Asset_Master"])
        }
        conditions = {
            r["asset_id"]: ConditionRow.model_validate(r) for r in _records(wb["Current_State"])
        }
        docs = {
            r["doc_id"]: KnowledgeDocRow.model_validate(r) for r in _records(wb["Knowledge_Docs"])
        }
        cases = [ServiceCaseRow.model_validate(r) for r in _records(wb["Historical_Service_Cases"])]

        telemetry = _group(wb["Telemetry_Sample_3d"], TelemetryRow, "timestamp", newest_first=False)
        alarms = _group(wb["Alarms_Events"], AlarmRow, "timestamp", newest_first=True)
        maintenance = _group(
            wb["Maintenance_History"], MaintenanceRow, "opened_at", newest_first=True
        )

        self._records: dict[str, AssetRecord] = {}
        for asset_id in sorted(identities):
            identity = identities[asset_id]
            condition = conditions.get(asset_id)
            if condition is None:
                raise ValueError(f"{asset_id} ({identity.source_ref}) has no Current_State row")
            self._records[asset_id] = AssetRecord(
                asset_id=asset_id,
                identity=identity,
                condition=condition,
                telemetry=telemetry.get(asset_id, []),
                alarms=alarms.get(asset_id, []),
                maintenance=maintenance.get(asset_id, []),
                knowledge_doc=docs.get(condition.knowledge_doc_id),
                similar_cases=[
                    c
                    for c in cases
                    if c.asset_class == identity.asset_class
                    and c.historical_issue_code == condition.issue_code
                ],
            )

    def get(self, asset_id: str) -> AssetRecord:
        try:
            return self._records[asset_id]
        except KeyError:
            raise KeyError(f"Unknown asset_id: {asset_id!r}") from None

    def all(self) -> list[AssetRecord]:
        return list(self._records.values())

    def by_class(self, asset_class: str) -> list[AssetRecord]:
        return [r for r in self._records.values() if r.asset_class == asset_class]

    def unhealthy(self) -> list[AssetRecord]:
        return [r for r in self._records.values() if not r.is_healthy]


@lru_cache(maxsize=1)
def get_registry() -> AssetRegistry:
    """Shared registry built once from the default workbook."""
    return AssetRegistry()
