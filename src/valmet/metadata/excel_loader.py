"""Load the equipment workbook (builder pack) sheets into pandas DataFrames.

Every row gets a ``_source`` column like ``"Current_State!R5"`` holding the real
Excel row number (header is row 1, so the first data row is R2).

Workbook path: ``$VALMET_EQUIPMENT_XLSX`` (relative paths are resolved against the
repo root), else ``data/raw/equipment.xlsx`` (committed).
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import openpyxl
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORKBOOK = REPO_ROOT / "data" / "raw" / "equipment.xlsx"
WORKBOOK_ENV = "VALMET_EQUIPMENT_XLSX"

SHEETS = (
    "Asset_Master",
    "Current_State",
    "Telemetry_Sample_3d",
    "Alarms_Events",
    "Maintenance_History",
    "Historical_Service_Cases",
    "Knowledge_Docs",
)

SOURCE_COL = "_source"


def _read_sheet(ws, sheet_name: str) -> pd.DataFrame:
    """Read one worksheet, keeping the true Excel row number of every record."""
    rows = ws.iter_rows(values_only=True)
    header = next(rows)
    # Ignore trailing columns with no header.
    keep = [i for i, h in enumerate(header) if h is not None]
    columns = [str(header[i]).strip() for i in keep]

    records, excel_rows = [], []
    for excel_row, row in enumerate(rows, start=2):  # header is row 1
        values = [row[i] if i < len(row) else None for i in keep]
        if all(v is None for v in values):
            continue  # fully blank row
        records.append(values)
        excel_rows.append(excel_row)

    df = pd.DataFrame(records, columns=columns)
    df[SOURCE_COL] = [f"{sheet_name}!R{r}" for r in excel_rows]

    # Normally the Excel row is index + 2; warn loudly if blank rows broke that.
    expected = [i + 2 for i in range(len(df))]
    if excel_rows != expected:
        print(
            f"[excel_loader] note: {sheet_name} has blank rows; "
            f"_source uses real Excel rows, not index + 2"
        )
    return df


def workbook_path() -> Path:
    """Workbook location: $VALMET_EQUIPMENT_XLSX, else data/raw/equipment.xlsx."""
    env = os.environ.get(WORKBOOK_ENV, "").strip()
    if not env:
        return DEFAULT_WORKBOOK
    path = Path(env).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_workbook(path: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all required sheets once per path; later calls return the cached result.

    Treat the returned DataFrames as read-only (use ``.copy()`` before modifying).
    """
    return _load_workbook(Path(path) if path is not None else workbook_path())


@cache
def _load_workbook(path: Path) -> dict[str, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(
            f"Equipment workbook not found at {path}. Put it in data/raw/ or set {WORKBOOK_ENV}."
        )
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        missing = [s for s in SHEETS if s not in wb.sheetnames]
        if missing:
            raise KeyError(f"Workbook {path} is missing sheets: {missing}")
        return {name: _read_sheet(wb[name], name) for name in SHEETS}
    finally:
        wb.close()


def get_sheet(name: str) -> pd.DataFrame:
    """Return one cached sheet by name."""
    return load_workbook()[name]


if __name__ == "__main__":
    for name, df in load_workbook().items():
        first = df[SOURCE_COL].iloc[0] if len(df) else "-"
        print(f"{name:<26} rows={len(df):>5}  first={first}")
