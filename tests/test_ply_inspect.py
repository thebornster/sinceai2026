from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import GS_PROPS, synthetic_gaussians, write_ply
from valmet.preprocessing.ply_inspect import PlyInspectionError, inspect_ply, read_header

REPO = Path(__file__).resolve().parents[1]
REAL_PLY = REPO / "data" / "raw" / "Equipment Room.ply"


def test_header_and_schema(gs_ply: Path):
    r = inspect_ply(gs_ply, sample_size=10_000)
    assert r["header"]["format"] == "binary_little_endian"
    assert r["header"]["elements"][0]["count"] == 1000
    assert r["file"]["expected_size_bytes"] == r["file"]["size_bytes"]
    gs = r["gaussian_schema"]
    assert gs["is_conventional_3dgs"] and gs["sh_degree"] == 1 and gs["has_normals"]
    assert r["sample"]["size"] == 1000  # capped at vertex count
    assert r["positions"]["nonfinite_rows"] == 3
    assert r["positions"]["axes"]["x"]["nonfinite"] == 3
    assert 0 <= r["opacity"]["activated"]["min"] <= r["opacity"]["activated"]["max"] <= 1
    assert "activated_max_axis" in r["scale"]


def test_sample_is_reproducible_and_matches_data(gs_ply: Path):
    a = inspect_ply(gs_ply, sample_size=100, seed=7)
    b = inspect_ply(gs_ply, sample_size=100, seed=7)
    c = inspect_ply(gs_ply, sample_size=100, seed=8)
    assert a["positions"] == b["positions"]
    assert a["positions"] != c["positions"]
    full = inspect_ply(gs_ply, sample_size=10_000)
    y = synthetic_gaussians()[:, GS_PROPS.index("y")]
    assert np.isclose(full["positions"]["axes"]["y"]["mean"], y.mean(), atol=1e-5)


@pytest.mark.parametrize("fmt", ["ascii", "binary_big_endian"])
def test_other_encodings(tmp_path: Path, fmt: str):
    rows = synthetic_gaussians(50)
    p = write_ply(tmp_path / "f.ply", GS_PROPS, rows, fmt=fmt)
    r = inspect_ply(p, sample_size=20, seed=1)
    assert r["header"]["format"] == fmt and r["sample"]["size"] == 20
    full = inspect_ply(p, sample_size=50)
    z = rows[:, GS_PROPS.index("z")]
    assert np.isclose(full["positions"]["axes"]["z"]["mean"], z.mean(), atol=1e-4)


def test_non_gaussian_ply_is_flagged(tmp_path: Path):
    p = write_ply(tmp_path / "pc.ply", ["x", "y", "z"], np.zeros((5, 3)))
    r = inspect_ply(p)
    assert not r["gaussian_schema"]["is_conventional_3dgs"]
    assert "opacity" in r["gaussian_schema"]["missing_required"]
    assert "opacity" not in r and r["warnings"]


def test_file_is_not_modified(gs_ply: Path):
    before = gs_ply.read_bytes()
    inspect_ply(gs_ply)
    assert gs_ply.read_bytes() == before


@pytest.mark.parametrize(
    "content, message",
    [
        (b"hello world", "magic"),
        (b"ply\nformat binary_little_endian 1.0\nelement vertex 1\n", "end_header"),
        (b"ply\nformat weird 1.0\nelement vertex 1\nproperty float x\nend_header\n", "format"),
        (b"ply\nformat ascii 1.0\nelement vertex x\nend_header\n", "malformed"),
        (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty quad x\nend_header\n", "malformed"),
        (b"ply\nformat ascii 1.0\nproperty float x\nend_header\n", "before any element"),
        (b"ply\nformat ascii 1.0\nelement face 0\nend_header\n", "vertex"),
        (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float a\nend_header\n1\n", "x/y/z"),
    ],
)
def test_malformed_headers(tmp_path: Path, content: bytes, message: str):
    p = tmp_path / "bad.ply"
    p.write_bytes(content)
    with pytest.raises(PlyInspectionError, match=message):
        inspect_ply(p)


def test_truncated_binary(gs_ply: Path, tmp_path: Path):
    p = tmp_path / "trunc.ply"
    p.write_bytes(gs_ply.read_bytes()[:-100])
    with pytest.raises(PlyInspectionError, match="truncated"):
        inspect_ply(p)


def test_missing_file(tmp_path: Path):
    with pytest.raises(PlyInspectionError, match="not found"):
        read_header(tmp_path / "nope.ply")


def test_cli_writes_report_and_fails_cleanly(gs_ply: Path, tmp_path: Path):
    out = tmp_path / "report.json"
    cli = [sys.executable, str(REPO / "scripts" / "inspect_ply.py")]
    ok = subprocess.run(
        [*cli, str(gs_ply), "-o", str(out), "--hash"], capture_output=True, text=True
    )
    assert ok.returncode == 0, ok.stderr
    report = json.loads(out.read_text())
    assert len(report["file"]["sha256"]) == 64

    bad = subprocess.run(
        [*cli, str(tmp_path / "missing.ply"), "-o", str(out)], capture_output=True, text=True
    )
    assert bad.returncode == 2 and "not found" in bad.stderr


@pytest.mark.real_data
@pytest.mark.skipif(not REAL_PLY.exists(), reason="real Equipment Room.ply not present")
def test_real_equipment_room():
    r = inspect_ply(REAL_PLY, sample_size=50_000)
    assert r["header"]["elements"][0]["count"] > 0
    assert r["gaussian_schema"]["is_conventional_3dgs"]
