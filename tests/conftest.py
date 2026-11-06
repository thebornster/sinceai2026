from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

GS_PROPS = (
    ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    + [f"f_rest_{i}" for i in range(9)]  # SH degree 1
    + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
)


def write_ply(
    path: Path,
    props: list[str],
    rows: np.ndarray,
    fmt: str = "binary_little_endian",
    extra_header: str = "",
) -> Path:
    """Write a float32-only vertex PLY. rows: (N, len(props))."""
    header = (
        f"ply\nformat {fmt} 1.0\ncomment synthetic test fixture\n"
        f"element vertex {len(rows)}\n"
        + "".join(f"property float {p}\n" for p in props)
        + extra_header
        + "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        if fmt == "ascii":
            for r in rows:
                f.write((" ".join(f"{v:.6g}" for v in r) + "\n").encode("ascii"))
        else:
            order = "<" if fmt == "binary_little_endian" else ">"
            f.write(np.asarray(rows, dtype=order + "f4").tobytes())
    return path


def synthetic_gaussians(n: int = 1000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(n, len(GS_PROPS))).astype(np.float32)
    col = {p: i for i, p in enumerate(GS_PROPS)}
    rows[:, col["x"]] = rng.uniform(-5, 5, n)
    rows[:, col["y"]] = rng.uniform(0, 3, n)
    rows[:, col["z"]] = rng.uniform(-2, 2, n)
    rows[:, col["opacity"]] = rng.normal(0, 3, n)
    rows[:, col["scale_0"] : col["scale_2"] + 1] = rng.normal(-4, 1, (n, 3))
    rows[:3, col["x"]] = np.nan  # a few broken positions
    return rows


@pytest.fixture
def gs_ply(tmp_path: Path) -> Path:
    return write_ply(tmp_path / "synthetic_gs.ply", GS_PROPS, synthetic_gaussians())
