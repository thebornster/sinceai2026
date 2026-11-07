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


def _plane(axis: int, value: float, lo: tuple, hi: tuple, step: float, rng) -> np.ndarray:
    """Grid-sampled axis-aligned rectangle; lo/hi are the bounds of the other two axes."""
    a, b = np.meshgrid(np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step))
    other = [k for k in range(3) if k != axis]
    pts = np.zeros((a.size, 3))
    pts[:, other[0]], pts[:, other[1]], pts[:, axis] = a.ravel(), b.ravel(), value
    return pts + rng.normal(0, 0.002, pts.shape)


def segment_room(seed: int = 0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """4 x 4 x 2.4 m room (world frame, floor z=0) with a 0.5 m crate and an x-aligned pipe.

    Returns the points and the point ranges of the crate and the pipe."""
    rng = np.random.default_rng(seed)
    step = 0.03
    room = [
        _plane(2, 0.0, (-2, -2), (2, 2), step, rng),
        _plane(2, 2.4, (-2, -2), (2, 2), step, rng),
        _plane(0, -2.0, (-2, 0), (2, 2.4), step, rng),
        _plane(0, 2.0, (-2, 0), (2, 2.4), step, rng),
        _plane(1, -2.0, (-2, 0), (2, 2.4), step, rng),
        _plane(1, 2.0, (-2, 0), (2, 2.4), step, rng),
    ]
    s = 0.015
    crate = np.concatenate(
        [
            _plane(0, 0.5, (-1.0, 0.2), (-0.5, 0.7), s, rng),
            _plane(0, 1.0, (-1.0, 0.2), (-0.5, 0.7), s, rng),
            _plane(1, -1.0, (0.5, 0.2), (1.0, 0.7), s, rng),
            _plane(1, -0.5, (0.5, 0.2), (1.0, 0.7), s, rng),
            _plane(2, 0.7, (0.5, -1.0), (1.0, -0.5), s, rng),
        ]
    )
    t = np.arange(-1.2, 0.3, 0.01)
    ang = np.arange(0, 2 * np.pi, np.pi / 12)
    tt, aa = np.meshgrid(t, ang)
    pipe = np.stack(
        [tt.ravel(), 1.0 + 0.05 * np.cos(aa.ravel()), 1.2 + 0.05 * np.sin(aa.ravel())], 1
    )
    pts = np.concatenate([*room, crate, pipe])
    n_room = sum(len(p) for p in room)
    ranges = {
        "crate": np.arange(n_room, n_room + len(crate)),
        "pipe": np.arange(n_room + len(crate), len(pts)),
    }
    return pts, ranges


@pytest.fixture(scope="session")
def segmented_scene(tmp_path_factory):
    """A scene dir with manifest + scene_clean.npz for segment_room(), after segmentation."""
    from valmet.common.artifacts import MANIFEST_FILE, save_model
    from valmet.common.contracts import CoordinateSystem, SceneManifest
    from valmet.preprocessing.export import SCENE_CLEAN_FILE
    from valmet.semantic.segment import SegmentParams, run

    root = tmp_path_factory.mktemp("segment") / "room"
    root.mkdir()
    pts, ranges = segment_room()
    save_model(
        root / MANIFEST_FILE,
        SceneManifest(
            scene_id="room",
            source_filename="room.ply",
            coordinate_system=CoordinateSystem(
                up_axis="+Z", units="meters", meters_per_unit=1.0, verified=True
            ),
        ),
    )
    n = len(pts)
    np.savez(
        root / SCENE_CLEAN_FILE,
        positions=pts.astype(np.float32),
        quats_wxyz=np.tile([1.0, 0, 0, 0], (n, 1)).astype(np.float32),
        scales=np.full((n, 3), 0.01, np.float32),
        opacities=np.full(n, 0.9, np.float32),
        rgbs=np.full((n, 3), 0.5, np.float32),
        orig_index=(np.arange(n) + 1000).astype(np.int32),
    )
    # In a room this small, 5 % of the leftover points is a whole crate face, so the extra
    # RANSAC planes (meant for large unlisted walls) are off here.
    objects = run(
        scene_dir=root,
        report_dir=root / "reports",
        sp=SegmentParams(ransac_max_planes=0),
    )
    return root, objects, ranges
