from __future__ import annotations

import numpy as np
import pytest

from valmet.common.geometry import apply_transform
from valmet.preprocessing.world_frame import FrameParams, ScaleSpec, estimate_world_frame

ROOM = (8.0, 6.0, 2.5)  # x, y, z in world meters


def _room_points(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Floor, ceiling, four walls and a crate as dense noisy points in a Z-up world, plus part labels
    (0 = floor, 1 = ceiling)."""
    rng = np.random.default_rng(seed)
    lx, ly, lz = ROOM

    def sheet(n, fixed_axis, value, ranges):
        p = np.empty((n, 3))
        free = [a for a in range(3) if a != fixed_axis]
        for a, (lo, hi) in zip(free, ranges, strict=True):
            p[:, a] = rng.uniform(lo, hi, n)
        p[:, fixed_axis] = value
        return p

    parts = [
        sheet(30_000, 2, 0.0, [(0, lx), (0, ly)]),
        sheet(20_000, 2, lz, [(0, lx), (0, ly)]),
        sheet(12_000, 0, 0.0, [(0, ly), (0, lz)]),
        sheet(12_000, 0, lx, [(0, ly), (0, lz)]),
        sheet(8_000, 1, 0.0, [(0, lx), (0, lz)]),
        sheet(8_000, 1, ly, [(0, lx), (0, lz)]),
        rng.uniform([2, 1, 0], [3, 2, 1], (3_000, 3)),
    ]
    labels = np.concatenate([np.full(len(p), i) for i, p in enumerate(parts)])
    pts = np.concatenate(parts)
    return pts + rng.normal(scale=0.003, size=pts.shape), labels


def _world_to_source(seed: int) -> np.ndarray:
    """Y-down camera-style source frame, with an arbitrary yaw, small tilt and offset."""
    rng = np.random.default_rng(seed)
    yaw = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(yaw), np.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    z_to_minus_y = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    tilt = np.deg2rad(4)
    rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
    m = np.eye(4)
    m[:3, :3] = 0.5 * (rx @ z_to_minus_y @ rz)  # source units = 2 m
    m[:3, 3] = rng.uniform(-3, 3, 3)
    return m


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recovers_floor_walls_and_scale(seed: int):
    world_true, labels = _room_points(seed=seed)
    source = apply_transform(_world_to_source(seed=seed), world_true)

    transform, info = estimate_world_frame(
        positions=source,
        opacities=np.ones(len(source)),
        params=FrameParams(voxel_size=0.01, slab_half_width=0.05, plane_threshold=0.005),
        scale_spec=ScaleSpec(ceiling_height_m=ROOM[2]),
    )
    world = apply_transform(transform, source)

    assert info["scale_source"] == "ceiling_height"
    assert info["scale"] == pytest.approx(2.0, rel=0.01)
    floor = world[labels == 0]
    assert np.abs(np.median(floor[:, 2])) < 0.01
    assert floor[:, 2].std() < 0.01  # level, not tilted
    ceiling = world[labels == 1]
    assert np.median(ceiling[:, 2]) == pytest.approx(ROOM[2], abs=0.02)

    # Walls on the X/Y axes, longest side along X, room centered in XY.
    lo, hi = np.percentile(world[:, :2], [0.5, 99.5], axis=0)
    np.testing.assert_allclose(hi - lo, ROOM[:2], atol=0.05)
    np.testing.assert_allclose((hi + lo) / 2, 0.0, atol=0.05)


def test_unset_scale_keeps_source_units():
    source = apply_transform(_world_to_source(seed=5), _room_points(seed=5)[0])
    transform, info = estimate_world_frame(
        positions=source,
        opacities=np.ones(len(source)),
        params=FrameParams(voxel_size=0.01, slab_half_width=0.05, plane_threshold=0.005),
        scale_spec=ScaleSpec(),
    )
    assert info["scale_source"] == "unset"
    assert np.cbrt(np.linalg.det(transform[:3, :3])) == pytest.approx(1.0)
    assert info["ceiling_plane_source"]["distance_from_floor"] == pytest.approx(1.25, abs=0.01)
