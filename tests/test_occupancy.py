from __future__ import annotations

import numpy as np

from valmet.spatial.occupancy import NavParams, build_occupancy

PARAMS = NavParams(resolution=0.05, radius=0.25, height=1.8, max_step_height=0.15)


def column(x: float, y: float, z_lo: float, z_hi: float, half: float, n: int, rng) -> np.ndarray:
    return rng.uniform([x - half, y - half, z_lo], [x + half, y + half, z_hi], (n, 3))


def synthetic_room(seed: int = 0) -> np.ndarray:
    """6 x 4 m room: walls, a crate at (1, 0), a 0.3 m pipe along x=-1 and a 0.05 m bump at (-2, 1)."""
    rng = np.random.default_rng(seed)
    parts = [rng.uniform([-3, -2, 0], [3, 2, 0.01], (40_000, 3))]  # sparse floor
    for x0, y0, x1, y1 in ((-3, -2, 3, -2), (-3, 2, 3, 2), (-3, -2, -3, 2), (3, -2, 3, 2)):
        t = rng.uniform(0, 1, 30_000)
        parts.append(
            np.stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0), rng.uniform(0, 2.5, len(t))], 1)
        )
    parts.append(column(1.0, 0.0, 0.0, 1.2, 0.3, 30_000, rng))
    parts.append(rng.uniform([-1.05, -2, 0.0], [-0.95, -0.5, 0.3], (8_000, 3)))  # pipe
    parts.append(column(-2.0, 1.0, 0.0, 0.05, 0.2, 5_000, rng))  # bump below step height
    return np.concatenate(parts)


def _cell_value(grid, navmap, x, y):
    i, j = navmap.world_to_cell(np.array([x, y]))[0]
    return grid[i, j]


def test_blocks_crate_and_pipe_but_not_low_bump():
    pts = synthetic_room()
    grid, navmap, info = build_occupancy(
        positions=pts,
        opacities=np.ones(len(pts)),
        scene_id="synthetic",
        params=PARAMS,
    )
    v = navmap.cell_values
    assert grid.dtype == np.uint8 and grid.shape == tuple(navmap.shape)
    assert _cell_value(grid, navmap, 1.0, 0.0) == v.occupied  # crate
    assert _cell_value(grid, navmap, -1.0, -1.0) == v.occupied  # pipe higher than a step
    assert _cell_value(grid, navmap, -2.0, 1.0) == v.free  # bump lower than a step
    assert _cell_value(grid, navmap, 0.0, -1.0) == v.free  # open floor
    assert _cell_value(grid, navmap, -3.2, 0.0) == v.unknown  # outside the room
    assert info["walkable_areas_m2"][0] > 10


def test_inflation_matches_radius():
    pts = synthetic_room()
    grid, navmap, _ = build_occupancy(
        positions=pts,
        opacities=np.ones(len(pts)),
        scene_id="synthetic",
        params=PARAMS,
    )
    v = navmap.cell_values
    # Crate spans x in [0.7, 1.3]; a 0.25 m radius blocks up to ~1.55 and frees beyond.
    assert _cell_value(grid, navmap, 1.48, 0.0) == v.occupied
    assert _cell_value(grid, navmap, 1.65, 0.0) == v.free


def test_faint_gaussians_are_ignored():
    rng = np.random.default_rng(1)
    pts = synthetic_room()
    haze = column(-1.5, 1.0, 0.3, 1.5, 0.3, 20_000, rng)
    grid, navmap, _ = build_occupancy(
        positions=np.concatenate([pts, haze]),
        opacities=np.concatenate([np.ones(len(pts)), np.full(len(haze), 0.2)]),
        scene_id="synthetic",
        params=PARAMS,
    )
    assert _cell_value(grid, navmap, -1.5, 1.0) == navmap.cell_values.free


def _room_with(*extras: np.ndarray, params: NavParams = PARAMS):
    pts = np.concatenate([synthetic_room(), *extras])
    return build_occupancy(
        positions=pts,
        opacities=np.ones(len(pts)),
        scene_id="synthetic",
        params=params,
    )


def test_overhang_blocks_the_floor_below_and_is_inflated():
    rng = np.random.default_rng(2)
    shelf = column(-2.0, -1.0, 1.0, 1.2, 0.2, 5_000, rng)  # nothing below it
    grid, navmap, _ = _room_with(shelf)
    v = navmap.cell_values
    assert _cell_value(grid, navmap, -2.0, -1.0) == v.occupied
    # Horizontal buffer applies at any body height: 0.15 m beside the shelf edge is blocked.
    assert _cell_value(grid, navmap, -2.0, -0.65) == v.occupied
    assert _cell_value(grid, navmap, -2.0, -0.4) == v.free


def test_overhead_pipe_above_body_height_does_not_block():
    rng = np.random.default_rng(3)
    rack = rng.uniform([2.15, -2.0, 2.3], [2.25, 2.0, 2.4], (5_000, 3))  # spans the room at 2.3 m
    grid, navmap, _ = _room_with(rack)
    assert _cell_value(grid, navmap, 2.2, 1.0) == navmap.cell_values.free


def test_head_clearance_extends_obstacle_band_above_body_height():
    rng = np.random.default_rng(4)
    bar = column(0.0, 1.2, 1.85, 1.9, 0.2, 5_000, rng)  # just above a 1.8 m avatar
    v = PARAMS
    with_headroom, map_a, _ = _room_with(bar)
    without, map_b, _ = _room_with(
        bar,
        params=NavParams(
            resolution=v.resolution,
            radius=v.radius,
            height=v.height,
            head_clearance=0.0,
            max_step_height=v.max_step_height,
        ),
    )
    assert _cell_value(with_headroom, map_a, 0.0, 1.2) == map_a.cell_values.occupied
    assert _cell_value(without, map_b, 0.0, 1.2) == map_b.cell_values.free
    assert map_a.clearance.height == 1.8  # contract keeps the real body height
