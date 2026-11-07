from __future__ import annotations

import numpy as np
import pytest

from test_occupancy import PARAMS, synthetic_room
from valmet.common.contracts import OrientedBox
from valmet.spatial.occupancy import build_layers, build_occupancy
from valmet.spatial.planner import (
    UnreachableTargetError,
    _line_free,
    path_clearance,
    plan_path,
    sideways_segments,
)
from valmet.spatial.proximity import footprint_distance

CRATE = OrientedBox(center=(1.0, 0.0, 0.6), extent=(0.6, 0.6, 1.2))


@pytest.fixture(scope="module")
def room():
    pts = synthetic_room()
    grid, navmap, _ = build_occupancy(
        positions=pts,
        opacities=np.ones(len(pts)),
        scene_id="synthetic",
        params=PARAMS,
    )
    return grid, navmap


def _segments_free(path, navmap, grid) -> bool:
    cells = navmap.world_to_cell(np.asarray(path.waypoints)[:, :2])
    free = grid == navmap.cell_values.free
    return all(
        _line_free(a=tuple(a), b=tuple(b), free=free)
        for a, b in zip(cells[:-1], cells[1:], strict=True)
    )


def test_route_from_given_start_avoids_crate(room):
    grid, navmap = room
    path = plan_path(
        navmap=navmap,
        grid=grid,
        target_box=CRATE,
        target_object_id="crate",
        start_xy=(2.5, 1.5),
    )
    assert np.allclose(path.start[:2], (2.5, 1.5), atol=0.1)
    standoff = footprint_distance(points_xy=np.array(path.inspection_position[:2]), box=CRATE)[0]
    assert navmap.clearance.radius + 0.1 - 1e-9 <= standoff <= 1.0
    assert _segments_free(path, navmap, grid)
    assert path_clearance(navmap=navmap, grid=grid, path=path) >= navmap.clearance.radius
    assert all(w[2] == 0.0 for w in path.waypoints)


def test_route_crosses_room_around_crate(room):
    grid, navmap = room
    path = plan_path(
        navmap=navmap,
        grid=grid,
        target_box=OrientedBox(center=(2.6, 0.0, 0.5), extent=(0.2, 0.2, 1.0)),
        target_object_id="far",
        start_xy=(-0.2, 0.0),
    )
    # The straight line passes through the crate, so the route must bend around it.
    assert len(path.waypoints) >= 3
    assert _segments_free(path, navmap, grid)


def test_auto_start_is_far_from_target(room):
    grid, navmap = room
    path = plan_path(
        navmap=navmap,
        grid=grid,
        target_box=CRATE,
        target_object_id="crate",
    )
    assert path.length() > 2.0
    assert _segments_free(path, navmap, grid)


def test_enclosed_target_is_unreachable(room):
    grid, navmap = room
    # Hollow square wall 1.4 m around the target: every stand-off cell is inside it.
    walled = grid.copy()
    i, j = navmap.world_to_cell(np.array([-2.0, -1.0]))[0]
    ring = np.zeros_like(grid, dtype=bool)
    ring[max(i - 30, 0) : i + 31, max(j - 30, 0) : j + 31] = True
    ring[max(i - 28, 0) : i + 29, max(j - 28, 0) : j + 29] = False
    walled[ring] = navmap.cell_values.occupied
    with pytest.raises(UnreachableTargetError):
        plan_path(
            navmap=navmap,
            grid=walled,
            target_box=OrientedBox(center=(-2.0, -1.0, 0.5), extent=(0.2, 0.2, 1.0)),
            target_object_id="boxed_in",
            start_xy=(2.5, 1.5),
        )


def test_route_passes_under_overhead_rack():
    rng = np.random.default_rng(5)
    # Rack at 2.3 m crossing the room between start and target: must not force a detour.
    rack = rng.uniform([-0.55, -2.0, 2.3], [-0.45, 2.0, 2.4], (5_000, 3))
    pts = np.concatenate([synthetic_room(), rack])
    grid, navmap, _ = build_occupancy(
        positions=pts,
        opacities=np.ones(len(pts)),
        scene_id="synthetic",
        params=PARAMS,
    )
    path = plan_path(
        navmap=navmap,
        grid=grid,
        target_box=OrientedBox(center=(0.2, 1.2, 0.5), extent=(0.2, 0.2, 1.0)),
        target_object_id="beyond_rack",
        start_xy=(-2.0, 1.2),
    )
    assert len(path.waypoints) == 2  # straight line under the rack
    assert _segments_free(path, navmap, grid)


def _room_with_gap(gap: float):
    """Wall across x = -2 with one gap of `gap` meters at y = 0, leaving a strip west of it."""
    rng = np.random.default_rng(6)
    half = gap / 2
    wall = np.concatenate(
        [
            rng.uniform([-2.05, -2.0, 0.0], [-1.95, -half, 2.0], (20_000, 3)),
            rng.uniform([-2.05, half, 0.0], [-1.95, 2.0, 2.0], (20_000, 3)),
        ]
    )
    pts = np.concatenate([synthetic_room(), wall])
    grid, obstacles, navmap, _ = build_layers(
        positions=pts,
        opacities=np.ones(len(pts)),
        scene_id="synthetic",
        params=PARAMS,
    )
    return grid, obstacles, navmap


FAR_TARGET = OrientedBox(center=(2.6, 0.0, 0.5), extent=(0.2, 0.2, 1.0))


def test_gap_narrower_than_shoulders_is_passed_sideways():
    grid, obstacles, navmap = _room_with_gap(gap=0.4)
    with pytest.raises(UnreachableTargetError):
        plan_path(
            navmap=navmap,
            grid=grid,
            target_box=FAR_TARGET,
            target_object_id="far",
            start_xy=(-2.5, 1.0),
        )
    path = plan_path(
        navmap=navmap,
        grid=grid,
        target_box=FAR_TARGET,
        target_object_id="far",
        start_xy=(-2.5, 1.0),
        obstacles=obstacles,
    )
    sideways = sideways_segments(
        navmap=navmap,
        grid=grid,
        path=path,
    )
    assert any(sideways) and not all(sideways)
    wp = np.asarray(path.waypoints)
    for a, b in np.stack([wp[:-1], wp[1:]], axis=1)[sideways]:
        # Sideways stretches run through the gap along the wall normal (x), not across it.
        assert abs(b[1] - a[1]) <= abs(b[0] - a[0]) + 1e-9
        assert abs(a[1]) < 0.2 and abs(b[1]) < 0.2
    clearance = path_clearance(
        navmap=navmap,
        grid=grid,
        path=path,
        obstacles=obstacles,
    )
    assert 0.15 <= clearance < 0.25


def test_gap_narrower_than_body_depth_stays_blocked():
    grid, obstacles, navmap = _room_with_gap(gap=0.15)
    with pytest.raises(UnreachableTargetError):
        plan_path(
            navmap=navmap,
            grid=grid,
            target_box=FAR_TARGET,
            target_object_id="far",
            start_xy=(-2.5, 1.0),
            obstacles=obstacles,
        )


def test_obstacles_do_not_change_routes_that_fit_normally():
    pts = synthetic_room()
    grid, obstacles, navmap, _ = build_layers(
        positions=pts,
        opacities=np.ones(len(pts)),
        scene_id="synthetic",
        params=PARAMS,
    )
    kwargs = {
        "navmap": navmap,
        "grid": grid,
        "target_box": CRATE,
        "target_object_id": "crate",
        "start_xy": (2.5, 1.5),
    }
    normal = plan_path(**kwargs)
    with_squeeze = plan_path(**kwargs, obstacles=obstacles)
    assert not any(sideways_segments(navmap=navmap, grid=grid, path=with_squeeze))
    assert with_squeeze.length() == pytest.approx(normal.length(), abs=1e-6)
