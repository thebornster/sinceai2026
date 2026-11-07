"""2-D occupancy grid on the world XY plane from cleaned, world-frame Gaussians.

A cell is blocked when enough opaque Gaussians sit in the avatar's body band
[max_step_height, height + head_clearance]. Floor evidence is not required: the reconstructed
floor is too sparse, so cells are free unless blocked. Cells with no Gaussians
at any height nearby are unknown, and so is everything outside the room
rectangle, which also acts as an obstacle so poorly scanned walls cannot leak.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from valmet.common.contracts import CellValues, NavigationMapMetadata, RobotClearance


@dataclass(frozen=True)
class NavParams:
    resolution: float = 0.05
    radius: float = 0.25
    height: float = 1.8
    head_clearance: float = 0.2  # extra headroom counted as obstacle above the body height
    max_step_height: float = 0.15
    min_opacity: float = 0.4
    min_points_per_cell: int = 2  # thin pipes and railings give few Gaussians per cell
    margin: float = 0.3
    observed_radius: float = 0.3  # cells farther than this from any Gaussian are unknown
    room_percentiles: tuple[float, float] = (0.5, 99.5)  # of body-band Gaussians, any opacity


OBSTACLES_FILE = "obstacles.npy"


def build_occupancy(
    positions: np.ndarray,
    opacities: np.ndarray,
    scene_id: str,
    params: NavParams,
) -> tuple[np.ndarray, NavigationMapMetadata, dict]:
    """Return (uint8 grid[ix, iy], metadata, info). Occupied cells include the radius inflation."""
    grid, _, navmap, info = build_layers(
        positions=positions,
        opacities=opacities,
        scene_id=scene_id,
        params=params,
    )
    return grid, navmap, info


def build_layers(
    positions: np.ndarray,
    opacities: np.ndarray,
    scene_id: str,
    params: NavParams,
) -> tuple[np.ndarray, np.ndarray, NavigationMapMetadata, dict]:
    """Like build_occupancy, plus the uninflated bool obstacle mask (walls and room boundary
    included), which the planner needs to fit a sideways body into gaps narrower than 2 * radius."""
    all_pts = np.asarray(positions, dtype=np.float64)
    band_top = params.height + params.head_clearance
    in_band = (all_pts[:, 2] >= params.max_step_height) & (all_pts[:, 2] <= band_top)
    room_lo, room_hi = np.percentile(all_pts[in_band, :2], params.room_percentiles, axis=0)
    pts = all_pts[np.asarray(opacities) >= params.min_opacity]
    lo = room_lo - params.margin
    hi = room_hi + params.margin
    shape = tuple(int(n) for n in np.ceil((hi - lo) / params.resolution))
    navmap = NavigationMapMetadata(
        scene_id=scene_id,
        origin=(float(lo[0]), float(lo[1]), 0.0),
        resolution=params.resolution,
        shape=shape,
        clearance=RobotClearance(
            radius=params.radius,
            height=params.height,
            max_step_height=params.max_step_height,
        ),
        obstacles_inflated=True,
    )

    cells = navmap.world_to_cell(pts[:, :2])
    inside = ((cells >= 0) & (cells < shape)).all(axis=1)
    cells, z = cells[inside], pts[inside, 2]

    body = (z >= params.max_step_height) & (z <= band_top)
    counts = np.zeros(shape, dtype=np.int64)
    np.add.at(counts, (cells[body, 0], cells[body, 1]), 1)
    blocked = counts >= params.min_points_per_cell

    seen = np.zeros(shape, dtype=bool)
    seen[cells[:, 0], cells[:, 1]] = True
    observed = ndimage.distance_transform_edt(~seen) * params.resolution <= params.observed_radius

    centers = navmap.cell_to_world(np.argwhere(np.ones(shape, dtype=bool))).reshape(*shape, 2)
    in_room = ((centers >= room_lo) & (centers <= room_hi)).all(axis=2)

    obstacles = blocked | ~in_room
    clearance = ndimage.distance_transform_edt(~obstacles) * params.resolution
    v = navmap.cell_values
    grid = np.full(shape, v.unknown, dtype=np.uint8)
    walkable = observed & in_room
    grid[walkable] = v.free
    grid[walkable & (clearance < params.radius)] = v.occupied

    info = {
        "params": asdict(params),
        "shape": list(shape),
        "room_rectangle": {"min": room_lo.tolist(), "max": room_hi.tolist()},
        "fractions": _fractions(grid=grid, values=v),
        "walkable_areas_m2": _area_sizes(grid=grid, values=v, resolution=params.resolution)[:5],
    }
    return grid, obstacles, navmap, info


def walkable_components(grid: np.ndarray, values: CellValues) -> tuple[np.ndarray, int]:
    """8-connected labels of free cells (0 = not free)."""
    return ndimage.label(grid == values.free, structure=np.ones((3, 3), dtype=bool))


def _fractions(grid: np.ndarray, values: CellValues) -> dict[str, float]:
    return {
        "free": float(np.mean(grid == values.free)),
        "occupied": float(np.mean(grid == values.occupied)),
        "unknown": float(np.mean(grid == values.unknown)),
    }


def _area_sizes(grid: np.ndarray, values: CellValues, resolution: float) -> list[float]:
    labels, n = walkable_components(
        grid=grid,
        values=values,
    )
    if n == 0:
        return []
    sizes = np.bincount(labels.ravel())[1:] * resolution**2
    return [round(float(s), 2) for s in np.sort(sizes)[::-1]]
