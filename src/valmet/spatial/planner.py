"""Collision-free route on the inflated occupancy grid to a stand-off point next to a target box.

The grid is already inflated by the avatar radius, so the avatar is a point here.
Dijkstra over 8-connected free cells, with a cost that prefers open space so routes
follow the middle of aisles, then greedy line-of-sight shortcuts.

With the uninflated obstacle mask, gaps narrower than the shoulder width are also passable
sideways: the body becomes an ellipse (shoulder width x depth) with its long axis along the
direction of travel, checked for the 4 grid directions, at a higher step cost.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from valmet.common.contracts import NavigationMapMetadata, NavigationPath, OrientedBox
from valmet.spatial.occupancy import walkable_components
from valmet.spatial.proximity import footprint_distance

NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
# Index into the 4 sideways layers (0, 45, 90, 135 degrees) for each step direction.
ORIENTATION = {
    (1, 0): 0,
    (-1, 0): 0,
    (1, 1): 1,
    (-1, -1): 1,
    (0, 1): 2,
    (0, -1): 2,
    (1, -1): 3,
    (-1, 1): 3,
}


class UnreachableTargetError(Exception):
    """No free cell within inspection range of the target, or none connected to the start."""


@dataclass(frozen=True)
class PlannerParams:
    preferred_clearance: float = 0.5  # meters beyond the inflated boundary
    clearance_weight: float = 3.0
    standoff_min: float = 0.10  # beyond the avatar radius
    standoff_max: float = 1.0
    start_cost_percentile: float = 90.0
    max_start_snap: float = 1.0  # meters from a given start to the nearest free cell
    body_depth: float = 0.3  # chest-to-back, the width when moving sideways
    sideways_cost: float = 3.0  # step cost multiplier while squeezing sideways


def plan_path(
    navmap: NavigationMapMetadata,
    grid: np.ndarray,
    target_box: OrientedBox,
    target_object_id: str,
    start_xy: tuple[float, float] | None = None,
    params: PlannerParams = PlannerParams(),
    obstacles: np.ndarray | None = None,
) -> NavigationPath:
    """Pass the uninflated `obstacles` mask to allow sideways squeezing through narrow gaps."""
    free = grid == navmap.cell_values.free
    passable = None
    if obstacles is not None:
        passable = sideways_passable(
            navmap=navmap,
            grid=grid,
            obstacles=obstacles,
            body_depth=params.body_depth,
        )
    res = navmap.resolution
    clearance = ndimage.distance_transform_edt(free) * res
    labels, _ = walkable_components(
        grid=grid,
        values=navmap.cell_values,
    )

    free_cells = np.argwhere(free)
    standoff = footprint_distance(
        points_xy=navmap.cell_to_world(free_cells),
        box=target_box,
    )
    lo = navmap.clearance.radius + params.standoff_min
    in_range = (standoff >= lo) & (standoff <= params.standoff_max)
    if not in_range.any():
        raise UnreachableTargetError(
            f"no free cell {lo:.2f}-{params.standoff_max:.2f} m from {target_object_id!r}; "
            f"nearest free cell is {standoff.min():.2f} m away"
        )
    candidates = free_cells[in_range]
    candidate_standoff = standoff[in_range]
    cost_grid = 1.0 + params.clearance_weight * np.clip(
        (params.preferred_clearance - clearance) / params.preferred_clearance, 0.0, 1.0
    )

    if start_xy is None:
        goal = _best_candidate(
            candidates=candidates,
            standoff=candidate_standoff,
            labels=labels,
            clearance=clearance,
        )
        costs, parents = _dijkstra(
            source=goal,
            free=free,
            cost_grid=cost_grid,
            passable=passable,
            sideways_cost=params.sideways_cost,
        )
        start = _auto_start(
            costs=np.where(
                labels == labels[goal], costs, np.inf
            ),  # start where one can walk normally
            clearance=clearance,
            percentile=params.start_cost_percentile,
        )
        cells = _backtrack(parents=parents, cell=start)  # start -> goal
    else:
        start = _snap_to_free(
            navmap=navmap,
            free=free,
            xy=start_xy,
            max_distance=params.max_start_snap,
        )
        costs, parents = _dijkstra(
            source=start,
            free=free,
            cost_grid=cost_grid,
            passable=passable,
            sideways_cost=params.sideways_cost,
        )
        reached = np.isfinite(costs[candidates[:, 0], candidates[:, 1]])
        if not reached.any():
            raise UnreachableTargetError(f"{target_object_id!r} is not reachable from {start_xy}")
        rank = np.lexsort(
            (
                costs[candidates[reached, 0], candidates[reached, 1]],
                np.round(candidate_standoff[reached] / res),
            )
        )
        goal = tuple(candidates[reached][rank[0]])
        cells = _backtrack(parents=parents, cell=goal)[::-1]

    cells = _drop_collinear(
        cells=_shortcut(cells=cells, free=free),
        free=free,
    )
    xy = navmap.cell_to_world(np.array(cells))
    waypoints = [(float(x), float(y), 0.0) for x, y in xy]
    return NavigationPath(
        scene_id=navmap.scene_id,
        start=waypoints[0],
        target_object_id=target_object_id,
        inspection_position=waypoints[-1],
        waypoints=waypoints,
    )


def path_clearance(
    navmap: NavigationMapMetadata,
    grid: np.ndarray,
    path: NavigationPath,
    obstacles: np.ndarray | None = None,
) -> float:
    """Minimum distance from the path's cell centers to an obstacle cell center. Without the
    obstacle mask it is estimated from the inflated grid (distance to a non-free cell + radius)."""
    if obstacles is None:
        free = grid == navmap.cell_values.free
        clearance = (
            ndimage.distance_transform_edt(free) * navmap.resolution + navmap.clearance.radius
        )
    else:
        clearance = ndimage.distance_transform_edt(~obstacles) * navmap.resolution
    pts = _densify(np.asarray(path.waypoints)[:, :2], step=navmap.resolution / 2)
    cells = navmap.world_to_cell(pts)
    return float(clearance[cells[:, 0], cells[:, 1]].min())


def sideways_passable(
    navmap: NavigationMapMetadata,
    grid: np.ndarray,
    obstacles: np.ndarray,
    body_depth: float,
) -> list[np.ndarray]:
    """For 0, 45, 90 and 135 degrees: cells where the body fits walking normally or sideways
    along that direction (ellipse with semi-axes radius along it and body_depth / 2 across)."""
    res = navmap.resolution
    a, b = navmap.clearance.radius, body_depth / 2
    n = int(np.ceil(a / res))
    dx, dy = np.mgrid[-n : n + 1, -n : n + 1] * res
    free = grid == navmap.cell_values.free
    known = grid != navmap.cell_values.unknown
    layers = []
    for angle in np.deg2rad([0.0, 45.0, 90.0, 135.0]):
        c, s = np.cos(angle), np.sin(angle)
        kernel = ((dx * c + dy * s) / a) ** 2 + ((dy * c - dx * s) / b) ** 2 < 1.0
        blocked = ndimage.binary_dilation(obstacles, structure=kernel)
        layers.append(free | (known & ~blocked))
    return layers


def sideways_segments(
    navmap: NavigationMapMetadata, grid: np.ndarray, path: NavigationPath
) -> list[bool]:
    """Per path segment: True if it leaves the normal free space, i.e. the avatar moves sideways."""
    free = grid == navmap.cell_values.free
    cells = navmap.world_to_cell(np.asarray(path.waypoints)[:, :2])
    return [
        not _line_free(a=tuple(a), b=tuple(b), free=free)
        for a, b in zip(cells[:-1], cells[1:], strict=True)
    ]


def _best_candidate(
    candidates: np.ndarray, standoff: np.ndarray, labels: np.ndarray, clearance: np.ndarray
) -> tuple[int, int]:
    """Closest stand-off cell inside the largest walkable area that reaches the target."""
    comp = labels[candidates[:, 0], candidates[:, 1]]
    sizes = np.bincount(labels.ravel())
    keep = comp == comp[np.argmax(sizes[comp])]
    c, s = candidates[keep], standoff[keep]
    rank = np.lexsort((-clearance[c[:, 0], c[:, 1]], np.round(s, 2)))
    return tuple(int(v) for v in c[rank[0]])


def _auto_start(costs: np.ndarray, clearance: np.ndarray, percentile: float) -> tuple[int, int]:
    """A far but open cell: among cells beyond the cost percentile, the one with most clearance."""
    reachable = np.argwhere(np.isfinite(costs))
    c = costs[reachable[:, 0], reachable[:, 1]]
    far = reachable[c >= np.percentile(c, percentile)]
    return tuple(int(v) for v in far[np.argmax(clearance[far[:, 0], far[:, 1]])])


def _snap_to_free(
    navmap: NavigationMapMetadata, free: np.ndarray, xy: tuple[float, float], max_distance: float
) -> tuple[int, int]:
    """Nearest free cell to xy. Raises if none is within max_distance."""
    cells = np.argwhere(free)
    d = np.linalg.norm(navmap.cell_to_world(cells) - np.asarray(xy, dtype=np.float64), axis=1)
    if d.min() > max_distance:
        raise UnreachableTargetError(f"start {xy} is {d.min():.2f} m from the nearest free cell")
    return tuple(int(v) for v in cells[np.argmin(d)])


def _dijkstra(
    source: tuple[int, int],
    free: np.ndarray,
    cost_grid: np.ndarray,
    passable: list[np.ndarray] | None = None,
    sideways_cost: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A step must stay inside the passable layer of its direction; default: free cells only."""
    layers = passable if passable is not None else [free] * 4
    nx, ny = free.shape
    costs = np.full(free.shape, np.inf)
    parents = np.full((*free.shape, 2), -1, dtype=np.int64)
    costs[source] = 0.0
    heap = [(0.0, source)]
    while heap:
        c, (i, j) = heapq.heappop(heap)
        if c > costs[i, j]:
            continue
        for di, dj in NEIGHBORS:
            a, b = i + di, j + dj
            ok = layers[ORIENTATION[(di, dj)]]
            if not (0 <= a < nx and 0 <= b < ny) or not (ok[a, b] and ok[i, j]):
                continue
            if di and dj and not (ok[i + di, j] and ok[i, j + dj]):  # no corner cutting
                continue
            step = (
                (1.4142135623730951 if di and dj else 1.0)
                * 0.5
                * (cost_grid[i, j] + cost_grid[a, b])
            )
            if not (free[i, j] and free[a, b]):
                step *= sideways_cost
            if c + step < costs[a, b]:
                costs[a, b] = c + step
                parents[a, b] = (i, j)
                heapq.heappush(heap, (c + step, (a, b)))
    return costs, parents


def _backtrack(parents: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    """Cells from `cell` back to the Dijkstra source."""
    out = [cell]
    while parents[out[-1]][0] >= 0:
        out.append(tuple(int(v) for v in parents[out[-1]]))
    return out


def _shortcut(cells: list[tuple[int, int]], free: np.ndarray) -> list[tuple[int, int]]:
    """Greedy line-of-sight smoothing: from each kept cell jump to the farthest visible one."""
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not _line_free(a=cells[i], b=cells[j], free=free):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def _drop_collinear(cells: list[tuple[int, int]], free: np.ndarray) -> list[tuple[int, int]]:
    """Merge same-direction steps (left by squeezes, which are not shortcut), without merging
    normal and sideways stretches."""
    out = cells[:2]
    for c in cells[2:]:
        (x0, y0), (x1, y1) = out[-2], out[-1]
        same_dir = (x1 - x0) * (c[1] - y1) == (y1 - y0) * (c[0] - x1)
        forward = (x1 - x0) * (c[0] - x1) + (y1 - y0) * (c[1] - y1) > 0
        if same_dir and forward and free[x0, y0] == free[x1, y1] == free[c]:
            out[-1] = c
        else:
            out.append(c)
    return out


def _line_free(a: tuple[int, int], b: tuple[int, int], free: np.ndarray) -> bool:
    pts = _densify(np.array([a, b], dtype=np.float64), step=0.5)
    # Check all cells a sample touches, so lines can't squeeze diagonally between blocked cells.
    for off in ((0.0, 0.0), (0.49, 0.49), (0.49, -0.49), (-0.49, 0.49), (-0.49, -0.49)):
        idx = np.rint(pts + off).astype(np.int64)
        if not free[idx[:, 0], idx[:, 1]].all():
            return False
    return True


def _densify(points: np.ndarray, step: float) -> np.ndarray:
    out = [points[:1]]
    for a, b in zip(points[:-1], points[1:], strict=True):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step)))
        out.append(a + (b - a) * (np.arange(1, n + 1)[:, None] / n))
    return np.concatenate(out)
