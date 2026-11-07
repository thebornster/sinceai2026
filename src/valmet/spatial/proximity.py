"""Horizontal distances from the avatar to object boxes, for the 'nearby assets' display."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from valmet.common.contracts import DetectedObject, OrientedBox


def footprint_distance(points_xy: np.ndarray, box: OrientedBox) -> np.ndarray:
    """(N,) XY distance from each point to the box's footprint (its corners projected to XY).

    0 inside the footprint. The footprint is convex, so the distance outside it equals the
    minimum distance to any segment between two projected corners.
    """
    pts = np.atleast_2d(np.asarray(points_xy, dtype=np.float64))[:, :2]
    corners = box.corners()[:, :2]
    a = np.array([corners[i] for i, _ in combinations(range(8), 2)])
    b = np.array([corners[j] for _, j in combinations(range(8), 2)])
    ab = b - a
    denom = np.maximum((ab**2).sum(axis=1), 1e-18)
    t = np.clip(((pts[:, None, :] - a) * ab).sum(axis=2) / denom, 0.0, 1.0)
    nearest = a + t[..., None] * ab
    dist = np.linalg.norm(pts[:, None, :] - nearest, axis=2).min(axis=1)

    try:
        hull = ConvexHull(corners)
    except QhullError:  # flat footprint (zero XY extent along one axis): no interior
        return dist
    inside = (pts @ hull.equations[:, :2].T + hull.equations[:, 2] <= 1e-12).all(axis=1)
    dist[inside] = 0.0
    return dist


def nearby_objects(
    position: np.ndarray, objects: list[DetectedObject], radius: float = 2.0
) -> list[tuple[str, float]]:
    """(object_id, distance) for objects whose footprint is within `radius`, nearest first."""
    xy = np.asarray(position, dtype=np.float64)[:2]
    hits = []
    for obj in objects:
        d = float(
            footprint_distance(
                points_xy=xy,
                box=obj.bbox,
            )[0]
        )
        if d <= radius:
            hits.append((obj.object_id, d))
    return sorted(hits, key=lambda h: h[1])
