"""Walking along waypoints and finding what is near the avatar. No viser needed."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

from valmet.common.contracts import OrientedBox


class PathPlayer:
    """Tracks how far along a list of waypoints the avatar is, at a constant speed."""

    def __init__(self, waypoints: Sequence[Sequence[float]]) -> None:
        self.load(waypoints)

    def load(self, waypoints: Sequence[Sequence[float]]) -> None:
        """Switch to a new route, stopped at its first waypoint."""
        self.waypoints = np.asarray(waypoints, dtype=float)
        segment_lengths = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        # cumulative[i] is the distance from the start to waypoint i.
        self.cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        self.total_length = float(self.cumulative[-1])
        # Legs with a length, so repeated waypoints never decide the heading.
        legs = np.flatnonzero(segment_lengths > 0.0)
        self.leg_starts = self.cumulative[legs]
        steps = np.diff(self.waypoints, axis=0)[legs]
        self.leg_headings = np.arctan2(steps[:, 1], steps[:, 0])
        self.distance = 0.0
        self.playing = False

    @property
    def finished(self) -> bool:
        return self.distance >= self.total_length

    def play(self) -> None:
        if self.finished:  # at the end: start over
            self.distance = 0.0
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def reset(self) -> None:
        self.playing = False
        self.distance = 0.0

    def update(self, dt: float, speed: float) -> float:
        """Move forward by speed (m/s) for dt seconds, stopping at the last waypoint.

        Returns the distance actually walked.
        """
        if not self.playing:
            return 0.0
        before = self.distance
        self.distance = min(before + speed * dt, self.total_length)
        if self.finished:
            self.playing = False
        return self.distance - before

    def heading(self) -> float:
        """Direction of the current leg in radians, or 0 if the route has no length."""
        if len(self.leg_headings) == 0:
            return 0.0
        leg = np.searchsorted(self.leg_starts, self.distance, side="right") - 1
        return float(self.leg_headings[leg])

    def position(self) -> np.ndarray:
        """Current point on the route, between two waypoints."""
        return np.array(
            [
                np.interp(self.distance, self.cumulative, self.waypoints[:, axis])
                for axis in range(3)
            ]
        )


def find_nearby(position: np.ndarray, centers: Mapping[str, np.ndarray], radius: float) -> set[str]:
    """Ids of the objects whose center is within radius of position."""
    return {
        object_id
        for object_id, center in centers.items()
        if np.linalg.norm(center - position) <= radius
    }


def find_nearest(position: np.ndarray, centers: Mapping[str, np.ndarray]) -> str:
    """Id of the object whose center is closest to position."""
    return min(centers, key=lambda object_id: np.linalg.norm(centers[object_id] - position))


def ray_box_distance(origin: np.ndarray, direction: np.ndarray, box: OrientedBox) -> float | None:
    """Distance along the ray to where it enters the box, or None if it misses.

    A ray starting inside the box does not count as a hit, so a big box around the
    camera never hides the objects in front of it.
    """
    rotation = np.asarray(box.rotation)
    o = (np.asarray(origin, dtype=float) - np.asarray(box.center)) @ rotation
    d = np.asarray(direction, dtype=float) @ rotation
    d = d / np.linalg.norm(d)
    half = np.asarray(box.extent) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-half - o) / d, (half - o) / d
    # Axes the ray runs parallel to: inside the slab -> no limit, outside -> miss.
    parallel = d == 0.0
    if np.any(parallel & (np.abs(o) > half)):
        return None
    near = np.where(parallel, -np.inf, np.minimum(t1, t2)).max()
    far = np.where(parallel, np.inf, np.maximum(t1, t2)).min()
    if near > far or near <= 0.0:
        return None
    return float(near)


def pick_object(
    origin: np.ndarray, direction: np.ndarray, boxes: Mapping[str, OrientedBox]
) -> str | None:
    """Id of the first box the ray enters, or None."""
    best_id, best_t = None, np.inf
    for object_id, box in boxes.items():
        t = ray_box_distance(origin, direction, box)
        if t is not None and t < best_t:
            best_id, best_t = object_id, t
    return best_id


class RayPick(NamedTuple):
    objects: list[int]  # object indices along the ray, front to back; [0] is under the cursor
    point: np.ndarray | None  # where the ray meets the picked object


def pick_along_ray(
    origin: np.ndarray,
    direction: np.ndarray,
    points: np.ndarray,
    labels: np.ndarray,
    transparent: np.ndarray,
    angle: float = 0.008,
    min_radius: float = 0.01,
    surface_depth: float = 0.08,
) -> RayPick:
    """What a click ray hits in a labelled point cloud.

    labels: object index per point, or negative for structure (walls, floor), which blocks
    the ray but is never picked. transparent[k]: object k is hidden and ignored. Points count
    when within `angle` radians of the ray (`min_radius` meters up close). The first surface is
    the nearest such point plus `surface_depth`; on it the object closest to the ray wins, so
    aim beats depth. A surface of only structure picks nothing.
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    keep = (labels < 0) | ~transparent[np.maximum(labels, 0)]
    pts, lab = points[keep], labels[keep]
    v = pts - np.asarray(origin, dtype=float)
    t = v @ d
    miss = np.linalg.norm(v - np.outer(t, d), axis=1)
    near = (t > 0.05) & (miss < np.maximum(angle * t, min_radius))
    if not near.any():
        return RayPick([], None)
    t, miss, lab, pts = t[near], miss[near], lab[near], pts[near]
    surface = (t < t.min() + surface_depth) & (lab >= 0)
    if not surface.any():
        return RayPick([], None)
    best = np.flatnonzero(surface)[np.argmin((miss / t)[surface])]
    first = int(lab[best])
    objects = lab >= 0
    by_depth = lab[objects][np.argsort(t[objects], kind="stable")]
    ids, first_seen = np.unique(by_depth, return_index=True)
    behind = [int(k) for k in ids[np.argsort(first_seen)] if k != first]
    return RayPick([first, *behind], pts[best])
