"""Walking along waypoints and finding what is near the avatar. No viser needed."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


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
