from __future__ import annotations

import numpy as np
import pytest

from valmet.common.contracts import DetectedObject, OrientedBox
from valmet.spatial.proximity import footprint_distance, nearby_objects


def _obj(object_id: str, center, extent, yaw_deg: float = 0.0) -> DetectedObject:
    c, s = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    box = OrientedBox(center=center, extent=extent, rotation=[[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return DetectedObject(object_id=object_id, scene_id="s", center=center, bbox=box)


def test_footprint_distance_axis_aligned_and_inside():
    box = OrientedBox(center=(0.0, 0.0, 1.0), extent=(2.0, 1.0, 2.0))
    d = footprint_distance(points_xy=np.array([[2.0, 0.0], [0.0, 0.0], [2.0, 1.5]]), box=box)
    np.testing.assert_allclose(d, [1.0, 0.0, np.hypot(1.0, 1.0)])


def test_footprint_distance_rotated_box():
    obj = _obj("r", (0.0, 0.0, 0.5), (2.0, 0.2, 1.0), yaw_deg=45)
    # A point along the box's long axis, 0.5 m past its end.
    p = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)]) * 1.5
    assert footprint_distance(points_xy=p, box=obj.bbox)[0] == pytest.approx(0.5, abs=1e-9)


def test_nearby_objects_sorted_and_cut_off():
    objects = [
        _obj("far", (5.0, 0.0, 0.5), (1.0, 1.0, 1.0)),
        _obj("near", (1.0, 0.0, 0.5), (1.0, 1.0, 1.0)),
        _obj("mid", (0.0, 2.0, 0.5), (1.0, 1.0, 1.0)),
        _obj("under", (0.0, 0.0, 0.5), (0.4, 0.4, 1.0)),
    ]
    hits = nearby_objects(position=(0.0, 0.0, 0.0), objects=objects, radius=2.0)
    assert [h[0] for h in hits] == ["under", "near", "mid"]
    np.testing.assert_allclose([h[1] for h in hits], [0.0, 0.5, 1.5])
