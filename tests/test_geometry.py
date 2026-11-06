from __future__ import annotations

import numpy as np
import pytest

from valmet.common import OrientedBox, SceneManifest
from valmet.common.geometry import apply_transform, as_array4, invert_transform, is_rotation

Y_UP_TO_Z_UP = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]


def test_column_vector_convention_translation_in_last_column():
    t = np.eye(4)
    t[:3, 3] = [1, 2, 3]
    assert np.allclose(apply_transform(t, np.zeros(3)), [1, 2, 3])
    assert np.allclose(apply_transform(t, np.ones((2, 3))), [[2, 3, 4]] * 2)


def test_y_up_source_maps_to_z_up_world():
    # Source "up" (+Y) must become world +Z; source +Z becomes world -Y. Right-handed preserved.
    out = apply_transform(Y_UP_TO_Z_UP, np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    assert np.allclose(out, [[0, 0, 1], [0, -1, 0], [1, 0, 0]])
    assert is_rotation(np.asarray(Y_UP_TO_Z_UP)[:3, :3])


def test_scale_and_inverse_roundtrip():
    t = np.diag([2.5, 2.5, 2.5, 1.0])
    t[:3, 3] = [-1, 0, 4]
    pts = np.random.default_rng(0).normal(size=(10, 3))
    assert np.allclose(apply_transform(invert_transform(t), apply_transform(t, pts)), pts)


def test_manifest_to_world_uses_its_transform():
    m = SceneManifest(scene_id="s", source_filename="a.ply", source_to_world=Y_UP_TO_Z_UP)
    assert np.allclose(m.to_world(np.array([[0.0, 2.0, 0.0]])), [[0, 0, 2]])
    # Default manifest is identity: nothing is silently normalized.
    assert np.allclose(SceneManifest(scene_id="s", source_filename="a.ply").to_world(np.ones(3)), 1)


@pytest.mark.parametrize(
    "bad",
    [
        np.eye(3),
        [[1, 0, 0, 0]] * 4,
        np.diag([1, 1, 0, 1]),
        [[np.nan] * 4] * 4,
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]],
    ],
)
def test_invalid_transforms(bad):
    with pytest.raises(ValueError):
        as_array4(bad)


def test_oriented_box_corners():
    box = OrientedBox(center=(1, 2, 3), extent=(2, 4, 6))
    c = box.corners()
    assert c.shape == (8, 3)
    assert np.allclose(c.min(0), [0, 0, 0]) and np.allclose(c.max(0), [2, 4, 6])
    # 90 deg yaw swaps the x/y footprint.
    rz = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    c = OrientedBox(center=(0, 0, 0), extent=(2, 4, 6), rotation=rz).corners()
    assert np.allclose(c.max(0), [2, 1, 3])
