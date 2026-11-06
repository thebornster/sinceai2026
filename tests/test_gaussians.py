from __future__ import annotations

from pathlib import Path

import numpy as np

from conftest import GS_PROPS, write_ply
from valmet.preprocessing.gaussians import (
    SH_C0,
    GaussianScene,
    load_gaussians,
    quat_to_matrix,
    transform_gaussians,
)


def _random_scene(n: int, seed: int) -> GaussianScene:
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return GaussianScene(
        positions=rng.normal(size=(n, 3)),
        quats_wxyz=q / np.linalg.norm(q, axis=1, keepdims=True),
        scales=rng.uniform(0.01, 0.5, (n, 3)),
        opacities=rng.uniform(0, 1, n),
        rgbs=rng.uniform(0, 1, (n, 3)),
        orig_index=np.arange(n),
    )


def test_load_activates_values(tmp_path: Path):
    rows = np.zeros((2, len(GS_PROPS)), dtype=np.float32)
    col = {p: i for i, p in enumerate(GS_PROPS)}
    rows[:, col["x"] : col["z"] + 1] = [[1, 2, 3], [4, 5, 6]]
    rows[:, col["opacity"]] = [0.0, 2.0]
    rows[:, col["scale_0"] : col["scale_2"] + 1] = np.log(0.1)
    rows[:, col["rot_0"]] = 2.0  # unnormalized identity
    rows[:, col["f_dc_0"]] = 1.0
    scene = load_gaussians(path=write_ply(tmp_path / "g.ply", GS_PROPS, rows))

    np.testing.assert_allclose(scene.positions, [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(scene.opacities, [0.5, 1 / (1 + np.exp(-2))], rtol=1e-6)
    np.testing.assert_allclose(scene.scales, 0.1, rtol=1e-6)
    np.testing.assert_allclose(scene.quats_wxyz, [[1, 0, 0, 0]] * 2)
    np.testing.assert_allclose(scene.rgbs[:, 0], 0.5 + SH_C0, rtol=1e-6)
    np.testing.assert_array_equal(scene.orig_index, [0, 1])


def test_transform_rotates_covariances():
    scene = _random_scene(n=200, seed=1)
    rng = np.random.default_rng(2)
    r, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    r *= np.sign(np.linalg.det(r))
    s, t = 1.7, np.array([0.3, -2.0, 5.0])
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = s * r, t

    out = transform_gaussians(
        scene=scene,
        matrix=m,
    )

    np.testing.assert_allclose(out.positions, scene.positions @ (s * r).T + t, atol=1e-12)
    expected = s**2 * np.einsum("ij,njk,lk->nil", r, scene.covariances(), r)
    np.testing.assert_allclose(out.covariances(), expected, atol=1e-10)
    np.testing.assert_allclose(np.linalg.norm(out.quats_wxyz, axis=1), 1.0)


def test_quat_to_matrix_is_rotation():
    r = quat_to_matrix(_random_scene(n=50, seed=3).quats_wxyz)
    np.testing.assert_allclose(r @ np.swapaxes(r, 1, 2), np.broadcast_to(np.eye(3), r.shape), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(r), 1.0)
