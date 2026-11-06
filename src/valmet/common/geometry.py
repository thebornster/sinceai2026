"""Coordinate conventions and 4x4 transform helpers shared by every pipeline.

Conventions (all modules):
- Points are XYZ triples. Arrays of points have shape (N, 3), float64.
- Transforms are 4x4 homogeneous matrices, row-major nested lists in JSON,
  applied to column vectors:  p_world = T @ [x, y, z, 1]^T.
  Translation is therefore the last *column* T[:3, 3].
- Rotations are 3x3 matrices that map box/local axes to world axes (columns
  are the local axes expressed in world coordinates).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Matrix4 = list[list[float]]


def identity4() -> Matrix4:
    return np.eye(4).tolist()


def as_array4(matrix: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return a validated (4, 4) float64 affine matrix."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got shape {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValueError("transform contains non-finite values")
    if not np.allclose(m[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"transform last row must be [0, 0, 0, 1], got {m[3].tolist()}")
    if abs(np.linalg.det(m[:3, :3])) < 1e-12:
        raise ValueError("transform linear part is singular")
    return m


def apply_transform(
    matrix: Sequence[Sequence[float]] | np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Apply a 4x4 transform to points of shape (N, 3) or (3,)."""
    m = as_array4(matrix)
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    if pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    out = pts @ m[:3, :3].T + m[:3, 3]
    return out[0] if single else out


def invert_transform(matrix: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    return np.linalg.inv(as_array4(matrix))


def is_rotation(r: Sequence[Sequence[float]] | np.ndarray, atol: float = 1e-6) -> bool:
    """True if r is a proper rotation (orthonormal, det = +1)."""
    r = np.asarray(r, dtype=np.float64)
    return (
        r.shape == (3, 3)
        and np.allclose(r.T @ r, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(r), 1.0, atol=atol)
    )
