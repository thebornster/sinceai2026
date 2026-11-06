"""Load 3DGS PLY vertices into activated arrays and apply similarity transforms to them.

Activations follow the reference 3DGS encoding (Kerbl et al. 2023): opacity is a
logit, scales are logs, rot_0..rot_3 is a wxyz quaternion and f_dc_* are degree-0
SH coefficients. Higher SH bands (f_rest_*) are not loaded: they are
view-dependent and would need an SH rotation under any transform.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from valmet.common.geometry import as_array4
from valmet.preprocessing.ply_inspect import (
    GS_REQUIRED,
    PlyInspectionError,
    _struct_dtype,
    read_header,
)

SH_C0 = 0.28209479177387814


@dataclass
class GaussianScene:
    positions: np.ndarray  # (N, 3) float64
    quats_wxyz: np.ndarray  # (N, 4) float64, unit norm
    scales: np.ndarray  # (N, 3) float64, activated
    opacities: np.ndarray  # (N,) float64 in [0, 1]
    rgbs: np.ndarray  # (N, 3) float64 in [0, 1]
    orig_index: np.ndarray  # (N,) int64, row in the source PLY

    def __len__(self) -> int:
        return len(self.positions)

    def subset(self, mask: np.ndarray) -> GaussianScene:
        return GaussianScene(**{f.name: getattr(self, f.name)[mask] for f in fields(self)})

    def covariances(self) -> np.ndarray:
        """(N, 3, 3) world covariances R diag(s^2) R^T."""
        r = quat_to_matrix(self.quats_wxyz)
        return np.einsum("nij,nj,nkj->nik", r, self.scales**2, r)


def load_gaussians(path: Path) -> GaussianScene:
    path = Path(path)
    header = read_header(path)
    vertex = header.element("vertex")
    if header.format == "ascii" or vertex is None or header.elements[0] is not vertex:
        raise PlyInspectionError("expected a binary PLY whose first element is 'vertex'")
    missing = set(GS_REQUIRED) - {p.name for p in vertex.properties}
    if missing:
        raise PlyInspectionError(f"vertex element lacks 3DGS properties: {sorted(missing)}")

    mm = np.memmap(
        path,
        dtype=_struct_dtype(vertex, header.format),
        mode="r",
        offset=header.header_bytes,
        shape=(vertex.count,),
    )
    try:
        col = lambda *names: np.stack([mm[n] for n in names], axis=1).astype(np.float64)  # noqa: E731
        positions = col("x", "y", "z")
        quats = col("rot_0", "rot_1", "rot_2", "rot_3")
        log_scales = col("scale_0", "scale_1", "scale_2")
        f_dc = col("f_dc_0", "f_dc_1", "f_dc_2")
        logit_opacity = np.asarray(mm["opacity"], dtype=np.float64)
    finally:
        del mm

    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = np.divide(quats, norms, out=np.tile([1.0, 0.0, 0.0, 0.0], (len(quats), 1)), where=norms > 0)
    return GaussianScene(
        positions=positions,
        quats_wxyz=quats,
        scales=np.exp(np.clip(log_scales, -60, 60)),
        opacities=1.0 / (1.0 + np.exp(-np.clip(logit_opacity, -60, 60))),
        rgbs=np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0),
        orig_index=np.arange(vertex.count, dtype=np.int64),
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz unit quaternions -> (N, 3, 3) rotation matrices."""
    w, x, y, z = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Single 3x3 rotation matrix -> wxyz unit quaternion."""
    r = np.asarray(r, dtype=np.float64)
    k = np.array(
        [
            [r[0, 0] + r[1, 1] + r[2, 2], r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]],
            [r[2, 1] - r[1, 2], r[0, 0] - r[1, 1] - r[2, 2], r[0, 1] + r[1, 0], r[0, 2] + r[2, 0]],
            [r[0, 2] - r[2, 0], r[0, 1] + r[1, 0], r[1, 1] - r[0, 0] - r[2, 2], r[1, 2] + r[2, 1]],
            [r[1, 0] - r[0, 1], r[0, 2] + r[2, 0], r[1, 2] + r[2, 1], r[2, 2] - r[0, 0] - r[1, 1]],
        ]
    )
    _, vecs = np.linalg.eigh(k)
    q = vecs[:, -1]
    return q if q[0] >= 0 else -q


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a ⊗ b for wxyz quaternions (broadcasting)."""
    aw, ax, ay, az = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def split_similarity(matrix: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Decompose T = [s R | t] into (s, R, t). Raises if T is not a similarity."""
    m = as_array4(matrix)
    lin = m[:3, :3]
    s = float(np.cbrt(np.linalg.det(lin)))
    r = lin / s
    if s <= 0 or not np.allclose(r.T @ r, np.eye(3), atol=1e-6):
        raise ValueError("transform is not a proper similarity (uniform scale + rotation)")
    return s, r, m[:3, 3]


def transform_gaussians(scene: GaussianScene, matrix: np.ndarray) -> GaussianScene:
    """Apply p' = s R p + t to positions, rotations and scales."""
    s, r, t = split_similarity(matrix=matrix)
    return GaussianScene(
        positions=scene.positions @ (s * r).T + t,
        quats_wxyz=quat_multiply(
            a=matrix_to_quat(r=r),
            b=scene.quats_wxyz,
        ),
        scales=scene.scales * s,
        opacities=scene.opacities,
        rgbs=scene.rgbs,
        orig_index=scene.orig_index,
    )
