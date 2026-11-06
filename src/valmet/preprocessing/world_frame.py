"""Estimate the source -> world similarity transform: +Z up, floor at z=0, walls on X/Y.

Assumes an indoor scan with a visible floor and roughly Manhattan walls. The up
direction comes from PCA (thinnest axis) and its sign from ``up_hint``: scans
reconstructed with OpenCV/COLMAP cameras have +Y pointing down, hence the
default hint (0, -1, 0).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class FrameParams:
    core_min_opacity: float = 0.5
    voxel_size: float = 0.02  # source units
    up_hint: tuple[float, float, float] = (0.0, -1.0, 0.0)
    slab_quantile: float = 0.3  # floor/ceiling peaks searched in the lowest/highest share of heights
    slab_half_width: float = 0.15  # source units around a peak passed to RANSAC
    plane_threshold: float = 0.01  # source units
    plane_iterations: int = 2000
    max_tilt_deg: float = 20.0
    seed: int = 0


@dataclass(frozen=True)
class ScaleSpec:
    """How to get meters: an explicit factor, or a known floor-to-ceiling height."""

    scale: float | None = None
    ceiling_height_m: float | None = None


def estimate_world_frame(
    positions: np.ndarray,
    opacities: np.ndarray,
    params: FrameParams,
    scale_spec: ScaleSpec,
) -> tuple[np.ndarray, dict]:
    """Return (4x4 source_to_world, info). positions/opacities are cleaned source data."""
    o3d.utility.random.seed(params.seed)
    pts = _core_points(
        positions=positions,
        opacities=opacities,
        params=params,
    )

    up = _up_direction(
        pts=pts,
        hint=np.asarray(params.up_hint, dtype=np.float64),
    )
    heights = pts @ up
    floor_n, floor_d, floor_inliers = _fit_slab_plane(
        pts=pts,
        heights=heights,
        up=up,
        side="low",
        params=params,
    )
    ceiling = None
    try:
        ceil_n, ceil_d, ceil_inliers = _fit_slab_plane(
            pts=pts,
            heights=heights,
            up=up,
            side="high",
            params=params,
        )
        ceiling = {
            "normal": ceil_n.tolist(),
            "d": ceil_d,
            "inliers": ceil_inliers,
            "distance_from_floor": floor_d - ceil_d,
        }
    except ValueError as exc:
        ceiling = {"error": str(exc)}

    level = _rotation_between(
        a=floor_n,
        b=np.array([0.0, 0.0, 1.0]),
    )
    leveled = pts @ level.T
    leveled[:, 2] += floor_d  # floor plane n.p + d = 0 -> z = -d before shift

    room_height = ceiling.get("distance_from_floor") or float(np.percentile(leveled[:, 2], 99))
    yaw_rad, manhattan_strength = _dominant_wall_yaw(
        pts=leveled,
        room_height=room_height,
        voxel_size=params.voxel_size,
    )
    yaw = _rot_z(-yaw_rad)
    aligned = leveled @ yaw.T
    lo, hi = np.percentile(aligned[:, :2], [1, 99], axis=0)
    if (hi - lo)[1] > (hi - lo)[0]:  # longest room side along X
        yaw = _rot_z(np.pi / 2) @ yaw
        aligned = leveled @ yaw.T
        lo, hi = np.percentile(aligned[:, :2], [1, 99], axis=0)
    xy_center = (lo + hi) / 2

    scale, scale_source = _resolve_scale(
        spec=scale_spec,
        floor_to_ceiling=ceiling.get("distance_from_floor"),
    )

    rotation = yaw @ level
    translation = np.array([-xy_center[0], -xy_center[1], 0.0]) + yaw @ np.array([0.0, 0.0, floor_d])
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = scale * translation

    info = {
        "params": asdict(params),
        "core_point_count": len(pts),
        "up_source": up.tolist(),
        "floor_plane_source": {"normal": floor_n.tolist(), "d": floor_d, "inliers": floor_inliers},
        "floor_tilt_from_pca_deg": float(np.degrees(np.arccos(np.clip(floor_n @ up, -1, 1)))),
        "ceiling_plane_source": ceiling,
        "wall_yaw_deg": float(np.degrees(yaw_rad)),
        "manhattan_strength": manhattan_strength,
        "scale": scale,
        "scale_source": scale_source,
    }
    return transform, info


def _core_points(positions: np.ndarray, opacities: np.ndarray, params: FrameParams) -> np.ndarray:
    pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(positions[opacities >= params.core_min_opacity])
    )
    return np.asarray(pcd.voxel_down_sample(voxel_size=params.voxel_size).points)


def _up_direction(pts: np.ndarray, hint: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(pts, [1, 99], axis=0)
    core = pts[((pts >= lo) & (pts <= hi)).all(axis=1)]
    _, vecs = np.linalg.eigh(np.cov((core - core.mean(axis=0)).T))
    up = vecs[:, 0]
    return up if up @ hint >= 0 else -up


def _fit_slab_plane(
    pts: np.ndarray, heights: np.ndarray, up: np.ndarray, side: str, params: FrameParams
) -> tuple[np.ndarray, float, int]:
    """RANSAC plane through the densest height slab at the low or high end. Normal along +up."""
    if side == "low":
        region = heights <= np.quantile(heights, params.slab_quantile)
    else:
        region = heights >= np.quantile(heights, 1 - params.slab_quantile)
    h = heights[region]
    bins = np.arange(h.min(), h.max() + params.voxel_size, params.voxel_size)
    counts, edges = np.histogram(h, bins=bins)
    peak = edges[np.argmax(counts)] + params.voxel_size / 2

    slab = pts[np.abs(heights - peak) <= params.slab_half_width]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(slab))
    (a, b, c, d), inliers = pcd.segment_plane(
        distance_threshold=params.plane_threshold,
        ransac_n=3,
        num_iterations=params.plane_iterations,
    )
    n = np.array([a, b, c])
    norm = np.linalg.norm(n)
    n, d = n / norm, d / norm
    if n @ up < 0:
        n, d = -n, -d
    tilt = np.degrees(np.arccos(np.clip(n @ up, -1, 1)))
    if tilt > params.max_tilt_deg:
        raise ValueError(f"{side} plane tilted {tilt:.1f} deg from the PCA up axis")
    return n, float(d), len(inliers)


def _dominant_wall_yaw(pts: np.ndarray, room_height: float, voxel_size: float) -> tuple[float, float]:
    """Wall direction mod 90 deg from near-horizontal normals (4-theta circular mean)."""
    band = pts[(pts[:, 2] > 0.15 * room_height) & (pts[:, 2] < 0.85 * room_height)]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(band))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_size, max_nn=30)
    )
    normals = np.asarray(pcd.normals)
    horizontal = normals[np.abs(normals[:, 2]) < 0.2]
    theta = np.arctan2(horizontal[:, 1], horizontal[:, 0])
    mean = np.exp(4j * theta).mean()
    return float(np.angle(mean) / 4), float(np.abs(mean))


def _resolve_scale(spec: ScaleSpec, floor_to_ceiling: float | None) -> tuple[float, str]:
    if spec.scale is not None and spec.ceiling_height_m is not None:
        raise ValueError("give either scale or ceiling_height_m, not both")
    if spec.scale is not None:
        return float(spec.scale), "explicit"
    if spec.ceiling_height_m is None:
        return 1.0, "unset"
    if floor_to_ceiling is None:
        raise ValueError("ceiling_height_m given but no ceiling plane was found")
    return float(spec.ceiling_height_m / floor_to_ceiling), "ceiling_height"


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    v = np.cross(a, b)
    c = float(a @ b)
    if c < -1 + 1e-9:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
