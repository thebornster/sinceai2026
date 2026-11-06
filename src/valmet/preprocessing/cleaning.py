"""Drop faint, oversized and stray Gaussians before any geometry is estimated."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import open3d as o3d

from valmet.preprocessing.gaussians import GaussianScene


@dataclass(frozen=True)
class CleaningParams:
    min_opacity: float = 0.1
    max_scale_percentile: float = 99.0
    crop_percentiles: tuple[float, float] = (0.2, 99.8)
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 2.0


def compute_keep_mask(scene: GaussianScene, params: CleaningParams) -> tuple[np.ndarray, dict]:
    """Boolean mask of Gaussians to keep, plus how many each filter removed."""
    removed: dict[str, int] = {}
    keep = np.isfinite(scene.positions).all(axis=1)
    removed["nonfinite"] = int((~keep).sum())

    step = keep & (scene.opacities >= params.min_opacity)
    removed["low_opacity"] = int(keep.sum() - step.sum())
    keep = step

    max_scale = scene.scales.max(axis=1)
    limit = np.percentile(max_scale[keep], params.max_scale_percentile)
    step = keep & (max_scale <= limit)
    removed["oversized"] = int(keep.sum() - step.sum())
    keep = step

    lo, hi = np.percentile(scene.positions[keep], params.crop_percentiles, axis=0)
    step = keep & ((scene.positions >= lo) & (scene.positions <= hi)).all(axis=1)
    removed["outside_crop"] = int(keep.sum() - step.sum())
    keep = step

    idx = np.flatnonzero(keep)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scene.positions[idx]))
    _, inliers = pcd.remove_statistical_outlier(
        nb_neighbors=params.outlier_neighbors,
        std_ratio=params.outlier_std_ratio,
    )
    step = np.zeros_like(keep)
    step[idx[np.asarray(inliers, dtype=np.int64)]] = True
    removed["statistical_outlier"] = int(keep.sum() - step.sum())
    keep = step

    stats = {
        "params": asdict(params),
        "input_count": len(scene),
        "kept_count": int(keep.sum()),
        "removed": removed,
        "max_scale_limit": float(limit),
        "crop_bounds_source": {"min": lo.tolist(), "max": hi.tolist()},
    }
    return keep, stats
