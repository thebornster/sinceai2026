"""Split the cleaned world-frame scene into structural planes and equipment objects.

Steps: voxel + SOR -> axis-aligned structural planes (floor, ceiling, walls) -> extra RANSAC
planes -> DBSCAN -> direction-class split of oversized clusters (pipe runs along x/y/z vs
fittings) -> noise reassignment -> overhead filter -> collinear pipe merge -> containment merge.

Ported from the field_copilot demo (PR #5) to the shared world frame (+Z up, meters, floor z=0).

Inputs : <scene_dir>/scene_manifest.json, <scene_dir>/scene_clean.npz
Outputs: <scene_dir>/objects.json                  list[DetectedObject], label=None
         <scene_dir>/semantic_details.json         per-object extras + structural planes
         <scene_dir>/semantic/segment_points.npz   voxel points + per-point object index
         <scene_dir>/semantic/members/obj_xxxx.npy source-PLY indices per object
         reports/semantic_<scene_id>.json, reports/preview/<scene_id>_segment_{front,top}.png

Usage:
    python -m valmet.semantic.segment --scene-dir data/processed/equipment_room
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from valmet.common.artifacts import (  # noqa: E402
    MANIFEST_FILE,
    OBJECTS_FILE,
    REPO_ROOT,
    load_model,
    save_list,
)
from valmet.common.contracts import (  # noqa: E402
    DetectedObject,
    GaussianMembership,
    OrientedBox,
    SceneManifest,
)
from valmet.preprocessing.export import SCENE_CLEAN_FILE  # noqa: E402

DETAILS_FILE = "semantic_details.json"
SEMANTIC_DIR = "semantic"
SEGMENT_POINTS_FILE = f"{SEMANTIC_DIR}/segment_points.npz"
MEMBERS_DIR = f"{SEMANTIC_DIR}/members"

NOISE = -1
STRUCTURE = -2
OTHER = 3  # direction class for non-linear points (x=0, y=1, z=2)
AXIS_NAMES = "xyz"


@dataclass(frozen=True)
class SegmentParams:
    """Distances in meters (the demo's file-unit values x 1.15)."""

    voxel: float = 0.023
    outlier_nb: int = 20
    outlier_std: float = 2.0
    normal_knn: int = 16
    opacity_knn: int = 4
    # structural planes: |normal . axis| above this, within plane_tol of the detected plane
    plane_normal_dot: float = 0.8
    plane_tol: float = 0.058
    plane_bin: float = 0.023
    plane_peak_frac: float = 0.25  # histogram peak must hold this fraction of the axis maximum
    ransac_dist: float = 0.023
    ransac_n: int = 3
    ransac_iters: int = 2000
    ransac_min_frac: float = 0.05
    ransac_max_planes: int = 4
    ransac_component_eps: float = 0.069
    ransac_component_min_frac: float = 0.02
    dbscan_eps: float = 0.052
    dbscan_min_points: int = 6
    min_cluster_points: int = 60
    noise_reassign_dist: float = 0.069
    contain_tol: float = 0.023
    collinear_min_cross_overlap: float = 0.5
    collinear_max_gap: float = 0.17
    overhead_min_points: int = 60
    overhead_min_opacity: float = 0.3
    overhead_margin: float = 0.17
    split_max_diag: float = 1.73
    split_max_points: int = 5000
    pca_knn: int = 24
    lin_thresh: float = 0.5
    cyl_thresh: float = 0.25
    axis_align: float = 0.85
    smooth_knn: int = 20
    class_eps: float = 0.052
    class_min_points: int = 6
    min_part_points: int = 40
    fitting_max_diag: float = 0.69
    pipe_linearity: float = 0.9
    member_dist: float = 0.035  # Gaussian -> object if within this of a segmented point
    seed: int = 0


# ---------- loading / cleanup ----------


def load_scene(scene_dir: Path) -> dict[str, np.ndarray]:
    d = np.load(scene_dir / SCENE_CLEAN_FILE)
    return {
        "xyz": d["positions"].astype(np.float64),
        "rgb": d["rgbs"].astype(np.float64),
        "opacity": d["opacities"].astype(np.float64),
        "orig_index": d["orig_index"].astype(np.int64),
    }


def downsample_and_denoise(d: dict[str, np.ndarray], sp: SegmentParams) -> tuple[np.ndarray, ...]:
    """Voxel + SOR. Returns xyz, rgb, opacity (from nearest originals), normals."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(d["xyz"]))
    pc.colors = o3d.utility.Vector3dVector(np.clip(d["rgb"], 0, 1))
    pc = pc.voxel_down_sample(sp.voxel)
    pc, _ = pc.remove_statistical_outlier(sp.outlier_nb, sp.outlier_std)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(sp.normal_knn))
    xyz = np.asarray(pc.points)
    _, nn = cKDTree(d["xyz"]).query(xyz, k=sp.opacity_knn)
    opacity = d["opacity"][nn].mean(1)
    print(f"[clean] voxel {sp.voxel} + SOR -> {len(xyz):,} points")
    return xyz, np.asarray(pc.colors), opacity, np.asarray(pc.normals)


# ---------- planes ----------


def _plane_value(values: np.ndarray, side: str, sp: SegmentParams) -> float | None:
    """Outermost histogram peak in one half of the axis range, refined to the median of its
    band. Peaks are relative to that half so the floor does not hide the ceiling."""
    if len(values) == 0:
        return None
    mid = (values.min() + values.max()) / 2
    values = values[values >= mid] if side == "max" else values[values <= mid]
    edges = np.arange(values.min(), values.max() + 2 * sp.plane_bin, sp.plane_bin)
    counts, edges = np.histogram(values, bins=edges)
    peaks = np.flatnonzero(counts >= sp.plane_peak_frac * counts.max())
    k = peaks[-1] if side == "max" else peaks[0]
    center = (edges[k] + edges[k + 1]) / 2
    return float(np.median(values[np.abs(values - center) < sp.plane_tol]))


def detect_planes(xyz: np.ndarray, normals: np.ndarray, sp: SegmentParams) -> list[dict]:
    """Floor at z=0, ceiling = highest z peak, walls = outermost x/y peaks (normal-aligned points)."""
    specs = [
        ("floor", "floor", 2, None),
        ("ceiling", "ceiling", 2, "max"),
        ("wall_x_min", "wall", 0, "min"),
        ("wall_x_max", "wall", 0, "max"),
        ("wall_y_min", "wall", 1, "min"),
        ("wall_y_max", "wall", 1, "max"),
    ]
    planes = []
    for name, label, axis, side in specs:
        aligned = np.abs(normals[:, axis]) > sp.plane_normal_dot
        value = 0.0 if side is None else _plane_value(xyz[aligned, axis], side, sp)
        if value is not None:
            planes.append({"name": name, "label": label, "axis": axis, "value": value})
    return planes


def remove_planes(
    xyz: np.ndarray, normals: np.ndarray, planes: list[dict], sp: SegmentParams
) -> np.ndarray:
    """Structural plane point = inside the band AND normal aligned with the plane axis.
    Adds `idx` to each plane, returns the mask of remaining points."""
    remaining = np.ones(len(xyz), bool)
    for p in planes:
        band = np.abs(xyz[:, p["axis"]] - p["value"]) < sp.plane_tol
        m = remaining & band & (np.abs(normals[:, p["axis"]]) > sp.plane_normal_dot)
        p["idx"] = np.flatnonzero(m)
        remaining &= ~m
        print(
            f"[plane] {p['name']} {AXIS_NAMES[p['axis']]}={p['value']:.3f}: {m.sum():,} points "
            f"({(band & ~m).sum():,} in band kept, normal not aligned)"
        )
    return remaining


def _dbscan(pts: np.ndarray, eps: float, min_pts: int) -> np.ndarray:
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    return np.asarray(pc.cluster_dbscan(eps=eps, min_points=min_pts))


def _groups(lab: np.ndarray, min_size: int) -> list[np.ndarray]:
    ids, cnt = np.unique(lab[lab >= 0], return_counts=True)
    return [np.flatnonzero(lab == i) for i, c in zip(ids, cnt, strict=True) if c >= min_size]


def ransac_extra_planes(xyz: np.ndarray, remaining: np.ndarray, sp: SegmentParams) -> list[dict]:
    """Peel off further large connected planes (> min_frac of remaining). Mutates `remaining`."""
    o3d.utility.random.seed(sp.seed)
    planes = []
    for k in range(sp.ransac_max_planes):
        idx = np.flatnonzero(remaining)
        if len(idx) < sp.ransac_n:
            break
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz[idx]))
        model, inl = pc.segment_plane(sp.ransac_dist, sp.ransac_n, sp.ransac_iters)
        inl = np.asarray(inl)
        comps = _groups(
            _dbscan(xyz[idx[inl]], sp.ransac_component_eps, 4),
            int(sp.ransac_component_min_frac * len(idx)),
        )
        sel = idx[inl[np.concatenate(comps)]] if comps else np.array([], int)
        if len(sel) < sp.ransac_min_frac * len(idx):
            break
        normal = np.round(np.asarray(model[:3]), 3).tolist()
        planes.append({"name": f"plane_{k}", "label": "wall", "normal": normal, "idx": sel})
        remaining[sel] = False
        print(f"[ransac] plane_{k} normal {normal}: {len(sel):,} points")
    return planes


# ---------- clustering ----------


def dbscan_clusters(xyz: np.ndarray, sp: SegmentParams) -> list[np.ndarray]:
    groups = _groups(_dbscan(xyz, sp.dbscan_eps, sp.dbscan_min_points), sp.min_cluster_points)
    print(f"[dbscan] {len(groups)} clusters >= {sp.min_cluster_points} pts")
    return groups


def direction_classes(xyz: np.ndarray, normals: np.ndarray, sp: SegmentParams) -> np.ndarray:
    """Per-point class 0/1/2 = pipe-like along x/y/z, 3 = other.
    Pipe-like if point-PCA linearity or normal-scatter cylindricity is high and
    the implied axis is near a coordinate axis. Smoothed by kNN majority vote."""
    k = min(sp.pca_knn, len(xyz))
    _, nn = cKDTree(xyz).query(xyz, k=k)
    nb = xyz[nn] - xyz[nn].mean(1, keepdims=True)
    w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", nb, nb))
    lin = (w[:, 2] - w[:, 1]) / np.maximum(w[:, 2], 1e-12)
    nrm = normals[nn]
    wn, vn = np.linalg.eigh(np.einsum("nki,nkj->nij", nrm, nrm) / k)
    cyl = (wn[:, 1] - wn[:, 0]) / np.maximum(wn[:, 2], 1e-12)
    use_cyl = cyl > sp.cyl_thresh
    axis = np.where(use_cyl[:, None], vn[:, :, 0], v[:, :, 2])
    aligned = np.abs(axis).max(1) > sp.axis_align
    pipe = (use_cyl | (lin > sp.lin_thresh)) & aligned
    cls = np.where(pipe, np.abs(axis).argmax(1), OTHER)
    _, nn = cKDTree(xyz).query(xyz, k=min(sp.smooth_knn, len(xyz)))
    votes = np.zeros((len(xyz), 4), np.int32)
    np.add.at(votes, (np.repeat(np.arange(len(xyz)), nn.shape[1]), cls[nn].ravel()), 1)
    return votes.argmax(1)


def is_oversized(pts: np.ndarray, sp: SegmentParams) -> bool:
    diag = float(np.linalg.norm(np.ptp(pts, 0)))
    return diag > sp.split_max_diag or len(pts) > sp.split_max_points


Part = tuple[np.ndarray, str | None, int]  # (indices, hint, axis class)


def split_by_direction(
    g: np.ndarray, xyz: np.ndarray, cls: np.ndarray, sp: SegmentParams
) -> list[Part]:
    """Split one cluster into straight pipe runs per axis + non-linear parts."""
    parts: list[Part] = []
    for c in range(4):
        sub = g[cls[g] == c]
        if len(sub) < sp.min_part_points:
            continue
        for p in _groups(_dbscan(xyz[sub], sp.class_eps, sp.class_min_points), sp.min_part_points):
            idx = sub[p]
            diag = float(np.linalg.norm(np.ptp(xyz[idx], 0)))
            hint = (
                "pipe segment"
                if c != OTHER
                else ("fitting/valve" if diag <= sp.fitting_max_diag else None)
            )
            parts.append((idx, hint, c))
    return parts


def split_oversized(
    groups: list[np.ndarray], xyz: np.ndarray, normals: np.ndarray, sp: SegmentParams
) -> list[Part]:
    cls = direction_classes(xyz, normals, sp)
    print(f"[split] direction classes x/y/z/other: {np.bincount(cls, minlength=4).tolist()}")
    out: list[Part] = []
    n_big = 0
    for g in groups:
        if not is_oversized(xyz[g], sp):
            out.append((g, None, OTHER))
            continue
        n_big += 1
        parts = split_by_direction(g, xyz, cls, sp) or [(g, None, OTHER)]
        kept = sum(len(p) for p, _, _ in parts)
        print(
            f"[split] {len(g):,} pts diag {np.linalg.norm(np.ptp(xyz[g], 0)):.2f} -> "
            f"{len(parts)} parts ({kept / len(g):.0%} of points)"
        )
        out += parts
    print(f"[split] {n_big} oversized clusters -> total {len(out)} clusters")
    return out


def reassign_noise(labels: np.ndarray, xyz: np.ndarray, max_dist: float) -> np.ndarray:
    """Give noise points the label of the nearest clustered point within max_dist."""
    lab = labels.copy()
    noise, clus = np.flatnonzero(lab == NOISE), np.flatnonzero(lab >= 0)
    if len(noise) == 0 or len(clus) == 0:
        return lab
    d, j = cKDTree(xyz[clus]).query(xyz[noise], distance_upper_bound=max_dist)
    ok = np.isfinite(d)
    lab[noise[ok]] = lab[clus[j[ok]]]
    print(f"[noise] reassigned {ok.sum():,} of {len(noise):,} noise points (<= {max_dist})")
    return lab


# ---------- post-filters on object list ----------


def filter_overhead(
    objs: list[dict], xyz: np.ndarray, opacity: np.ndarray, ceiling_z: float | None, sp: SegmentParams
) -> list[dict]:
    """Clusters above the ceiling are tagged 'overhead'. Those lying entirely more than
    `overhead_margin` above it are kept only if big and opaque enough (else floaters)."""
    if ceiling_z is None:
        return objs
    out, dropped = [], 0
    for o in objs:
        z = xyz[o["idx"], 2]
        if z.mean() > ceiling_z:
            clearly_above = z.min() - ceiling_z > sp.overhead_margin
            weak = (
                len(o["idx"]) < sp.overhead_min_points
                or opacity[o["idx"]].mean() <= sp.overhead_min_opacity
            )
            if clearly_above and weak:
                dropped += 1
                continue
            o["tags"].append("overhead")
        out.append(o)
    print(f"[overhead] kept {sum('overhead' in o['tags'] for o in out)}, dropped {dropped}")
    return out


def merge_collinear_pipes(objs: list[dict], xyz: np.ndarray, sp: SegmentParams) -> list[dict]:
    """Union pipe segments on the same axis whose cross-sections overlap and whose
    gap along the axis is small (one physical pipe broken by junctions)."""
    pipes = [i for i, o in enumerate(objs) if o["hint"] == "pipe segment" and o["axis"] < OTHER]
    box = {i: (xyz[objs[i]["idx"]].min(0), xyz[objs[i]["idx"]].max(0)) for i in pipes}
    parent = {i: i for i in pipes}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a_i, i in enumerate(pipes):
        for j in pipes[a_i + 1 :]:
            ax = objs[i]["axis"]
            if objs[j]["axis"] != ax:
                continue
            (lo1, hi1), (lo2, hi2) = box[i], box[j]
            cross = [k for k in range(3) if k != ax]
            inter = np.prod([max(0.0, min(hi1[k], hi2[k]) - max(lo1[k], lo2[k])) for k in cross])
            smaller = min(
                np.prod([hi1[k] - lo1[k] for k in cross]),
                np.prod([hi2[k] - lo2[k] for k in cross]),
            )
            gap = max(lo1[ax], lo2[ax]) - min(hi1[ax], hi2[ax])
            if (
                smaller > 0
                and inter / smaller > sp.collinear_min_cross_overlap
                and gap < sp.collinear_max_gap
            ):
                parent[find(j)] = find(i)
    merged: dict[int, dict] = {}
    for i in pipes:
        r = find(i)
        if r in merged:
            merged[r]["idx"] = np.concatenate([merged[r]["idx"], objs[i]["idx"]])
            merged[r]["tags"] = sorted(set(merged[r]["tags"]) | set(objs[i]["tags"]))
        else:
            merged[r] = objs[i]
    out = [o for i, o in enumerate(objs) if i not in parent] + list(merged.values())
    print(f"[collinear] {len(pipes)} pipe segments -> {len(merged)} pipe runs")
    return out


def merge_contained(objs: list[dict], xyz: np.ndarray, tol: float) -> list[dict]:
    """Merge an object into a larger one of the same hint whose box fully contains it."""
    objs = sorted(objs, key=lambda o: -len(o["idx"]))
    boxes = [(xyz[o["idx"]].min(0), xyz[o["idx"]].max(0)) for o in objs]
    alive = [True] * len(objs)
    for i in range(len(objs)):
        for j in range(i):
            if (
                alive[j]
                and objs[i]["hint"] == objs[j]["hint"]
                and (boxes[i][0] >= boxes[j][0] - tol).all()
                and (boxes[i][1] <= boxes[j][1] + tol).all()
            ):
                objs[j]["idx"] = np.concatenate([objs[j]["idx"], objs[i]["idx"]])
                alive[i] = False
                break
    print(f"[contain] merged {alive.count(False)} contained boxes")
    return [o for o, a in zip(objs, alive, strict=True) if a]


# ---------- features / output ----------


def shape_features(pts: np.ndarray) -> dict[str, float]:
    if len(pts) < 2:
        return {"linearity": 0.0, "planarity": 0.0, "sphericity": 1.0}
    ev = np.sort(np.linalg.eigvalsh(np.cov((pts - pts.mean(0)).T)))[::-1]
    ev = np.maximum(ev, 1e-12)
    return {
        "linearity": float((ev[0] - ev[1]) / ev[0]),
        "planarity": float((ev[1] - ev[2]) / ev[0]),
        "sphericity": float(ev[2] / ev[0]),
    }


def object_details(
    idx: np.ndarray,
    xyz: np.ndarray,
    rgb: np.ndarray,
    opacity: np.ndarray,
    hint: str | None,
    tags: list[str],
    sp: SegmentParams,
) -> dict[str, Any]:
    pts = xyz[idx]
    lo, hi = pts.min(0), pts.max(0)
    f = shape_features(pts)
    f.update(
        {
            "height_bottom": float(lo[2]),
            "height_top": float(hi[2]),
            "diag": float(np.linalg.norm(hi - lo)),
            "mean_opacity": float(opacity[idx].mean()),
        }
    )
    if hint is None and f["linearity"] > sp.pipe_linearity:
        hint = "pipe segment"
    return {
        "kind": "object",
        "hint": hint,
        "tags": tags,
        "n_points": int(len(idx)),
        "color": rgb[idx].mean(0).round(3).tolist(),
        "bbox_min": lo.round(4).tolist(),
        "bbox_max": hi.round(4).tolist(),
        "features": {k: round(v, 4) for k, v in f.items()},
    }


def order_objects(objs: list[dict], xyz: np.ndarray) -> list[dict]:
    """Stable id order: most points first, ties by centroid."""
    return sorted(objs, key=lambda o: (-len(o["idx"]), *np.round(xyz[o["idx"]].mean(0), 4)))


def membership(
    gaussians_xyz: np.ndarray, xyz: np.ndarray, labels: np.ndarray, max_dist: float
) -> np.ndarray:
    """Per-Gaussian label of the nearest segmented point within max_dist, else NOISE."""
    dist, j = cKDTree(xyz).query(gaussians_xyz, k=1, distance_upper_bound=max_dist)
    ok = np.isfinite(dist)
    out = np.full(len(gaussians_xyz), NOISE, np.int32)
    out[ok] = labels[j[ok]]
    return out


def object_id(i: int) -> str:
    return f"obj_{i + 1:04d}"


# ---------- plots ----------


def plot_view(
    xyz: np.ndarray,
    labels: np.ndarray,
    details: list[dict],
    axes: tuple[int, int],
    title: str,
    path: Path,
    max_points: int = 200_000,
) -> None:
    rng = np.random.default_rng(0)
    cols = rng.uniform(0.15, 0.95, (max(len(details), 1), 3))
    sel = np.arange(len(xyz))
    if len(sel) > max_points:
        sel = rng.choice(sel, max_points, replace=False)
    a, b = axes
    lab = labels[sel]
    c = np.where(
        lab[:, None] >= 0,
        cols[np.maximum(lab, 0)],
        np.where(lab[:, None] == STRUCTURE, [0.75, 0.75, 0.75], [0.45, 0.0, 0.0]),
    )
    order = np.argsort(lab >= 0)  # draw noise and planes first
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.scatter(xyz[sel[order], a], xyz[sel[order], b], c=c[order], s=0.3, linewidths=0)
    for i, d in enumerate(details):
        lo, hi = d["bbox_min"], d["bbox_max"]
        ax.add_patch(
            Rectangle(
                (lo[a], lo[b]),
                hi[a] - lo[a],
                hi[b] - lo[b],
                fill=False,
                ec=cols[i],
                lw=0.6,
                ls="--" if d["hint"] == "fitting/valve" else "-",
            )
        )
    noise = (labels == NOISE).mean()
    ax.set_title(
        f"{title} - {len(details)} objects - noise {noise:.1%} (dark red) - planes grey - "
        "dashed = fitting/valve"
    )
    ax.set_xlabel(AXIS_NAMES[a])
    ax.set_ylabel(AXIS_NAMES[b])
    ax.set_aspect("equal")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[plot] {path}")


# ---------- main ----------


def segment(
    xyz: np.ndarray, opacity: np.ndarray, normals: np.ndarray, sp: SegmentParams
) -> tuple[list[dict], list[dict], np.ndarray]:
    """Voxel-level segmentation. Returns (planes, ordered objects, per-point labels) where
    labels are object indices, NOISE or STRUCTURE."""
    planes = detect_planes(xyz, normals, sp)
    remaining = remove_planes(xyz, normals, planes, sp)
    planes += ransac_extra_planes(xyz, remaining, sp)
    oi = np.flatnonzero(remaining)
    groups = dbscan_clusters(xyz[oi], sp)
    parts = split_oversized(groups, xyz[oi], normals[oi], sp) if groups else []

    lab = np.full(len(xyz), NOISE, np.int32)
    lab[~remaining] = STRUCTURE  # excluded from reassignment
    for k, (p, _, _) in enumerate(parts):
        lab[oi[p]] = k
    lab[oi] = reassign_noise(lab[oi], xyz[oi], sp.noise_reassign_dist)
    objs = [
        {"idx": np.flatnonzero(lab == k), "hint": h, "axis": c, "tags": []}
        for k, (_, h, c) in enumerate(parts)
    ]
    ceiling = next((p["value"] for p in planes if p["name"] == "ceiling"), None)
    objs = filter_overhead(objs, xyz, opacity, ceiling, sp)
    objs = merge_collinear_pipes(objs, xyz, sp)
    objs = merge_contained(objs, xyz, sp.contain_tol)
    objs = order_objects(objs, xyz)

    labels = np.where(lab == STRUCTURE, STRUCTURE, NOISE).astype(np.int32)
    for k, o in enumerate(objs):
        labels[o["idx"]] = k
    return planes, objs, labels


def run(scene_dir: Path, report_dir: Path, sp: SegmentParams = SegmentParams()) -> list[DetectedObject]:
    t0 = time.time()
    scene_dir = Path(scene_dir)
    manifest = load_model(scene_dir / MANIFEST_FILE, SceneManifest)
    if manifest.coordinate_system.up_axis != "+Z":
        raise ValueError(f"segmentation needs a +Z-up world, manifest has {manifest.coordinate_system.up_axis!r}")
    scene = load_scene(scene_dir)
    xyz, rgb, opacity, normals = downsample_and_denoise(scene, sp)
    planes, objs, labels = segment(
        xyz=xyz,
        opacity=opacity,
        normals=normals,
        sp=sp,
    )
    details = [
        object_details(o["idx"], xyz, rgb, opacity, o["hint"], o["tags"], sp) for o in objs
    ]
    gaussian_labels = membership(scene["xyz"], xyz, labels, sp.member_dist)

    members_dir = scene_dir / MEMBERS_DIR
    shutil.rmtree(members_dir, ignore_errors=True)
    members_dir.mkdir(parents=True)
    order = np.argsort(gaussian_labels, kind="stable")
    bounds = np.searchsorted(gaussian_labels[order], np.arange(len(objs) + 1))
    objects = []
    for k, (o, d) in enumerate(zip(objs, details, strict=True)):
        oid = object_id(k)
        members = np.sort(scene["orig_index"][order[bounds[k] : bounds[k + 1]]])
        rel = f"{MEMBERS_DIR}/{oid}.npy"
        np.save(scene_dir / rel, members.astype(np.int64))
        lo, hi = np.asarray(d["bbox_min"]), np.asarray(d["bbox_max"])
        objects.append(
            DetectedObject(
                object_id=oid,
                scene_id=manifest.scene_id,
                center=tuple(np.round(xyz[o["idx"]].mean(0), 4).tolist()),
                bbox=OrientedBox(
                    center=tuple(np.round((lo + hi) / 2, 4).tolist()),
                    extent=tuple(np.round(hi - lo, 4).tolist()),
                ),
                gaussians=GaussianMembership(indices_file=rel, count=len(members)),
            )
        )
    save_list(scene_dir / OBJECTS_FILE, objects)

    plane_records = [
        {
            "name": p["name"],
            "label": p["label"],
            "axis": AXIS_NAMES[p["axis"]] if "axis" in p else None,
            "value": round(p["value"], 4) if "value" in p else None,
            "normal": p.get("normal"),
            "n_points": int(len(p["idx"])),
        }
        for p in planes
    ]
    detail_doc = {
        "scene_id": manifest.scene_id,
        "params": asdict(sp),
        "planes": plane_records,
        "objects": {object_id(k): d for k, d in enumerate(details)},
    }
    (scene_dir / DETAILS_FILE).write_text(json.dumps(detail_doc, indent=1) + "\n", encoding="utf-8")
    np.savez_compressed(
        scene_dir / SEGMENT_POINTS_FILE,
        xyz=xyz.astype(np.float32),
        rgb=rgb.astype(np.float32),
        opacity=opacity.astype(np.float32),
        labels=labels,
        gaussian_labels=gaussian_labels,
    )

    hints: dict[str, int] = {}
    for d in details:
        hints[d["hint"] or "unlabelled"] = hints.get(d["hint"] or "unlabelled", 0) + 1
    noise = float((labels == NOISE).mean())
    preview = report_dir / "preview"
    plot_view(xyz, labels, details, (0, 2), "Front (x vs z)", preview / f"{manifest.scene_id}_segment_front.png")
    plot_view(xyz, labels, details, (0, 1), "Top (x vs y)", preview / f"{manifest.scene_id}_segment_top.png")
    report = {
        "scene_id": manifest.scene_id,
        "n_objects": len(objects),
        "n_planes": len(planes),
        "noise_fraction": round(noise, 4),
        "hints": hints,
        "overhead": sum("overhead" in d["tags"] for d in details),
        "planes": plane_records,
        "seconds": round(time.time() - t0, 1),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"semantic_{manifest.scene_id}.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"\nEXTRACTED OBJECTS: {len(objects)}  (+{len(planes)} structural)  "
        f"noise {noise:.1%}  hints {hints}  [{report['seconds']:.0f}s]"
    )
    return objects


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--scene-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "equipment_room"
    )
    ap.add_argument("--report-dir", type=Path, default=REPO_ROOT / "reports")
    args = ap.parse_args(argv)
    run(
        scene_dir=args.scene_dir,
        report_dir=args.report_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
