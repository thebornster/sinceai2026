"""Write preprocessing outputs: manifest, cleaned world-frame Gaussians, viewer splats, previews."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from valmet.common.artifacts import MANIFEST_FILE, save_model  # noqa: E402
from valmet.common.contracts import CoordinateSystem, SceneManifest  # noqa: E402
from valmet.preprocessing.gaussians import GaussianScene  # noqa: E402

SCENE_CLEAN_FILE = "scene_clean.npz"
VIEWER_SPLATS_FILE = "viewer_splats.npz"


def build_manifest(
    scene_id: str,
    source: Path,
    source_sha256: str | None,
    source_vertex_count: int,
    source_to_world: np.ndarray,
    scale_source: str,
) -> SceneManifest:
    metric = scale_source != "unset"
    return SceneManifest(
        scene_id=scene_id,
        source_filename=Path(source).name,
        source_sha256=source_sha256,
        source_vertex_count=source_vertex_count,
        coordinate_system=CoordinateSystem(
            up_axis="+Z",
            units="meters" if metric else "unknown",
            meters_per_unit=1.0 if metric else None,
            verified=metric,
            notes=(
                f"Floor at z=0, walls aligned to X/Y, XY centered on the room. Scale: {scale_source}."
            ),
        ),
        source_to_world=np.asarray(source_to_world).tolist(),
    )


def write_outputs(
    out_dir: Path,
    manifest: SceneManifest,
    world: GaussianScene,
    viewer: GaussianScene,
    viewer_max_splats: int | None,
) -> dict[str, Path]:
    """world: cleaned Gaussians for geometry. viewer: uncleaned Gaussians for rendering, since
    faint splats are what fill in surfaces visually."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_model(out_dir / MANIFEST_FILE, manifest)

    np.savez(
        out_dir / SCENE_CLEAN_FILE,
        positions=world.positions.astype(np.float32),
        quats_wxyz=world.quats_wxyz.astype(np.float32),
        scales=world.scales.astype(np.float32),
        opacities=world.opacities.astype(np.float32),
        rgbs=world.rgbs.astype(np.float32),
        orig_index=world.orig_index.astype(np.int32),
    )

    splats = viewer
    if viewer_max_splats is not None and len(viewer) > viewer_max_splats:
        splats = viewer.subset(np.sort(np.argsort(-viewer.opacities)[:viewer_max_splats]))
    # Shapes match viser's SceneApi.add_gaussian_splats(centers, covariances, rgbs, opacities).
    np.savez(
        out_dir / VIEWER_SPLATS_FILE,
        centers=splats.positions.astype(np.float32),
        covariances=splats.covariances().astype(np.float32),
        rgbs=splats.rgbs.astype(np.float32),
        opacities=splats.opacities[:, None].astype(np.float32),
    )
    return {
        "manifest": out_dir / MANIFEST_FILE,
        "scene_clean": out_dir / SCENE_CLEAN_FILE,
        "viewer_splats": out_dir / VIEWER_SPLATS_FILE,
    }


def write_previews(world: GaussianScene, out_dir: Path, scene_id: str) -> list[Path]:
    """Opacity-weighted density images: top-down XY plus XZ and YZ side views."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p, w = world.positions, world.opacities
    lo, hi = np.percentile(p, [0.5, 99.5], axis=0)
    views = {
        "topdown": (0, 1, "x", "y"),
        "side_xz": (0, 2, "x", "z"),
        "side_yz": (1, 2, "y", "z"),
    }
    written = []
    for name, (i, j, xi, xj) in views.items():
        fig, ax = plt.subplots(figsize=(10, 10 * (hi[j] - lo[j]) / (hi[i] - lo[i]) + 1))
        hist, ei, ej = np.histogram2d(
            p[:, i], p[:, j], bins=(400, max(50, int(400 * (hi[j] - lo[j]) / (hi[i] - lo[i])))),
            range=[[lo[i], hi[i]], [lo[j], hi[j]]], weights=w,
        )  # fmt: skip
        ax.imshow(
            np.log1p(hist.T), origin="lower", extent=(ei[0], ei[-1], ej[0], ej[-1]), cmap="magma"
        )
        if j == 2:
            ax.axhline(0.0, color="cyan", linewidth=0.8, linestyle="--", label="z = 0 (floor)")
            ax.legend(loc="upper right")
        ax.set_xlabel(xi)
        ax.set_ylabel(xj)
        ax.set_title(f"{scene_id}: {name} (opacity-weighted density, world frame)")
        path = out_dir / f"{scene_id}_{name}.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written
