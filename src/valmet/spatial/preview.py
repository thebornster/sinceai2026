"""Top-down PNG of the occupancy grid, optionally with a path and target box."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from valmet.common.contracts import NavigationMapMetadata, NavigationPath, OrientedBox  # noqa: E402

CMAP = ListedColormap(["#f2f2f2", "#3b3b3b", "#c9d6e8"])  # free, occupied, unknown


def render_navmap(
    path_png: Path,
    navmap: NavigationMapMetadata,
    grid: np.ndarray,
    title: str,
    path: NavigationPath | None = None,
    target_box: OrientedBox | None = None,
) -> Path:
    v = navmap.cell_values
    index = np.full(grid.shape, 2, dtype=np.uint8)
    index[grid == v.free] = 0
    index[grid == v.occupied] = 1
    ox, oy = navmap.origin[:2]
    nx, ny = navmap.shape
    extent = (ox, ox + nx * navmap.resolution, oy, oy + ny * navmap.resolution)

    fig, ax = plt.subplots(figsize=(10, 10 * ny / nx + 1))
    ax.imshow(index.T, origin="lower", extent=extent, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
    if target_box is not None:
        corners = target_box.corners()[[0, 1, 3, 2, 0], :2]
        ax.plot(corners[:, 0], corners[:, 1], color="#d62728", linewidth=2, label="target")
    if path is not None:
        wp = np.asarray(path.waypoints)
        ax.plot(wp[:, 0], wp[:, 1], color="#1f77b4", linewidth=2, marker="o", markersize=3, label="path")
        ax.plot(*wp[0, :2], marker="s", color="#2ca02c", markersize=9, label="start")
        ax.plot(*wp[-1, :2], marker="*", color="#d62728", markersize=14, label="inspection point")
        ax.legend(loc="upper right")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path_png
