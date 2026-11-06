"""Open the preprocessed scene in Viser and measure distances to calibrate the metric scale.

Turn on "Measure" in the side panel, click two points on the scene (e.g. floor and the top of
a door frame), enter that object's real length in meters and read off the --scale value.
Distances are in world units of the current manifest (source units while scale is unset).

Usage:
    python scripts/measure_scale.py [--scene-dir data/processed/equipment_room] [--port 8080]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import viser

from valmet.common.artifacts import REPO_ROOT

PICK_RADIUS = 0.03  # world units around the click ray
PICK_MIN_OPACITY = 0.5


def pick_point(centers: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
    """Nearest splat center (along the ray) within PICK_RADIUS of the click ray."""
    d = direction / np.linalg.norm(direction)
    rel = centers - origin
    t = rel @ d
    dist = np.linalg.norm(rel - t[:, None] * d, axis=1)
    hit = np.flatnonzero((t > 0) & (dist < PICK_RADIUS))
    if hit.size == 0:
        return None
    return centers[hit[np.argmin(t[hit])]]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "equipment_room")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args(argv)

    splats = np.load(args.scene_dir / "viewer_splats.npz")
    centers = splats["centers"].astype(np.float64)
    pickable = centers[splats["opacities"][:, 0] >= PICK_MIN_OPACITY]

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    splat_handle = server.scene.add_gaussian_splats(
        "/scene",
        centers=splats["centers"],
        covariances=splats["covariances"],
        rgbs=splats["rgbs"],
        opacities=splats["opacities"],
    )
    server.scene.add_grid("/floor_grid", width=12.0, height=10.0, cell_size=0.5, section_size=1.0)

    clean = np.load(args.scene_dir / "scene_clean.npz")
    cleaned_cloud = server.scene.add_point_cloud(
        "/cleaned",
        points=clean["positions"],
        colors=(clean["rgbs"] * 255).astype(np.uint8),
        point_size=0.01,
        visible=False,
    )
    show_cleaned = server.gui.add_checkbox("Show cleaned geometry (points)", initial_value=False)
    show_splats = server.gui.add_checkbox("Show splats", initial_value=True)

    @show_cleaned.on_update
    def _(_) -> None:
        cleaned_cloud.visible = show_cleaned.value

    @show_splats.on_update
    def _(_) -> None:
        splat_handle.visible = show_splats.value

    measure = server.gui.add_checkbox("Measure (click 2 points)", initial_value=False)
    readout = server.gui.add_text("Distance (world units)", initial_value="-", disabled=True)
    known = server.gui.add_number("Known length (m)", initial_value=2.0, min=0.01, step=0.01)
    result = server.gui.add_text("Scale flag", initial_value="-", disabled=True)
    clear = server.gui.add_button("Clear")
    picked: list[np.ndarray] = []

    def refresh() -> None:
        if len(picked) < 2:
            readout.value = f"{len(picked)}/2 points picked"
            result.value = "-"
            return
        dist = float(np.linalg.norm(picked[1] - picked[0]))
        readout.value = f"{dist:.4f}"
        result.value = f"--scale {known.value / dist:.4f}"

    def draw() -> None:
        for i, p in enumerate(picked):
            server.scene.add_icosphere(f"/measure/p{i}", radius=0.03, color=(0, 255, 255), position=p)
        if len(picked) == 2:
            server.scene.add_line_segments(
                "/measure/line",
                points=np.array([picked]),
                colors=(0, 255, 255),
                thickness=0.01,
            )

    def on_click(event: viser.ScenePointerEvent) -> None:
        p = pick_point(
            centers=pickable,
            origin=np.asarray(event.ray_origin),
            direction=np.asarray(event.ray_direction),
        )
        if p is None:
            return
        if len(picked) == 2:
            picked.clear()
            server.scene.remove_by_name("/measure")
        picked.append(p)
        draw()
        refresh()

    @measure.on_update
    def _(_) -> None:
        if measure.value:
            server.scene.on_pointer_event(event_type="click")(on_click)
            return
        server.scene.remove_pointer_callback()

    @known.on_update
    def _(_) -> None:
        refresh()

    @clear.on_click
    def _(_) -> None:
        picked.clear()
        server.scene.remove_by_name("/measure")
        refresh()

    print(f"Viser running at http://localhost:{args.port} (Ctrl+C to stop)")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
