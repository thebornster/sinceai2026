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
from collections.abc import Callable
from pathlib import Path

import numpy as np
import viser

from valmet.common.artifacts import NAVMAP_FILE, PATHS_FILE, REPO_ROOT, load_list, load_model
from valmet.common.contracts import NavigationMapMetadata, NavigationPath
from valmet.spatial.planner import sideways_segments

PICK_RADIUS = 0.03  # world units around the click ray
PICK_MIN_OPACITY = 0.5
WALK_EYE_HEIGHT = 1.6
WALK_LOOK_AHEAD = 0.5  # meters either side of the camera used for the heading
WALK_LOOK_DOWN = 0.3  # meters the gaze drops over 2 m
WALK_FPS = 30.0


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


def add_navigation(server: viser.ViserServer, scene_dir: Path) -> None:
    """Toggles for the occupancy grid (blocked cells) and the planned route, if they exist.
    Path segments where the avatar squeezes through sideways are orange."""
    navmap = grid = None
    if (scene_dir / NAVMAP_FILE).exists():
        navmap = load_model(scene_dir / NAVMAP_FILE, NavigationMapMetadata)
        grid = np.load(scene_dir / navmap.grid_file)
        blocked = navmap.cell_to_world(np.argwhere(grid == navmap.cell_values.occupied))
        handle = server.scene.add_point_cloud(
            "/navigation/blocked",
            points=np.column_stack([blocked, np.full(len(blocked), 0.02)]),
            colors=(200, 40, 40),
            point_size=navmap.resolution,
            point_shape="square",
            visible=False,
        )
        toggle = server.gui.add_checkbox("Show blocked cells (navmap)", initial_value=False)
        toggle.on_update(lambda _: setattr(handle, "visible", toggle.value))

    if (scene_dir / PATHS_FILE).exists():
        handles = []
        for i, path in enumerate(load_list(scene_dir / PATHS_FILE, NavigationPath)):
            wp = np.asarray(path.waypoints) + [0.0, 0.0, 0.05]
            colors = np.tile(np.array([30, 120, 255], dtype=np.uint8), (len(wp) - 1, 2, 1))
            if navmap is not None:
                sideways = sideways_segments(
                    navmap=navmap,
                    grid=grid,
                    path=path,
                )
                colors[np.array(sideways, dtype=bool)] = (255, 140, 0)
            handles.append(
                server.scene.add_line_segments(
                    f"/navigation/path_{i}",
                    points=np.stack([wp[:-1], wp[1:]], axis=1),
                    colors=colors,
                    thickness=0.04,
                )
            )
            handles.append(
                server.scene.add_icosphere(
                    f"/navigation/start_{i}", radius=0.08, color=(40, 180, 40), position=wp[0]
                )
            )
            handles.append(
                server.scene.add_icosphere(
                    f"/navigation/goal_{i}", radius=0.08, color=(220, 40, 40), position=wp[-1]
                )
            )
        toggle_path = server.gui.add_checkbox("Show path", initial_value=True)
        toggle_path.on_update(lambda _: [setattr(h, "visible", toggle_path.value) for h in handles])


def add_walkthrough(server: viser.ViserServer, scene_dir: Path) -> Callable[[float], None]:
    """Slider and Play button that move the camera along the first path at eye height.
    Returns a tick(dt) function for the main loop to advance playback."""
    if not (scene_dir / PATHS_FILE).exists():
        return lambda _: None
    xy = np.asarray(load_list(scene_dir / PATHS_FILE, NavigationPath)[0].waypoints)[:, :2]
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    length = float(s[-1])

    def at(dist: float) -> np.ndarray:
        dist = float(np.clip(dist, 0.0, length))
        return np.array([np.interp(dist, s, xy[:, 0]), np.interp(dist, s, xy[:, 1])])

    with server.gui.add_folder("Walk along path"):
        slider = server.gui.add_slider(
            "Distance (m)", min=0.0, max=length, step=0.05, initial_value=0.0
        )
        speed = server.gui.add_number("Speed (m/s)", initial_value=0.8, min=0.1, max=5.0, step=0.1)
        eye = server.gui.add_number(
            "Eye height (m)", initial_value=WALK_EYE_HEIGHT, min=0.2, max=3.0, step=0.1
        )
        play = server.gui.add_button("Play / Pause")
    playing = [False]

    def place_camera(_=None) -> None:
        here = at(slider.value)
        heading = at(slider.value + WALK_LOOK_AHEAD) - at(slider.value - WALK_LOOK_AHEAD)
        if np.linalg.norm(heading) < 1e-6:
            heading = xy[-1] - xy[-2]
        heading = heading / np.linalg.norm(heading)
        position = np.array([*here, eye.value])
        look_at = np.array([*(here + heading * 2.0), eye.value - WALK_LOOK_DOWN])
        for client in server.get_clients().values():
            with client.atomic():
                client.camera.up_direction = (0.0, 0.0, 1.0)
                client.camera.position = position
                client.camera.look_at = look_at

    slider.on_update(place_camera)
    eye.on_update(place_camera)

    @play.on_click
    def _(_) -> None:
        if not playing[0] and slider.value >= length:
            slider.value = 0.0
        playing[0] = not playing[0]

    def tick(dt: float) -> None:
        if not playing[0]:
            return
        nxt = min(slider.value + speed.value * dt, length)
        playing[0] = nxt < length
        slider.value = nxt

    return tick


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--scene-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "equipment_room"
    )
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
    add_navigation(server=server, scene_dir=args.scene_dir)
    tick = add_walkthrough(server=server, scene_dir=args.scene_dir)
    show_splats = server.gui.add_checkbox("Show splats", initial_value=True)

    @show_cleaned.on_update
    def _(_) -> None:
        cleaned_cloud.visible = show_cleaned.value

    @show_splats.on_update
    def _(_) -> None:
        splat_handle.visible = show_splats.value

    orbit = server.gui.add_checkbox("Click sets orbit center", initial_value=False)
    measure = server.gui.add_checkbox("Measure (click 2 points)", initial_value=False)
    readout = server.gui.add_text("Distance (world units)", initial_value="-", disabled=True)
    known = server.gui.add_number("Known length (m)", initial_value=2.0, min=0.01, step=0.01)
    result = server.gui.add_text("Scale flag", initial_value="-", disabled=True)
    last_point = server.gui.add_text(
        "Last picked point (x, y, z)", initial_value="-", disabled=True
    )
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
            server.scene.add_icosphere(
                f"/measure/p{i}", radius=0.03, color=(0, 255, 255), position=p
            )
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
        last_point.value = ", ".join(f"{v:.2f}" for v in p)
        if orbit.value:
            event.client.camera.look_at = p
            return
        if len(picked) == 2:
            picked.clear()
            server.scene.remove_by_name("/measure")
        picked.append(p)
        draw()
        refresh()

    # Viser allows one click callback, so the two click modes are exclusive.
    def set_click_mode(active: viser.GuiCheckboxHandle, other: viser.GuiCheckboxHandle) -> None:
        if active.value:
            other.value = False
            server.scene.on_pointer_event(event_type="click")(on_click)
        elif not other.value:
            server.scene.remove_pointer_callback()

    measure.on_update(lambda _: set_click_mode(active=measure, other=orbit))
    orbit.on_update(lambda _: set_click_mode(active=orbit, other=measure))

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
        tick(1.0 / WALK_FPS)
        time.sleep(1.0 / WALK_FPS)


if __name__ == "__main__":
    raise SystemExit(main())
