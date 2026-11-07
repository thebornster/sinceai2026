"""The Viser app: boxes, routes, the walking avatar and the side panels.

Pick an object, press Navigate, and the avatar walks that object's route from
paths.json. Objects without one get a route planned on the navmap from where the avatar
stands.
Click a box to inspect it, fly the camera to it, or ask the LLM copilot for a diagnosis.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import imageio.v3 as iio
import numpy as np
import viser
import viser.transforms as tf

from valmet import api
from valmet.common.contracts import NavigationPath
from valmet.integration.live_scene import LiveObject, LiveScene
from valmet.spatial.planner import UnreachableTargetError, plan_path
from valmet.viewer.avatar import Person
from valmet.viewer.panels import format_diagnosis, format_metadata
from valmet.viewer.playback import (
    PathPlayer,
    find_nearby,
    find_nearest,
    pick_along_ray,
    pick_object,
)

BOX_COLOR = (120, 144, 170)
HIGHLIGHT_COLOR = (60, 220, 120)  # boxes close to the avatar
FAULT_COLOR = (230, 40, 40)  # faulty boxes close to the avatar
SELECTED_COLOR = (255, 0, 255)  # the clicked box
ROUTE_COLOR = (40, 120, 255)
ROUTE_LIFT = 0.02  # keeps the route line just above the floor grid
FRAME_TIME = 1.0 / 30.0  # seconds between updates (30 per second)
SELECTED_FILL_OPACITY = 0.35
SAME_SPOT_COS = np.cos(np.radians(0.5))  # a click this close to the last one goes deeper
HIT_MARKER_RADIUS = 0.02
SHOW_ALL = "All objects"
SHOW_FAULTY = "Faulty only"
FLY_MIN_DISTANCE = 1.2  # meters from the camera to a small object after "Fly to"
FLY_EXTENT_FACTOR = 2.0  # camera distance per meter of box diagonal


def target_label(obj: LiveObject) -> str:
    return f"FAULT: {obj.name}" if obj.is_faulty else obj.name


class ViewerApp:
    def __init__(self, scene: LiveScene, port: int = 8080, offline: bool = False) -> None:
        self.scene = scene
        self.objects = {obj.object_id: obj for obj in scene.objects}
        self.centers = {
            obj.object_id: np.asarray(obj.detected.center, dtype=float) for obj in scene.objects
        }
        # Faulty objects first, so the default choice is the one that needs a visit.
        ordered = sorted(scene.objects, key=lambda obj: not obj.is_faulty)
        self.targets = {target_label(obj): obj for obj in ordered}

        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("+z")
        self.offline = offline
        self._add_ground()
        self._add_splats()
        self.labels: dict[str, viser.LabelHandle] = {}
        self.boxes = {obj.object_id: self._add_object(obj) for obj in scene.objects}

        # The object at the end of the loaded route.
        self.route_target = next((obj for obj in ordered if obj.route), None)
        waypoints = self.route_target.route.waypoints if self.route_target else [(0.0, 0.0, 0.0)]
        self.player = PathPlayer(waypoints)
        self.person = Person(self.server, self.player.position(), self.player.heading())
        self._draw_route(waypoints)

        self.destination: LiveObject | None = None  # the object being walked to
        # Set by the GUI thread, used by the frame loop.
        self.pending_target: str | None = None
        self.pending_play: str | None = None
        self.pending_reset = False
        self.highlighted: set[str] = set()
        self.shown_id: str | None = None  # the object the panels are showing
        self.selected: LiveObject | None = None  # the clicked object
        self.last_click: tuple[np.ndarray, np.ndarray] | None = None  # ray origin, direction
        self.asking = False  # an LLM diagnosis is running
        self._add_gui()
        self.server.scene.on_click()(self._on_scene_click)

    def _add_ground(self) -> None:
        navmap = self.scene.navmap
        if navmap:
            size = np.asarray(navmap.shape) * navmap.resolution
            middle = np.asarray(navmap.origin[:2]) + size / 2
            floor = navmap.origin[2]
        else:
            points = np.array(list(self.centers.values()))[:, :2]
            size = np.ptp(points, axis=0) + 4.0
            middle = (points.min(axis=0) + points.max(axis=0)) / 2
            floor = 0.0
        self.server.scene.add_grid(
            "/ground",
            width=float(size[0]),
            height=float(size[1]),
            cell_size=1.0,
            position=(float(middle[0]), float(middle[1]), floor),
        )

    def _add_splats(self) -> None:
        splats = self.scene.splats
        if splats is None:
            return
        self.server.scene.add_gaussian_splats(
            "/splats",
            centers=splats["centers"],
            covariances=splats["covariances"],
            rgbs=splats["rgbs"],
            opacities=splats["opacities"],
        )

    def _add_object(self, obj: LiveObject) -> viser.BoxHandle:
        """Draw one object's box and label. Returns the box, to recolor it later."""
        bbox = obj.detected.bbox
        x, y, z = bbox.center
        box = self.server.scene.add_box(
            f"/objects/{obj.object_id}/box",
            color=BOX_COLOR,
            dimensions=bbox.extent,
            wxyz=tf.SO3.from_matrix(np.asarray(bbox.rotation)).wxyz,
            position=bbox.center,
            wireframe=self.scene.splats is not None,  # solid boxes would hide the splats
        )
        self.labels[obj.object_id] = self.server.scene.add_label(
            f"/objects/{obj.object_id}/label",
            target_label(obj),
            position=(x, y, z + bbox.extent[2] / 2 + 0.15),
            anchor="bottom-center",
            visible=False,  # only the selected object's label, unless "Show labels" is on
        )
        return box

    def _draw_route(self, waypoints: list) -> None:
        if len(waypoints) < 2:
            return
        points = np.asarray(waypoints, dtype=np.float32) + (0.0, 0.0, ROUTE_LIFT)
        self.server.scene.add_line_segments(
            "/route",
            points=np.stack([points[:-1], points[1:]], axis=1),
            colors=ROUTE_COLOR,
            thickness=3.0,
            thickness_units="screen",
        )

    def _add_gui(self) -> None:
        gui = self.server.gui
        with gui.add_folder("Navigate"):
            self.target_dropdown = gui.add_dropdown("Object", tuple(self.targets))
            navigate_button = gui.add_button("Navigate to object")
            self.status_panel = gui.add_markdown(self._summary())
        with gui.add_folder("Display"):
            self.show_dropdown = gui.add_dropdown("Show", (SHOW_ALL, SHOW_FAULTY))
            self.labels_checkbox = gui.add_checkbox("Show labels", initial_value=False)
            self.isolate_checkbox = gui.add_checkbox("Only show selected", initial_value=False)
        with gui.add_folder("Playback"):
            play_button = gui.add_button("Play")
            pause_button = gui.add_button("Pause")
            reset_button = gui.add_button("Reset")
            self.speed_slider = gui.add_slider(
                "Speed (m/s)", min=0.2, max=5.0, step=0.1, initial_value=1.0
            )
            self.radius_slider = gui.add_slider(
                "Highlight radius (m)", min=0.5, max=5.0, step=0.1, initial_value=1.5
            )
        with gui.add_folder("Selected object"):
            self.selection_panel = gui.add_markdown("Click a box in the scene to inspect it.")
            self.crop_image = gui.add_image(np.full((8, 24, 3), 235, np.uint8), visible=False)
            fly_button = gui.add_button("Fly to object", disabled=True)
            self.ask_button = gui.add_button("Ask copilot", disabled=True)
            self.copilot_panel = gui.add_markdown("")
        with gui.add_folder("Nearest object"):
            self.metadata_panel = gui.add_markdown("")
        with gui.add_folder("Diagnosis"):
            self.diagnosis_panel = gui.add_markdown("")

        navigate_button.on_click(self._on_navigate)
        play_button.on_click(self._on_play)
        pause_button.on_click(lambda _: self.player.pause())
        reset_button.on_click(self._on_reset)
        self.fly_button = fly_button
        self.show_dropdown.on_update(lambda _: self.apply_visibility())
        self.labels_checkbox.on_update(lambda _: self.apply_visibility())
        self.isolate_checkbox.on_update(lambda _: self.apply_visibility())
        fly_button.on_click(self._on_fly)
        self.ask_button.on_click(self._on_ask)

    def _summary(self) -> str:
        count = len(self.scene.faulty)
        return f"{count} faulty object{'' if count == 1 else 's'}. Pick one and press Navigate."

    def _on_navigate(self, _event: object) -> None:
        self.pending_target = self.target_dropdown.value

    def _on_play(self, _event: object) -> None:
        self.pending_play = self.target_dropdown.value

    def _on_reset(self, _event: object) -> None:
        self.pending_reset = True

    def is_shown(self, obj: LiveObject) -> bool:
        """Faulty-only mode still shows the selected object and the one being walked to."""
        if self.isolate_checkbox.value and self.selected is not None:
            return obj is self.selected
        if self.show_dropdown.value == SHOW_ALL or obj.is_faulty:
            return True
        return obj in (self.selected, self.destination, self.route_target)

    def apply_visibility(self) -> None:
        for object_id, obj in self.objects.items():
            shown = self.is_shown(obj)
            self.boxes[object_id].visible = shown
            is_selected = obj is self.selected
            self.labels[object_id].visible = shown and (self.labels_checkbox.value or is_selected)

    def _on_scene_click(self, event: viser.SceneClickEvent) -> None:
        if event.ray_origin is None or event.ray_direction is None:
            return
        self.click(np.asarray(event.ray_origin), np.asarray(event.ray_direction))

    def click(self, origin: np.ndarray, direction: np.ndarray) -> None:
        """Select the object under the click; clicking the same spot again selects the next
        object behind it."""
        origin = np.asarray(origin, dtype=float)
        direction = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
        previous, self.last_click = self.last_click, (origin, direction)
        hits, point = self.hits(origin, direction)
        if not hits:
            return
        object_id = hits[0]
        same_spot = (
            previous is not None
            and np.allclose(previous[0], origin, atol=1e-3)
            and float(previous[1] @ direction) > SAME_SPOT_COS
        )
        if same_spot and self.selected is not None and self.selected.object_id in hits:
            object_id = hits[(hits.index(self.selected.object_id) + 1) % len(hits)]
        self.select(object_id)
        if point is not None:
            self.server.scene.add_icosphere(
                "/selection_hit",
                radius=HIT_MARKER_RADIUS,
                color=SELECTED_COLOR,
                position=tuple(float(v) for v in point),
            )
        if len(hits) > 1:
            self.selection_panel.content += (
                "\n\n*Click again on the same spot to select the object behind.*"
            )

    def pick(self, origin: np.ndarray, direction: np.ndarray) -> str | None:
        """The visible object under the click, or None."""
        hits, _ = self.hits(origin, direction)
        return hits[0] if hits else None

    def hits(
        self, origin: np.ndarray, direction: np.ndarray
    ) -> tuple[list[str], np.ndarray | None]:
        """Visible objects along the click ray, front to back, and where the first is hit.

        Uses the segmented Gaussians when the scene has them (walls block the ray, a big box
        in front can't steal the click), else the first box entered."""
        scene = self.scene
        visible = np.array([self.boxes[obj.object_id].visible for obj in scene.objects])
        if scene.pick_points is not None:
            ray = pick_along_ray(
                origin=origin,
                direction=direction,
                points=scene.pick_points,
                labels=scene.pick_labels,
                transparent=~visible,
            )
            return [scene.objects[k].object_id for k in ray.objects], ray.point
        boxes = {
            obj.object_id: obj.detected.bbox
            for obj, shown in zip(scene.objects, visible, strict=True)
            if shown
        }
        object_id = pick_object(origin, direction, boxes)
        return ([object_id] if object_id is not None else []), None

    def select(self, object_id: str) -> None:
        """Show one object in the selection panel and color its box."""
        previous = self.selected
        self.selected = self.objects[object_id]
        if previous is not None:
            self.boxes[previous.object_id].color = self._box_color(previous.object_id)
        self.boxes[object_id].color = SELECTED_COLOR
        bbox = self.selected.detected.bbox
        self.server.scene.add_box(
            "/selection",
            color=SELECTED_COLOR,
            dimensions=bbox.extent,
            wxyz=tf.SO3.from_matrix(np.asarray(bbox.rotation)).wxyz,
            position=bbox.center,
            opacity=SELECTED_FILL_OPACITY,
        )
        self.selection_panel.content = format_metadata(self.selected)
        crop = self.selected.crop
        self.crop_image.visible = crop is not None
        if crop is not None:
            self.crop_image.image = iio.imread(crop)[..., :3]
        self.fly_button.disabled = False
        self.apply_visibility()
        self.ask_button.disabled = self.asking or self.selected.asset is None
        if not self.asking:
            self.copilot_panel.content = ""

    def _on_fly(self, event: viser.GuiEvent) -> None:
        if self.selected is None or event.client is None:
            return
        bbox = self.selected.detected.bbox
        center = np.asarray(bbox.center)
        distance = max(FLY_MIN_DISTANCE, FLY_EXTENT_FACTOR * float(np.linalg.norm(bbox.extent)))
        camera = event.client.camera
        # Keep the current viewing direction, just move closer.
        away = np.asarray(camera.position) - center
        away = away / np.linalg.norm(away) if np.linalg.norm(away) > 0 else np.array([0, -1, 0.5])
        camera.position = tuple(center + away * distance)
        camera.look_at = tuple(center)

    def _on_ask(self, _event: object) -> None:
        if self.selected is None or self.selected.asset is None or self.asking:
            return
        self.ask_copilot(self.selected)

    def ask_copilot(self, obj: LiveObject) -> threading.Thread:
        """Run the LLM diagnosis in the background, showing elapsed seconds until it returns."""
        self.asking = True
        self.ask_button.disabled = True
        started = time.time()

        def progress() -> None:
            while self.asking:
                elapsed = time.time() - started
                self.copilot_panel.content = (
                    f"Asking the copilot about {obj.name}: evidence, model, citation check "
                    f"({elapsed:.0f} s)"
                )
                time.sleep(0.5)

        def work() -> None:
            try:
                diagnosis = api.diagnose_fault(obj.object_id, use_llm=True, offline=self.offline)
                text = format_diagnosis(replace(obj, diagnosis=diagnosis))
                method = diagnosis.get("method", "")
                if method:
                    text += f"\n\n_method: {method}_"
            except Exception as exc:  # noqa: BLE001 - the panel shows any failure
                text = f"Copilot failed: {exc}"
            finally:
                self.asking = False
            self.copilot_panel.content = text
            self.ask_button.disabled = self.selected is None or self.selected.asset is None

        threading.Thread(target=progress, daemon=True).start()
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        return worker

    def _plan_route(self, obj: LiveObject) -> NavigationPath | str:
        """Route from the avatar's position on the navmap, or why there is none."""
        scene = self.scene
        if scene.navmap is None or scene.grid is None:
            return f"No route to {obj.name} in paths.json and no navmap to plan one."
        try:
            return plan_path(
                navmap=scene.navmap,
                grid=scene.grid,
                target_box=obj.detected.bbox,
                target_object_id=obj.object_id,
                start_xy=tuple(self.player.position()[:2]),
                obstacles=scene.obstacles,
            )
        except UnreachableTargetError as exc:
            return f"Can't reach {obj.name}: {exc}"

    def navigate(self, label: str) -> None:
        """Start walking the route to the object with this dropdown label."""
        obj = self.targets[label]
        route = obj.route or self._plan_route(obj)
        if isinstance(route, str):
            self.player.pause()
            self.destination = None
            self.status_panel.content = route
            return
        self.player.load(route.waypoints)
        self._draw_route(route.waypoints)
        self.player.play()
        self.route_target = obj
        self.destination = obj
        self.apply_visibility()
        self.status_panel.content = f"Walking to {obj.name}."

    def play(self, label: str) -> None:
        """Continue the walk to the object with this label, or start one if it's a new target."""
        obj = self.targets[label]
        if obj is not self.route_target or self.player.finished:
            self.navigate(label)
            return
        self.player.play()
        self.destination = obj
        self.status_panel.content = f"Walking to {obj.name}."

    def reset(self) -> None:
        """Go back to the start of the loaded route, ready to walk it again."""
        self.player.reset()
        self.destination = self.route_target
        if self.route_target is None:
            self.status_panel.content = self._summary()
        else:
            self.status_panel.content = (
                f"Back at the start. Press Play to walk to {self.route_target.name}."
            )

    def _box_color(self, object_id: str) -> tuple[int, int, int]:
        if self.selected is not None and self.selected.object_id == object_id:
            return SELECTED_COLOR
        if object_id in self.highlighted:
            return FAULT_COLOR if self.objects[object_id].is_faulty else HIGHLIGHT_COLOR
        return BOX_COLOR

    def _update_highlights(self, position: np.ndarray) -> None:
        # Only recolor the boxes whose highlight state changed this frame.
        nearby = find_nearby(position, self.centers, self.radius_slider.value)
        changed = nearby ^ self.highlighted
        self.highlighted = nearby
        for object_id in changed:
            self.boxes[object_id].color = self._box_color(object_id)

    def _update_panels(self, position: np.ndarray) -> None:
        # Only rewrite the panels when a different object becomes the nearest.
        nearest_id = find_nearest(position, self.centers)
        if nearest_id == self.shown_id:
            return
        self.shown_id = nearest_id
        self.metadata_panel.content = format_metadata(self.objects[nearest_id])
        self.diagnosis_panel.content = format_diagnosis(self.objects[nearest_id])

    def tick(self, dt: float) -> None:
        """Advance the simulation by one frame."""
        if self.pending_target is not None:
            self.navigate(self.pending_target)
            self.pending_target = None
        if self.pending_play is not None:
            self.play(self.pending_play)
            self.pending_play = None
        if self.pending_reset:
            self.reset()
            self.pending_reset = False

        walked = self.player.update(dt, self.speed_slider.value)
        position = self.player.position()
        self.person.move_to(position, self.player.heading(), walked, dt)
        self._update_highlights(position)
        self._update_panels(position)

        if self.destination is not None and self.player.finished:
            self.status_panel.content = f"Arrived at {self.destination.name}."
            self.destination = None

    def run(self) -> None:
        print(f"Viewer running at http://localhost:{self.server.get_port()} (Ctrl+C to stop)")
        last_time = time.time()
        while True:
            now = time.time()
            self.tick(now - last_time)
            last_time = now
            time.sleep(FRAME_TIME)
