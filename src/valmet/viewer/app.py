"""The Viser app: boxes, routes, the walking avatar and the side panels.

Pick an object, press Navigate, and the avatar walks that object's route from
paths.json. Routes are computed by the spatial pipeline; the viewer only plays them.
"""

from __future__ import annotations

import time

import numpy as np
import viser
import viser.transforms as tf

from valmet.integration.live_scene import LiveObject, LiveScene
from valmet.viewer.avatar import Person
from valmet.viewer.panels import format_diagnosis, format_metadata
from valmet.viewer.playback import PathPlayer, find_nearby, find_nearest

BOX_COLOR = (120, 144, 170)
HIGHLIGHT_COLOR = (60, 220, 120)  # boxes close to the avatar
FAULT_COLOR = (230, 40, 40)  # faulty boxes close to the avatar
ROUTE_COLOR = (40, 120, 255)
ROUTE_LIFT = 0.02  # keeps the route line just above the floor grid
FRAME_TIME = 1.0 / 30.0  # seconds between updates (30 per second)


def target_label(obj: LiveObject) -> str:
    return f"FAULT: {obj.name}" if obj.is_faulty else obj.name


class ViewerApp:
    def __init__(self, scene: LiveScene, port: int = 8080) -> None:
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
        self._add_ground()
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
        self._add_gui()

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
        )
        self.server.scene.add_label(
            f"/objects/{obj.object_id}/label",
            target_label(obj),
            position=(x, y, z + bbox.extent[2] / 2 + 0.15),
            anchor="bottom-center",
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
        with gui.add_folder("Nearest object"):
            self.metadata_panel = gui.add_markdown("")
        with gui.add_folder("Diagnosis"):
            self.diagnosis_panel = gui.add_markdown("")

        navigate_button.on_click(self._on_navigate)
        play_button.on_click(self._on_play)
        pause_button.on_click(lambda _: self.player.pause())
        reset_button.on_click(self._on_reset)

    def _summary(self) -> str:
        count = len(self.scene.faulty)
        return f"{count} faulty object{'' if count == 1 else 's'}. Pick one and press Navigate."

    def _on_navigate(self, _event: object) -> None:
        self.pending_target = self.target_dropdown.value

    def _on_play(self, _event: object) -> None:
        self.pending_play = self.target_dropdown.value

    def _on_reset(self, _event: object) -> None:
        self.pending_reset = True

    def navigate(self, label: str) -> None:
        """Start walking the route to the object with this dropdown label."""
        obj = self.targets[label]
        if obj.route is None:
            self.player.pause()
            self.destination = None
            self.status_panel.content = f"No route to {obj.name} in paths.json yet."
            return
        self.player.load(obj.route.waypoints)
        self._draw_route(obj.route.waypoints)
        self.player.play()
        self.route_target = obj
        self.destination = obj
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

    def _update_highlights(self, position: np.ndarray) -> None:
        # Only recolor the boxes whose highlight state changed this frame.
        nearby = find_nearby(position, self.centers, self.radius_slider.value)
        for object_id in nearby - self.highlighted:
            is_faulty = self.objects[object_id].is_faulty
            self.boxes[object_id].color = FAULT_COLOR if is_faulty else HIGHLIGHT_COLOR
        for object_id in self.highlighted - nearby:
            self.boxes[object_id].color = BOX_COLOR
        self.highlighted = nearby

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
