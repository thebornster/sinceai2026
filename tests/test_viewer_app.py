import socket
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("viser")

from valmet.integration.live_scene import load_live_scene  # noqa: E402
from valmet.metadata.mapping import METADATA_DIR_ENV  # noqa: E402
from valmet.viewer.app import (  # noqa: E402
    BOX_COLOR,
    FAULT_COLOR,
    HIGHLIGHT_COLOR,
    SELECTED_COLOR,
    ViewerApp,
    target_label,
)
from valmet.viewer.avatar import STRIDE_LENGTH  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv(METADATA_DIR_ENV, raising=False)
    viewer = ViewerApp(load_live_scene(offline=True), port=free_port(), offline=True)
    yield viewer
    viewer.server.stop()


def _without_valve_route(scene):
    objects = [
        replace(obj, route=None) if obj.object_id == "obj_0002" else obj for obj in scene.objects
    ]
    return replace(scene, objects=objects)


@pytest.fixture
def app_without_valve_route(monkeypatch):
    """No paths.json route to the valve and no navmap to plan one."""
    monkeypatch.delenv(METADATA_DIR_ENV, raising=False)
    scene = replace(_without_valve_route(load_live_scene(offline=True)), navmap=None, grid=None)
    viewer = ViewerApp(scene, port=free_port())
    yield viewer
    viewer.server.stop()


@pytest.fixture
def app_planning_valve_route(monkeypatch):
    """No paths.json route to the valve, but a navmap to plan one."""
    monkeypatch.delenv(METADATA_DIR_ENV, raising=False)
    viewer = ViewerApp(_without_valve_route(load_live_scene(offline=True)), port=free_port())
    yield viewer
    viewer.server.stop()


def color(box) -> tuple:
    return tuple(int(c) for c in box.color)


def run_until_stopped(app: ViewerApp) -> None:
    for _ in range(10_000):
        app.tick(0.1)
        if not app.player.playing:
            return
    raise AssertionError("never stopped")


def label_of(app: ViewerApp, object_id: str) -> str:
    return target_label(app.objects[object_id])


def test_faulty_object_is_the_default_target(app):
    assert app.target_dropdown.value == "FAULT: P-002 pump (obj_0001)"
    assert "1 faulty object." in app.status_panel.content


def test_rotated_box_keeps_its_rotation(app):
    # obj_0002 is yawed 30 degrees in the mock scene.
    np.testing.assert_allclose(
        app.boxes["obj_0002"].wxyz,
        (np.cos(np.radians(15)), 0.0, 0.0, np.sin(np.radians(15))),
        atol=1e-6,
    )


def test_navigate_walks_the_route_and_reports_arrival(app):
    app.pending_target = label_of(app, "obj_0001")
    app.tick(0.1)
    assert app.player.playing
    assert app.status_panel.content == "Walking to P-002 pump (obj_0001)."

    run_until_stopped(app)
    route = app.objects["obj_0001"].route
    np.testing.assert_allclose(app.player.position(), route.inspection_position)
    assert app.status_panel.content == "Arrived at P-002 pump (obj_0001)."


def test_arriving_at_the_faulty_object_shows_it_red_with_its_diagnosis(app):
    app.pending_target = label_of(app, "obj_0001")
    app.tick(0.1)
    run_until_stopped(app)
    app.tick(0.1)
    assert color(app.boxes["obj_0001"]) == FAULT_COLOR
    assert "P-002 (ASSET-014)" in app.metadata_panel.content
    assert "CAVITATION_RISK" in app.diagnosis_panel.content


def test_target_without_a_route_says_so_and_does_not_move(app_without_valve_route):
    app = app_without_valve_route
    before = app.player.position()
    app.pending_target = label_of(app, "obj_0002")
    app.tick(0.1)
    assert not app.player.playing
    assert "No route to CV-002 valve (obj_0002)" in app.status_panel.content
    np.testing.assert_allclose(app.player.position(), before)


def test_highlight_follows_the_radius_slider(app):
    app.radius_slider.value = 5.0
    app.tick(0.1)
    assert color(app.boxes["obj_0001"]) == FAULT_COLOR
    assert color(app.boxes["obj_0002"]) == HIGHLIGHT_COLOR
    app.radius_slider.value = 0.5
    app.tick(0.1)
    assert color(app.boxes["obj_0001"]) == BOX_COLOR
    assert color(app.boxes["obj_0002"]) == BOX_COLOR


def walk_frames(app: ViewerApp, frames: int) -> None:
    app.pending_target = label_of(app, "obj_0001")
    for _ in range(frames):
        app.tick(1 / 30)


def test_picking_a_target_without_a_route_stops_a_walk_in_progress(app_without_valve_route):
    app = app_without_valve_route
    walk_frames(app, 10)
    app.pending_target = label_of(app, "obj_0002")
    app.tick(1 / 30)
    stopped_at = app.player.position()
    assert not app.player.playing
    for _ in range(10):
        app.tick(1 / 30)
    np.testing.assert_allclose(app.player.position(), stopped_at)


def test_reset_near_the_start_is_a_jump_not_a_step(app):
    walk_frames(app, 10)  # about 0.33 m along the route
    phase = app.person.phase
    app.pending_reset = True
    app.tick(1 / 30)
    assert app.person.phase == phase
    assert app.person.heading == pytest.approx(app.player.heading())


def test_reset_faces_the_first_leg_and_updates_the_status(app):
    app.pending_target = label_of(app, "obj_0001")
    app.tick(0.1)
    run_until_stopped(app)
    app.pending_reset = True
    app.tick(0.1)
    first_leg = app.player.waypoints[1] - app.player.waypoints[0]
    assert app.person.heading == pytest.approx(np.arctan2(first_leg[1], first_leg[0]))
    assert app.status_panel.content == (
        "Back at the start. Press Play to walk to P-002 pump (obj_0001)."
    )


def test_walking_again_after_reset_reports_arrival(app):
    walk_frames(app, 10)
    app.pending_reset = True
    app.tick(0.1)
    app.pending_play = label_of(app, "obj_0001")
    app.tick(0.1)
    run_until_stopped(app)
    assert app.status_panel.content == "Arrived at P-002 pump (obj_0001)."


def test_a_slow_frame_at_top_speed_keeps_the_walk_going(app):
    app.speed_slider.value = 5.0
    app.player.load([(0.0, 0.0, 0.0), (40.0, 0.0, 0.0)])
    app.player.play()
    for _ in range(30):
        app.tick(1 / 30)
    phase = app.person.phase
    app.tick(0.15)  # 0.75 m in one frame
    assert app.person.swing > 0.9
    assert app.person.phase - phase == pytest.approx(2 * np.pi * 0.75 / STRIDE_LENGTH)


def test_walking_cycle_follows_the_distance_walked(app):
    app.pending_target = label_of(app, "obj_0001")
    app.tick(0.1)
    run_until_stopped(app)
    assert app.person.phase == pytest.approx(2 * np.pi * app.player.total_length / STRIDE_LENGTH)


@pytest.mark.parametrize("object_id", ["obj_0001", "obj_0002", "obj_0003"])
def test_navigate_reaches_the_selected_object(app, object_id):
    app.pending_target = label_of(app, object_id)
    app.tick(0.1)
    run_until_stopped(app)
    obj = app.objects[object_id]
    np.testing.assert_allclose(app.player.position(), obj.route.inspection_position)
    assert app.status_panel.content == f"Arrived at {obj.name}."


def test_play_walks_to_the_selected_object_not_the_loaded_route(app):
    app.pending_play = label_of(app, "obj_0002")
    app.tick(0.1)
    run_until_stopped(app)
    route = app.objects["obj_0002"].route
    np.testing.assert_allclose(app.player.position(), route.inspection_position)


def test_play_continues_a_paused_walk_to_the_same_object(app):
    walk_frames(app, 10)
    app.player.pause()
    app.tick(1 / 30)
    paused_at = app.player.distance
    app.pending_play = label_of(app, "obj_0001")
    app.tick(1 / 30)
    assert app.player.distance > paused_at
    assert app.status_panel.content == "Walking to P-002 pump (obj_0001)."


def test_play_on_a_target_without_a_route_does_not_walk_the_old_route(app_without_valve_route):
    app = app_without_valve_route
    app.pending_play = label_of(app, "obj_0002")
    app.tick(0.1)
    assert not app.player.playing
    assert "No route to CV-002 valve (obj_0002)" in app.status_panel.content


def test_selecting_an_object_shows_it_and_keeps_its_color_while_walking(app):
    app.select("obj_0002")
    assert color(app.boxes["obj_0002"]) == SELECTED_COLOR
    assert "obj_0002" in app.selection_panel.content
    assert not app.crop_image.visible
    assert not app.ask_button.disabled

    app.radius_slider.value = 5.0
    app.tick(0.1)
    assert color(app.boxes["obj_0002"]) == SELECTED_COLOR
    app.select("obj_0001")
    assert color(app.boxes["obj_0002"]) == HIGHLIGHT_COLOR


def test_ask_copilot_fills_the_panel_without_blocking(app):
    app.select("obj_0001")
    worker = app.ask_copilot(app.selected)
    worker.join(timeout=30)
    assert not app.asking
    assert "CAVITATION_RISK" in app.copilot_panel.content
    assert "method:" in app.copilot_panel.content


def test_target_without_a_route_gets_one_planned_from_the_avatar(app_planning_valve_route):
    app = app_planning_valve_route
    start = app.player.position()
    app.pending_target = label_of(app, "obj_0002")
    app.tick(0.1)
    assert app.player.playing
    assert app.status_panel.content == "Walking to CV-002 valve (obj_0002)."
    np.testing.assert_allclose(app.player.waypoints[0][:2], start[:2], atol=0.2)
    run_until_stopped(app)
    assert app.status_panel.content == "Arrived at CV-002 valve (obj_0002)."


def test_labels_are_off_except_for_the_selection(app):
    assert not any(label.visible for label in app.labels.values())
    app.select("obj_0002")
    assert [i for i, label in app.labels.items() if label.visible] == ["obj_0002"]


def test_faulty_only_hides_healthy_objects_but_keeps_selection_and_destination(app):
    app.labels_checkbox.value = True
    app.show_dropdown.value = "Faulty only"
    app.apply_visibility()
    assert app.boxes["obj_0001"].visible and app.labels["obj_0001"].visible  # faulty
    assert not app.boxes["obj_0003"].visible and not app.labels["obj_0003"].visible

    app.select("obj_0003")
    assert app.boxes["obj_0003"].visible

    app.labels_checkbox.value = False
    app.apply_visibility()
    assert app.boxes["obj_0001"].visible and not app.labels["obj_0001"].visible

    app.show_dropdown.value = "All objects"
    app.apply_visibility()
    assert all(box.visible for box in app.boxes.values())


def test_selection_is_filled_and_can_be_shown_alone(app):
    app.select("obj_0002")
    assert app.labels["obj_0002"].visible  # the selected label stays on
    assert not app.labels["obj_0001"].visible
    app.isolate_checkbox.value = True
    app.apply_visibility()
    assert [i for i, box in app.boxes.items() if box.visible] == ["obj_0002"]


def test_click_picks_the_visible_box_along_the_ray(app):
    center = np.asarray(app.objects["obj_0002"].detected.bbox.center)
    origin = center + np.array([0.0, 0.0, 20.0])
    assert app.pick(origin, np.array([0.0, 0.0, -1.0])) == "obj_0002"
    app.show_dropdown.value = "Faulty only"
    app.apply_visibility()
    assert app.pick(origin, np.array([0.0, 0.0, -1.0])) != "obj_0002"


def test_click_uses_segment_points_over_boxes(monkeypatch):
    monkeypatch.delenv(METADATA_DIR_ENV, raising=False)
    scene = load_live_scene(offline=True)
    center = np.asarray(scene.objects[1].detected.center)  # obj_0002
    # One point of obj_0003 placed in front of obj_0002 along the ray: points decide, not boxes.
    points = np.array([center, center + (0.0, 0.0, 5.0)], np.float32)
    viewer = ViewerApp(
        replace(scene, pick_points=points, pick_labels=np.array([1, 2])),
        port=free_port(),
    )
    try:
        origin = center + np.array([0.0, 0.0, 20.0])
        down = np.array([0.0, 0.0, -1.0])
        assert viewer.pick(origin, down) == "obj_0003"
        viewer.click(origin, down)
        assert viewer.selected.object_id == "obj_0003"
        viewer.click(origin, down)  # same spot again: the object behind
        assert viewer.selected.object_id == "obj_0002"
        viewer.click(origin, down)  # and around
        assert viewer.selected.object_id == "obj_0003"
        viewer.boxes["obj_0003"].visible = False
        assert viewer.pick(origin, down) == "obj_0002"
    finally:
        viewer.server.stop()
