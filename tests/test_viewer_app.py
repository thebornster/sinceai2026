import socket

import numpy as np
import pytest

pytest.importorskip("viser")

from valmet.integration.live_scene import load_live_scene  # noqa: E402
from valmet.metadata.mapping import METADATA_DIR_ENV  # noqa: E402
from valmet.viewer.app import (  # noqa: E402
    BOX_COLOR,
    FAULT_COLOR,
    HIGHLIGHT_COLOR,
    ViewerApp,
    target_label,
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def app(monkeypatch):
    monkeypatch.delenv(METADATA_DIR_ENV, raising=False)
    viewer = ViewerApp(load_live_scene(offline=True), port=free_port())
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


def test_target_without_a_route_says_so_and_does_not_move(app):
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
