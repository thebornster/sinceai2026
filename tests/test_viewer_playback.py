import numpy as np
import pytest

from valmet.common.contracts import OrientedBox
from valmet.viewer.playback import (
    PathPlayer,
    find_nearby,
    find_nearest,
    pick_object,
    pick_point,
    ray_box_distance,
)

ROUTE = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 4.0, 0.0)]


def walk(player: PathPlayer, speed: float = 1.0, dt: float = 0.1) -> list[np.ndarray]:
    trace = [player.position()]
    for _ in range(10_000):
        if not player.playing:
            break
        player.update(dt, speed)
        trace.append(player.position())
    return trace


def test_starts_stopped_at_first_waypoint():
    player = PathPlayer(ROUTE)
    assert player.total_length == pytest.approx(7.0)
    player.update(1.0, 1.0)
    np.testing.assert_allclose(player.position(), ROUTE[0])


def test_walks_every_waypoint_at_constant_speed_and_stops_at_the_end():
    player = PathPlayer(ROUTE)
    player.play()
    trace = np.array(walk(player))
    steps = np.linalg.norm(np.diff(trace, axis=0), axis=1)
    assert steps.max() <= 0.1 + 1e-9
    np.testing.assert_allclose(trace[-1], ROUTE[-1])
    assert np.linalg.norm(trace - np.array(ROUTE[1]), axis=1).min() < 0.1
    assert player.finished and not player.playing


def test_pause_holds_and_play_resumes():
    player = PathPlayer(ROUTE)
    player.play()
    player.update(1.0, 2.0)
    player.pause()
    player.update(1.0, 2.0)
    assert player.distance == pytest.approx(2.0)
    player.play()
    player.update(1.0, 2.0)
    assert player.distance == pytest.approx(4.0)


def test_play_at_the_end_starts_over():
    player = PathPlayer(ROUTE)
    player.play()
    walk(player)
    player.play()
    assert player.distance == 0.0 and player.playing


def test_reset_returns_to_start_and_stops():
    player = PathPlayer(ROUTE)
    player.play()
    player.update(1.0, 2.0)
    player.reset()
    assert player.distance == 0.0 and not player.playing


def test_load_switches_route_and_stops():
    player = PathPlayer(ROUTE)
    player.play()
    player.update(1.0, 2.0)
    player.load([(1.0, 1.0, 0.0), (1.0, 2.0, 0.0)])
    assert not player.playing
    np.testing.assert_allclose(player.position(), (1.0, 1.0, 0.0))
    assert player.total_length == pytest.approx(1.0)


def test_single_waypoint_route_is_already_finished():
    player = PathPlayer([(1.0, 2.0, 0.0)])  # contracts allow a one-point path
    player.play()
    player.update(1.0, 1.0)
    assert player.finished and not player.playing
    np.testing.assert_allclose(player.position(), (1.0, 2.0, 0.0))


def test_repeated_waypoints_are_harmless():
    player = PathPlayer([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0)])
    player.play()
    player.update(1.0, 1.0)
    np.testing.assert_allclose(player.position(), (1.0, 0.0, 0.0))


def test_find_nearby_and_nearest():
    centers = {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([3.0, 0.0, 0.0])}
    position = np.zeros(3)
    assert find_nearby(position, centers, 1.0) == {"a"}
    assert find_nearby(position, centers, 0.5) == set()
    assert find_nearby(position, centers, 5.0) == {"a", "b"}
    assert find_nearest(position, centers) == "a"
    assert find_nearest(np.array([2.9, 0.0, 0.0]), centers) == "b"


def test_update_returns_the_distance_walked():
    player = PathPlayer([(0, 0, 0), (1, 0, 0)])
    assert player.update(0.1, 1.0) == 0.0  # not playing yet
    player.play()
    assert player.update(0.25, 1.0) == pytest.approx(0.25)
    assert player.update(10.0, 1.0) == pytest.approx(0.75)  # stops at the end


def test_heading_follows_the_current_leg():
    player = PathPlayer([(0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 2, 0), (1, 2, 0)])
    assert player.heading() == pytest.approx(0.0)
    player.distance = 1.5
    assert player.heading() == pytest.approx(np.pi / 2)
    player.distance = player.total_length  # a repeated last waypoint keeps the last real leg
    assert player.heading() == pytest.approx(np.pi / 2)


def test_heading_of_a_route_with_no_length_is_zero():
    assert PathPlayer([(1, 2, 0)]).heading() == 0.0


UNIT_BOX = OrientedBox(center=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0))


def test_ray_enters_box_at_its_face():
    t = ray_box_distance(np.array([-5.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]), UNIT_BOX)
    assert t == pytest.approx(4.5)


@pytest.mark.parametrize(
    "origin, direction",
    [
        ((-5.0, 2.0, 0.0), (1.0, 0.0, 0.0)),  # parallel, outside the slab
        ((-5.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),  # pointing away
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),  # starts inside
    ],
)
def test_ray_misses(origin, direction):
    assert ray_box_distance(np.array(origin), np.array(direction), UNIT_BOX) is None


def test_rotated_box_is_hit_in_its_own_frame():
    c, s = np.cos(np.radians(45)), np.sin(np.radians(45))
    box = OrientedBox(
        center=(0.0, 0.0, 0.0),
        extent=(2.0, 0.2, 0.2),
        rotation=[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
    )
    # The bar reaches 1.0 along the diagonal; a ray along y=1 passes beyond its end.
    t = ray_box_distance(np.array([-5.0, -5.0, 0.0]), np.array([1.0, 1.0, 0.0]), box)
    assert t == pytest.approx(np.hypot(5.0, 5.0) - 1.0)
    assert ray_box_distance(np.array([-5.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), box) is None


def test_pick_returns_the_first_box_along_the_ray():
    far = OrientedBox(center=(3.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0))
    boxes = {"far": far, "near": UNIT_BOX, "aside": OrientedBox(center=(0, 5, 0), extent=(1, 1, 1))}
    assert pick_object(np.array([-5.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), boxes) == "near"
    assert pick_object(np.array([-5.0, 0.0, 3.0]), np.array([1.0, 0.0, 0.0]), boxes) is None


def test_point_pick_takes_the_nearest_point_near_the_ray():
    points = np.array([[5.0, 0.0, 0.0], [3.0, 0.02, 0.0], [2.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    origin, direction = np.zeros(3), np.array([1.0, 0.0, 0.0])
    assert pick_point(origin, direction, points, 0.03, 0.01) == 1
    assert pick_point(origin, -direction, points, 0.03, 0.01) == 3
    assert pick_point(origin, np.array([0.0, 0.0, 1.0]), points, 0.03, 0.01) is None
