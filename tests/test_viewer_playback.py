import numpy as np
import pytest

from valmet.viewer.playback import PathPlayer, find_nearby, find_nearest

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
