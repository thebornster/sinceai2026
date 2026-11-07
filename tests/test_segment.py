import json

import numpy as np

from valmet.common.artifacts import OBJECTS_FILE, SEMANTIC_DETAILS_FILE, SceneArtifacts


def _object_holding(objects, root, indices):
    """The object whose members hold most of the given source indices (orig_index = i + 1000)."""
    wanted = set((indices + 1000).tolist())
    best = max(
        objects,
        key=lambda o: len(wanted & set(np.load(root / o.gaussians.indices_file).tolist())),
    )
    return best


def test_finds_crate_and_pipe_and_drops_the_room(segmented_scene):
    root, objects, ranges = segmented_scene
    assert 2 <= len(objects) <= 4

    crate = _object_holding(objects, root, ranges["crate"])
    np.testing.assert_allclose(crate.bbox.center, (0.75, -0.75, 0.45), atol=0.05)
    pipe = _object_holding(objects, root, ranges["pipe"])
    assert pipe.object_id != crate.object_id
    assert pipe.bbox.extent[0] > 1.3  # one run along x, not broken up

    details = json.loads((root / SEMANTIC_DETAILS_FILE).read_text())
    assert details["objects"][pipe.object_id]["hint"] == "pipe segment"
    planes = {p["name"]: p for p in details["planes"]}
    assert abs(planes["ceiling"]["value"] - 2.4) < 0.03
    assert abs(planes["wall_x_min"]["value"] + 2.0) < 0.03
    assert abs(planes["wall_y_max"]["value"] - 2.0) < 0.03


def test_outputs_follow_the_contracts(segmented_scene):
    root, objects, _ = segmented_scene
    artifacts = SceneArtifacts.load(root)
    assert artifacts.check_consistency() == []
    assert [o.object_id for o in artifacts.objects] == [f"obj_{i:04d}" for i in range(1, len(objects) + 1)]
    for o in artifacts.objects:
        assert o.label is None and o.scene_id == "room"
        members = np.load(root / o.gaussians.indices_file)
        assert len(members) == o.gaussians.count > 0
        assert np.all(members >= 1000)
    assert (root / OBJECTS_FILE).exists()


def test_tilted_noisy_floor_is_removed_and_debris_dropped():
    from valmet.semantic.segment import (
        SegmentParams,
        filter_floor_debris,
        floor_height,
        remove_planes,
    )

    rng = np.random.default_rng(0)
    sp = SegmentParams()
    xy = rng.uniform(-3, 3, (60_000, 2))
    z = 0.01 * xy[:, 0]  # 6 cm rise across the room
    xyz = np.column_stack([xy, z + rng.normal(0, 0.005, len(z))])
    normals = np.tile([0.0, 0.0, 1.0], (len(xyz), 1))
    noisy = rng.random(len(xyz)) < 0.2
    normals[noisy] = [0.8, 0.0, 0.6]  # normal estimate off on 20 % of the floor

    floor_z = floor_height(xyz, normals, sp)
    assert np.abs(floor_z - z).max() < 0.02
    planes = [{"name": "floor", "label": "floor", "axis": 2, "value": 0.0}]
    remaining = remove_planes(
        xyz=xyz,
        normals=normals,
        planes=planes,
        floor_z=floor_z,
        sp=sp,
    )
    assert remaining.mean() < 0.01

    objs = [
        {"idx": np.arange(10), "tags": []},  # floor-level speckle
        {"idx": np.arange(10, 20), "tags": []},
    ]
    xyz[10:20, 2] += 0.5  # something standing on the floor
    kept = filter_floor_debris(objs, xyz, floor_z, sp)
    assert [o["idx"][0] for o in kept] == [10]
