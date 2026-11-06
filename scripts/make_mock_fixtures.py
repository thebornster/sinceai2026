"""Regenerate the mock scene artifacts in data/fixtures/mock_scene/.

The mock is hand-made demo data (not derived from the real PLY). It lets every
workstream develop against valid contracts before the real pipelines exist.
Run after any schema change:  python scripts/make_mock_fixtures.py
"""

from __future__ import annotations

import math

import numpy as np

from valmet.common import (
    AssetMapping,
    CoordinateSystem,
    DetectedObject,
    NavigationMapMetadata,
    NavigationPath,
    OrientedBox,
    RobotClearance,
    SceneManifest,
)
from valmet.common.artifacts import (
    ASSET_MAPPING_FILE,
    MANIFEST_FILE,
    MOCK_SCENE_DIR,
    NAVMAP_FILE,
    OBJECTS_FILE,
    PATHS_FILE,
    SceneArtifacts,
    save_list,
    save_model,
)

SCENE = "mock_equipment_room"


def yaw(deg: float) -> list[list[float]]:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def main() -> None:
    out = MOCK_SCENE_DIR
    manifest = SceneManifest(
        scene_id=SCENE,
        source_filename="mock_equipment_room.ply",
        source_vertex_count=0,
        coordinate_system=CoordinateSystem(
            up_axis="+Z",
            units="meters",
            meters_per_unit=1.0,
            verified=True,
            notes="MOCK. Pretends the source was Y-up; source_to_world rotates +90 deg about X.",
        ),
        # (x, y, z)_source -> (x, -z, y)_world
        source_to_world=[[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    )
    objects = [
        DetectedObject(
            object_id="obj_0001",
            scene_id=SCENE,
            label="pump",
            confidence=0.91,
            center=(1.0, 0.5, 0.45),
            bbox=OrientedBox(center=(1.0, 0.5, 0.45), extent=(0.8, 0.5, 0.9)),
        ),
        DetectedObject(
            object_id="obj_0002",
            scene_id=SCENE,
            label="valve",
            confidence=0.64,
            center=(-1.0, -0.8, 1.1),
            bbox=OrientedBox(center=(-1.0, -0.8, 1.1), extent=(0.3, 0.2, 0.3), rotation=yaw(30)),
        ),
        DetectedObject(
            object_id="obj_0003",
            scene_id=SCENE,
            label=None,
            confidence=None,
            center=(0.2, -1.0, 0.6),
            bbox=OrientedBox(center=(0.2, -1.0, 0.6), extent=(0.6, 0.4, 1.2), rotation=yaw(-15)),
        ),
    ]

    navmap = NavigationMapMetadata(
        scene_id=SCENE,
        origin=(-2.0, -1.5, 0.0),
        resolution=0.1,
        shape=(40, 30),
        clearance=RobotClearance(radius=0.25, height=1.7, max_step_height=0.05),
    )
    v = navmap.cell_values
    grid = np.full(navmap.shape, v.free, dtype=np.uint8)
    grid[[0, -1], :] = v.occupied  # walls
    grid[:, [0, -1]] = v.occupied
    for o in objects:  # crude inflated footprints
        lo = navmap.world_to_cell(np.asarray(o.bbox.corners())[:, :2].min(0) - 0.25)[0]
        hi = navmap.world_to_cell(np.asarray(o.bbox.corners())[:, :2].max(0) + 0.25)[0]
        grid[max(lo[0], 0) : hi[0] + 1, max(lo[1], 0) : hi[1] + 1] = v.occupied
    grid[34:, 25:] = v.unknown  # unscanned corner

    start = (-1.5, 1.0, 0.0)
    routes = {  # waypoints after the start, ending at the inspection position
        "obj_0001": [(-0.5, 1.1, 0.0), (0.3, 1.1, 0.0)],
        "obj_0002": [(-1.25, 0.0, 0.0), (-0.95, -0.25, 0.0)],
        "obj_0003": [(-0.5, 0.2, 0.0), (0.25, -0.35, 0.0)],
    }
    paths = [
        NavigationPath(
            scene_id=SCENE,
            start=start,
            target_object_id=object_id,
            inspection_position=waypoints[-1],
            waypoints=[start, *waypoints],
        )
        for object_id, waypoints in routes.items()
    ]
    mappings = [
        AssetMapping(
            scene_id=SCENE, scene_object_id="obj_0001", equipment_id="PMP-101", method="matched"
        ),
        AssetMapping(
            scene_id=SCENE, scene_object_id="obj_0002", equipment_id="VLV-204", method="random"
        ),
    ]

    save_model(out / MANIFEST_FILE, manifest)
    save_list(out / OBJECTS_FILE, objects)
    save_model(out / NAVMAP_FILE, navmap)
    np.save(out / navmap.grid_file, grid)
    save_list(out / PATHS_FILE, paths)
    save_list(out / ASSET_MAPPING_FILE, mappings)

    problems = SceneArtifacts.load(out).check_consistency()
    if problems:
        raise SystemExit("inconsistent mock artifacts:\n" + "\n".join(problems))
    print(f"wrote mock scene to {out}")


if __name__ == "__main__":
    main()
