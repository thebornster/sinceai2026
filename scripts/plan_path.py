"""Plan a collision-free route to a target object and write paths.json.

The target comes from objects.json, or from a hand-placed box while segmentation is pending
(box center and full extents in world meters, optional yaw in degrees).

Usage:
    python scripts/plan_path.py --target-object-id obj_0012
    python scripts/plan_path.py --target-box 1.2,-0.5,0.6,0.8,0.6,1.2 --start -4,2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from valmet.common.artifacts import (
    NAVMAP_FILE,
    OBJECTS_FILE,
    PATHS_FILE,
    REPO_ROOT,
    load_list,
    load_model,
    save_list,
)
from valmet.common.contracts import DetectedObject, NavigationMapMetadata, OrientedBox
from valmet.spatial.occupancy import OBSTACLES_FILE
from valmet.spatial.planner import (
    PlannerParams,
    UnreachableTargetError,
    path_clearance,
    plan_path,
    sideways_segments,
)
from valmet.spatial.preview import render_navmap

PLACEHOLDER_ID = "target_placeholder"


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split(",")]


def _box_from_arg(values: list[float]) -> OrientedBox:
    if len(values) not in (6, 7):
        raise argparse.ArgumentTypeError("--target-box takes cx,cy,cz,ex,ey,ez[,yaw_deg]")
    yaw = np.deg2rad(values[6]) if len(values) == 7 else 0.0
    c, s = np.cos(yaw), np.sin(yaw)
    return OrientedBox(
        center=tuple(values[:3]),
        extent=tuple(values[3:6]),
        rotation=[[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--scene-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "equipment_room"
    )
    ap.add_argument("--report-dir", type=Path, default=REPO_ROOT / "reports")
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-object-id", help="object_id from objects.json")
    target.add_argument("--target-box", type=_floats, help="cx,cy,cz,ex,ey,ez[,yaw_deg]")
    ap.add_argument("--start", type=_floats, help="x,y start in world meters (default: auto)")
    ap.add_argument(
        "--body-depth",
        type=float,
        default=PlannerParams.body_depth,
        help="chest-to-back depth (m) for squeezing sideways through gaps narrower than the shoulders",
    )
    ap.add_argument(
        "--no-sideways", action="store_true", help="only routes that fit walking normally"
    )
    args = ap.parse_args(argv)

    navmap = load_model(args.scene_dir / NAVMAP_FILE, NavigationMapMetadata)
    grid = np.load(args.scene_dir / navmap.grid_file)
    obstacles = None
    if not args.no_sideways and (args.scene_dir / OBSTACLES_FILE).exists():
        obstacles = np.load(args.scene_dir / OBSTACLES_FILE)
    if args.target_object_id:
        objects = {o.object_id: o for o in load_list(args.scene_dir / OBJECTS_FILE, DetectedObject)}
        if args.target_object_id not in objects:
            print(f"error: {args.target_object_id!r} not in {OBJECTS_FILE}")
            return 2
        box, target_id = objects[args.target_object_id].bbox, args.target_object_id
    else:
        box, target_id = _box_from_arg(values=args.target_box), PLACEHOLDER_ID

    try:
        path = plan_path(
            navmap=navmap,
            grid=grid,
            target_box=box,
            target_object_id=target_id,
            start_xy=tuple(args.start[:2]) if args.start else None,
            params=PlannerParams(body_depth=args.body_depth),
            obstacles=obstacles,
        )
    except UnreachableTargetError as exc:
        print(f"error: {exc}")
        return 3

    save_list(args.scene_dir / PATHS_FILE, [path])
    preview = render_navmap(
        path_png=args.report_dir / "preview" / f"{navmap.scene_id}_path.png",
        navmap=navmap,
        grid=grid,
        title=f"{navmap.scene_id}: route to {target_id}",
        path=path,
        target_box=box,
    )
    clearance = path_clearance(
        navmap=navmap,
        grid=grid,
        path=path,
        obstacles=obstacles,
    )
    sideways = sideways_segments(
        navmap=navmap,
        grid=grid,
        path=path,
    )
    wp = np.asarray(path.waypoints)
    sideways_m = (
        float(np.linalg.norm(np.diff(wp, axis=0), axis=1)[sideways].sum()) if sideways else 0.0
    )
    print(
        f"start {np.round(path.start[:2], 2).tolist()} -> inspection {np.round(path.inspection_position[:2], 2).tolist()}"
    )
    print(
        f"{len(path.waypoints)} waypoints, length {path.length():.2f} m, min clearance ~{clearance:.2f} m"
    )
    print(f"sideways squeeze: {sideways_m:.2f} m over {sum(sideways)} segments")
    print(
        f"wrote {(args.scene_dir / PATHS_FILE).relative_to(REPO_ROOT)}, preview {preview.relative_to(REPO_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
