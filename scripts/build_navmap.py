"""Build the 2-D occupancy grid (navmap.json + occupancy.npy) from the preprocessed scene.

Usage:
    python scripts/build_navmap.py
    python scripts/build_navmap.py --radius 0.2 --max-step-height 0.3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from valmet.common.artifacts import MANIFEST_FILE, NAVMAP_FILE, REPO_ROOT, load_model, save_model
from valmet.common.contracts import SceneManifest
from valmet.preprocessing.export import SCENE_CLEAN_FILE
from valmet.spatial.occupancy import OBSTACLES_FILE, NavParams, build_layers
from valmet.spatial.preview import render_navmap


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--scene-dir", type=Path, default=REPO_ROOT / "data" / "processed" / "equipment_room"
    )
    ap.add_argument("--report-dir", type=Path, default=REPO_ROOT / "reports")
    ap.add_argument("--resolution", type=float, default=NavParams.resolution)
    ap.add_argument("--radius", type=float, default=NavParams.radius)
    ap.add_argument("--height", type=float, default=NavParams.height)
    ap.add_argument("--head-clearance", type=float, default=NavParams.head_clearance)
    ap.add_argument("--max-step-height", type=float, default=NavParams.max_step_height)
    args = ap.parse_args(argv)

    manifest = load_model(args.scene_dir / MANIFEST_FILE, SceneManifest)
    cs = manifest.coordinate_system
    if cs.up_axis != "+Z" or cs.units != "meters":
        print(
            f"error: navigation needs a +Z-up metric scene, manifest has {cs.up_axis!r}/{cs.units!r}"
        )
        return 2

    scene = np.load(args.scene_dir / SCENE_CLEAN_FILE)
    grid, obstacles, navmap, info = build_layers(
        positions=scene["positions"],
        opacities=scene["opacities"],
        scene_id=manifest.scene_id,
        params=NavParams(
            resolution=args.resolution,
            radius=args.radius,
            height=args.height,
            head_clearance=args.head_clearance,
            max_step_height=args.max_step_height,
        ),
    )
    save_model(args.scene_dir / NAVMAP_FILE, navmap)
    np.save(args.scene_dir / navmap.grid_file, grid)
    np.save(args.scene_dir / OBSTACLES_FILE, obstacles)

    preview = render_navmap(
        path_png=args.report_dir / "preview" / f"{manifest.scene_id}_navmap.png",
        navmap=navmap,
        grid=grid,
        title=f"{manifest.scene_id}: occupancy (light = free, dark = blocked incl. radius, blue = unknown)",
    )
    report = {"scene_id": manifest.scene_id, **info, "preview": str(preview.relative_to(REPO_ROOT))}
    report_path = args.report_dir / f"navigation_{manifest.scene_id}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    f = info["fractions"]
    print(f"grid {info['shape']} at {navmap.resolution} m")
    print(f"free {f['free']:.2f}  occupied {f['occupied']:.2f}  unknown {f['unknown']:.2f}")
    print(f"largest walkable areas (m2): {info['walkable_areas_m2']}")
    print(f"report: {report_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
