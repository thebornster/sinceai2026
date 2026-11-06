"""Clean a Gaussian-splat PLY, estimate the world frame and write the preprocessing artifacts.

Outputs (data/processed/<scene_id>/): scene_manifest.json, scene_clean.npz, viewer_splats.npz.
Report: reports/preprocessing_<scene_id>.json plus preview PNGs in reports/preview/.

Usage:
    python scripts/preprocess_scene.py
    python scripts/preprocess_scene.py --ceiling-height-m 3.2
    python scripts/preprocess_scene.py --scale 1.6 --flip-up
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from valmet.common.artifacts import REPO_ROOT
from valmet.preprocessing.cleaning import CleaningParams, compute_keep_mask
from valmet.preprocessing.export import build_manifest, write_outputs, write_previews
from valmet.preprocessing.gaussians import load_gaussians, transform_gaussians
from valmet.preprocessing.ply_inspect import _sha256
from valmet.preprocessing.world_frame import FrameParams, ScaleSpec, estimate_world_frame

DEFAULT_INPUT = REPO_ROOT / "data" / "raw" / "Equipment Room.ply"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--scene-id", default="equipment_room")
    ap.add_argument("--output-root", type=Path, default=REPO_ROOT / "data" / "processed")
    ap.add_argument("--report-dir", type=Path, default=REPO_ROOT / "reports")
    ap.add_argument("--min-opacity", type=float, default=CleaningParams.min_opacity)
    ap.add_argument(
        "--viewer-max-splats", type=int, default=800_000, help="most opaque splats kept for Viser"
    )
    ap.add_argument("--flip-up", action="store_true", help="use +Y (not -Y) as the source up hint")
    scale = ap.add_mutually_exclusive_group()
    scale.add_argument("--scale", type=float, help="source units -> meters factor")
    scale.add_argument("--ceiling-height-m", type=float, help="known floor-to-ceiling height")
    ap.add_argument("--no-hash", action="store_true", help="skip SHA-256 of the source file")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    def lap(step: str, message: str) -> None:
        timings[step] = round(time.perf_counter() - t0 - sum(timings.values()), 2)
        print(f"[{timings[step]:6.1f}s] {message}")

    scene = load_gaussians(path=args.input)
    lap("load", f"loaded {len(scene):,} Gaussians")

    keep, cleaning = compute_keep_mask(
        scene=scene,
        params=CleaningParams(min_opacity=args.min_opacity),
    )
    clean = scene.subset(keep)
    lap("clean", f"cleaned -> {len(clean):,} kept {cleaning['removed']}")

    transform, frame = estimate_world_frame(
        positions=clean.positions,
        opacities=clean.opacities,
        params=FrameParams(up_hint=(0.0, 1.0, 0.0) if args.flip_up else (0.0, -1.0, 0.0)),
        scale_spec=ScaleSpec(
            scale=args.scale,
            ceiling_height_m=args.ceiling_height_m,
        ),
    )
    world = transform_gaussians(
        scene=clean,
        matrix=transform,
    )
    lap("world_frame", "estimated world frame")

    manifest = build_manifest(
        scene_id=args.scene_id,
        source=args.input,
        source_sha256=None if args.no_hash else _sha256(args.input),
        source_vertex_count=len(scene),
        source_to_world=transform,
        scale_source=frame["scale_source"],
    )
    outputs = write_outputs(
        out_dir=args.output_root / args.scene_id,
        manifest=manifest,
        world=world,
        viewer=transform_gaussians(
            scene=scene,
            matrix=transform,
        ),
        viewer_max_splats=args.viewer_max_splats,
    )
    previews = write_previews(
        world=world,
        out_dir=args.report_dir / "preview",
        scene_id=args.scene_id,
    )
    lap("write", "wrote outputs")

    lo, hi = np.percentile(world.positions, [1, 99], axis=0)
    report = {
        "scene_id": args.scene_id,
        "source": args.input.name,
        "cleaning": cleaning,
        "world_frame": frame,
        "world_bounds_p1_p99": {"min": lo.tolist(), "max": hi.tolist()},
        "outputs": {k: str(v.relative_to(REPO_ROOT)) for k, v in outputs.items()},
        "previews": [str(p.relative_to(REPO_ROOT)) for p in previews],
        "timings_s": timings,
    }
    report_path = args.report_dir / f"preprocessing_{args.scene_id}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    ceiling = frame["ceiling_plane_source"].get("distance_from_floor")
    print(f"up (source): {np.round(frame['up_source'], 3).tolist()}")
    print(f"floor-to-ceiling (source units): {ceiling}")
    print(f"wall yaw: {frame['wall_yaw_deg']:.1f} deg, manhattan strength {frame['manhattan_strength']:.2f}")
    print(f"scale: {frame['scale']} ({frame['scale_source']})")
    print(f"world bounds p1..p99: {np.round(lo, 2).tolist()} .. {np.round(hi, 2).tolist()}")
    print(f"report: {report_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
