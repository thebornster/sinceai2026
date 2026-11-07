"""Start the live viewer on a scene's artifacts.

Usage:
    python scripts/run_viewer.py
    python scripts/run_viewer.py --scene-dir data/processed/<scene_id> --port 8081
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

from valmet.common.artifacts import MOCK_SCENE_DIR
from valmet.integration.live_scene import load_live_scene
from valmet.viewer.app import ViewerApp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--scene-dir", type=Path, default=MOCK_SCENE_DIR, help="artifact folder")
    ap.add_argument("--port", type=int, default=8080, help="preferred port")
    ap.add_argument("--offline", action="store_true", help="read equipment data from the snapshot")
    args = ap.parse_args(argv)

    try:
        scene = load_live_scene(args.scene_dir, offline=args.offline)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not scene.manifest.coordinate_system.verified:
        print("warning: the world frame is not verified, so positions may be wrong")
    print(f"{scene.manifest.scene_id}: {len(scene.objects)} objects, {len(scene.faulty)} faulty")

    with contextlib.suppress(KeyboardInterrupt):
        ViewerApp(scene, port=args.port).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
