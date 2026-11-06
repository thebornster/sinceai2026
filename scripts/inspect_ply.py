"""Inspect a Gaussian-splat PLY without modifying it and write a JSON report.

Usage:
    python scripts/inspect_ply.py "data/raw/Equipment Room.ply"
    python scripts/inspect_ply.py scene.ply --sample-size 500000 --seed 1 --hash
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from valmet.preprocessing.ply_inspect import PlyInspectionError, inspect_ply

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "reports" / "ply_inspection.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("ply", type=Path, help="path to the PLY file")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT, help="JSON report path")
    ap.add_argument("--sample-size", type=int, default=200_000, help="vertices to sample")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (reproducible)")
    ap.add_argument("--hash", action="store_true", help="also compute SHA-256 of the file")
    args = ap.parse_args(argv)

    try:
        report = inspect_ply(args.ply, args.sample_size, args.seed, compute_hash=args.hash)
    except PlyInspectionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    v = report["header"]["elements"]
    gs = report["gaussian_schema"]
    print(
        f"{report['file']['name']}: {report['file']['size_bytes'] / 2**20:.1f} MiB, "
        f"{report['header']['format']}"
    )
    for e in v:
        print(f"  element {e['name']}: {e['count']} x {len(e['properties'])} properties")
    print(
        f"  conventional 3DGS: {gs['is_conventional_3dgs']} (SH degree {gs['sh_degree']}, "
        f"missing {gs['missing_required'] or 'none'})"
    )
    if "positions" in report:
        for axis, s in report["positions"]["axes"].items():
            if "min" in s:
                p = s["percentiles"]
                print(
                    f"  {axis}: min {s['min']:.3f}  p1 {p['1']:.3f}  p50 {p['50']:.3f}  "
                    f"p99 {p['99']:.3f}  max {s['max']:.3f}"
                )
        print(f"  non-finite position rows in sample: {report['positions']['nonfinite_rows']}")
    for w in report["warnings"]:
        print(f"  warning: {w}")
    print(f"report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
