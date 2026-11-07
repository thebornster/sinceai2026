"""Map scene objects (``DetectedObject``) to Excel devices, reproducibly.

Rules, in order:
1. The target object gets the target asset (detail mapping "target").
2. Every other object gets a healthy device only.
3. Label maps to an Excel asset_class and an unused healthy device of that class
   exists -> "class_match".
4. Otherwise pick a healthy device with the seeded RNG -> "synthetic".
   - Known classes whose healthy devices are all used reuse one of their own
     class (a pump never shows valve data).
   - Other labels and ``label=None`` take unused healthy devices first.
5. A device that is already used gets a derived id (ASSET-005-c2, -c3, ...) that
   keeps source_asset_id and the original Excel refs -> "clone".

Two files are written per scene:
- ``asset_mapping.json``: ``list[AssetMapping]`` exactly as in valmet.common.contracts.
  method: target -> "procedural"; label class == device class -> "matched";
  anything else -> "random". equipment_id is the Excel asset_id.
- ``asset_mapping_details.json``: our richer entries (tag, condition, derived
  device_id, excel_refs), read by valmet.api.

Run:
    python -m valmet.metadata.mapping                       # mock scene
    python -m valmet.metadata.mapping --scene-dir data/processed/<scene_id> --target obj_0042
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from valmet.common.artifacts import (
    ASSET_MAPPING_FILE,
    MANIFEST_FILE,
    MOCK_SCENE_DIR,
    OBJECTS_FILE,
    PATHS_FILE,
    REPO_ROOT,
    SceneArtifacts,
    load_list,
    load_model,
    save_list,
)
from valmet.common.contracts import AssetMapping, DetectedObject, NavigationPath, SceneManifest
from valmet.metadata.registry import AssetRecord, AssetRegistry

DETAILS_FILE = "asset_mapping_details.json"
METADATA_FIXTURES_DIR = REPO_ROOT / "data" / "fixtures" / "metadata"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
METADATA_DIR_ENV = "VALMET_METADATA_DIR"

DEFAULT_TARGET_ASSET = "ASSET-014"

# Normalised label -> Excel asset_class. Labels are lower-cased, "_"/"-" -> " ".
LABEL_TO_ASSET_CLASS = {
    "pump": "PROCESS_PUMP",
    "process pump": "PROCESS_PUMP",
    "centrifugal pump": "PROCESS_PUMP",
    "motor": "ELECTRIC_MOTOR",
    "electric motor": "ELECTRIC_MOTOR",
    "valve": "CONTROL_VALVE",
    "control valve": "CONTROL_VALVE",
    "pressure transmitter": "PRESSURE_TRANSMITTER",
    "pressure sensor": "PRESSURE_TRANSMITTER",
    "pressure transmitter/gauge": "PRESSURE_TRANSMITTER",
    "gauge": "PRESSURE_TRANSMITTER",
    "vibration sensor": "VIBRATION_SENSOR",
}


def normalize_label(label: str | None) -> str | None:
    if label is None:
        return None
    norm = " ".join(label.lower().replace("_", " ").replace("-", " ").split())
    return norm or None


def label_asset_class(label: str | None) -> str | None:
    """Excel asset_class for a detector label, or None if it has no equivalent."""
    return LABEL_TO_ASSET_CLASS.get(normalize_label(label) or "")


def default_output_dir(scene_dir: Path, scene_id: str) -> Path:
    """Mock scene -> data/fixtures/metadata/<scene_id>/, real -> data/processed/<scene_id>/."""
    if Path(scene_dir).resolve() == MOCK_SCENE_DIR.resolve():
        return METADATA_FIXTURES_DIR / scene_id
    return PROCESSED_DIR / scene_id


_DIR_OVERRIDE: ContextVar[Path | None] = ContextVar("metadata_dir_override", default=None)


def metadata_dir() -> Path:
    """Folder valmet.api reads asset_mapping_details.json (and the snapshot) from.

    An active ``use_metadata_dir(...)`` block wins; then $VALMET_METADATA_DIR
    (relative to repo root if not absolute); else the mock scene.
    """
    override = _DIR_OVERRIDE.get()
    if override is not None:
        return override
    env = os.environ.get(METADATA_DIR_ENV, "").strip()
    if not env:
        return METADATA_FIXTURES_DIR / "mock_equipment_room"
    path = Path(env).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


@contextmanager
def use_metadata_dir(path: Path) -> Iterator[Path]:
    """Temporarily make ``metadata_dir()`` return ``path`` (used by the snapshot export)."""
    token = _DIR_OVERRIDE.set(Path(path))
    try:
        yield Path(path)
    finally:
        _DIR_OVERRIDE.reset(token)


def resolve_target(
    objects: list[DetectedObject],
    target_object: str | None = None,
    scene_dir: Path | None = None,
) -> str:
    """Pick the faulty target object.

    Explicit ``target_object`` wins (needed when navigation wants the target before
    any path exists); else the first path's target_object_id in ``scene_dir``;
    else the first object (by id) labelled as a pump.
    """
    ids = {o.object_id for o in objects}
    if target_object is not None:
        if target_object not in ids:
            raise ValueError(f"Target object {target_object!r} is not in objects")
        return target_object
    if scene_dir is not None and (Path(scene_dir) / PATHS_FILE).exists():
        paths = load_list(Path(scene_dir) / PATHS_FILE, NavigationPath)
        if paths and paths[0].target_object_id in ids:
            return paths[0].target_object_id
    pumps = sorted(o.object_id for o in objects if label_asset_class(o.label) == "PROCESS_PUMP")
    if pumps:
        return pumps[0]
    raise ValueError("No target given, no path target, and no object labelled as a pump")


# --------------------------------------------------------------------------- #
# Mapping
# --------------------------------------------------------------------------- #
def _contract_method(detail_mapping: str, label: str | None, asset_class: str) -> str:
    if detail_mapping == "target":
        return "procedural"
    return "matched" if label_asset_class(label) == asset_class else "random"


def _entry(
    obj: DetectedObject, rec: AssetRecord, device_id: str, mapping: str, is_target: bool
) -> dict[str, Any]:
    return {
        "object_id": obj.object_id,
        "scene_id": obj.scene_id,
        "object_class": obj.label,
        "device_id": device_id,
        "source_asset_id": rec.asset_id,
        "tag": rec.identity.tag,
        "asset_class": rec.asset_class,
        "condition_state": rec.condition.condition_state,
        "health_score": rec.condition.health_score,
        "mapping": mapping,
        "method": _contract_method(mapping, obj.label, rec.asset_class),
        "is_target": is_target,
        "excel_refs": {
            "Asset_Master": rec.identity.source_ref,
            "Current_State": rec.condition.source_ref,
        },
    }


def build_mapping(
    objects: list[DetectedObject],
    registry: AssetRegistry,
    target_object: str,
    seed: int = 42,
    target_asset: str = DEFAULT_TARGET_ASSET,
) -> list[dict[str, Any]]:
    """Detail entries, one per object, ordered by object_id."""
    rng = random.Random(seed)
    objects = sorted(objects, key=lambda o: o.object_id)  # order-independent input
    ids = [o.object_id for o in objects]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate object_id in objects")
    if target_object not in ids:
        raise ValueError(f"Target object {target_object!r} is not in objects")
    target_rec = registry.get(target_asset)  # raises KeyError if unknown

    healthy = sorted((r for r in registry.all() if r.is_healthy), key=lambda r: r.asset_id)
    if not healthy:
        raise ValueError("Registry has no healthy devices to map")
    uses: dict[str, int] = {target_rec.asset_id: 1}

    def assign(obj: DetectedObject, rec: AssetRecord, preferred: str) -> dict[str, Any]:
        n = uses.get(rec.asset_id, 0) + 1
        uses[rec.asset_id] = n
        if n == 1:
            return _entry(obj, rec, rec.asset_id, preferred, False)
        return _entry(obj, rec, f"{rec.asset_id}-c{n}", "clone", False)

    out: dict[str, dict[str, Any]] = {}
    rest: list[DetectedObject] = []

    # Pass 1: target, then class matches. Done before any synthetic pick so an
    # unlabelled object can never take the last healthy device a pump needs.
    for obj in objects:
        if obj.object_id == target_object:
            out[obj.object_id] = _entry(obj, target_rec, target_rec.asset_id, "target", True)
            continue
        asset_class = label_asset_class(obj.label)
        candidates = [r for r in healthy if r.asset_class == asset_class and r.asset_id not in uses]
        if candidates:
            out[obj.object_id] = assign(obj, rng.choice(candidates), "class_match")
        else:
            rest.append(obj)

    # Pass 2: synthetic picks (clones when the device is already used).
    for obj in rest:
        asset_class = label_asset_class(obj.label)
        if asset_class:
            pool = [r for r in healthy if r.asset_class == asset_class] or healthy
        else:
            pool = [r for r in healthy if r.asset_id not in uses] or healthy
        out[obj.object_id] = assign(obj, rng.choice(pool), "synthetic")

    return [out[i] for i in ids]


def to_contract(details: list[dict[str, Any]]) -> list[AssetMapping]:
    """The shared-contract view of the detail entries."""
    return [
        AssetMapping(
            scene_id=d["scene_id"],
            scene_object_id=d["object_id"],
            equipment_id=d["source_asset_id"],
            method=d["method"],
        )
        for d in details
    ]


def check_against_scene(
    scene_dir: Path, objects: list[DetectedObject], mappings: list[AssetMapping]
) -> list[str]:
    """Their cross-artifact checks (ids exist, scene_id matches, mapped once)."""
    manifest_path = Path(scene_dir) / MANIFEST_FILE
    if manifest_path.exists():
        manifest = load_model(manifest_path, SceneManifest)
    else:
        manifest = SceneManifest(scene_id=objects[0].scene_id, source_filename="unknown")
    arts = SceneArtifacts(
        root=Path(scene_dir), manifest=manifest, objects=objects, asset_mappings=mappings
    )
    return arts.check_consistency()


def write_outputs(details: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    save_list(out_dir / ASSET_MAPPING_FILE, to_contract(details))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / DETAILS_FILE).write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")


def run(
    scene_dir: Path = MOCK_SCENE_DIR,
    target_object: str | None = None,
    target_asset: str = DEFAULT_TARGET_ASSET,
    seed: int = 42,
    out_dir: Path | None = None,
    registry: AssetRegistry | None = None,
) -> tuple[list[dict[str, Any]], Path]:
    """Load objects, build the mapping, validate it, write both files."""
    scene_dir = Path(scene_dir)
    objects = load_list(scene_dir / OBJECTS_FILE, DetectedObject)
    if not objects:
        raise ValueError(f"No objects in {scene_dir / OBJECTS_FILE}")
    target = resolve_target(objects, target_object, scene_dir)
    details = build_mapping(objects, registry or AssetRegistry(), target, seed, target_asset)
    problems = check_against_scene(scene_dir, objects, to_contract(details))
    if problems:
        raise ValueError(f"Mapping is inconsistent with the scene: {problems}")
    out = Path(out_dir) if out_dir else default_output_dir(scene_dir, objects[0].scene_id)
    write_outputs(details, out)
    return details, out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Map scene objects to Excel devices.")
    ap.add_argument("--scene-dir", type=Path, default=MOCK_SCENE_DIR)
    ap.add_argument("--target", default=None, help="Faulty target object_id (default: path target)")
    ap.add_argument("--target-asset", default=DEFAULT_TARGET_ASSET)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    details, out = run(args.scene_dir, args.target, args.target_asset, args.seed, args.out)
    try:
        shown = out.relative_to(REPO_ROOT)
    except ValueError:
        shown = out
    print(f"Wrote {len(details)} entries to {shown}/{{{ASSET_MAPPING_FILE}, {DETAILS_FILE}}}\n")
    print(
        f"{'object_id':<10} {'label':<20} {'device_id':<13} {'tag':<7} "
        f"{'condition':<19} {'mapping':<12} method"
    )
    for d in details:
        flag = "  <- TARGET" if d["is_target"] else ""
        print(
            f"{d['object_id']:<10} {str(d['object_class']):<20} {d['device_id']:<13} "
            f"{d['tag']:<7} {d['condition_state']:<19} {d['mapping']:<12} {d['method']}{flag}"
        )


if __name__ == "__main__":
    main()
