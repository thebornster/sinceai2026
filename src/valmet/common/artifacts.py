"""On-disk layout and JSON I/O for scene artifacts.

One directory per scene (``data/processed/<scene_id>/``, or the mock under
``data/fixtures/mock_scene/``):

    scene_manifest.json     SceneManifest              (preprocessing)
    objects.json            list[DetectedObject]       (semantic)
    navmap.json             NavigationMapMetadata      (spatial)
    occupancy.npy           uint8 grid, shape = navmap.shape (spatial)
    paths.json              list[NavigationPath]       (spatial)
    asset_mapping.json      list[AssetMapping]         (metadata)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
from pydantic import BaseModel, TypeAdapter

from valmet.common.contracts import (
    AssetMapping,
    DetectedObject,
    NavigationMapMetadata,
    NavigationPath,
    SceneManifest,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MOCK_SCENE_DIR = REPO_ROOT / "data" / "fixtures" / "mock_scene"

MANIFEST_FILE = "scene_manifest.json"
OBJECTS_FILE = "objects.json"
NAVMAP_FILE = "navmap.json"
PATHS_FILE = "paths.json"
ASSET_MAPPING_FILE = "asset_mapping.json"

M = TypeVar("M", bound=BaseModel)


def save_model(path: Path, model: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_model(path: Path, cls: type[M]) -> M:
    return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def save_list(path: Path, items: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps([m.model_dump(mode="json") for m in items], indent=2)
    path.write_text(data + "\n", encoding="utf-8")


def load_list(path: Path, cls: type[M]) -> list[M]:
    return TypeAdapter(list[cls]).validate_json(Path(path).read_text(encoding="utf-8"))


@dataclass
class SceneArtifacts:
    """Whatever artifacts exist in a scene directory. Missing ones are None."""

    root: Path
    manifest: SceneManifest
    objects: list[DetectedObject] | None = None
    navmap: NavigationMapMetadata | None = None
    paths: list[NavigationPath] | None = None
    asset_mappings: list[AssetMapping] | None = None

    @classmethod
    def load(cls, root: Path) -> SceneArtifacts:
        root = Path(root)
        opt = lambda name: (root / name) if (root / name).exists() else None  # noqa: E731
        return cls(
            root=root,
            manifest=load_model(root / MANIFEST_FILE, SceneManifest),
            objects=load_list(p, DetectedObject) if (p := opt(OBJECTS_FILE)) else None,
            navmap=load_model(p, NavigationMapMetadata) if (p := opt(NAVMAP_FILE)) else None,
            paths=load_list(p, NavigationPath) if (p := opt(PATHS_FILE)) else None,
            asset_mappings=load_list(p, AssetMapping) if (p := opt(ASSET_MAPPING_FILE)) else None,
        )

    def load_occupancy(self) -> np.ndarray:
        if self.navmap is None:
            raise FileNotFoundError(f"no {NAVMAP_FILE} in {self.root}")
        return np.load(self.root / self.navmap.grid_file)

    def check_consistency(self) -> list[str]:
        """Cross-artifact checks. Returns a list of problems (empty = consistent)."""
        problems: list[str] = []
        sid = self.manifest.scene_id
        cs = self.manifest.coordinate_system
        object_ids = {o.object_id for o in self.objects or []}

        def same_scene(kind: str, items) -> None:
            for it in items:
                if it.scene_id != sid:
                    problems.append(f"{kind}: scene_id {it.scene_id!r} != manifest {sid!r}")

        if self.objects is not None:
            same_scene("object", self.objects)
            if len(object_ids) != len(self.objects):
                problems.append("objects: duplicate object_id")
        if self.navmap is not None:
            same_scene("navmap", [self.navmap])
            if cs.up_axis != "+Z":
                problems.append(f"navmap requires world up_axis '+Z', manifest has {cs.up_axis!r}")
            grid_path = self.root / self.navmap.grid_file
            if not grid_path.exists():
                problems.append(f"navmap: grid file {grid_path.name} missing")
            else:
                grid = np.load(grid_path)
                want = tuple(self.navmap.shape)
                if grid.shape != want or grid.dtype != np.uint8:
                    problems.append(f"navmap: grid {grid.dtype}{grid.shape} != uint8{want}")
        if self.paths is not None:
            same_scene("path", self.paths)
            for p in self.paths:
                if self.objects is not None and p.target_object_id not in object_ids:
                    problems.append(f"path: unknown target_object_id {p.target_object_id!r}")
        if self.asset_mappings is not None:
            same_scene("asset_mapping", self.asset_mappings)
            for a in self.asset_mappings:
                if self.objects is not None and a.scene_object_id not in object_ids:
                    problems.append(f"asset_mapping: unknown scene_object_id {a.scene_object_id!r}")
            mapped = [a.scene_object_id for a in self.asset_mappings]
            if len(set(mapped)) != len(mapped):
                problems.append("asset_mapping: object mapped more than once")
        return problems
