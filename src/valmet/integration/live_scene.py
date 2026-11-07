"""Joins scene artifacts with equipment data for the live viewer.

This is the only place the viewer's data comes from: geometry and routes from
``SceneArtifacts``, device identity and diagnoses from ``valmet.api``. Everything
is loaded once at startup, so the viewer's frame loop never touches the workbook.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from valmet import api
from valmet.common import DetectedObject, NavigationMapMetadata, NavigationPath, SceneManifest
from valmet.common.artifacts import MOCK_SCENE_DIR, SceneArtifacts

HEALTHY_STATE = "healthy"


@dataclass(frozen=True)
class LiveObject:
    """One detected object plus whatever is known about it."""

    detected: DetectedObject
    asset: dict[str, Any] | None = None  # api.get_asset(); None = no equipment record
    diagnosis: dict[str, Any] | None = None  # api.diagnose_fault()
    route: NavigationPath | None = None  # shortest path in paths.json; None = no route yet

    @property
    def object_id(self) -> str:
        return self.detected.object_id

    @property
    def is_faulty(self) -> bool:
        return self.asset is not None and self.asset["condition_state"] != HEALTHY_STATE

    @property
    def name(self) -> str:
        """Unique display name, for example 'P-002 pump (obj_0001)'."""
        label = self.detected.label or "unclassified"
        tag = f"{self.asset['device']['tag']} " if self.asset else ""
        return f"{tag}{label} ({self.object_id})"


@dataclass(frozen=True)
class LiveScene:
    manifest: SceneManifest
    objects: list[LiveObject]
    navmap: NavigationMapMetadata | None = None

    @property
    def faulty(self) -> list[LiveObject]:
        return [obj for obj in self.objects if obj.is_faulty]


def _shortest_routes(paths: list[NavigationPath] | None) -> dict[str, NavigationPath]:
    routes: dict[str, NavigationPath] = {}
    for path in paths or []:
        best = routes.get(path.target_object_id)
        if best is None or path.length() < best.length():
            routes[path.target_object_id] = path
    return routes


def _equipment(object_id: str, scene_id: str, offline: bool) -> tuple[dict | None, dict | None]:
    """Asset and diagnosis for one object, or (None, None) if it has no record in this scene."""
    try:
        asset = api.get_asset(object_id, offline=offline)
    except KeyError:  # api.UnknownObjectError: the viewer may show unmapped objects
        return None, None
    if asset["scene_id"] != scene_id:  # metadata belongs to another scene; ids may collide
        return None, None
    return asset, api.diagnose_fault(object_id, offline=offline)


def load_live_scene(root: Path = MOCK_SCENE_DIR, offline: bool = False) -> LiveScene:
    """Load and join everything the viewer shows. Raises ValueError if the scene is unusable."""
    artifacts = SceneArtifacts.load(root)
    problems = artifacts.check_consistency()
    if problems:
        raise ValueError(f"inconsistent artifacts in {root}: " + "; ".join(problems))
    if not artifacts.objects:
        raise ValueError(f"no objects in {root}")
    up_axis = artifacts.manifest.coordinate_system.up_axis
    if up_axis not in (None, "+Z"):
        raise ValueError(f"the viewer needs a +Z-up world, manifest has {up_axis!r}")

    routes = _shortest_routes(artifacts.paths)
    objects = []
    for detected in artifacts.objects:
        asset, diagnosis = _equipment(detected.object_id, artifacts.manifest.scene_id, offline)
        objects.append(LiveObject(detected, asset, diagnosis, routes.get(detected.object_id)))
    return LiveScene(artifacts.manifest, objects, artifacts.navmap)
