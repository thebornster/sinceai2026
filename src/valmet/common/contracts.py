"""Shared data contracts exchanged between the spatial, semantic, metadata and viewer modules.

Read this before producing or consuming any artifact.

World frame
    Every position in every artifact is in the *world* frame defined by
    ``SceneManifest``:  p_world = SceneManifest.source_to_world @ p_source,
    where p_source are the raw PLY vertex coordinates. Only preprocessing writes
    the manifest. No other pipeline may rescale, recenter or re-orient
    coordinates on its own; ask preprocessing for a new manifest instead.

    The target world frame is right-handed, +Z up and in meters. Until
    preprocessing has verified this, the manifest says so explicitly
    (``up_axis=None``, ``units="unknown"``, ``verified=False``) and the transform
    is identity.

Bounding boxes
    One convention: ``OrientedBox`` = world-space ``center``, full side lengths
    ``extent`` along the box's local x/y/z axes, and a 3x3 ``rotation`` whose
    columns are those local axes in world coordinates. An axis-aligned box is
    simply ``rotation = identity``. This matches ``open3d.geometry.OrientedBoundingBox``
    (center, R, extent).

Versioning
    Every model carries ``schema_version``. Bump ``SCHEMA_VERSION`` when a
    change breaks existing artifacts. Models forbid unknown fields so typos fail loudly.
"""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from valmet.common.geometry import Matrix4, apply_transform, as_array4, identity4, is_rotation

SCHEMA_VERSION = "1.0"

Vec3 = tuple[float, float, float]
Id = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:\-]+$")]
Axis = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]


def _check_matrix4(m: Matrix4) -> Matrix4:
    as_array4(m)
    return m


def _check_rotation(r: list[list[float]]) -> list[list[float]]:
    if not is_rotation(r):
        raise ValueError("rotation must be a proper 3x3 rotation matrix (orthonormal, det=+1)")
    return r


Transform4 = Annotated[Matrix4, AfterValidator(_check_matrix4)]
Rotation3 = Annotated[list[list[float]], AfterValidator(_check_rotation)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["1.0"] = SCHEMA_VERSION


# --------------------------------------------------------------------------- scene


class CoordinateSystem(BaseModel):
    """What is known about the world frame. Unknowns stay explicitly unknown."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    handedness: Literal["right"] = "right"
    up_axis: Axis | None = Field(None, description="World up axis; None until verified.")
    units: Literal["meters", "unknown"] = "unknown"
    meters_per_unit: float | None = Field(
        None, gt=0, description="World units -> meters; None if scale unknown."
    )
    verified: bool = Field(False, description="True once preprocessing checked up axis and scale.")
    notes: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> CoordinateSystem:
        if self.units == "meters" and self.meters_per_unit not in (None, 1.0):
            raise ValueError("units='meters' implies meters_per_unit == 1.0")
        if self.verified and (self.up_axis is None or self.units == "unknown"):
            raise ValueError("verified=True requires a known up_axis and units")
        return self


class SceneManifest(Contract):
    scene_id: Id
    source_filename: str = Field(min_length=1)
    source_sha256: str | None = Field(None, pattern=r"^[0-9a-f]{64}$")
    source_vertex_count: int | None = Field(None, ge=0)
    coordinate_system: CoordinateSystem = CoordinateSystem()
    source_to_world: Transform4 = Field(default_factory=identity4)

    def to_world(self, source_points: np.ndarray) -> np.ndarray:
        """Map raw PLY coordinates (N, 3) into the world frame."""
        return apply_transform(self.source_to_world, source_points)


# --------------------------------------------------------------------------- semantic


class OrientedBox(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    center: Vec3
    extent: tuple[
        Annotated[float, Field(ge=0)], Annotated[float, Field(ge=0)], Annotated[float, Field(ge=0)]
    ] = Field(description="Full side lengths along local x, y, z (not half-sizes).")
    rotation: Rotation3 = Field(
        default_factory=lambda: np.eye(3).tolist(),
        description="Columns = box local axes in world coordinates. Identity = axis-aligned.",
    )

    def corners(self) -> np.ndarray:
        """(8, 3) world-space corners. Bit i of the corner index: 0=-half, 1=+half on axis i."""
        half = np.asarray(self.extent) / 2.0
        signs = np.array([[(i >> a) & 1 for a in range(3)] for i in range(8)]) * 2 - 1
        return np.asarray(self.center) + (signs * half) @ np.asarray(self.rotation).T


class GaussianMembership(BaseModel):
    """Reference to the source-PLY vertex indices belonging to one object."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    indices_file: str = Field(
        min_length=1, description="Path (relative to the scene artifact dir) to a 1-D int .npy."
    )
    count: int = Field(ge=0)


class DetectedObject(Contract):
    object_id: Id
    scene_id: Id
    label: str | None = Field(None, description="Semantic label; None = unknown/unclassified.")
    confidence: float | None = Field(None, ge=0.0, le=1.0)
    center: Vec3 = Field(description="Representative world point, e.g. Gaussian centroid.")
    bbox: OrientedBox
    gaussians: GaussianMembership | None = None

    @model_validator(mode="after")
    def _confidence_needs_label(self) -> DetectedObject:
        if self.label is None and self.confidence is not None:
            raise ValueError("confidence given without a label")
        return self


# --------------------------------------------------------------------------- spatial


class CellValues(BaseModel):
    """uint8 values stored in the occupancy grid file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    free: int = Field(0, ge=0, le=255)
    occupied: int = Field(1, ge=0, le=255)
    unknown: int = Field(255, ge=0, le=255)

    @model_validator(mode="after")
    def _distinct(self) -> CellValues:
        if len({self.free, self.occupied, self.unknown}) != 3:
            raise ValueError("free/occupied/unknown values must be distinct")
        return self


class RobotClearance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    radius: float = Field(gt=0, description="Footprint radius in world units.")
    height: float = Field(gt=0, description="Body height in world units.")
    max_step_height: float | None = Field(None, ge=0)


class NavigationMapMetadata(Contract):
    """2-D occupancy grid on the world XY plane (requires a +Z-up world).

    Cell (ix, iy) of the array ``grid[ix, iy]`` covers world
    x in [origin.x + ix*res, origin.x + (ix+1)*res), likewise for y.
    ``origin`` is the world position of the outer corner of cell (0, 0);
    origin.z is the floor height the grid was computed for.
    """

    scene_id: Id
    origin: Vec3
    resolution: float = Field(gt=0, description="Cell edge length in world units.")
    shape: tuple[Annotated[int, Field(gt=0)], Annotated[int, Field(gt=0)]] = Field(
        description="(nx, ny) = grid array shape."
    )
    grid_file: str = Field("occupancy.npy", min_length=1)
    cell_values: CellValues = CellValues()
    clearance: RobotClearance
    obstacles_inflated: bool = Field(
        True, description="True if occupied cells already include the clearance radius."
    )

    def world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))[:, :2]
        return np.floor((xy - np.asarray(self.origin[:2])) / self.resolution).astype(np.int64)

    def cell_to_world(self, ij: np.ndarray) -> np.ndarray:
        """Cell centers in world XY."""
        ij = np.atleast_2d(np.asarray(ij, dtype=np.float64))
        return np.asarray(self.origin[:2]) + (ij + 0.5) * self.resolution


class NavigationPath(Contract):
    scene_id: Id
    start: Vec3
    target_object_id: Id
    inspection_position: Vec3 = Field(description="Reachable stand-off point near the target.")
    waypoints: list[Vec3] = Field(min_length=1, description="Ordered, start -> inspection point.")

    @model_validator(mode="after")
    def _endpoints(self) -> NavigationPath:
        if not np.allclose(self.waypoints[0], self.start, atol=1e-6):
            raise ValueError("waypoints[0] must equal start")
        if not np.allclose(self.waypoints[-1], self.inspection_position, atol=1e-6):
            raise ValueError("waypoints[-1] must equal inspection_position")
        return self

    def length(self) -> float:
        pts = np.asarray(self.waypoints)
        return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


# --------------------------------------------------------------------------- metadata


class AssetMapping(Contract):
    scene_id: Id
    scene_object_id: Id
    equipment_id: str = Field(min_length=1, description="Equipment ID as written in the Excel.")
    method: Literal["matched", "random", "procedural"] = Field(
        description="matched = evidence-based; random/procedural = demo assignment."
    )
