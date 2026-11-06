"""Shared contracts, coordinate conventions and artifact I/O. The only cross-module import."""

from valmet.common.contracts import (
    SCHEMA_VERSION,
    AssetMapping,
    CellValues,
    CoordinateSystem,
    DetectedObject,
    GaussianMembership,
    NavigationMapMetadata,
    NavigationPath,
    OrientedBox,
    RobotClearance,
    SceneManifest,
)

__all__ = [
    "SCHEMA_VERSION",
    "AssetMapping",
    "CellValues",
    "CoordinateSystem",
    "DetectedObject",
    "GaussianMembership",
    "NavigationMapMetadata",
    "NavigationPath",
    "OrientedBox",
    "RobotClearance",
    "SceneManifest",
]
