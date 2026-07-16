"""Shared camera contracts and lazily loaded P1 pinhole geometry."""

from __future__ import annotations

from .camera import (
    CameraObservation,
    CoordinateConvention,
    DepthDefinition,
    PixelCenterConvention,
    TransformConvention,
)

__all__ = [
    "CameraObservation",
    "CoordinateConvention",
    "DepthDefinition",
    "PixelCenterConvention",
    "TransformConvention",
    "backproject_pixel",
    "backproject_points",
    "camera_center_world",
    "camera_to_world",
    "project_point",
    "project_points",
    "transform_points",
    "world_to_camera",
]


def __getattr__(name: str) -> object:
    """Load point geometry lazily to avoid a contract import cycle."""

    if name in {"backproject_pixel", "backproject_points"}:
        from . import backprojection

        return getattr(backprojection, name)
    if name in {"project_point", "project_points"}:
        from . import projection

        return getattr(projection, name)
    if name in {
        "camera_center_world",
        "camera_to_world",
        "transform_points",
        "world_to_camera",
    }:
        from . import transforms

        return getattr(transforms, name)
    raise AttributeError(f"module 'semantic3d.geometry' has no attribute {name!r}")
