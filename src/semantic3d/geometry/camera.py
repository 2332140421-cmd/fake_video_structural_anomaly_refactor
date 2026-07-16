"""Camera calibration and pose contracts for shared 3D geometry.

The canonical camera convention is OpenCV-like: camera X points right, Y
points down, and Z points forward. Image coordinates use ``u`` to the right
and ``v`` down. Integer pixel indices denote pixel centres; geometry functions
therefore do not add an implicit 0.5 offset. Depth is optical-axis Z depth,
not Euclidean range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional

import numpy as np

from ..validity import MissingReason


class CoordinateConvention(str, Enum):
    """Supported explicit camera-coordinate conventions."""

    OPENCV = "opencv_x_right_y_down_z_forward"
    UNKNOWN = "unknown"


class DepthDefinition(str, Enum):
    """Meaning of a positive depth value used by projection geometry."""

    Z_DEPTH = "camera_optical_axis_z"
    UNKNOWN = "unknown"


class TransformConvention(str, Enum):
    """How rigid transforms act on homogeneous points."""

    COLUMN_VECTOR = "column_vector_left_multiply"
    UNKNOWN = "unknown"


class PixelCenterConvention(str, Enum):
    """Image-sampling convention for pixel coordinates."""

    INTEGER_CENTERS = "integer_coordinates_are_pixel_centers_no_half_offset"
    UNKNOWN = "unknown"


def _array_or_none(
    value: Optional[np.ndarray], shape: tuple[int, ...], name: str
) -> Optional[np.ndarray]:
    """Convert an optional matrix and validate its shape and finite values."""

    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def validate_intrinsics(K: np.ndarray) -> np.ndarray:
    """Validate and return a pinhole camera intrinsic matrix."""

    matrix = _array_or_none(K, (3, 3), "K")
    assert matrix is not None
    if np.allclose(matrix, np.eye(3), atol=1e-12):
        raise ValueError("A 3x3 identity matrix cannot be valid camera intrinsics.")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("Camera focal lengths fx and fy must be positive.")
    if abs(matrix[2, 2]) <= 1e-12 or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError("Camera intrinsic matrix K must be invertible.")
    if not np.allclose(matrix[2], np.asarray([0.0, 0.0, 1.0]), atol=1e-9):
        raise ValueError("Only standard pinhole K with bottom row [0, 0, 1] is supported.")
    return matrix


def validate_rigid_transform(transform: np.ndarray, name: str) -> np.ndarray:
    """Validate a finite 4x4 homogeneous rigid transform."""

    matrix = _array_or_none(transform, (4, 4), name)
    assert matrix is not None
    if not np.allclose(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
        raise ValueError(f"{name} must have homogeneous bottom row [0, 0, 0, 1].")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation block must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation block must have determinant +1.")
    return matrix


@dataclass(frozen=True)
class CameraObservation:
    """Camera intrinsics and optional pose with explicit validity semantics.

    The stored ``T_world_camera`` and ``T_camera_world`` names are retained for
    backward-compatible construction. New code must use the unambiguous
    :attr:`T_world_from_camera` and :attr:`T_camera_from_world` properties.

    ``valid`` means the intrinsics and coordinate semantics are usable for
    camera-frame geometry. Pose is optional and reported separately through
    :attr:`pose_valid`, allowing camera-frame reconstruction without pretending
    that world-frame coordinates are available.
    """

    K: Optional[np.ndarray]
    distortion: Optional[np.ndarray]
    T_world_camera: Optional[np.ndarray]
    T_camera_world: Optional[np.ndarray]
    image_width: int
    image_height: int
    coordinate_convention: CoordinateConvention | str
    intrinsics_source: str
    pose_source: str
    valid: bool
    quality: float
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    depth_definition: DepthDefinition | str = DepthDefinition.Z_DEPTH
    transform_convention: TransformConvention | str = TransformConvention.COLUMN_VECTOR
    pixel_center_convention: PixelCenterConvention | str = PixelCenterConvention.INTEGER_CENTERS

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("Camera image_width and image_height must be positive.")
        quality = float(self.quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("CameraObservation.quality must be finite and in [0, 1].")
        convention = CoordinateConvention(self.coordinate_convention)
        depth_definition = DepthDefinition(self.depth_definition)
        transform_convention = TransformConvention(self.transform_convention)
        pixel_convention = PixelCenterConvention(self.pixel_center_convention)
        k = _array_or_none(self.K, (3, 3), "K")
        if k is not None:
            k = validate_intrinsics(k)

        twc = _array_or_none(self.T_world_camera, (4, 4), "T_world_from_camera")
        tcw = _array_or_none(self.T_camera_world, (4, 4), "T_camera_from_world")
        pose_error = ""
        try:
            if twc is not None:
                twc = validate_rigid_transform(twc, "T_world_from_camera")
            if tcw is not None:
                tcw = validate_rigid_transform(tcw, "T_camera_from_world")
            if twc is None and tcw is not None:
                twc = np.linalg.inv(tcw)
            elif tcw is None and twc is not None:
                tcw = np.linalg.inv(twc)
            elif twc is not None and tcw is not None and not (
                np.allclose(twc @ tcw, np.eye(4), atol=1e-6)
                and np.allclose(tcw @ twc, np.eye(4), atol=1e-6)
            ):
                pose_error = "inconsistent_camera_transforms"
        except (ValueError, np.linalg.LinAlgError):
            pose_error = "invalid_camera_transform"

        distortion = None
        if self.distortion is not None:
            distortion = np.asarray(self.distortion, dtype=float).reshape(-1)
            if not np.isfinite(distortion).all():
                raise ValueError("distortion must contain only finite values.")

        valid = bool(self.valid)
        missing_reason = str(self.missing_reason)
        if valid:
            if k is None:
                raise ValueError("Valid CameraObservation requires K.")
            if convention != CoordinateConvention.OPENCV:
                raise ValueError("Only the OpenCV camera coordinate convention is supported.")
            if depth_definition != DepthDefinition.Z_DEPTH:
                raise ValueError("Only optical-axis Z depth is supported.")
            if transform_convention != TransformConvention.COLUMN_VECTOR:
                raise ValueError("Only column-vector left-multiply transforms are supported.")
            if pixel_convention != PixelCenterConvention.INTEGER_CENTERS:
                raise ValueError("Only integer-coordinate pixel centres are supported.")
            if not self.intrinsics_source.strip():
                raise ValueError("Valid CameraObservation requires intrinsics provenance.")
            if self.intrinsics_source.strip().lower() == "approximate" and quality >= 1.0:
                raise ValueError("Approximate intrinsics must use quality < 1.")
            if missing_reason:
                raise ValueError("Valid CameraObservation cannot have missing_reason.")
            if pose_error:
                valid = False
                missing_reason = pose_error
        elif not missing_reason:
            missing_reason = pose_error
            if not missing_reason:
                raise ValueError("Invalid CameraObservation requires missing_reason.")

        object.__setattr__(self, "K", k)
        object.__setattr__(self, "distortion", distortion)
        object.__setattr__(self, "T_world_camera", twc)
        object.__setattr__(self, "T_camera_world", tcw)
        object.__setattr__(self, "coordinate_convention", convention)
        object.__setattr__(self, "depth_definition", depth_definition)
        object.__setattr__(self, "transform_convention", transform_convention)
        object.__setattr__(self, "pixel_center_convention", pixel_convention)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "missing_reason", missing_reason)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def T_world_from_camera(self) -> Optional[np.ndarray]:
        """Return the transform mapping camera points into the world frame."""

        return self.T_world_camera

    @property
    def T_camera_from_world(self) -> Optional[np.ndarray]:
        """Return the transform mapping world points into the camera frame."""

        return self.T_camera_world

    @property
    def pose_valid(self) -> bool:
        """Return whether a mutually consistent camera/world pose is available."""

        return bool(
            self.valid
            and self.T_world_from_camera is not None
            and self.T_camera_from_world is not None
        )

    @classmethod
    def from_parameters(
        cls,
        *,
        K: np.ndarray,
        image_width: int,
        image_height: int,
        intrinsics_source: str,
        quality: float,
        distortion: Optional[np.ndarray] = None,
        T_world_from_camera: Optional[np.ndarray] = None,
        T_camera_from_world: Optional[np.ndarray] = None,
        pose_source: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "CameraObservation":
        """Construct a camera using direction-explicit transform names."""

        return cls(
            K=K,
            distortion=distortion,
            T_world_camera=T_world_from_camera,
            T_camera_world=T_camera_from_world,
            image_width=image_width,
            image_height=image_height,
            coordinate_convention=CoordinateConvention.OPENCV,
            intrinsics_source=intrinsics_source,
            pose_source=pose_source,
            valid=True,
            quality=quality,
            metadata=dict(metadata or {}),
            depth_definition=DepthDefinition.Z_DEPTH,
            transform_convention=TransformConvention.COLUMN_VECTOR,
            pixel_center_convention=PixelCenterConvention.INTEGER_CENTERS,
        )

    @classmethod
    def missing(
        cls,
        image_width: int,
        image_height: int,
        reason: MissingReason | str = MissingReason.MISSING_CAMERA,
        metadata: Mapping[str, Any] | None = None,
    ) -> "CameraObservation":
        """Create an invalid camera record without fake identity calibration."""

        reason_value = reason.value if isinstance(reason, MissingReason) else str(reason)
        return cls(
            K=None,
            distortion=None,
            T_world_camera=None,
            T_camera_world=None,
            image_width=image_width,
            image_height=image_height,
            coordinate_convention=CoordinateConvention.UNKNOWN,
            intrinsics_source="",
            pose_source="",
            valid=False,
            quality=0.0,
            missing_reason=reason_value,
            metadata=dict(metadata or {}),
            depth_definition=DepthDefinition.UNKNOWN,
            transform_convention=TransformConvention.UNKNOWN,
            pixel_center_convention=PixelCenterConvention.UNKNOWN,
        )
