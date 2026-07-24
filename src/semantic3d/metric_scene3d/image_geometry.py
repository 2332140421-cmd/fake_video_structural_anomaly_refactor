"""Explicit source-image to model-array coordinate mappings."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..geometry.camera import validate_intrinsics


@dataclass(frozen=True)
class ImageCoordinateTransform:
    """Affine image mapping for resize, crop, padding, or letterbox.

    The mapping is ``target = scale * (source - crop) + padding``. Both axes
    use integer pixel-centre coordinates without an implicit half-pixel shift.
    """

    source_width: int
    source_height: int
    target_width: int
    target_height: int
    scale_x: float = 1.0
    scale_y: float = 1.0
    crop_x: float = 0.0
    crop_y: float = 0.0
    pad_x: float = 0.0
    pad_y: float = 0.0
    operation: str = "identity"
    distortion_status: str = "not_corrected"

    def __post_init__(self) -> None:
        if min(
            self.source_width,
            self.source_height,
            self.target_width,
            self.target_height,
        ) <= 0:
            raise ValueError("Image dimensions must be positive.")
        if self.scale_x <= 0.0 or self.scale_y <= 0.0:
            raise ValueError("Image transform scales must be positive.")

    @classmethod
    def identity(cls, width: int, height: int) -> "ImageCoordinateTransform":
        """Return a source-to-target identity transform."""

        return cls(width, height, width, height)

    @property
    def matrix(self) -> np.ndarray:
        """Return the 3x3 homogeneous source-to-target pixel transform."""

        return np.asarray(
            [
                [self.scale_x, 0.0, self.pad_x - self.scale_x * self.crop_x],
                [0.0, self.scale_y, self.pad_y - self.scale_y * self.crop_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def source_to_target(self, pixels: np.ndarray) -> np.ndarray:
        """Map ``[N,2]`` source pixels into target coordinates."""

        array = np.asarray(pixels, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("pixels must have shape [N, 2].")
        homogeneous = np.column_stack((array, np.ones(len(array), dtype=float)))
        mapped = (self.matrix @ homogeneous.T).T
        return mapped[:, :2] / mapped[:, 2:3]

    def target_to_source(self, pixels: np.ndarray) -> np.ndarray:
        """Map ``[N,2]`` target pixels back to source coordinates."""

        array = np.asarray(pixels, dtype=float)
        if array.ndim != 2 or array.shape[1] != 2:
            raise ValueError("pixels must have shape [N, 2].")
        homogeneous = np.column_stack((array, np.ones(len(array), dtype=float)))
        mapped = (np.linalg.inv(self.matrix) @ homogeneous.T).T
        return mapped[:, :2] / mapped[:, 2:3]

    def transform_intrinsics(self, K_source: np.ndarray) -> np.ndarray:
        """Transform source-image intrinsics into target-array coordinates."""

        return validate_intrinsics(self.matrix @ validate_intrinsics(K_source))


def align_binary_mask(
    mask: np.ndarray,
    *,
    mask_transform: ImageCoordinateTransform,
    target_transform: ImageCoordinateTransform,
) -> np.ndarray:
    """Warp a binary mask into a depth/target array with nearest-neighbour sampling."""

    binary = np.asarray(mask, dtype=bool)
    if binary.shape != (mask_transform.target_height, mask_transform.target_width):
        raise ValueError("mask shape does not match mask_transform target dimensions.")
    mask_to_target = target_transform.matrix @ np.linalg.inv(mask_transform.matrix)
    aligned = cv2.warpPerspective(
        binary.astype(np.uint8),
        mask_to_target,
        (target_transform.target_width, target_transform.target_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned.astype(bool)


def validate_transform_coverage(transform: ImageCoordinateTransform) -> tuple[bool, str]:
    """Check whether the target samples map inside the declared source image."""

    target_corners = np.asarray(
        [
            [0.0, 0.0],
            [transform.target_width - 1.0, 0.0],
            [0.0, transform.target_height - 1.0],
            [transform.target_width - 1.0, transform.target_height - 1.0],
        ]
    )
    source = transform.target_to_source(target_corners)
    finite = np.isfinite(source).all()
    if not finite:
        return False, "non_finite_image_transform"
    return True, ""
