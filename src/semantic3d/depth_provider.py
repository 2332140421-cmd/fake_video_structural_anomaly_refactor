"""Depth provider interfaces and bbox-level depth aggregation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional, Sequence, Union

import cv2
import numpy as np

PathLike = Union[str, Path]
DepthMethod = Literal["median", "mean"]


class BaseDepthProvider(ABC):
    """Interface for frame-level depth estimation providers."""

    @abstractmethod
    def predict_depth(self, frame_path: PathLike) -> np.ndarray:
        """Return a HxW depth map for the input frame image."""


class MockDepthProvider(BaseDepthProvider):
    """Generate a deterministic relative depth map without a real model.

    The map combines a vertical and horizontal gradient. It is useful for
    testing that object bbox regions receive different median depths.
    """

    def __init__(
        self,
        min_depth: float = 1.0,
        max_depth: float = 10.0,
        direction: Literal["vertical", "horizontal", "diagonal"] = "diagonal",
    ) -> None:
        """Create a mock depth provider."""

        if min_depth <= 0:
            raise ValueError(f"min_depth must be > 0, got {min_depth}.")
        if max_depth <= min_depth:
            raise ValueError(
                f"max_depth must be > min_depth, got {max_depth} <= {min_depth}."
            )
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.direction = direction

    def predict_depth(self, frame_path: PathLike) -> np.ndarray:
        """Return a smooth relative depth map with the same HxW as the image."""

        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read frame image with OpenCV: {frame_path}")
        height, width = image.shape[:2]

        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        if self.direction == "vertical":
            gradient = np.repeat(y, width, axis=1)
        elif self.direction == "horizontal":
            gradient = np.repeat(x, height, axis=0)
        elif self.direction == "diagonal":
            gradient = (np.repeat(y, width, axis=1) + np.repeat(x, height, axis=0)) / 2.0
        else:
            raise ValueError(
                "direction must be one of 'vertical', 'horizontal', or 'diagonal'."
            )

        return self.min_depth + gradient * (self.max_depth - self.min_depth)


class RealDepthProvider(BaseDepthProvider):
    """Placeholder adapter for a future local depth-estimation model."""

    def __init__(self, model: Optional[object] = None) -> None:
        """Create a real depth provider wrapper.

        A concrete implementation can inject Depth Anything, MiDaS, or another
        local model object here later.
        """

        self.model = model
        if self.model is None:
            raise RuntimeError(
                "RealDepthProvider is unavailable. Please install depth model "
                "dependencies or use mock_depth."
            )

    def predict_depth(self, frame_path: PathLike) -> np.ndarray:
        """Predict a depth map using the injected model."""

        if not hasattr(self.model, "predict_depth"):
            raise RuntimeError(
                "RealDepthProvider is unavailable. Please install depth model "
                "dependencies or use mock_depth."
            )
        depth = self.model.predict_depth(frame_path)  # type: ignore[attr-defined]
        depth_map = np.asarray(depth, dtype=float)
        if depth_map.ndim != 2:
            raise ValueError(f"Depth model must return HxW depth, got {depth_map.shape}.")
        return depth_map


def compute_object_depth_from_bbox(
    depth_map: np.ndarray,
    bbox: Optional[Sequence[float]],
    method: DepthMethod = "median",
    default_depth: float = 5.0,
) -> float:
    """Aggregate object depth from a bbox region in a frame-level depth map."""

    if default_depth <= 0:
        raise ValueError(f"default_depth must be > 0, got {default_depth}.")

    depth_array = np.asarray(depth_map, dtype=float)
    if depth_array.ndim != 2:
        raise ValueError(f"depth_map must be HxW, got shape {depth_array.shape}.")
    if bbox is None or len(bbox) != 4:
        return float(default_depth)

    height, width = depth_array.shape
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1_i = max(0, min(width, int(np.floor(x1))))
    y1_i = max(0, min(height, int(np.floor(y1))))
    x2_i = max(0, min(width, int(np.ceil(x2))))
    y2_i = max(0, min(height, int(np.ceil(y2))))

    if x2_i <= x1_i or y2_i <= y1_i:
        return float(default_depth)

    region = depth_array[y1_i:y2_i, x1_i:x2_i]
    valid = region[np.isfinite(region)]
    if valid.size == 0:
        return float(default_depth)

    if method == "median":
        return float(np.median(valid))
    if method == "mean":
        return float(np.mean(valid))
    raise ValueError("method must be 'median' or 'mean'.")
