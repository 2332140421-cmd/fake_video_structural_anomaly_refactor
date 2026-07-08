"""Depth provider interfaces and bbox-level depth aggregation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union

import cv2
import numpy as np
from PIL import Image

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
    """Depth-estimation provider backed by ``transformers.pipeline``.

    Project convention: larger depth values mean objects are farther away.
    Some monocular models output inverse depth or disparity, where larger values
    mean closer. In that case, create this provider with ``invert_depth=True`` or
    pass ``--invert_depth`` in the CLI. The provider does not guess the model's
    direction automatically.
    """

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Small",
        device: str = "cpu",
        normalize: bool = True,
        invert_depth: bool = False,
        min_depth_value: float = 1e-6,
        pipeline_instance: Optional[Any] = None,
    ) -> None:
        """Load a real monocular depth-estimation pipeline.

        Args:
            model_name: Hugging Face model id or local model path.
            device: Device passed to transformers, such as ``cpu`` or ``cuda:0``.
            normalize: If true, linearly normalize output to the stable positive
                range [1, 10]. R_sd only uses depth ratios, so this keeps values
                well-conditioned without claiming metric depth.
            invert_depth: Reverse output direction before normalization. Use this
                for models that output inverse depth/disparity.
            min_depth_value: Lower positive clamp for non-normalized outputs.
            pipeline_instance: Optional injected pipeline-like callable for tests.
        """

        if min_depth_value <= 0:
            raise ValueError(
                f"min_depth_value must be > 0, got {min_depth_value}."
            )
        self.model_name = model_name
        self.resolved_model_name = _resolve_transformers_depth_model_name(model_name)
        self.device = device
        self.normalize = normalize
        self.invert_depth = invert_depth
        self.min_depth_value = float(min_depth_value)
        self.pipeline = pipeline_instance or self._load_pipeline(
            self.resolved_model_name, device
        )

    @staticmethod
    def _load_pipeline(model_name: str, device: str) -> Any:
        """Load the transformers depth-estimation pipeline."""

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "RealDepthProvider requires transformers and pillow. Install them "
                "with: pip install transformers pillow"
            ) from exc

        try:
            return pipeline("depth-estimation", model=model_name, device=device)
        except Exception as exc:
            raise RuntimeError(
                "RealDepthProvider could not load the depth-estimation model "
                f"{model_name!r}. Check network access, model name, local cache, "
                "or use --depth_provider mock_depth."
            ) from exc

    def predict_depth(self, frame_path: PathLike) -> np.ndarray:
        """Predict a HxW relative depth map for a frame image."""

        path = Path(frame_path)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Could not read frame image with PIL: {path}") from exc

        width, height = image.size
        try:
            result = self.pipeline(image)
        except Exception as exc:
            raise RuntimeError(
                f"Depth-estimation pipeline failed for frame {path}: {exc}"
            ) from exc

        raw_depth = _extract_depth_output(result)
        depth_map = _to_numpy_depth(raw_depth)
        if depth_map.ndim != 2:
            raise ValueError(f"Depth model must return HxW depth, got {depth_map.shape}.")
        depth_map = _resize_depth(depth_map, width=width, height=height)
        depth_map = _sanitize_depth(depth_map, min_depth_value=self.min_depth_value)

        if self.invert_depth:
            finite = depth_map[np.isfinite(depth_map)]
            if finite.size:
                depth_map = float(finite.max()) + float(finite.min()) - depth_map
                depth_map = _sanitize_depth(
                    depth_map, min_depth_value=self.min_depth_value
                )

        if self.normalize:
            depth_map = _normalize_depth(depth_map, min_value=1.0, max_value=10.0)

        return depth_map.astype(np.float32, copy=False)


def _extract_depth_output(result: Any) -> Any:
    """Get the depth tensor/PIL image from a transformers pipeline result."""

    if isinstance(result, dict):
        if "predicted_depth" in result:
            return result["predicted_depth"]
        if "depth" in result:
            return result["depth"]
    raise ValueError(
        "Depth-estimation pipeline output must contain 'predicted_depth' or 'depth'."
    )


def _resolve_transformers_depth_model_name(model_name: str) -> str:
    """Map common Depth Anything names to the Transformers-compatible id."""

    aliases = {
        "depth-anything/Depth-Anything-V2-Small": (
            "depth-anything/Depth-Anything-V2-Small-hf"
        ),
    }
    return aliases.get(model_name, model_name)


def _to_numpy_depth(raw_depth: Any) -> np.ndarray:
    """Convert a tensor, PIL image, or array-like depth output to a numpy array."""

    if hasattr(raw_depth, "detach"):
        raw_depth = raw_depth.detach().cpu().numpy()
    elif isinstance(raw_depth, Image.Image):
        raw_depth = np.asarray(raw_depth)
    depth_map = np.asarray(raw_depth, dtype=np.float32)
    depth_map = np.squeeze(depth_map)
    return depth_map


def _resize_depth(depth_map: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize depth output to the original frame size."""

    if depth_map.shape == (height, width):
        return depth_map.astype(np.float32, copy=False)
    return cv2.resize(
        depth_map.astype(np.float32, copy=False),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )


def _sanitize_depth(depth_map: np.ndarray, min_depth_value: float) -> np.ndarray:
    """Replace NaN/inf and clamp depth to positive finite values."""

    depth = np.asarray(depth_map, dtype=np.float32)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.full(depth.shape, min_depth_value, dtype=np.float32)
    fill = float(np.median(finite))
    depth = np.nan_to_num(depth, nan=fill, posinf=fill, neginf=fill)
    return np.maximum(depth, min_depth_value).astype(np.float32, copy=False)


def _normalize_depth(
    depth_map: np.ndarray,
    min_value: float = 1.0,
    max_value: float = 10.0,
) -> np.ndarray:
    """Normalize a finite depth map to a stable positive range."""

    depth = np.asarray(depth_map, dtype=np.float32)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return np.full(depth.shape, min_value, dtype=np.float32)
    d_min = float(finite.min())
    d_max = float(finite.max())
    if d_max <= d_min:
        return np.full(depth.shape, (min_value + max_value) / 2.0, dtype=np.float32)
    normalized = (depth - d_min) / (d_max - d_min)
    return (min_value + normalized * (max_value - min_value)).astype(np.float32)


def save_depth_visualization(depth_map: np.ndarray, output_path: PathLike) -> Path:
    """Save a color PNG visualization for a depth map."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    depth = _normalize_depth(np.asarray(depth_map, dtype=np.float32), 0.0, 255.0)
    depth_uint8 = np.clip(depth, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
    cv2.imwrite(str(output), colored)
    return output


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
