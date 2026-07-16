"""Depth provider interfaces and bbox-level depth aggregation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import cv2
import numpy as np
from PIL import Image

PathLike = Union[str, Path]
DepthMethod = Literal["median", "mean"]


class DepthRepresentation(str, Enum):
    """Semantic representation of values in a depth observation."""

    METRIC_DEPTH = "metric_depth"
    RELATIVE_DEPTH = "relative_depth"
    INVERSE_DEPTH = "inverse_depth"
    DISPARITY_LIKE = "disparity_like"
    UNKNOWN = "unknown"


class DepthScaleStatus(str, Enum):
    """Calibration scope of a depth observation."""

    METRIC_CALIBRATED = "metric_calibrated"
    RELATIVE_SHARED_SEQUENCE = "relative_shared_sequence"
    RELATIVE_PER_FRAME = "relative_per_frame"
    UNKNOWN = "unknown"


class LargerValueMeans(str, Enum):
    """Direction represented by increasing depth-map values."""

    FARTHER = "farther"
    CLOSER = "closer"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DepthObservation:
    """Canonical frame-level depth evidence for future geometry modules.

    ``depth_map`` is the non-visualization numeric representation. New geometry
    code must call :meth:`require_geometry_depth` before using it. The legacy
    per-frame [1, 10] visualization/compatibility array is stored separately and
    is never accepted as geometry depth.
    """

    depth_map: Optional[np.ndarray]
    raw_model_output: Optional[np.ndarray]
    visualization_depth: Optional[np.ndarray]
    depth_representation: DepthRepresentation | str
    scale_status: DepthScaleStatus | str
    larger_value_means: LargerValueMeans | str
    valid_mask: Optional[np.ndarray]
    confidence_map: Optional[np.ndarray]
    provider_name: str
    frame_index: Optional[int]
    valid: bool
    quality: float
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        representation = DepthRepresentation(self.depth_representation)
        scale_status = DepthScaleStatus(self.scale_status)
        direction = LargerValueMeans(self.larger_value_means)
        quality = float(self.quality)
        if not np.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("DepthObservation.quality must be finite and in [0, 1].")

        depth_map = None if self.depth_map is None else np.asarray(self.depth_map, dtype=np.float32)
        raw = (
            None
            if self.raw_model_output is None
            else np.asarray(self.raw_model_output, dtype=np.float32)
        )
        visualization = (
            None
            if self.visualization_depth is None
            else np.asarray(self.visualization_depth, dtype=np.float32)
        )
        valid_mask = (
            None if self.valid_mask is None else np.asarray(self.valid_mask, dtype=bool)
        )
        confidence = (
            None
            if self.confidence_map is None
            else np.asarray(self.confidence_map, dtype=np.float32)
        )
        if depth_map is not None and depth_map.ndim != 2:
            raise ValueError(f"depth_map must be HxW, got {depth_map.shape}.")
        if valid_mask is not None and depth_map is not None and valid_mask.shape != depth_map.shape:
            raise ValueError("valid_mask must have the same shape as depth_map.")
        if confidence is not None and depth_map is not None and confidence.shape != depth_map.shape:
            raise ValueError("confidence_map must have the same shape as depth_map.")
        if visualization is not None and depth_map is not None and visualization.shape != depth_map.shape:
            raise ValueError("visualization_depth must have the same shape as depth_map.")
        if self.valid:
            if depth_map is None or valid_mask is None or not np.any(valid_mask):
                raise ValueError("Valid DepthObservation requires depth_map and valid pixels.")
            if representation == DepthRepresentation.UNKNOWN:
                raise ValueError("Valid DepthObservation requires a known representation.")
            if scale_status == DepthScaleStatus.UNKNOWN:
                raise ValueError("Valid DepthObservation requires a known scale_status.")
            if direction == LargerValueMeans.UNKNOWN:
                raise ValueError("Valid DepthObservation requires a value direction.")
            if self.missing_reason:
                raise ValueError("Valid DepthObservation cannot have missing_reason.")
        elif not self.missing_reason:
            raise ValueError("Invalid DepthObservation requires missing_reason.")

        object.__setattr__(self, "depth_map", depth_map)
        object.__setattr__(self, "raw_model_output", raw)
        object.__setattr__(self, "visualization_depth", visualization)
        object.__setattr__(self, "depth_representation", representation)
        object.__setattr__(self, "scale_status", scale_status)
        object.__setattr__(self, "larger_value_means", direction)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "confidence_map", confidence)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def require_geometry_depth(self, require_metric: bool = False) -> np.ndarray:
        """Return usable depth or reject unknown/inverse/legacy representations."""

        if not self.valid or self.depth_map is None:
            raise ValueError(f"Depth observation is invalid: {self.missing_reason}.")
        if bool(self.metadata.get("legacy_normalized_depth", False)):
            raise ValueError("legacy_normalized_depth cannot be used for 3D geometry.")
        if self.depth_representation not in {
            DepthRepresentation.METRIC_DEPTH,
            DepthRepresentation.RELATIVE_DEPTH,
        }:
            raise ValueError(
                f"{self.depth_representation.value} must be converted to depth before geometry."
            )
        if self.larger_value_means != LargerValueMeans.FARTHER:
            raise ValueError(
                "Geometry Z depth must use larger_value_means='farther'; inverse or "
                "ambiguous outputs must be converted explicitly."
            )
        if require_metric and self.scale_status != DepthScaleStatus.METRIC_CALIBRATED:
            raise ValueError("Metric geometry requires metric_calibrated depth.")
        return self.depth_map

    @classmethod
    def missing(
        cls,
        provider_name: str,
        frame_index: Optional[int],
        reason: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "DepthObservation":
        """Create an invalid observation without a fabricated depth array."""

        return cls(
            depth_map=None,
            raw_model_output=None,
            visualization_depth=None,
            depth_representation=DepthRepresentation.UNKNOWN,
            scale_status=DepthScaleStatus.UNKNOWN,
            larger_value_means=LargerValueMeans.UNKNOWN,
            valid_mask=None,
            confidence_map=None,
            provider_name=provider_name,
            frame_index=frame_index,
            valid=False,
            quality=0.0,
            missing_reason=reason,
            metadata=dict(metadata or {}),
        )


class BaseDepthProvider(ABC):
    """Canonical frame-level depth provider interface.

    New geometry code uses ``predict_observation``. ``predict_depth`` remains a
    compatibility array API for existing 2D/2.5D experiments.
    """

    @abstractmethod
    def predict_observation(
        self, frame_path: PathLike, frame_index: Optional[int] = None
    ) -> DepthObservation:
        """Return semantic depth evidence for one frame."""

    def predict_depth(self, frame_path: PathLike) -> np.ndarray:
        """Legacy array adapter; subclasses may preserve older normalization."""

        return self.predict_observation(frame_path).require_geometry_depth()

    def legacy_depth_metadata(self) -> dict[str, Any]:
        """Describe the compatibility array returned by ``predict_depth``."""

        return {
            "interface": "predict_depth",
            "legacy": True,
            "legacy_normalized_depth": False,
        }


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

    def predict_observation(
        self, frame_path: PathLike, frame_index: Optional[int] = None
    ) -> DepthObservation:
        """Return deterministic sequence-consistent synthetic relative depth."""

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

        depth = self.min_depth + gradient * (self.max_depth - self.min_depth)
        valid_mask = np.isfinite(depth) & (depth > 0.0)
        return DepthObservation(
            depth_map=depth,
            raw_model_output=depth.copy(),
            visualization_depth=_normalize_depth(depth, 0.0, 1.0),
            depth_representation=DepthRepresentation.RELATIVE_DEPTH,
            scale_status=DepthScaleStatus.RELATIVE_SHARED_SEQUENCE,
            larger_value_means=LargerValueMeans.FARTHER,
            valid_mask=valid_mask,
            confidence_map=None,
            provider_name="mock_gradient_depth",
            frame_index=frame_index,
            valid=bool(np.any(valid_mask)),
            quality=float(np.mean(valid_mask)),
            missing_reason="" if np.any(valid_mask) else "no_valid_depth_pixels",
            metadata={"synthetic": True, "direction": self.direction},
        )

    def predict_depth(self, frame_path: PathLike) -> np.ndarray:
        """Return the legacy synthetic array used by existing tests."""

        observation = self.predict_observation(frame_path)
        assert observation.depth_map is not None
        return observation.depth_map

    def legacy_depth_metadata(self) -> dict[str, Any]:
        """Describe the synthetic relative-depth compatibility array."""

        return {
            "interface": "predict_depth",
            "legacy": True,
            "legacy_normalized_depth": False,
            "depth_representation": DepthRepresentation.RELATIVE_DEPTH.value,
            "scale_status": DepthScaleStatus.RELATIVE_SHARED_SEQUENCE.value,
            "larger_value_means": LargerValueMeans.FARTHER.value,
        }


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
        (
            self.raw_depth_representation,
            self.raw_larger_value_means,
        ) = _infer_model_depth_semantics(self.resolved_model_name)

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

    def _predict_raw(
        self, frame_path: PathLike
    ) -> tuple[np.ndarray, np.ndarray, int, int]:
        """Run the model once and return raw and resized model-space arrays."""

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
        raw_array = _to_numpy_depth(raw_depth)
        if raw_array.ndim != 2:
            raise ValueError(f"Depth model must return HxW depth, got {raw_array.shape}.")
        resized = _resize_depth(raw_array, width=width, height=height)
        return raw_array, resized, width, height

    def predict_observation(
        self, frame_path: PathLike, frame_index: Optional[int] = None
    ) -> DepthObservation:
        """Return raw model semantics and non-affine geometry depth separately.

        Depth Anything V2 is treated as relative inverse-depth-like output:
        larger raw predictions indicate closer regions. With ``invert_depth``
        enabled, canonical relative depth is produced by a reciprocal transform,
        not by the legacy per-frame affine [1, 10] normalization. No metric unit
        is claimed.
        """

        raw_array, resized, _, _ = self._predict_raw(frame_path)
        valid_mask = np.isfinite(resized) & (resized > self.min_depth_value)
        quality = float(np.mean(valid_mask)) if valid_mask.size else 0.0
        representation = self.raw_depth_representation
        direction = self.raw_larger_value_means
        geometry = np.full(resized.shape, np.nan, dtype=np.float32)
        if np.any(valid_mask):
            geometry[valid_mask] = resized[valid_mask]

        if (
            representation in {DepthRepresentation.INVERSE_DEPTH, DepthRepresentation.DISPARITY_LIKE}
            and self.invert_depth
        ):
            converted = np.full(resized.shape, np.nan, dtype=np.float32)
            converted[valid_mask] = 1.0 / np.maximum(
                resized[valid_mask], self.min_depth_value
            )
            geometry = converted
            representation = DepthRepresentation.RELATIVE_DEPTH
            direction = LargerValueMeans.FARTHER

        known = representation != DepthRepresentation.UNKNOWN
        valid = bool(np.any(valid_mask) and known)
        missing_reason = "" if valid else "unknown_depth_representation"
        visualization_source = geometry if np.any(np.isfinite(geometry)) else resized
        visualization = _normalize_depth(visualization_source, 0.0, 1.0)
        return DepthObservation(
            depth_map=geometry,
            raw_model_output=raw_array,
            visualization_depth=visualization,
            depth_representation=representation,
            scale_status=(
                DepthScaleStatus.RELATIVE_PER_FRAME
                if known
                else DepthScaleStatus.UNKNOWN
            ),
            larger_value_means=direction,
            valid_mask=valid_mask,
            confidence_map=None,
            provider_name=f"transformers:{self.resolved_model_name}",
            frame_index=frame_index,
            valid=valid,
            quality=quality,
            missing_reason=missing_reason,
            metadata={
                "model_name": self.model_name,
                "resolved_model_name": self.resolved_model_name,
                "raw_depth_representation": self.raw_depth_representation.value,
                "raw_larger_value_means": self.raw_larger_value_means.value,
                "conversion": (
                    "reciprocal_inverse_to_relative"
                    if self.invert_depth
                    and self.raw_depth_representation
                    in {DepthRepresentation.INVERSE_DEPTH, DepthRepresentation.DISPARITY_LIKE}
                    else "none"
                ),
                "metric_depth": False,
                "legacy_normalized_depth": False,
            },
        )

    def predict_depth(self, frame_path: PathLike) -> np.ndarray:
        """Return the frozen legacy [1, 10] array used by existing baselines.

        This method intentionally preserves the historical affine inversion and
        normalization. It is not the canonical geometry interface.
        """

        _, resized, _, _ = self._predict_raw(frame_path)
        depth_map = _sanitize_depth(resized, min_depth_value=self.min_depth_value)

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

    def legacy_depth_metadata(self) -> dict[str, Any]:
        """Mark the old affine-normalized output as non-geometric evidence."""

        return {
            "interface": "predict_depth",
            "legacy": True,
            "legacy_normalized_depth": bool(self.normalize),
            "depth_representation": "legacy_normalized_depth" if self.normalize else self.raw_depth_representation.value,
            "scale_status": DepthScaleStatus.RELATIVE_PER_FRAME.value,
            "larger_value_means": (
                LargerValueMeans.FARTHER.value
                if self.invert_depth
                else self.raw_larger_value_means.value
            ),
            "provider_name": f"transformers:{self.resolved_model_name}",
        }


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


def _infer_model_depth_semantics(
    model_name: str,
) -> tuple[DepthRepresentation, LargerValueMeans]:
    """Return documented model-family output semantics without claiming meters.

    Depth Anything V2 predicts relative inverse-depth-like values. Other model
    families stay unknown until an adapter explicitly declares their semantics.
    """

    normalized = model_name.strip().lower()
    if "depth-anything" in normalized:
        return DepthRepresentation.INVERSE_DEPTH, LargerValueMeans.CLOSER
    return DepthRepresentation.UNKNOWN, LargerValueMeans.UNKNOWN


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


class LegacyDepthProviderAdapter(BaseDepthProvider):
    """Adapt an array-returning legacy provider to the canonical contract.

    The safe default treats its array as legacy visualization/compatibility
    output and marks it invalid for geometry. Callers must explicitly declare
    representation, scale, and direction before a non-normalized legacy source
    can become valid geometry evidence.
    """

    def __init__(
        self,
        legacy_provider: Any,
        provider_name: str = "legacy_depth_provider",
        *,
        legacy_normalized_depth: bool = True,
        depth_representation: DepthRepresentation | str = DepthRepresentation.UNKNOWN,
        scale_status: DepthScaleStatus | str = DepthScaleStatus.UNKNOWN,
        larger_value_means: LargerValueMeans | str = LargerValueMeans.UNKNOWN,
    ) -> None:
        if not (
            hasattr(legacy_provider, "predict_depth")
            or hasattr(legacy_provider, "estimate_depth")
        ):
            raise TypeError(
                "legacy_provider must define predict_depth(frame_path) or "
                "estimate_depth(image, frame_index)."
            )
        self.legacy_provider = legacy_provider
        self.provider_name = provider_name
        self.legacy_normalized_depth = bool(legacy_normalized_depth)
        self.depth_representation = DepthRepresentation(depth_representation)
        self.scale_status = DepthScaleStatus(scale_status)
        self.larger_value_means = LargerValueMeans(larger_value_means)

    def predict_observation(
        self, frame_path: PathLike, frame_index: Optional[int] = None
    ) -> DepthObservation:
        """Wrap one legacy array with explicit unsafe/safe semantics."""

        if hasattr(self.legacy_provider, "predict_depth"):
            raw_array = self.legacy_provider.predict_depth(frame_path)
        else:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read frame image with OpenCV: {frame_path}")
            raw_array = self.legacy_provider.estimate_depth(
                image, -1 if frame_index is None else int(frame_index)
            )
        array = np.asarray(raw_array, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Legacy depth provider must return HxW, got {array.shape}.")
        valid_mask = np.isfinite(array) & (array > 0.0)
        if self.legacy_normalized_depth:
            return DepthObservation(
                depth_map=array,
                raw_model_output=array.copy(),
                visualization_depth=_normalize_depth(array, 0.0, 1.0),
                depth_representation=DepthRepresentation.UNKNOWN,
                scale_status=DepthScaleStatus.UNKNOWN,
                larger_value_means=LargerValueMeans.UNKNOWN,
                valid_mask=valid_mask,
                confidence_map=None,
                provider_name=self.provider_name,
                frame_index=frame_index,
                valid=False,
                quality=0.0,
                missing_reason="legacy_normalized_depth_not_geometry",
                metadata={"legacy_normalized_depth": True},
            )
        valid = bool(
            np.any(valid_mask)
            and self.depth_representation != DepthRepresentation.UNKNOWN
            and self.scale_status != DepthScaleStatus.UNKNOWN
            and self.larger_value_means != LargerValueMeans.UNKNOWN
        )
        return DepthObservation(
            depth_map=array,
            raw_model_output=array.copy(),
            visualization_depth=_normalize_depth(array, 0.0, 1.0),
            depth_representation=self.depth_representation,
            scale_status=self.scale_status,
            larger_value_means=self.larger_value_means,
            valid_mask=valid_mask,
            confidence_map=None,
            provider_name=self.provider_name,
            frame_index=frame_index,
            valid=valid,
            quality=float(np.mean(valid_mask)) if valid else 0.0,
            missing_reason="" if valid else "incomplete_legacy_depth_semantics",
            metadata={"legacy_normalized_depth": False, "adapter": "legacy"},
        )


# Explicit name for the dense mock; semantic3d.providers.MockDepthProvider is a
# separate legacy label-to-depth helper retained for old mock-object tests.
FrameMockDepthProvider = MockDepthProvider


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
