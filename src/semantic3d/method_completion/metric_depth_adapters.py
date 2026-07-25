"""Offline metric-depth adapters used by the P4-C3B metric branch.

Only :class:`UniDepthV2Adapter` has a real inference implementation. The
remaining adapters deliberately stay interface-only until their dependencies,
weights, and output semantics are independently verified.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import math
import os
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

import cv2
import numpy as np

from ..depth_provider import (
    DepthObservation,
    DepthRepresentation,
    DepthScaleStatus,
    LargerValueMeans,
)
from ..geometry.camera import CameraObservation, CoordinateConvention
from ..runtime.external_sources import (
    ExternalSourceError,
    activate_unidepth_source,
    verify_unidepth_source,
)
from .metric_scale import (
    MetricDepthDefinition,
    MetricDepthEvidence,
    MetricDepthType,
    MetricScaleStatus,
)
from .scale_evidence import ProviderStatus


UNIDEPTH_V2_SOURCE_REVISION = "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"
UNIDEPTH_V2_MODEL_REVISION = "038c238f06c87b6c2f5b3749fd51fbf442b1f218"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one local file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_weight_file(weights_path: str | Path | None) -> Optional[Path]:
    """Resolve a model file from a file path or Hugging Face-style directory."""

    if weights_path is None:
        return None
    path = Path(weights_path)
    if path.is_file():
        return path
    candidate = path / "model.safetensors"
    return candidate if candidate.is_file() else None


@dataclass(frozen=True)
class MetricDepthAdapterDescriptor:
    """Machine-readable local environment and adapter readiness."""

    provider_name: str
    expected_module: str
    expected_model_family: str
    adapter_status: str
    output_unit: str
    depth_definition: str
    metric_scale_status: str
    sensor_ground_truth: bool
    automatic_download_allowed: bool
    weights_path: str
    dependency_available: bool
    weights_available: bool
    expected_weight_sha256: str = ""
    actual_weight_sha256: str = ""
    weight_hash_verified: bool = False
    device: str = "cpu"
    precision: str = "fp32"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe descriptor."""

        return asdict(self)


@dataclass(frozen=True)
class MetricDepthFrameResult:
    """One metric-depth inference with canonical depth and camera contracts."""

    depth_observation: DepthObservation
    metric_depth_evidence: MetricDepthEvidence
    camera_observation: CameraObservation
    uncertainty_map: Optional[np.ndarray]
    intrinsics_confidence: float
    runtime_seconds: float
    peak_gpu_memory_bytes: int
    provider_version: str
    weight_sha256: str
    raw_depth_definition: str
    standardized_depth_definition: str
    original_image_size: tuple[int, int]
    provider_status: str
    failure_reason: str = ""


class MetricProviderRuntimeError(RuntimeError):
    """A stable provider failure that cannot become anomaly evidence."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = str(reason)


class BaseMetricDepthAdapter(ABC):
    """Canonical metric provider contract with explicit local weights."""

    provider_name = "base_metric_depth"
    expected_module = ""
    expected_model_family = ""
    output_unit = "meter"
    depth_definition = "unknown"
    adapter_implemented = False

    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        device: str = "cpu",
        precision: str = "fp32",
        expected_weight_sha256: str = "",
    ) -> None:
        self.weights_path = None if weights_path is None else Path(weights_path)
        self.device = str(device).strip().lower()
        self.precision = str(precision).strip().lower()
        self.expected_weight_sha256 = str(expected_weight_sha256).strip().lower()
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError("precision must be 'fp32' or 'fp16'.")

    def describe(self) -> MetricDepthAdapterDescriptor:
        """Report readiness without importing a large optional model package."""

        dependency = bool(
            self.expected_module and importlib.util.find_spec(self.expected_module) is not None
        )
        weight_file = resolve_weight_file(self.weights_path)
        weights = weight_file is not None
        actual_sha = sha256_file(weight_file) if weight_file is not None else ""
        hash_verified = bool(
            actual_sha
            and self.expected_weight_sha256
            and actual_sha == self.expected_weight_sha256
        )
        if self.adapter_implemented and dependency and weights and hash_verified:
            status = "not_executed"
        elif self.adapter_implemented:
            status = "blocked_by_input"
        else:
            status = "interface_only"
        return MetricDepthAdapterDescriptor(
            provider_name=self.provider_name,
            expected_module=self.expected_module,
            expected_model_family=self.expected_model_family,
            adapter_status=status,
            output_unit=self.output_unit,
            depth_definition=self.depth_definition,
            metric_scale_status="model_predicted",
            sensor_ground_truth=False,
            automatic_download_allowed=False,
            weights_path="" if self.weights_path is None else str(self.weights_path),
            dependency_available=dependency,
            weights_available=weights,
            expected_weight_sha256=self.expected_weight_sha256,
            actual_weight_sha256=actual_sha,
            weight_hash_verified=hash_verified,
            device=self.device,
            precision=self.precision,
        )

    def _require_local_runtime(self) -> None:
        descriptor = self.describe()
        if not descriptor.dependency_available:
            raise MetricProviderRuntimeError(
                "metric_provider_dependency_missing",
                f"{self.provider_name} dependency {self.expected_module!r} is unavailable; "
                "install it explicitly. No automatic installation was attempted.",
            )
        if not descriptor.weights_available:
            raise MetricProviderRuntimeError(
                "metric_provider_weights_missing",
                f"{self.provider_name} requires an explicit local weights_path. "
                "No network download was attempted.",
            )
        if not self.expected_weight_sha256:
            raise MetricProviderRuntimeError(
                "metric_provider_weight_sha256_missing",
                "Metric inference is blocked until expected_weight_sha256 is registered.",
            )
        if not descriptor.weight_hash_verified:
            raise MetricProviderRuntimeError(
                "metric_provider_weight_sha256_mismatch",
                "Local metric-depth weight SHA-256 does not match the registered digest.",
            )

    @abstractmethod
    def predict_frame(
        self, frame_path: str | Path, *, frame_index: Optional[int] = None
    ) -> MetricDepthFrameResult:
        """Return canonical model-predicted metric depth and intrinsics."""

    def predict_metric_depth(self, frame_path: str | Path) -> MetricDepthEvidence:
        """Compatibility adapter returning only the MD2 metric evidence."""

        return self.predict_frame(frame_path).metric_depth_evidence


class _InterfaceOnlyMetricAdapter(BaseMetricDepthAdapter):
    def predict_frame(
        self, frame_path: str | Path, *, frame_index: Optional[int] = None
    ) -> MetricDepthFrameResult:
        self._require_local_runtime()
        raise NotImplementedError(
            f"{self.provider_name} loading/inference remains interface_only in P4-C3B-M1."
        )


@contextmanager
def _offline_model_environment() -> Iterator[None]:
    """Temporarily force common model hubs into offline mode."""

    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "WANDB_MODE")
    previous = {name: os.environ.get(name) for name in names}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "offline"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _rank_quality_from_error(error_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Convert frame-relative error ranking to a bounded quality ranking.

    UniDepthV2 calls this output ``confidence``, but its documentation defines
    it as an estimated scale-invariant log error whose magnitude is relative
    within one input. Lower error is therefore mapped to higher rank quality.
    This is a deterministic heuristic quality score, not a calibrated
    probability.
    """

    quality = np.zeros(error_map.shape, dtype=np.float32)
    values = np.asarray(error_map[valid_mask], dtype=np.float64)
    if values.size == 0:
        return quality
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(order.shape, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    if order.size == 1:
        ranked = np.ones(1, dtype=np.float64)
    else:
        ranked = 1.0 - ranks / float(order.size - 1)
    quality[valid_mask] = ranked.astype(np.float32)
    return quality


def _visualization_depth(depth: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Create a robust [0, 1] visualization without changing geometry depth."""

    output = np.full(depth.shape, np.nan, dtype=np.float32)
    values = depth[valid_mask]
    if values.size == 0:
        return output
    low, high = np.quantile(values, [0.02, 0.98])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        output[valid_mask] = 0.5
        return output
    output[valid_mask] = np.clip((values - low) / (high - low), 0.0, 1.0)
    return output


class UniDepthV2Adapter(BaseMetricDepthAdapter):
    """Offline UniDepthV2 adapter producing model-predicted metric Z depth.

    The model decoder predicts ray radius and three-dimensional points. The
    public ``infer`` method returns ``depth = points[:, 2]``; that optical-axis
    Z depth is the canonical geometry array. The radius is retained as raw
    model output. Neither array is sensor ground truth.
    """

    provider_name = "unidepth_v2_vits14"
    expected_module = "unidepth"
    expected_model_family = "UniDepthV2 ViT-S14"
    depth_definition = "camera_optical_axis_z"
    adapter_implemented = True

    def __init__(
        self,
        weights_path: str | Path | None = None,
        *,
        device: str = "cpu",
        precision: str = "fp32",
        expected_weight_sha256: str = "",
        resolution_level: int = 0,
        model_factory: Optional[Callable[[Path], Any]] = None,
    ) -> None:
        super().__init__(
            weights_path,
            device=device,
            precision=precision,
            expected_weight_sha256=expected_weight_sha256,
        )
        if not 0 <= int(resolution_level) < 10:
            raise ValueError("resolution_level must be in [0, 10).")
        if self.device == "cpu" and self.precision == "fp16":
            raise ValueError("FP16 UniDepthV2 inference is not supported on CPU.")
        self.resolution_level = int(resolution_level)
        self._model_factory = model_factory
        self._model: Any = None
        self._torch: Any = None

    def describe(self) -> MetricDepthAdapterDescriptor:
        """Report a pinned source checkout as an available dependency."""

        source_verified = False
        if self._model_factory is None and self.expected_module == "unidepth":
            try:
                verify_unidepth_source()
            except ExternalSourceError:
                pass
            else:
                source_verified = True
        descriptor = super().describe()
        if source_verified and not descriptor.dependency_available:
            status = (
                "not_executed"
                if descriptor.weights_available and descriptor.weight_hash_verified
                else "blocked_by_input"
            )
            return replace(
                descriptor,
                dependency_available=True,
                adapter_status=status,
            )
        return descriptor

    def _activate_external_source(self) -> None:
        if self._model_factory is not None or self.expected_module != "unidepth":
            return
        try:
            activate_unidepth_source()
        except ExternalSourceError as exc:
            raise MetricProviderRuntimeError(
                "metric_provider_dependency_missing",
                f"Verified UniDepth source is unavailable: {exc}",
            ) from exc

    def _resolved_device(self, torch_module: Any) -> str:
        if self.device == "auto":
            return "cuda" if bool(torch_module.cuda.is_available()) else "cpu"
        if self.device.startswith("cuda") and not bool(torch_module.cuda.is_available()):
            raise MetricProviderRuntimeError(
                "metric_provider_device_unavailable",
                f"Requested device {self.device!r}, but CUDA is unavailable.",
            )
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise MetricProviderRuntimeError(
                "metric_provider_device_unsupported",
                f"Unsupported UniDepthV2 device {self.device!r}.",
            )
        return self.device

    def _load_model(self) -> tuple[Any, Any, str]:
        """Lazy-load the model strictly from verified local files."""

        self._activate_external_source()
        self._require_local_runtime()
        if self._model is not None:
            assert self._torch is not None
            return self._model, self._torch, self._resolved_device(self._torch)

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency gate catches normal path
            raise MetricProviderRuntimeError(
                "metric_provider_dependency_missing", "PyTorch is unavailable."
            ) from exc
        resolved_device = self._resolved_device(torch)
        try:
            with _offline_model_environment():
                if self._model_factory is not None:
                    model = self._model_factory(Path(self.weights_path or ""))
                else:
                    from unidepth.models import UniDepthV2

                    model = UniDepthV2.from_pretrained(str(self.weights_path))
            model = model.to(resolved_device).eval()
            model.resolution_level = self.resolution_level
            if self.precision == "fp16":
                model = model.half()
        except MetricProviderRuntimeError:
            raise
        except Exception as exc:
            raise MetricProviderRuntimeError(
                "metric_provider_offline_load_failed",
                f"UniDepthV2 failed to load from local weights: {exc}",
            ) from exc
        self._model = model
        self._torch = torch
        return model, torch, resolved_device

    @staticmethod
    def _squeeze_map(value: Any, name: str) -> np.ndarray:
        array = value.detach().float().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        array = np.asarray(array, dtype=np.float32).squeeze()
        if array.ndim != 2:
            raise MetricProviderRuntimeError(
                "metric_provider_output_schema_invalid",
                f"UniDepthV2 {name} must reduce to HxW, got {array.shape}.",
            )
        return array

    def predict_frame(
        self, frame_path: str | Path, *, frame_index: Optional[int] = None
    ) -> MetricDepthFrameResult:
        """Run one verified local UniDepthV2 inference on an RGB frame."""

        path = Path(frame_path)
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise MetricProviderRuntimeError(
                "metric_provider_frame_unreadable", f"Could not read frame: {path}"
            )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        model, torch, resolved_device = self._load_model()
        input_tensor = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1)
        if self.precision == "fp16":
            input_tensor = input_tensor.half()
        if resolved_device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(resolved_device)
            torch.cuda.synchronize(resolved_device)
        start = time.perf_counter()
        try:
            with _offline_model_environment():
                outputs = model.infer(input_tensor)
            if resolved_device.startswith("cuda"):
                torch.cuda.synchronize(resolved_device)
            runtime = time.perf_counter() - start
        except Exception as exc:
            raise MetricProviderRuntimeError(
                "metric_provider_inference_failed", f"UniDepthV2 inference failed: {exc}"
            ) from exc

        required = {"depth", "radius", "confidence", "intrinsics"}
        missing = sorted(required.difference(outputs))
        if missing:
            raise MetricProviderRuntimeError(
                "metric_provider_output_schema_invalid",
                f"UniDepthV2 output is missing fields: {', '.join(missing)}.",
            )
        depth = self._squeeze_map(outputs["depth"], "depth")
        radius = self._squeeze_map(outputs["radius"], "radius")
        uncertainty = self._squeeze_map(outputs["confidence"], "confidence/error")
        if depth.shape != (height, width):
            raise MetricProviderRuntimeError(
                "metric_provider_output_resolution_mismatch",
                f"Expected {(height, width)} depth, got {depth.shape}.",
            )
        valid_mask = np.isfinite(depth) & (depth > 0.0) & np.isfinite(uncertainty)
        if not np.any(valid_mask):
            raise MetricProviderRuntimeError(
                "metric_provider_no_valid_depth", "UniDepthV2 returned no valid depth pixels."
            )
        rank_quality = _rank_quality_from_error(uncertainty, valid_mask)
        quality = float(np.median(rank_quality[valid_mask]))
        intrinsics_raw = outputs["intrinsics"]
        K = (
            intrinsics_raw.detach().float().cpu().numpy()
            if hasattr(intrinsics_raw, "detach")
            else np.asarray(intrinsics_raw)
        )
        K = np.asarray(K, dtype=np.float64).squeeze()
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise MetricProviderRuntimeError(
                "metric_provider_intrinsics_invalid",
                f"UniDepthV2 intrinsics must be finite 3x3, got {K.shape}.",
            )
        weight_file = resolve_weight_file(self.weights_path)
        assert weight_file is not None
        weight_sha = sha256_file(weight_file)
        try:
            provider_version = importlib.metadata.version("unidepth")
        except importlib.metadata.PackageNotFoundError:
            provider_version = f"source@{UNIDEPTH_V2_SOURCE_REVISION}"
        peak_memory = (
            int(torch.cuda.max_memory_allocated(resolved_device))
            if resolved_device.startswith("cuda")
            else 0
        )
        metadata = {
            "depth_type": MetricDepthType.METRIC.value,
            "depth_unit": "meter",
            "depth_definition": MetricDepthDefinition.Z_DEPTH.value,
            "raw_depth_definition": MetricDepthDefinition.RAY_DISTANCE.value,
            "metric_scale_status": MetricScaleStatus.MODEL_PREDICTED.value,
            "sensor_ground_truth": False,
            "confidence_semantics": "derived_inverse_rank_of_frame_relative_log_error",
            "confidence_is_calibrated_probability": False,
            "uncertainty_semantics": "unidepth_estimated_scale_invariant_log_error_relative_within_frame",
            "intrinsics_source": "model_predicted",
            "coordinate_frame": CoordinateConvention.OPENCV.value,
            "pose_source": "unavailable",
            "provider_version": provider_version,
            "provider_source_revision": UNIDEPTH_V2_SOURCE_REVISION,
            "provider_model_revision": UNIDEPTH_V2_MODEL_REVISION,
            "weight_sha256": weight_sha,
            "device": resolved_device,
            "precision": self.precision,
            "resolution_level": self.resolution_level,
            "input_image_size_hw": [height, width],
            "output_depth_size_hw": list(depth.shape),
            "runtime_seconds": float(runtime),
            "peak_gpu_memory_bytes": peak_memory,
            "provenance": "official_UniDepth_source_and_Hugging_Face_model",
        }
        depth_observation = DepthObservation(
            depth_map=depth,
            raw_model_output=radius,
            visualization_depth=_visualization_depth(depth, valid_mask),
            depth_representation=DepthRepresentation.METRIC_DEPTH,
            scale_status=DepthScaleStatus.METRIC_CALIBRATED,
            larger_value_means=LargerValueMeans.FARTHER,
            valid_mask=valid_mask,
            confidence_map=rank_quality,
            provider_name=self.provider_name,
            frame_index=frame_index,
            valid=True,
            quality=quality,
            metadata=metadata,
        )
        metric_evidence = MetricDepthEvidence.from_depth_observation(depth_observation)
        camera_observation = CameraObservation.from_parameters(
            K=K,
            image_width=width,
            image_height=height,
            intrinsics_source="model_predicted",
            quality=0.0,
            pose_source="unavailable",
            metadata={
                "provider_name": self.provider_name,
                "provider_version": provider_version,
                "sensor_calibrated": False,
                "intrinsics_confidence": float("nan"),
                "intrinsics_confidence_available": False,
                "quality_reason": "provider_does_not_expose_intrinsics_confidence",
                "weight_sha256": weight_sha,
            },
        )
        return MetricDepthFrameResult(
            depth_observation=depth_observation,
            metric_depth_evidence=metric_evidence,
            camera_observation=camera_observation,
            uncertainty_map=uncertainty,
            intrinsics_confidence=float("nan"),
            runtime_seconds=float(runtime),
            peak_gpu_memory_bytes=peak_memory,
            provider_version=provider_version,
            weight_sha256=weight_sha,
            raw_depth_definition=MetricDepthDefinition.RAY_DISTANCE.value,
            standardized_depth_definition=MetricDepthDefinition.Z_DEPTH.value,
            original_image_size=(height, width),
            provider_status="executed_valid",
        )


class DepthProAdapter(_InterfaceOnlyMetricAdapter):
    provider_name = "depth_pro"
    expected_module = "depth_pro"
    expected_model_family = "Depth Pro"
    depth_definition = "z_depth_or_model_documented_depth_required"


class Metric3Dv2Adapter(_InterfaceOnlyMetricAdapter):
    provider_name = "metric3d_v2"
    expected_module = "metric3d"
    expected_model_family = "Metric3Dv2"
    depth_definition = "z_depth_or_model_documented_depth_required"


class DepthAnythingV2MetricAdapter(_InterfaceOnlyMetricAdapter):
    provider_name = "depth_anything_v2_metric"
    expected_module = "depth_anything_v2"
    expected_model_family = "Depth Anything V2 Metric"
    depth_definition = "z_depth_or_model_documented_depth_required"


METRIC_DEPTH_ADAPTERS: Mapping[str, type[BaseMetricDepthAdapter]] = {
    "unidepth_v2": UniDepthV2Adapter,
    "depth_pro": DepthProAdapter,
    "metric3d_v2": Metric3Dv2Adapter,
    "depth_anything_v2_metric": DepthAnythingV2MetricAdapter,
}


def get_metric_depth_adapter(
    name: str,
    *,
    weights_path: Optional[str | Path] = None,
    device: str = "cpu",
    precision: str = "fp32",
    expected_weight_sha256: str = "",
    **kwargs: Any,
) -> BaseMetricDepthAdapter:
    """Create an adapter without importing its optional model package."""

    try:
        adapter = METRIC_DEPTH_ADAPTERS[str(name).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown metric depth adapter {name!r}.") from exc
    return adapter(
        weights_path=weights_path,
        device=device,
        precision=precision,
        expected_weight_sha256=expected_weight_sha256,
        **kwargs,
    )
