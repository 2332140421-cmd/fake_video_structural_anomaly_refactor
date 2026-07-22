"""Lazy, no-download adapter skeletons for future metric-depth providers."""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Optional

from .metric_scale import MetricDepthEvidence
from .scale_evidence import ProviderStatus


@dataclass(frozen=True)
class MetricDepthAdapterDescriptor:
    """Machine-readable status for a provider that has not run inference."""

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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class BaseMetricDepthAdapter(ABC):
    """Canonical future metric provider contract with explicit local weights."""

    provider_name = "base_metric_depth"
    expected_module = ""
    expected_model_family = ""
    output_unit = "meter"
    depth_definition = "unknown"

    def __init__(self, weights_path: str | Path | None = None, *, device: str = "cpu") -> None:
        self.weights_path = None if weights_path is None else Path(weights_path)
        self.device = str(device)

    def describe(self) -> MetricDepthAdapterDescriptor:
        """Report interface readiness without importing a large dependency."""

        dependency = bool(
            self.expected_module and importlib.util.find_spec(self.expected_module) is not None
        )
        weights = bool(self.weights_path is not None and self.weights_path.is_file())
        return MetricDepthAdapterDescriptor(
            provider_name=self.provider_name,
            expected_module=self.expected_module,
            expected_model_family=self.expected_model_family,
            adapter_status="not_executed" if dependency and weights else "interface_only",
            output_unit=self.output_unit,
            depth_definition=self.depth_definition,
            metric_scale_status="model_predicted",
            sensor_ground_truth=False,
            automatic_download_allowed=False,
            weights_path="" if self.weights_path is None else str(self.weights_path),
            dependency_available=dependency,
            weights_available=weights,
        )

    def _require_local_runtime(self) -> None:
        descriptor = self.describe()
        if not descriptor.dependency_available:
            raise RuntimeError(
                f"{self.provider_name} dependency {self.expected_module!r} is unavailable; "
                "install it explicitly on the server. No automatic installation was attempted."
            )
        if not descriptor.weights_available:
            raise RuntimeError(
                f"{self.provider_name} requires an explicit local weights_path. "
                "No network download was attempted."
            )

    @abstractmethod
    def predict_metric_depth(self, frame_path: str | Path) -> MetricDepthEvidence:
        """Return model-predicted metric depth with explicit semantics."""


class _InterfaceOnlyMetricAdapter(BaseMetricDepthAdapter):
    def predict_metric_depth(self, frame_path: str | Path) -> MetricDepthEvidence:
        self._require_local_runtime()
        raise NotImplementedError(
            f"{self.provider_name} loading/inference is intentionally interface_only in P4-C3A-MD2."
        )


class UniDepthV2Adapter(_InterfaceOnlyMetricAdapter):
    provider_name = "unidepth_v2"
    expected_module = "unidepth"
    expected_model_family = "UniDepthV2"
    depth_definition = "z_depth_or_model_documented_depth_required"


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
    name: str, *, weights_path: Optional[str | Path] = None, device: str = "cpu"
) -> BaseMetricDepthAdapter:
    """Create an adapter without importing its optional model package."""

    try:
        adapter = METRIC_DEPTH_ADAPTERS[str(name).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown metric depth adapter {name!r}.") from exc
    return adapter(weights_path=weights_path, device=device)
