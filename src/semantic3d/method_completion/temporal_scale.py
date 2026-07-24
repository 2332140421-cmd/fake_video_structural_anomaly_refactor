"""Same-object cross-frame scale stability for metric and local-relative modes."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from .scale_evidence import (
    ProviderStatus,
    ScaleBranchName,
    ScaleEvidenceRole,
    ScaleGeometryEvidence,
)


class TemporalScaleMode(str, Enum):
    """Scale domain used by the same-object temporal route."""

    METRIC = "metric"
    RELATIVE_LOCAL = "relative_local"


class TemporalReferenceMethod(str, Enum):
    """Supported causal/history reference estimators."""

    PREVIOUS_VALID = "previous_valid"
    ROLLING_MEDIAN = "rolling_median"
    TRACK_MEDIAN = "track_median"
    ROBUST_TRACK_MEDIAN = "robust_track_median"
    ROBUST_REFERENCE_WINDOW = "robust_reference_window"


@dataclass(frozen=True)
class ScaleHistoryObservation:
    """One dimension-specific scale observation for a stable track."""

    video_id: str
    clip_id: str
    frame_id: str
    frame_index: int
    track_id: str
    object_id: str
    dimension_type: str
    size_value: float
    size_unit: str
    temporal_mode: TemporalScaleMode | str
    depth_provider: str
    depth_definition: str
    intrinsics_fingerprint: str
    depth_scale_alignment_status: str
    pose_change_status: str = "stable"
    occlusion_status: str = "visible"
    truncated: bool = False
    out_of_frame: bool = False
    mask_stable: bool = True
    provider_status: ProviderStatus | str = ProviderStatus.OK
    quality: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    viewpoint_class: str = "unknown"
    dimension_observable: bool = True
    intrinsics_source: str = "unknown"
    valid: bool = True
    failure_reason: str = ""
    track_continuity_status: str = "continuous"
    scene_cut: bool = False

    def __post_init__(self) -> None:
        mode = TemporalScaleMode(self.temporal_mode)
        provider = ProviderStatus(self.provider_status)
        value = float(self.size_value)
        quality = float(self.quality)
        valid_observation = bool(self.valid) and bool(self.dimension_observable)
        if valid_observation and (not math.isfinite(value) or value <= 0.0):
            raise ValueError("Valid scale history size_value must be finite and positive.")
        if not valid_observation:
            value = float("nan")
            if not self.failure_reason:
                raise ValueError("Invalid scale history observations require failure_reason.")
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("Scale history quality must be in [0, 1].")
        required_unit = "meter" if mode == TemporalScaleMode.METRIC else "relative_local_unit"
        if self.size_unit not in {required_unit, "m" if mode == TemporalScaleMode.METRIC else required_unit}:
            raise ValueError(f"{mode.value} requires size_unit={required_unit}.")
        object.__setattr__(self, "temporal_mode", mode)
        object.__setattr__(self, "provider_status", provider)
        object.__setattr__(self, "size_value", value)
        object.__setattr__(self, "size_unit", required_unit)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "valid", valid_observation)
        object.__setattr__(self, "dimension_observable", bool(self.dimension_observable))
        object.__setattr__(self, "scene_cut", bool(self.scene_cut))


class TemporalSameObjectScaleBranch:
    """Priority-2 temporal size stability with explicit history isolation."""

    def __init__(
        self,
        *,
        reference_method: TemporalReferenceMethod | str = TemporalReferenceMethod.ROLLING_MEDIAN,
        min_valid_history: int = 2,
        reference_window: int = 5,
        max_frame_gap: int | None = None,
        minimum_quality: float = 0.0,
        max_intrinsics_relative_change: float = 0.0,
        config_sha256: str = "",
        software_commit: str = "",
    ) -> None:
        self.reference_method = TemporalReferenceMethod(reference_method)
        self.min_valid_history = int(min_valid_history)
        self.reference_window = int(reference_window)
        self.max_frame_gap = None if max_frame_gap is None else int(max_frame_gap)
        self.minimum_quality = float(minimum_quality)
        self.max_intrinsics_relative_change = float(max_intrinsics_relative_change)
        self.config_sha256 = config_sha256
        self.software_commit = software_commit
        if self.min_valid_history < 1 or self.reference_window < 1:
            raise ValueError("History lengths must be positive.")
        if self.max_frame_gap is not None and self.max_frame_gap < 1:
            raise ValueError("max_frame_gap must be positive when provided.")
        if not 0.0 <= self.minimum_quality <= 1.0:
            raise ValueError("minimum_quality must be in [0, 1].")
        if self.max_intrinsics_relative_change < 0.0:
            raise ValueError("max_intrinsics_relative_change cannot be negative.")

    def evaluate(
        self,
        current: ScaleHistoryObservation,
        history: Sequence[ScaleHistoryObservation],
    ) -> ScaleGeometryEvidence:
        """Compare current scale with a valid same-track historical reference."""

        base = self._base(current)
        if not current.valid or not current.dimension_observable:
            return ScaleGeometryEvidence.missing(
                failure_reason=current.failure_reason or "dimension_not_observable",
                **base,
            )
        if current.provider_status == ProviderStatus.PROVIDER_FAILED:
            return ScaleGeometryEvidence.missing(
                failure_reason="scale_provider_failed",
                provider_status=ProviderStatus.PROVIDER_FAILED,
                **base,
            )
        if current.provider_status != ProviderStatus.OK:
            return ScaleGeometryEvidence.missing(failure_reason="scale_provider_not_ready", **base)
        if current.quality < self.minimum_quality:
            return ScaleGeometryEvidence.missing(
                failure_reason="current_scale_quality_below_threshold", **base
            )
        if current.scene_cut:
            return ScaleGeometryEvidence.missing(failure_reason="scene_cut_boundary", **base)
        if current.track_continuity_status not in {"continuous", "recovered_verified"}:
            return ScaleGeometryEvidence.missing(
                failure_reason="track_interruption_or_id_switch", **base
            )
        if current.truncated or current.out_of_frame:
            return ScaleGeometryEvidence.missing(
                failure_reason="current_object_truncated_or_out_of_frame", **base
            )
        if current.occlusion_status in {"heavy_occlusion", "fully_occluded"}:
            return ScaleGeometryEvidence.missing(failure_reason="current_object_occluded", **base)
        if current.pose_change_status not in {"stable", "compatible"}:
            return ScaleGeometryEvidence.missing(failure_reason="object_pose_change_not_compatible", **base)
        if not current.mask_stable:
            return ScaleGeometryEvidence.missing(failure_reason="unstable_object_mask", **base)
        if current.temporal_mode == TemporalScaleMode.RELATIVE_LOCAL and (
            current.depth_scale_alignment_status not in {"aligned", "relative_shared_clip"}
        ):
            return ScaleGeometryEvidence.missing(
                failure_reason="relative_depth_scale_not_locally_aligned", **base
            )

        earlier = [item for item in history if item.frame_index < current.frame_index]
        if earlier and any(item.video_id != current.video_id for item in earlier):
            return ScaleGeometryEvidence.missing(failure_reason="cross_video_history_forbidden", **base)
        if earlier and any(item.clip_id != current.clip_id for item in earlier):
            return ScaleGeometryEvidence.missing(failure_reason="cross_clip_history_forbidden", **base)
        if earlier and any(item.track_id != current.track_id for item in earlier):
            return ScaleGeometryEvidence.missing(failure_reason="track_id_switch_or_mismatch", **base)
        compatible: list[ScaleHistoryObservation] = []
        for item in earlier:
            if item.dimension_type != current.dimension_type:
                continue
            if item.temporal_mode != current.temporal_mode or item.size_unit != current.size_unit:
                return ScaleGeometryEvidence.missing(failure_reason="scale_domain_or_unit_changed", **base)
            if item.depth_provider != current.depth_provider:
                return ScaleGeometryEvidence.missing(failure_reason="depth_provider_changed", **base)
            if item.depth_definition != current.depth_definition:
                return ScaleGeometryEvidence.missing(failure_reason="depth_definition_changed", **base)
            if not self._intrinsics_compatible(item, current):
                return ScaleGeometryEvidence.missing(failure_reason="camera_intrinsics_changed", **base)
            if not item.valid or not item.dimension_observable:
                continue
            if item.provider_status != ProviderStatus.OK:
                continue
            if item.quality < self.minimum_quality:
                continue
            if item.truncated or item.out_of_frame or not item.mask_stable:
                continue
            if item.occlusion_status in {"heavy_occlusion", "fully_occluded"}:
                continue
            if item.scene_cut:
                continue
            compatible.append(item)
        if self.max_frame_gap is not None and compatible:
            latest = max(item.frame_index for item in compatible)
            if current.frame_index - latest > self.max_frame_gap:
                return ScaleGeometryEvidence.missing(
                    failure_reason="track_history_frame_gap_too_large", **base
                )
        required = 1 if self.reference_method == TemporalReferenceMethod.PREVIOUS_VALID else self.min_valid_history
        if len(compatible) < required:
            return ScaleGeometryEvidence.missing(
                failure_reason="insufficient_valid_scale_history", **base
            )
        reference_items = self._reference_items(compatible)
        reference = (
            reference_items[-1].size_value
            if self.reference_method == TemporalReferenceMethod.PREVIOUS_VALID
            else float(np.median([item.size_value for item in reference_items]))
        )
        residual = abs(math.log(current.size_value) - math.log(reference))
        quality = min(current.quality, float(np.median([item.quality for item in reference_items])))
        return ScaleGeometryEvidence.observed(
            residual_value=residual,
            confidence=quality,
            uncertainty=float(np.median(np.abs(
                np.log([item.size_value for item in reference_items]) - math.log(reference)
            ))),
            provenance={
                "temporal_mode": current.temporal_mode.value,
                "dimension_type": current.dimension_type,
                "current_size": current.size_value,
                "reference_size": reference,
                "size_unit": current.size_unit,
                "reference_method": self.reference_method.value,
                "track_length": len(earlier) + 1,
                "valid_history_length": len(compatible),
                "depth_scale_alignment_status": current.depth_scale_alignment_status,
                "pose_change_status": current.pose_change_status,
                "occlusion_status": current.occlusion_status,
                "viewpoint_class": current.viewpoint_class,
                "intrinsics_source": current.intrinsics_source,
                "formula": "abs(log(S_t)-log(S_reference))",
            },
            **{key: value for key, value in base.items() if key != "provenance"},
        )

    def _reference_items(
        self, history: Sequence[ScaleHistoryObservation]
    ) -> list[ScaleHistoryObservation]:
        ordered = sorted(history, key=lambda item: item.frame_index)
        if self.reference_method == TemporalReferenceMethod.PREVIOUS_VALID:
            return ordered[-1:]
        if self.reference_method in {
            TemporalReferenceMethod.ROLLING_MEDIAN,
            TemporalReferenceMethod.ROBUST_REFERENCE_WINDOW,
        }:
            return ordered[-self.reference_window :]
        return ordered

    def _intrinsics_compatible(
        self,
        previous: ScaleHistoryObservation,
        current: ScaleHistoryObservation,
    ) -> bool:
        if previous.intrinsics_fingerprint == current.intrinsics_fingerprint:
            return True
        if self.max_intrinsics_relative_change <= 0.0:
            return False
        if previous.intrinsics_source != current.intrinsics_source:
            return False
        first = previous.metadata.get("intrinsics_parameters")
        second = current.metadata.get("intrinsics_parameters")
        if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)):
            return False
        if len(first) != 4 or len(second) != 4:
            return False
        left = np.asarray(first, dtype=float)
        right = np.asarray(second, dtype=float)
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            return False
        relative = np.max(np.abs(right - left) / np.maximum(np.abs(left), 1.0))
        return bool(relative <= self.max_intrinsics_relative_change)

    def _base(self, current: ScaleHistoryObservation) -> dict[str, Any]:
        return {
            "video_id": current.video_id,
            "clip_id": current.clip_id,
            "frame_id": current.frame_id,
            "object_id": current.object_id,
            "track_id": current.track_id,
            "branch_name": ScaleBranchName.TEMPORAL_SAME_OBJECT,
            "branch_priority": 2,
            "evidence_role": ScaleEvidenceRole.TEMPORAL_SUPPORT,
            "residual_name": "R_size_temporal",
            "depth_type": (
                "metric" if current.temporal_mode == TemporalScaleMode.METRIC else "relative"
            ),
            "depth_unit": current.size_unit,
            "depth_definition": current.depth_definition,
            "coordinate_system": "same_track_local_scale_history",
            "localization_reference": f"track:{current.track_id}",
            "provenance": {"depth_provider": current.depth_provider},
            "config_sha256": self.config_sha256,
            "software_commit": self.software_commit,
        }


# Compatibility alias for the previous route name.
CrossFrameScaleStabilityV2Branch = TemporalSameObjectScaleBranch
