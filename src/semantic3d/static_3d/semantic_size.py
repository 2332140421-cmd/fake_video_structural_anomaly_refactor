"""Unary semantic 3D scale evidence with metric and non-circular relative modes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..scale_depth import ScalePrior
from ..scale_prior import normalize_label
from ..shared_3d_observation import (
    GeometryScaleStatus,
    GeometryScaleUnit,
    Object3DObservation,
    Shared3DFrameObservation,
)
from ..validity import ResidualEvidence
from .residual_types import EvidenceRole, Static3DContext


@dataclass(frozen=True)
class _PriorInterval:
    label: str
    low: float
    high: float
    source: str


def _interval_from_resolver(resolver: Any, label: str) -> Optional[_PriorInterval]:
    """Resolve a reliable metric interval from a resolver or mapping."""

    normalized = normalize_label(label)
    if hasattr(resolver, "resolve"):
        resolved = resolver.resolve(label)
        if not bool(getattr(resolved, "reliable", False)):
            return None
        low_value = getattr(resolved, "low", getattr(resolved, "min_size", math.nan))
        high_value = getattr(
            resolved, "high", getattr(resolved, "max_size", math.nan)
        )
        low, high = float(low_value), float(high_value)
        resolved_label = str(getattr(resolved, "resolved_label", normalized))
        source = str(
            getattr(
                resolved,
                "prior_source",
                getattr(resolved, "source", "physical"),
            )
        )
    elif isinstance(resolver, Mapping):
        entry = resolver.get(normalized) or resolver.get(label)
        if entry is None:
            return None
        if isinstance(entry, ScalePrior):
            low, high = float(entry.min_size), float(entry.max_size)
        elif isinstance(entry, Mapping):
            low, high = float(entry["min"]), float(entry["max"])
        else:
            low, high = (float(value) for value in entry)
        resolved_label, source = normalized, "physical_test_mapping"
    else:
        raise TypeError("prior_resolver must define resolve(label) or be a mapping.")
    if not math.isfinite(low) or not math.isfinite(high) or low <= 0.0 or high <= low:
        return None
    return _PriorInterval(resolved_label, low, high, source)


def _distance_to_interval(value: float, low: float, high: float) -> float:
    return max(0.0, low - value, value - high)


class SemanticSize3DResidual:
    """Produce one independent semantic-size ResidualEvidence per target object."""

    def __init__(self, prior_resolver: Any) -> None:
        self.prior_resolver = prior_resolver

    @staticmethod
    def _object(
        frame: Shared3DFrameObservation, object_id: str
    ) -> Object3DObservation | None:
        try:
            return Static3DContext(frame).object_by_id(object_id)
        except KeyError:
            return None

    @staticmethod
    def _metadata(
        obj: Object3DObservation | None,
        *,
        calibration_status: str,
        calibration_source: str,
        anchor_object_ids: Sequence[str] = (),
        excluded_from_evaluation_ids: Sequence[str] = (),
        estimated_scene_scale: float | None = None,
        expected_min: float = math.nan,
        expected_max: float = math.nan,
    ) -> dict[str, object]:
        return {
            "evidence_role": EvidenceRole.ANOMALY_RESIDUAL.value,
            "object_id": None if obj is None else obj.source_object_2d_id,
            "semantic_label": None if obj is None else obj.semantic_label,
            "observed_scale_3d": None if obj is None else obj.observed_scale_3d,
            "scale_status": None if obj is None else obj.scale_status.value,
            "scale_unit": None if obj is None else obj.scale_unit.value,
            "calibration_status": calibration_status,
            "calibration_source": calibration_source,
            "anchor_object_ids": list(anchor_object_ids),
            "excluded_from_evaluation_ids": list(excluded_from_evaluation_ids),
            "estimated_scene_scale": estimated_scene_scale,
            "expected_min": expected_min,
            "expected_max": expected_max,
            "prior_source": "physical",
            "one_to_n_scene_object_representation": True,
        }

    def _missing(
        self,
        object_id: str,
        reason: str,
        obj: Object3DObservation | None,
        **metadata: object,
    ) -> ResidualEvidence:
        base = self._metadata(
            obj,
            calibration_status=str(metadata.pop("calibration_status", "unavailable")),
            calibration_source=str(metadata.pop("calibration_source", "none")),
            anchor_object_ids=metadata.pop("anchor_object_ids", ()),  # type: ignore[arg-type]
            excluded_from_evaluation_ids=metadata.pop(
                "excluded_from_evaluation_ids", ()
            ),  # type: ignore[arg-type]
        )
        base.update(metadata)
        return ResidualEvidence.missing(
            f"semantic_size_3d:{object_id}",
            reason,
            source_ids=(object_id,),
            metadata=base,
        )

    def evaluate_metric(
        self, frame: Shared3DFrameObservation, object_id: str
    ) -> ResidualEvidence:
        """Compare calibrated metric object scale with a metric category interval."""

        obj = self._object(frame, object_id)
        if obj is None or not obj.valid or obj.observed_scale_3d is None:
            return self._missing(object_id, "invalid_object_scale", obj)
        if (
            obj.scale_status != GeometryScaleStatus.METRIC_3D
            or obj.scale_unit != GeometryScaleUnit.METER
        ):
            return self._missing(
                object_id,
                "metric_calibration_required",
                obj,
                calibration_status="relative_uncalibrated",
            )
        metric_source = str(obj.metadata.get("metric_scale_source", "none"))
        if metric_source in {"", "none", "unknown"}:
            return self._missing(
                object_id,
                "missing_metric_scale_source",
                obj,
                calibration_status="metric_source_missing",
            )
        prior = _interval_from_resolver(self.prior_resolver, obj.canonical_label)
        if prior is None:
            return self._missing(object_id, "missing_or_unreliable_physical_prior", obj)
        observed = float(obj.observed_scale_3d)
        residual = _distance_to_interval(observed, prior.low, prior.high)
        metadata = self._metadata(
            obj,
            calibration_status="metric_calibrated",
            calibration_source=metric_source,
            expected_min=prior.low,
            expected_max=prior.high,
        )
        metadata.update(
            {
                "resolved_label": prior.label,
                "prior_source": prior.source,
                "estimated_metric_scale": observed,
            }
        )
        return ResidualEvidence.observed(
            f"semantic_size_3d:{object_id}",
            residual,
            quality=min(obj.reconstruction_quality, obj.scale_quality or 0.0),
            source_ids=(object_id,),
            metadata=metadata,
        )

    def evaluate_metric_scene(
        self, frame: Shared3DFrameObservation
    ) -> dict[str, ResidualEvidence]:
        """Return one metric evidence record per object, never an n-by-n matrix."""

        return {
            obj.source_object_2d_id: self.evaluate_metric(
                frame, obj.source_object_2d_id
            )
            for obj in frame.objects
        }

    def evaluate_relative_with_anchors(
        self,
        frame: Shared3DFrameObservation,
        object_id: str,
        anchor_object_ids: Sequence[str],
        *,
        anchors_are_independent: bool,
        calibration_source: str = "independent_anchor_calibration",
    ) -> ResidualEvidence:
        """Calibrate a target from independent other objects in the same frame."""

        target = self._object(frame, object_id)
        anchors = tuple(dict.fromkeys(str(item) for item in anchor_object_ids))
        if target is None or not target.valid or target.observed_scale_3d is None:
            return self._missing(object_id, "invalid_object_scale", target)
        if not anchors:
            return self._missing(
                object_id,
                "no_independent_scale_anchor",
                target,
                calibration_status="relative_uncalibrated",
            )
        if object_id in anchors:
            return self._missing(
                object_id,
                "anchor_self_evaluation_forbidden",
                target,
                calibration_status="circular_anchor_rejected",
                anchor_object_ids=anchors,
                excluded_from_evaluation_ids=anchors,
            )
        if not anchors_are_independent:
            return self._missing(
                object_id,
                "anchor_independence_not_established",
                target,
                calibration_status="relative_uncalibrated",
                anchor_object_ids=anchors,
            )
        if target.scale_status != GeometryScaleStatus.RELATIVE_3D:
            return self._missing(
                object_id,
                "relative_calibration_requires_relative_3d",
                target,
                anchor_object_ids=anchors,
            )
        target_prior = _interval_from_resolver(
            self.prior_resolver, target.canonical_label
        )
        if target_prior is None:
            return self._missing(
                object_id,
                "missing_or_unreliable_physical_prior",
                target,
                anchor_object_ids=anchors,
            )

        factors: list[float] = []
        anchor_qualities: list[float] = []
        used_anchors: list[str] = []
        for anchor_id in anchors:
            anchor = self._object(frame, anchor_id)
            if (
                anchor is None
                or not anchor.valid
                or anchor.observed_scale_3d is None
                or anchor.scale_status != GeometryScaleStatus.RELATIVE_3D
            ):
                continue
            prior = _interval_from_resolver(
                self.prior_resolver, anchor.canonical_label
            )
            if prior is None:
                continue
            expected_midpoint = 0.5 * (prior.low + prior.high)
            factors.append(expected_midpoint / float(anchor.observed_scale_3d))
            anchor_qualities.append(anchor.reconstruction_quality)
            used_anchors.append(anchor_id)
        if not factors:
            return self._missing(
                object_id,
                "no_valid_independent_scale_anchor",
                target,
                calibration_status="relative_uncalibrated",
                anchor_object_ids=anchors,
            )
        scene_scale = float(np.median(factors))
        estimated_metric = float(target.observed_scale_3d) * scene_scale
        residual = _distance_to_interval(
            estimated_metric, target_prior.low, target_prior.high
        )
        metadata = self._metadata(
            target,
            calibration_status="relative_or_calibrated",
            calibration_source=calibration_source,
            anchor_object_ids=used_anchors,
            excluded_from_evaluation_ids=used_anchors,
            estimated_scene_scale=scene_scale,
            expected_min=target_prior.low,
            expected_max=target_prior.high,
        )
        metadata.update(
            {
                "estimated_metric_scale": estimated_metric,
                "target_used_as_anchor": False,
                "anchor_independence_asserted": True,
                "resolved_label": target_prior.label,
            }
        )
        return ResidualEvidence.observed(
            f"semantic_size_3d:{object_id}",
            residual,
            quality=min(
                target.reconstruction_quality,
                float(np.mean(anchor_qualities)),
            ),
            source_ids=(object_id, *used_anchors),
            metadata=metadata,
        )

    def evaluate_leave_one_out(
        self,
        frame: Shared3DFrameObservation,
        object_id: str,
        scene_object_ids: Sequence[str],
    ) -> ResidualEvidence:
        """Calibrate from all eligible scene objects except the target itself."""

        anchors = [item for item in scene_object_ids if item != object_id]
        evidence = self.evaluate_relative_with_anchors(
            frame,
            object_id,
            anchors,
            anchors_are_independent=True,
            calibration_source="leave_one_out_scene_scale_calibration",
        )
        metadata = dict(evidence.metadata)
        metadata["leave_one_out_target_excluded"] = True
        metadata["excluded_target_id"] = object_id
        if evidence.valid:
            return ResidualEvidence.observed(
                evidence.name,
                evidence.value,
                quality=evidence.quality,
                source_ids=evidence.source_ids,
                metadata=metadata,
            )
        return ResidualEvidence.missing(
            evidence.name,
            evidence.missing_reason,
            source_ids=evidence.source_ids,
            metadata=metadata,
        )

    def evaluate_leave_one_out_scene(
        self,
        frame: Shared3DFrameObservation,
        scene_object_ids: Sequence[str] | None = None,
    ) -> dict[str, ResidualEvidence]:
        """Return 1:n leave-one-out evidence with each target excluded in turn."""

        object_ids = tuple(
            scene_object_ids
            if scene_object_ids is not None
            else (obj.source_object_2d_id for obj in frame.objects)
        )
        return {
            object_id: self.evaluate_leave_one_out(frame, object_id, object_ids)
            for object_id in object_ids
        }

    def evaluate_same_frame_relative_ratio(
        self,
        frame: Shared3DFrameObservation,
        object_id: str,
        reference_object_id: str,
    ) -> ResidualEvidence:
        """Evaluate a target/reference scale ratio without claiming metric scale."""

        target = self._object(frame, object_id)
        reference = self._object(frame, reference_object_id)
        if object_id == reference_object_id:
            return self._missing(
                object_id,
                "reference_self_evaluation_forbidden",
                target,
                calibration_status="circular_anchor_rejected",
            )
        if (
            target is None
            or reference is None
            or not target.valid
            or not reference.valid
            or target.observed_scale_3d is None
            or reference.observed_scale_3d is None
            or target.scale_status != GeometryScaleStatus.RELATIVE_3D
            or reference.scale_status != GeometryScaleStatus.RELATIVE_3D
        ):
            return self._missing(
                object_id,
                "invalid_relative_scale_pair",
                target,
                calibration_status="relative_uncalibrated",
            )
        target_prior = _interval_from_resolver(
            self.prior_resolver, target.canonical_label
        )
        reference_prior = _interval_from_resolver(
            self.prior_resolver, reference.canonical_label
        )
        if target_prior is None or reference_prior is None:
            return self._missing(
                object_id,
                "missing_or_unreliable_physical_prior",
                target,
                calibration_status="relative_uncalibrated",
            )
        observed_ratio = float(target.observed_scale_3d) / float(
            reference.observed_scale_3d
        )
        expected_low = target_prior.low / reference_prior.high
        expected_high = target_prior.high / reference_prior.low
        log_ratio = math.log(observed_ratio)
        residual = _distance_to_interval(
            log_ratio, math.log(expected_low), math.log(expected_high)
        )
        metadata = self._metadata(
            target,
            calibration_status="relative_or_calibrated",
            calibration_source="same_frame_relative_ratio",
            anchor_object_ids=(reference_object_id,),
            excluded_from_evaluation_ids=(reference_object_id,),
            expected_min=expected_low,
            expected_max=expected_high,
        )
        metadata.update(
            {
                "observed_relative_ratio": observed_ratio,
                "residual_space": "log_ratio",
                "estimated_scene_scale": None,
                "metric_unary_claim": False,
            }
        )
        return ResidualEvidence.observed(
            f"semantic_size_3d:{object_id}",
            residual,
            quality=min(
                target.reconstruction_quality, reference.reconstruction_quality
            ),
            source_ids=(object_id, reference_object_id),
            metadata=metadata,
        )
