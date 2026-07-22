"""Three explicit semantic-geometry routes used by P4-C3A-M.

The routes deliberately keep metric unary scale, same-frame relative
scale-depth, and cross-frame scale stability separate.  In particular, a
monocular relative reconstruction cannot silently enter the metric branch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from ..scale_depth import (
    ObjectObservation,
    ScalePrior,
    scale_depth_residual,
    scale_depth_residual_log,
)
from ..shared_3d_observation import Object3DObservation, Shared3DFrameObservation
from ..static_3d.semantic_size import SemanticSize3DResidual
from ..validity import ResidualEvidence


@dataclass(frozen=True)
class AbsoluteSemanticScaleBranch:
    """Route A: compare one metric object scale with a physical prior."""

    prior_resolver: Any

    def evaluate(
        self, frame: Shared3DFrameObservation, object_id: str
    ) -> ResidualEvidence:
        """Return a metric residual or an explicit metric-scale block."""

        obj = next(
            (item for item in frame.objects if item.source_object_2d_id == object_id),
            None,
        )
        if obj is None:
            return ResidualEvidence.missing(
                "r_semantic_absolute_scale_3d",
                "object_not_found",
                source_ids=(object_id,),
                metadata={"semantic_geometry_route": "absolute_metric"},
            )
        if obj.scale_status.value != "metric_3d" or obj.scale_unit.value != "meter":
            return ResidualEvidence.missing(
                "r_semantic_absolute_scale_3d",
                "blocked_by_metric_scale_unavailable",
                source_ids=(object_id,),
                metadata={
                    "semantic_geometry_route": "absolute_metric",
                    "observed_scale_status": obj.scale_status.value,
                    "observed_scale_unit": obj.scale_unit.value,
                    "metric_value_fabricated": False,
                },
            )
        evidence = SemanticSize3DResidual(self.prior_resolver).evaluate_metric(
            frame, object_id
        )
        metadata = {
            **dict(evidence.metadata),
            "semantic_geometry_route": "absolute_metric",
        }
        if evidence.valid:
            return ResidualEvidence.observed(
                "r_semantic_absolute_scale_3d",
                evidence.value,
                quality=evidence.quality,
                source_ids=evidence.source_ids,
                metadata=metadata,
            )
        return ResidualEvidence.missing(
            "r_semantic_absolute_scale_3d",
            evidence.missing_reason,
            source_ids=evidence.source_ids,
            metadata=metadata,
        )


@dataclass(frozen=True)
class RelativeScaleDepthBranch:
    """Route B: evidence-aware wrapper around the frozen 2D pair baseline."""

    scale_priors: Mapping[str, ScalePrior]
    use_log: bool = True

    def evaluate(
        self, obj_a: ObjectObservation, obj_b: ObjectObservation
    ) -> ResidualEvidence:
        """Compute same-frame pair R_sd without converting invalid input to zero."""

        source_ids = (obj_a.object_id, obj_b.object_id)
        try:
            function = scale_depth_residual_log if self.use_log else scale_depth_residual
            value, details = function(obj_a, obj_b, self.scale_priors)
        except KeyError as exc:
            return ResidualEvidence.missing(
                "rsd_2d_relative_scale_depth",
                "missing_or_unreliable_scale_prior",
                source_ids=source_ids,
                metadata={
                    "semantic_geometry_route": "same_frame_relative_pair",
                    "error": str(exc),
                },
            )
        except ValueError as exc:
            return ResidualEvidence.missing(
                "rsd_2d_relative_scale_depth",
                "invalid_relative_scale_depth_input",
                source_ids=source_ids,
                metadata={
                    "semantic_geometry_route": "same_frame_relative_pair",
                    "error": str(exc),
                },
            )
        return ResidualEvidence.observed(
            "rsd_2d_relative_scale_depth",
            value,
            quality=min(float(obj_a.confidence), float(obj_b.confidence)),
            source_ids=source_ids,
            metadata={
                "semantic_geometry_route": "same_frame_relative_pair",
                "space": "log" if self.use_log else "ratio",
                "dimensionless": True,
                "legacy_formula_reused": True,
                **details,
            },
        )


@dataclass(frozen=True)
class CrossFrameScaleStabilityBranch:
    """Route C: compare one tracked object's 3D scale across two frames.

    The residual is ``max(0, abs(log(S_t / S_prev)) - tolerance)``.  It is
    defined only when both scale observations share a metric calibration or a
    sequence-consistent relative-depth scale.
    """

    tolerance: float = 0.02

    def __post_init__(self) -> None:
        if not math.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")

    def evaluate(
        self,
        previous: Object3DObservation,
        current: Object3DObservation,
    ) -> ResidualEvidence:
        """Return cross-frame scale stability for one stable track ID."""

        previous_id = previous.track_id or previous.source_object_2d_id
        current_id = current.track_id or current.source_object_2d_id
        source_ids = (
            f"{previous_id}:{previous.frame_index}",
            f"{current_id}:{current.frame_index}",
        )
        if previous_id != current_id:
            return ResidualEvidence.missing(
                "r_cross_frame_scale_stability",
                "object_track_id_mismatch",
                source_ids=source_ids,
            )
        if not previous.valid or not current.valid:
            return ResidualEvidence.missing(
                "r_cross_frame_scale_stability",
                "invalid_object_3d_observation",
                source_ids=source_ids,
            )
        try:
            previous_scale = previous.require_cross_frame_scale_comparable()
            current_scale = current.require_cross_frame_scale_comparable()
        except ValueError as exc:
            return ResidualEvidence.missing(
                "r_cross_frame_scale_stability",
                "blocked_by_cross_frame_scale_unavailable",
                source_ids=source_ids,
                metadata={"error": str(exc)},
            )
        if previous.scale_status != current.scale_status:
            return ResidualEvidence.missing(
                "r_cross_frame_scale_stability",
                "cross_frame_scale_domain_mismatch",
                source_ids=source_ids,
            )
        raw = abs(math.log(current_scale) - math.log(previous_scale))
        residual = max(0.0, raw - self.tolerance)
        return ResidualEvidence.observed(
            "r_cross_frame_scale_stability",
            residual,
            quality=min(
                previous.reconstruction_quality,
                current.reconstruction_quality,
                previous.scale_quality or 0.0,
                current.scale_quality or 0.0,
            ),
            source_ids=source_ids,
            metadata={
                "semantic_geometry_route": "same_object_cross_frame",
                "previous_scale": previous_scale,
                "current_scale": current_scale,
                "raw_log_scale_change": raw,
                "tolerance": self.tolerance,
                "formula": "max(0, abs(log(S_t/S_prev)) - tolerance)",
                "scale_status": previous.scale_status.value,
            },
        )
