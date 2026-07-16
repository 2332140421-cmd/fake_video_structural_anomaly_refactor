"""Multi-granularity residual structures and aggregation logic.

R_sd remains an object-pair residual from scale_depth.py. Pixel-, point-, and
region-level residuals are first pooled through object masks into single-object
residuals, then object-level and object-pair residuals are summarized at clip
level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .aggregation import (
    aggregate_map_by_mask,
    aggregate_points_by_mask,
    aggregate_values,
)
from .validity import MissingReason, ResidualEvidence


LEGACY_ZERO_MISSING_BEHAVIOR = True


@dataclass(frozen=True)
class ObjectMaskObservation:
    """Object mask used to pool low-level residuals into object-level values."""

    object_id: str
    label: str
    mask: np.ndarray
    confidence: float = 1.0

    def __post_init__(self) -> None:
        mask_array = np.asarray(self.mask).astype(bool)
        if mask_array.ndim != 2:
            raise ValueError(f"mask must be HxW, got shape {mask_array.shape}.")
        object.__setattr__(self, "mask", mask_array)

    @property
    def mask_area(self) -> int:
        """Return the number of pixels inside the object mask."""

        return int(np.sum(self.mask))

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return bounding box as (x1, y1, x2, y2), inclusive pixel bounds."""

        ys, xs = np.nonzero(self.mask)
        if xs.size == 0:
            return (0, 0, 0, 0)
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    @property
    def center(self) -> tuple[float, float]:
        """Return mask centroid as (x, y)."""

        ys, xs = np.nonzero(self.mask)
        if xs.size == 0:
            return (0.0, 0.0)
        return float(xs.mean()), float(ys.mean())


@dataclass(frozen=True)
class ObjectLevelResidual:
    """Legacy scalar residuals; absent sources are historically represented as zero."""

    object_id: str
    label: str
    flow: float
    track: float
    depth_cons: float
    corr: float
    confidence: float = 1.0


@dataclass(frozen=True)
class ObjectPairResidual:
    """Object-pair residuals, including R_sd from scale-depth consistency."""

    object_id_a: str
    object_id_b: str
    label_a: str
    label_b: str
    scale_depth: float
    occ: float = 0.0
    relative_motion: float = 0.0


@dataclass(frozen=True)
class ClipResidualSummary:
    """Clip-level residual summary."""

    object_residuals: list[ObjectLevelResidual]
    pair_residuals: list[ObjectPairResidual]
    clip_score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectLevelResidualEvidence:
    """Evidence-aware object residuals for new 3D-compatible modules."""

    object_id: str
    label: str
    flow: ResidualEvidence
    track: ResidualEvidence
    depth_cons: ResidualEvidence
    corr: ResidualEvidence
    confidence: float = 1.0


def _matrix_value(matrix: Optional[np.ndarray], i: int, j: int, name: str) -> float:
    """Fetch an optional NxN matrix value with validation."""

    if matrix is None:
        return 0.0
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be an NxN matrix, got shape {array.shape}.")
    if i >= array.shape[0] or j >= array.shape[1]:
        raise ValueError(f"{name} shape {array.shape} does not match object count.")
    return float(array[i, j])


def build_object_level_residuals(
    objects: Sequence[ObjectMaskObservation],
    flow_residual_map: Optional[np.ndarray] = None,
    depth_residual_map: Optional[np.ndarray] = None,
    corr_residual_map: Optional[np.ndarray] = None,
    track_points_xy: Optional[np.ndarray] = None,
    track_residuals: Optional[np.ndarray] = None,
    aggregation_method: str = "topk_mean",
) -> list[ObjectLevelResidual]:
    """Legacy pooling where absent maps historically become zero.

    New 3D code must call ``build_object_level_residual_evidence``.
    """

    residuals = []
    for obj in objects:
        flow = (
            aggregate_map_by_mask(flow_residual_map, obj.mask, method=aggregation_method)
            if flow_residual_map is not None
            else 0.0
        )
        depth_cons = (
            aggregate_map_by_mask(depth_residual_map, obj.mask, method=aggregation_method)
            if depth_residual_map is not None
            else 0.0
        )
        corr = (
            aggregate_map_by_mask(corr_residual_map, obj.mask, method=aggregation_method)
            if corr_residual_map is not None
            else 0.0
        )
        track = (
            aggregate_points_by_mask(
                track_points_xy,
                track_residuals,
                obj.mask,
                method=aggregation_method,
            )
            if track_points_xy is not None and track_residuals is not None
            else 0.0
        )
        residuals.append(
            ObjectLevelResidual(
                object_id=obj.object_id,
                label=obj.label,
                flow=flow,
                track=track,
                depth_cons=depth_cons,
                corr=corr,
                confidence=obj.confidence,
            )
        )
    return residuals


def build_object_level_residuals_with_details(
    objects: Sequence[ObjectMaskObservation],
    flow_residual_map: Optional[np.ndarray] = None,
    depth_residual_map: Optional[np.ndarray] = None,
    corr_residual_map: Optional[np.ndarray] = None,
    track_points_xy: Optional[np.ndarray] = None,
    track_residuals: Optional[np.ndarray] = None,
    aggregation_method: str = "topk_mean",
) -> tuple[list[ObjectLevelResidual], dict[str, Any]]:
    """Pool residuals and return missing-source details for diagnostics."""

    residuals = build_object_level_residuals(
        objects,
        flow_residual_map=flow_residual_map,
        depth_residual_map=depth_residual_map,
        corr_residual_map=corr_residual_map,
        track_points_xy=track_points_xy,
        track_residuals=track_residuals,
        aggregation_method=aggregation_method,
    )
    missing = []
    if flow_residual_map is None:
        missing.append("flow_residual_map")
    if depth_residual_map is None:
        missing.append("depth_residual_map")
    if corr_residual_map is None:
        missing.append("corr_residual_map")
    if track_points_xy is None or track_residuals is None:
        missing.append("track_points_xy/track_residuals")
    return residuals, {"missing_sources": missing, "aggregation_method": aggregation_method}


def _map_evidence(
    name: str,
    residual_map: Optional[np.ndarray],
    obj: ObjectMaskObservation,
    aggregation_method: str,
) -> ResidualEvidence:
    """Aggregate a present map or return explicit missing evidence."""

    source_id = f"object:{obj.object_id}"
    if residual_map is None:
        return ResidualEvidence.missing(
            name,
            MissingReason.MISSING_RESIDUAL_SOURCE,
            source_ids=(source_id,),
        )
    try:
        value = aggregate_map_by_mask(
            residual_map,
            obj.mask,
            method=aggregation_method,
            empty_policy="raise",
        )
    except ValueError as error:
        return ResidualEvidence.missing(
            name,
            "empty_or_invalid_object_region",
            source_ids=(source_id,),
            metadata={"error": str(error)},
        )
    return ResidualEvidence.observed(
        name,
        value,
        quality=min(1.0, max(0.0, float(obj.confidence))),
        source_ids=(source_id,),
    )


def build_object_level_residual_evidence(
    objects: Sequence[ObjectMaskObservation],
    flow_residual_map: Optional[np.ndarray] = None,
    depth_residual_map: Optional[np.ndarray] = None,
    corr_residual_map: Optional[np.ndarray] = None,
    track_points_xy: Optional[np.ndarray] = None,
    track_residuals: Optional[np.ndarray] = None,
    aggregation_method: str = "topk_mean",
) -> list[ObjectLevelResidualEvidence]:
    """Pool residuals with NaN + validity semantics instead of legacy zeros."""

    output: list[ObjectLevelResidualEvidence] = []
    for obj in objects:
        if track_points_xy is None or track_residuals is None:
            track = ResidualEvidence.missing(
                "track",
                MissingReason.MISSING_RESIDUAL_SOURCE,
                source_ids=(f"object:{obj.object_id}",),
            )
        else:
            try:
                value = aggregate_points_by_mask(
                    track_points_xy,
                    track_residuals,
                    obj.mask,
                    method=aggregation_method,
                    empty_policy="raise",
                )
                track = ResidualEvidence.observed(
                    "track",
                    value,
                    quality=min(1.0, max(0.0, float(obj.confidence))),
                    source_ids=(f"object:{obj.object_id}",),
                )
            except ValueError as error:
                track = ResidualEvidence.missing(
                    "track",
                    "empty_or_invalid_track_region",
                    source_ids=(f"object:{obj.object_id}",),
                    metadata={"error": str(error)},
                )
        output.append(
            ObjectLevelResidualEvidence(
                object_id=obj.object_id,
                label=obj.label,
                flow=_map_evidence("flow", flow_residual_map, obj, aggregation_method),
                track=track,
                depth_cons=_map_evidence(
                    "depth_cons", depth_residual_map, obj, aggregation_method
                ),
                corr=_map_evidence("corr", corr_residual_map, obj, aggregation_method),
                confidence=obj.confidence,
            )
        )
    return output


def build_object_pair_residuals(
    objects: Sequence[ObjectMaskObservation],
    scale_depth_matrix: Optional[np.ndarray] = None,
    occ_matrix: Optional[np.ndarray] = None,
    relative_motion_matrix: Optional[np.ndarray] = None,
) -> list[ObjectPairResidual]:
    """Convert NxN object-pair matrices to unique ObjectPairResidual records."""

    pair_residuals = []
    for i, obj_a in enumerate(objects):
        for j in range(i + 1, len(objects)):
            obj_b = objects[j]
            pair_residuals.append(
                ObjectPairResidual(
                    object_id_a=obj_a.object_id,
                    object_id_b=obj_b.object_id,
                    label_a=obj_a.label,
                    label_b=obj_b.label,
                    scale_depth=_matrix_value(
                        scale_depth_matrix, i, j, "scale_depth_matrix"
                    ),
                    occ=_matrix_value(occ_matrix, i, j, "occ_matrix"),
                    relative_motion=_matrix_value(
                        relative_motion_matrix, i, j, "relative_motion_matrix"
                    ),
                )
            )
    return pair_residuals


def summarize_clip_residuals(
    object_residuals: Sequence[ObjectLevelResidual],
    pair_residuals: Sequence[ObjectPairResidual],
    weights: Optional[Mapping[str, float]] = None,
    aggregation_method: str = "topk_mean",
) -> ClipResidualSummary:
    """Fuse object-level and pair-level residual scores into a clip score."""

    default_weights = {
        "object_flow": 1.0,
        "object_track": 1.0,
        "object_depth_cons": 1.0,
        "object_corr": 1.0,
        "pair_scale_depth": 1.0,
        "pair_occ": 1.0,
        "pair_relative_motion": 1.0,
    }
    if weights is not None:
        default_weights.update({key: float(value) for key, value in weights.items()})

    object_scores = {
        residual.object_id: (
            default_weights["object_flow"] * residual.flow
            + default_weights["object_track"] * residual.track
            + default_weights["object_depth_cons"] * residual.depth_cons
            + default_weights["object_corr"] * residual.corr
        )
        for residual in object_residuals
    }
    pair_scores = {
        f"{residual.object_id_a}->{residual.object_id_b}": (
            default_weights["pair_scale_depth"] * residual.scale_depth
            + default_weights["pair_occ"] * residual.occ
            + default_weights["pair_relative_motion"] * residual.relative_motion
        )
        for residual in pair_residuals
    }

    all_scores = list(object_scores.values()) + list(pair_scores.values())
    clip_score = aggregate_values(all_scores, method=aggregation_method)
    details = {
        "object_scores": object_scores,
        "pair_scores": pair_scores,
        "aggregation_method": aggregation_method,
        "weights": dict(default_weights),
    }
    return ClipResidualSummary(
        object_residuals=list(object_residuals),
        pair_residuals=list(pair_residuals),
        clip_score=clip_score,
        details=details,
    )
