"""Robust sequence depth alignment without semantic physical-size anchors."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from ..depth_provider import DepthObservation
from .observation import DepthAlignmentMode, DepthAlignmentObservation


def _fit_linear(x: np.ndarray, y: np.ndarray, mode: DepthAlignmentMode) -> tuple[float, float]:
    if mode == DepthAlignmentMode.SCALE_ONLY:
        denominator = float(np.dot(x, x))
        if denominator <= 1e-12:
            raise ValueError("Degenerate scale-only support.")
        return float(np.dot(x, y) / denominator), 0.0
    if mode in {
        DepthAlignmentMode.AFFINE_DEPTH,
        DepthAlignmentMode.AFFINE_INVERSE_DEPTH,
    }:
        design = np.column_stack((x, np.ones_like(x)))
        coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank < 2:
            raise ValueError("Degenerate affine alignment support.")
        return float(coefficients[0]), float(coefficients[1])
    if mode == DepthAlignmentMode.PROVIDER_SHARED_SCALE:
        return 1.0, 0.0
    raise ValueError(f"Unsupported alignment mode: {mode.value}")


def estimate_depth_alignment(
    source_values: Sequence[float] | np.ndarray,
    target_values: Sequence[float] | np.ndarray,
    *,
    source_frame: int,
    target_frame: int,
    mode: DepthAlignmentMode | str,
    minimum_support: int = 12,
    maximum_iterations: int = 8,
    mad_threshold: float = 3.5,
    values_are_alignment_domain: bool = False,
    metadata: Optional[dict[str, object]] = None,
) -> DepthAlignmentObservation:
    """Fit a robust source-to-target depth mapping from stable background samples."""

    alignment_mode = DepthAlignmentMode(mode)
    if alignment_mode == DepthAlignmentMode.UNSUPPORTED:
        return DepthAlignmentObservation.missing(
            source_frame,
            target_frame,
            "unsupported_depth_alignment_mode",
            mode=alignment_mode,
            metadata=metadata,
        )
    source = np.asarray(source_values, dtype=float).reshape(-1)
    target = np.asarray(target_values, dtype=float).reshape(-1)
    if source.shape != target.shape:
        raise ValueError("source_values and target_values must have the same shape.")
    valid = np.isfinite(source) & np.isfinite(target) & (source > 0.0) & (target > 0.0)
    source, target = source[valid], target[valid]
    support_count = int(source.size)
    if support_count < minimum_support:
        return DepthAlignmentObservation.missing(
            source_frame,
            target_frame,
            "insufficient_background_depth_support",
            mode=alignment_mode,
            support_count=support_count,
            metadata=metadata,
        )
    if (
        alignment_mode == DepthAlignmentMode.AFFINE_INVERSE_DEPTH
        and not values_are_alignment_domain
    ):
        x, y = 1.0 / source, 1.0 / target
    else:
        x, y = source, target
    if alignment_mode == DepthAlignmentMode.PROVIDER_SHARED_SCALE:
        residual = y - x
        error = float(np.sqrt(np.mean(np.square(residual))))
        scale_reference = max(float(np.median(np.abs(y))), 1e-12)
        quality = float(1.0 / (1.0 + error / scale_reference))
        return DepthAlignmentObservation(
            source_frame=source_frame,
            target_frame=target_frame,
            alignment_mode=alignment_mode,
            scale=1.0,
            shift=0.0,
            support_count=support_count,
            inlier_ratio=1.0,
            fitting_error=error,
            quality=quality,
            valid=True,
            metadata={
                **dict(metadata or {}),
                "fit_domain": "depth",
                "mapping": "target_domain = scale * source_domain + shift",
                "provider_shared_scale_asserted": True,
            },
        )

    inliers = np.ones(source.size, dtype=bool)
    scale = shift = math.nan
    for _ in range(maximum_iterations):
        if int(np.sum(inliers)) < minimum_support:
            break
        try:
            scale, shift = _fit_linear(x[inliers], y[inliers], alignment_mode)
        except ValueError:
            break
        residuals = y - (scale * x + shift)
        centre = float(np.median(residuals[inliers]))
        mad = float(np.median(np.abs(residuals[inliers] - centre)))
        robust_sigma = 1.4826 * mad
        floor = max(1e-8, 1e-4 * float(np.median(np.abs(y[inliers]))))
        threshold = max(floor, mad_threshold * robust_sigma)
        updated = np.abs(residuals - centre) <= threshold
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    inlier_count = int(np.sum(inliers))
    if (
        inlier_count < minimum_support
        or not math.isfinite(scale)
        or not math.isfinite(shift)
        or scale <= 0.0
    ):
        return DepthAlignmentObservation.missing(
            source_frame,
            target_frame,
            "robust_depth_alignment_failed",
            mode=alignment_mode,
            support_count=support_count,
            metadata={
                **dict(metadata or {}),
                "inlier_count": inlier_count,
            },
        )
    fitted = scale * x[inliers] + shift
    error = float(np.sqrt(np.mean(np.square(y[inliers] - fitted))))
    reference = max(float(np.median(np.abs(y[inliers]))), 1e-12)
    normalized_error = error / reference
    inlier_ratio = inlier_count / support_count
    quality = float(np.clip(inlier_ratio / (1.0 + normalized_error), 0.0, 1.0))
    return DepthAlignmentObservation(
        source_frame=source_frame,
        target_frame=target_frame,
        alignment_mode=alignment_mode,
        scale=scale,
        shift=shift,
        support_count=support_count,
        inlier_ratio=inlier_ratio,
        fitting_error=error,
        quality=quality,
        valid=True,
        metadata={
            **dict(metadata or {}),
            "fit_domain": (
                "inverse_depth"
                if alignment_mode == DepthAlignmentMode.AFFINE_INVERSE_DEPTH
                else "depth"
            ),
            "values_are_alignment_domain": values_are_alignment_domain,
            "mapping": "target_domain = scale * source_domain + shift",
            "inlier_count": inlier_count,
            "normalized_fitting_error": normalized_error,
            "semantic_physical_prior_used": False,
            "detection_object_used_as_scale_anchor": False,
        },
    )


def _sample_nearest(
    depth: np.ndarray,
    points: np.ndarray,
    valid_mask: np.ndarray,
    foreground_mask: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    columns = np.floor(points[:, 0] + 0.5).astype(int)
    rows = np.floor(points[:, 1] + 0.5).astype(int)
    inside = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    values = np.full(points.shape[0], np.nan, dtype=float)
    accepted = np.zeros(points.shape[0], dtype=bool)
    valid_indices = np.flatnonzero(inside)
    if valid_indices.size:
        r, c = rows[valid_indices], columns[valid_indices]
        local_valid = valid_mask[r, c]
        if foreground_mask is not None:
            local_valid &= ~foreground_mask[r, c]
        selected = valid_indices[local_valid]
        values[selected] = depth[rows[selected], columns[selected]]
        accepted[selected] = True
    return values, accepted


def estimate_depth_alignment_from_correspondences(
    source_depth: DepthObservation,
    target_depth: DepthObservation,
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    source_frame: int,
    target_frame: int,
    mode: DepthAlignmentMode | str,
    source_foreground_mask: Optional[np.ndarray] = None,
    target_foreground_mask: Optional[np.ndarray] = None,
    minimum_support: int = 12,
    metadata: Optional[dict[str, object]] = None,
) -> DepthAlignmentObservation:
    """Fit alignment from paired background pixels in two canonical depth maps."""

    source_map = source_depth.require_geometry_depth()
    target_map = target_depth.require_geometry_depth()
    source_xy = np.asarray(source_points, dtype=float).reshape(-1, 2)
    target_xy = np.asarray(target_points, dtype=float).reshape(-1, 2)
    if source_xy.shape != target_xy.shape:
        raise ValueError("source_points and target_points must have matching Nx2 shape.")
    if source_foreground_mask is not None and source_foreground_mask.shape != source_map.shape:
        raise ValueError("source_foreground_mask shape must match source depth.")
    if target_foreground_mask is not None and target_foreground_mask.shape != target_map.shape:
        raise ValueError("target_foreground_mask shape must match target depth.")
    source_valid = (
        np.asarray(source_depth.valid_mask, dtype=bool)
        if source_depth.valid_mask is not None
        else np.isfinite(source_map) & (source_map > 0.0)
    )
    target_valid = (
        np.asarray(target_depth.valid_mask, dtype=bool)
        if target_depth.valid_mask is not None
        else np.isfinite(target_map) & (target_map > 0.0)
    )
    source_values, source_accepted = _sample_nearest(
        source_map, source_xy, source_valid, source_foreground_mask
    )
    target_values, target_accepted = _sample_nearest(
        target_map, target_xy, target_valid, target_foreground_mask
    )
    accepted = source_accepted & target_accepted
    return estimate_depth_alignment(
        source_values[accepted],
        target_values[accepted],
        source_frame=source_frame,
        target_frame=target_frame,
        mode=mode,
        minimum_support=minimum_support,
        metadata={
            **dict(metadata or {}),
            "candidate_correspondence_count": int(source_xy.shape[0]),
            "background_depth_correspondence_count": int(np.sum(accepted)),
            "foreground_excluded": True,
        },
    )


def apply_depth_alignment(
    source_depth: np.ndarray,
    alignment: DepthAlignmentObservation,
) -> np.ndarray:
    """Apply a valid source-to-target alignment while preserving invalid pixels."""

    if not alignment.valid:
        raise ValueError(f"Cannot apply invalid alignment: {alignment.missing_reason}")
    source = np.asarray(source_depth, dtype=float)
    output = np.full(source.shape, np.nan, dtype=float)
    valid = np.isfinite(source) & (source > 0.0)
    if alignment.alignment_mode == DepthAlignmentMode.AFFINE_INVERSE_DEPTH:
        inverse = alignment.scale / source[valid] + alignment.shift
        positive = inverse > 1e-12
        valid_indices = np.flatnonzero(valid)
        output_flat = output.reshape(-1)
        output_flat[valid_indices[positive]] = 1.0 / inverse[positive]
    else:
        aligned = alignment.scale * source[valid] + alignment.shift
        positive = aligned > 0.0
        valid_indices = np.flatnonzero(valid)
        output_flat = output.reshape(-1)
        output_flat[valid_indices[positive]] = aligned[positive]
    return output


@dataclass(frozen=True)
class DepthAlignmentModelSelection:
    """Candidate depth models and the holdout-selected edge."""

    source_frame: int
    target_frame: int
    candidates: tuple[DepthAlignmentObservation, ...]
    selected: DepthAlignmentObservation
    fitting_count: int
    holdout_count: int
    raw_model_output_used: bool
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.fitting_count < 0 or self.holdout_count < 0:
            raise ValueError("Depth fitting/holdout counts must be non-negative.")
        if self.valid and (not self.selected.valid or self.missing_reason):
            raise ValueError("Valid selection requires a valid selected model.")
        if not self.valid and not self.missing_reason:
            raise ValueError("Invalid selection requires missing_reason.")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "metadata", dict(self.metadata))


class GlobalDepthAlignmentMethod(str, Enum):
    """How pairwise depth edges are tied to a sequence reference."""

    DIRECT_TO_REFERENCE = "direct_to_reference"
    PAIRWISE_GRAPH_PROPAGATION = "pairwise_graph_propagation"
    ROBUST_GLOBAL_REFINEMENT = "robust_global_refinement"


@dataclass(frozen=True)
class PerFrameDepthAlignment:
    """Mapping from one frame's selected domain into the reference domain."""

    frame_index: int
    reference_frame: int
    scale: float
    shift: float
    alignment_domain: str
    supporting_edges: tuple[tuple[int, int], ...]
    valid: bool
    quality: float
    missing_reason: str = ""

    def __post_init__(self) -> None:
        if self.valid:
            if not all(math.isfinite(value) for value in (self.scale, self.shift)):
                raise ValueError("Valid per-frame depth alignment requires finite parameters.")
            if self.scale <= 0.0 or self.missing_reason:
                raise ValueError("Valid per-frame depth alignment requires scale > 0.")
        else:
            if not (math.isnan(self.scale) and math.isnan(self.shift)):
                raise ValueError("Invalid per-frame alignment parameters must be NaN.")
            if not self.missing_reason:
                raise ValueError("Invalid per-frame alignment requires missing_reason.")
        if not math.isfinite(self.quality) or not 0.0 <= self.quality <= 1.0:
            raise ValueError("Per-frame alignment quality must be in [0, 1].")


@dataclass(frozen=True)
class SequenceDepthAlignmentResult:
    """Sequence-wide depth alignment graph rooted at a reference frame."""

    frame_indices: tuple[int, ...]
    reference_frame: int
    method: GlobalDepthAlignmentMethod | str
    alignment_mode: DepthAlignmentMode | str
    alignment_domain: str
    per_frame: Mapping[int, PerFrameDepthAlignment]
    supporting_edges: tuple[DepthAlignmentObservation, ...]
    global_consistency_error: float
    scale_drift_before: float
    scale_drift_after: float
    connected_frame_ratio: float
    valid: bool
    quality: float
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        method = GlobalDepthAlignmentMethod(self.method)
        mode = DepthAlignmentMode(self.alignment_mode)
        for name in ("global_consistency_error", "scale_drift_before", "scale_drift_after"):
            value = float(getattr(self, name))
            if not (math.isnan(value) or (math.isfinite(value) and value >= 0.0)):
                raise ValueError(f"{name} must be non-negative or NaN.")
            object.__setattr__(self, name, value)
        for name in ("connected_frame_ratio", "quality"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
            object.__setattr__(self, name, value)
        if self.valid and self.missing_reason:
            raise ValueError("Valid sequence depth alignment cannot have missing_reason.")
        if not self.valid and not self.missing_reason:
            raise ValueError("Invalid sequence depth alignment requires missing_reason.")
        object.__setattr__(self, "frame_indices", tuple(self.frame_indices))
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "alignment_mode", mode)
        object.__setattr__(self, "per_frame", dict(self.per_frame))
        object.__setattr__(self, "supporting_edges", tuple(self.supporting_edges))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return sequence alignment diagnostics for JSON output."""

        return {
            "frame_indices": list(self.frame_indices),
            "reference_frame": self.reference_frame,
            "method": self.method.value,
            "alignment_mode": self.alignment_mode.value,
            "alignment_domain": self.alignment_domain,
            "per_frame": {
                str(index): {
                    "frame_index": item.frame_index,
                    "reference_frame": item.reference_frame,
                    "scale": None if not item.valid else item.scale,
                    "shift": None if not item.valid else item.shift,
                    "alignment_domain": item.alignment_domain,
                    "supporting_edges": [list(edge) for edge in item.supporting_edges],
                    "valid": item.valid,
                    "quality": item.quality,
                    "missing_reason": item.missing_reason,
                }
                for index, item in self.per_frame.items()
            },
            "supporting_edges": [
                {
                    "source_frame": item.source_frame,
                    "target_frame": item.target_frame,
                    "mode": item.alignment_mode.value,
                    "scale": item.scale if item.valid else None,
                    "shift": item.shift if item.valid else None,
                    "fitting_error": item.fitting_error if item.valid else None,
                    "holdout_error": item.holdout_error if item.valid else None,
                    "valid": item.valid,
                    "missing_reason": item.missing_reason,
                }
                for item in self.supporting_edges
            ],
            "global_consistency_error": self.global_consistency_error,
            "scale_drift_before": self.scale_drift_before,
            "scale_drift_after": self.scale_drift_after,
            "connected_frame_ratio": self.connected_frame_ratio,
            "valid": self.valid,
            "quality": self.quality,
            "missing_reason": self.missing_reason,
            "metadata": dict(self.metadata),
        }


def _alignment_domain_values(
    source_values: np.ndarray,
    target_values: np.ndarray,
    mode: DepthAlignmentMode,
    *,
    values_are_alignment_domain: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == DepthAlignmentMode.AFFINE_INVERSE_DEPTH and not values_are_alignment_domain:
        return 1.0 / source_values, 1.0 / target_values
    return source_values, target_values


def estimate_depth_alignment_with_holdout(
    source_values: Sequence[float] | np.ndarray,
    target_values: Sequence[float] | np.ndarray,
    *,
    source_frame: int,
    target_frame: int,
    mode: DepthAlignmentMode | str,
    minimum_fitting_support: int = 12,
    minimum_holdout_support: int = 4,
    holdout_fraction: float = 0.25,
    values_are_alignment_domain: bool = False,
    maximum_normalized_holdout_error: float = 0.10,
    metadata: Mapping[str, Any] | None = None,
) -> DepthAlignmentObservation:
    """Fit on one deterministic subset and validate on a disjoint holdout set."""

    alignment_mode = DepthAlignmentMode(mode)
    source = np.asarray(source_values, dtype=float).reshape(-1)
    target = np.asarray(target_values, dtype=float).reshape(-1)
    if source.shape != target.shape:
        raise ValueError("source_values and target_values must have matching shape.")
    valid = np.isfinite(source) & np.isfinite(target) & (source > 0.0) & (target > 0.0)
    source, target = source[valid], target[valid]
    required = minimum_fitting_support + minimum_holdout_support
    if source.size < required:
        return DepthAlignmentObservation.missing(
            source_frame,
            target_frame,
            "insufficient_fitting_and_holdout_support",
            mode=alignment_mode,
            support_count=int(source.size),
            metadata=dict(metadata or {}),
        )
    holdout_step = max(2, int(round(1.0 / max(min(holdout_fraction, 0.5), 1e-6))))
    indices = np.arange(source.size)
    holdout_mask = indices % holdout_step == 0
    if int(np.sum(holdout_mask)) < minimum_holdout_support:
        holdout_mask[:minimum_holdout_support] = True
    fitting_mask = ~holdout_mask
    if int(np.sum(fitting_mask)) < minimum_fitting_support:
        return DepthAlignmentObservation.missing(
            source_frame,
            target_frame,
            "insufficient_fitting_support_after_split",
            mode=alignment_mode,
            support_count=int(source.size),
            metadata=dict(metadata or {}),
        )
    fitted = estimate_depth_alignment(
        source[fitting_mask],
        target[fitting_mask],
        source_frame=source_frame,
        target_frame=target_frame,
        mode=alignment_mode,
        minimum_support=minimum_fitting_support,
        values_are_alignment_domain=values_are_alignment_domain,
        metadata={
            **dict(metadata or {}),
            "fitting_holdout_split": "deterministic_interleaved",
            "visualization_depth_used": False,
        },
    )
    if not fitted.valid:
        return fitted
    source_domain, target_domain = _alignment_domain_values(
        source[holdout_mask],
        target[holdout_mask],
        alignment_mode,
        values_are_alignment_domain=values_are_alignment_domain,
    )
    predicted = fitted.scale * source_domain + fitted.shift
    physical_valid = bool(
        np.isfinite(predicted).all()
        and np.all(predicted > 1e-12)
        and fitted.scale > 0.0
    )
    holdout_error = float(np.sqrt(np.mean(np.square(target_domain - predicted))))
    reference = max(float(np.median(np.abs(target_domain))), 1e-12)
    normalized_holdout = holdout_error / reference
    if not physical_valid or normalized_holdout > maximum_normalized_holdout_error:
        return DepthAlignmentObservation.missing(
            source_frame,
            target_frame,
            (
                "non_physical_depth_alignment"
                if not physical_valid
                else "depth_alignment_holdout_rejected"
            ),
            mode=alignment_mode,
            support_count=int(source.size),
            metadata={
                **dict(fitted.metadata),
                "fitting_error_before_rejection": fitted.fitting_error,
                "holdout_error_before_rejection": holdout_error,
                "normalized_holdout_error": normalized_holdout,
                "fitting_count": int(np.sum(fitting_mask)),
                "holdout_count": int(np.sum(holdout_mask)),
            },
        )
    quality = float(
        np.clip(
            fitted.inlier_ratio / (1.0 + normalized_holdout),
            0.0,
            1.0,
        )
    )
    return replace(
        fitted,
        support_count=int(source.size),
        holdout_error=holdout_error,
        holdout_count=int(np.sum(holdout_mask)),
        physical_valid=True,
        quality=quality,
        metadata={
            **dict(fitted.metadata),
            "fitting_count": int(np.sum(fitting_mask)),
            "holdout_count": int(np.sum(holdout_mask)),
            "normalized_holdout_error": normalized_holdout,
            "monotonic_positive_mapping": True,
        },
    )


def select_depth_alignment_model(
    source_depth_values: Sequence[float] | np.ndarray,
    target_depth_values: Sequence[float] | np.ndarray,
    *,
    source_frame: int,
    target_frame: int,
    source_raw_values: Sequence[float] | np.ndarray | None = None,
    target_raw_values: Sequence[float] | np.ndarray | None = None,
    minimum_fitting_support: int = 12,
    minimum_holdout_support: int = 4,
    simplicity_tolerance: float = 1.05,
    metadata: Mapping[str, Any] | None = None,
) -> DepthAlignmentModelSelection:
    """Select the simplest physically valid model near the best holdout error."""

    source_depth = np.asarray(source_depth_values, dtype=float).reshape(-1)
    target_depth = np.asarray(target_depth_values, dtype=float).reshape(-1)
    candidates: list[DepthAlignmentObservation] = []
    for mode in (DepthAlignmentMode.SCALE_ONLY, DepthAlignmentMode.AFFINE_DEPTH):
        candidates.append(
            estimate_depth_alignment_with_holdout(
                source_depth,
                target_depth,
                source_frame=source_frame,
                target_frame=target_frame,
                mode=mode,
                minimum_fitting_support=minimum_fitting_support,
                minimum_holdout_support=minimum_holdout_support,
                metadata={**dict(metadata or {}), "candidate_domain": "geometry_z_depth"},
            )
        )
    raw_used = source_raw_values is not None and target_raw_values is not None
    inverse_source = (
        np.asarray(source_raw_values, dtype=float).reshape(-1)
        if raw_used
        else source_depth
    )
    inverse_target = (
        np.asarray(target_raw_values, dtype=float).reshape(-1)
        if raw_used
        else target_depth
    )
    candidates.append(
        estimate_depth_alignment_with_holdout(
            inverse_source,
            inverse_target,
            source_frame=source_frame,
            target_frame=target_frame,
            mode=DepthAlignmentMode.AFFINE_INVERSE_DEPTH,
            minimum_fitting_support=minimum_fitting_support,
            minimum_holdout_support=minimum_holdout_support,
            values_are_alignment_domain=raw_used,
            metadata={
                **dict(metadata or {}),
                "candidate_domain": (
                    "raw_model_output_inverse_like" if raw_used else "reciprocal_z_depth"
                ),
                "raw_model_output_used": raw_used,
                "visualization_depth_used": False,
            },
        )
    )
    valid = [candidate for candidate in candidates if candidate.valid and candidate.physical_valid]
    if not valid:
        missing = DepthAlignmentObservation.missing(
            source_frame,
            target_frame,
            "all_depth_alignment_models_rejected",
            metadata={
                "candidate_reasons": [candidate.missing_reason for candidate in candidates],
            },
        )
        return DepthAlignmentModelSelection(
            source_frame,
            target_frame,
            tuple(candidates),
            missing,
            0,
            0,
            raw_used,
            False,
            "all_depth_alignment_models_rejected",
        )
    normalized = {
        id(candidate): float(candidate.metadata.get("normalized_holdout_error", math.inf))
        for candidate in valid
    }
    best_error = min(normalized.values())
    near_best = [
        candidate
        for candidate in valid
        if normalized[id(candidate)] <= best_error * simplicity_tolerance + 1e-12
    ]
    complexity = {
        DepthAlignmentMode.SCALE_ONLY: 0,
        DepthAlignmentMode.AFFINE_DEPTH: 1,
        DepthAlignmentMode.AFFINE_INVERSE_DEPTH: 1,
        DepthAlignmentMode.PROVIDER_SHARED_SCALE: 0,
    }
    selected = min(
        near_best,
        key=lambda candidate: (
            complexity[candidate.alignment_mode],
            normalized[id(candidate)],
        ),
    )
    return DepthAlignmentModelSelection(
        source_frame,
        target_frame,
        tuple(candidates),
        selected,
        int(selected.metadata.get("fitting_count", 0)),
        selected.holdout_count,
        bool(selected.metadata.get("raw_model_output_used", False)),
        True,
        metadata={
            "selection_rule": "simplest_model_within_holdout_tolerance",
            "simplicity_tolerance": simplicity_tolerance,
            "semantic_scale_prior_used": False,
            "anomaly_residual_used": False,
        },
    )


def _resized_raw_output(observation: DepthObservation) -> Optional[np.ndarray]:
    if observation.raw_model_output is None or observation.depth_map is None:
        return None
    raw = np.asarray(observation.raw_model_output, dtype=float)
    if raw.shape != observation.depth_map.shape:
        raw = cv2.resize(
            raw.astype(np.float32),
            (observation.depth_map.shape[1], observation.depth_map.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(float)
    return raw


def select_depth_alignment_from_correspondences(
    source_depth: DepthObservation,
    target_depth: DepthObservation,
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    source_frame: int,
    target_frame: int,
    source_foreground_mask: Optional[np.ndarray] = None,
    target_foreground_mask: Optional[np.ndarray] = None,
    minimum_fitting_support: int = 12,
    minimum_holdout_support: int = 4,
    metadata: Mapping[str, Any] | None = None,
) -> DepthAlignmentModelSelection:
    """Evaluate all supported domains on stable background correspondences."""

    source_map = source_depth.require_geometry_depth()
    target_map = target_depth.require_geometry_depth()
    source_xy = np.asarray(source_points, dtype=float).reshape(-1, 2)
    target_xy = np.asarray(target_points, dtype=float).reshape(-1, 2)
    source_valid = np.asarray(source_depth.valid_mask, dtype=bool)
    target_valid = np.asarray(target_depth.valid_mask, dtype=bool)
    source_values, source_ok = _sample_nearest(
        source_map, source_xy, source_valid, source_foreground_mask
    )
    target_values, target_ok = _sample_nearest(
        target_map, target_xy, target_valid, target_foreground_mask
    )
    accepted = source_ok & target_ok
    raw_source_map = _resized_raw_output(source_depth)
    raw_target_map = _resized_raw_output(target_depth)
    raw_source_values = raw_target_values = None
    if raw_source_map is not None and raw_target_map is not None:
        raw_source_valid = np.isfinite(raw_source_map) & (raw_source_map > 0.0)
        raw_target_valid = np.isfinite(raw_target_map) & (raw_target_map > 0.0)
        raw_source_all, raw_source_ok = _sample_nearest(
            raw_source_map,
            source_xy,
            raw_source_valid,
            source_foreground_mask,
        )
        raw_target_all, raw_target_ok = _sample_nearest(
            raw_target_map,
            target_xy,
            raw_target_valid,
            target_foreground_mask,
        )
        accepted &= raw_source_ok & raw_target_ok
        raw_source_values = raw_source_all[accepted]
        raw_target_values = raw_target_all[accepted]
    return select_depth_alignment_model(
        source_values[accepted],
        target_values[accepted],
        source_frame=source_frame,
        target_frame=target_frame,
        source_raw_values=raw_source_values,
        target_raw_values=raw_target_values,
        minimum_fitting_support=minimum_fitting_support,
        minimum_holdout_support=minimum_holdout_support,
        metadata={
            **dict(metadata or {}),
            "candidate_correspondence_count": int(source_xy.shape[0]),
            "background_depth_correspondence_count": int(np.sum(accepted)),
            "foreground_excluded": True,
            "visualization_depth_used": False,
        },
    )


def _solve_global_affine_parameters(
    frame_indices: tuple[int, ...],
    reference: int,
    edges: Sequence[DepthAlignmentObservation],
) -> tuple[dict[int, tuple[float, float]], float]:
    variables = [index for index in frame_indices if index != reference]
    variable_index = {frame: position for position, frame in enumerate(variables)}
    if not variables:
        return {reference: (1.0, 0.0)}, 0.0
    rows: list[np.ndarray] = []
    targets: list[float] = []
    weights: list[float] = []
    for edge in edges:
        row = np.zeros(len(variables), dtype=float)
        if edge.target_frame != reference:
            row[variable_index[edge.target_frame]] += 1.0
        if edge.source_frame != reference:
            row[variable_index[edge.source_frame]] -= 1.0
        rows.append(row)
        targets.append(-math.log(edge.scale))
        weights.append(max(edge.quality, 1e-3))
    matrix = np.asarray(rows)
    target = np.asarray(targets)
    root_weights = np.sqrt(np.asarray(weights))
    log_scales, *_ = np.linalg.lstsq(
        matrix * root_weights[:, None], target * root_weights, rcond=None
    )
    scales = {reference: 1.0}
    scales.update({frame: float(math.exp(log_scales[pos])) for frame, pos in variable_index.items()})
    shift_rows: list[np.ndarray] = []
    shift_targets: list[float] = []
    for edge in edges:
        row = np.zeros(len(variables), dtype=float)
        if edge.target_frame != reference:
            row[variable_index[edge.target_frame]] += 1.0
        if edge.source_frame != reference:
            row[variable_index[edge.source_frame]] -= 1.0
        shift_rows.append(row)
        shift_targets.append(-scales[edge.source_frame] * edge.shift / edge.scale)
    shift_matrix = np.asarray(shift_rows)
    shifts_vector, *_ = np.linalg.lstsq(
        shift_matrix * root_weights[:, None],
        np.asarray(shift_targets) * root_weights,
        rcond=None,
    )
    shifts = {reference: 0.0}
    shifts.update({frame: float(shifts_vector[pos]) for frame, pos in variable_index.items()})
    residuals = []
    for edge in edges:
        residuals.append(
            (math.log(scales[edge.target_frame]) - math.log(scales[edge.source_frame]))
            + math.log(edge.scale)
        )
    consistency = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else 0.0
    return {frame: (scales[frame], shifts[frame]) for frame in frame_indices}, consistency


def align_depth_sequence_to_reference(
    frame_indices: Sequence[int],
    selections: Sequence[DepthAlignmentModelSelection],
    *,
    reference_frame: Optional[int] = None,
    minimum_edge_quality: float = 0.20,
    method: GlobalDepthAlignmentMethod | str = GlobalDepthAlignmentMethod.ROBUST_GLOBAL_REFINEMENT,
) -> SequenceDepthAlignmentResult:
    """Build a depth-alignment graph so one failed adjacent edge is not fatal."""

    indices = tuple(int(index) for index in frame_indices)
    if not indices:
        raise ValueError("frame_indices cannot be empty.")
    reference = indices[0] if reference_frame is None else int(reference_frame)
    selected_method = GlobalDepthAlignmentMethod(method)
    modes = (
        DepthAlignmentMode.SCALE_ONLY,
        DepthAlignmentMode.AFFINE_DEPTH,
        DepthAlignmentMode.AFFINE_INVERSE_DEPTH,
    )
    score_by_mode: dict[DepthAlignmentMode, float] = {mode: 0.0 for mode in modes}
    candidates_by_mode: dict[DepthAlignmentMode, list[DepthAlignmentObservation]] = {
        mode: [] for mode in modes
    }
    for selection in selections:
        for candidate in selection.candidates:
            if candidate.valid and candidate.quality >= minimum_edge_quality:
                candidates_by_mode[candidate.alignment_mode].append(candidate)
                score_by_mode[candidate.alignment_mode] += candidate.quality
    connected_by_mode: dict[DepthAlignmentMode, set[int]] = {}
    mean_holdout_by_mode: dict[DepthAlignmentMode, float] = {}
    for mode in modes:
        mode_adjacency: dict[int, set[int]] = {index: set() for index in indices}
        for edge in candidates_by_mode[mode]:
            mode_adjacency[edge.source_frame].add(edge.target_frame)
            mode_adjacency[edge.target_frame].add(edge.source_frame)
        mode_connected = {reference}
        mode_queue = [reference]
        while mode_queue:
            current = mode_queue.pop(0)
            for neighbour in mode_adjacency[current]:
                if neighbour not in mode_connected:
                    mode_connected.add(neighbour)
                    mode_queue.append(neighbour)
        connected_by_mode[mode] = mode_connected
        connected_edges = [
            edge
            for edge in candidates_by_mode[mode]
            if edge.source_frame in mode_connected and edge.target_frame in mode_connected
        ]
        mean_holdout_by_mode[mode] = _mean_finite(
            [
                float(edge.metadata.get("normalized_holdout_error", math.nan))
                for edge in connected_edges
            ]
        )
    maximum_coverage = max(len(connected_by_mode[mode]) for mode in modes)
    coverage_modes = [
        mode for mode in modes if len(connected_by_mode[mode]) == maximum_coverage
    ]
    finite_holdout = [
        mean_holdout_by_mode[mode]
        for mode in coverage_modes
        if math.isfinite(mean_holdout_by_mode[mode])
    ]
    best_holdout = min(finite_holdout) if finite_holdout else math.inf
    near_best_modes = [
        mode
        for mode in coverage_modes
        if (
            not finite_holdout
            or mean_holdout_by_mode[mode] <= best_holdout * 1.05 + 1e-12
        )
    ]
    model_complexity = {
        DepthAlignmentMode.SCALE_ONLY: 0,
        DepthAlignmentMode.AFFINE_DEPTH: 1,
        DepthAlignmentMode.AFFINE_INVERSE_DEPTH: 1,
    }
    selected_mode = min(
        near_best_modes,
        key=lambda mode: (
            model_complexity[mode],
            mean_holdout_by_mode[mode]
            if math.isfinite(mean_holdout_by_mode[mode])
            else math.inf,
            -score_by_mode[mode],
        ),
    )
    edges = candidates_by_mode[selected_mode]
    adjacency: dict[int, set[int]] = {index: set() for index in indices}
    for edge in edges:
        adjacency[edge.source_frame].add(edge.target_frame)
        adjacency[edge.target_frame].add(edge.source_frame)
    connected = {reference}
    queue = [reference]
    while queue:
        current = queue.pop(0)
        for neighbour in adjacency[current]:
            if neighbour not in connected:
                connected.add(neighbour)
                queue.append(neighbour)
    usable_edges = [
        edge
        for edge in edges
        if edge.source_frame in connected and edge.target_frame in connected
    ]
    parameters: dict[int, tuple[float, float]] = {reference: (1.0, 0.0)}
    consistency = float("nan")
    if usable_edges and len(connected) > 1:
        connected_indices = tuple(index for index in indices if index in connected)
        parameters, consistency = _solve_global_affine_parameters(
            connected_indices, reference, usable_edges
        )
    elif len(indices) == 1:
        consistency = 0.0
    per_frame: dict[int, PerFrameDepthAlignment] = {}
    for index in indices:
        if index in parameters:
            supporting = tuple(
                (edge.source_frame, edge.target_frame)
                for edge in usable_edges
                if index in {edge.source_frame, edge.target_frame}
            )
            local_quality = (
                float(
                    np.mean(
                        [
                            edge.quality
                            for edge in usable_edges
                            if index in {edge.source_frame, edge.target_frame}
                        ]
                    )
                )
                if supporting
                else 1.0
            )
            per_frame[index] = PerFrameDepthAlignment(
                index,
                reference,
                parameters[index][0],
                parameters[index][1],
                (
                    "inverse_depth_raw_or_reciprocal"
                    if selected_mode == DepthAlignmentMode.AFFINE_INVERSE_DEPTH
                    else "geometry_z_depth"
                ),
                supporting,
                True,
                local_quality,
            )
        else:
            per_frame[index] = PerFrameDepthAlignment(
                index,
                reference,
                float("nan"),
                float("nan"),
                "unknown",
                (),
                False,
                0.0,
                "depth_alignment_graph_disconnected",
            )
    edge_scales = [edge.scale for edge in usable_edges if edge.scale > 0.0]
    before = (
        float(np.std(np.log(edge_scales))) if len(edge_scales) >= 2 else float("nan")
    )
    # Before refinement, edge-scale dispersion measures pairwise drift.  After
    # refinement, the graph equation residual measures remaining inconsistency;
    # the dispersion of absolute per-frame scale parameters is not itself an
    # error because genuine frame normalisations may require different scales.
    after = consistency
    connected_ratio = len(connected) / len(indices)
    mean_holdout = _mean_finite([edge.holdout_error for edge in usable_edges])
    quality = float(
        connected_ratio
        * (
            np.mean([edge.quality for edge in usable_edges])
            if usable_edges
            else (1.0 if len(indices) == 1 else 0.0)
        )
        / (1.0 + (consistency if math.isfinite(consistency) else 1.0))
    )
    valid = connected_ratio == 1.0 and bool(usable_edges or len(indices) == 1)
    direct_edges = sum(
        edge.source_frame == reference or edge.target_frame == reference
        for edge in usable_edges
    )
    effective_method = (
        GlobalDepthAlignmentMethod.DIRECT_TO_REFERENCE
        if valid and direct_edges >= len(indices) - 1
        else selected_method
    )
    return SequenceDepthAlignmentResult(
        frame_indices=indices,
        reference_frame=reference,
        method=effective_method,
        alignment_mode=selected_mode,
        alignment_domain=(
            "inverse_depth_raw_or_reciprocal"
            if selected_mode == DepthAlignmentMode.AFFINE_INVERSE_DEPTH
            else "geometry_z_depth"
        ),
        per_frame=per_frame,
        supporting_edges=tuple(usable_edges),
        global_consistency_error=consistency,
        scale_drift_before=before,
        scale_drift_after=after,
        connected_frame_ratio=connected_ratio,
        valid=valid,
        quality=float(np.clip(quality, 0.0, 1.0)),
        missing_reason="" if valid else "depth_alignment_graph_disconnected",
        metadata={
            "candidate_scores_by_mode": {
                mode.value: score_by_mode[mode] for mode in modes
            },
            "connected_frame_ratio_by_mode": {
                mode.value: len(connected_by_mode[mode]) / len(indices)
                for mode in modes
            },
            "mean_normalized_holdout_error_by_mode": {
                mode.value: mean_holdout_by_mode[mode] for mode in modes
            },
            "global_model_selection_rule": (
                "reference_connectivity_then_holdout_then_simplicity"
            ),
            "mean_holdout_error": mean_holdout,
            "direct_reference_edge_count": direct_edges,
            "pairwise_graph_propagation": True,
            "robust_global_refinement": selected_method
            == GlobalDepthAlignmentMethod.ROBUST_GLOBAL_REFINEMENT,
            "semantic_scale_prior_used": False,
            "visualization_depth_used": False,
        },
    )


def _mean_finite(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def apply_sequence_depth_alignment(
    values: np.ndarray,
    frame_alignment: PerFrameDepthAlignment,
    *,
    values_are_inverse_domain: bool = False,
) -> np.ndarray:
    """Map one frame to the reference domain while preserving invalid pixels."""

    if not frame_alignment.valid:
        raise ValueError(
            f"Cannot apply invalid frame alignment: {frame_alignment.missing_reason}"
        )
    source = np.asarray(values, dtype=float)
    output = np.full(source.shape, np.nan, dtype=float)
    valid = np.isfinite(source) & (source > 0.0)
    mapped = frame_alignment.scale * source[valid] + frame_alignment.shift
    positive = mapped > 1e-12
    flat_indices = np.flatnonzero(valid)
    if values_are_inverse_domain:
        output.reshape(-1)[flat_indices[positive]] = 1.0 / mapped[positive]
    else:
        output.reshape(-1)[flat_indices[positive]] = mapped[positive]
    return output
