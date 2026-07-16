"""Frozen coarse 2D object-level scale-depth consistency residuals.

This module implements the legacy/coarse 2D R_sd baseline used by the prototype.
It does not depend on pretrained models. The inputs are object observations,
class-level physical scale priors, and estimated depths.

For each object i:

    projection_ratio_i = mask_area_i / frame_area_i
    p_i = sqrt(mask_area_i / frame_area_i)

For object A and object B:

    r_min = H_A_min / H_B_max
    r_max = H_A_max / H_B_min

The depth ratio Z_A / Z_B should lie inside:

    [r_min * p_B / p_A, r_max * p_B / p_A]

The ratio-space residual is the distance from Z_A / Z_B to that interval.
The log-space residual computes the same interval-distance idea after applying
log to the depth ratio and interval bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class ObjectObservation:
    """Observation of one segmented object in a video frame.

    Attributes:
        object_id: Stable object or instance identifier.
        label: Semantic class label used to index the scale prior.
        mask_area: Instance mask area in pixels.
        frame_area: Full frame area in pixels.
        depth: Median depth of the object region.
        confidence: Optional confidence score for the observation.
    """

    object_id: str
    label: str
    mask_area: float
    frame_area: float
    depth: float
    confidence: float = 1.0

    @property
    def projection_ratio(self) -> float:
        """Return the object projection area ratio mask_area / frame_area."""

        _validate_object_geometry(self)
        return self.mask_area / self.frame_area

    @property
    def equivalent_projection_scale(self) -> float:
        """Return p_i = sqrt(mask_area_i / frame_area_i)."""

        return sqrt(self.projection_ratio)


@dataclass(frozen=True)
class ScalePrior:
    """Coarse physical scale interval for an object class."""

    min_size: float
    max_size: float


ScalePriors = Mapping[str, ScalePrior]


def _validate_object_geometry(obj: ObjectObservation) -> None:
    """Validate object-level positive geometric quantities."""

    if obj.mask_area <= 0:
        raise ValueError(
            f"Object '{obj.object_id}' has invalid mask_area={obj.mask_area}; "
            "mask_area must be > 0."
        )
    if obj.frame_area <= 0:
        raise ValueError(
            f"Object '{obj.object_id}' has invalid frame_area={obj.frame_area}; "
            "frame_area must be > 0."
        )
    if obj.depth <= 0:
        raise ValueError(
            f"Object '{obj.object_id}' has invalid depth={obj.depth}; "
            "depth must be > 0."
        )


def _get_valid_scale_prior(obj: ObjectObservation, scale_priors: ScalePriors) -> ScalePrior:
    """Fetch and validate the scale prior for an object's label."""

    if obj.label not in scale_priors:
        raise KeyError(
            f"Missing scale prior for label '{obj.label}' "
            f"used by object '{obj.object_id}'."
        )

    prior = scale_priors[obj.label]
    if prior.min_size <= 0:
        raise ValueError(
            f"Scale prior for label '{obj.label}' has min_size={prior.min_size}; "
            "min_size must be > 0."
        )
    if prior.max_size <= prior.min_size:
        raise ValueError(
            f"Scale prior for label '{obj.label}' has min_size={prior.min_size}, "
            f"max_size={prior.max_size}; max_size must be > min_size."
        )
    return prior


def _validate_pair(
    obj_a: ObjectObservation, obj_b: ObjectObservation, scale_priors: ScalePriors
) -> Tuple[ScalePrior, ScalePrior]:
    """Validate two observations and return their class scale priors."""

    _validate_object_geometry(obj_a)
    _validate_object_geometry(obj_b)
    prior_a = _get_valid_scale_prior(obj_a, scale_priors)
    prior_b = _get_valid_scale_prior(obj_b, scale_priors)
    return prior_a, prior_b


def _distance_to_interval(value: float, lower: float, upper: float) -> float:
    """Return distance from a scalar value to the closed interval [lower, upper]."""

    if value < lower:
        return lower - value
    if value > upper:
        return value - upper
    return 0.0


def compute_scale_depth_interval(
    obj_a: ObjectObservation,
    obj_b: ObjectObservation,
    scale_priors: ScalePriors,
) -> Tuple[float, float]:
    """Compute the reasonable interval for the depth ratio Z_A / Z_B.

    Formula:
        p_A = sqrt(mask_area_A / frame_area_A)
        p_B = sqrt(mask_area_B / frame_area_B)
        r_min = H_A_min / H_B_max
        r_max = H_A_max / H_B_min
        lower = r_min * p_B / p_A
        upper = r_max * p_B / p_A
    """

    prior_a, prior_b = _validate_pair(obj_a, obj_b, scale_priors)

    p_a = obj_a.equivalent_projection_scale
    p_b = obj_b.equivalent_projection_scale

    r_min = prior_a.min_size / prior_b.max_size
    r_max = prior_a.max_size / prior_b.min_size

    lower = r_min * p_b / p_a
    upper = r_max * p_b / p_a
    return float(lower), float(upper)


def scale_depth_residual(
    obj_a: ObjectObservation,
    obj_b: ObjectObservation,
    scale_priors: ScalePriors,
    eps: float = 1e-8,
) -> Tuple[float, Dict[str, float]]:
    """Compute ratio-space R_sd for an ordered object pair (A, B).

    The observed ratio is:
        ratio = (Z_A + eps) / (Z_B + eps)

    R_sd is the distance from ratio to the interval returned by
    compute_scale_depth_interval.
    """

    if eps <= 0:
        raise ValueError(f"eps must be > 0, got eps={eps}.")
    _validate_pair(obj_a, obj_b, scale_priors)

    lower, upper = compute_scale_depth_interval(obj_a, obj_b, scale_priors)

    # Observed depth ratio Z_A / Z_B. eps only guards numerical division.
    depth_ratio = (obj_a.depth + eps) / (obj_b.depth + eps)
    residual = _distance_to_interval(depth_ratio, lower, upper)

    details = {
        "projection_ratio_a": obj_a.projection_ratio,
        "projection_ratio_b": obj_b.projection_ratio,
        "projection_scale_a": obj_a.equivalent_projection_scale,
        "projection_scale_b": obj_b.equivalent_projection_scale,
        "depth_ratio": depth_ratio,
        "lower": lower,
        "upper": upper,
        "residual": residual,
    }
    return float(residual), details


def scale_depth_residual_log(
    obj_a: ObjectObservation,
    obj_b: ObjectObservation,
    scale_priors: ScalePriors,
    eps: float = 1e-8,
) -> Tuple[float, Dict[str, float]]:
    """Compute log-space R_sd for an ordered object pair (A, B).

    Formula:
        log_ratio = log(Z_A + eps) - log(Z_B + eps)
        log_lower = log(r_min) + log(p_B + eps) - log(p_A + eps)
        log_upper = log(r_max) + log(p_B + eps) - log(p_A + eps)

    The returned residual is the distance from log_ratio to
    [log_lower, log_upper].
    """

    if eps <= 0:
        raise ValueError(f"eps must be > 0, got eps={eps}.")
    prior_a, prior_b = _validate_pair(obj_a, obj_b, scale_priors)

    p_a = obj_a.equivalent_projection_scale
    p_b = obj_b.equivalent_projection_scale
    r_min = prior_a.min_size / prior_b.max_size
    r_max = prior_a.max_size / prior_b.min_size

    log_ratio = log(obj_a.depth + eps) - log(obj_b.depth + eps)
    log_lower = log(r_min) + log(p_b + eps) - log(p_a + eps)
    log_upper = log(r_max) + log(p_b + eps) - log(p_a + eps)
    residual = _distance_to_interval(log_ratio, log_lower, log_upper)

    details = {
        "projection_ratio_a": obj_a.projection_ratio,
        "projection_ratio_b": obj_b.projection_ratio,
        "projection_scale_a": p_a,
        "projection_scale_b": p_b,
        "log_depth_ratio": log_ratio,
        "log_lower": log_lower,
        "log_upper": log_upper,
        "residual": residual,
    }
    return float(residual), details


def pairwise_scale_depth_residuals(
    objects: Iterable[ObjectObservation],
    scale_priors: ScalePriors,
    use_log: bool = True,
) -> Tuple[np.ndarray, Dict[Tuple[str, str], Dict[str, float]]]:
    """Compute pairwise scale-depth residuals for all ordered object pairs.

    Args:
        objects: Objects detected in one frame.
        scale_priors: Mapping from class label to physical ScalePrior.
        use_log: If True, compute log-space residuals; otherwise ratio-space.

    Returns:
        residual_matrix: N x N matrix. Diagonal entries are 0 because an object
            is not compared with itself.
        details: Dictionary keyed by (object_id_a, object_id_b), containing
            per-pair calculation details.
    """

    object_list = list(objects)
    n_objects = len(object_list)
    residual_matrix = np.zeros((n_objects, n_objects), dtype=float)
    details: Dict[Tuple[str, str], Dict[str, float]] = {}

    residual_fn = scale_depth_residual_log if use_log else scale_depth_residual

    for i, obj_a in enumerate(object_list):
        _validate_object_geometry(obj_a)
        _get_valid_scale_prior(obj_a, scale_priors)
        for j, obj_b in enumerate(object_list):
            if i == j:
                continue
            residual, pair_details = residual_fn(obj_a, obj_b, scale_priors)
            residual_matrix[i, j] = residual
            details[(obj_a.object_id, obj_b.object_id)] = pair_details

    return residual_matrix, details


def rsd_2d_coarse(
    obj_a: ObjectObservation,
    obj_b: ObjectObservation,
    scale_priors: ScalePriors,
    eps: float = 1e-8,
) -> Tuple[float, Dict[str, float]]:
    """Explicitly named compatibility entry point for coarse 2D R_sd."""

    return scale_depth_residual(obj_a, obj_b, scale_priors, eps=eps)


def rsd_2d_coarse_log(
    obj_a: ObjectObservation,
    obj_b: ObjectObservation,
    scale_priors: ScalePriors,
    eps: float = 1e-8,
) -> Tuple[float, Dict[str, float]]:
    """Explicitly named log-space compatibility entry point for coarse 2D R_sd."""

    return scale_depth_residual_log(obj_a, obj_b, scale_priors, eps=eps)
