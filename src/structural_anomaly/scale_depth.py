"""Scale-depth consistency residuals.

The module implements the object-level residual described in the thesis
prototype:

    p_i = sqrt(mask_area_i / frame_area)

For object A and object B:

    r_min = H_A_min / H_B_max
    r_max = H_A_max / H_B_min

The depth ratio Z_A / Z_B should lie in:

    [r_min * p_B / p_A, r_max * p_B / p_A]

The residual is the distance from the depth ratio to this interval. A log-space
version is also provided for better numerical stability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple, Union

import numpy as np

NumberLike = Union[float, int, np.ndarray, Any]
ScalePrior = Mapping[str, Tuple[float, float]]


@dataclass(frozen=True)
class ObjectObservation:
    """Observed object information needed by the scale-depth residual.

    Attributes:
        class_name: Object category name used to look up the scale prior.
        mask_area: Instance mask area in pixels.
        depth: Median depth of the object region.
    """

    class_name: str
    mask_area: NumberLike
    depth: NumberLike


@dataclass(frozen=True)
class ScaleDepthResidualResult:
    """Detailed result of one scale-depth residual computation.

    Attributes:
        residual: Distance from the observed depth ratio to the valid interval.
        ratio: Observed depth ratio, either Z_A / Z_B or log(Z_A) - log(Z_B).
        lower: Lower bound of the valid interval.
        upper: Upper bound of the valid interval.
        projection_a: Equivalent projection scale p_A.
        projection_b: Equivalent projection scale p_B.
    """

    residual: NumberLike
    ratio: NumberLike
    lower: NumberLike
    upper: NumberLike
    projection_a: NumberLike
    projection_b: NumberLike


def _is_torch_tensor(value: Any) -> bool:
    """Return True when value looks like a torch Tensor without importing torch."""

    return value.__class__.__module__.startswith("torch")


def _backend_from_values(*values: Any) -> str:
    """Select numpy or torch by inspecting the provided values."""

    return "torch" if any(_is_torch_tensor(value) for value in values) else "numpy"


def _as_backend_value(value: NumberLike, backend: str) -> NumberLike:
    """Convert Python scalars to the selected array/tensor backend."""

    if backend == "torch":
        import torch

        if _is_torch_tensor(value):
            return value
        return torch.as_tensor(value, dtype=torch.float64)
    return np.asarray(value, dtype=float)


def _sqrt(value: NumberLike, backend: str) -> NumberLike:
    if backend == "torch":
        import torch

        return torch.sqrt(value)
    return np.sqrt(value)


def _log(value: NumberLike, backend: str) -> NumberLike:
    if backend == "torch":
        import torch

        return torch.log(value)
    return np.log(value)


def _maximum(a: NumberLike, b: NumberLike, backend: str) -> NumberLike:
    if backend == "torch":
        import torch

        return torch.maximum(a, b)
    return np.maximum(a, b)


def _zeros_like(value: NumberLike, backend: str) -> NumberLike:
    if backend == "torch":
        import torch

        return torch.zeros_like(value)
    return np.zeros_like(value, dtype=float)


def _validate_positive(name: str, value: Any) -> None:
    """Validate that every scalar/array/tensor element is positive."""

    if _is_torch_tensor(value):
        if bool((value <= 0).any().item()):
            raise ValueError(f"{name} must be positive.")
        return
    if bool(np.any(np.asarray(value) <= 0)):
        raise ValueError(f"{name} must be positive.")


def _get_scale_prior(class_name: str, scale_prior: ScalePrior) -> Tuple[float, float]:
    """Return and validate the physical scale interval for one class."""

    if class_name not in scale_prior:
        raise KeyError(f"Missing scale prior for class '{class_name}'.")

    min_size, max_size = scale_prior[class_name]
    if min_size <= 0 or max_size <= 0:
        raise ValueError(f"Scale prior for '{class_name}' must be positive.")
    if min_size > max_size:
        raise ValueError(
            f"Scale prior for '{class_name}' must satisfy min_size <= max_size."
        )
    return float(min_size), float(max_size)


def compute_equivalent_projection_scale(
    mask_area: NumberLike, frame_area: NumberLike
) -> NumberLike:
    """Compute the equivalent projection scale p_i.

    Formula:
        p_i = sqrt(mask_area_i / frame_area)

    Args:
        mask_area: Object instance mask area in pixels.
        frame_area: Whole frame area in pixels.

    Returns:
        Equivalent projection scale. The return type follows the input backend:
        numpy for numeric/numpy inputs, torch Tensor for torch inputs.
    """

    backend = _backend_from_values(mask_area, frame_area)
    mask_area_value = _as_backend_value(mask_area, backend)
    frame_area_value = _as_backend_value(frame_area, backend)
    _validate_positive("mask_area", mask_area_value)
    _validate_positive("frame_area", frame_area_value)
    return _sqrt(mask_area_value / frame_area_value, backend)


def _compute_equivalent_projection_scale_with_backend(
    mask_area: NumberLike, frame_area: NumberLike, backend: str
) -> NumberLike:
    """Compute p_i while preserving the backend chosen by the caller."""

    mask_area_value = _as_backend_value(mask_area, backend)
    frame_area_value = _as_backend_value(frame_area, backend)
    _validate_positive("mask_area", mask_area_value)
    _validate_positive("frame_area", frame_area_value)
    return _sqrt(mask_area_value / frame_area_value, backend)


def _interval_distance(
    ratio: NumberLike, lower: NumberLike, upper: NumberLike, backend: str
) -> NumberLike:
    """Distance from ratio to interval [lower, upper].

    Formula:
        if ratio < lower: lower - ratio
        if ratio > upper: ratio - upper
        otherwise: 0
    """

    below = lower - ratio
    above = ratio - upper
    zero = _zeros_like(ratio, backend)
    return _maximum(_maximum(below, above, backend), zero, backend)


def compute_scale_depth_residual(
    object_a: ObjectObservation,
    object_b: ObjectObservation,
    frame_area: NumberLike,
    scale_prior: ScalePrior,
) -> ScaleDepthResidualResult:
    """Compute ratio-space object-level scale-depth residual R_sd(A, B).

    The depth ratio Z_A / Z_B is required to lie in:

        [r_min * p_B / p_A, r_max * p_B / p_A]

    where:

        p_i = sqrt(mask_area_i / frame_area)
        r_min = H_A_min / H_B_max
        r_max = H_A_max / H_B_min

    Args:
        object_a: Observation of object A.
        object_b: Observation of object B.
        frame_area: Whole frame area in pixels.
        scale_prior: Mapping from class name to (min_size, max_size).

    Returns:
        ScaleDepthResidualResult with residual, ratio, interval bounds, and
        equivalent projection scales.
    """

    backend = _backend_from_values(
        object_a.mask_area, object_a.depth, object_b.mask_area, object_b.depth, frame_area
    )
    depth_a = _as_backend_value(object_a.depth, backend)
    depth_b = _as_backend_value(object_b.depth, backend)
    _validate_positive("object_a.depth", depth_a)
    _validate_positive("object_b.depth", depth_b)

    h_a_min, h_a_max = _get_scale_prior(object_a.class_name, scale_prior)
    h_b_min, h_b_max = _get_scale_prior(object_b.class_name, scale_prior)

    projection_a = _compute_equivalent_projection_scale_with_backend(
        object_a.mask_area, frame_area, backend
    )
    projection_b = _compute_equivalent_projection_scale_with_backend(
        object_b.mask_area, frame_area, backend
    )

    # r_min = H_A_min / H_B_max, r_max = H_A_max / H_B_min.
    r_min = _as_backend_value(h_a_min / h_b_max, backend)
    r_max = _as_backend_value(h_a_max / h_b_min, backend)

    # Valid interval for Z_A / Z_B:
    # [r_min * p_B / p_A, r_max * p_B / p_A].
    lower = r_min * projection_b / projection_a
    upper = r_max * projection_b / projection_a

    # Observed depth ratio Z_A / Z_B.
    ratio = depth_a / depth_b
    residual = _interval_distance(ratio, lower, upper, backend)

    return ScaleDepthResidualResult(
        residual=residual,
        ratio=ratio,
        lower=lower,
        upper=upper,
        projection_a=projection_a,
        projection_b=projection_b,
    )


def compute_scale_depth_residual_log(
    object_a: ObjectObservation,
    object_b: ObjectObservation,
    frame_area: NumberLike,
    scale_prior: ScalePrior,
) -> ScaleDepthResidualResult:
    """Compute log-space object-level scale-depth residual R_sd_log(A, B).

    Formula:
        log_ratio = log(Z_A) - log(Z_B)
        log_lower = log(r_min) + log(p_B) - log(p_A)
        log_upper = log(r_max) + log(p_B) - log(p_A)

    The residual is the distance from log_ratio to [log_lower, log_upper].

    Args:
        object_a: Observation of object A.
        object_b: Observation of object B.
        frame_area: Whole frame area in pixels.
        scale_prior: Mapping from class name to (min_size, max_size).

    Returns:
        ScaleDepthResidualResult whose ratio/lower/upper/residual are in
        log-space while projection_a/projection_b remain p_A and p_B.
    """

    backend = _backend_from_values(
        object_a.mask_area, object_a.depth, object_b.mask_area, object_b.depth, frame_area
    )
    depth_a = _as_backend_value(object_a.depth, backend)
    depth_b = _as_backend_value(object_b.depth, backend)
    _validate_positive("object_a.depth", depth_a)
    _validate_positive("object_b.depth", depth_b)

    h_a_min, h_a_max = _get_scale_prior(object_a.class_name, scale_prior)
    h_b_min, h_b_max = _get_scale_prior(object_b.class_name, scale_prior)

    projection_a = _compute_equivalent_projection_scale_with_backend(
        object_a.mask_area, frame_area, backend
    )
    projection_b = _compute_equivalent_projection_scale_with_backend(
        object_b.mask_area, frame_area, backend
    )

    # r_min = H_A_min / H_B_max, r_max = H_A_max / H_B_min.
    r_min = _as_backend_value(h_a_min / h_b_max, backend)
    r_max = _as_backend_value(h_a_max / h_b_min, backend)

    # log-space valid interval:
    # [log(r_min) + log(p_B) - log(p_A),
    #  log(r_max) + log(p_B) - log(p_A)].
    lower = _log(r_min, backend) + _log(projection_b, backend) - _log(projection_a, backend)
    upper = _log(r_max, backend) + _log(projection_b, backend) - _log(projection_a, backend)

    # log_ratio = log(Z_A) - log(Z_B).
    ratio = _log(depth_a, backend) - _log(depth_b, backend)
    residual = _interval_distance(ratio, lower, upper, backend)

    return ScaleDepthResidualResult(
        residual=residual,
        ratio=ratio,
        lower=lower,
        upper=upper,
        projection_a=projection_a,
        projection_b=projection_b,
    )
