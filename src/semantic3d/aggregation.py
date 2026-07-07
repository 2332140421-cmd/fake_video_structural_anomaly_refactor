"""Aggregation helpers for multi-granularity structural residuals."""

from __future__ import annotations

from typing import Literal

import numpy as np

EmptyPolicy = Literal["zero", "raise"]


def _as_1d_float_array(values: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Convert input values to a finite one-dimensional float array."""

    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def topk_mean(
    values: np.ndarray | list[float] | tuple[float, ...],
    k_ratio: float = 0.2,
    empty_policy: EmptyPolicy = "zero",
) -> float:
    """Return the mean of the largest k_ratio proportion of values.

    Args:
        values: One-dimensional values or any array-like input.
        k_ratio: Proportion of largest values to average. At least one value is
            selected when the input is non-empty.
        empty_policy: "zero" returns 0.0 for empty input; "raise" raises a
            ValueError.
    """

    if k_ratio <= 0 or k_ratio > 1:
        raise ValueError(f"k_ratio must be in (0, 1], got {k_ratio}.")

    array = _as_1d_float_array(values)
    if array.size == 0:
        if empty_policy == "zero":
            return 0.0
        if empty_policy == "raise":
            raise ValueError("Cannot compute topk_mean for empty values.")
        raise ValueError(f"Unknown empty_policy: {empty_policy!r}.")

    k = max(1, int(np.ceil(array.size * k_ratio)))
    return float(np.sort(array)[-k:].mean())


def aggregate_values(
    values: np.ndarray | list[float] | tuple[float, ...],
    method: str = "topk_mean",
    empty_policy: EmptyPolicy = "zero",
    k_ratio: float = 0.2,
) -> float:
    """Aggregate one-dimensional residual values."""

    array = _as_1d_float_array(values)
    if array.size == 0:
        if empty_policy == "zero":
            return 0.0
        if empty_policy == "raise":
            raise ValueError("Cannot aggregate empty values.")
        raise ValueError(f"Unknown empty_policy: {empty_policy!r}.")

    if method == "mean":
        return float(array.mean())
    if method == "median":
        return float(np.median(array))
    if method == "max":
        return float(array.max())
    if method == "topk_mean":
        return topk_mean(array, k_ratio=k_ratio, empty_policy=empty_policy)
    raise ValueError(
        f"Unknown aggregation method '{method}'. "
        "Use 'mean', 'median', 'max', or 'topk_mean'."
    )


def aggregate_map_by_mask(
    residual_map: np.ndarray,
    mask: np.ndarray,
    method: str = "topk_mean",
    empty_policy: EmptyPolicy = "zero",
    k_ratio: float = 0.2,
) -> float:
    """Aggregate an HxW residual map inside a boolean or 0/1 object mask."""

    residual_array = np.asarray(residual_map, dtype=float)
    mask_array = np.asarray(mask).astype(bool)
    if residual_array.ndim != 2:
        raise ValueError(f"residual_map must be HxW, got shape {residual_array.shape}.")
    if mask_array.shape != residual_array.shape:
        raise ValueError(
            f"mask shape {mask_array.shape} must match residual_map shape "
            f"{residual_array.shape}."
        )
    return aggregate_values(
        residual_array[mask_array],
        method=method,
        empty_policy=empty_policy,
        k_ratio=k_ratio,
    )


def aggregate_points_by_mask(
    points_xy: np.ndarray,
    point_residuals: np.ndarray,
    mask: np.ndarray,
    method: str = "topk_mean",
    empty_policy: EmptyPolicy = "zero",
    k_ratio: float = 0.2,
) -> float:
    """Aggregate residuals for points that fall inside an object mask.

    Args:
        points_xy: N x 2 point coordinates in (x, y) order.
        point_residuals: N point residual values.
        mask: H x W boolean or 0/1 object mask.

    Points outside the image boundary are ignored safely.
    """

    points = np.asarray(points_xy, dtype=float)
    residuals = np.asarray(point_residuals, dtype=float).reshape(-1)
    mask_array = np.asarray(mask).astype(bool)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points_xy must have shape Nx2, got {points.shape}.")
    if residuals.shape[0] != points.shape[0]:
        raise ValueError(
            f"point_residuals length {residuals.shape[0]} does not match "
            f"points count {points.shape[0]}."
        )
    if mask_array.ndim != 2:
        raise ValueError(f"mask must be HxW, got shape {mask_array.shape}.")

    height, width = mask_array.shape
    xs = np.rint(points[:, 0]).astype(int)
    ys = np.rint(points[:, 1]).astype(int)
    in_bounds = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    if not np.any(in_bounds):
        return aggregate_values([], method=method, empty_policy=empty_policy)

    valid_indices = np.where(in_bounds)[0]
    inside = mask_array[ys[valid_indices], xs[valid_indices]]
    selected = residuals[valid_indices[inside]]
    return aggregate_values(
        selected,
        method=method,
        empty_policy=empty_policy,
        k_ratio=k_ratio,
    )
