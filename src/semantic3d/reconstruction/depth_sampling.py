"""Robust sampling from canonical metric or relative Z-depth observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from ..depth_provider import DepthObservation


class DepthSamplingMethod(str, Enum):
    """Supported image-space depth sampling neighbourhoods."""

    EXACT_PIXEL = "exact_pixel"
    LOCAL_MEDIAN_3X3 = "local_median_3x3"
    LOCAL_MEDIAN_5X5 = "local_median_5x5"


@dataclass(frozen=True)
class DepthSample:
    """One sampled Z-depth value with local robustness diagnostics."""

    source_pixel: tuple[float, float]
    sampled_depth: Optional[float]
    sampling_method: DepthSamplingMethod | str
    local_valid_ratio: float
    local_depth_iqr: Optional[float]
    point_quality: float
    valid: bool
    missing_reason: str = ""

    def __post_init__(self) -> None:
        method = DepthSamplingMethod(self.sampling_method)
        valid_ratio = float(self.local_valid_ratio)
        quality = float(self.point_quality)
        if not 0.0 <= valid_ratio <= 1.0:
            raise ValueError("local_valid_ratio must be in [0, 1].")
        if not 0.0 <= quality <= 1.0:
            raise ValueError("point_quality must be in [0, 1].")
        if self.valid:
            if self.sampled_depth is None or not math.isfinite(self.sampled_depth):
                raise ValueError("Valid DepthSample requires finite sampled_depth.")
            if self.sampled_depth <= 0.0:
                raise ValueError("Valid DepthSample requires positive Z depth.")
            if self.missing_reason:
                raise ValueError("Valid DepthSample cannot have missing_reason.")
        elif not self.missing_reason:
            raise ValueError("Invalid DepthSample requires missing_reason.")
        object.__setattr__(self, "sampling_method", method)
        object.__setattr__(self, "local_valid_ratio", valid_ratio)
        object.__setattr__(self, "point_quality", quality)


def _pixel_index(value: float) -> int:
    """Map a pixel-centre coordinate to its nearest integer index."""

    return int(np.floor(float(value) + 0.5))


def _window_radius(method: DepthSamplingMethod) -> int:
    if method == DepthSamplingMethod.EXACT_PIXEL:
        return 0
    if method == DepthSamplingMethod.LOCAL_MEDIAN_3X3:
        return 1
    return 2


def sample_depth(
    depth: DepthObservation,
    u: float,
    v: float,
    method: DepthSamplingMethod | str = DepthSamplingMethod.LOCAL_MEDIAN_3X3,
    *,
    filter_local_outliers: bool = True,
) -> DepthSample:
    """Sample canonical Z depth using a robust local median.

    The function calls ``require_geometry_depth`` and therefore rejects legacy
    per-frame [1, 10] visualization arrays and unconverted inverse depth.
    """

    sampling_method = DepthSamplingMethod(method)
    if not np.isfinite([u, v]).all():
        return DepthSample(
            (float(u), float(v)),
            None,
            sampling_method,
            0.0,
            None,
            0.0,
            False,
            "non_finite_source_pixel",
        )
    try:
        depth_map = depth.require_geometry_depth()
    except ValueError as error:
        return DepthSample(
            (float(u), float(v)),
            None,
            sampling_method,
            0.0,
            None,
            0.0,
            False,
            f"invalid_geometry_depth:{error}",
        )
    row, column = _pixel_index(v), _pixel_index(u)
    height, width = depth_map.shape
    if row < 0 or row >= height or column < 0 or column >= width:
        return DepthSample(
            (float(u), float(v)),
            None,
            sampling_method,
            0.0,
            None,
            0.0,
            False,
            "source_pixel_out_of_bounds",
        )

    radius = _window_radius(sampling_method)
    row_min, row_max = max(0, row - radius), min(height, row + radius + 1)
    col_min, col_max = max(0, column - radius), min(width, column + radius + 1)
    patch = np.asarray(depth_map[row_min:row_max, col_min:col_max], dtype=float)
    mask = np.isfinite(patch) & (patch > 0.0)
    if depth.valid_mask is not None:
        mask &= depth.valid_mask[row_min:row_max, col_min:col_max]
    local_valid_ratio = float(mask.mean()) if mask.size else 0.0
    values = patch[mask]
    if values.size == 0:
        return DepthSample(
            (float(u), float(v)),
            None,
            sampling_method,
            local_valid_ratio,
            None,
            0.0,
            False,
            "no_valid_depth_in_window",
        )

    q1, q3 = np.percentile(values, [25.0, 75.0])
    iqr = float(q3 - q1)
    filtered = values
    if filter_local_outliers and values.size >= 4:
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        candidate = values[(values >= lower) & (values <= upper)]
        if candidate.size:
            filtered = candidate
    sampled = float(filtered[0]) if radius == 0 else float(np.median(filtered))
    relative_iqr = iqr / max(abs(sampled), 1e-12)
    stability = 1.0 / (1.0 + relative_iqr)
    confidence_quality = 1.0
    if depth.confidence_map is not None:
        local_confidence = np.asarray(
            depth.confidence_map[row_min:row_max, col_min:col_max], dtype=float
        )[mask]
        finite_confidence = local_confidence[np.isfinite(local_confidence)]
        if finite_confidence.size:
            confidence_quality = float(np.clip(np.mean(finite_confidence), 0.0, 1.0))
    quality = float(np.clip(local_valid_ratio * stability * confidence_quality, 0.0, 1.0))
    return DepthSample(
        source_pixel=(float(u), float(v)),
        sampled_depth=sampled,
        sampling_method=sampling_method,
        local_valid_ratio=local_valid_ratio,
        local_depth_iqr=iqr,
        point_quality=quality,
        valid=True,
    )
