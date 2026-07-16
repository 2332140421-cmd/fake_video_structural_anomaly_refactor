"""Read-only diagnostics for dimension-aligned strict R_sd v2 errors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np

from .observations import ObjectObservationJSON


DEPTH_STRATEGIES = (
    "full_bbox_median",
    "center_70_bbox_median",
    "center_50_bbox_median",
    "center_30_bbox_median",
    "bbox_depth_p25",
    "bbox_depth_p50",
    "bbox_depth_p75",
    "center_depth_trimmed_mean",
    "foreground_depth_cluster",
    "mask_median_depth",
)


@dataclass(frozen=True)
class DepthStatistic:
    """One object-region depth estimate and its diagnostic quality values."""

    depth: float
    valid_depth_ratio: float
    depth_iqr: float
    valid: bool
    method_detail: str


def deterministic_track_id(obj: ObjectObservationJSON) -> tuple[str, bool]:
    """Return the real track id or a stable, explicit fallback identifier."""

    if obj.track_id:
        return str(obj.track_id), False
    return f"fallback:{obj.label}:{obj.object_id}", True


def make_track_pair_id(video_id: str, track_id_a: str, track_id_b: str) -> str:
    """Create an order-independent video-level track-pair identifier."""

    first, second = sorted((str(track_id_a), str(track_id_b)))
    return f"{video_id}:{first}:{second}"


def make_frame_pair_id(
    video_id: str,
    global_frame_index: int,
    track_id_a: str,
    track_id_b: str,
) -> str:
    """Create an order-independent globally indexed frame-pair identifier."""

    first, second = sorted((str(track_id_a), str(track_id_b)))
    return f"{video_id}:{int(global_frame_index)}:{first}:{second}"


def safe_bbox_bounds(
    bbox: Optional[Sequence[float]],
    width: int,
    height: int,
    center_fraction: float = 1.0,
) -> Optional[tuple[int, int, int, int]]:
    """Clip a bbox crop safely and optionally retain its centered fraction."""

    if bbox is None or len(bbox) != 4 or width <= 0 or height <= 0:
        return None
    values = np.asarray(bbox, dtype=float)
    if not np.all(np.isfinite(values)) or not 0 < center_fraction <= 1:
        return None
    x1, y1, x2, y2 = values.tolist()
    if x2 <= x1 or y2 <= y1:
        return None
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half_w = (x2 - x1) * center_fraction / 2.0
    half_h = (y2 - y1) * center_fraction / 2.0
    ix1 = max(0, min(width, int(math.floor(cx - half_w))))
    iy1 = max(0, min(height, int(math.floor(cy - half_h))))
    ix2 = max(0, min(width, int(math.ceil(cx + half_w))))
    iy2 = max(0, min(height, int(math.ceil(cy + half_h))))
    return None if ix2 <= ix1 or iy2 <= iy1 else (ix1, iy1, ix2, iy2)


def _valid_values(crop: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return finite positive values, valid ratio, and IQR."""

    array = np.asarray(crop, dtype=float)
    valid_mask = np.isfinite(array) & (array > 0)
    values = array[valid_mask]
    ratio = float(values.size / array.size) if array.size else 0.0
    iqr = float(np.percentile(values, 75) - np.percentile(values, 25)) if values.size else math.nan
    return values, ratio, iqr


def _fallback_median(crop: np.ndarray, detail: str) -> DepthStatistic:
    values, ratio, iqr = _valid_values(crop)
    if values.size == 0:
        return DepthStatistic(math.nan, ratio, iqr, False, detail + ":no_valid_depth")
    return DepthStatistic(float(np.median(values)), ratio, iqr, True, detail + ":median_fallback")


def foreground_depth_cluster(crop: np.ndarray) -> DepthStatistic:
    """Select a deterministic 1D depth cluster using center support and continuity.

    When two clusters cannot be separated safely, this function falls back to
    the full-crop median and records that fallback in ``method_detail``.
    """

    array = np.asarray(crop, dtype=float)
    values, ratio, iqr = _valid_values(array)
    if values.size < 20 or not math.isfinite(iqr) or iqr <= 1e-8:
        return _fallback_median(array, "foreground_cluster_unstable")
    centers = np.asarray(np.percentile(values, [25, 75]), dtype=float)
    labels = np.zeros(values.size, dtype=np.int8)
    for _ in range(20):
        distances = np.abs(values[:, None] - centers[None, :])
        new_labels = np.argmin(distances, axis=1).astype(np.int8)
        new_centers = np.asarray(
            [np.mean(values[new_labels == index]) if np.any(new_labels == index) else centers[index] for index in range(2)]
        )
        if np.allclose(new_centers, centers, rtol=1e-6, atol=1e-8):
            labels, centers = new_labels, new_centers
            break
        labels, centers = new_labels, new_centers
    counts = np.bincount(labels, minlength=2)
    if np.min(counts) < max(5, int(0.05 * values.size)) or abs(centers[1] - centers[0]) < 0.25 * iqr:
        return _fallback_median(array, "foreground_cluster_unstable")

    valid_mask = np.isfinite(array) & (array > 0)
    label_image = np.full(array.shape, -1, dtype=np.int8)
    label_image[valid_mask] = labels
    h, w = array.shape[:2]
    center = np.zeros(array.shape, dtype=bool)
    center[max(0, int(0.25 * h)) : max(1, int(0.75 * h)), max(0, int(0.25 * w)) : max(1, int(0.75 * w))] = True
    scores: list[float] = []
    for index in range(2):
        cluster_mask = (label_image == index).astype(np.uint8)
        cluster_count = int(cluster_mask.sum())
        center_support = float(cluster_mask[center].sum() / max(1, center.sum()))
        components, _, stats, _ = cv2.connectedComponentsWithStats(cluster_mask, connectivity=8)
        largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if components > 1 else 0
        continuity = float(largest / cluster_count) if cluster_count else 0.0
        scores.append(0.65 * center_support + 0.35 * continuity)
    selected = int(np.argmax(scores))
    selected_values = values[labels == selected]
    return DepthStatistic(
        depth=float(np.median(selected_values)),
        valid_depth_ratio=ratio,
        depth_iqr=float(np.percentile(selected_values, 75) - np.percentile(selected_values, 25)),
        valid=True,
        method_detail=f"foreground_cluster_{selected}",
    )


def compute_depth_strategy(
    depth_map: np.ndarray,
    obj: ObjectObservationJSON,
    strategy: str,
    mask: Optional[np.ndarray] = None,
) -> DepthStatistic:
    """Compute one diagnostic object-depth statistic without mutating inputs."""

    if strategy not in DEPTH_STRATEGIES:
        raise ValueError(f"Unknown depth strategy: {strategy}")
    array = np.asarray(depth_map)
    if array.ndim != 2:
        return DepthStatistic(math.nan, 0.0, math.nan, False, "invalid_depth_map_shape")
    original_bbox = None if obj.bbox is None else tuple(obj.bbox)
    fractions = {
        "center_70_bbox_median": 0.70,
        "center_50_bbox_median": 0.50,
        "center_30_bbox_median": 0.30,
        "center_depth_trimmed_mean": 0.70,
    }
    fraction = fractions.get(strategy, 1.0)
    bounds = safe_bbox_bounds(obj.bbox, array.shape[1], array.shape[0], fraction)
    if bounds is None:
        return DepthStatistic(math.nan, 0.0, math.nan, False, "invalid_bbox")
    x1, y1, x2, y2 = bounds
    crop = array[y1:y2, x1:x2]
    values, ratio, iqr = _valid_values(crop)
    if strategy == "mask_median_depth":
        if mask is None or np.asarray(mask).shape != array.shape:
            return DepthStatistic(math.nan, 0.0, math.nan, False, "mask_unavailable")
        mask_crop = np.asarray(mask[y1:y2, x1:x2], dtype=bool)
        masked = np.where(mask_crop, crop, np.nan)
        values, ratio, iqr = _valid_values(masked)
    if values.size == 0:
        return DepthStatistic(math.nan, ratio, iqr, False, "no_valid_depth")
    if strategy == "foreground_depth_cluster":
        result = foreground_depth_cluster(crop)
    elif strategy == "bbox_depth_p25":
        result = DepthStatistic(float(np.percentile(values, 25)), ratio, iqr, True, "percentile_25")
    elif strategy in {"full_bbox_median", "bbox_depth_p50", "center_70_bbox_median", "center_50_bbox_median", "center_30_bbox_median", "mask_median_depth"}:
        result = DepthStatistic(float(np.median(values)), ratio, iqr, True, "median")
    elif strategy == "bbox_depth_p75":
        result = DepthStatistic(float(np.percentile(values, 75)), ratio, iqr, True, "percentile_75")
    else:
        sorted_values = np.sort(values)
        trim = int(0.1 * sorted_values.size)
        trimmed = sorted_values[trim:-trim] if trim and 2 * trim < sorted_values.size else sorted_values
        result = DepthStatistic(float(np.mean(trimmed)), ratio, iqr, True, "center70_trimmed_mean_10pct")
    if original_bbox is not None and tuple(obj.bbox or []) != original_bbox:
        raise AssertionError("Depth diagnostics must not mutate object bbox.")
    return result


def interval_distance(value: float, low: float, high: float) -> float:
    """Return distance from a scalar to a closed interval."""

    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def recompute_scale_depth_formula(
    depth_a: float,
    depth_b: float,
    prior_a_min: float,
    prior_a_max: float,
    prior_b_min: float,
    prior_b_max: float,
    projected_a: float,
    projected_b: float,
) -> dict[str, float]:
    """Recompute ratio/log strict R_sd with explicit invalid-value handling."""

    values = (
        depth_a, depth_b, prior_a_min, prior_a_max, prior_b_min, prior_b_max, projected_a, projected_b
    )
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in values):
        return {key: math.nan for key in (
            "observed_depth_ratio", "expected_ratio_low", "expected_ratio_high", "observed_log_ratio",
            "expected_log_low", "expected_log_high", "distance_below_interval", "distance_above_interval",
            "rsd_ratio", "rsd_log",
        )}
    low = prior_a_min / prior_b_max * projected_b / projected_a
    high = prior_a_max / prior_b_min * projected_b / projected_a
    observed = depth_a / depth_b
    log_observed = math.log(depth_a) - math.log(depth_b)
    log_low, log_high = math.log(low), math.log(high)
    return {
        "observed_depth_ratio": observed,
        "expected_ratio_low": low,
        "expected_ratio_high": high,
        "observed_log_ratio": log_observed,
        "expected_log_low": log_low,
        "expected_log_high": log_high,
        "distance_below_interval": max(0.0, log_low - log_observed),
        "distance_above_interval": max(0.0, log_observed - log_high),
        "rsd_ratio": interval_distance(observed, low, high),
        "rsd_log": interval_distance(log_observed, log_low, log_high),
    }


def swapped_log_residual_consistent(arguments: Mapping[str, float], tolerance: float = 1e-10) -> bool:
    """Verify order invariance of log-space interval distance under A/B swap."""

    forward = recompute_scale_depth_formula(
        arguments["depth_a"], arguments["depth_b"], arguments["prior_a_min"], arguments["prior_a_max"],
        arguments["prior_b_min"], arguments["prior_b_max"], arguments["projected_a"], arguments["projected_b"],
    )
    swapped = recompute_scale_depth_formula(
        arguments["depth_b"], arguments["depth_a"], arguments["prior_b_min"], arguments["prior_b_max"],
        arguments["prior_a_min"], arguments["prior_a_max"], arguments["projected_b"], arguments["projected_a"],
    )
    return math.isclose(forward["rsd_log"], swapped["rsd_log"], rel_tol=tolerance, abs_tol=tolerance)


def coefficient_of_variation(values: Iterable[float]) -> float:
    """Return population CV over finite positive values, or NaN."""

    array = np.asarray([float(value) for value in values if math.isfinite(float(value)) and float(value) > 0])
    if array.size < 2 or float(np.mean(array)) == 0:
        return math.nan
    return float(np.std(array) / np.mean(array))


def boundary_contacts(
    bbox: Optional[Sequence[float]], width: int, height: int, margin_px: float = 2.0
) -> dict[str, bool]:
    """Report whether a detection touches a frame boundary."""

    if bbox is None or len(bbox) != 4:
        return {side: True for side in ("top", "bottom", "left", "right")}
    x1, y1, x2, y2 = (float(value) for value in bbox)
    return {
        "top": y1 <= margin_px,
        "bottom": y2 >= height - margin_px,
        "left": x1 <= margin_px,
        "right": x2 >= width - margin_px,
    }


def diagnostic_labels(row: Mapping[str, object]) -> tuple[str, ...]:
    """Assign transparent heuristic diagnostic labels, not model predictions."""

    labels: list[str] = []
    if any(bool(row.get(f"person_touches_{side}")) for side in ("top", "bottom", "left", "right")):
        labels.append("likely_person_truncated")
    person_aspect = float(row.get("person_aspect_ratio", math.nan))
    # This diagnostic range is intentionally narrower than the formal frozen
    # gate. A wide person box can pass the generic gate while still being a
    # seated/bent observation that does not represent upright body height.
    if not math.isfinite(person_aspect) or not 0.20 <= person_aspect <= 0.58 or float(row.get("person_height_temporal_cv", 0.0)) > 0.15:
        labels.append("likely_person_pose_mismatch")
    cup_width = float(row.get("cup_bbox_width", math.nan))
    cup_height = float(row.get("cup_bbox_height", math.nan))
    if min(cup_width, cup_height) < 16 or float(row.get("cup_bbox_area_ratio", 0.0)) < 0.0002:
        labels.append("likely_cup_too_small")
    cup_aspect = float(row.get("cup_aspect_ratio", math.nan))
    if not math.isfinite(cup_aspect) or not 0.45 <= cup_aspect <= 1.60 or float(row.get("cup_aspect_ratio_temporal_cv", 0.0)) > 0.20:
        labels.append("likely_cup_pose_mismatch")
    cup_depth = float(row.get("cup_current_depth", math.nan))
    cup_iqr = float(row.get("cup_full_depth_iqr", math.nan))
    if math.isfinite(cup_depth) and cup_depth > 0 and math.isfinite(cup_iqr) and cup_iqr / cup_depth > 0.20:
        labels.append("likely_depth_background_contamination")
        if float(row.get("cup_bbox_area_ratio", 0.0)) > 0.05:
            labels.append("likely_bbox_background_contamination")
    if float(row.get("depth_ratio_temporal_cv", 0.0)) > 0.10:
        labels.append("likely_depth_model_instability")
    if float(row.get("rsd_log", 0.0)) > 0.1 and not any(
        label in labels
        for label in (
            "likely_person_truncated", "likely_person_pose_mismatch", "likely_cup_pose_mismatch",
            "likely_depth_background_contamination", "likely_depth_model_instability",
        )
    ):
        labels.append("likely_prior_domain_mismatch")
    return tuple(labels or ["no_obvious_issue"])
