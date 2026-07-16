"""Depth temporal consistency residuals for tracked objects.

For the same physical object, its real-world scale is approximately stable
over a short time span. Under perspective projection, moving closer decreases
depth while increasing projection scale; moving farther increases depth while
decreasing projection scale. Therefore

    log(relative_depth) + log(projection_scale)

should remain relatively stable for a correctly tracked rigid object. This is
a lightweight consistency approximation, not strict 3D reconstruction. It can
be affected by monocular depth scale drift, bbox_area approximating mask_area,
object pose changes, non-rigid deformation, camera zoom, wrong association, and
detector jitter.
"""

from __future__ import annotations

import math
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .observations import FrameObservationJSON, ObjectObservationJSON
from .validity import MissingReason, ResidualEvidence, aggregate_residual_evidence


LEGACY_ZERO_MISSING_BEHAVIOR = True


@dataclass(frozen=True)
class DepthTemporalResidualResult:
    """Residual result for one same-track object transition."""

    track_id: str
    label: str
    previous_frame_index: int
    current_frame_index: int
    previous_depth: float
    current_depth: float
    previous_depth_reference: float
    current_depth_reference: float
    previous_relative_depth: float
    current_relative_depth: float
    previous_projection_scale: float
    current_projection_scale: float
    previous_geometry_state: float
    current_geometry_state: float
    raw_residual: float
    tolerance: float
    residual: float
    confidence_weight: float
    weighted_residual: float
    valid: bool
    skip_reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a CSV/JSON-friendly dictionary."""

        return asdict(self)


def compute_frame_depth_reference(
    frame: FrameObservationJSON,
    depth_map: Optional[np.ndarray] = None,
    eps: float = 1e-6,
) -> Optional[float]:
    """Return a robust frame-level relative-depth reference.

    If a depth map is available, this returns the median of all finite positive
    pixels. Otherwise it falls back to the median of finite positive object
    depths in the frame. The reference reduces monocular depth's frame-to-frame
    global scale drift; it is still relative depth, not metric distance.
    """

    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}.")

    if depth_map is not None:
        values = np.asarray(depth_map, dtype=float)
        valid = values[np.isfinite(values) & (values > eps)]
        if valid.size:
            return float(np.median(valid))

    depths = np.asarray([float(obj.depth) for obj in frame.objects], dtype=float)
    valid_depths = depths[np.isfinite(depths) & (depths > eps)]
    if valid_depths.size:
        return float(np.median(valid_depths))
    return None


def compute_depth_temporal_residual(
    previous_object: ObjectObservationJSON,
    current_object: ObjectObservationJSON,
    previous_depth_reference: float,
    current_depth_reference: float,
    tolerance: float = 0.10,
    eps: float = 1e-6,
) -> DepthTemporalResidualResult:
    """Compute legacy R_depth_cons for one same-track object transition.

    The residual is independent from scale priors. It only compares how one
    tracked object's relative depth and image projection scale evolve between
    two frames:

        p_t = sqrt(mask_area_t / frame_area_t)
        z_rel_t = object_depth_t / frame_depth_reference_t
        g_t = log(z_rel_t + eps) + log(p_t + eps)
        R_depth_cons = max(0, abs(g_current - g_previous) - tolerance)

    Invalid transitions retain historical zero-valued numeric fields and must
    be converted with ``depth_transition_evidence`` in new code. The explicitly
    named baseline is ``r_depth_cons_2p5d``.

    ``weighted_residual`` multiplies the residual by the geometric mean of the
    two detection confidences.
    """

    if tolerance < 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance}.")
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps}.")

    base = _invalid_result_base(
        previous_object,
        current_object,
        previous_depth_reference,
        current_depth_reference,
        tolerance,
    )
    skip_reason = _validate_transition(
        previous_object,
        current_object,
        previous_depth_reference,
        current_depth_reference,
        eps,
    )
    if skip_reason:
        return DepthTemporalResidualResult(
            **base,
            raw_residual=0.0,
            residual=0.0,
            confidence_weight=_confidence_weight(previous_object, current_object),
            weighted_residual=0.0,
            valid=False,
            skip_reason=skip_reason,
        )

    previous_projection_scale = _projection_scale(previous_object)
    current_projection_scale = _projection_scale(current_object)
    previous_relative_depth = float(previous_object.depth) / float(previous_depth_reference)
    current_relative_depth = float(current_object.depth) / float(current_depth_reference)
    previous_geometry_state = math.log(previous_relative_depth + eps) + math.log(
        previous_projection_scale + eps
    )
    current_geometry_state = math.log(current_relative_depth + eps) + math.log(
        current_projection_scale + eps
    )
    raw_residual = abs(current_geometry_state - previous_geometry_state)
    residual = max(0.0, raw_residual - tolerance)
    confidence_weight = _confidence_weight(previous_object, current_object)

    return DepthTemporalResidualResult(
        track_id=str(current_object.track_id or previous_object.track_id or ""),
        label=str(
            current_object.canonical_label
            or previous_object.canonical_label
            or current_object.label
        ),
        previous_frame_index=int(_frame_index_from_object_id(previous_object.object_id)),
        current_frame_index=int(_frame_index_from_object_id(current_object.object_id)),
        previous_depth=float(previous_object.depth),
        current_depth=float(current_object.depth),
        previous_depth_reference=float(previous_depth_reference),
        current_depth_reference=float(current_depth_reference),
        previous_relative_depth=previous_relative_depth,
        current_relative_depth=current_relative_depth,
        previous_projection_scale=previous_projection_scale,
        current_projection_scale=current_projection_scale,
        previous_geometry_state=previous_geometry_state,
        current_geometry_state=current_geometry_state,
        raw_residual=raw_residual,
        tolerance=float(tolerance),
        residual=residual,
        confidence_weight=confidence_weight,
        weighted_residual=confidence_weight * residual,
        valid=True,
        skip_reason="",
    )


def r_depth_cons_2p5d(
    previous_object: ObjectObservationJSON,
    current_object: ObjectObservationJSON,
    previous_depth_reference: float,
    current_depth_reference: float,
    tolerance: float = 0.10,
    eps: float = 1e-6,
) -> DepthTemporalResidualResult:
    """Explicitly named 2.5D compatibility entry point for R_depth_cons."""

    return compute_depth_temporal_residual(
        previous_object,
        current_object,
        previous_depth_reference,
        current_depth_reference,
        tolerance=tolerance,
        eps=eps,
    )


def depth_transition_evidence(
    result: DepthTemporalResidualResult,
) -> ResidualEvidence:
    """Convert a legacy transition result to NaN-aware 2.5D evidence."""

    source_ids = (
        result.track_id,
        f"frame:{result.previous_frame_index}",
        f"frame:{result.current_frame_index}",
    )
    if not result.valid:
        return ResidualEvidence.missing(
            "r_depth_cons_2p5d",
            result.skip_reason or MissingReason.NO_VALID_EVIDENCE,
            source_ids=source_ids,
            metadata={"legacy_zero_values_ignored": True},
        )
    return ResidualEvidence.observed(
        "r_depth_cons_2p5d",
        result.residual,
        quality=min(1.0, max(0.0, result.confidence_weight)),
        source_ids=source_ids,
        metadata={"raw_residual": result.raw_residual, "tolerance": result.tolerance},
    )


def aggregate_depth_transition_evidence(
    results: Sequence[DepthTemporalResidualResult],
    method: str = "mean",
) -> ResidualEvidence:
    """Aggregate valid transitions and return NaN when no evidence exists."""

    return aggregate_residual_evidence(
        "r_depth_cons_2p5d",
        [depth_transition_evidence(result) for result in results],
        method=method,
        require_all_valid=False,
    )


def _invalid_result_base(
    previous_object: ObjectObservationJSON,
    current_object: ObjectObservationJSON,
    previous_depth_reference: float,
    current_depth_reference: float,
    tolerance: float,
) -> dict[str, Any]:
    """Create common fields for invalid residual outputs."""

    return {
        "track_id": str(current_object.track_id or previous_object.track_id or ""),
        "label": str(
            current_object.canonical_label
            or previous_object.canonical_label
            or current_object.label
        ),
        "previous_frame_index": int(_frame_index_from_object_id(previous_object.object_id)),
        "current_frame_index": int(_frame_index_from_object_id(current_object.object_id)),
        "previous_depth": float(previous_object.depth),
        "current_depth": float(current_object.depth),
        "previous_depth_reference": float(previous_depth_reference),
        "current_depth_reference": float(current_depth_reference),
        "previous_relative_depth": 0.0,
        "current_relative_depth": 0.0,
        "previous_projection_scale": 0.0,
        "current_projection_scale": 0.0,
        "previous_geometry_state": 0.0,
        "current_geometry_state": 0.0,
        "tolerance": float(tolerance),
    }


def _validate_transition(
    previous_object: ObjectObservationJSON,
    current_object: ObjectObservationJSON,
    previous_depth_reference: float,
    current_depth_reference: float,
    eps: float,
) -> str:
    """Return a skip reason for invalid transitions, or an empty string."""

    if not previous_object.track_id or not current_object.track_id:
        return "missing_track_id"
    if previous_object.track_id != current_object.track_id:
        return "track_id_mismatch"
    for name, value in {
        "previous_depth": previous_object.depth,
        "current_depth": current_object.depth,
        "previous_depth_reference": previous_depth_reference,
        "current_depth_reference": current_depth_reference,
        "previous_mask_area": previous_object.mask_area,
        "current_mask_area": current_object.mask_area,
        "previous_frame_area": previous_object.frame_area,
        "current_frame_area": current_object.frame_area,
    }.items():
        number = float(value)
        if not math.isfinite(number) or number <= eps:
            return f"invalid_{name}"
    return ""


def _projection_scale(obj: ObjectObservationJSON) -> float:
    """Return sqrt(mask_area / frame_area) for an object."""

    return math.sqrt(float(obj.mask_area) / float(obj.frame_area))


def _confidence_weight(
    previous_object: ObjectObservationJSON,
    current_object: ObjectObservationJSON,
) -> float:
    """Return sqrt(previous_confidence * current_confidence), clamped to [0, inf)."""

    previous_confidence = max(0.0, float(previous_object.confidence))
    current_confidence = max(0.0, float(current_object.confidence))
    return math.sqrt(previous_confidence * current_confidence)


def _frame_index_from_object_id(object_id: str) -> int:
    """Best-effort frame index extraction for direct function calls.

    Pipeline callers should overwrite frame indices via ``with_frame_indices``.
    This fallback keeps standalone unit tests and demo objects concise.
    """

    marker = "_f"
    if marker in object_id:
        suffix = object_id.rsplit(marker, maxsplit=1)[-1].split("_", maxsplit=1)[0]
        if suffix.isdigit():
            return int(suffix)
    return -1


def with_frame_indices(
    result: DepthTemporalResidualResult,
    previous_frame_index: int,
    current_frame_index: int,
) -> DepthTemporalResidualResult:
    """Return a result with explicit frame indices."""

    data = result.to_dict()
    data["previous_frame_index"] = int(previous_frame_index)
    data["current_frame_index"] = int(current_frame_index)
    return DepthTemporalResidualResult(**data)


def aggregate_track_depth_residuals(
    results: Sequence[DepthTemporalResidualResult],
    topk: int = 3,
) -> list[dict[str, Any]]:
    """Aggregate R_depth_cons transitions into track-level summaries."""

    grouped: dict[str, list[DepthTemporalResidualResult]] = {}
    for result in results:
        grouped.setdefault(result.track_id, []).append(result)

    summaries: list[dict[str, Any]] = []
    for track_id, track_results in sorted(grouped.items()):
        valid_results = [result for result in track_results if result.valid]
        residuals = [result.residual for result in valid_results]
        weighted = [result.weighted_residual for result in valid_results]
        summaries.append(
            {
                "track_id": track_id,
                "label": _first_label(track_results),
                "num_transitions": len(track_results),
                "mean_residual": _mean(residuals),
                "max_residual": max(residuals) if residuals else 0.0,
                "topk_mean_residual": _topk_mean(residuals, topk=topk),
                "mean_weighted_residual": _mean(weighted),
                "valid_ratio": (
                    float(len(valid_results) / len(track_results))
                    if track_results
                    else 0.0
                ),
            }
        )
    return summaries


def aggregate_clip_depth_residuals(
    results: Sequence[DepthTemporalResidualResult],
    frame_indices: Sequence[int],
    clip_id: str = "clip",
    topk: int = 3,
) -> dict[str, Any]:
    """Legacy clip aggregation; no valid transitions historically aggregate to zero."""

    frame_set = {int(index) for index in frame_indices}
    clip_results = [
        result
        for result in results
        if result.previous_frame_index in frame_set and result.current_frame_index in frame_set
    ]
    valid_results = [result for result in clip_results if result.valid]
    residuals = [result.residual for result in valid_results]
    weighted = [result.weighted_residual for result in valid_results]
    tracks = {result.track_id for result in clip_results if result.track_id}
    return {
        "clip_id": clip_id,
        "num_tracks": len(tracks),
        "num_valid_transitions": len(valid_results),
        "mean_R_depth_cons": _mean(residuals),
        "max_R_depth_cons": max(residuals) if residuals else 0.0,
        "topk_mean_R_depth_cons": _topk_mean(residuals, topk=topk),
        "weighted_clip_score": _mean(weighted),
    }


def compute_track_transitions(
    frames: Sequence[FrameObservationJSON],
    depth_references: Mapping[int, Optional[float]],
    max_frame_gap: int = 1,
    tolerance: float = 0.10,
) -> tuple[list[DepthTemporalResidualResult], dict[str, int]]:
    """Compute R_depth_cons for same-track objects across sorted frames."""

    sorted_frames = sorted(frames, key=lambda item: int(item.frame_index))
    previous_by_track: dict[str, tuple[FrameObservationJSON, ObjectObservationJSON]] = {}
    results: list[DepthTemporalResidualResult] = []
    stats = {
        "valid_transitions": 0,
        "skipped_invalid_depth": 0,
        "skipped_missing_reference": 0,
        "skipped_frame_gap": 0,
    }

    for frame in sorted_frames:
        current_reference = depth_references.get(int(frame.frame_index))
        for obj in frame.objects:
            if not obj.track_id:
                continue
            previous = previous_by_track.get(obj.track_id)
            if previous is not None:
                previous_frame, previous_obj = previous
                frame_gap = int(frame.frame_index) - int(previous_frame.frame_index)
                if frame_gap <= 0:
                    stats["skipped_frame_gap"] += 1
                elif frame_gap > max_frame_gap:
                    stats["skipped_frame_gap"] += 1
                else:
                    previous_reference = depth_references.get(
                        int(previous_frame.frame_index)
                    )
                    if previous_reference is None or current_reference is None:
                        stats["skipped_missing_reference"] += 1
                    else:
                        result = compute_depth_temporal_residual(
                            previous_obj,
                            obj,
                            previous_reference,
                            current_reference,
                            tolerance=tolerance,
                        )
                        result = with_frame_indices(
                            result,
                            previous_frame_index=int(previous_frame.frame_index),
                            current_frame_index=int(frame.frame_index),
                        )
                        if result.valid:
                            stats["valid_transitions"] += 1
                        elif "depth" in result.skip_reason or "area" in result.skip_reason:
                            stats["skipped_invalid_depth"] += 1
                        results.append(result)
            previous_by_track[obj.track_id] = (frame, obj)
    return results, stats


def save_depth_consistency_tracks_plot(
    results: Sequence[DepthTemporalResidualResult],
    output_path: str | Path,
) -> Path:
    """Save a PNG showing relative depth, projection scale, and residual by track."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    valid_results = [result for result in results if result.valid]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    if not valid_results:
        axes[1].text(0.5, 0.5, "No valid R_depth_cons transitions", ha="center")
        for axis in axes:
            axis.grid(True, alpha=0.25)
        fig.suptitle("Depth Temporal Consistency Tracks")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path

    for track_id, track_results in _group_valid_by_track(valid_results).items():
        frames = [result.current_frame_index for result in track_results]
        rel_depth = [result.current_relative_depth for result in track_results]
        projection = [result.current_projection_scale for result in track_results]
        residual = [result.residual for result in track_results]
        label = _first_label(track_results)
        line_label = f"{track_id} ({label})"
        axes[0].plot(frames, rel_depth, marker="o", label=line_label)
        axes[1].plot(frames, projection, marker="o", label=line_label)
        axes[2].plot(frames, residual, marker="o", label=line_label)

    axes[0].set_ylabel("relative depth")
    axes[1].set_ylabel("projection scale")
    axes[2].set_ylabel("R_depth_cons")
    axes[2].set_xlabel("frame_index")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Depth Temporal Consistency Tracks")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def depth_consistency_plot_series_from_csv(
    pair_csv_path: str | Path,
) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Load valid transition series from the pair CSV for plotting.

    Residual points are taken only from valid CSV rows and are placed at
    ``current_frame_index``. Invalid transitions and first observations without
    predecessors are not filled with zero; they are simply absent from the
    residual series.
    """

    path = Path(pair_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing depth consistency pair CSV: {path}")

    grouped: dict[str, dict[str, dict[int, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if not _csv_bool(row.get("valid", "")):
                continue
            video_id = str(row.get("video_id", "video"))
            track_id = str(row.get("track_id", "track"))
            label = str(row.get("label", "object"))
            key = f"{video_id}:{track_id} ({label})"
            series = grouped.setdefault(
                key,
                {
                    "relative_depth": {},
                    "projection_scale": {},
                    "raw_residual": {},
                    "residual": {},
                },
            )
            previous_frame = int(float(row["previous_frame_index"]))
            current_frame = int(float(row["current_frame_index"]))
            series["relative_depth"].setdefault(
                previous_frame,
                float(row["previous_relative_depth"]),
            )
            series["relative_depth"][current_frame] = float(row["current_relative_depth"])
            series["projection_scale"].setdefault(
                previous_frame,
                float(row["previous_projection_scale"]),
            )
            series["projection_scale"][current_frame] = float(
                row["current_projection_scale"]
            )
            series["raw_residual"][current_frame] = float(row["raw_residual"])
            series["residual"][current_frame] = float(row["residual"])

    return {
        key: {
            name: sorted(values.items())
            for name, values in value.items()
        }
        for key, value in grouped.items()
    }


def save_depth_consistency_tracks_plot_from_csv(
    pair_csv_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Save a depth consistency plot using valid rows from pair CSV only."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    series_by_track = depth_consistency_plot_series_from_csv(pair_csv_path)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    if not series_by_track:
        axes[1].text(0.5, 0.5, "No valid R_depth_cons transitions", ha="center")
        for axis in axes:
            axis.grid(True, alpha=0.25)
        fig.suptitle("Depth Temporal Consistency Tracks")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return path

    for label, series in series_by_track.items():
        for axis, field in zip(
            axes,
            ["relative_depth", "projection_scale", "residual"],
        ):
            points = series[field]
            if not points:
                continue
            frames = [frame for frame, _ in points]
            values = [value for _, value in points]
            axis.plot(frames, values, marker="o", label=label)

    axes[0].set_ylabel("relative depth")
    axes[1].set_ylabel("projection scale")
    axes[2].set_ylabel("R_depth_cons")
    axes[2].set_xlabel("frame_index")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Depth Temporal Consistency Tracks")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_raw_and_thresholded_residual_plots_from_csv(
    pair_csv_path: str | Path,
    raw_output_path: str | Path,
    thresholded_output_path: str | Path,
    tolerance: float,
) -> tuple[Path, Path]:
    """Save separate raw and thresholded R_depth_cons plots from pair CSV."""

    raw_path = _save_residual_plot_from_csv(
        pair_csv_path,
        raw_output_path,
        field="raw_residual",
        ylabel="raw residual",
        title=f"Raw Depth Temporal Residual (tolerance={tolerance:g})",
        tolerance=tolerance,
    )
    thresholded_path = _save_residual_plot_from_csv(
        pair_csv_path,
        thresholded_output_path,
        field="residual",
        ylabel="R_depth_cons",
        title=f"Thresholded R_depth_cons (tolerance={tolerance:g})",
        tolerance=None,
    )
    return raw_path, thresholded_path


def _save_residual_plot_from_csv(
    pair_csv_path: str | Path,
    output_path: str | Path,
    field: str,
    ylabel: str,
    title: str,
    tolerance: Optional[float] = None,
) -> Path:
    """Save one residual plot from CSV-valid transitions."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    series_by_track = depth_consistency_plot_series_from_csv(pair_csv_path)

    fig, ax = plt.subplots(figsize=(11, 5))
    point_count = 0
    for label, series in series_by_track.items():
        points = series.get(field, [])
        if not points:
            continue
        frames = [frame for frame, _ in points]
        values = [value for _, value in points]
        point_count += len(points)
        ax.plot(frames, values, marker="o", label=label)
    if tolerance is not None:
        ax.axhline(
            float(tolerance),
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"tolerance={tolerance:g}",
        )
    if point_count == 0:
        ax.text(0.5, 0.5, "No valid R_depth_cons transitions", ha="center")
    ax.set_title(title)
    ax.set_xlabel("current_frame_index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if point_count > 0 or tolerance is not None:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _group_valid_by_track(
    results: Iterable[DepthTemporalResidualResult],
) -> dict[str, list[DepthTemporalResidualResult]]:
    """Group valid residuals by track id."""

    grouped: dict[str, list[DepthTemporalResidualResult]] = {}
    for result in results:
        grouped.setdefault(result.track_id, []).append(result)
    return {track_id: sorted(items, key=lambda item: item.current_frame_index) for track_id, items in grouped.items()}


def _first_label(results: Sequence[DepthTemporalResidualResult]) -> str:
    """Return the first non-empty label in a result sequence."""

    for result in results:
        if result.label:
            return result.label
    return ""


def _mean(values: Sequence[float]) -> float:
    """Return mean value or zero for an empty sequence."""

    return float(np.mean(values)) if values else 0.0


def _topk_mean(values: Sequence[float], topk: int = 3) -> float:
    """Return mean of the largest k values."""

    if not values:
        return 0.0
    k = max(1, min(int(topk), len(values)))
    return float(np.mean(sorted(values, reverse=True)[:k]))


def _csv_bool(value: str) -> bool:
    """Parse bool-like CSV values."""

    return str(value).strip().lower() in {"1", "true", "yes", "y"}
