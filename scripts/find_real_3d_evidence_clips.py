#!/usr/bin/env python3
"""Select observation-rich 3D evidence clips without labels or residual values."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CANDIDATE_TYPES = (
    "person_structure",
    "ordinary_object_structure",
    "stable_mask_tracking",
    "mask_overlap",
    "partial_occlusion",
    "full_occlusion",
    "reappearance",
    "out_of_frame",
    "detector_missing_control",
    "multi_object_depth_order",
    "stable_multi_object",
)

OCCLUSION_WINDOW_SIGNALS = (
    "multi_object_mask_overlap",
    "sustained_mask_contact",
    "visible_area_drop",
    "possible_occluder",
    "depth_order_available",
    "disappearance",
    "reappearance",
    "out_of_frame",
    "detector_missing_control",
)

FORBIDDEN_SELECTION_FIELDS = {
    "label",
    "label_name",
    "expected_label",
    "truth_label",
    "is_fake",
    "residual",
    "anomaly_score",
    "final_score",
    "final_anomaly_residual",
}


def find_evidence_clips(
    frame_records: Sequence[Mapping[str, Any]],
    *,
    minimum_duration: int = 2,
) -> list[dict[str, Any]]:
    """Group contiguous observation events using no truth label or residual score."""

    if minimum_duration < 1:
        raise ValueError("minimum_duration must be positive.")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in frame_records:
        forbidden = FORBIDDEN_SELECTION_FIELDS.intersection(row)
        if forbidden:
            raise ValueError("Clip selection records must not contain truth labels or residual scores.")
        video_id, frame_index = str(row["video_id"]), int(row["frame_index"])
        event_flags = {
            "person_structure": float(row.get("keypoint_valid_ratio", 0.0)) > 0.0 and str(row.get("geometry_mode", "")) in {"static_camera_3d", "full_se3_3d"},
            "ordinary_object_structure": int(row.get("ordinary_structure_graph_count", 0)) > 0,
            "stable_mask_tracking": int(row.get("stable_mask_track_count", 0)) > 0,
            "mask_overlap": bool(row.get("has_formal_mask_overlap", False)),
            "partial_occlusion": str(row.get("visibility_state", "")) == "partially_occluded",
            "full_occlusion": str(row.get("visibility_state", "")) == "fully_occluded",
            "reappearance": str(row.get("visibility_state", "")) == "reappeared",
            "out_of_frame": str(row.get("visibility_state", "")) == "out_of_frame",
            "detector_missing_control": str(row.get("visibility_state", "")) == "detector_missing" and bool(row.get("has_history_prediction", False)),
            "multi_object_depth_order": int(row.get("valid_depth_order_count", 0)) > 0,
            "stable_multi_object": int(row.get("formal_mask_object_count", 0)) >= 2 and float(row.get("mean_tracking_quality", 0.0)) > 0.0,
        }
        for candidate_type, enabled in event_flags.items():
            if enabled:
                grouped[(video_id, candidate_type)].append({**dict(row), "frame_index": frame_index})

    candidates = []
    for (video_id, candidate_type), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: item["frame_index"])
        runs: list[list[Mapping[str, Any]]] = []
        for row in rows:
            if not runs or row["frame_index"] > runs[-1][-1]["frame_index"] + 1:
                runs.append([row])
            else:
                runs[-1].append(row)
        for run in runs:
            if len(run) < minimum_duration:
                continue
            object_ids = sorted({value for row in run for value in str(row.get("object_track_ids", "")).split(";") if value})
            labels = sorted({value for row in run for value in str(row.get("semantic_labels", "")).split(";") if value})
            candidates.append({
                "video_id": video_id,
                "start_frame": run[0]["frame_index"],
                "end_frame": run[-1]["frame_index"],
                "object_track_ids": ";".join(object_ids),
                "semantic_labels": ";".join(labels),
                "mask_valid_ratio": sum(float(row.get("mask_valid_ratio", 0.0)) for row in run) / len(run),
                "keypoint_valid_ratio": sum(float(row.get("keypoint_valid_ratio", 0.0)) for row in run) / len(run),
                "overlap_duration": sum(bool(row.get("has_formal_mask_overlap", False)) for row in run),
                "depth_order_confidence": sum(float(row.get("depth_order_confidence", 0.0)) for row in run) / len(run),
                "scene_cut_status": any(bool(row.get("scene_cut", False)) for row in run),
                "geometry_mode": str(run[0].get("geometry_mode", "unavailable")),
                "candidate_type": candidate_type,
                "selection_reason": "observation_availability_and_event_type_only",
                "quality": sum(float(row.get("observation_quality", 0.0)) for row in run) / len(run),
            })
    return candidates


def write_candidates(rows: Sequence[Mapping[str, Any]], output_csv: Path) -> None:
    """Write candidate rows, preserving an explicit empty report."""

    columns = (
        "video_id", "start_frame", "end_frame", "object_track_ids",
        "semantic_labels", "mask_valid_ratio", "keypoint_valid_ratio",
        "overlap_duration", "depth_order_confidence", "scene_cut_status",
        "geometry_mode", "candidate_type", "selection_reason", "quality",
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def scan_occlusion_event_windows(
    frame_records: Sequence[Mapping[str, Any]],
    *,
    window_sizes: Sequence[int] = (8, 16, 24, 32),
    stride: int | None = None,
) -> list[dict[str, Any]]:
    """Audit multi-window occlusion evidence formation without anomaly scores.

    Every tested window is returned. A rejected window retains all failed
    checks; an event-free window is explicitly ``no_observable_occlusion_event``.
    """

    if not window_sizes or any(int(size) < 2 for size in window_sizes):
        raise ValueError("window_sizes must contain integers >= 2.")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in frame_records:
        forbidden = FORBIDDEN_SELECTION_FIELDS.intersection(row)
        if forbidden:
            raise ValueError("Occlusion window scanning must not consume truth labels or anomaly scores.")
        grouped[str(row["video_id"])].append(row)
    output: list[dict[str, Any]] = []
    for video_id, raw_rows in sorted(grouped.items()):
        rows = sorted(raw_rows, key=lambda item: int(item["frame_index"]))
        for raw_size in window_sizes:
            size = int(raw_size)
            if not rows:
                continue
            starts = list(range(0, max(1, len(rows) - size + 1), stride or max(1, size // 2)))
            last_start = max(0, len(rows) - size)
            if last_start not in starts:
                starts.append(last_start)
            for start in sorted(set(starts)):
                window = rows[start:start + size]
                if len(window) < 2:
                    continue
                counts = {
                    signal: sum(bool(row.get(signal, False)) for row in window)
                    for signal in OCCLUSION_WINDOW_SIGNALS
                }
                has_contact = counts["multi_object_mask_overlap"] > 0 or counts["sustained_mask_contact"] >= 2
                has_change = counts["visible_area_drop"] > 0 or counts["disappearance"] > 0
                observable_event = has_contact and has_change and counts["possible_occluder"] > 0
                failed = []
                if not has_contact:
                    failed.append("no_sustained_mask_contact_or_overlap")
                if not has_change:
                    failed.append("no_visible_area_drop_or_disappearance")
                if counts["possible_occluder"] == 0:
                    failed.append("possible_occluder_missing")
                if counts["depth_order_available"] == 0:
                    failed.append("depth_order_unavailable")
                scene_cut = any(bool(row.get("scene_cut", False)) for row in window)
                if scene_cut:
                    failed.append("scene_cut_in_window")
                accepted = observable_event and counts["depth_order_available"] > 0 and not scene_cut
                if accepted:
                    status = "formal_occlusion_candidate"
                    reason = ""
                elif not observable_event:
                    status = "no_observable_occlusion_event"
                    reason = ";".join(failed)
                else:
                    status = "candidate_rejected"
                    reason = ";".join(failed)
                output.append({
                    "video_id": video_id,
                    "window_size": size,
                    "start_frame": int(window[0]["frame_index"]),
                    "end_frame": int(window[-1]["frame_index"]),
                    **counts,
                    "scene_cut": scene_cut,
                    "accepted": accepted,
                    "status": status,
                    "rejection_reasons": reason,
                    "selection_source": "observation_signals_only",
                    "truth_labels_used": False,
                    "anomaly_scores_used": False,
                })
    return output


def write_occlusion_window_diagnostics(
    rows: Sequence[Mapping[str, Any]], output_csv: Path,
) -> None:
    """Write accepted and rejected multi-window occlusion diagnostics."""

    columns = (
        "video_id", "window_size", "start_frame", "end_frame",
        *OCCLUSION_WINDOW_SIGNALS, "scene_cut", "accepted", "status",
        "rejection_reasons", "selection_source", "truth_labels_used",
        "anomaly_scores_used",
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame_records", type=Path, default=PROJECT_ROOT / "outputs/real_3d_evidence_coverage/frame_observation_availability.json")
    parser.add_argument("--output_csv", type=Path, default=PROJECT_ROOT / "outputs/real_3d_evidence_coverage/evidence_clip_candidates.csv")
    parser.add_argument("--min_duration", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.frame_records.exists():
        raise FileNotFoundError(f"Observation availability file not found: {args.frame_records}")
    payload = json.loads(args.frame_records.read_text(encoding="utf-8"))
    candidates = find_evidence_clips(payload, minimum_duration=args.min_duration)
    write_candidates(candidates, args.output_csv)
    print(f"Saved {len(candidates)} observation-only candidate clip(s) to {args.output_csv}")


if __name__ == "__main__":
    main()
