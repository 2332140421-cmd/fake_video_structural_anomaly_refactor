#!/usr/bin/env python3
"""Run label-independent partial P4 aggregation on readiness-approved clips."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from semantic3d.aggregation_v2 import (  # noqa: E402
    AggregationEvidence,
    EvidenceApplicability,
    aggregate_multilevel_evidence,
    get_evidence_registry,
)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _discover_dynamic_clips(dynamic_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    output = {}
    for report_path in sorted(dynamic_root.glob("*/smoke_report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        video_id = str(report.get("video_id", ""))
        if video_id:
            output[video_id] = (report_path.parent, report)
    return output


def _object_metadata(coverage_root: Path, video_id: str) -> dict[tuple[str, int, str], dict[str, Any]]:
    output = {}
    for row in _read_csv(coverage_root / "mask_coverage.csv"):
        if row["video_id"] != video_id or not _bool(row["valid"]):
            continue
        try:
            bbox = json.loads(row["mask_bbox"])
        except (TypeError, json.JSONDecodeError):
            bbox = None
        output[(video_id, int(row["frame_index"]), row["object_track_id"])] = {
            "semantic_label": row["class_name"],
            "localization_bbox": bbox,
            "localization_mask_reference": row["visible_mask_path"],
            "localization_source": "formal_visible_instance_mask",
        }
    return output


def _point_evidences(
    clip_dir: Path,
    *,
    video_id: str,
    geometry_mode: str,
) -> list[AggregationEvidence]:
    """Load only persisted formal residuals; no value is reconstructed for plotting."""

    output: list[AggregationEvidence] = []
    engineering_quality = 0.50
    common_metadata = {
        "video_id": video_id,
        "geometry_mode": geometry_mode,
        "quality_source": "conservative_legacy_artifact_default",
        "quality_is_probability": False,
        "truth_label_used": False,
    }
    for row in _read_csv(clip_dir / "direction_residuals.csv"):
        for field, suffix in (("own_history", "own"), ("object_median", "object_median")):
            if _bool(row.get(f"{field}_valid", False)) and math.isfinite(float(row[field])):
                output.append(AggregationEvidence.observed(
                    float(row[field]), quality=engineering_quality,
                    branch_name="direction_consistency",
                    source_id=f"direction:{row['point_id']}:{row['current_frame_index']}:{suffix}",
                    frame_index=int(row["current_frame_index"]),
                    object_track_id=row["object_track_id"],
                    point_or_edge_id=row["point_id"], metadata=common_metadata,
                ))
    for row in _read_csv(clip_dir / "relative_velocity.csv"):
        for field, valid_field, suffix in (
            ("speed_change", "speed_change_valid", "history"),
            ("point_vs_object_median_speed", "point_vs_object_median_speed_valid", "object_median"),
        ):
            if _bool(row.get(valid_field, False)) and math.isfinite(float(row[field])):
                output.append(AggregationEvidence.observed(
                    float(row[field]), quality=engineering_quality,
                    branch_name="relative_velocity_change",
                    source_id=f"velocity:{row['point_id']}:{row['current_frame_index']}:{suffix}",
                    frame_index=int(row["current_frame_index"]),
                    object_track_id=row["object_track_id"],
                    point_or_edge_id=row["point_id"], metadata=common_metadata,
                ))
    for row in _read_csv(clip_dir / "dynamic_reprojection_residuals.csv"):
        if _bool(row.get("formal_residual_valid", False)) and math.isfinite(float(row["formal_residual"])):
            output.append(AggregationEvidence.observed(
                float(row["formal_residual"]), quality=engineering_quality,
                branch_name="dynamic_reprojection",
                source_id=f"reprojection:{row['point_id']}:{row['target_frame_index']}",
                frame_index=int(row["target_frame_index"]),
                object_track_id=row["object_track_id"],
                point_or_edge_id=row["point_id"], metadata=common_metadata,
            ))
    return output


def _edge_evidences(coverage_root: Path, *, video_id: str) -> list[AggregationEvidence]:
    """Load persisted fixed-edge residuals from the shared-3D smoke."""

    path = coverage_root / "shared_3d_smoke/structure_temporal_evidence.csv"
    output = []
    for row in _read_csv(path):
        if row["video_id"] != video_id or row["evidence_level"] != "edge" or not _bool(row["valid"]):
            continue
        value = float(row["value"])
        if not math.isfinite(value):
            continue
        output.append(AggregationEvidence.observed(
            value, quality=float(row["quality"]),
            branch_name="structure_temporal",
            source_id=f"structure:{row['object_track_id']}:{row['frame_index']}:{row['point_or_edge_id']}",
            frame_index=int(row["frame_index"]),
            object_track_id=row["object_track_id"],
            point_or_edge_id=row["point_or_edge_id"],
            metadata={
                "video_id": video_id,
                "semantic_label": row["semantic_label"],
                "truth_label_used": False,
                "source_artifact": str(path),
            },
        ))
    return output


def _aggregate_rows(result: Any) -> dict[str, list[dict[str, Any]]]:
    return {
        "point": [_json_safe(asdict(item)) for item in result.point_aggregates],
        "edge": [_json_safe(asdict(item)) for item in result.edge_aggregates],
        "object": [_json_safe(asdict(item)) for item in result.object_aggregates],
        "frame": [_json_safe(asdict(item)) for item in result.frame_aggregates],
    }


def _plot_localization(result: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    frames = [item.frame_index for item in result.frame_aggregates]
    raw = [item.value for item in result.frame_aggregates]
    axes[0].plot(frames, raw, marker="o", label="raw frame score")
    axes[0].plot(frames, result.smoothed_frame_scores, marker="s", label="causal median")
    axes[0].set_title("Temporal evidence localization")
    axes[0].set_xlabel("global frame index")
    axes[0].legend(fontsize=8)
    object_values = [(item.object_track_id, item.value) for item in result.object_aggregates if item.valid]
    if object_values:
        labels, values = zip(*object_values)
        axes[1].bar(range(len(values)), values, color="#4e79a7")
        axes[1].set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
    axes[1].set_title("Object-frame aggregates")
    branches = result.clip_aggregate.branch_coverage
    axes[2].bar(range(len(branches)), list(branches.values()), color="#59a14f")
    axes[2].set_xticks(range(len(branches)), list(branches), rotation=60, ha="right", fontsize=7)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Active branch coverage")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Partial P4 Aggregation Smoke (No Authenticity Decision)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_partial_p4_aggregation_smoke(
    *,
    coverage_root: Path,
    dynamic_root: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Aggregate only readiness-approved persisted evidence."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    aggregation = config["aggregation"]
    temporal = config["temporal_localization"]
    registry = get_evidence_registry()
    output_dir.mkdir(parents=True, exist_ok=True)
    registry_payload = {
        "branches": {name: spec.to_dict() for name, spec in registry.items()},
        "diagnostic_auto_registration": False,
        "truth_labels_used": False,
    }
    (output_dir / "evidence_registry.json").write_text(
        json.dumps(registry_payload, indent=2), encoding="utf-8",
    )
    coverage_rows = _read_csv(coverage_root / "per_video_summary.csv")
    dynamic = _discover_dynamic_clips(dynamic_root)
    branch_rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    all_rows = {"point": [], "edge": [], "object": [], "frame": []}
    interval_rows: list[dict[str, Any]] = []
    clip_payload: list[dict[str, Any]] = []
    ready_results = []
    for video in coverage_rows:
        video_id = video["video_id"]
        ready = _bool(video.get("ready_for_partial_p4", False))
        if not ready:
            reason = video.get("primary_missing_reason") or "no_formal_structure_or_occlusion_residual_branch"
            reports.append({"video_id": video_id, "status": "not_ready", "reason": reason})
            for name in registry:
                branch_rows.append({
                    "video_id": video_id, "branch_name": name,
                    "applicability": EvidenceApplicability.OBSERVATION_MISSING.value,
                    "valid_evidence_count": 0, "coverage": 0.0,
                    "missing_reason": reason,
                })
            continue
        if video_id not in dynamic:
            reports.append({"video_id": video_id, "status": "not_ready", "reason": "dynamic_evidence_artifact_missing"})
            continue
        clip_dir, dynamic_report = dynamic[video_id]
        geometry_mode = str(dynamic_report["geometry_mode"])
        points = _point_evidences(clip_dir, video_id=video_id, geometry_mode=geometry_mode)
        edges = _edge_evidences(coverage_root, video_id=video_id)
        metadata = _object_metadata(coverage_root, video_id)
        result = aggregate_multilevel_evidence(
            point_evidences=points, edge_evidences=edges, video_id=video_id,
            clip_id=str(dynamic_report["clip_id"]), object_metadata=metadata,
            method=str(aggregation["method"]), top_k=int(aggregation["top_k"]),
            quality_floor=float(aggregation["quality_floor"]),
            temporal_threshold=float(temporal["high_evidence_threshold"]),
            moving_median_window=int(temporal["moving_median_window"]),
            max_gap=int(temporal["max_gap"]),
            minimum_duration=int(temporal["minimum_duration"]),
        )
        ready_results.append(result)
        rows = _aggregate_rows(result)
        for level in all_rows:
            all_rows[level].extend(rows[level])
        interval_rows.extend({"video_id": video_id, **_json_safe(asdict(item))} for item in result.intervals)
        counts: dict[str, int] = {}
        for item in points:
            counts[item.branch_name] = counts.get(item.branch_name, 0) + int(item.valid)
        for item in edges:
            counts[item.branch_name] = counts.get(item.branch_name, 0) + int(item.valid)
        current_branch_rows = []
        for name, spec in registry.items():
            supported = geometry_mode in spec.supported_geometry_modes
            count = counts.get(name, 0)
            if count:
                applicability, reason = EvidenceApplicability.APPLICABLE.value, ""
            elif not supported:
                applicability, reason = EvidenceApplicability.UNSUPPORTED_MODE.value, "geometry_mode_not_supported"
            elif spec.event_conditioned and float(video.get("formal_mask_valid_ratio", 0.0)) >= 0.999:
                applicability, reason = EvidenceApplicability.NOT_APPLICABLE.value, "no_observable_event"
            elif spec.event_conditioned:
                applicability, reason = EvidenceApplicability.OBSERVATION_MISSING.value, "incomplete_event_observation"
            else:
                applicability, reason = EvidenceApplicability.OBSERVATION_MISSING.value, "formal_residual_artifact_unavailable"
            branch_row = {
                "video_id": video_id, "branch_name": name,
                "applicability": applicability, "valid_evidence_count": count,
                "coverage": 1.0 if count or applicability == "not_applicable" else 0.0,
                "missing_reason": reason,
            }
            current_branch_rows.append(branch_row)
            branch_rows.append(branch_row)
        clip_row = _json_safe(asdict(result.clip_aggregate))
        clip_row["branch_coverage"] = {
            row["branch_name"]: row["coverage"] for row in current_branch_rows
        }
        clip_row["coverage_dimensions"]["branch_coverage"] = float(np.mean([
            row["coverage"] for row in current_branch_rows
        ])) if current_branch_rows else 0.0
        clip_payload.append(clip_row)
        reports.append({
            "video_id": video_id, "status": "partial_p4_aggregated",
            "reason": "", "geometry_mode": geometry_mode,
            "point_aggregate_count": len(result.point_aggregates),
            "edge_aggregate_count": len(result.edge_aggregates),
            "object_aggregate_count": len(result.object_aggregates),
            "frame_aggregate_count": len(result.frame_aggregates),
            "clip_aggregate_count": 1,
            "temporal_interval_count": len(result.intervals),
            "classification_output": False,
        })
    columns = {
        "point": ("value", "valid", "quality", "coverage", "missing_reason", "point_id", "object_track_id", "video_id", "frame_index", "contributing_source_ids", "contributing_branch_names", "top_contributors", "coverage_dimensions", "applicable_count", "valid_count", "observation_missing_count", "not_applicable_count"),
        "edge": ("value", "valid", "quality", "coverage", "missing_reason", "edge_id", "object_track_id", "video_id", "frame_index", "contributing_source_ids", "contributing_branch_names", "top_contributors", "coverage_dimensions", "applicable_count", "valid_count", "observation_missing_count", "not_applicable_count"),
        "object": ("value", "valid", "quality", "coverage", "missing_reason", "object_track_id", "semantic_label", "video_id", "frame_index", "branch_scores", "valid_point_ratio", "valid_edge_ratio", "top_anomalous_point_ids", "top_anomalous_edge_ids", "localization_bbox", "localization_mask_reference", "contributing_source_ids", "contributing_branch_names", "coverage_dimensions"),
        "frame": ("value", "valid", "quality", "coverage", "missing_reason", "video_id", "frame_index", "object_scores", "active_branches", "branch_coverage", "top_object_ids", "top_point_ids", "top_edge_ids", "contributing_source_ids", "coverage_dimensions"),
    }
    for level, filename in (("point", "point_aggregates.csv"), ("edge", "edge_aggregates.csv"), ("object", "object_aggregates.csv"), ("frame", "frame_aggregates.csv")):
        serial = [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()} for row in all_rows[level]]
        _write_csv(output_dir / filename, serial, columns[level])
    _write_csv(output_dir / "temporal_intervals.csv", interval_rows, (
        "video_id", "start_frame", "end_frame", "score", "quality",
        "valid_frame_count", "missing_frame_count", "source_frame_indices", "metadata",
    ))
    _write_csv(output_dir / "branch_coverage.csv", branch_rows, (
        "video_id", "branch_name", "applicability", "valid_evidence_count", "coverage", "missing_reason",
    ))
    (output_dir / "clip_aggregates.json").write_text(json.dumps(clip_payload, indent=2), encoding="utf-8")
    report = {
        "experiment": "partial_p4_quality_aware_aggregation_smoke",
        "ready_video_count": len(ready_results),
        "not_ready_video_count": len(coverage_rows) - len(ready_results),
        "per_video": reports,
        "truth_labels_used": False,
        "weights_learned_from_six_videos": False,
        "authenticity_probability_output": False,
    }
    (output_dir / "smoke_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if ready_results:
        _plot_localization(ready_results[0], output_dir / "localization_diagnostics.png")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage_root", type=Path, default=PROJECT_ROOT / "outputs/real_3d_evidence_coverage_v2")
    parser.add_argument("--dynamic_root", type=Path, default=PROJECT_ROOT / "outputs/real_object_dynamic_3d_smoke")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/partial_p4_aggregation.yaml")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs/partial_p4_aggregation_smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_partial_p4_aggregation_smoke(
        coverage_root=args.coverage_root, dynamic_root=args.dynamic_root,
        config_path=args.config, output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
