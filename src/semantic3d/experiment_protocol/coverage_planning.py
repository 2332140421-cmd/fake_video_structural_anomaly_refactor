"""Coverage targets and gaps for future formal data acquisition."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from .schema import BranchEligibilityRecord, SplitAssignment, VideoInventoryRecord


DEFAULT_TARGETS: dict[str, tuple[str, int, int]] = {
    "static_3d_video": ("video", 30, 150),
    "d1_dynamic_clip": ("clip", 30, 150),
    "d2_rotation_clip": ("clip", 30, 150),
    "d3_full_se3_clip": ("clip", 30, 150),
    "person_structure_track": ("track", 30, 200),
    "ordinary_structure_track": ("track", 50, 300),
    "partial_occlusion_event": ("event", 30, 200),
    "full_occlusion_event": ("event", 30, 200),
    "reappearance_event": ("event", 20, 100),
    "temporal_localization_annotated_video": ("video", 50, 300),
    "spatial_localization_annotated_video": ("video", 50, 300),
}


def build_coverage_targets(
    inventory: Iterable[VideoInventoryRecord],
    eligibility: Iterable[BranchEligibilityRecord],
    acceptance_summary: dict[str, Any],
    *,
    targets: dict[str, tuple[str, int, int]] | None = None,
) -> dict[str, Any]:
    """Compare observation counts with planning targets, without fitting residuals."""

    inventory_rows = list(inventory)
    branch_rows = list(eligibility)
    current = {
        "static_3d_video": sum(row.entity_type == "video" and row.tier == "S" and row.applicable for row in branch_rows),
        "d1_dynamic_clip": sum(row.entity_type == "clip" and row.tier == "D1" and row.applicable for row in branch_rows),
        "d2_rotation_clip": sum(row.entity_type == "clip" and row.tier == "D2" and row.applicable for row in branch_rows),
        "d3_full_se3_clip": sum(row.entity_type == "clip" and row.tier == "D3" and row.applicable for row in branch_rows),
        "person_structure_track": int(acceptance_summary["ordinary_structure_funnel"]["person_structure_track_count"]),
        "ordinary_structure_track": int(acceptance_summary["ordinary_formal_structure_graph_count"]),
        "partial_occlusion_event": int(acceptance_summary["occlusion_depth_order"]["formal_occlusion_event_count"]),
        "full_occlusion_event": 0,
        "reappearance_event": 0,
        "temporal_localization_annotated_video": sum(row.temporal_annotation_available for row in inventory_rows),
        "spatial_localization_annotated_video": sum(row.spatial_annotation_available for row in inventory_rows),
    }
    target_spec = targets or DEFAULT_TARGETS
    rows = []
    for name, (unit, minimum, desired) in target_spec.items():
        available = int(current.get(name, 0))
        rows.append(
            {
                "target_name": name,
                "unit": unit,
                "minimum_engineering_target": minimum,
                "desired_experiment_target": desired,
                "currently_available": available,
                "minimum_gap": max(0, minimum - available),
                "desired_gap": max(0, desired - available),
            }
        )
    return {
        "planning_only": True,
        "targets_not_fitted_from_residuals": True,
        "target_count": len(rows),
        "targets": rows,
    }


def build_missingness_bias_report(
    inventory: Iterable[VideoInventoryRecord],
    assignments: Iterable[SplitAssignment],
    structural_dataset_root: str | Path,
    eligibility: Iterable[BranchEligibilityRecord],
    *,
    shortcut_difference_threshold: float = 0.20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Summarize provider coverage by split/class and flag possible shortcuts."""

    inventory_rows = list(inventory)
    split_by_video = {row.video_id: row.split for row in assignments}
    root = Path(structural_dataset_root)
    coverage_rows = pq.read_table(root / "reports/coverage_metrics.parquet").to_pylist()
    metric_by_video: dict[str, dict[str, float]] = defaultdict(dict)
    for row in coverage_rows:
        if row["scope_type"] == "video":
            metric_by_video[str(row["scope_id"])][str(row["metric_name"])] = float(row["ratio"])
    objects = pq.read_table(root / "observations/objects.parquet").to_pylist()
    masks = pq.read_table(root / "observations/masks.parquet").to_pylist()
    keypoints = pq.read_table(root / "observations/keypoints.parquet").to_pylist()
    readiness = pq.read_table(root / "observations/dynamic_readiness.parquet").to_pylist()
    object_count = Counter(str(row["video_id"]) for row in objects if row["valid"])
    valid_mask_count = Counter(
        str(row["video_id"]) for row in masks if row["valid"] and not row["bbox_fallback"]
    )
    person_count = Counter(str(row["video_id"]) for row in keypoints)
    keypoint_valid_count = Counter(str(row["video_id"]) for row in keypoints if row["valid"])
    geometry_modes: dict[str, Counter[str]] = defaultdict(Counter)
    failure_reasons: dict[str, Counter[str]] = defaultdict(Counter)
    for row in readiness:
        video_id = str(row["video_id"])
        geometry_modes[video_id][str(row["geometry_mode"])] += 1
        if row["missing_reason"]:
            failure_reasons[video_id][str(row["missing_reason"])] += 1
    eligibility_rows = [row for row in eligibility if row.entity_type == "clip"]
    applicable_branches = Counter(row.video_id for row in eligibility_rows if row.applicable)
    total_branches = Counter(row.video_id for row in eligibility_rows)

    per_video: dict[str, dict[str, Any]] = {}
    for row in inventory_rows:
        metrics = metric_by_video[row.video_id]
        dominant_mode = geometry_modes[row.video_id].most_common(1)
        per_video[row.video_id] = {
            "split": split_by_video[row.video_id],
            "binary_label": row.binary_label,
            "depth_coverage": metrics.get("frame_depth_coverage", 0.0),
            "frame_shared_3d_coverage": metrics.get("frame_shared_3d_coverage", 0.0),
            "sequence_aligned_coverage": metrics.get("sequence_depth_aligned_coverage", 0.0),
            "dynamic_readiness": metrics.get("dynamic_3d_ready_coverage", 0.0),
            "mask_coverage": (
                valid_mask_count[row.video_id] / object_count[row.video_id]
                if object_count[row.video_id]
                else None
            ),
            "keypoint_coverage": (
                keypoint_valid_count[row.video_id] / person_count[row.video_id]
                if person_count[row.video_id]
                else None
            ),
            "branch_applicability_ratio": (
                applicable_branches[row.video_id] / total_branches[row.video_id]
                if total_branches[row.video_id]
                else None
            ),
            "mask_applicable": object_count[row.video_id] > 0,
            "keypoint_applicable": person_count[row.video_id] > 0,
            "dominant_geometry_mode": dominant_mode[0][0] if dominant_mode else "unavailable",
            "provider_failure_reasons": dict(failure_reasons[row.video_id]),
        }
    grouped: dict[tuple[str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for values in per_video.values():
        grouped[(values["split"], values["binary_label"])].append(values)
    numeric_fields = (
        "depth_coverage",
        "frame_shared_3d_coverage",
        "sequence_aligned_coverage",
        "dynamic_readiness",
        "mask_coverage",
        "keypoint_coverage",
        "branch_applicability_ratio",
    )
    output = []
    for (split, label), rows in sorted(grouped.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        means = {}
        for field in numeric_fields:
            values = [float(row[field]) for row in rows if row[field] is not None]
            means[field] = sum(values) / len(values) if values else None
        output.append(
            {
                "split": split,
                "binary_label": label,
                "video_count": len(rows),
                **means,
                "mask_applicable_video_count": sum(bool(row["mask_applicable"]) for row in rows),
                "keypoint_applicable_video_count": sum(bool(row["keypoint_applicable"]) for row in rows),
                "geometry_mode_counts": dict(Counter(row["dominant_geometry_mode"] for row in rows)),
                "provider_failure_reasons": dict(
                    sum((Counter(row["provider_failure_reasons"]) for row in rows), Counter())
                ),
                "coverage_features_must_not_be_anomaly_by_default": True,
            }
        )
    risks = []
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output:
        by_split[str(row["split"])].append(row)
    for split, rows in by_split.items():
        by_label = {row["binary_label"]: row for row in rows}
        if 0 not in by_label or 1 not in by_label:
            continue
        for field in numeric_fields:
            if by_label[0][field] is None or by_label[1][field] is None:
                continue
            difference = abs(float(by_label[0][field]) - float(by_label[1][field]))
            if difference >= shortcut_difference_threshold:
                risks.append(
                    {
                        "split": split,
                        "metric": field,
                        "absolute_class_difference": difference,
                        "shortcut_risk": True,
                        "recommended_action": "stratify_or_balance_and_ablate_coverage_features",
                    }
                )
    summary = {
        "shortcut_risk": bool(risks),
        "risk_count": len(risks),
        "risks": risks,
        "small_sample_warning": len(inventory_rows) <= 6,
        "provider_failure_is_anomaly_evidence": False,
        "coverage_feature_ablation_required_if_used": True,
    }
    return output, summary
