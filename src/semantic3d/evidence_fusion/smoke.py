"""Offline P4-C3B-M6 fusion and localization smoke over persisted evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

from ..method_completion.observability import ClipObservability
from .adapters import load_persisted_unified_evidence
from .audits import (
    branch_availability_rows,
    coverage_by_group,
    missingness_only_features,
    provider_failure_balance,
)
from .contracts import (
    EvidenceBranchGroup,
    UnifiedEvidence,
    provider_status_is_failure,
)
from .fusion import branch_dropout_audit, fuse_unified_evidence
from .localization import (
    map_unified_evidence_spatially,
    rank_object_and_track_evidence,
)
from .routing import BranchRouteDecision, route_evidence_branches
from .temporal import build_frame_fusions, build_temporal_evidence_sequences


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    records = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in records for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(_json_safe(value), sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "git_unavailable"


def _identity_fields(item: UnifiedEvidence) -> Mapping[str, Any]:
    return {
        "object_id": item.object_id,
        "track_id": item.track_id,
        "frame_id": item.frame_id,
        "video_id": item.video_id,
        "clip_id": item.clip_id,
        "frame_index": item.frame_index,
        "spatial_reference": item.spatial_reference,
        "temporal_reference": item.temporal_reference,
        "provenance": {
            **dict(item.provenance),
            "route_original_evidence_id": item.evidence_id,
        },
    }


def _apply_route(
    item: UnifiedEvidence,
    decision: BranchRouteDecision,
) -> UnifiedEvidence:
    if decision.status == "enabled":
        return item
    return UnifiedEvidence.unavailable(
        evidence_id=f"{item.evidence_id}:route:{decision.status}",
        applicable=decision.applicable,
        provider_status=decision.status,
        failure_reason=decision.reason,
        branch_name=item.branch_name,
        branch_group=item.branch_group,
        **_identity_fields(item),
    )


def _placeholder(
    *,
    video_id: str,
    clip_id: str,
    group: EvidenceBranchGroup,
    decision: BranchRouteDecision,
) -> UnifiedEvidence:
    applicable = decision.applicable
    status = decision.status
    reason = decision.reason
    if status == "enabled":
        status = "blocked_by_input"
        reason = "branch_evidence_unavailable"
        applicable = True
    return UnifiedEvidence.unavailable(
        evidence_id=f"{clip_id}:{group.value}:availability",
        applicable=applicable,
        provider_status=status,
        failure_reason=reason,
        branch_name=f"{group.value}_availability",
        branch_group=group,
        video_id=video_id,
        clip_id=clip_id,
        frame_id="",
        frame_index=None,
        spatial_reference={"kind": "reference_only"},
        temporal_reference={"clip_id": clip_id},
        provenance={
            "synthetic_residual_value": False,
            "availability_marker_only": True,
            "authenticity_label_used": False,
        },
    )


def _route_all(
    evidences: Sequence[UnifiedEvidence],
    clip_routes: Sequence[Mapping[str, Any]],
) -> tuple[tuple[UnifiedEvidence, ...], tuple[Mapping[str, Any], ...]]:
    by_clip: dict[tuple[str, str], list[UnifiedEvidence]] = {}
    for item in evidences:
        by_clip.setdefault((item.video_id, item.clip_id), []).append(item)
    routed: list[UnifiedEvidence] = []
    rows: list[Mapping[str, Any]] = []
    configured_keys = set()
    for config in clip_routes:
        video_id, clip_id = str(config["video_id"]), str(config["clip_id"])
        configured_keys.add((video_id, clip_id))
        decisions = route_evidence_branches(
            ClipObservability(str(config["observability"])),
            pose_available=bool(config["pose_available"]),
            event_observed=bool(config["event_observed"]),
        )
        source_rows = by_clip.get((video_id, clip_id), [])
        for group, decision in decisions.items():
            matches = [item for item in source_rows if item.branch_group == group]
            if matches:
                routed.extend(_apply_route(item, decision) for item in matches)
            else:
                routed.append(
                    _placeholder(
                        video_id=video_id,
                        clip_id=clip_id,
                        group=group,
                        decision=decision,
                    )
                )
            rows.append(
                {
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "observability": str(config["observability"]),
                    "pose_available": bool(config["pose_available"]),
                    "event_observed": bool(config["event_observed"]),
                    "branch_group": group.value,
                    "route_status": decision.status,
                    "route_applicable": decision.applicable,
                    "route_input_expected": decision.input_expected,
                    "route_reason": decision.reason,
                    "persisted_input_count": len(matches),
                    "persisted_valid_count": sum(item.valid for item in matches),
                }
            )
    for key, rows_for_clip in by_clip.items():
        if key not in configured_keys:
            routed.extend(rows_for_clip)
    return tuple(routed), tuple(rows)


def _group_by_clip(
    evidences: Sequence[UnifiedEvidence],
) -> Mapping[tuple[str, str], tuple[UnifiedEvidence, ...]]:
    grouped: dict[tuple[str, str], list[UnifiedEvidence]] = {}
    for item in evidences:
        grouped.setdefault((item.video_id, item.clip_id), []).append(item)
    return {
        key: tuple(rows)
        for key, rows in sorted(grouped.items())
    }


def _save_spatial_maps(
    output_path: Path,
    maps: Mapping[str, Mapping[tuple[str, str, int], np.ndarray]],
) -> None:
    payload = {}
    for level, level_maps in maps.items():
        for (video_id, clip_id, frame_index), array in level_maps.items():
            safe_clip = clip_id.replace(":", "_").replace("/", "_")
            payload[f"{level}__{video_id}__{safe_clip}__{frame_index:06d}"] = array
    np.savez_compressed(output_path, **payload)


def run_evidence_fusion_smoke(
    project_root: str | Path,
    config_path: str | Path = "configs/p4c3b_evidence_fusion_v1.yaml",
) -> Mapping[str, Any]:
    """Run deterministic fusion over existing artifacts without model inference."""

    root = Path(project_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    persisted, source_audit = load_persisted_unified_evidence(root, config)
    routed, route_rows = _route_all(persisted, config["clip_routes"])
    branch_weights = {
        EvidenceBranchGroup(key): float(value)
        for key, value in config["branch_weights"].items()
    }
    clip_groups = _group_by_clip(routed)
    risk_rows = []
    contribution_rows = []
    dropout_rows = []
    missingness_rows = []
    for (video_id, clip_id), rows in clip_groups.items():
        result = fuse_unified_evidence(
            rows,
            branch_weights=branch_weights,
            top_k=int(config["fusion"]["top_k"]),
        )
        risk_rows.append(
            {
                "video_id": video_id,
                "clip_id": clip_id,
                "risk_score": result.risk_score,
                "evidence_confidence": result.evidence_confidence,
                "active_branch_count": result.active_branch_count,
                "available_weight_ratio": result.available_weight_ratio,
                "missing_reason_summary": result.missing_reason_summary,
                "valid": result.valid,
                "failure_reason": result.failure_reason,
                "authenticity_label_used": False,
                "formal_threshold_selected": False,
                "classification_output": False,
            }
        )
        contribution_rows.extend(
            {
                "video_id": video_id,
                "clip_id": clip_id,
                **asdict(item),
            }
            for item in result.branch_contributions
        )
        dropout_rows.extend(
            {
                "video_id": video_id,
                "clip_id": clip_id,
                **row,
            }
            for row in branch_dropout_audit(rows, branch_weights=branch_weights)
        )
        missingness_rows.append(
            {
                "video_id": video_id,
                "clip_id": clip_id,
                **missingness_only_features(rows),
            }
        )
    availability = branch_availability_rows(routed)
    frame_evidence = build_frame_fusions(routed, branch_weights=branch_weights)
    temporal_config = config["temporal_localization"]
    sequences = build_temporal_evidence_sequences(
        frame_evidence,
        smoothing_window=int(temporal_config["smoothing_window"]),
        diagnostic_threshold=temporal_config["diagnostic_threshold"],
        max_gap=int(temporal_config["max_gap"]),
        minimum_duration=int(temporal_config["minimum_duration"]),
    )
    map_shape = tuple(int(value) for value in config["localization"]["audit_map_shape"])
    spatial = map_unified_evidence_spatially(routed, image_shape=map_shape)
    rankings = rank_object_and_track_evidence(spatial)
    _save_spatial_maps(
        output / "spatial_evidence_maps.npz",
        {
            "object": spatial.object_evidence_maps,
            "boundary": spatial.boundary_evidence_maps,
            "point": spatial.point_evidence_maps,
            "track": spatial.track_evidence_maps,
            "frame": spatial.frame_spatial_evidence_maps,
        },
    )
    coverage_groups = {
        str(key): str(value)
        for key, value in config.get("coverage_groups", {}).items()
    }
    grouped_coverage = coverage_by_group(routed, coverage_groups)
    provider_balance = provider_failure_balance(routed, coverage_groups)
    schema_audit = {
        "schema_version": config["schema_version"],
        "required_fields": [
            "residual_value",
            "applicable",
            "valid",
            "confidence",
            "uncertainty",
            "provider_status",
            "failure_reason",
            "branch_name",
            "object_id",
            "track_id",
            "frame_id",
            "spatial_reference",
            "temporal_reference",
            "provenance",
        ],
        "branch_groups": [group.value for group in EvidenceBranchGroup],
        "evidence_count": len(routed),
        "valid_evidence_count": sum(item.valid for item in routed),
        "invalid_non_nan_count": sum(
            not item.valid and math.isfinite(item.residual_value) for item in routed
        ),
        "provider_failure_valid_count": sum(
            item.valid and provider_status_is_failure(item.provider_status)
            for item in routed
        ),
        "source_audit": source_audit,
        "config_sha256": _sha256(config_file),
    }
    missingness_audit = {
        "missingness_only_baseline_interface_complete": True,
        "missingness_only_baseline_trained": False,
        "branch_dropout_interface_complete": True,
        "real_fake_coverage_audit_interface_complete": True,
        "real_fake_coverage_audit_executed": False,
        "real_fake_coverage_audit_reason": (
            "Smoke uses unlabeled source groups; authenticity labels are not read."
        ),
        "unlabeled_source_group_coverage": grouped_coverage,
        "provider_failure_balance_audit": provider_balance,
        "per_clip_missingness_features": missingness_rows,
        "branch_dropout": dropout_rows,
        "missingness_used_in_risk": False,
        "performance_metric_computed": False,
    }
    localization_audit = {
        "point_map_count": len(spatial.point_evidence_maps),
        "boundary_map_count": len(spatial.boundary_evidence_maps),
        "object_map_count": len(spatial.object_evidence_maps),
        "track_map_count": len(spatial.track_evidence_maps),
        "frame_map_count": len(spatial.frame_spatial_evidence_maps),
        "rasterized_evidence_count": sum(
            bool(row["rasterized"]) for row in spatial.manifest_rows
        ),
        "reference_only_or_skipped_count": sum(
            not bool(row["rasterized"]) for row in spatial.manifest_rows
        ),
        "skipped_reason_counts": spatial.skipped_reason_counts,
        "missing_pixels_are_nan": True,
        "object_mask_fabricated": False,
        "object_track_ranking_count": len(rankings),
        "formal_threshold_selected": False,
    }
    _write_json(output / "evidence_schema_audit.json", schema_audit)
    _write_csv(
        output / "branch_availability_audit.csv",
        [*availability, *route_rows],
    )
    _write_csv(output / "risk_confidence_baseline.csv", risk_rows)
    _write_csv(output / "branch_contribution_audit.csv", contribution_rows)
    _write_json(
        output / "temporal_evidence_sequences.json",
        [asdict(item) for item in sequences],
    )
    _write_csv(output / "spatial_evidence_manifest.csv", spatial.manifest_rows)
    _write_csv(output / "object_track_rankings.csv", rankings)
    _write_json(output / "missingness_shortcut_audit.json", missingness_audit)
    _write_json(output / "localization_interface_audit.json", localization_audit)
    risk_by_clip = {
        (str(row["video_id"]), str(row["clip_id"])): bool(row["valid"])
        for row in risk_rows
    }
    static_supported = any(
        str(route["observability"]) in {"static", "low_motion"}
        and risk_by_clip.get(
            (str(route["video_id"]), str(route["clip_id"])), False
        )
        for route in config["clip_routes"]
    )
    statuses = {
        "unified_evidence_schema_complete": (
            schema_audit["invalid_non_nan_count"] == 0
            and schema_audit["provider_failure_valid_count"] == 0
        ),
        "missing_aware_fusion_complete": bool(risk_rows),
        "risk_and_confidence_separated": all(
            "risk_score" in row and "evidence_confidence" in row
            for row in risk_rows
        ),
        "static_video_detection_supported": static_supported,
        "pose_missing_fallback_supported": any(
            row["route_status"] == "blocked_by_input"
            and row["branch_group"]
            in {
                EvidenceBranchGroup.D2_POSE_REPROJECTION.value,
                EvidenceBranchGroup.D3_STRUCTURAL_RELATION.value,
            }
            and any(
                str(route["video_id"]) == str(row["video_id"])
                and str(route["clip_id"]) == str(row["clip_id"])
                and not bool(route["pose_available"])
                for route in config["clip_routes"]
            )
            for row in route_rows
        ),
        "missingness_shortcut_controls_complete": True,
        "temporal_localization_pipeline_complete": bool(sequences),
        "spatial_localization_pipeline_complete": (
            localization_audit["rasterized_evidence_count"] > 0
        ),
        "formal_threshold_selected": False,
        "formal_training_executed": False,
        "method_effectiveness_established": False,
    }
    validation = {
        **statuses,
        "persisted_input_evidence_count": len(persisted),
        "routed_evidence_count": len(routed),
        "clip_fusion_count": len(risk_rows),
        "valid_clip_fusion_count": sum(bool(row["valid"]) for row in risk_rows),
        "active_branch_counts": {
            row["clip_id"]: row["active_branch_count"] for row in risk_rows
        },
        "config_sha256": _sha256(config_file),
        "software_commit": _commit(root),
        "provider_inference_executed": False,
        "authenticity_labels_read": False,
    }
    _write_json(output / "validation_report.json", validation)
    report_lines = [
        "# P4-C3B-M6 Evidence Fusion Report",
        "",
        "M6 performs deterministic, missing-aware evidence aggregation and localization.",
        "It does not train a classifier, select a formal threshold, or evaluate authenticity.",
        "",
        "## Evidence",
        "",
        f"- Persisted input evidence: {len(persisted)}.",
        f"- Routed evidence: {len(routed)}.",
        f"- Clip fusion records: {len(risk_rows)}.",
        "- Risk uses valid residual values only; missingness affects confidence and coverage.",
        "- The fixed residual transform is monotonic and untrained, not calibrated.",
        "",
        "## Localization",
        "",
        f"- Rasterized evidence records: {localization_audit['rasterized_evidence_count']}.",
        "- References without persisted pixel/Mask support remain manifest-only.",
        "- Empty pixels remain NaN; no object Mask is fabricated.",
        "",
        "## Status",
        "",
        *[
            f"- `{name}`: `{str(value).lower()}`"
            for name, value in statuses.items()
        ],
        "",
        "## Limits",
        "",
        "- Residual branches are not statistically calibrated to a common distribution.",
        "- Current smoke covers only persisted M3/M4/M5 short clips.",
        "- Missingness-only and real/fake coverage interfaces are audit-only and untrained.",
        "- Output risk is a deterministic diagnostic score, not an authenticity probability.",
        "",
    ]
    (output / "EVIDENCE_FUSION_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return validation
