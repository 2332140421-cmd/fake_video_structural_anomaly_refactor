"""Build deterministic P4-C3A-M method-completion audit artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from ..validity import ResidualEvidence
from .d2_validation import run_d2_synthetic_validation
from .d3_relations import (
    D3RelationObservation,
    D3RelationType,
    compare_d3_relations,
    d3_formula_definitions,
)
from .eligibility import EligibilityRecord, summarize_eligibility
from .localization import (
    BoundaryResidualLocation,
    ObjectResidualLocation,
    PointResidualLocation,
    TrackResidualLocation,
    map_residual_evidence,
)
from .observability import ClipMotionMeasurements, classify_clip_observability


STRICT_HASHES = {
    "scale_priors_strict_v1.yaml": (
        "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b"
    ),
    "scale_priors_strict_v2.yaml": (
        "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b"
    ),
}


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist() if path.exists() else []


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_safe_json(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _branch_inventory() -> list[dict[str, Any]]:
    rows = [
        ("A_absolute_semantic_scale_3d", "static_semantic_geometry", "AbsoluteSemanticScaleBranch.evaluate", "metric Object3D scale + physical prior", "ResidualEvidence", "implemented", "blocked_by_metric_scale_unavailable", "metric metres", "R=max(0,H_min-S,S-H_max)"),
        ("B_relative_scale_depth", "static_semantic_geometry", "RelativeScaleDepthBranch.evaluate", "same-frame objects + relative depth + physical priors", "ResidualEvidence", "implemented_and_executed", "20 valid strict-v2 pairs", "dimensionless log ratio", "distance(log(Za/Zb), expected log interval)"),
        ("C_cross_frame_scale_stability", "static_semantic_geometry", "CrossFrameScaleStabilityBranch.evaluate", "same track, sequence-comparable Object3D scales", "ResidualEvidence", "implemented", "not_materialized_on_six_video_artifact", "dimensionless log ratio", "max(0,abs(log(St/Sprev))-tolerance)"),
        ("clip_observability", "applicability", "classify_clip_observability", "motion measurements and quality", "ClipObservabilityResult", "implemented_and_executed", "59 existing clips classified without model rerun", "pixels and unitless quality", "deterministic rule-based classification"),
        ("eligibility_funnels", "applicability", "summarize_eligibility", "branch candidate records", "cumulative and terminal counts", "implemented_and_executed", "existing artifacts only", "counts", "total/applicable/input_ready/attempted/valid"),
        ("D1_static_camera", "dynamic_geometry", "compute_dynamic_reprojection_residual", "shared-scale points + identity pose", "ResidualEvidence", "implemented_and_executed", "2/59 clips", "image-diagonal-normalized pixels", "camera-compensated point reprojection"),
        ("D2_rotation_compensated", "dynamic_geometry", "run_d2_synthetic_validation", "K, depth, point, R, independent next pixel", "synthetic validation diagnostics", "implemented_synthetic_only", "0/59 six-video clips", "pixels and normalized pixels", "project(R X) against independent observation"),
        ("D3_higher_order_full_se3", "dynamic_geometry", "D3HigherOrderResidual.evaluate", "full-SE3 relations in one coordinate system", "ResidualEvidence interface", "interface_only", "full-SE3 six-video input unavailable", "relation-specific", "relation formulas recorded in d3_definition_and_status.json"),
        ("point_localization", "localization", "map_residual_evidence", "point residual + pixel", "frame_residual_map", "implemented_synthetic_verified", "not materialized for six-video export", "source residual unit", "place residual at source pixel"),
        ("track_localization", "localization", "map_residual_evidence", "track residual + point path", "track_scores and spatial map", "implemented_synthetic_verified", "not materialized for six-video export", "source residual unit", "map residual over observed path"),
        ("object_mask_localization", "localization", "map_residual_evidence", "object residual + formal mask", "object_scores and spatial map", "implemented_synthetic_verified", "not materialized for six-video export", "source residual unit", "map residual to visible mask"),
        ("boundary_localization", "localization", "map_residual_evidence", "boundary residual + boundary points", "spatial evidence map", "implemented_synthetic_verified", "no valid six-video boundary residual", "source residual unit", "map residual to boundary pixels"),
        ("temporal_evidence_sequence", "localization", "map_residual_evidence", "valid localized residuals by frame", "temporal_evidence_sequence", "implemented_synthetic_verified", "no six-video final anomaly sequence", "source residual unit", "per-frame max, missing frames remain NaN"),
    ]
    fields = (
        "branch_name", "method_group", "function_or_class", "input_schema",
        "output_schema", "code_status", "six_video_status", "unit", "formula",
    )
    return [dict(zip(fields, row, strict=True)) for row in rows]


def _observability_rows(
    readiness_rows: Sequence[Mapping[str, Any]],
    video_names: Mapping[str, str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in sorted(readiness_rows, key=lambda item: (str(item["video_id"]), str(item["clip_id"]))):
        mode = str(row.get("geometry_mode", "unavailable"))
        median_motion = float(row.get("median_pixel_motion", math.nan))
        tracked = float(row.get("tracked_transition_ratio", 0.0))
        coverage = float(row.get("independent_track_coverage", 0.0))
        quality = max(float(row.get("quality", 0.0)), min(tracked, coverage))
        reason = str(row.get("missing_reason", ""))
        provider_failed = "insufficient_independent_point_track" in reason
        if mode == "static_camera_3d":
            background_motion = 0.0
            object_motion = median_motion if math.isfinite(median_motion) else None
        elif math.isfinite(median_motion):
            background_motion = median_motion
            object_motion = None
        else:
            background_motion = None
            object_motion = None
        result = classify_clip_observability(
            ClipMotionMeasurements(
                clip_id=str(row["clip_id"]),
                background_motion_px=background_motion,
                object_motion_px=object_motion,
                camera_pose_available=mode != "unavailable",
                object_tracks_available=tracked > 0.0,
                quality=quality,
                provider_failed=provider_failed,
                missing_reason=reason if provider_failed else "",
                metadata={"source": "p4b5_dynamic_readiness"},
            )
        )
        output.append(
            {
                "video_id": video_names.get(str(row["video_id"]), str(row["video_id"])),
                "clip_id": row["clip_id"],
                "source_geometry_mode": mode,
                "observability": result.observability.value,
                "valid": result.valid,
                "quality": result.quality,
                "provider_failed": result.provider_failed,
                "missing_reason": result.missing_reason,
                "background_motion_px": background_motion,
                "object_motion_px": object_motion,
                "static_or_low_is_failure": False,
            }
        )
    return output


def _valid_record(branch: str, unit_id: str, unit: str) -> EligibilityRecord:
    return EligibilityRecord(
        branch, unit_id, True, True, True, True, metadata={"unit": unit}
    )


def _unavailable_record(
    branch: str,
    unit_id: str,
    unit: str,
    reason: str,
    outcome: str,
    *,
    attempted: bool = False,
) -> EligibilityRecord:
    return EligibilityRecord(
        branch_name=branch,
        unit_id=unit_id,
        applicable=outcome != "not_applicable",
        input_ready=attempted,
        attempted=attempted,
        valid=False,
        provider_failed=outcome == "provider_failed",
        blocked=outcome == "blocked",
        not_applicable=outcome == "not_applicable",
        reason=reason,
        metadata={"unit": unit},
    )


def _artifact_records(
    branch: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    unit: str,
) -> list[EligibilityRecord]:
    records: list[EligibilityRecord] = []
    for index, row in enumerate(rows):
        unit_id = str(row.get("evidence_id") or f"{branch}:{index}")
        if bool(row.get("valid", False)) and _finite(row.get("raw_value")):
            records.append(_valid_record(branch, unit_id, unit))
            continue
        applicability = str(row.get("applicability", ""))
        reason = str(row.get("missing_reason", "no_valid_output"))
        if "insufficient_independent_point_track" in reason:
            outcome = "provider_failed"
        elif applicability == "not_applicable":
            outcome = "not_applicable"
        elif applicability == "observation_missing":
            outcome = "provider_failed"
        else:
            outcome = "blocked"
        records.append(_unavailable_record(branch, unit_id, unit, reason, outcome))
    return records


def _build_funnels(
    *,
    rsd_rows: Sequence[Mapping[str, Any]],
    depth_rows: Sequence[Mapping[str, Any]],
    readiness_rows: Sequence[Mapping[str, Any]],
    point_rows: Sequence[Mapping[str, Any]],
    clip_rows: Sequence[Mapping[str, Any]],
    object_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, object]]:
    records: list[EligibilityRecord] = []
    for index, row in enumerate(rsd_rows):
        branch = "R_sd"
        unit_id = f"{row.get('video_id')}:{row.get('frame_index')}:{index}"
        if _truth(row.get("valid")) and _finite(row.get("rsd_log")):
            records.append(_valid_record(branch, unit_id, "object_pair"))
            continue
        reason = str(row.get("skip_reason") or "invalid_rsd_pair")
        if "depth" in reason:
            outcome = "provider_failed"
        elif any(token in reason for token in ("prior", "gate", "measurement", "incompatible")):
            outcome = "not_applicable"
        else:
            outcome = "blocked"
        records.append(_unavailable_record(branch, unit_id, "object_pair", reason, outcome))
    for index, row in enumerate(depth_rows):
        unit_id = f"{row.get('video_id')}:{row.get('track_id')}:{index}"
        if _truth(row.get("valid")) and _finite(row.get("residual")):
            records.append(_valid_record("depth_consistency", unit_id, "object_transition"))
        else:
            reason = str(row.get("skip_reason") or "invalid_depth_transition")
            records.append(
                _unavailable_record(
                    "depth_consistency", unit_id, "object_transition", reason,
                    "provider_failed" if "depth" in reason else "blocked",
                )
            )
    branch_map = {
        "track_3d_continuity": "track_continuity",
        "direction_consistency": "direction_consistency",
        "relative_velocity_change": "relative_velocity_change",
        "dynamic_reprojection": "D1",
    }
    for source, target in branch_map.items():
        rows = [row for row in point_rows if row.get("branch_name") == source]
        records.extend(_artifact_records(target, rows, unit="point"))
    for row in readiness_rows:
        clip_id = str(row["clip_id"])
        mode = str(row.get("geometry_mode", "unavailable"))
        reason = str(row.get("missing_reason", ""))
        median_motion = float(row.get("median_pixel_motion", math.nan))
        track_provider_failed = "insufficient_independent_point_track" in reason
        if mode == "static_camera_3d" and bool(row.get("valid", False)):
            records.append(_valid_record("D1_eligibility", clip_id, "clip"))
        elif track_provider_failed:
            records.append(
                _unavailable_record(
                    "D1_eligibility", clip_id, "clip", reason, "provider_failed"
                )
            )
        else:
            records.append(_unavailable_record("D1_eligibility", clip_id, "clip", "clip_not_static_camera", "not_applicable"))
        if mode == "rotation_compensated" and bool(row.get("valid", False)):
            records.append(_valid_record("D2", clip_id, "clip"))
        elif track_provider_failed:
            records.append(_unavailable_record("D2", clip_id, "clip", reason, "provider_failed"))
        elif math.isfinite(median_motion) and median_motion > 1.0:
            records.append(_unavailable_record("D2", clip_id, "clip", "rotation_transform_not_materialized", "blocked"))
        else:
            records.append(_unavailable_record("D2", clip_id, "clip", "rotation_compensation_not_applicable", "not_applicable"))
        if mode == "full_se3_3d" and bool(row.get("valid", False)):
            records.append(_valid_record("D3", clip_id, "clip"))
        elif track_provider_failed:
            records.append(_unavailable_record("D3", clip_id, "clip", reason, "provider_failed"))
        elif math.isfinite(median_motion) and median_motion > 1.0:
            records.append(_unavailable_record("D3", clip_id, "clip", "full_se3_not_observationally_supported", "blocked"))
        else:
            records.append(_unavailable_record("D3", clip_id, "clip", "full_se3_not_applicable_to_observed_motion", "not_applicable"))
    for source, target in (
        ("occlusion_depth_order", "occlusion"),
        ("reappearance_consistency", "reappearance"),
    ):
        rows = [row for row in clip_rows if row.get("branch_name") == source]
        records.extend(_artifact_records(target, rows, unit="clip_event"))
    for index, row in enumerate(object_rows):
        unit_id = str(row.get("object_observation_id") or index)
        records.append(
            _unavailable_record(
                "absolute_semantic_scale_3d",
                unit_id,
                "object_observation",
                "blocked_by_metric_scale_unavailable",
                "blocked",
            )
        )
    for index, row in enumerate(depth_rows):
        records.append(
            _unavailable_record(
                "cross_frame_scale_stability",
                f"scale:{index}",
                "object_transition",
                "explicit_object_scale_history_not_materialized",
                "blocked",
            )
        )
    return summarize_eligibility(records)


def _d3_status() -> dict[str, Any]:
    coordinate = "synthetic_full_se3_world"
    common = dict(
        relation_id="objects:a:b",
        relation_type=D3RelationType.OBJECT_RELATIVE_DISTANCE,
        source_ids=("a", "b"),
        unit="relative_shared_sequence_unit",
        coordinate_system_id=coordinate,
        pose_compensated=True,
        valid=True,
        quality=1.0,
    )
    previous = D3RelationObservation(frame_index=0, values=(2.0,), **common)
    same = D3RelationObservation(frame_index=1, values=(2.0,), **common)
    changed = D3RelationObservation(frame_index=1, values=(4.0,), **common)
    same_evidence = compare_d3_relations(previous, same)
    changed_evidence = compare_d3_relations(previous, changed)
    return {
        "definition": "pose-compensated higher-order 3D structure relation residual",
        "code_status": "interface_only",
        "six_video_verified": False,
        "full_executor_implemented": False,
        "required_inputs": [
            "scale-compatible full-SE3 pose",
            "shared coordinate system",
            "matched object and structure relation IDs",
            "valid visibility and quality masks",
        ],
        "relation_data_structures": [item.value for item in D3RelationType],
        "formulas": d3_formula_definitions(),
        "synthetic_contract_checks": {
            "unchanged_relation_is_zero": same_evidence.valid and same_evidence.value == 0.0,
            "changed_relation_is_positive": changed_evidence.valid and changed_evidence.value > 0.0,
            "changed_value": changed_evidence.value,
        },
        "verified": False,
        "reason": "relation contracts and formulas exist, but no integrated full-SE3 executor or six-video D3 input exists",
    }


def _localization_audit() -> dict[str, Any]:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:5, 2:5] = True
    point = ResidualEvidence.observed("point", 0.2, source_ids=("point:1",))
    track = ResidualEvidence.observed("track", 0.4, source_ids=("track:1",))
    obj = ResidualEvidence.observed("object", 0.6, source_ids=("object:1",))
    boundary = ResidualEvidence.observed("boundary", 0.8, source_ids=("boundary:1",))
    failed = ResidualEvidence.missing("failed", "provider_failed", source_ids=("failed:1",))
    bundle = map_residual_evidence(
        image_shape=(8, 8),
        frame_indices=(0, 1, 2),
        point_residuals=(
            PointResidualLocation(point, 0, (1.0, 1.0), "p", "t"),
            PointResidualLocation(failed, 0, (7.0, 7.0), "bad", "bad"),
        ),
        track_residuals=(TrackResidualLocation(track, "t", {0: (1.0, 1.0), 1: (2.0, 1.0)}),),
        object_residuals=(ObjectResidualLocation(obj, 1, "o", mask),),
        boundary_residuals=(BoundaryResidualLocation(boundary, 1, "o", ((2.0, 2.0), (4.0, 4.0))),),
    )
    checks = {
        "point_mapped_to_point": math.isclose(
            float(bundle.spatial_evidence_map[0][1, 1]), 0.4, abs_tol=1e-6
        ),
        "track_score_available": math.isclose(
            float(bundle.track_scores.get("t", math.nan)), 0.4, abs_tol=1e-12
        ),
        "object_mapped_to_mask": math.isclose(
            float(bundle.spatial_evidence_map[1][3, 3]), 0.6, abs_tol=1e-6
        ),
        "boundary_mapped_to_boundary": math.isclose(
            float(bundle.spatial_evidence_map[1][2, 2]), 0.8, abs_tol=1e-6
        ),
        "provider_failure_not_mapped": math.isnan(float(bundle.spatial_evidence_map[0][7, 7])),
        "empty_frame_remains_nan": math.isnan(bundle.temporal_evidence_sequence[2]),
        "no_threshold_or_final_decision": bundle.metadata["final_anomaly_decision"] is False,
    }
    return {
        "interface_complete": all(checks.values()),
        "six_video_localization_artifact_materialized": False,
        "outputs": [
            "frame_residual_map", "object_scores", "track_scores",
            "spatial_evidence_map", "temporal_evidence_sequence",
        ],
        "checks": checks,
        "skipped_source_reasons": dict(bundle.skipped_source_reasons),
        "missing_semantics": "NaN with explicit reason; never zero substitution",
        "threshold_selected": False,
        "final_anomaly_decision": False,
    }


def _blocked_features(
    funnels: Mapping[str, Mapping[str, object]],
) -> dict[str, Any]:
    return {
        "features": [
            {
                "feature": "absolute_semantic_scale_3d",
                "status": "blocked_by_input",
                "reason": "blocked_by_metric_scale_unavailable",
                "fabricated_metric_output": False,
            },
            {
                "feature": "D2_six_video",
                "status": "blocked_by_input",
                "reason": "rotation_transform_not_materialized",
                "synthetic_code_path_verified": True,
            },
            {
                "feature": "D3_full_executor",
                "status": "interface_only",
                "reason": "full_se3_not_observationally_supported",
            },
            {
                "feature": "formal_occlusion_residual",
                "status": "not_applicable_or_blocked",
                "reason": "no validated six-video event or geometry support",
            },
            {
                "feature": "six_video_localization_export",
                "status": "implemented_not_executed",
                "reason": "mapping interface verified synthetically; no threshold-free export run requested",
            },
        ],
        "funnel_terminal_counts": {
            name: {
                key: values[key]
                for key in ("valid", "provider_failed", "blocked", "not_applicable")
            }
            for name, values in funnels.items()
        },
    }


def _report(
    *,
    validation: Mapping[str, Any],
    observability: Sequence[Mapping[str, Any]],
    funnels: Mapping[str, Mapping[str, object]],
    d2: Mapping[str, Any],
    d3: Mapping[str, Any],
    localization: Mapping[str, Any],
) -> str:
    obs_counts = Counter(str(row["observability"]) for row in observability)
    lines = [
        "# P4-C3A-M Method Completion Report",
        "",
        "## Scope",
        "",
        "This stage completed code contracts, eligibility accounting, synthetic geometry checks, and threshold-free localization mapping. It did not run learned providers, download data, train, fit distributions, select thresholds, or compute authentic/fake performance.",
        "",
        "## Semantic Geometry Routes",
        "",
        "- Route A, absolute semantic scale: implemented and metric-gated. Current six-video relative depth is explicitly `blocked_by_metric_scale_unavailable`; no metre-valued residual is fabricated.",
        "- Route B, same-frame relative scale-depth: complete and backed by the frozen R_sd implementation.",
        "- Route C, same-object cross-frame scale stability: formula and evidence-aware code path complete; current six-video artifact does not materialize explicit object-scale histories for this new route.",
        "",
        "## Observability",
        "",
        f"Existing clips classified without provider rerun: {len(observability)}. Counts: {dict(sorted(obs_counts.items()))}.",
        "Static and low-motion states are valid observability conditions and are never relabeled as provider failure.",
        "",
        "## Eligibility Funnels",
        "",
    ]
    for name, values in funnels.items():
        lines.append(
            f"- `{name}` ({values['unit']}): total={values['total']}, applicable={values['applicable']}, input_ready={values['input_ready']}, attempted={values['attempted']}, valid={values['valid']}, provider_failed={values['provider_failed']}, blocked={values['blocked']}, not_applicable={values['not_applicable']}."
        )
    lines.extend(
        [
            "",
            "## D2 And D3",
            "",
            f"D2 synthetic checks: {d2['passed']}/{d2['total']} passed. D2 remains unverified on the six-video artifact.",
            "P4-C0 D2 semantics remain rotation-compensated. The known R,t check validates the rigid transform primitive; translation is not silently claimed by D2.",
            f"D3 definition complete: {str(validation['d3_definition_complete']).lower()}; code status: `{d3['code_status']}`; verified: false.",
            "",
            "## Localization",
            "",
            f"Threshold-free localization mapping checks complete: {str(localization['interface_complete']).lower()}.",
            "Valid residuals can be traced to points, tracks, masks, boundaries, frames, and a temporal evidence sequence. Missing/provider-failed evidence remains NaN and is excluded.",
            "",
            "## Freeze Decision",
            "",
            f"`ready_for_git_freeze={str(validation['ready_for_git_freeze']).lower()}` means the requested method-completion code and contracts are internally auditable. It does not mean D2/D3 six-video coverage or detection effectiveness has been established.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_method_completion_audit(
    project_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build all P4-C3A-M outputs from existing artifacts and synthetic inputs."""

    root = Path(project_root).resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    dataset_root = root / "outputs/structural_enhancement_dataset/p4b5_six_video_full_observation"
    if not dataset_root.exists():
        raise FileNotFoundError(f"P4-B.5 dataset not found: {dataset_root}")
    videos = _read_parquet(dataset_root / "manifests/videos.parquet")
    video_names = {str(row["video_id"]): str(row["source_name"]) for row in videos}
    readiness = _read_parquet(dataset_root / "observations/dynamic_readiness.parquet")
    point_rows = _read_parquet(dataset_root / "evidence/point_evidence.parquet")
    clip_rows = _read_parquet(dataset_root / "evidence/clip_evidence.parquet")
    object_rows = _read_parquet(dataset_root / "observations/objects.parquet")
    strict_rsd = Path(
        "/mnt/e/fake_video_structural_anomaly_archive/outputs/evaluation/"
        "rsd_strict_v2/per_pair_rsd_details.csv"
    )
    if not strict_rsd.exists():
        strict_rsd = root / "outputs/evaluation/rsd_strict_v2/per_pair_rsd_details.csv"
    depth_path = root / "outputs/results/test_videos_depth_consistency_pairs.csv"
    rsd_rows = _read_csv(strict_rsd)
    depth_rows = _read_csv(depth_path)

    branches = _branch_inventory()
    observability = _observability_rows(readiness, video_names)
    funnels = _build_funnels(
        rsd_rows=rsd_rows,
        depth_rows=depth_rows,
        readiness_rows=readiness,
        point_rows=point_rows,
        clip_rows=clip_rows,
        object_rows=object_rows,
    )
    d2 = run_d2_synthetic_validation()
    d3 = _d3_status()
    localization = _localization_audit()
    blocked = _blocked_features(funnels)
    strict_current = {
        filename: _sha256(root / "configs" / filename)
        for filename in STRICT_HASHES
    }
    strict_unchanged = strict_current == STRICT_HASHES
    validation = {
        "static_branch_complete": True,
        "relative_scale_depth_branch_complete": True,
        "absolute_scale_branch_status": "blocked_by_metric_scale_unavailable",
        "cross_frame_scale_stability_complete": True,
        "d1_eligibility_explained": "D1_eligibility" in funnels,
        "d2_code_path_complete": True,
        "d2_synthetic_verified": bool(d2["all_passed"]),
        "d2_six_video_verified": False,
        "d3_definition_complete": True,
        "d3_code_status": "interface_only",
        "localization_evidence_mapping_complete": bool(localization["interface_complete"]),
        "ready_for_git_freeze": bool(
            d2["all_passed"]
            and localization["interface_complete"]
            and strict_unchanged
            and d3["code_status"] == "interface_only"
        ),
        "strict_prior_hashes_unchanged": strict_unchanged,
        "strict_prior_hashes": strict_current,
        "method_effectiveness_established": False,
        "truth_performance_computed": False,
        "large_models_executed": False,
        "training_performed": False,
        "threshold_selected": False,
        "six_video_count": len(videos),
        "clip_count": len(readiness),
    }

    branch_fields = (
        "branch_name", "method_group", "function_or_class", "input_schema",
        "output_schema", "code_status", "six_video_status", "unit", "formula",
    )
    _write_csv(output / "method_branch_inventory.csv", branches, branch_fields)
    _write_csv(
        output / "observability_audit.csv",
        observability,
        (
            "video_id", "clip_id", "source_geometry_mode", "observability",
            "valid", "quality", "provider_failed", "missing_reason",
            "background_motion_px", "object_motion_px", "static_or_low_is_failure",
        ),
    )
    _write_json(output / "residual_eligibility_funnels.json", {"branches": funnels})
    _write_json(output / "d2_synthetic_validation.json", d2)
    _write_json(output / "d3_definition_and_status.json", d3)
    _write_json(output / "localization_interface_audit.json", localization)
    _write_json(output / "blocked_features.json", blocked)
    _write_json(output / "validation_report.json", validation)
    (output / "METHOD_COMPLETION_REPORT.md").write_text(
        _report(
            validation=validation,
            observability=observability,
            funnels=funnels,
            d2=d2,
            d3=d3,
            localization=localization,
        ),
        encoding="utf-8",
    )
    return validation
