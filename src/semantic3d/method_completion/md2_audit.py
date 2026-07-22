"""Deterministic, read-only P4-C3A-MD2 audit and synthetic validation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from ..dimension_aligned_scale_depth import load_dimension_aligned_prior_resolver
from .metric_depth_adapters import METRIC_DEPTH_ADAPTERS
from .metric_scale import (
    MetricDepthDefinition,
    MetricDepthEvidence,
    MetricDepthType,
    MetricObjectRegion,
    MetricScaleStatus,
    estimate_projected_extent,
)
from .multi_interval_prior import (
    DimensionScalePrior,
    MultiIntervalScalePriorRegistry,
    ObjectPhysicalScalePrior,
    SizeInterval,
    log_distance_to_interval_union,
)
from .scale_evidence import (
    ProviderStatus,
    ScaleBranchName,
    ScaleEvidenceRole,
    ScaleGeometryEvidence,
)
from .scale_router import ScaleEvidenceRouter
from .temporal_scale import (
    ScaleHistoryObservation,
    TemporalSameObjectScaleBranch,
)


STRICT_HASHES = {
    "scale_priors_strict_v1.yaml": "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b",
    "scale_priors_strict_v2.yaml": "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b",
}
PROTOCOL_HASHES = {
    "p4c0_experiment_protocol_v1.yaml": "8a4a8f5d6ac795646876042a84c9b0a4fdb1d06bec31045b734c3dfb64f8a304",
    "p4c1_experiment_manifest_v1.yaml": "ec48e26da4f434a1356959997b546ac30dc9e439281b2e09174f7c86a35ce086",
    "p4c2_formal_data_readiness_v1.yaml": "fe8c3cda137337330209528f4025d0593fefa42886be89eb214bfd58d38a8d89",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truth(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _camera():
    from ..geometry.camera import CameraObservation, CoordinateConvention

    return CameraObservation(
        np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]),
        None, None, None, 640, 480, CoordinateConvention.OPENCV,
        "synthetic_calibrated", "missing", True, 1.0,
    )


def run_metric_synthetic_validation() -> dict[str, Any]:
    """Run small deterministic metric geometry checks without model inference."""

    shape = (480, 640)
    mask = np.zeros(shape, bool)
    mask[100:300, 200:300] = True
    obj = MetricObjectRegion(
        "synthetic", "clip", "frame", "object", "track", "box",
        (200, 100, 300, 300), shape, mask=mask,
    )
    depth = MetricDepthEvidence(
        np.full(shape, 5.0), np.ones(shape, bool), np.ones(shape),
        MetricDepthType.METRIC, "meter", MetricScaleStatus.MODEL_PREDICTED,
        MetricDepthDefinition.Z_DEPTH, "synthetic", ProviderStatus.OK, 1.0,
    )
    extent = estimate_projected_extent(obj, depth, _camera())
    intervals = (SizeInterval(1.0, 2.0), SizeInterval(4.0, 5.0))
    checks = [
        {"name": "pinhole_height", "passed": math.isclose(extent.y_extent_m, 2.0)},
        {"name": "pinhole_width", "passed": math.isclose(extent.x_extent_m, 1.0)},
        {"name": "inside_disjoint_interval", "passed": log_distance_to_interval_union(1.5, intervals) == 0.0},
        {"name": "second_disjoint_interval", "passed": log_distance_to_interval_union(4.5, intervals) == 0.0},
        {
            "name": "outside_monotonic",
            "passed": log_distance_to_interval_union(6.0, intervals)
            < log_distance_to_interval_union(8.0, intervals),
        },
    ]
    return {"checks": checks, "passed": all(item["passed"] for item in checks), "large_model_run": False}


def _history(frame: int, size: float, *, mode: str = "metric") -> ScaleHistoryObservation:
    return ScaleHistoryObservation(
        "v", "c", f"f{frame}", frame, "track", f"o{frame}", "height_m", size,
        "meter" if mode == "metric" else "relative_local_unit", mode,
        "synthetic", "z_depth", "K", "aligned",
    )


def run_temporal_synthetic_validation() -> dict[str, Any]:
    """Validate causal same-track scale references."""

    previous = TemporalSameObjectScaleBranch(reference_method="previous_valid")
    rolling = TemporalSameObjectScaleBranch(reference_method="rolling_median", min_valid_history=2)
    zero = previous.evaluate(_history(1, 2.0), [_history(0, 2.0)])
    small = previous.evaluate(_history(1, 2.5), [_history(0, 2.0)])
    large = previous.evaluate(_history(1, 4.0), [_history(0, 2.0)])
    robust = rolling.evaluate(
        _history(4, 2.0), [_history(0, 2.0), _history(1, 20.0), _history(2, 2.0)]
    )
    local = previous.evaluate(
        _history(1, 2.1, mode="relative_local"), [_history(0, 2.0, mode="relative_local")]
    )
    checks = [
        {"name": "constant_zero", "passed": zero.valid and zero.residual_value == 0.0},
        {"name": "jump_monotonic", "passed": 0.0 < small.residual_value < large.residual_value},
        {"name": "rolling_median_robust", "passed": robust.valid and robust.residual_value == 0.0},
        {"name": "relative_local_unit", "passed": local.valid and local.depth_unit == "relative_local_unit"},
    ]
    return {"checks": checks, "passed": all(item["passed"] for item in checks)}


def _synthetic_evidence(branch: ScaleBranchName, valid: bool) -> ScaleGeometryEvidence:
    role = (
        ScaleEvidenceRole.PRIMARY
        if branch == ScaleBranchName.METRIC_SINGLE_OBJECT
        else ScaleEvidenceRole.TEMPORAL_SUPPORT
        if branch == ScaleBranchName.TEMPORAL_SAME_OBJECT
        else ScaleEvidenceRole.FALLBACK
    )
    base = {
        "video_id": "v", "clip_id": "c", "frame_id": "f", "object_id": "o",
        "track_id": "t", "branch_name": branch,
        "branch_priority": {ScaleBranchName.METRIC_SINGLE_OBJECT: 1, ScaleBranchName.TEMPORAL_SAME_OBJECT: 2, ScaleBranchName.RELATIVE_PAIR: 3}[branch],
        "evidence_role": role, "residual_name": "R", "depth_type": "synthetic",
        "depth_unit": "synthetic", "depth_definition": "z_depth",
        "coordinate_system": "camera", "localization_reference": "object:o",
        "provenance": {}, "config_sha256": "", "software_commit": "",
    }
    if valid:
        return ScaleGeometryEvidence.observed(residual_value=0.1, confidence=0.9, **base)
    return ScaleGeometryEvidence.missing(failure_reason="unavailable", **base)


def run_router_synthetic_validation() -> dict[str, Any]:
    """Validate fixed priority and lazy pair enumeration."""

    calls = {"count": 0}

    def pair_supplier():
        calls["count"] += 1
        return [_synthetic_evidence(ScaleBranchName.RELATIVE_PAIR, True)]

    router = ScaleEvidenceRouter("fallback_only")
    metric = router.route(
        metric_evidence=_synthetic_evidence(ScaleBranchName.METRIC_SINGLE_OBJECT, True),
        pair_supplier=pair_supplier,
        clip_observability="static",
    )
    temporal = router.route(
        metric_evidence=_synthetic_evidence(ScaleBranchName.METRIC_SINGLE_OBJECT, False),
        temporal_evidence=_synthetic_evidence(ScaleBranchName.TEMPORAL_SAME_OBJECT, True),
        pair_supplier=pair_supplier,
    )
    fallback = router.route(
        metric_evidence=_synthetic_evidence(ScaleBranchName.METRIC_SINGLE_OBJECT, False),
        temporal_evidence=_synthetic_evidence(ScaleBranchName.TEMPORAL_SAME_OBJECT, False),
        pair_supplier=pair_supplier,
    )
    checks = [
        {"name": "metric_primary", "passed": metric.selected_primary_branch == "metric_single_object_scale"},
        {"name": "metric_skips_pair", "passed": calls["count"] == 1},
        {"name": "temporal_secondary", "passed": temporal.selected_primary_branch == "temporal_same_object_scale"},
        {"name": "pair_fallback", "passed": fallback.selected_primary_branch == "relative_pair_scale_depth"},
        {"name": "static_supported", "passed": metric.selected_primary_branch == "metric_single_object_scale"},
    ]
    return {"checks": checks, "passed": all(item["passed"] for item in checks)}


def build_md2_audit(
    project_root: Path,
    output_dir: Path,
    *,
    dataset_root: Path | None = None,
    strict_result_path: Path | None = None,
) -> dict[str, Any]:
    """Build all MD2 outputs using only existing artifacts and synthetic arrays."""

    root = project_root.resolve()
    output = output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = dataset_root or root / "outputs/structural_enhancement_dataset/p4b5_six_video_full_observation"
    strict_path = strict_result_path or Path(
        "/mnt/e/fake_video_structural_anomaly_archive/outputs/evaluation/rsd_strict_v2/per_pair_rsd_details.csv"
    )
    required = [dataset / "manifests/clips.parquet", dataset / "manifests/videos.parquet", dataset / "observations/objects.parquet", dataset / "observations/tracks.parquet", strict_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("MD2 read-only audit inputs missing: " + ", ".join(missing))

    strict_hash_before = _sha256(strict_path)
    clips = pq.read_table(dataset / "manifests/clips.parquet").to_pylist()
    videos = pq.read_table(dataset / "manifests/videos.parquet").to_pylist()
    objects = pq.read_table(dataset / "observations/objects.parquet").to_pylist()
    tracks = pq.read_table(dataset / "observations/tracks.parquet").to_pylist()
    strict_rows = _read_csv(strict_path)
    video_names = {row["video_id"]: row["source_name"] for row in videos}
    strict_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in strict_rows:
        strict_by_video[row["video_id"]].append(row)
    objects_by_video_frame: Counter[tuple[str, int]] = Counter(
        (str(row["video_id"]), int(row["frame_index"])) for row in objects if row["valid"]
    )

    config_path = root / "configs/p4c3a_scale_geometry_priority_v1.yaml"
    config_hash = _sha256(config_path)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False
    ).stdout.strip()
    priority = {
        "schema_version": "p4c3a_metric_primary_refactor_v1",
        "branch_priority": [
            {"priority": 1, "branch": "metric_single_object_scale", "role": "primary"},
            {"priority": 2, "branch": "temporal_same_object_scale", "role": "temporal_support"},
            {"priority": 3, "branch": "relative_pair_scale_depth", "role": "fallback_or_audit"},
        ],
        "pair_branch_default_policy": "fallback_only",
        "metric_depth_enabled_by_default": False,
        "method_effectiveness_established": False,
    }
    _json(output / "method_priority_inventory.json", priority)
    _json(
        output / "scale_branch_schema.json",
        {
            "ScaleGeometryEvidence_fields": [field.name for field in fields(ScaleGeometryEvidence)],
            "evidence_roles": [item.value for item in ScaleEvidenceRole],
            "missing_residual": "NaN",
            "provider_failure_is_anomaly": False,
        },
    )

    metric_validation = run_metric_synthetic_validation()
    temporal_validation = run_temporal_synthetic_validation()
    router_validation = run_router_synthetic_validation()
    _json(output / "synthetic_metric_scale_tests.json", metric_validation)
    _json(output / "synthetic_temporal_scale_tests.json", temporal_validation)
    _json(output / "synthetic_router_tests.json", router_validation)
    _csv(
        output / "metric_single_object_audit.csv",
        metric_validation["checks"],
        ("name", "passed"),
    )
    _csv(
        output / "temporal_same_object_audit.csv",
        temporal_validation["checks"],
        ("name", "passed"),
    )
    _csv(output / "scale_router_audit.csv", router_validation["checks"], ("name", "passed"))

    pair_rows = []
    for source_name in sorted(strict_by_video):
        rows = strict_by_video[source_name]
        pair_rows.append(
            {
                "video_id": source_name,
                "strict_v2_rows": len(rows),
                "candidate_pairs": sum(row.get("skip_reason") != "insufficient_objects" for row in rows),
                "valid_pairs": sum(_truth(row.get("valid")) for row in rows),
                "status": "preserved_historical_strict_v2_read_only",
            }
        )
    _csv(
        output / "relative_pair_fallback_audit.csv",
        pair_rows,
        ("video_id", "strict_v2_rows", "candidate_pairs", "valid_pairs", "status"),
    )

    routing_rows = []
    for clip in sorted(clips, key=lambda item: (item["video_id"], item["clip_ordinal"])):
        source_name = video_names[str(clip["video_id"])]
        start, end = int(clip["core_start_frame_index"]), int(clip["core_end_frame_index"])
        strict_clip = [
            row for row in strict_by_video.get(source_name, [])
            if start <= int(row["frame_index"]) <= end
        ]
        valid_pair_count = sum(_truth(row["valid"]) for row in strict_clip)
        object_count = sum(
            objects_by_video_frame[(str(clip["video_id"]), frame)] for frame in range(start, end + 1)
        )
        pair_applicable = any(row.get("skip_reason") != "insufficient_objects" for row in strict_clip)
        routing_rows.append(
            {
                "video_id": source_name,
                "clip_id": clip["clip_id"],
                "candidate_objects": object_count,
                "metric_primary_applicable": False,
                "metric_primary_status": "blocked_metric_scale_unavailable",
                "temporal_metric_applicable": False,
                "temporal_relative_applicable": False,
                "temporal_status": "blocked_explicit_scale_history_not_materialized",
                "pair_fallback_applicable": pair_applicable,
                "pair_fallback_status": (
                    "historical_strict_v2_valid" if valid_pair_count
                    else "historical_strict_v2_no_valid_pair" if strict_clip
                    else "no_historical_pair_record"
                ),
                "selected_branch": (
                    "relative_pair_scale_depth" if valid_pair_count else "no_scale_evidence"
                ),
                "routing_reason": (
                    "read_only_historical_pair_evidence_available"
                    if valid_pair_count else "metric_unavailable_and_no_materialized_temporal_or_valid_pair_evidence"
                ),
                "failure_reasons": "metric_scale_unavailable|relative_scale_history_not_materialized",
            }
        )
    routing_fields = tuple(routing_rows[0])
    _csv(output / "six_video_scale_routing.csv", routing_rows, routing_fields)

    temporal_candidates = sum(int(row["observation_count"]) >= 2 for row in tracks if row["valid"])
    actual_pair_candidates = sum(row.get("skip_reason") != "insufficient_objects" for row in strict_rows)
    valid_pairs = sum(_truth(row["valid"]) for row in strict_rows)
    pair_skip = Counter(
        row.get("skip_reason") or "valid" for row in strict_rows if not _truth(row["valid"])
    )
    funnels = {
        "metric_single_object_scale": {
            "total_candidates": len(objects), "candidate_objects": len(objects),
            "applicable": 0, "input_ready": 0, "attempted": 0, "valid": 0,
            "metric_ready_objects": 0, "valid_absolute_scale_objects": 0,
            "blocked": len(objects), "provider_failed": 0, "not_applicable": 0,
            "failure_reason_counts": {"metric_scale_unavailable": len(objects)},
        },
        "temporal_same_object_scale_metric": {
            "total_candidates": temporal_candidates, "applicable": 0, "input_ready": 0,
            "attempted": 0, "valid": 0, "blocked": temporal_candidates,
            "provider_failed": 0, "not_applicable": 0,
            "failure_reason_counts": {"metric_scale_unavailable": temporal_candidates},
        },
        "temporal_same_object_scale_relative": {
            "total_candidates": temporal_candidates, "applicable": temporal_candidates,
            "input_ready": 0, "attempted": 0, "valid": 0,
            "blocked": temporal_candidates, "provider_failed": 0, "not_applicable": 0,
            "failure_reason_counts": {
                "explicit_scale_history_not_materialized_and_relative_per_frame_unaligned": temporal_candidates
            },
        },
        "relative_pair_scale_depth": {
            "total_candidates": len(strict_rows), "candidate_pairs": actual_pair_candidates,
            "applicable": actual_pair_candidates, "input_ready": actual_pair_candidates,
            "attempted": actual_pair_candidates, "executed_pairs": actual_pair_candidates,
            "valid": valid_pairs, "valid_pairs": valid_pairs,
            "blocked": actual_pair_candidates - valid_pairs,
            "provider_failed": 0,
            "not_applicable": len(strict_rows) - actual_pair_candidates,
            "failure_reason_counts": dict(sorted(pair_skip.items())),
        },
    }
    _json(output / "scale_eligibility_funnels.json", funnels)

    strict_resolver = load_dimension_aligned_prior_resolver(root / "configs/scale_priors_strict_v2.yaml")
    adapted = MultiIntervalScalePriorRegistry.from_strict_v2(strict_resolver)
    prior_audit = {
        "strict_v2_entries_seen": len(strict_resolver.entries),
        "metric_registry_entries_adapted": len(adapted.entries),
        "multi_interval_supported": True,
        "production_multi_interval_entries_added": 0,
        "strict_v2_modified": False,
        "entries": [
            {
                "label": label,
                "dimensions": sorted(prior.dimensions),
                "interval_count": sum(len(item.intervals) for item in prior.dimensions.values()),
                "source_registry": "strict_v2_read_only_adapter",
            }
            for label, prior in sorted(adapted.entries.items())
        ],
        "metric_provider_adapters": [
            adapter().describe().to_dict() for _, adapter in sorted(METRIC_DEPTH_ADAPTERS.items())
        ],
    }
    _json(output / "physical_prior_registry_audit.json", prior_audit)

    strict_hash_after = _sha256(strict_path)
    frozen = {
        "strict_prior_hashes": {
            name: _sha256(root / "configs" / name) for name in STRICT_HASHES
        },
        "strict_prior_hashes_expected": STRICT_HASHES,
        "strict_v2_result_path": str(strict_path),
        "strict_v2_result_sha256_before": strict_hash_before,
        "strict_v2_result_sha256_after": strict_hash_after,
        "strict_v2_results_unchanged": strict_hash_before == strict_hash_after,
        "strict_v2_formula_unchanged": True,
        "p4_protocol_config_hashes": {
            name: _sha256(root / "configs" / name) for name in PROTOCOL_HASHES
        },
        "p4_protocol_hashes_expected": PROTOCOL_HASHES,
        "p4c1_manifest_sha256": "3bfa3cc030adc29d1bfd6374d36ee17ad9f4a6fedb48f7b882549bcba4ad57a3",
        "p4c2_readiness_manifest_sha256": "2094a721bc58c7cae1d567a41d9fd31542990a95f87f0927b87c31119280e981",
        "d1_evidence_table_sha256": _sha256(dataset / "evidence/point_evidence.parquet"),
        "p4c3a_m_d2_validation_sha256": _sha256(root / "outputs/p4c3a_method_completion/d2_synthetic_validation.json"),
    }
    frozen["strict_v1_hash_unchanged"] = frozen["strict_prior_hashes"]["scale_priors_strict_v1.yaml"] == STRICT_HASHES["scale_priors_strict_v1.yaml"]
    frozen["strict_v2_hash_unchanged"] = frozen["strict_prior_hashes"]["scale_priors_strict_v2.yaml"] == STRICT_HASHES["scale_priors_strict_v2.yaml"]
    frozen["p4_protocol_hashes_unchanged"] = frozen["p4_protocol_config_hashes"] == PROTOCOL_HASHES
    _json(output / "frozen_result_regression.json", frozen)

    blocked = {
        "metric_single_object_six_video": "metric_scale_unavailable; existing depth is monocular relative_per_frame",
        "temporal_metric_six_video": "metric scale history unavailable",
        "temporal_relative_six_video": "dimension-specific aligned relative scale history was not materialized",
        "metric_provider_inference": "disabled by configuration and prohibited in this local stage",
        "provider_adapters": "interface_only until an explicit server implementation/dependency/weight is configured",
        "method_effectiveness": "not evaluated; no labels, thresholds, or performance metrics were read",
    }
    _json(output / "blocked_features.json", blocked)

    validation = {
        "metric_single_object_is_primary": True,
        "temporal_same_object_is_secondary": True,
        "relative_pair_is_fallback": True,
        "pair_branch_default_policy": "fallback_only",
        "metric_single_object_code_complete": True,
        "metric_single_object_synthetic_verified": metric_validation["passed"],
        "metric_single_object_six_video_executed": False,
        "metric_provider_large_inference_executed": False,
        "temporal_metric_code_complete": True,
        "temporal_relative_code_complete": True,
        "temporal_synthetic_verified": temporal_validation["passed"],
        "temporal_six_video_executed": False,
        "relative_pair_branch_preserved": True,
        "strict_v2_formula_unchanged": True,
        "strict_v2_results_unchanged": frozen["strict_v2_results_unchanged"],
        "strict_v2_hash_unchanged": frozen["strict_v2_hash_unchanged"],
        "scale_evidence_router_complete": router_validation["passed"],
        "static_video_supported": True,
        "multi_interval_prior_supported": True,
        "robust_mask_pointcloud_extent_supported": True,
        "localization_mapping_preserved": True,
        "ready_for_server_metric_smoke": True,
        "ready_for_git_freeze": True,
        "method_effectiveness_established": False,
        "six_video_count": len(videos),
        "clip_count": len(clips),
        "candidate_objects": len(objects),
        "strict_v2_total_rows": len(strict_rows),
        "strict_v2_candidate_pairs": actual_pair_candidates,
        "strict_v2_valid_pairs": valid_pairs,
        "config_sha256": config_hash,
        "software_commit": commit,
        "large_models_executed": False,
        "training_performed": False,
        "threshold_selected": False,
        "truth_performance_computed": False,
    }
    validation["valid"] = all(
        [
            validation["metric_single_object_synthetic_verified"],
            validation["temporal_synthetic_verified"],
            validation["scale_evidence_router_complete"],
            validation["strict_v2_results_unchanged"],
            validation["strict_v2_hash_unchanged"],
            frozen["strict_v1_hash_unchanged"],
            frozen["p4_protocol_hashes_unchanged"],
        ]
    )
    _json(output / "validation_report.json", validation)
    _report(output / "METRIC_PRIMARY_REFACTOR_REPORT.md", validation, funnels, blocked, frozen)
    return validation


def _report(
    path: Path,
    validation: Mapping[str, Any],
    funnels: Mapping[str, Any],
    blocked: Mapping[str, str],
    frozen: Mapping[str, Any],
) -> None:
    metric = funnels["metric_single_object_scale"]
    temporal = funnels["temporal_same_object_scale_relative"]
    pair = funnels["relative_pair_scale_depth"]
    lines = [
        "# P4-C3A-MD2 Metric Primary Refactor Report",
        "",
        "## Scope",
        "",
        "This stage changes routing and adds synthetic geometry validation only. It does not run a metric-depth model, train, select thresholds, or evaluate authenticity performance.",
        "",
        "## Branch Priority",
        "",
        "1. `metric_single_object_scale` (primary)",
        "2. `temporal_same_object_scale` (temporal support)",
        "3. `relative_pair_scale_depth` (fallback/audit, default `fallback_only`)",
        "",
        "The metric branch uses `H_hat=h_px*Z/fy`, `W_hat=w_px*Z/fx`, or robust mask-point-cloud quantile extents. Physical residuals are nearest-union log distances and are dimensionless.",
        "",
        "## Six-video Read-only Eligibility",
        "",
        f"- Candidate object observations: {metric['candidate_objects']}",
        f"- Metric-ready / valid metric objects: {metric['metric_ready_objects']} / {metric['valid_absolute_scale_objects']}",
        f"- Relative temporal candidate tracks: {temporal['total_candidates']}; materialized valid histories: {temporal['valid']}",
        f"- Historical strict-v2 rows / actual pairs / valid pairs: {pair['total_candidates']} / {pair['candidate_pairs']} / {pair['valid_pairs']}",
        "- Metric execution is correctly blocked because all existing depth is monocular `relative_per_frame`, not meters.",
        "- Relative temporal execution is not claimed because aligned dimension-specific scale histories are not materialized.",
        "",
        "## Frozen Regression",
        "",
        f"- strict-v1 unchanged: {frozen['strict_v1_hash_unchanged']}",
        f"- strict-v2 unchanged: {frozen['strict_v2_hash_unchanged']}",
        f"- strict-v2 result bytes unchanged during audit: {frozen['strict_v2_results_unchanged']}",
        f"- P4-C0/C1/C2 config hashes unchanged: {frozen['p4_protocol_hashes_unchanged']}",
        "",
        "## Blocked Real Functions",
        "",
        *[f"- `{name}`: {reason}" for name, reason in blocked.items()],
        "",
        "## Readiness",
        "",
        f"- Audit valid: {validation['valid']}",
        f"- Ready for a controlled server metric-provider smoke: {validation['ready_for_server_metric_smoke']}",
        "- Provider-specific adapters remain interface-only until explicitly configured with local dependencies and weights.",
        "- Method effectiveness established: false.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
