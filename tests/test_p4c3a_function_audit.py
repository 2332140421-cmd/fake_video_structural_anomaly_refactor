from __future__ import annotations

import csv
import json
from pathlib import Path

from semantic3d.function_audit import (
    ALLOWED_FEATURE_STATUSES,
    build_function_audit,
    run_synthetic_formula_checks,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "outputs/structural_enhancement_dataset/p4b5_six_video_full_observation"
ARCHIVE = Path("/mnt/e/fake_video_structural_anomaly_archive")


def test_synthetic_formula_checks_cover_geometry_and_missing_evidence() -> None:
    result = run_synthetic_formula_checks()
    assert result["all_passed"]
    assert result["total"] == 10
    names = {row["test_name"] for row in result["checks"]}
    assert "reversed_pose_direction_is_detectable" in names
    assert "provider_failure_is_masked_not_high_residual" in names
    assert "occluded_point_is_not_scored_as_correspondence" in names


def test_audit_builds_required_deterministic_artifacts(tmp_path: Path) -> None:
    if not DATASET.exists():
        raise AssertionError(f"Required local P4-B.5 artifact is missing: {DATASET}")
    first = tmp_path / "first"
    second = tmp_path / "second"
    report_a = build_function_audit(ROOT, first, dataset_root=DATASET, archive_root=ARCHIVE)
    report_b = build_function_audit(ROOT, second, dataset_root=DATASET, archive_root=ARCHIVE)
    expected = {
        "method_feature_inventory.csv",
        "per_video_coverage.csv",
        "per_clip_coverage.csv",
        "residual_artifact_inventory.csv",
        "residual_numeric_audit.json",
        "coordinate_and_unit_audit.json",
        "synthetic_formula_tests.json",
        "blocked_features.json",
        "FUNCTION_AUDIT_REPORT.md",
        "validation_report.json",
    }
    assert {path.name for path in first.iterdir()} == expected
    assert report_a == report_b
    for name in expected:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_inventory_uses_controlled_statuses_and_real_execution_counts(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    build_function_audit(ROOT, output, dataset_root=DATASET, archive_root=ARCHIVE)
    with (output / "method_feature_inventory.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["status"] for row in rows} <= ALLOWED_FEATURE_STATUSES
    reprojection = next(row for row in rows if row["feature_name"] == "point_reprojection_residual")
    assert int(reprojection["videos_succeeded"]) == 2
    d2 = next(row for row in rows if row["feature_name"] == "D2_rotation_compensated_geometry")
    d3 = next(row for row in rows if row["feature_name"] == "D3_full_SE3_geometry")
    assert d2["status"] == d3["status"] == "blocked_by_input"


def test_validation_never_claims_effectiveness_or_full_closure(tmp_path: Path) -> None:
    output = tmp_path / "audit"
    build_function_audit(ROOT, output, dataset_root=DATASET, archive_root=ARCHIVE)
    report = json.loads((output / "validation_report.json").read_text(encoding="utf-8"))
    assert report["method_effectiveness_established"] is False
    assert report["all_paper_functions_implemented"] is False
    assert report["all_paper_functions_executed_on_six_videos"] is False
    assert report["D1_verified"] is True
    assert report["D2_verified"] is False
    assert report["D3_verified"] is False
