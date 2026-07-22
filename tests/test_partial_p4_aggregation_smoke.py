from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.run_partial_p4_aggregation_smoke import run_partial_p4_aggregation_smoke


def write_csv(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def test_partial_p4_runs_only_ready_video_without_truth_labels(tmp_path: Path) -> None:
    coverage = tmp_path / "coverage"
    dynamic = tmp_path / "dynamic" / "clip"
    output = tmp_path / "output"
    write_csv(coverage / "per_video_summary.csv", [
        "video_id", "ready_for_partial_p4", "primary_missing_reason", "formal_mask_valid_ratio",
    ], [["ready", True, "", 1.0], ["blocked", False, "no_structure", 0.5]])
    write_csv(coverage / "mask_coverage.csv", [
        "video_id", "frame_index", "object_track_id", "valid", "mask_bbox", "class_name", "visible_mask_path",
    ], [["ready", 2, "o", True, "[1, 1, 5, 5]", "person", "mask.npy"]])
    dynamic.mkdir(parents=True)
    (dynamic / "smoke_report.json").write_text(json.dumps({
        "video_id": "ready", "clip_id": "clip", "geometry_mode": "static_camera_3d",
    }))
    write_csv(dynamic / "direction_residuals.csv", [
        "point_id", "object_track_id", "current_frame_index", "own_history", "own_history_valid", "object_median", "object_median_valid",
    ], [["p", "o", 2, 1.2, True, 0.2, True]])
    write_csv(dynamic / "relative_velocity.csv", [
        "point_id", "object_track_id", "current_frame_index", "speed_change", "speed_change_valid", "point_vs_object_median_speed", "point_vs_object_median_speed_valid",
    ], [["p", "o", 2, 0.3, True, 0.1, True]])
    write_csv(dynamic / "dynamic_reprojection_residuals.csv", [
        "point_id", "object_track_id", "target_frame_index", "formal_residual", "formal_residual_valid",
    ], [["p", "o", 2, 0.01, True]])
    config = tmp_path / "config.yaml"
    config.write_text("""
aggregation: {method: hybrid_median_top_k, top_k: 2, trim_fraction: 0.1, quality_floor: 0.25}
temporal_localization: {high_evidence_threshold: 0.5, moving_median_window: 1, max_gap: 1, minimum_duration: 1}
""")
    report = run_partial_p4_aggregation_smoke(
        coverage_root=coverage, dynamic_root=tmp_path / "dynamic",
        config_path=config, output_dir=output,
    )
    assert report["ready_video_count"] == 1
    assert report["not_ready_video_count"] == 1
    assert report["truth_labels_used"] is False
    assert {row["status"] for row in report["per_video"]} == {"partial_p4_aggregated", "not_ready"}
    for name in ("evidence_registry.json", "point_aggregates.csv", "object_aggregates.csv", "frame_aggregates.csv", "clip_aggregates.json", "branch_coverage.csv", "localization_diagnostics.png"):
        assert (output / name).exists()
