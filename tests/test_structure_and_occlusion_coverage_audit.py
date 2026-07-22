from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from scripts.audit_ordinary_object_structure_coverage import audit_ordinary_object_structure_coverage
from scripts.find_real_3d_evidence_clips import scan_occlusion_event_windows
from semantic3d.occlusion import adaptive_erosion_pixels


def write_csv(path: Path, columns: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)


def test_adaptive_erosion_depends_on_pixels_not_label() -> None:
    small = np.zeros((40, 40), dtype=bool)
    small[10:20, 10:20] = True
    large = np.zeros((200, 200), dtype=bool)
    large[20:180, 20:180] = True
    assert adaptive_erosion_pixels(small) < adaptive_erosion_pixels(large)


def test_structure_audit_does_not_drop_stable_track(tmp_path: Path) -> None:
    coverage = tmp_path / "coverage"
    mask_dir = coverage / "masks"
    frame_dir = coverage / "frames" / "v"
    mask_dir.mkdir(parents=True)
    frame_dir.mkdir(parents=True)
    mask_rows = []
    tracking_rows = []
    for frame in range(3):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.circle(image, (24 + frame, 24), 8, (255, 255, 255), -1)
        cv2.imwrite(str(frame_dir / f"frame_{frame:06d}.jpg"), image)
        mask = np.zeros((64, 64), dtype=bool)
        mask[16:33, 16 + frame:33 + frame] = True
        mask_path = mask_dir / f"m{frame}.npy"
        np.save(mask_path, mask)
        mask_rows.append(["v", frame, "t", "cup", True, str(mask_path), int(mask.sum()), 0.9, "real_instance_mask_provider"])
        tracking_rows.append(["v", "t", frame, True, 0.8])
    write_csv(coverage / "mask_coverage.csv", [
        "video_id", "frame_index", "object_track_id", "class_name", "valid", "visible_mask_path", "mask_area", "confidence", "source_provider",
    ], mask_rows)
    write_csv(coverage / "mask_tracking_quality.csv", [
        "video_id", "object_track_id", "frame_index", "valid", "track_quality",
    ], tracking_rows)
    write_csv(coverage / "structure_graph_coverage.csv", ["video_id", "object_track_id", "valid", "edge_count"], [])
    write_csv(coverage / "structure_residual_coverage.csv", ["video_id", "object_track_id", "valid", "valid_residual_count"], [])
    result = audit_ordinary_object_structure_coverage(
        coverage_root=coverage, dynamic_root=tmp_path / "dynamic",
        output_dir=tmp_path / "audit", max_point_audit_frames=3,
    )
    assert result["funnel"]["formal_mask_tracks"] == 1
    assert result["funnel"]["tracks_silently_dropped"] == 0
    assert len(result["rows"]) == 1
    assert result["rows"][0]["primary_failure_reason"]
    assert result["rows"][0]["truth_labels_used"] is False


def test_occlusion_windows_retain_rejection_reasons_and_forbid_scores() -> None:
    rows = [{
        "video_id": "v", "frame_index": frame,
        "multi_object_mask_overlap": frame in (2, 3),
        "sustained_mask_contact": frame in (2, 3),
        "visible_area_drop": frame == 3,
        "possible_occluder": frame in (2, 3),
        "depth_order_available": False,
        "disappearance": False, "reappearance": False,
        "out_of_frame": False, "detector_missing_control": False,
        "scene_cut": False,
    } for frame in range(8)]
    result = scan_occlusion_event_windows(rows, window_sizes=(8,))
    assert len(result) == 1
    assert result[0]["status"] == "candidate_rejected"
    assert "depth_order_unavailable" in result[0]["rejection_reasons"]
    assert result[0]["truth_labels_used"] is False
    rows[0]["anomaly_score"] = 9.0
    import pytest
    with pytest.raises(ValueError, match="truth labels or anomaly"):
        scan_occlusion_event_windows(rows, window_sizes=(8,))
