from __future__ import annotations

import pytest

from scripts.find_real_3d_evidence_clips import find_evidence_clips


def test_no_event_returns_empty_candidates() -> None:
    rows = [{"video_id": "v", "frame_index": i, "geometry_mode": "static_camera_3d", "visibility_state": "no_occlusion_event"} for i in range(4)]
    assert find_evidence_clips(rows) == []


def test_finder_uses_observation_type_not_residual_or_truth_label() -> None:
    rows = [{"video_id": "v", "frame_index": i, "geometry_mode": "static_camera_3d", "keypoint_valid_ratio": 0.8, "object_track_ids": "p", "semantic_labels": "person", "observation_quality": 0.8} for i in range(3)]
    result = find_evidence_clips(rows)
    assert len(result) == 1 and result[0]["candidate_type"] == "person_structure"
    with pytest.raises(ValueError, match="truth labels or residual"):
        find_evidence_clips([{**rows[0], "label": 1}])
    with pytest.raises(ValueError, match="truth labels or residual"):
        find_evidence_clips([{**rows[0], "anomaly_score": 3.0}])


def test_full_video_candidate_types_use_events_only() -> None:
    rows = [{
        "video_id": "v", "frame_index": index,
        "geometry_mode": "full_video_mask_observation_only",
        "stable_mask_track_count": 2,
        "formal_mask_object_count": 2,
        "mean_tracking_quality": 0.8,
        "object_track_ids": "a;b", "semantic_labels": "cup;person",
        "mask_valid_ratio": 1.0, "observation_quality": 0.8,
        "visibility_state": "no_occlusion_event",
    } for index in range(3)]
    types = {row["candidate_type"] for row in find_evidence_clips(rows)}
    assert "stable_mask_tracking" in types
    assert "stable_multi_object" in types
