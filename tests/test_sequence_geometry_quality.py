from __future__ import annotations

from semantic3d.sequence_geometry import SequenceGeometryQuality, SyntheticSequenceGeometryProvider

from synthetic_sequence_geometry import make_world_consistent_sequence


def test_sequence_geometry_quality_is_not_anomaly_score() -> None:
    frames, poses = make_world_consistent_sequence(
        [(0.0, 0.0, 0.0), (0.2, 0.0, 0.0)]
    )
    clip = SyntheticSequenceGeometryProvider(
        poses,
        background_track_ids=("bg_1", "bg_2"),
        reprojection_error_by_frame={0: 0.0, 1: 0.01},
    ).predict_clip(frames, [0, 1])
    quality = SequenceGeometryQuality.from_clip(clip)
    assert quality.valid
    assert quality.valid_pose_ratio == 1.0
    assert quality.valid_shared_3d_frame_ratio == 1.0
    assert quality.metadata["quality_is_probability"] is False
    assert quality.metadata["anomaly_score"] is False
    assert 0.0 <= quality.quality <= 1.0

