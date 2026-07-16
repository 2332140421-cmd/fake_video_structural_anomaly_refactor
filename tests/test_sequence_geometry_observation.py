from __future__ import annotations

import hashlib
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from semantic3d.sequence_geometry import (
    LegacyDepthPoseSequenceAdapter,
    MockSequenceGeometryProvider,
    RelativePoseObservation,
    SequenceScaleStatus,
    SyntheticSequenceGeometryProvider,
    UnifiedSequenceGeometryProvider,
    Shared3DClipObservation,
)

from synthetic_sequence_geometry import make_world_consistent_sequence


def test_pose_missing_is_not_identity() -> None:
    missing = RelativePoseObservation.missing(1, "pose_missing", source_frame_index=0)
    assert not missing.valid
    assert missing.T_world_from_camera is None
    assert missing.relative_pose_from_previous is None
    assert not missing.is_identity_relative_pose
    assert math.isnan(missing.reprojection_error)


def test_valid_static_identity_requires_and_preserves_evidence() -> None:
    pose = RelativePoseObservation.from_transforms(
        source_frame_index=0,
        target_frame_index=1,
        T_world_from_camera=np.eye(4),
        relative_pose_from_previous=np.eye(4),
        pose_source="synthetic_static_camera",
        pose_quality=1.0,
        background_support_count=40,
        background_inlier_ratio=1.0,
        reprojection_error=0.0,
        metadata={"identity_evidence": "static_background", "background_support_ratio": 1.0},
    )
    assert pose.valid
    assert pose.is_identity_relative_pose
    with pytest.raises(ValueError, match="requires background"):
        RelativePoseObservation.from_transforms(
            source_frame_index=0,
            target_frame_index=1,
            T_world_from_camera=np.eye(4),
            relative_pose_from_previous=np.eye(4),
            pose_source="unsupported_identity",
            pose_quality=1.0,
            background_support_count=0,
            background_inlier_ratio=0.0,
            reprojection_error=0.0,
        )


def test_moving_camera_static_world_center_is_invariant_after_compensation() -> None:
    frames, poses = make_world_consistent_sequence(
        [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
    )
    clip = SyntheticSequenceGeometryProvider(
        poses,
        background_track_ids=("bg_1", "bg_2"),
    ).predict_clip(frames, [0, 1, 2])
    centers = [frame.objects[0].center_3d_world.as_array() for frame in clip.frames]
    np.testing.assert_allclose(centers, np.asarray([[0.0, 0.0, 8.0]] * 3), atol=1e-9)
    for index in clip.frame_indices:
        twc = clip.T_world_from_camera_by_frame[index]
        tcw = clip.T_camera_from_world_by_frame[index]
        assert twc is not None and tcw is not None
        np.testing.assert_allclose(twc @ tcw, np.eye(4), atol=1e-9)
    assert clip.sequence_scale_status == SequenceScaleStatus.METRIC_SEQUENCE


def test_moving_camera_and_moving_object_preserve_true_world_motion() -> None:
    frames, poses = make_world_consistent_sequence(
        [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)],
        moving_object_offsets=[(0.0, 0.0, 0.0), (0.0, 0.4, 0.0)],
    )
    clip = SyntheticSequenceGeometryProvider(poses).predict_clip(frames, [0, 1])
    first = clip.frames[0].objects[0].center_3d_world.as_array()
    second = clip.frames[1].objects[0].center_3d_world.as_array()
    np.testing.assert_allclose(second - first, [0.0, 0.4, 0.0], atol=1e-9)


def test_relative_per_frame_without_alignment_disallows_dynamic_3d() -> None:
    frames, poses = make_world_consistent_sequence(
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], metric=False
    )
    clip = SyntheticSequenceGeometryProvider(poses).predict_clip(frames, [0, 1])
    assert clip.sequence_scale_status == SequenceScaleStatus.RELATIVE_PER_FRAME
    assert not clip.scale_allows_dynamic_3d
    assert not clip.allows_dynamic_3d


def test_mock_provider_never_fabricates_pose() -> None:
    frames, _ = make_world_consistent_sequence(
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)], metric=False
    )
    clip = MockSequenceGeometryProvider().predict_clip(frames, [0, 1])
    assert not clip.valid
    assert all(not pose.valid for pose in clip.relative_poses)
    assert all(value is None for value in clip.T_world_from_camera_by_frame.values())


def test_legacy_adapter_outputs_same_shared_clip_contract() -> None:
    frames, poses = make_world_consistent_sequence(
        [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)]
    )
    clip = LegacyDepthPoseSequenceAdapter(
        T_world_from_camera_by_frame=poses,
        pose_scale_compatible_with_depth=True,
    ).predict_clip(frames, [0, 1])
    assert isinstance(clip, Shared3DClipObservation)
    assert clip.provider_name == "legacy_depth_pose_sequence_adapter"


def test_unified_sequence_geometry_provider_is_reserved_abstract_interface() -> None:
    assert inspect.isabstract(UnifiedSequenceGeometryProvider)


def test_scene_cut_truncates_clip_and_no_pose_crosses_boundary() -> None:
    frames, poses = make_world_consistent_sequence(
        [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)]
    )
    clip = SyntheticSequenceGeometryProvider(
        poses, scene_cut_flags={2: True}
    ).predict_clip(frames, [0, 1, 2])
    assert clip.frame_indices == (0, 1)
    assert all(pose.target_frame_index != 2 for pose in clip.relative_poses)
    assert clip.metadata["terminated_at_scene_cut"] is True


def test_strict_prior_hashes_still_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "scale_priors_strict_v1.yaml": "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b",
        "scale_priors_strict_v2.yaml": "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b",
    }
    for filename, digest in expected.items():
        assert hashlib.sha256((root / "configs" / filename).read_bytes()).hexdigest() == digest
