import numpy as np

from data.schemas import ClipObservation, FrameObservation, ObjectObservation, TrackObservation
from models.motion_residuals import compute_motion_residuals
from models.relation_residuals import compute_relation_residuals
from models.reprojection_residuals import compute_d2_projection_residual, compute_reprojection_residuals
from semantic3d.d3.events import OcclusionEventInputsV2, classify_occlusion_event, compute_reappearance_residual
from semantic3d.occlusion.visibility_state import ObjectVisibilityObservation, VisibilityState
from semantic3d.pose_d2.contracts import PairwisePoseObservation, PoseProviderStatus


def _clip():
    intrinsics = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    frames = []
    for index in range(3):
        first = ObjectObservation(
            "object_a", "track_a", "person", (40, 30, 60, 60), 1.0,
            metric_surface_xyz=np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]]),
        )
        second = ObjectObservation(
            "object_b", "track_b", "car", (60, 30, 90, 60), 1.0,
            metric_surface_xyz=np.array([[1.0 + 0.2 * index, 0.0, 2.0], [1.1 + 0.2 * index, 0.0, 2.0]]),
        )
        frames.append(
            FrameObservation(
                "video", "clip", index, float(index),
                np.zeros((80, 100, 3), dtype=np.uint8),
                [first, second],
                np.full((80, 100), 2.0),
                np.ones((80, 100), dtype=bool),
                intrinsics=intrinsics,
                relative_pose_from_previous=None if index == 0 else np.eye(4),
                confidence={"relative_pose": 1.0, "metric_depth": 1.0},
                occlusion_states={"track_a": "none"},
                reappearance_states={"track_a": "none"},
            )
        )
    frames[-1].visibility_observations["track_a"] = ObjectVisibilityObservation(
        object_track_id="track_a",
        frame_index=2,
        previous_state=VisibilityState.FULLY_VISIBLE,
        current_state=VisibilityState.UNCERTAIN,
        predicted_support_area=100.0,
        observed_visible_area=0.0,
        visible_ratio=0.0,
        occluded_ratio=0.0,
        in_frame_ratio=0.9,
        possible_occluder_ids=(),
        state_quality=0.9,
        valid=True,
    )
    frames[1].boundary_correspondences = np.array([[50.0, 40.0, 50.0, 40.0]])
    frames[-1].reappearance_observations.append(
        {
            "previous_object_track_id": "track_a",
            "candidate_object_track_id": "track_a",
            "frame_index": 2,
            "predicted_reappearance_region": (40.0, 30.0, 60.0, 60.0),
            "semantic_label_match": True,
            "appearance_similarity": 0.9,
            "structure_similarity": 0.9,
            "relative_depth_consistency": 0.9,
            "motion_direction_consistency": 0.9,
            "reid_source": "synthetic_formal_reid",
        }
    )
    track = TrackObservation(
        "point_track",
        "object_a",
        (0, 1, 2),
        np.array([[50.0, 40.0], [55.0, 40.0], [60.0, 40.0]]),
        points_3d=np.array([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0], [0.2, 0.0, 2.0]]),
    )
    return ClipObservation("video", "clip", frames, [track])


def test_d1_dynamic_and_track_residuals():
    rows = compute_motion_residuals(_clip())
    names = {row.name for row in rows if row.valid_mask}
    assert "dynamic_reprojection" in names
    assert "track_3d_continuity" in names
    assert "direction_consistency" in names


def test_d2_point_depth_and_boundary_entry():
    clip = _clip()
    rows = compute_reprojection_residuals(clip)
    names = {row.name for row in rows if row.valid_mask}
    assert {"point_reprojection", "boundary_reprojection", "depth_reprojection"} <= names
    pose = PairwisePoseObservation(
        frame_t=0, frame_t1=1, rotation=np.eye(3), translation=np.zeros(3),
        T_target_from_source=np.eye(4),
        pose_convention="X_target_camera=T_target_from_source@X_source_camera",
        camera_to_world_or_world_to_camera="camera_t_to_camera_t1",
        translation_scale_status="metric_model_depth", inlier_count=10, inlier_ratio=1.0,
        reprojection_error=0.0, static_background_ratio=1.0, dynamic_foreground_ratio=0.0,
        confidence=1.0, provider_status=PoseProviderStatus.ESTIMATED_VALID,
        failure_reason="", background_candidates=10, foreground_rejected=0,
        geometric_inliers=10, degeneracy_status="none", provider_name="synthetic", valid=True,
    )
    boundary = compute_d2_projection_residual(
        evidence_id="boundary", evidence_type="boundary", video_id="video", clip_id="clip",
        pose=pose, source_point_camera_m=np.array([0.0, 0.0, 2.0]),
        target_observed_uv=np.array([50.0, 40.0]),
        K_target=clip.frames[1].intrinsics, image_width=100, image_height=80,
        target_depth_m=clip.frames[1].metric_depth,
        target_depth_valid_mask=clip.frames[1].depth_valid_mask,
        object_id="object_a", track_id="track_a", point_id="boundary_0", point_confidence=1.0,
    )
    assert boundary.valid and boundary.boundary_reprojection_residual < 1e-8


def test_d3_relation_and_explicit_missing_events():
    rows = compute_relation_residuals(_clip())
    assert any(row.name == "relation" and row.valid_mask for row in rows)
    assert any(row.name == "occlusion" and row.valid_mask for row in rows)
    assert any(row.name == "reappearance" and row.valid_mask for row in rows)
    assert any(row.name == "occlusion" and not row.valid_mask for row in rows)
    assert any(row.name == "reappearance" and not row.valid_mask for row in rows)
    assert callable(classify_occlusion_event)
    assert callable(compute_reappearance_residual)
    assert OcclusionEventInputsV2 is not None
