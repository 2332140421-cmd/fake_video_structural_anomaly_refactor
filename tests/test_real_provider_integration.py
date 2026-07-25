import numpy as np

from data.schemas import ClipObservation, FrameObservation, ObjectObservation, VideoClip
from models.providers import LegacyPoseProviderAdapter, LegacyTrackProviderAdapter
from models.motion_residuals import compute_motion_residuals


def _frame(index, image, obj):
    shape = image.shape[:2]
    return FrameObservation(
        video_id="video",
        clip_id="clip",
        frame_index=index,
        timestamp=index / 30.0,
        image=image,
        objects=[] if obj is None else [obj],
        metric_depth=np.full(shape, 2.0),
        depth_valid_mask=np.ones(shape, dtype=bool),
        intrinsics=np.asarray([[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]]),
    )


def _object(index, mask):
    return ObjectObservation(
        object_id=f"detection_{index}",
        track_id=f"raw_{index}",
        category="person",
        bbox_xyxy=(4.0, 4.0, 60.0, 60.0),
        confidence=0.9,
        instance_mask=mask,
        mask_quality=0.9,
    )


def test_real_adapter_stabilizes_ids_and_keeps_detector_miss_unavailable():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[4:60, 4:60] = True
    frames = [
        _frame(0, image, _object(0, mask)),
        _frame(1, image, None),
        _frame(2, image, _object(2, mask)),
    ]
    clip = VideoClip("clip", "video", (0, 1, 2), (0.0, 1 / 30, 2 / 30), tuple(frame.image for frame in frames))
    LegacyTrackProviderAdapter(LegacyPoseProviderAdapter()).track(clip, frames)

    assert frames[0].objects[0].track_id == frames[2].objects[0].track_id
    assert frames[0].objects[0].track_id.startswith("clip:trk_")
    assert frames[2].objects[0].metadata["track_status"] == "REAPPEARED"
    missing = frames[1].visibility_observations[frames[0].objects[0].track_id]
    assert not missing.valid
    assert missing.current_state.value != "fully_occluded"
    assert frames[2].reappearance_states[frames[2].objects[0].track_id] == "UNAVAILABLE_NO_FORMAL_REID"


def test_real_adapter_reuses_klt_for_metric_point_tracks():
    rng = np.random.default_rng(7)
    first = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    images = [np.roll(first, shift=index, axis=1) for index in range(3)]
    mask = np.ones((64, 64), dtype=bool)
    frames = [_frame(index, image, _object(index, mask)) for index, image in enumerate(images)]
    clip = VideoClip("clip", "video", (0, 1, 2), (0.0, 1 / 30, 2 / 30), tuple(images))

    tracks = LegacyTrackProviderAdapter(
        LegacyPoseProviderAdapter(), maximum_point_tracks=16
    ).track(clip, frames)

    assert tracks
    assert any(len(track.frame_indices) == 3 for track in tracks)
    assert all(track.metadata["source_tracker"] == "opencv_lk_forward_backward" for track in tracks)
    assert all(track.metadata["support_type"] == "OBJECT" for track in tracks)
    assert all(np.any(track.valid_mask) for track in tracks)
    assert frames[1].actual_correspondences is not None
    clip_observation = ClipObservation(
        video_id="video", clip_id="clip", frames=frames, tracks=list(tracks)
    )
    compute_motion_residuals(clip_observation)
