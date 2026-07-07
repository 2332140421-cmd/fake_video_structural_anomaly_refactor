from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import pytest

from semantic3d.data_structures import (
    ClipObservation,
    FrameObservation,
    ObjectTrack,
    ResidualReport,
)
from semantic3d.interfaces import (
    DepthProvider,
    FlowProvider,
    SegmentationProvider,
    TrackerProvider,
)
from semantic3d.scale_depth import ObjectObservation


def _object() -> ObjectObservation:
    return ObjectObservation(
        object_id="obj_1",
        label="soccer_ball",
        mask_area=100.0,
        frame_area=1_000.0,
        depth=4.0,
        confidence=0.9,
    )


def test_frame_observation_accepts_image_path_or_image_id() -> None:
    frame = FrameObservation(
        frame_index=0,
        image_path="/tmp/frame_000.png",
        width=1920,
        height=1080,
        objects=[_object()],
        depth_map_path="/tmp/depth_000.npy",
        flow_path="/tmp/flow_000.npy",
    )

    assert frame.frame_area == 1920 * 1080
    assert frame.objects[0].object_id == "obj_1"


def test_frame_observation_requires_image_path_or_image_id() -> None:
    with pytest.raises(ValueError, match="image_path or image_id"):
        FrameObservation(frame_index=0, width=1920, height=1080)


def test_clip_observation_requires_valid_range_and_frames() -> None:
    frame = FrameObservation(
        frame_index=3,
        image_id="frame_003",
        width=640,
        height=480,
        objects=[_object()],
    )
    clip = ClipObservation(
        clip_id="clip_a",
        frames=[frame],
        clip_start=3,
        clip_end=3,
    )

    assert clip.frames[0].image_id == "frame_003"

    with pytest.raises(ValueError, match="clip_end must be >= clip_start"):
        ClipObservation(clip_id="bad", frames=[frame], clip_start=5, clip_end=4)


def test_object_track_requires_aligned_lengths() -> None:
    track = ObjectTrack(
        object_id="obj_1",
        label="soccer_ball",
        frame_indices=[0, 1],
        centers=[(10.0, 20.0), (11.0, 21.0)],
        depths=[4.0, 4.2],
        mask_areas=[100.0, 110.0],
        projection_scales=[0.1, 0.11],
    )

    assert track.centers[1] == (11.0, 21.0)

    with pytest.raises(ValueError, match="equal lengths"):
        ObjectTrack(
            object_id="bad",
            label="soccer_ball",
            frame_indices=[0, 1],
            centers=[(10.0, 20.0)],
            depths=[4.0, 4.2],
            mask_areas=[100.0, 110.0],
            projection_scales=[0.1, 0.11],
        )


def test_residual_report_stores_scores_and_explanations() -> None:
    report = ResidualReport(
        clip_id="clip_a",
        frame_scores=[0.1, 0.4],
        object_pair_residuals={
            "obj_1->obj_2": {"scale_depth": 0.3, "flow": 0.1}
        },
        total_score=0.25,
        explanations=["scale-depth residual is elevated for obj_1->obj_2"],
    )

    assert report.total_score == pytest.approx(0.25)
    assert "obj_1->obj_2" in report.object_pair_residuals


def test_provider_interfaces_are_abstract() -> None:
    for interface in [
        SegmentationProvider,
        DepthProvider,
        FlowProvider,
        TrackerProvider,
    ]:
        with pytest.raises(TypeError):
            interface()
