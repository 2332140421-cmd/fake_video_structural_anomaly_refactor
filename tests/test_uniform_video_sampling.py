import cv2
import numpy as np
import pytest

from data.video import (
    InsufficientUniqueFramesError,
    read_uniform_video_sample,
    split_uniform_sample,
    uniform_frame_indices,
)


def _video(path, frame_count):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 24),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        writer.write(np.full((24, 32, 3), index % 255, dtype=np.uint8))
    writer.release()


def test_uniform_sampler_spans_full_video_with_unique_integer_indices(tmp_path):
    video = tmp_path / "video.avi"
    _video(video, 120)
    sample = read_uniform_video_sample(video, requested_frame_count=32)

    assert sample.frame_indices == uniform_frame_indices(120, 32)
    assert sample.frame_indices[0] == 0
    assert sample.frame_indices[-1] == 119
    assert len(set(sample.frame_indices)) == 32
    assert all(
        left < right
        for left, right in zip(sample.frame_indices, sample.frame_indices[1:])
    )
    assert all(
        left <= right
        for left, right in zip(sample.timestamps, sample.timestamps[1:])
    )
    clips = split_uniform_sample(sample, clip_length=8, clip_count=4)
    assert len(clips) == 4
    assert [len(clip.frames) for clip in clips] == [8, 8, 8, 8]
    assert tuple(index for clip in clips for index in clip.frame_indices) == tuple(
        range(32)
    )
    assert sample.frame_indices == uniform_frame_indices(120, 32)


def test_short_video_is_rejected_without_looping_or_duplicate_frames(tmp_path):
    video = tmp_path / "short.avi"
    _video(video, 31)
    with pytest.raises(InsufficientUniqueFramesError, match="INSUFFICIENT_UNIQUE_FRAMES"):
        read_uniform_video_sample(video, requested_frame_count=32)


def test_uniform_index_rule_is_label_blind_and_validates_target():
    expected = uniform_frame_indices(37, 32)
    assert expected == uniform_frame_indices(37, 32)
    assert expected[0] == 0 and expected[-1] == 36
    with pytest.raises(ValueError, match="at least 2"):
        uniform_frame_indices(100, 1)
