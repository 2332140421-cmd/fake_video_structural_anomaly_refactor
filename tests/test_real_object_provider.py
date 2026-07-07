from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import importlib.util
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from semantic3d.build_observations import build_frame_observation
from semantic3d.io import load_clip_observation, save_clip_observation
from semantic3d.provider_registry import get_object_provider
from semantic3d.real_object_provider import (
    RealObjectProvider,
    bbox_area_to_mask_area,
    normalize_label,
)
from semantic3d.build_observations import build_clip_observation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_test_frame(path: Path) -> None:
    """Write a small image for provider schema tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((120, 160, 3), 240, dtype=np.uint8)
    cv2.rectangle(image, (50, 35), (120, 90), (80, 80, 80), -1)
    cv2.imwrite(str(path), image)


def test_provider_registry_mock() -> None:
    provider = get_object_provider("mock", mock_mode="reasonable")

    objects = provider.predict("frame.png", frame_index=0, width=320, height=180)

    assert len(objects) == 2
    assert objects[0].label == "soccer_ball"
    assert objects[1].label == "elephant"


def test_real_provider_missing_dependency() -> None:
    if importlib.util.find_spec("ultralytics") is not None:
        pytest.skip("ultralytics is installed; missing-dependency path is not active.")

    with pytest.raises(RuntimeError, match="requires the optional dependency 'ultralytics'"):
        RealObjectProvider(backend="ultralytics")


def test_bbox_area_to_mask_area() -> None:
    assert bbox_area_to_mask_area([10, 20, 30, 50]) == pytest.approx(600.0)
    assert normalize_label("sports ball") == "soccer_ball"
    assert normalize_label("traffic light") == "traffic_light"


def test_build_real_observation_json_schema(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    _write_test_frame(frame_path)

    provider = RealObjectProvider(
        detector=lambda _path: [
            {
                "label": "person",
                "bbox": [20, 25, 70, 105],
                "confidence": 0.91,
            },
            {
                "label": "car",
                "bbox": [80, 40, 145, 95],
                "confidence": 0.86,
            },
        ],
        default_depth=6.0,
        confidence_threshold=0.3,
    )
    frame_obs = build_frame_observation(frame_path, 3, provider)
    clip_obs = build_clip_observation(
        "schema_video", "schema_clip_000", [frame_obs], metadata={"provider": "test"}
    )
    output_path = tmp_path / "observations" / "schema_clip_000.json"
    save_clip_observation(clip_obs, output_path)
    loaded = load_clip_observation(output_path)

    assert loaded.frames[0].frame_index == 3
    assert loaded.frames[0].objects[0].label == "person"
    assert loaded.frames[0].objects[0].mask_area == pytest.approx(4000.0)
    assert loaded.frames[0].objects[0].frame_area == pytest.approx(160.0 * 120.0)
    assert loaded.frames[0].objects[0].depth == pytest.approx(6.0)
    assert loaded.frames[0].objects[0].bbox == [20.0, 25.0, 70.0, 105.0]


def test_missing_scale_prior_is_handled(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    _write_test_frame(frame_path)
    provider = RealObjectProvider(
        detector=lambda _path: [
            {
                "label": "traffic light",
                "bbox": [10, 10, 40, 50],
                "confidence": 0.95,
            },
            {
                "label": "person",
                "bbox": [60, 20, 100, 110],
                "confidence": 0.9,
            },
        ],
        default_depth=7.0,
        confidence_threshold=0.3,
    )

    with pytest.warns(RuntimeWarning, match="no scale prior is available"):
        objects = provider.predict(frame_path, frame_index=0, width=160, height=120)

    assert len(objects) == 1
    assert objects[0].label == "person"


def test_build_real_observations_script_with_mock_provider(tmp_path: Path) -> None:
    video_path = tmp_path / "test_real.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (160, 96),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create test video.")
    try:
        for _index in range(8):
            writer.write(np.full((96, 160, 3), 240, dtype=np.uint8))
    finally:
        writer.release()

    observation_dir = tmp_path / "real_observations"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_real_object_observations_from_video.py"),
            "--video_path",
            str(video_path),
            "--output_dir",
            str(observation_dir),
            "--max_frames",
            "8",
            "--clip_len",
            "4",
            "--stride",
            "4",
            "--object_provider",
            "mock",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    json_paths = sorted(observation_dir.glob("*.json"))
    assert len(json_paths) == 2
    clip_obs = load_clip_observation(json_paths[0])
    assert clip_obs.metadata["object_provider"] == "mock"
    assert len(clip_obs.frames[0].objects) == 2
