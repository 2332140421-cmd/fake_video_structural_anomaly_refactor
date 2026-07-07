from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from semantic3d.build_observations import build_clip_observation, build_frame_observation
from semantic3d.io import load_clip_observation, save_clip_observation
from semantic3d.provider_registry import get_object_provider
from semantic3d.providers import MockObjectProvider
from semantic3d.real_object_provider import (
    RealObjectProvider,
    bbox_area_to_mask_area,
    normalize_label,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
YOLO_MODEL = PROJECT_ROOT / "checkpoints" / "yolov8n.pt"
TEST_VIDEO = PROJECT_ROOT / "data" / "videos" / "test_real.mp4"


def _write_test_frame(path: Path) -> None:
    """Write a small image for provider schema tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((120, 160, 3), 240, dtype=np.uint8)
    cv2.rectangle(image, (50, 35), (120, 90), (80, 80, 80), -1)
    cv2.imwrite(str(path), image)


def test_real_object_provider_import() -> None:
    assert RealObjectProvider.__name__ == "RealObjectProvider"


def test_real_object_provider_init() -> None:
    if not YOLO_MODEL.exists():
        pytest.skip("YOLO weights are not available.")

    provider = RealObjectProvider(
        model_path=YOLO_MODEL,
        confidence_threshold=0.3,
        default_depth=5.0,
        device="cpu",
    )

    assert isinstance(provider, RealObjectProvider)


def test_provider_registry_mock() -> None:
    provider = get_object_provider("mock", mock_mode="reasonable")

    objects = provider.predict("frame.png", frame_index=0, width=320, height=180)

    assert isinstance(provider, MockObjectProvider)
    assert len(objects) == 2
    assert objects[0].label == "soccer_ball"
    assert objects[1].label == "elephant"


def test_provider_registry_real_detector() -> None:
    if not YOLO_MODEL.exists():
        pytest.skip("YOLO weights are not available.")

    provider = get_object_provider(
        "real_detector",
        model_path=YOLO_MODEL,
        confidence_threshold=0.3,
        default_depth=5.0,
        device="cpu",
    )

    assert isinstance(provider, RealObjectProvider)


def test_bbox_area_to_mask_area() -> None:
    assert bbox_area_to_mask_area([10, 20, 30, 50]) == pytest.approx(600.0)


def test_label_mapping() -> None:
    assert normalize_label("sports ball") == "soccer_ball"
    assert normalize_label("person") == "person"
    assert normalize_label("car") == "car"


def test_unknown_label_handling(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    _write_test_frame(frame_path)
    detections = [
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
    ]

    skip_provider = RealObjectProvider(
        detector=lambda _path: detections,
        default_depth=7.0,
        confidence_threshold=0.3,
        skip_unknown_scale_prior=True,
    )
    with pytest.warns(RuntimeWarning, match="missing scale prior"):
        skipped_objects = skip_provider.predict(
            frame_path, frame_index=0, width=160, height=120
        )
    assert [obj.label for obj in skipped_objects] == ["person"]

    keep_provider = RealObjectProvider(
        detector=lambda _path: detections,
        default_depth=7.0,
        confidence_threshold=0.3,
        skip_unknown_scale_prior=False,
    )
    kept_objects = keep_provider.predict(frame_path, frame_index=0, width=160, height=120)
    assert [obj.label for obj in kept_objects] == ["traffic_light", "person"]


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


def test_real_detection_on_sample_frame(tmp_path: Path) -> None:
    if not YOLO_MODEL.exists():
        pytest.skip("YOLO weights are not available.")
    if not TEST_VIDEO.exists():
        pytest.skip("test_real.mp4 is not available.")

    capture = cv2.VideoCapture(str(TEST_VIDEO))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        pytest.skip("Could not read the first frame from test_real.mp4.")

    frame_path = tmp_path / "sample_frame.jpg"
    cv2.imwrite(str(frame_path), frame)
    height, width = frame.shape[:2]
    provider = RealObjectProvider(
        model_path=YOLO_MODEL,
        confidence_threshold=0.3,
        default_depth=5.0,
        device="cpu",
        skip_unknown_scale_prior=False,
    )

    objects = provider.predict(frame_path, frame_index=0, width=width, height=height)

    assert isinstance(objects, list)
    assert len(objects) > 0
    assert all(obj.mask_area > 0 for obj in objects)
    assert all(obj.depth == pytest.approx(5.0) for obj in objects)


def test_build_real_observations_script_with_real_detector(tmp_path: Path) -> None:
    if not YOLO_MODEL.exists():
        pytest.skip("YOLO weights are not available.")
    if not TEST_VIDEO.exists():
        pytest.skip("test_real.mp4 is not available.")

    observation_dir = tmp_path / "real_observations"
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_real_object_observations_from_video.py"),
            "--video_path",
            str(TEST_VIDEO),
            "--output_dir",
            str(observation_dir),
            "--max_frames",
            "4",
            "--clip_len",
            "4",
            "--stride",
            "4",
            "--object_provider",
            "real_detector",
            "--model_path",
            str(YOLO_MODEL),
            "--confidence_threshold",
            "0.3",
            "--default_depth",
            "5.0",
            "--device",
            "cpu",
            "--keep_unknown_scale_prior",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    json_paths = sorted(observation_dir.glob("*.json"))
    assert len(json_paths) == 1
    clip_obs = load_clip_observation(json_paths[0])
    assert clip_obs.metadata["object_provider"] == "real_detector"
    assert clip_obs.metadata["mask_area_source"] == "bbox_area"
    assert len(clip_obs.frames) == 4
    assert any(frame.objects for frame in clip_obs.frames)
