from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from semantic3d.build_observations import build_frame_observation
from semantic3d.io import load_clip_observation
from semantic3d.providers import MockObjectProvider
from semantic3d.scale_depth import ScalePrior, scale_depth_residual_log
from semantic3d.video_preprocess import build_clips, extract_frames


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCALE_PRIORS = {
    "soccer_ball": ScalePrior(min_size=0.20, max_size=0.24),
    "elephant": ScalePrior(min_size=2.40, max_size=3.40),
}


def _create_test_video(video_path: Path, num_frames: int = 12) -> None:
    """Create a small video for pipeline tests."""

    video_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 160, 96
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create test video: {video_path}")

    try:
        for frame_index in range(num_frames):
            frame = np.full((height, width, 3), 240, dtype=np.uint8)
            cv2.circle(frame, (30 + frame_index, 70), 7, (50, 90, 230), -1)
            cv2.rectangle(frame, (92, 35), (142, 82), (120, 120, 120), -1)
            writer.write(frame)
    finally:
        writer.release()


def _compute_mock_pair_residual(mode: str) -> float:
    """Compute log-space R_sd for the mock soccer_ball-elephant pair."""

    provider = MockObjectProvider(mode=mode)  # type: ignore[arg-type]
    objects = provider.predict("frame.png", frame_index=0, width=320, height=180)
    soccer_ball = objects[0].to_scale_depth_observation()
    elephant = objects[1].to_scale_depth_observation()
    residual, _details = scale_depth_residual_log(
        soccer_ball, elephant, SCALE_PRIORS
    )
    return residual


def test_extract_frames(tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    _create_test_video(video_path, num_frames=10)

    frame_paths, frame_indices = extract_frames(
        video_path, tmp_path / "frames", max_frames=5
    )

    assert len(frame_paths) == 5
    assert frame_indices == [0, 1, 2, 3, 4]
    assert all(path.exists() for path in frame_paths)


def test_build_clips() -> None:
    frame_paths = [Path(f"frame_{index:03d}.png") for index in range(10)]

    clips = build_clips(frame_paths, clip_len=4, stride=3)

    assert len(clips) == 3
    assert clips[0].frame_indices == [0, 1, 2, 3]
    assert clips[1].frame_indices == [3, 4, 5, 6]
    assert clips[2].frame_indices == [6, 7, 8, 9]


def test_mock_object_provider_reasonable() -> None:
    residual = _compute_mock_pair_residual("reasonable")

    assert residual == pytest.approx(0.0, abs=1e-6)


def test_mock_object_provider_anomaly() -> None:
    residual = _compute_mock_pair_residual("anomaly")

    assert residual > 1.0


def test_build_observations_from_video(tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    observation_dir = tmp_path / "observations"
    _create_test_video(video_path, num_frames=10)

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_observations_from_video.py"),
            "--video_path",
            str(video_path),
            "--output_dir",
            str(observation_dir),
            "--max_frames",
            "10",
            "--clip_len",
            "4",
            "--stride",
            "3",
            "--mock_mode",
            "reasonable",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    json_paths = sorted(observation_dir.glob("*.json"))
    assert len(json_paths) == 3
    clip_obs = load_clip_observation(json_paths[0])
    assert clip_obs.metadata["mock_mode"] == "reasonable"
    assert len(clip_obs.frames) == 4
    assert len(clip_obs.frames[0].objects) == 2


def test_build_frame_observation_reads_image_size(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    cv2.imwrite(str(frame_path), image)

    frame_obs = build_frame_observation(
        frame_path, 7, MockObjectProvider(mode="reasonable")
    )

    assert frame_obs.width == 96
    assert frame_obs.height == 64
    assert frame_obs.frame_index == 7
    assert len(frame_obs.objects) == 2


def test_run_observation_rsd_pipeline(tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    observation_dir = tmp_path / "observations"
    output_csv = tmp_path / "results" / "video_rsd_results.csv"
    output_png = tmp_path / "visualizations" / "video_rsd_clip_scores.png"
    _create_test_video(video_path, num_frames=12)

    build_script = PROJECT_ROOT / "scripts" / "build_observations_from_video.py"
    for mode in ("reasonable", "anomaly"):
        subprocess.run(
            [
                sys.executable,
                str(build_script),
                "--video_path",
                str(video_path),
                "--output_dir",
                str(observation_dir),
                "--max_frames",
                "12",
                "--clip_len",
                "4",
                "--stride",
                "4",
                "--mock_mode",
                mode,
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_observation_rsd_pipeline.py"),
            "--observation_dir",
            str(observation_dir),
            "--output_csv",
            str(output_csv),
            "--visualization_path",
            str(output_png),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_csv.exists()
    assert output_png.exists()
    with output_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows
    reasonable_scores = [
        float(row["clip_score"])
        for row in rows
        if row["expected_mode"] == "reasonable"
    ]
    anomaly_scores = [
        float(row["clip_score"]) for row in rows if row["expected_mode"] == "anomaly"
    ]
    assert reasonable_scores
    assert anomaly_scores
    assert max(anomaly_scores) > max(reasonable_scores) + 1.0
