"""Demo for real-object observation construction from a local video.

The demo first tries ``real_detector``. If no local detector dependency/weights
are available, it prints the error and falls back to ``mock`` so the observation
and R_sd pipeline can still be exercised end to end.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from _bootstrap import ensure_project_environment

PROJECT_ROOT = ensure_project_environment()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def create_fallback_video(video_path: Path, num_frames: int = 20) -> None:
    """Create a tiny local video when no user video is available."""

    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (320, 180),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create fallback video: {video_path}")
    try:
        for index in range(num_frames):
            frame = np.full((180, 320, 3), 245, dtype=np.uint8)
            cv2.rectangle(frame, (170, 70), (285, 145), (120, 120, 120), -1)
            cv2.circle(frame, (45 + index * 2, 130), 13, (60, 110, 235), -1)
            writer.write(frame)
    finally:
        writer.release()


def ensure_demo_video() -> Path:
    """Return data/videos/test_real.mp4, copying or creating it if needed."""

    target = PROJECT_ROOT / "data" / "videos" / "test_real.mp4"
    if target.exists():
        return target

    source = PROJECT_ROOT / "data" / "real_videos" / "real_1.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        shutil.copyfile(source, target)
    else:
        create_fallback_video(target)
    return target


def run_command(command: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a project command and return the completed process."""

    print("$ " + " ".join(command))
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def main() -> None:
    """Run real-object observation construction, with mock fallback if needed."""

    video_path = ensure_demo_video()
    observation_dir = PROJECT_ROOT / "outputs" / "real_observations"
    output_csv = PROJECT_ROOT / "outputs" / "results" / "real_object_video_rsd_results.csv"
    output_png = (
        PROJECT_ROOT / "outputs" / "visualizations" / "real_object_video_rsd_scores.png"
    )

    build_script = PROJECT_ROOT / "scripts" / "build_real_object_observations_from_video.py"
    rsd_script = PROJECT_ROOT / "scripts" / "run_observation_rsd_pipeline.py"

    build_base = [
        sys.executable,
        str(build_script),
        "--video_path",
        str(video_path),
        "--output_dir",
        str(observation_dir),
        "--max_frames",
        "20",
        "--clip_len",
        "8",
        "--stride",
        "4",
        "--confidence_threshold",
        "0.3",
    ]

    result = run_command([*build_base, "--object_provider", "real_detector"], check=False)
    if result.returncode != 0:
        print("\nreal_detector is not available in this environment.")
        print("Reason:")
        print(result.stderr.strip() or result.stdout.strip())
        print("\nFalling back to --object_provider mock for pipeline validation.")
        fallback = run_command(
            [*build_base, "--object_provider", "mock", "--mock_mode", "reasonable"]
        )
        print(fallback.stdout.strip())
    else:
        print(result.stdout.strip())

    rsd_result = run_command(
        [
            sys.executable,
            str(rsd_script),
            "--observation_dir",
            str(observation_dir),
            "--output_csv",
            str(output_csv),
            "--visualization_path",
            str(output_png),
        ]
    )
    print(rsd_result.stdout.strip())

    print("\nOutputs")
    print(f"  Observation JSON directory: {observation_dir}")
    print(f"  R_sd CSV: {output_csv}")
    print(f"  Visualization: {output_png}")


if __name__ == "__main__":
    main()
