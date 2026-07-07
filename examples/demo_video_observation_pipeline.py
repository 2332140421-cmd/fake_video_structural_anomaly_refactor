"""End-to-end demo: local video -> observation JSON -> scale-depth R_sd CSV."""

from __future__ import annotations

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


def create_demo_video(video_path: Path, num_frames: int = 20) -> None:
    """Create a small synthetic video if no local demo video exists."""

    video_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 320, 180
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create demo video: {video_path}")

    try:
        for index in range(num_frames):
            frame = np.full((height, width, 3), 245, dtype=np.uint8)
            ball_x = 40 + index * 3
            cv2.circle(frame, (ball_x, 125), 12, (70, 130, 240), -1)
            elephant_x = 190 - index
            cv2.rectangle(frame, (elephant_x, 60), (295 - index, 145), (120, 120, 120), -1)
            cv2.putText(
                frame,
                f"frame {index:02d}",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()


def run_command(command: list[str]) -> None:
    """Run a subprocess command from the project root and print it."""

    print("$ " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    """Run the minimal real-video-to-observation pipeline demo."""

    video_path = PROJECT_ROOT / "data" / "demo_videos" / "demo.mp4"
    if not video_path.exists():
        create_demo_video(video_path)
        print(f"Created synthetic demo video: {video_path}")
    else:
        print(f"Using existing demo video: {video_path}")

    observation_dir = PROJECT_ROOT / "outputs" / "observations"
    output_csv = PROJECT_ROOT / "outputs" / "results" / "video_rsd_results.csv"
    visualization_path = (
        PROJECT_ROOT / "outputs" / "visualizations" / "video_rsd_clip_scores.png"
    )

    build_script = PROJECT_ROOT / "scripts" / "build_observations_from_video.py"
    rsd_script = PROJECT_ROOT / "scripts" / "run_observation_rsd_pipeline.py"

    for mode in ("reasonable", "anomaly"):
        run_command(
            [
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
                "--mock_mode",
                mode,
            ]
        )

    run_command(
        [
            sys.executable,
            str(rsd_script),
            "--observation_dir",
            str(observation_dir),
            "--output_csv",
            str(output_csv),
            "--visualization_path",
            str(visualization_path),
        ]
    )

    print("\nOutputs")
    print(f"  Observation JSON directory: {observation_dir}")
    print(f"  R_sd CSV: {output_csv}")
    print(f"  Clip score plot: {visualization_path}")


if __name__ == "__main__":
    main()
