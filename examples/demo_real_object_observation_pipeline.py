"""Demo for YOLO-based real-object observation construction from a local video."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _bootstrap import ensure_project_environment

PROJECT_ROOT = ensure_project_environment()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


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
    """Run YOLO real-object observation construction and R_sd scoring."""

    video_path = PROJECT_ROOT / "data" / "videos" / "test_real.mp4"
    model_path = PROJECT_ROOT / "checkpoints" / "yolov8n.pt"
    if not video_path.exists():
        raise FileNotFoundError(
            f"Demo video is missing: {video_path}. Put a video at this path first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO weights are missing: {model_path}. "
            "Place yolov8n.pt in checkpoints/yolov8n.pt first."
        )

    observation_dir = PROJECT_ROOT / "outputs" / "real_observations" / "real_detector_demo"
    output_csv = PROJECT_ROOT / "outputs" / "results" / "real_detector_rsd_results.csv"
    output_png = (
        PROJECT_ROOT / "outputs" / "visualizations" / "real_detector_rsd_scores.png"
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
        "--object_provider",
        "real_detector",
        "--model_path",
        str(model_path),
        "--default_depth",
        "5.0",
        "--device",
        "cpu",
        "--keep_unknown_scale_prior",
    ]

    result = run_command(build_base)
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
