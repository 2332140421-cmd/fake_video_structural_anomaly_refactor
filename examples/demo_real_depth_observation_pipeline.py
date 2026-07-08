"""Demo for YOLO observations with real monocular depth aggregation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _bootstrap import ensure_project_environment

PROJECT_ROOT = ensure_project_environment()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic3d.io import load_clip_observation  # noqa: E402


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a project command and print it."""

    print("$ " + " ".join(command))
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> None:
    """Build real detector observations with real depth and run R_sd."""

    video_path = PROJECT_ROOT / "data" / "videos" / "test_real.mp4"
    model_path = PROJECT_ROOT / "checkpoints" / "yolov8n.pt"
    if not video_path.exists():
        raise FileNotFoundError(f"Video is missing: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"YOLO weights are missing: {model_path}. Place yolov8n.pt there first."
        )

    observation_dir = PROJECT_ROOT / "outputs" / "real_observations_depth"
    depth_output_dir = PROJECT_ROOT / "outputs" / "depth_maps"
    output_csv = PROJECT_ROOT / "outputs" / "results" / "real_depth_rsd_results.csv"
    output_png = PROJECT_ROOT / "outputs" / "visualizations" / "real_depth_rsd_scores.png"

    build_script = PROJECT_ROOT / "scripts" / "build_real_object_observations_from_video.py"
    rsd_script = PROJECT_ROOT / "scripts" / "run_observation_rsd_pipeline.py"

    build_result = run_command(
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
            "--object_provider",
            "real_detector",
            "--model_path",
            str(model_path),
            "--confidence_threshold",
            "0.3",
            "--depth_provider",
            "real_depth",
            "--depth_model_name",
            "depth-anything/Depth-Anything-V2-Small",
            "--depth_output_dir",
            str(depth_output_dir),
            "--save_depth_maps",
            "--device",
            "cpu",
        ]
    )
    print(build_result.stdout.strip())

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

    print("\nDetected objects with bbox-depth aggregation")
    for path in sorted(observation_dir.glob("*.json"))[:1]:
        clip = load_clip_observation(path)
        for frame in clip.frames[:2]:
            print(f"Frame {frame.frame_index}:")
            for obj in frame.objects:
                bbox = [round(value, 1) for value in (obj.bbox or [])]
                print(
                    f"  {obj.label}: bbox={bbox}, bbox_area={obj.mask_area:.2f}, "
                    f"depth={obj.depth:.3f}, confidence={obj.confidence:.3f}"
                )

    print("\nOutputs")
    print(f"  Observation JSON directory: {observation_dir}")
    print(f"  Depth maps directory: {depth_output_dir}")
    print(f"  R_sd CSV: {output_csv}")
    print(f"  Visualization: {output_png}")


if __name__ == "__main__":
    main()
