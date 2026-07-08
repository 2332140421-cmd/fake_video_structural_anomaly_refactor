from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from semantic3d.io import load_clip_observation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_video(path: Path, width: int = 96, height: int = 64, frames: int = 8) -> None:
    """Write a tiny deterministic video for tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (width, height),
    )
    for index in range(frames):
        image = np.full((height, width, 3), 235, dtype=np.uint8)
        cv2.rectangle(image, (10 + index, 32), (28 + index, 50), (30, 80, 220), -1)
        cv2.rectangle(image, (54 - index // 2, 14), (88 - index // 2, 46), (80, 180, 60), -1)
        writer.write(image)
    writer.release()


def _run_compare(tmp_path: Path) -> Path:
    """Run the comparison script with mock providers and return output dir."""

    video_path = tmp_path / "tiny.mp4"
    output_dir = tmp_path / "depth_direction"
    _write_video(video_path)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "compare_depth_direction_rsd.py"),
        "--video_path",
        str(video_path),
        "--output_dir",
        str(output_dir),
        "--max_frames",
        "8",
        "--clip_len",
        "4",
        "--stride",
        "4",
        "--object_provider",
        "mock",
        "--mock_mode",
        "anomaly",
        "--real_depth_backend",
        "mock_depth",
        "--device",
        "cpu",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    return output_dir


def _all_depths(observation_dir: Path) -> list[float]:
    """Collect object depths from clip observation JSON files."""

    depths = []
    for path in sorted(observation_dir.glob("*.json")):
        clip = load_clip_observation(path)
        depths.extend(float(obj.depth) for frame in clip.frames for obj in frame.objects)
    return depths


def test_compare_depth_direction_script_runs(tmp_path: Path) -> None:
    output_dir = _run_compare(tmp_path)

    assert (output_dir / "no_depth_observations").exists()
    assert (output_dir / "real_depth_no_invert_observations").exists()
    assert (output_dir / "real_depth_invert_observations").exists()


def test_summary_csv_created(tmp_path: Path) -> None:
    output_dir = _run_compare(tmp_path)
    summary_path = output_dir / "depth_direction_summary.csv"

    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert {row["mode"] for row in rows} == {
        "no_depth",
        "real_depth_no_invert",
        "real_depth_invert",
    }


def test_no_depth_has_default_depth(tmp_path: Path) -> None:
    output_dir = _run_compare(tmp_path)
    depths = _all_depths(output_dir / "no_depth_observations")

    assert depths
    assert set(round(depth, 4) for depth in depths) == {5.0}


def test_real_depth_or_mock_depth_changes_depth(tmp_path: Path) -> None:
    output_dir = _run_compare(tmp_path)
    depths = _all_depths(output_dir / "real_depth_no_invert_observations")

    assert depths
    assert len(set(round(depth, 4) for depth in depths)) > 1
    assert set(round(depth, 4) for depth in depths) != {5.0}


def test_visualize_depth_direction_debug_runs(tmp_path: Path) -> None:
    output_dir = _run_compare(tmp_path)
    debug_dir = tmp_path / "debug"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "visualize_depth_direction_debug.py"),
        "--no_invert_observation_dir",
        str(output_dir / "real_depth_no_invert_observations"),
        "--invert_observation_dir",
        str(output_dir / "real_depth_invert_observations"),
        "--no_invert_depth_dir",
        str(output_dir / "real_depth_no_invert_depth_maps"),
        "--invert_depth_dir",
        str(output_dir / "real_depth_invert_depth_maps"),
        "--no_invert_rsd_csv",
        str(output_dir / "real_depth_no_invert_rsd_results.csv"),
        "--invert_rsd_csv",
        str(output_dir / "real_depth_invert_rsd_results.csv"),
        "--output_dir",
        str(debug_dir),
        "--max_frames",
        "2",
    ]

    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

    assert list(debug_dir.glob("*.png"))


def test_empty_rsd_is_handled(tmp_path: Path) -> None:
    from scripts.compare_depth_direction_rsd import save_summary, summarize_mode

    observation_dir = tmp_path / "empty_observations"
    observation_dir.mkdir()
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text(
        "video_id,clip_id,frame_index,object_pair,R_sd,R_sd_log,clip_score,expected_mode\n",
        encoding="utf-8",
    )

    summary = summarize_mode("empty", observation_dir, csv_path)
    assert summary["rsd_rows"] == 0
    assert summary["num_objects"] == 0

    summary_path = tmp_path / "summary.csv"
    save_summary([summary], summary_path)
    assert summary_path.exists()
