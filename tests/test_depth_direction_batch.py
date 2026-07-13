from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import csv
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_video(path: Path, width: int = 96, height: int = 64, frames: int = 6) -> None:
    """Write a small deterministic video for batch pipeline tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (width, height),
    )
    for index in range(frames):
        image = np.full((height, width, 3), 238, dtype=np.uint8)
        cv2.rectangle(image, (8 + index, 34), (25 + index, 51), (40, 90, 220), -1)
        cv2.rectangle(
            image,
            (52 - index // 2, 12),
            (88 - index // 2, 45),
            (70, 180, 70),
            -1,
        )
        writer.write(image)
    writer.release()


def _run_batch(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Run the batch comparison script with mock providers."""

    video_dir = tmp_path / "videos"
    _write_video(video_dir / "case_a.mp4")
    _write_video(video_dir / "case_b.mp4")

    output_dir = tmp_path / "batch_outputs"
    summary_csv = tmp_path / "results" / "depth_direction_batch_summary.csv"
    visualization_dir = tmp_path / "visualizations"
    report_csv = tmp_path / "results" / "scale_prior_coverage.csv"
    candidates_yaml = tmp_path / "results" / "scale_prior_candidates.yaml"

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "batch_compare_depth_direction_rsd.py"),
        "--video_dir",
        str(video_dir),
        "--output_dir",
        str(output_dir),
        "--summary_csv",
        str(summary_csv),
        "--visualization_dir",
        str(visualization_dir),
        "--scale_prior_report_csv",
        str(report_csv),
        "--scale_prior_candidates_yaml",
        str(candidates_yaml),
        "--max_frames",
        "6",
        "--clip_len",
        "3",
        "--stride",
        "3",
        "--object_provider",
        "mock",
        "--mock_mode",
        "anomaly",
        "--real_depth_backend",
        "mock_depth",
        "--max_debug_frames",
        "1",
        "--device",
        "cpu",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    return summary_csv, visualization_dir, report_csv


def test_batch_depth_direction_summary_created(tmp_path: Path) -> None:
    summary_csv, _, _ = _run_batch(tmp_path)

    assert summary_csv.exists()
    with summary_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 4
    assert {row["video_id"] for row in rows} == {"case_a", "case_b"}
    assert {row["mode"] for row in rows} == {
        "real_depth_no_invert",
        "real_depth_invert",
    }


def test_batch_depth_direction_visualizations_created(tmp_path: Path) -> None:
    _, visualization_dir, _ = _run_batch(tmp_path)

    assert list((visualization_dir / "case_a").glob("*.png"))
    assert list((visualization_dir / "case_b").glob("*.png"))


def test_batch_scale_prior_coverage_report_created(tmp_path: Path) -> None:
    _, _, report_csv = _run_batch(tmp_path)

    assert report_csv.exists()
    with report_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    statuses = {row["status"] for row in rows}
    assert statuses
    assert statuses <= {"exact", "alias", "missing", "unreliable"}


def test_batch_scale_prior_candidates_created(tmp_path: Path) -> None:
    video_dir = tmp_path / "videos"
    _write_video(video_dir / "case_a.mp4")

    output_dir = tmp_path / "batch_outputs"
    summary_csv = tmp_path / "results" / "depth_direction_batch_summary.csv"
    visualization_dir = tmp_path / "visualizations"
    report_csv = tmp_path / "results" / "scale_prior_coverage.csv"
    candidates_yaml = tmp_path / "results" / "scale_prior_candidates.yaml"

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "batch_compare_depth_direction_rsd.py"),
        "--video_dir",
        str(video_dir),
        "--output_dir",
        str(output_dir),
        "--summary_csv",
        str(summary_csv),
        "--visualization_dir",
        str(visualization_dir),
        "--scale_prior_report_csv",
        str(report_csv),
        "--scale_prior_candidates_yaml",
        str(candidates_yaml),
        "--max_frames",
        "4",
        "--clip_len",
        "2",
        "--stride",
        "2",
        "--object_provider",
        "mock",
        "--mock_mode",
        "anomaly",
        "--real_depth_backend",
        "mock_depth",
        "--max_debug_frames",
        "1",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

    assert candidates_yaml.exists()
    data = yaml.safe_load(candidates_yaml.read_text(encoding="utf-8"))
    assert "candidate_scale_priors" in data
