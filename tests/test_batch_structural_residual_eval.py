from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import csv
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from scripts.analyze_structural_residual_distributions import analyze
from scripts.build_video_manifest import (
    build_manifest_rows,
    save_manifest,
    select_smoke_rows,
    validate_manifest_rows,
)
from scripts.run_batch_structural_residual_eval import (
    PER_VIDEO_FIELDS,
    _summary,
    load_config,
    run_evaluation,
    validate_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_video(path: Path, frames: int = 4) -> None:
    """Create a small deterministic video for pipeline tests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (96, 64)
    )
    for index in range(frames):
        image = np.full((64, 96, 3), 235, dtype=np.uint8)
        cv2.rectangle(image, (8 + index, 36), (23 + index, 52), (20, 80, 220), -1)
        cv2.rectangle(image, (52, 10), (88, 47), (60, 170, 80), -1)
        writer.write(image)
    writer.release()
    assert path.exists()


def _test_config(path: Path) -> None:
    """Write a fast mock configuration using the same pipeline contracts."""

    data = {
        "video": {"fps": None, "max_frames": 4, "clip_len": 2, "stride": 2},
        "object_detection": {
            "provider": "mock",
            "model_path": "checkpoints/yolov8n.pt",
            "confidence_threshold": 0.3,
            "device": "cpu",
            "keep_unknown_scale_prior": True,
            "mock_mode": "anomaly",
        },
        "depth": {
            "provider": "mock_depth",
            "model_name": "unused",
            "device": "cpu",
            "invert_depth": True,
            "depth_mode": "real_depth_invert",
            "save_depth_maps": True,
            "default_depth": 5.0,
            "convention": "larger_is_farther",
            "metric_depth": False,
        },
        "scale_depth": {
            "scale_prior_path": "configs/scale_priors.yaml",
            "topk": 2,
        },
        "depth_consistency": {
            "tolerance": 0.02,
            "topk": 2,
            "iou_threshold": 0.1,
            "center_distance_threshold": 0.25,
            "max_area_ratio": 3.0,
            "max_frame_gap": 1,
        },
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _prepare_and_run(tmp_path: Path) -> Path:
    """Build a two-video manifest and run the lightweight batch pipeline."""

    real_path = tmp_path / "real" / "real_case.mp4"
    fake_path = tmp_path / "fake" / "fake_case.mp4"
    _write_video(real_path)
    _write_video(fake_path)
    rows = build_manifest_rows([real_path], [fake_path])
    manifest = tmp_path / "manifest.csv"
    save_manifest(rows, manifest)
    config = tmp_path / "eval.yaml"
    _test_config(config)
    output = tmp_path / "evaluation"
    run_evaluation(
        manifest,
        config,
        output,
        overwrite=True,
        allow_test_providers=True,
    )
    return output


def test_current_video_directories_and_production_config() -> None:
    """Use the existing test-video layout and enforce the real pilot settings."""

    real_dir = PROJECT_ROOT / "data/tests_videos/tests_real_videos"
    fake_dir = PROJECT_ROOT / "data/tests_videos/tests_fake_videos"
    assert sorted(path.name for path in real_dir.glob("*.mp4")) == [
        "real_1.mp4",
        "real_2.mp4",
        "real_3.mp4",
        "real_4.mp4",
    ]
    assert sorted(path.name for path in fake_dir.glob("*.mp4")) == [
        "fake_1.mp4",
        "fake_2.mp4",
    ]
    assert not (PROJECT_ROOT / "data/eval_videos").exists()

    config = load_config(PROJECT_ROOT / "configs/structural_residual_eval.yaml")
    validate_config(config)
    assert config["depth_consistency"]["tolerance"] == pytest.approx(0.02)


def test_manifest_labels_split_paths_and_smoke_selection(tmp_path: Path) -> None:
    """Manifest rows preserve real/fake labels and select one of each for smoke."""

    real_paths = [tmp_path / "real_1.mp4", tmp_path / "real_2.mp4"]
    fake_paths = [tmp_path / "fake_1.mp4"]
    for path in [*real_paths, *fake_paths]:
        path.touch()
    rows = build_manifest_rows(real_paths, fake_paths)
    validate_manifest_rows(rows)
    assert [row["label"] for row in rows] == [0, 0, 1]
    assert {row["split"] for row in rows} == {"val"}
    smoke = select_smoke_rows(rows)
    assert [row["label"] for row in smoke] == [0, 1]


def test_duplicate_video_id_is_rejected(tmp_path: Path) -> None:
    """The manifest must not silently merge videos with the same stem."""

    real = tmp_path / "real" / "same.mp4"
    fake = tmp_path / "fake" / "same.mp4"
    real.parent.mkdir()
    fake.parent.mkdir()
    real.touch()
    fake.touch()
    with pytest.raises(ValueError, match="Duplicate video_id"):
        build_manifest_rows([real], [fake])


def test_batch_outputs_quality_and_analysis(tmp_path: Path) -> None:
    """The batch pipeline creates tables, diagnostics, and six plots."""

    output = _prepare_and_run(tmp_path)
    required = [
        output / "per_video_features.csv",
        output / "per_clip_features.csv",
        output / "quality_report.csv",
        output / "scale_prior_coverage.csv",
    ]
    assert all(path.exists() for path in required)

    with required[0].open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert set(PER_VIDEO_FIELDS) <= set(rows[0])
    assert {row["video_id"] for row in rows} == {"real_case", "fake_case"}
    assert all(int(row["num_objects"]) > 0 for row in rows)
    assert all(int(row["duplicate_track_frame_count"]) == 0 for row in rows)
    assert all(int(row["one_to_many_assignment_count"]) == 0 for row in rows)

    analysis_dir = output / "analysis"
    analyze(output, analysis_dir)
    png_names = {
        "rsd_log_distribution.png",
        "depth_cons_raw_distribution.png",
        "valid_rsd_pairs_per_video.png",
        "valid_depth_transitions_per_video.png",
        "scale_prior_coverage.png",
        "track_quality.png",
    }
    assert {path.name for path in analysis_dir.glob("*.png")} == png_names
    assert (analysis_dir / "group_summary.csv").exists()
    text = (analysis_dir / "analysis_summary.txt").read_text(encoding="utf-8")
    assert "cannot define final thresholds" in text
    assert "Clips are not treated as independent video samples" in text


def test_empty_evidence_uses_nan() -> None:
    """An empty evidence set must not be summarized as a zero residual."""

    summary = _summary([], topk=3)
    assert all(math.isnan(value) for value in summary.values())
