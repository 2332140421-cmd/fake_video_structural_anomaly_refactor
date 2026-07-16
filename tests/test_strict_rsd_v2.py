from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import csv
import hashlib
import math
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

from semantic3d.dimension_aligned_scale_depth import load_dimension_aligned_prior_resolver
from semantic3d.io import save_clip_observation
from semantic3d.observations import ClipObservationJSON, FrameObservationJSON, ObjectObservationJSON


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_strict_rsd_versions import compare  # noqa: E402
from run_strict_rsd_v2 import aggregate_video, run_v2_baseline  # noqa: E402


V1_HASH = "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_strict_v1_remains_frozen() -> None:
    v1 = PROJECT_ROOT / "configs/scale_priors_strict_v1.yaml"
    assert _sha256(v1) == V1_HASH


def test_v2_statuses_and_dimension_alignment_are_explicit() -> None:
    resolver = load_dimension_aligned_prior_resolver(
        PROJECT_ROOT / "configs/scale_priors_strict_v2.yaml"
    )
    assert resolver.resolve("soccer_ball").entry.reliability_status == "strict_high"  # type: ignore[union-attr]
    assert resolver.resolve("person").entry.reliability_status == "conditional_physical"  # type: ignore[union-attr]
    assert resolver.resolve("cup").entry.compatibility_group == "vertical_extent"  # type: ignore[union-attr]
    assert resolver.resolve("person").entry.projected_measurement == "bbox_height_norm"  # type: ignore[union-attr]
    assert resolver.resolve("book").entry.reliability_status == "pose_sensitive"  # type: ignore[union-attr]


def test_v2_configuration_was_not_fit_from_pilot_videos() -> None:
    path = PROJECT_ROOT / "configs/scale_priors_strict_v2.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    statement = data["metadata"]["independence_statement"]
    assert "pilot six videos" in statement
    assert data["metadata"]["empirical_pair_prior_enabled"] is False
    text = path.read_text(encoding="utf-8")
    assert "data/tests_videos" not in text
    assert "outputs/evaluation" not in text


def test_strict_high_and_conditional_are_aggregated_separately() -> None:
    rows = [
        {
            "frame_index": 0,
            "valid": True,
            "skip_reason": "",
            "evidence_tier": "strict_high",
            "rsd_log": 0.2,
        },
        {
            "frame_index": 1,
            "valid": True,
            "skip_reason": "",
            "evidence_tier": "conditional_physical",
            "rsd_log": 0.8,
        },
        {
            "frame_index": 1,
            "valid": False,
            "skip_reason": "pose_sensitive_prior_a",
            "evidence_tier": "unavailable",
            "rsd_log": float("nan"),
        },
    ]
    frame = FrameObservationJSON(frame_index=0, frame_id="f0", width=10, height=10, objects=[])
    result = aggregate_video(
        {"video_id": "video", "label": 0, "label_name": "real"},
        [frame, frame],
        rows,
        Counter({"strict_high": 1, "conditional_physical": 1}),
        topk=3,
    )
    assert result["strict_high_num_pairs"] == 1
    assert result["strict_high_rsd_log_mean"] == pytest.approx(0.2)
    assert result["conditional_num_pairs"] == 1
    assert result["conditional_rsd_log_mean"] == pytest.approx(0.8)
    assert result["combined_rsd_log_mean"] == pytest.approx(0.5)


def _object(object_id: str, label: str, bbox: list[float], depth: float) -> ObjectObservationJSON:
    width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return ObjectObservationJSON(
        object_id=object_id,
        label=label,
        mask_area=width * height,
        frame_area=1_000_000.0,
        depth=depth,
        confidence=0.9,
        bbox=bbox,
    )


def test_v2_end_to_end_uses_separate_directory_and_preserves_nan(tmp_path: Path) -> None:
    person = _object("p", "person", [100, 100, 300, 600], 2.0)
    cup = _object("c", "cup", [700, 400, 740, 450], 1.0)
    frame = FrameObservationJSON(
        frame_index=0,
        frame_id="f0",
        width=1000,
        height=1000,
        objects=[person, cup],
    )
    clip = ClipObservationJSON(
        clip_id="clip",
        video_id="video",
        frame_indices=[0],
        frames=[frame],
        metadata={"depth_mode": "real_depth_invert"},
    )
    observation_root = tmp_path / "pilot"
    save_clip_observation(
        clip,
        observation_root / "videos/video/associated_observations/associated.json",
    )
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "video_id,video_path,label,label_name,split\nvideo,unused.mp4,0,real,val\n",
        encoding="utf-8",
    )
    v1_dir = tmp_path / "rsd_strict_baseline"
    v1_dir.mkdir()
    sentinel = v1_dir / "sentinel.txt"
    sentinel.write_text("v1 frozen", encoding="utf-8")
    v2_dir = tmp_path / "rsd_strict_v2"

    metadata = run_v2_baseline(
        manifest,
        observation_root,
        PROJECT_ROOT / "configs/scale_priors_strict_v2.yaml",
        v2_dir,
        overwrite=False,
    )

    assert sentinel.read_text(encoding="utf-8") == "v1 frozen"
    assert v2_dir != v1_dir
    assert metadata["empirical_pair_prior_enabled"] is False
    with (v2_dir / "per_video_rsd_features.csv").open("r", encoding="utf-8", newline="") as file:
        video = next(csv.DictReader(file))
    assert video["conditional_num_pairs"] == "1"
    assert video["strict_high_rsd_log_mean"] == "NaN"
    assert video["strict_high_rsd_log_mean"] != "0"


def test_current_six_video_v2_outputs_are_separate_when_present() -> None:
    v1_dir = PROJECT_ROOT / "outputs/evaluation/rsd_strict_baseline"
    v2_dir = PROJECT_ROOT / "outputs/evaluation/rsd_strict_v2"
    if not v1_dir.exists() or not v2_dir.exists():
        pytest.skip("Local six-video evaluation outputs are not committed fixtures.")
    assert v1_dir.resolve() != v2_dir.resolve()
    assert (v1_dir / "run_metadata.json").exists()
    assert (v2_dir / "run_metadata.json").exists()


def test_v1_v2_comparison_outputs_engineering_coverage_only(tmp_path: Path) -> None:
    v1_dir, v2_dir = tmp_path / "v1", tmp_path / "v2"
    v1_dir.mkdir()
    v2_dir.mkdir()
    video_header = "video_id,label,label_name\nvideo,0,real\n"
    (v1_dir / "per_video_rsd_features.csv").write_text(video_header, encoding="utf-8")
    (v2_dir / "per_video_rsd_features.csv").write_text(video_header, encoding="utf-8")
    (v1_dir / "per_pair_rsd_details.csv").write_text(
        "video_id,valid,skip_reason\nvideo,False,unreliable_prior_a\n",
        encoding="utf-8",
    )
    (v2_dir / "per_pair_rsd_details.csv").write_text(
        "video_id,valid,skip_reason,evidence_tier\n"
        "video,True,,conditional_physical\n",
        encoding="utf-8",
    )
    output = tmp_path / "comparison"

    rows = compare(v1_dir, v2_dir, output)

    assert len(rows) == 4
    assert rows[0]["valid_pairs"] == 0
    assert rows[2]["valid_pairs"] == 1
    assert (output / "v1_v2_comparison.csv").exists()
    assert (output / "visualizations/v1_v2_pair_coverage.png").exists()
    assert (output / "visualizations/v1_v2_video_coverage.png").exists()
    assert (output / "visualizations/v2_skip_reasons.png").exists()
