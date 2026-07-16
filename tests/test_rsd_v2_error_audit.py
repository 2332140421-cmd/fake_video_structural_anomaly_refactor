from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import csv
import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from semantic3d.io import save_clip_observation
from semantic3d.observations import (
    ClipObservationJSON,
    FrameObservationJSON,
    ObjectObservationJSON,
)
from semantic3d.rsd_v2_error_audit import (
    DEPTH_STRATEGIES,
    compute_depth_strategy,
    deterministic_track_id,
    diagnostic_labels,
    foreground_depth_cluster,
    make_frame_pair_id,
    make_track_pair_id,
    recompute_scale_depth_formula,
    safe_bbox_bounds,
    swapped_log_residual_consistent,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_strict_rsd_v2_errors import (  # noqa: E402
    DEPTH_COMPARISON_FIELDS,
    PAIR_AUDIT_FIELDS,
    build_audit_data,
    run_audit,
    save_csv,
)


V1_HASH = "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b"
V2_HASH = "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(
    object_id: str,
    label: str,
    bbox: list[float],
    depth: float,
    track_id: str | None,
) -> ObjectObservationJSON:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return ObjectObservationJSON(
        object_id=object_id,
        label=label,
        mask_area=width * height,
        frame_area=10_000.0,
        depth=depth,
        confidence=0.9,
        bbox=bbox,
        track_id=track_id,
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    observation_root = tmp_path / "observations"
    video_root = observation_root / "videos/demo"
    image_path = tmp_path / "frame.png"
    depth_path = tmp_path / "depth.npy"
    image = np.full((100, 100, 3), 220, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    depth = np.full((100, 100), 6.0, dtype=np.float32)
    depth[10:90, 10:40] = 4.0
    depth[60:80, 60:80] = 2.0
    np.save(depth_path, depth)

    person = _object("person_0", "person", [10, 10, 40, 90], 4.0, "trk_person")
    cup = _object("cup_0", "cup", [60, 60, 80, 80], 2.0, "trk_cup")
    frame = FrameObservationJSON(
        frame_index=0,
        frame_id="demo_frame_000000",
        width=100,
        height=100,
        objects=[person, cup],
        image_path=str(image_path),
        depth_map_path=str(depth_path),
    )
    associated = ClipObservationJSON(
        clip_id="demo_associated",
        video_id="demo",
        frame_indices=[0],
        frames=[frame],
        metadata={"depth_mode": "real_depth_invert"},
    )
    save_clip_observation(
        associated,
        video_root / "associated_observations/demo_associated.json",
    )

    raw_person = _object("person_0", "person", [10, 10, 40, 90], 4.0, None)
    raw_cup = _object("cup_0", "cup", [60, 60, 80, 80], 2.0, None)
    raw_frame = FrameObservationJSON(
        frame_index=0,
        frame_id="demo_frame_000000",
        width=100,
        height=100,
        objects=[raw_person, raw_cup],
        image_path=str(image_path),
        depth_map_path=str(depth_path),
    )
    for index in range(2):
        clip = ClipObservationJSON(
            clip_id=f"demo_clip_{index}",
            video_id="demo",
            frame_indices=[0],
            frames=[raw_frame],
            metadata={"depth_mode": "real_depth_invert"},
        )
        save_clip_observation(clip, video_root / f"observations/demo_clip_{index}.json")

    input_dir = tmp_path / "strict_v2"
    input_dir.mkdir()
    (input_dir / "run_metadata.json").write_text(
        json.dumps({"observation_root": str(observation_root)}), encoding="utf-8"
    )
    result = recompute_scale_depth_formula(4.0, 2.0, 1.5, 2.0, 0.08, 0.12, 0.8, 0.2)
    pair = {
        "video_id": "demo",
        "frame_index": 0,
        "object_a_id": "person_0",
        "object_b_id": "cup_0",
        "object_a_label": "person",
        "object_b_label": "cup",
        "object_a_prior_status": "conditional_physical",
        "object_b_prior_status": "conditional_physical",
        "object_a_prior_low": 1.5,
        "object_a_prior_high": 2.0,
        "object_b_prior_low": 0.08,
        "object_b_prior_high": 0.12,
        "characteristic_dimension_a": "body_height",
        "characteristic_dimension_b": "upright_height",
        "projected_measurement_a": 0.8,
        "projected_measurement_b": 0.2,
        "measurement_quality_a": "bbox_extent",
        "measurement_quality_b": "bbox_extent",
        "gate_passed_a": True,
        "gate_passed_b": True,
        "gate_score_a": 1.0,
        "gate_score_b": 1.0,
        "gate_reasons_a": "valid",
        "gate_reasons_b": "valid",
        "failed_gate_reasons_a": "",
        "failed_gate_reasons_b": "",
        "rsd_ratio": result["rsd_ratio"],
        "rsd_log": result["rsd_log"],
        "evidence_tier": "conditional_physical",
        "depth_mode": "real_depth_invert",
        "valid": True,
    }
    fields = list(pair)
    with (input_dir / "per_pair_rsd_details.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(pair)
        writer.writerow(pair)
    return input_dir, tmp_path / "videos", tmp_path / "audit"


def test_frame_pair_id_is_unique_by_global_frame() -> None:
    first = make_frame_pair_id("v", 1, "track_a", "track_b")
    second = make_frame_pair_id("v", 2, "track_a", "track_b")
    assert first != second


def test_track_pair_id_is_order_independent_and_stable() -> None:
    assert make_track_pair_id("v", "b", "a") == make_track_pair_id("v", "a", "b")


def test_missing_track_id_has_deterministic_fallback() -> None:
    obj = _object("object_1", "cup", [0, 0, 10, 10], 2.0, None)
    assert deterministic_track_id(obj) == ("fallback:cup:object_1", True)


def test_formula_recomputation_and_interval_order_are_correct() -> None:
    result = recompute_scale_depth_formula(4.0, 2.0, 1.5, 2.0, 0.08, 0.12, 0.8, 0.2)
    assert result["observed_depth_ratio"] == pytest.approx(2.0)
    assert result["expected_ratio_low"] == pytest.approx(3.125)
    assert result["expected_ratio_high"] == pytest.approx(6.25)
    assert result["expected_ratio_low"] <= result["expected_ratio_high"]
    assert result["rsd_ratio"] == pytest.approx(1.125)
    assert result["rsd_log"] > 0


def test_swapped_pair_preserves_log_residual() -> None:
    arguments = {
        "depth_a": 4.0,
        "depth_b": 2.0,
        "prior_a_min": 1.5,
        "prior_a_max": 2.0,
        "prior_b_min": 0.08,
        "prior_b_max": 0.12,
        "projected_a": 0.8,
        "projected_b": 0.2,
    }
    assert swapped_log_residual_consistent(arguments)


def test_bbox_crop_is_clipped_to_image_bounds() -> None:
    assert safe_bbox_bounds([-5, -4, 15, 14], 10, 10) == (0, 0, 10, 10)
    assert safe_bbox_bounds([5, 5, 5, 9], 10, 10) is None


def test_depth_strategies_do_not_modify_observation() -> None:
    obj = _object("cup", "cup", [1, 1, 9, 9], 3.0, "t")
    before = deepcopy(obj.to_dict())
    depth = np.arange(100, dtype=float).reshape(10, 10) + 1
    for strategy in DEPTH_STRATEGIES:
        compute_depth_strategy(depth, obj, strategy)
    assert obj.to_dict() == before


def test_invalid_depth_is_nan_not_zero() -> None:
    obj = _object("cup", "cup", [1, 1, 9, 9], 3.0, "t")
    statistic = compute_depth_strategy(np.full((10, 10), np.nan), obj, "full_bbox_median")
    formula = recompute_scale_depth_formula(math.nan, 2.0, 1.0, 2.0, 0.1, 0.2, 0.5, 0.1)
    assert not statistic.valid
    assert math.isnan(statistic.depth)
    assert math.isnan(formula["rsd_log"])


def test_foreground_cluster_safely_falls_back_when_unstable() -> None:
    result = foreground_depth_cluster(np.full((10, 10), 4.0))
    assert result.valid
    assert result.depth == pytest.approx(4.0)
    assert "fallback" in result.method_detail


def test_mask_depth_without_mask_stays_nan() -> None:
    obj = _object("cup", "cup", [1, 1, 9, 9], 3.0, "t")
    result = compute_depth_strategy(np.ones((10, 10)), obj, "mask_median_depth")
    assert not result.valid
    assert math.isnan(result.depth)


def test_diagnostic_labels_are_explicit_heuristics() -> None:
    labels = diagnostic_labels(
        {
            "person_touches_top": False,
            "person_touches_bottom": False,
            "person_touches_left": False,
            "person_touches_right": False,
            "person_aspect_ratio": 0.5,
            "person_height_temporal_cv": 0.01,
            "cup_bbox_width": 200,
            "cup_bbox_height": 200,
            "cup_bbox_area_ratio": 0.09,
            "cup_aspect_ratio": 1.0,
            "cup_aspect_ratio_temporal_cv": 0.01,
            "cup_current_depth": 4.0,
            "cup_full_depth_iqr": 1.0,
            "depth_ratio_temporal_cv": 0.01,
            "rsd_log": 1.0,
        }
    )
    assert "likely_bbox_background_contamination" in labels
    assert "likely_depth_background_contamination" in labels


def test_overlapping_clips_and_pair_rows_are_deduplicated(tmp_path: Path) -> None:
    input_dir, _, _ = _fixture(tmp_path)
    audit_rows, depth_rows, projection_rows, stats, _ = build_audit_data(
        input_dir, "demo", "person", "cup"
    )
    assert len(audit_rows) == 1
    assert len(depth_rows) == len(DEPTH_STRATEGIES)
    assert len(projection_rows) == 3
    assert stats["input_duplicate_frame_pair_count"] == 1
    assert stats["duplicate_frame_pair_count"] == 0
    assert stats["repeated_clip_pair_count"] == 1


def test_diagnostic_csv_preserves_nan_marker(tmp_path: Path) -> None:
    path = tmp_path / "diagnostic.csv"
    save_csv([{"rsd_log": math.nan}], path, ["rsd_log"])
    assert path.read_text(encoding="utf-8").splitlines()[1] == "NaN"


def test_end_to_end_report_and_diagnostic_images_are_generated(tmp_path: Path) -> None:
    input_dir, video_dir, output_dir = _fixture(tmp_path)
    report = run_audit(input_dir, video_dir, "demo", "person", "cup", output_dir)
    assert report["num_valid_frame_pairs"] == 1
    assert report["num_unique_track_pairs"] == 1
    assert report["duplicate_frame_pair_count"] == 0
    assert (output_dir / "error_attribution_report.json").exists()
    assert (output_dir / "frames/frame_000000_person_cup_audit.png").stat().st_size > 0
    assert (output_dir / "person_cup_rsd_by_depth_strategy.png").stat().st_size > 0
    with (output_dir / "person_cup_depth_strategy_comparison.csv").open(
        "r", encoding="utf-8", newline=""
    ) as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len(DEPTH_STRATEGIES)
    assert set(PAIR_AUDIT_FIELDS).issubset(
        (output_dir / "person_cup_pair_audit.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    )
    assert set(DEPTH_COMPARISON_FIELDS).issubset(rows[0])


def test_strict_v1_v2_hashes_are_unchanged() -> None:
    assert _hash(PROJECT_ROOT / "configs/scale_priors_strict_v1.yaml") == V1_HASH
    assert _hash(PROJECT_ROOT / "configs/scale_priors_strict_v2.yaml") == V2_HASH


def test_real3_was_not_used_to_modify_physical_min_max() -> None:
    config_path = PROJECT_ROOT / "configs/scale_priors_strict_v2.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    text = config_path.read_text(encoding="utf-8")
    assert config["metadata"]["empirical_pair_prior_enabled"] is False
    assert "real_3" not in text
    assert "data/tests_videos" not in text
    assert "outputs/evaluation" not in text
