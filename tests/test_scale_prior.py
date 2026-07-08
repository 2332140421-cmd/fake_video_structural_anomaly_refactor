from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import sys
from pathlib import Path

import pytest

from semantic3d.io import save_clip_observation
from semantic3d.observations import (
    ClipObservationJSON,
    FrameObservationJSON,
    ObjectObservationJSON,
)
from semantic3d.scale_prior import (
    ScalePriorResolver,
    default_scale_prior_resolver,
    load_scale_prior_resolver,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_observation_rsd_pipeline import compute_rows  # noqa: E402
from report_missing_scale_priors import (  # noqa: E402
    build_report_rows,
    collect_label_counts,
    save_report,
)
from generate_scale_prior_candidates import (  # noqa: E402
    build_candidate_data,
    load_report_rows,
    save_candidates,
)


def test_exact_prior() -> None:
    resolver = default_scale_prior_resolver(PROJECT_ROOT)

    resolved = resolver.resolve("person")

    assert resolved.resolved_label == "person"
    assert resolved.source == "exact"
    assert resolved.reliable is True
    assert resolved.prior.min_size == pytest.approx(1.30)

    cup = resolver.resolve("cup")
    assert cup.resolved_label == "cup"
    assert cup.source == "exact"
    assert cup.prior.min_size == pytest.approx(0.07)


def test_alias_prior() -> None:
    resolver = default_scale_prior_resolver(PROJECT_ROOT)

    dining_table = resolver.resolve("dining_table")
    sports_ball = resolver.resolve("sports ball")
    mouse = resolver.resolve("mouse")

    assert dining_table.resolved_label == "table"
    assert dining_table.source == "alias"
    assert sports_ball.resolved_label == "ball"
    assert sports_ball.source == "alias"
    assert mouse.resolved_label == "handheld_object"
    assert mouse.source == "alias"


def test_missing_prior() -> None:
    resolver = default_scale_prior_resolver(PROJECT_ROOT)

    resolved = resolver.resolve("unknown_label")

    assert resolved.source == "missing"
    assert resolved.resolved_label == "unknown_label"
    assert resolved.reliable is False


def test_unreliable_prior() -> None:
    resolver = default_scale_prior_resolver(PROJECT_ROOT)

    vase = resolver.resolve("vase", require_reliable=True)
    plant = resolver.resolve("potted_plant", require_reliable=True)
    vase_without_requirement = resolver.resolve("vase", require_reliable=False)

    assert vase.source == "unreliable"
    assert vase.reliable is False
    assert plant.source == "unreliable"
    assert plant.reliable is False
    assert vase_without_requirement.source == "exact"
    assert vase_without_requirement.reliable is False


def test_yaml_loader_returns_resolver() -> None:
    resolver = load_scale_prior_resolver(
        PROJECT_ROOT / "configs" / "scale_priors.yaml"
    )

    assert isinstance(resolver, ScalePriorResolver)
    assert resolver.resolve("coffee_table").resolved_label == "table"  # type: ignore[union-attr]


def test_rsd_pipeline_skips_unknown_labels_safely(tmp_path: Path) -> None:
    frame_area = 640.0 * 480.0
    clip = ClipObservationJSON(
        clip_id="clip_scale_prior_test",
        video_id="video_scale_prior_test",
        frame_indices=[0],
        frames=[
            FrameObservationJSON(
                frame_index=0,
                frame_id="frame_000000",
                width=640,
                height=480,
                objects=[
                    ObjectObservationJSON(
                        object_id="person_1",
                        label="person",
                        mask_area=20_000.0,
                        frame_area=frame_area,
                        depth=5.0,
                        bbox=[100, 100, 220, 300],
                    ),
                    ObjectObservationJSON(
                        object_id="table_1",
                        label="dining_table",
                        mask_area=40_000.0,
                        frame_area=frame_area,
                        depth=5.0,
                        bbox=[260, 220, 520, 380],
                    ),
                    ObjectObservationJSON(
                        object_id="mystery_1",
                        label="unknown_label",
                        mask_area=10_000.0,
                        frame_area=frame_area,
                        depth=5.0,
                        bbox=[20, 20, 80, 120],
                    ),
                ],
            )
        ],
        metadata={"expected_mode": "test"},
    )
    input_dir = tmp_path / "observations"
    save_clip_observation(clip, input_dir / "clip_scale_prior_test.json")

    rows, stats = compute_rows(
        input_dir,
        resolver=default_scale_prior_resolver(PROJECT_ROOT),
        return_stats=True,
    )

    assert stats["total_objects"] == 3
    assert stats["exact_prior_objects"] == 1
    assert stats["alias_prior_objects"] == 1
    assert stats["skipped_missing_prior_objects"] == 1
    assert stats["skipped_unreliable_prior_objects"] == 0
    assert stats["total_candidate_pairs"] == 3
    assert stats["computed_pairs"] == 1
    assert stats["skipped_pairs_missing_prior"] == 2
    assert stats["skipped_pairs_unreliable_prior"] == 0
    assert len(rows) == 1
    assert rows[0]["object_pair"] == "person_1->table_1"


def test_report_missing_scale_priors(tmp_path: Path) -> None:
    frame_area = 640.0 * 480.0
    clip = ClipObservationJSON(
        clip_id="clip_report_test",
        video_id="video_report_test",
        frame_indices=[0],
        frames=[
            FrameObservationJSON(
                frame_index=0,
                frame_id="frame_000000",
                width=640,
                height=480,
                objects=[
                    ObjectObservationJSON(
                        object_id="cup_1",
                        label="cup",
                        mask_area=2_000.0,
                        frame_area=frame_area,
                        depth=2.0,
                    ),
                    ObjectObservationJSON(
                        object_id="plant_1",
                        label="potted plant",
                        mask_area=10_000.0,
                        frame_area=frame_area,
                        depth=4.0,
                    ),
                    ObjectObservationJSON(
                        object_id="unknown_1",
                        label="unknown_label",
                        mask_area=1_000.0,
                        frame_area=frame_area,
                        depth=3.0,
                    ),
                ],
            )
        ],
    )
    input_dir = tmp_path / "observations"
    save_clip_observation(clip, input_dir / "clip_report_test.json")

    resolver = default_scale_prior_resolver(PROJECT_ROOT)
    rows = build_report_rows(collect_label_counts(input_dir), resolver, min_count=1)
    output_csv = tmp_path / "missing_scale_prior_report.csv"
    save_report(rows, output_csv)

    by_label = {str(row["label"]): row for row in rows}
    assert by_label["cup"]["status"] == "exact"
    assert by_label["potted_plant"]["status"] == "unreliable"
    assert by_label["unknown_label"]["status"] == "missing"
    assert output_csv.exists()

    candidates = build_candidate_data(load_report_rows(output_csv), min_count=1)
    candidate_yaml = tmp_path / "scale_prior_candidates.yaml"
    save_candidates(candidates, candidate_yaml)
    text = candidate_yaml.read_text(encoding="utf-8")
    main_config = (PROJECT_ROOT / "configs" / "scale_priors.yaml").read_text(
        encoding="utf-8"
    )

    assert "candidate_scale_priors" in text
    assert "unknown_label" in candidates
    assert candidates["unknown_label"]["action"] == "review"
    assert "potted_plant" in candidates
    assert candidates["potted_plant"]["action"] == "review_or_skip"
    assert "candidate_scale_priors" not in main_config


def test_rsd_pipeline_skips_unreliable(tmp_path: Path) -> None:
    frame_area = 640.0 * 480.0
    clip = ClipObservationJSON(
        clip_id="clip_unreliable_test",
        video_id="video_unreliable_test",
        frame_indices=[0],
        frames=[
            FrameObservationJSON(
                frame_index=0,
                frame_id="frame_000000",
                width=640,
                height=480,
                objects=[
                    ObjectObservationJSON(
                        object_id="cup_1",
                        label="cup",
                        mask_area=2_000.0,
                        frame_area=frame_area,
                        depth=2.0,
                    ),
                    ObjectObservationJSON(
                        object_id="vase_1",
                        label="vase",
                        mask_area=8_000.0,
                        frame_area=frame_area,
                        depth=3.0,
                    ),
                ],
            )
        ],
    )
    input_dir = tmp_path / "observations"
    save_clip_observation(clip, input_dir / "clip_unreliable_test.json")

    rows, stats = compute_rows(
        input_dir,
        resolver=default_scale_prior_resolver(PROJECT_ROOT),
        return_stats=True,
    )

    assert stats["total_objects"] == 2
    assert stats["exact_prior_objects"] == 1
    assert stats["skipped_unreliable_prior_objects"] == 1
    assert stats["total_candidate_pairs"] == 1
    assert stats["computed_pairs"] == 0
    assert stats["skipped_pairs_unreliable_prior"] == 1
    assert rows == []
