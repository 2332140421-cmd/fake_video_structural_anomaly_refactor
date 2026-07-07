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


def test_person_hits_exact_prior() -> None:
    resolver = default_scale_prior_resolver(PROJECT_ROOT)

    resolved = resolver.resolve("person")

    assert resolved is not None
    assert resolved.resolved_label == "person"
    assert resolved.resolution == "exact"
    assert resolved.prior.min_size == pytest.approx(1.30)


def test_dining_table_alias_maps_to_table() -> None:
    resolver = default_scale_prior_resolver(PROJECT_ROOT)

    resolved = resolver.resolve("dining_table")

    assert resolved is not None
    assert resolved.resolved_label == "table"
    assert resolved.resolution == "alias"


def test_sports_ball_alias_maps_to_ball() -> None:
    resolver = default_scale_prior_resolver(PROJECT_ROOT)

    resolved = resolver.resolve("sports ball")

    assert resolved is not None
    assert resolved.resolved_label == "ball"
    assert resolved.resolution == "alias"


def test_mouse_alias_maps_to_handheld_object() -> None:
    resolver = default_scale_prior_resolver(PROJECT_ROOT)

    resolved = resolver.resolve("mouse")

    assert resolved is not None
    assert resolved.resolved_label == "handheld_object"
    assert resolved.resolution == "alias"


def test_unknown_label_returns_none() -> None:
    resolver = default_scale_prior_resolver(PROJECT_ROOT)

    assert resolver.resolve("unknown_label") is None


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
    assert stats["skipped_unknown_objects"] == 1
    assert stats["computed_pairs"] == 1
    assert stats["skipped_pairs_missing_prior"] == 0
    assert len(rows) == 1
    assert rows[0]["object_pair"] == "person_1->table_1"
