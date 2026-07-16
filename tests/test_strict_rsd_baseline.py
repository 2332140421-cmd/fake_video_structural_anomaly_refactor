from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import csv
import math
import sys
from collections import Counter
from pathlib import Path

import pytest

from semantic3d.empirical_pair_prior import EmpiricalPairPriorResolver
from semantic3d.observations import FrameObservationJSON, ObjectObservationJSON
from semantic3d.strict_scale_prior import (
    StrictPhysicalPriorEntry,
    StrictPhysicalScalePriorResolver,
    load_strict_physical_prior_resolver,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from audit_physical_scale_priors import audit_candidate  # noqa: E402
from run_strict_rsd_baseline import (  # noqa: E402
    aggregate_video,
    evaluate_frame,
    evaluate_object_pair,
    plot_score_by_video,
    save_csv,
)


def _entry(
    label: str,
    low: float | None,
    high: float | None,
    *,
    status: str = "reliable_single",
    reliable: bool = True,
) -> StrictPhysicalPriorEntry:
    """Build a compact strict-prior entry for isolated unit tests."""

    return StrictPhysicalPriorEntry(
        label=label,
        min_size=low,
        max_size=high,
        characteristic_dimension="equivalent_reference_silhouette_linear_scale",
        dimension_definition="sqrt(reference height * reference width)",
        unit="m",
        estimation_method="test_fixture_only",
        source_count=1,
        sources=({"source_id": "fixture"},),
        audit_status=status,
        reliable=reliable,
        reliability_reason="deterministic test fixture",
        pose_sensitivity="low",
        multimodal_warning=False,
        reviewed_at="2026-07-14",
        prior_version="test_v1",
    )


def _resolver() -> StrictPhysicalScalePriorResolver:
    """Return a resolver containing reliable, alias, and unreliable paths."""

    return StrictPhysicalScalePriorResolver(
        {
            "person": _entry("person", 1.0, 2.0),
            "cup": _entry("cup", 0.1, 0.2),
            "table": _entry("table", 1.0, 2.0),
            "book": _entry(
                "book", 0.1, 0.5, status="pose_sensitive", reliable=False
            ),
        },
        aliases={"dining_table": "table"},
        metadata={"prior_version": "test_v1"},
    )


def _object(
    object_id: str,
    label: str,
    mask_area: float,
    depth: float = 5.0,
) -> ObjectObservationJSON:
    """Build one geometric test observation."""

    return ObjectObservationJSON(
        object_id=object_id,
        label=label,
        mask_area=mask_area,
        frame_area=10_000.0,
        depth=depth,
        bbox=[0.0, 0.0, 10.0, 10.0],
    )


def test_reliable_pair_computes_rsd() -> None:
    row = evaluate_object_pair(
        "video", 0, _object("a", "person", 400.0), _object("b", "cup", 4.0), 5.0, _resolver()
    )

    assert row["valid"] is True
    assert row["skip_reason"] == ""
    assert row["rsd_log"] == pytest.approx(0.0)
    assert row["prior_source"] == "physical"


def test_reliable_alias_pair_computes_rsd() -> None:
    row = evaluate_object_pair(
        "video", 0, _object("a", "dining_table", 400.0), _object("b", "cup", 4.0), 5.0, _resolver()
    )

    assert row["valid"] is True
    assert row["object_a_prior_status"] == "alias"
    assert row["object_a_canonical_label"] == "table"


@pytest.mark.parametrize(
    ("label", "reason"),
    [("book", "unreliable_prior_a"), ("unknown_label", "missing_prior_a")],
)
def test_unavailable_prior_stays_nan(label: str, reason: str) -> None:
    row = evaluate_object_pair(
        "video", 0, _object("a", label, 100.0), _object("b", "cup", 4.0), 5.0, _resolver()
    )

    assert row["valid"] is False
    assert row["skip_reason"] == reason
    assert math.isfinite(float(row["observed_depth_ratio"]))
    assert math.isfinite(float(row["observed_log_ratio"]))
    assert math.isnan(float(row["rsd_ratio"]))
    assert math.isnan(float(row["rsd_log"]))
    assert "证据不足" in str(row["explanation_text"])
    assert "残差为零" in str(row["explanation_text"])


def test_invalid_geometry_has_specific_skip_reason() -> None:
    row = evaluate_object_pair(
        "video", 0, _object("a", "person", 400.0, depth=0.0), _object("b", "cup", 4.0), 5.0, _resolver()
    )

    assert row["skip_reason"] == "invalid_depth_a"
    assert math.isnan(float(row["rsd_log"]))


def test_no_valid_pair_aggregates_to_nan(tmp_path: Path) -> None:
    frame = FrameObservationJSON(
        frame_index=0,
        frame_id="frame_0",
        width=100,
        height=100,
        objects=[_object("a", "book", 100.0), _object("b", "cup", 4.0)],
    )
    pairs = evaluate_frame("video", frame, _resolver())
    aggregate = aggregate_video(
        {"video_id": "video", "label": 1, "label_name": "fake"},
        [frame],
        pairs,
        Counter({"unreliable": 1, "exact": 1}),
        topk=3,
    )

    assert aggregate["label"] == 1
    assert aggregate["label_name"] == "fake"
    assert aggregate["status"] == "insufficient_rsd_evidence"
    assert aggregate["valid_pair_ratio"] == pytest.approx(0.0)
    assert math.isnan(float(aggregate["rsd_log_mean"]))

    output = tmp_path / "features.csv"
    save_csv([aggregate], output, list(aggregate))
    with output.open("r", encoding="utf-8", newline="") as file:
        loaded = next(csv.DictReader(file))
    assert loaded["rsd_log_mean"] == "NaN"
    assert loaded["rsd_log_mean"] != "0"


def test_valid_pair_ratio_is_computed_from_candidates() -> None:
    frames = [
        FrameObservationJSON(
            frame_index=0,
            frame_id="frame_0",
            width=100,
            height=100,
            objects=[_object("a", "person", 400.0), _object("b", "cup", 4.0)],
        )
    ]
    valid = evaluate_frame("video", frames[0], _resolver())[0]
    skipped = evaluate_object_pair(
        "video", 1, _object("a", "book", 100.0), _object("b", "cup", 4.0), 5.0, _resolver()
    )
    aggregate = aggregate_video(
        {"video_id": "video", "label": 0, "label_name": "real"},
        frames,
        [valid, skipped],
        Counter(),
        topk=3,
    )

    assert aggregate["num_valid_pairs"] == 1
    assert aggregate["num_candidate_pairs"] == 2
    assert aggregate["valid_pair_ratio"] == pytest.approx(0.5)


def test_nan_video_is_not_plotted_as_zero(tmp_path: Path) -> None:
    output = tmp_path / "score.png"
    rows = [
        {"video_id": "has_score", "label_name": "real", "rsd_log_topk_mean": 0.4},
        {"video_id": "no_evidence", "label_name": "fake", "rsd_log_topk_mean": float("nan")},
    ]

    plotted = plot_score_by_video(rows, output)

    assert output.exists() and output.stat().st_size > 0
    assert plotted == ["has_score"]


def test_frozen_prior_is_independent_and_strict() -> None:
    resolver = load_strict_physical_prior_resolver(
        PROJECT_ROOT / "configs/scale_priors_strict_v1.yaml"
    )

    assert resolver.metadata["independent_of_evaluation_videos"] is True
    assert resolver.metadata["prior_source"] == "physical"
    assert resolver.resolve("sports ball").resolved_label == "ball"
    assert resolver.resolve("sports ball").resolution == "unreliable"


def test_audit_rules_reject_multimodal_candidate() -> None:
    candidate = {
        "characteristic_dimension": "equivalent_reference_silhouette_linear_scale",
        "dimension_definition": "sqrt(height * width)",
        "unit": "m",
        "min": 0.1,
        "max": 0.2,
        "estimation_method": "official standard",
        "interval_basis": "official_standard",
        "source_count": 1,
        "sources": [
            {
                "url": "https://example.org/standard",
                "title": "Standard",
                "publisher": "Standards body",
                "source_type": "official_standard",
                "used_for_interval": True,
                "dimensions": "diameter",
                "conversion_to_m": 1.0,
            }
        ],
        "pose_sensitivity": "low",
        "multimodal_warning": True,
        "multimodal_reason": "multiple physical subtypes",
    }

    row = audit_candidate("item", candidate, {"allowed_unit": "m"})

    assert row["audit_status"] == "conditional_multimodal"
    assert row["reliable"] is False


def test_empirical_pair_prior_is_only_an_abstract_interface() -> None:
    with pytest.raises(TypeError):
        EmpiricalPairPriorResolver()
