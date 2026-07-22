"""Tests for verified source lineage, official split priority, and leakage audit."""

from __future__ import annotations

from semantic3d.experiment_protocol.formal_schema import FormalSplitRecord, SourceLineageRecord
from semantic3d.experiment_protocol.source_lineage import (
    audit_formal_split_leakage,
    plan_verified_formal_splits,
)


def _lineage(video_id: str, original_id: str, *, verified: bool = True) -> SourceLineageRecord:
    return SourceLineageRecord(
        dataset_name="formal_unit",
        original_source_id=original_id if verified else "",
        source_group_id=f"group-{original_id or video_id}",
        derived_video_id=video_id,
        source_video_name=video_id,
        video_path=f"data/{video_id}.mp4",
        source_sha256=(video_id * 64)[:64],
        authenticity_label=0 if video_id.startswith("real") else 1,
        manipulation_type="none" if video_id.startswith("real") else "synthetic",
        derivation_status="verified" if verified else "unresolved",
        identity_evidence="official_metadata" if verified else "file_hash_only",
        verification_status="verified" if verified else "unverified",
        formal_split_eligible=verified,
        ineligibility_reason="" if verified else "original_source_identity_unverified",
        dataset_role="formal",
    )


def test_official_split_has_priority_for_complete_source_group() -> None:
    rows = (_lineage("real-a", "origin-a"), _lineage("fake-a", "origin-a"))
    plan = plan_verified_formal_splits(
        rows,
        official_split_by_video={"real-a": "val", "fake-a": "validation"},
    )
    assert {row.formal_split for row in plan} == {"validation"}
    assert all(row.official_split_preserved for row in plan)
    assert all(row.split_source == "official" for row in plan)


def test_group_aware_fallback_keeps_original_and_derivatives_together() -> None:
    rows = (_lineage("real-a", "origin-a"), _lineage("fake-a", "origin-a"))
    first = plan_verified_formal_splits(rows, random_seed=7)
    second = plan_verified_formal_splits(tuple(reversed(rows)), random_seed=7)
    assert {row.formal_split for row in first} == {first[0].formal_split}
    assert [(row.derived_video_id, row.formal_split) for row in first] == [
        (row.derived_video_id, row.formal_split) for row in second
    ]
    assert all(row.split_source == "group_aware_deterministic" for row in first)


def test_unverified_identity_is_blocked_before_split() -> None:
    plan = plan_verified_formal_splits((_lineage("unknown", "", verified=False),))
    assert plan[0].formal_split == ""
    assert not plan[0].eligible
    assert plan[0].blocked_reason == "unverified_source_lineage"


def test_conflicting_official_splits_block_the_whole_source_group() -> None:
    rows = (_lineage("real-a", "origin-a"), _lineage("fake-a", "origin-a"))
    plan = plan_verified_formal_splits(
        rows,
        official_split_by_video={"real-a": "train", "fake-a": "test"},
    )
    assert all(not row.formal_split for row in plan)
    assert all(row.blocked_reason == "source_group_official_split_conflict" for row in plan)


def test_leakage_audit_detects_manual_origin_cross_split() -> None:
    rows = (_lineage("real-a", "origin-a"), _lineage("fake-a", "origin-a"))
    splits = (
        FormalSplitRecord(
            "formal_unit",
            "real-a",
            "origin-a",
            "group-origin-a",
            "train",
            "official",
            True,
            "",
            True,
            "unit",
            1,
        ),
        FormalSplitRecord(
            "formal_unit",
            "fake-a",
            "origin-a",
            "group-origin-a",
            "test",
            "official",
            True,
            "",
            True,
            "unit",
            1,
        ),
    )
    audit = audit_formal_split_leakage(rows, splits)
    assert audit["cross_split_leakage_detected"]
    assert {row["type"] for row in audit["findings"]} >= {
        "original_source_cross_split",
        "source_group_cross_split",
    }

