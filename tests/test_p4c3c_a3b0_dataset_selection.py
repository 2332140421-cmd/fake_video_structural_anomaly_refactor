"""P4-C3C-A3-B0 candidate registry and metadata-only audit invariants."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from semantic3d.dataset_builder.candidate_audit import (
    CandidateAuditError,
    FORMAL_MAPPING_FIELDS,
    SCORE_FIELDS,
    summarize_candidate_registry,
    validate_candidate_registry,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = (
    PROJECT_ROOT
    / "configs/data_registry/p4c3c_a3b0_dataset_candidates_v1.yaml"
)


@pytest.fixture
def registry() -> dict:
    return yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))


def _record(registry: dict, dataset_id: str) -> dict:
    return next(
        item for item in registry["datasets"] if item["dataset_id"] == dataset_id
    )


def test_candidate_registry_validates_and_classifies_all_candidates(registry: dict) -> None:
    validate_candidate_registry(registry)
    summary = summarize_candidate_registry(registry)
    assert summary["candidate_count"] == 4
    assert set(summary["dataset_ids"]) == {
        "genvideo_100k",
        "genvidbench",
        "decof",
        "brokenvideos",
    }


def test_every_dataset_remains_a_non_ready_candidate(registry: dict) -> None:
    for record in registry["datasets"]:
        assert record["selection_status"] == "candidate"
        assert record["data_downloaded"] is False
        assert record["production_adapter_ready"] is False
        assert record["official_schema_verified"] is False


def test_score_fields_are_complete_and_bounded(registry: dict) -> None:
    for record in registry["datasets"]:
        assert tuple(record["scores"]) == SCORE_FIELDS
        assert all(0 <= score <= 5 for score in record["scores"].values())


def test_license_status_is_never_empty(registry: dict) -> None:
    for record in registry["datasets"]:
        assert str(record["license"]).strip()
        assert str(record["license_status"]).strip()


def test_random_split_and_official_test_media_are_forbidden(registry: dict) -> None:
    policies = registry["policies"]
    assert policies["random_split_allowed"] is False
    assert policies["official_test_media_allowed_in_small_sample"] is False
    assert policies["media_download_bytes"] == 0


def test_unresolved_formal_fields_remain_explicit(registry: dict) -> None:
    for record in registry["datasets"]:
        assert tuple(record["formal_mapping"]) == FORMAL_MAPPING_FIELDS
    genvideo = _record(registry, "genvideo_100k")
    assert genvideo["formal_mapping"]["video_path"] == "UNRESOLVED_SCHEMA"
    assert genvideo["formal_mapping"]["label"] == "UNRESOLVED_SCHEMA"
    broken = _record(registry, "brokenvideos")
    assert broken["formal_mapping"]["sample_id"] == "UNRESOLVED_SCHEMA"


def test_no_path_or_filename_inference_is_authorized(registry: dict) -> None:
    policies = registry["policies"]
    assert policies["label_from_path_allowed"] is False
    assert policies["generator_from_filename_allowed"] is False


def test_small_sample_plan_stays_blocked(registry: dict) -> None:
    recommendation = registry["recommendation"]
    assert recommendation["small_sample_source_identified"] is True
    assert recommendation["small_sample_download_ready"] is False
    assert all(
        record["small_sample_selection_without_test_possible"] == "BLOCKED"
        for record in registry["datasets"]
    )


def test_genvideo_adapter_semantics_remain_blocked(registry: dict) -> None:
    record = _record(registry, "genvideo_100k")
    assert record["schema_status"] == "BLOCKED"
    assert record["record_metadata_available"] == "NOT_AVAILABLE"
    assert record["small_sample_selection_possible"] == "BLOCKED"


def test_genvidbench_released_label_count_is_scoped_to_143k(registry: dict) -> None:
    record = _record(registry, "genvidbench")
    assert record["metadata_record_count"] == 141995
    assert record["train_count"] == 74135
    assert record["test_count"] == 67860
    assert record["schema_status"] == "PARTIAL"


def test_decof_official_splits_are_preserved(registry: dict) -> None:
    record = _record(registry, "decof")
    assert record["split_values"] == {
        "train": "train",
        "val": "validation",
        "test": "test",
    }
    assert (record["train_count"], record["validation_count"], record["test_count"]) == (
        771,
        96,
        97,
    )


def test_brokenvideos_is_spatial_only_not_a_real_fake_source(registry: dict) -> None:
    record = _record(registry, "brokenvideos")
    assert record["recommended_role"] == "PRIMARY_SPATIAL_LOCALIZATION_DATASET"
    assert record["mask_available"] == "VERIFIED"
    assert record["is_real_field"] == "NOT_APPLICABLE"
    assert record["formal_schema_mapping_possible"] == "BLOCKED"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("policies", "media_download_bytes"), 1),
        (("policies", "random_split_allowed"), True),
        (("policies", "label_from_path_allowed"), True),
        (("policies", "generator_from_filename_allowed"), True),
    ],
)
def test_validator_rejects_unsafe_global_policies(
    registry: dict,
    path: tuple[str, str],
    value: object,
) -> None:
    broken = deepcopy(registry)
    broken[path[0]][path[1]] = value
    with pytest.raises(CandidateAuditError, match="unsafe audit policy"):
        validate_candidate_registry(broken)


def test_validator_rejects_unverified_dataset_marked_ready(registry: dict) -> None:
    broken = deepcopy(registry)
    broken["datasets"][0]["production_adapter_ready"] = True
    with pytest.raises(CandidateAuditError, match="must remain false"):
        validate_candidate_registry(broken)


def test_validator_rejects_missing_score_field(registry: dict) -> None:
    broken = deepcopy(registry)
    del broken["datasets"][0]["scores"]["J"]
    with pytest.raises(CandidateAuditError, match="score fields"):
        validate_candidate_registry(broken)


def test_registry_contains_no_server_absolute_path() -> None:
    text = REGISTRY_PATH.read_text(encoding="utf-8")
    assert "/" + "root/autodl-tmp" not in text
    assert "/" + "home/" not in text
    assert "/" + "mnt/" not in text
