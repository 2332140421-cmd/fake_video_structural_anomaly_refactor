"""P4-C3C-A3-B0.2 DeCoF metadata-only and download-gate invariants."""

from __future__ import annotations

import json
import socket
import struct
from pathlib import Path

import pytest
import yaml

from semantic3d.dataset_adapters.decof import (
    DECOF_ADAPTER_STATUS,
    DECOF_ARCHIVE_LAYOUTS,
    DECOF_DATA_DOWNLOADED,
    DECOF_MEDIA_RESOLUTION_READY,
    DECOF_NETWORK_ACCESS_ALLOWED,
    DECOF_OFFICIAL_SCHEMA_VERIFIED,
    DECOF_PRODUCTION_ADAPTER_READY,
    DeCoFMetadataAdapter,
    DeCoFMetadataError,
    build_decof_small_sample_plan,
    parse_decof_zip_central_directory,
    summarize_decof_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = (
    PROJECT_ROOT / "configs/data_registry/decof_metadata_readiness_v1.yaml"
)


def _write_metadata_fixture(root: Path) -> Path:
    prompts = [
        {
            "video_id": "train_real_a",
            "prompt": "train prompt a",
            "source": "MSRVTT",
            "video_url": "https://example.invalid/train-a",
        },
        {
            "video_id": "train_real_b",
            "prompt": "train prompt b",
            "source": "MSRVTT",
            "video_url": "https://example.invalid/train-b",
        },
        {
            "video_id": "validation_real",
            "prompt": "validation prompt",
            "source": "MSRVTT",
            "video_url": "https://example.invalid/validation",
        },
        {
            "video_id": "test_real",
            "prompt": "test prompt",
            "source": "MSVD",
        },
    ]
    prompt_path = root / "datas/prompts/data.json"
    prompt_path.parent.mkdir(parents=True)
    prompt_path.write_text(
        "".join(json.dumps(row) + "\n" for row in prompts),
        encoding="utf-8",
    )
    split_root = root / "datas/split"
    split_root.mkdir(parents=True)
    (split_root / "train.json").write_text(
        "train_real_a\ntrain_real_b\n",
        encoding="utf-8",
    )
    (split_root / "val.json").write_text(
        "validation_real\n",
        encoding="utf-8",
    )
    (split_root / "test.json").write_text("test_real\n", encoding="utf-8")
    return root


def _central_directory_bytes(prefix: str, source_ids: list[str]) -> bytes:
    rows = []
    local_offset = 0
    for index, source_id in enumerate(source_ids, start=1):
        member = f"{prefix}{source_id}.mp4".encode("utf-8")
        compressed_size = 100 + index
        uncompressed_size = 200 + index
        rows.append(
            struct.pack(
                "<4s6H3L5H2L",
                b"PK\x01\x02",
                20,
                20,
                0,
                0,
                0,
                0,
                index,
                compressed_size,
                uncompressed_size,
                len(member),
                0,
                0,
                0,
                0,
                0,
                local_offset,
            )
            + member
        )
        local_offset += compressed_size
    return b"".join(rows)


def _member_rows(tmp_path: Path, metadata_root: Path) -> list[dict]:
    adapter = DeCoFMetadataAdapter()
    placeholders = adapter.read_records(metadata_root)
    source_ids = sorted(
        record.source_id for record in placeholders if record.is_real
    )
    split_by_source_id = {
        record.source_id: record.split
        for record in placeholders
        if record.is_real
    }
    rows: list[dict] = []
    for archive_name, layout in DECOF_ARCHIVE_LAYOUTS.items():
        central = tmp_path / f"{archive_name}.central"
        central.write_bytes(
            _central_directory_bytes(layout["member_prefix"], source_ids)
        )
        rows.extend(
            parse_decof_zip_central_directory(
                central,
                archive_name=archive_name,
                archive_size=10_000,
                source_ids=source_ids,
                split_by_source_id=split_by_source_id,
            )
        )
    return rows


def test_registry_freezes_official_counts_hashes_and_blocked_license() -> None:
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert registry["repository_revision"] == (
        "5f73e8dc4f8cd9faeed07d767c9517e3f2c66044"
    )
    assert registry["official_split"]["counts"] == {
        "total": 964,
        "train": 771,
        "validation": 96,
        "test": 97,
    }
    hashes = {
        item["relative_path"]: item["sha256"]
        for item in registry["metadata_files"]
    }
    assert hashes == {
        "datas/prompts/data.json": (
            "d9552693cd83bc8fabd90703827272beea32634ca098282beaf082ef5098ddd7"
        ),
        "datas/split/train.json": (
            "246bea61a97e78773e864ec9d58271831617696cac13c0d95401f367184d9b0e"
        ),
        "datas/split/val.json": (
            "f6ef4064377b12fc8e64a876c902ec10e79f8e6001fd703c47f9a6025ca678f1"
        ),
        "datas/split/test.json": (
            "0218c5c5831333483c6152204aff44e9e99295ea3c6ec6118d316f02485a4e49"
        ),
    }
    assert registry["license"]["overall_status"] == "BLOCKED"
    assert registry["small_sample_download_ready"] is False


def test_official_split_audit_is_disjoint_and_exhaustive_in_registry() -> None:
    split = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))[
        "official_split"
    ]
    assert split["status"] == "VERIFIED"
    assert split["random_split_used"] is False
    assert split["duplicate_id_count"] == 0
    assert split["empty_id_count"] == 0
    assert split["train_validation_overlap"] == 0
    assert split["train_test_overlap"] == 0
    assert split["validation_test_overlap"] == 0
    assert split["unassigned_id_count"] == 0
    assert split["unknown_id_count"] == 0


def test_adapter_rejects_duplicate_prompt_id(tmp_path: Path) -> None:
    root = _write_metadata_fixture(tmp_path / "metadata")
    prompt_path = root / "datas/prompts/data.json"
    first = prompt_path.read_text(encoding="utf-8").splitlines()[0]
    with prompt_path.open("a", encoding="utf-8") as handle:
        handle.write(first + "\n")
    with pytest.raises(DeCoFMetadataError, match="duplicate prompt video_id"):
        DeCoFMetadataAdapter().read_records(root)


def test_adapter_rejects_source_id_in_multiple_splits(tmp_path: Path) -> None:
    root = _write_metadata_fixture(tmp_path / "metadata")
    with (root / "datas/split/val.json").open("a", encoding="utf-8") as handle:
        handle.write("train_real_a\n")
    with pytest.raises(DeCoFMetadataError, match="multiple official splits"):
        DeCoFMetadataAdapter().read_records(root)


def test_adapter_rejects_unknown_and_unassigned_split_ids(tmp_path: Path) -> None:
    root = _write_metadata_fixture(tmp_path / "unknown")
    with (root / "datas/split/train.json").open("a", encoding="utf-8") as handle:
        handle.write("unknown\n")
    with pytest.raises(DeCoFMetadataError, match="unknown IDs"):
        DeCoFMetadataAdapter().read_records(root)

    root = _write_metadata_fixture(tmp_path / "unassigned")
    (root / "datas/split/test.json").write_text("", encoding="utf-8")
    with pytest.raises(DeCoFMetadataError, match="absent from official splits"):
        DeCoFMetadataAdapter().read_records(root)


def test_central_directory_requires_frozen_archive_and_layout(
    tmp_path: Path,
) -> None:
    central = tmp_path / "members.bin"
    central.write_bytes(_central_directory_bytes("unknown/", ["source"]))
    with pytest.raises(DeCoFMetadataError, match="unknown DeCoF archive"):
        parse_decof_zip_central_directory(
            central,
            archive_name="unknown.zip",
            archive_size=1000,
            source_ids={"source"},
            split_by_source_id={"source": "train"},
        )
    with pytest.raises(DeCoFMetadataError, match="frozen layout"):
        parse_decof_zip_central_directory(
            central,
            archive_name="T2V.zip",
            archive_size=1000,
            source_ids={"source"},
            split_by_source_id={"source": "train"},
        )


def test_central_directory_maps_every_member_without_media_payload(
    tmp_path: Path,
) -> None:
    source_ids = ["first", "second"]
    central = tmp_path / "members.bin"
    central.write_bytes(_central_directory_bytes("T2V/", source_ids))
    rows = parse_decof_zip_central_directory(
        central,
        archive_name="T2V.zip",
        archive_size=1000,
        source_ids=source_ids,
        split_by_source_id={"first": "train", "second": "test"},
    )
    assert [row["source_id"] for row in rows] == source_ids
    assert all(row["generator"] == "Text2Video-Zero" for row in rows)
    assert all(row["mapping_status"] == "LAYOUT_DEFINED_MAPPING" for row in rows)
    assert all(row["media_payload_bytes"] == 0 for row in rows)
    assert rows[1]["split"] == "test"


def test_central_directory_rejects_duplicate_or_missing_source_id(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.bin"
    duplicate.write_bytes(_central_directory_bytes("Show1/", ["same", "same"]))
    with pytest.raises(DeCoFMetadataError, match="duplicate source ID"):
        parse_decof_zip_central_directory(
            duplicate,
            archive_name="Show1.zip",
            archive_size=1000,
            source_ids={"same"},
            split_by_source_id={"same": "train"},
        )

    missing = tmp_path / "missing.bin"
    missing.write_bytes(_central_directory_bytes("Show1/", ["first"]))
    with pytest.raises(DeCoFMetadataError, match="misses official IDs"):
        parse_decof_zip_central_directory(
            missing,
            archive_name="Show1.zip",
            archive_size=1000,
            source_ids={"first", "second"},
            split_by_source_id={"first": "train", "second": "validation"},
        )


def test_metadata_only_records_keep_null_video_paths_and_official_test(
    tmp_path: Path,
) -> None:
    root = _write_metadata_fixture(tmp_path / "metadata")
    records = DeCoFMetadataAdapter().read_records(
        root,
        member_index=_member_rows(tmp_path, root),
    )
    assert len(records) == 20
    assert all(record.video_path is None for record in records)
    assert all(record.official_split for record in records)
    assert sum(record.split == "test" for record in records) == 5
    assert all(not record.downloadable for record in records)


def test_labels_come_from_pair_role_not_path_or_stem(tmp_path: Path) -> None:
    root = _write_metadata_fixture(tmp_path / "metadata")
    records = DeCoFMetadataAdapter().read_records(
        root,
        member_index=_member_rows(tmp_path, root),
    )
    assert all(record.label == 0 and record.is_real for record in records if record.is_real)
    assert all(
        record.label == 1 and not record.is_real
        for record in records
        if not record.is_real
    )

    broken = _member_rows(tmp_path, root)
    broken[0]["member_path"] = "T2V/name_that_looks_fake.mp4"
    with pytest.raises(DeCoFMetadataError, match="explicit mapping"):
        DeCoFMetadataAdapter().read_records(root, member_index=broken)


def test_generator_comes_only_from_explicit_archive_mapping(tmp_path: Path) -> None:
    root = _write_metadata_fixture(tmp_path / "metadata")
    members = _member_rows(tmp_path, root)
    members[0]["generator"] = "guessed-from-stem"
    with pytest.raises(DeCoFMetadataError, match="frozen mapping"):
        DeCoFMetadataAdapter().read_records(root, member_index=members)


def test_pair_lineage_and_stable_sample_ids_are_preserved(tmp_path: Path) -> None:
    root = _write_metadata_fixture(tmp_path / "metadata")
    members = _member_rows(tmp_path, root)
    adapter = DeCoFMetadataAdapter()
    first = adapter.read_records(root, member_index=members)
    second = adapter.read_records(root, member_index=members)
    assert [record.sample_id for record in first] == [
        record.sample_id for record in second
    ]
    group = [record for record in first if record.source_id == "train_real_a"]
    assert len(group) == 5
    assert {record.pair_source_id for record in group} == {"train_real_a"}
    assert all(
        record.source_lineage["pair_source_id"] == "train_real_a"
        for record in group
    )


def test_missing_real_source_and_archive_member_are_explicit_blockers(
    tmp_path: Path,
) -> None:
    root = _write_metadata_fixture(tmp_path / "metadata")
    records = DeCoFMetadataAdapter().read_records(root)
    test_real = next(
        record
        for record in records
        if record.source_id == "test_real" and record.is_real
    )
    assert test_real.real_source_locator is None
    assert "real_source_locator_missing" in test_real.blocked_reasons
    assert all(
        "archive_member_missing" in record.blocked_reasons
        for record in records
        if not record.is_real
    )


def test_formal_draft_preserves_verified_fields_and_missing_media(
    tmp_path: Path,
) -> None:
    root = _write_metadata_fixture(tmp_path / "metadata")
    record = DeCoFMetadataAdapter().read_records(root)[0]
    draft = record.to_formal_draft()
    assert draft["sample_id"] == record.sample_id
    assert draft["source_id"] == record.source_id
    assert draft["split"] == record.split
    assert draft["video_path"] is None
    assert draft["file_size"] is None
    assert draft["sha256"] is None
    assert draft["formal_schema_ready"] is False
    assert draft["metadata_status"]["temporal_annotation"] == "unresolved_schema"
    assert draft["metadata_status"]["spatial_annotation"] == "unresolved_schema"


def test_summary_does_not_confuse_missing_generator_with_real_not_applicable(
    tmp_path: Path,
) -> None:
    root = _write_metadata_fixture(tmp_path / "metadata")
    records = DeCoFMetadataAdapter().read_records(
        root,
        member_index=_member_rows(tmp_path, root),
    )
    summary = summarize_decof_records(records)
    assert summary["records_total"] == 20
    assert summary["real_records"] == 4
    assert summary["fake_records"] == 16
    assert summary["missing_generator"] == 0
    assert summary["generator_not_applicable_real"] == 4
    assert summary["missing_real_source"] == 1
    assert summary["missing_archive_member"] == 0
    assert summary["downloadable_non_test_records"] == 0


def test_blocked_plan_selects_no_unlicensed_or_test_record(
    tmp_path: Path,
) -> None:
    root = _write_metadata_fixture(tmp_path / "metadata")
    records = DeCoFMetadataAdapter().read_records(
        root,
        member_index=_member_rows(tmp_path, root),
    )
    plan = build_decof_small_sample_plan(
        records,
        repository_revision="frozen",
        license_verified=False,
        real_source_recovery_ready=False,
    )
    assert plan["selected"] == {"train": [], "validation": [], "test": []}
    assert plan["counts"]["test"] == 0
    assert plan["binary_small_sample_plan_ready"] is False
    assert plan["fake_only_engineering_smoke_plan_ready"] is False
    assert plan["small_sample_download_ready"] is False


def test_unblocked_fixture_plan_is_deterministic_balanced_and_test_free(
    tmp_path: Path,
) -> None:
    root = _write_metadata_fixture(tmp_path / "metadata")
    records = DeCoFMetadataAdapter().read_records(
        root,
        member_index=_member_rows(tmp_path, root),
    )
    kwargs = {
        "repository_revision": "frozen",
        "license_verified": True,
        "real_source_recovery_ready": True,
    }
    first = build_decof_small_sample_plan(records, **kwargs)
    second = build_decof_small_sample_plan(records, **kwargs)
    assert first == second
    assert first["counts"] == {"train": 4, "validation": 2, "test": 0}
    for split in ("train", "validation"):
        rows = first["selected"][split]
        assert sum(row["label"] == 0 for row in rows) == len(rows) // 2
        assert sum(row["label"] == 1 for row in rows) == len(rows) // 2
        assert len({row["source_id"] for row in rows}) == len(rows) // 2
    assert first["binary_small_sample_plan_ready"] is True
    assert first["test_exclusion_enforced"] is True


def test_adapter_executes_without_network_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_metadata_fixture(tmp_path / "metadata")

    def reject_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket, "socket", reject_network)
    records = DeCoFMetadataAdapter().read_records(root)
    assert records
    assert DECOF_NETWORK_ACCESS_ALLOWED is False


def test_adapter_does_not_scan_or_read_media(tmp_path: Path) -> None:
    root = _write_metadata_fixture(tmp_path / "metadata")
    media = root / "data/unrelated.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"must-not-be-read")
    before = media.stat()
    records = DeCoFMetadataAdapter().read_records(root)
    after = media.stat()
    assert records
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    assert all(record.video_path is None for record in records)


def test_adapter_and_registry_cannot_claim_production_or_download_readiness() -> None:
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert DECOF_ADAPTER_STATUS == "metadata_only"
    assert DECOF_MEDIA_RESOLUTION_READY is False
    assert DECOF_DATA_DOWNLOADED is False
    assert DECOF_OFFICIAL_SCHEMA_VERIFIED is False
    assert DECOF_PRODUCTION_ADAPTER_READY is False
    assert registry["adapter_status"] == "metadata_only"
    assert registry["data_downloaded"] is False
    assert registry["official_schema_verified"] is False
    assert registry["production_adapter_ready"] is False
    assert registry["media_download_bytes"] == 0


def test_registry_enforces_test_exclusion_and_zero_ready_flags() -> None:
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert registry["test_exclusion_enforced"] is True
    assert registry["official_split"]["test_exclusion_required"] is True
    assert registry["binary_small_sample_plan_ready"] is False
    assert registry["fake_only_engineering_smoke_plan_ready"] is False
    assert registry["small_sample_download_ready"] is False
    assert registry["real_source_probe"]["recovery_status"] == "BLOCKED"
