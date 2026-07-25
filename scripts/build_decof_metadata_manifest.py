#!/usr/bin/env python3
"""Build DeCoF metadata-only audit artifacts without network or media access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from semantic3d.dataset_adapters.decof import (
    DECOF_ARCHIVE_LAYOUTS,
    DeCoFMetadataAdapter,
    build_decof_small_sample_plan,
    parse_decof_zip_central_directory,
    summarize_decof_records,
)

DEFAULT_REGISTRY = "configs/data_registry/decof_metadata_readiness_v1.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a DeCoF formal-manifest draft and download-gate reports from "
            "official text metadata plus pre-fetched ZIP central directories. "
            "This command performs no network requests and never reads media."
        )
    )
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--central-directory",
        action="append",
        default=[],
        metavar="ARCHIVE=PATH",
        help="Exact ZIP central-directory range for one frozen archive.",
    )
    parser.add_argument("--archive-member-index-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--plan-output", required=True)
    parser.add_argument("--checklist-output", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_registry(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DeCoF registry must be a YAML object")
    if payload.get("dataset_id") != "decof":
        raise ValueError("registry dataset_id must be decof")
    if payload.get("media_download_bytes") != 0:
        raise ValueError("registry must preserve zero media bytes")
    if payload.get("production_adapter_ready") is not False:
        raise ValueError("metadata-only registry cannot be production ready")
    if payload.get("data_downloaded") is not False:
        raise ValueError("metadata-only registry cannot claim downloaded data")
    return payload


def _parse_assignments(values: Iterable[str]) -> dict[str, Path]:
    assignments: dict[str, Path] = {}
    for value in values:
        archive_name, separator, raw_path = value.partition("=")
        if not separator or not archive_name or not raw_path:
            raise ValueError(
                "--central-directory must use the form ARCHIVE=PATH"
            )
        if archive_name in assignments:
            raise ValueError(f"duplicate central-directory input: {archive_name}")
        assignments[archive_name] = Path(raw_path).expanduser().resolve()
    return assignments


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def _build_checklist(
    *,
    registry: Mapping[str, Any],
    summary: Mapping[str, Any],
    plan: Mapping[str, Any],
    member_rows: list[dict[str, Any]],
    test_ids: list[str],
) -> str:
    indexed = registry["archive_member_audit"]["indexed_archives"]
    unindexed = registry["archive_member_audit"]["unindexed_archives"]
    archive_lines = [
        f"- `{item['archive_name']}`: {item['archive_size']} bytes"
        for item in [*indexed, *unindexed]
    ]
    range_lines = [
        (
            f"- `{item['archive_name']}`: EOCD `{item['eocd_range']}` "
            f"({item['eocd_received_bytes']} bytes); central directory "
            f"`{item['central_directory_range']}` "
            f"({item['central_directory_received_bytes']} bytes); "
            "`media_payload_bytes=0`."
        )
        for item in indexed
    ]
    return "\n".join(
        [
            "# P4-C3C-A3-B0.2 DeCoF download unlock checklist",
            "",
            "This checklist is generated from frozen official metadata and exact ZIP",
            "central-directory ranges. It is not download authorization.",
            "",
            "```text",
            "DECOF_LICENSE_STATUS=BLOCKED",
            "DECOF_OFFICIAL_SPLIT_VERIFIED=YES",
            "DECOF_METADATA_SCHEMA_VERIFIED=YES",
            "DECOF_NON_TEST_MEMBER_INDEX_READY=YES",
            "DECOF_REAL_SOURCE_RECOVERY_READY=NO",
            "DECOF_METADATA_ADAPTER_READY=YES",
            "DECOF_TEST_EXCLUSION_ENFORCED=YES",
            "DECOF_BINARY_SAMPLE_PLAN_READY=NO",
            "DECOF_FAKE_ONLY_PLAN_READY=NO",
            "SMALL_SAMPLE_DOWNLOAD_READY=NO",
            "REAL_DATA_DRY_RUN_READY=NO",
            "FORMAL_TRAINING_READY=NO",
            "```",
            "",
            "## Evidence",
            "",
            f"- Official prompt IDs: {registry['official_split']['counts']['total']}.",
            (
                "- Official split: train="
                f"{registry['official_split']['counts']['train']}, validation="
                f"{registry['official_split']['counts']['validation']}, test="
                f"{registry['official_split']['counts']['test']}; no overlap, "
                "unknown ID, or unassigned ID."
            ),
            (
                f"- Explicit member-index rows: {len(member_rows)}; missing generated "
                f"members: {summary['missing_archive_member']}."
            ),
            (
                "- All four indexed base-generator archives contain one member for "
                "each of the 964 official source IDs. Generator identity comes from "
                "the frozen archive/prefix table; unknown layouts are rejected."
            ),
            (
                "- The repository has no LICENSE/COPYING file and its GitHub license "
                "field is null. Generated-video, prompts/split, and original-real "
                "source terms are not established."
            ),
            (
                f"- Real rows missing a published locator: "
                f"{summary['missing_real_source']}. Sampled MSR-VTT locator metadata "
                "includes successful and unavailable responses."
            ),
            (
                f"- Current non-test downloadable records: "
                f"{summary['downloadable_non_test_records']}; blocked non-test "
                f"records: {summary['blocked_non_test_records']}."
            ),
            f"- Binary plan selected counts: {plan['counts']}.",
            "- Media payload downloaded in this stage: 0 bytes.",
            "",
            "## Recorded Range requests",
            "",
            *range_lines,
            "",
            "## Absolute prohibitions before remediation",
            "",
            "- No sample ID is currently authorized for media download.",
            "- Every official test ID is excluded:",
            "",
            "```text",
            *test_ids,
            "```",
            "",
            "- Do not download any full archive:",
            "",
            *archive_lines,
            "",
            "## Minimum remaining blockers",
            "",
            "1. Obtain an explicit DeCoF generated-video and metadata license.",
            "2. Verify MSVD/MSR-VTT research-use terms and stable per-ID recovery.",
            "3. Resolve missing/deleted real-source media and obtain media checksums.",
            "4. Obtain or compute archive checksums without weakening test exclusion.",
            "",
            "A3-B1 may not start. The current allowed media byte budget remains zero.",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    metadata_root = Path(args.metadata_root).expanduser().resolve()
    registry = _load_registry(Path(args.registry).expanduser().resolve())
    central_directories = _parse_assignments(args.central_directory)

    indexed_archive_entries = {
        item["archive_name"]: item
        for item in registry["archive_member_audit"]["indexed_archives"]
    }
    if set(central_directories) != set(indexed_archive_entries):
        raise ValueError(
            "central-directory inputs must exactly match the four indexed archives"
        )
    if set(central_directories) != set(DECOF_ARCHIVE_LAYOUTS):
        raise ValueError("registry and adapter archive mappings disagree")

    adapter = DeCoFMetadataAdapter()
    placeholder_records = adapter.read_records(metadata_root)
    source_ids = {
        record.source_id for record in placeholder_records if record.is_real
    }
    split_by_source_id = {
        record.source_id: record.split
        for record in placeholder_records
        if record.is_real
    }

    member_rows: list[dict[str, Any]] = []
    for archive_name in sorted(central_directories):
        item = indexed_archive_entries[archive_name]
        path = central_directories[archive_name]
        if not path.is_file():
            raise FileNotFoundError(f"central directory is missing: {path}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != item["central_directory_sha256"]:
            raise ValueError(f"{archive_name}: central-directory checksum mismatch")
        parsed = parse_decof_zip_central_directory(
            path,
            archive_name=archive_name,
            archive_size=int(item["archive_size"]),
            source_ids=source_ids,
            split_by_source_id=split_by_source_id,
        )
        if len(parsed) != int(item["member_count"]):
            raise ValueError(f"{archive_name}: member count mismatch")
        member_rows.extend(parsed)

    member_rows.sort(
        key=lambda row: (
            row["source_id"],
            row["generator"],
            row["member_path"],
        )
    )
    member_index_path = Path(args.archive_member_index_output).expanduser().resolve()
    _write_jsonl(member_index_path, member_rows)

    records = adapter.read_records(metadata_root, member_index=member_rows)
    summary = summarize_decof_records(records)
    summary.update(
        {
            "official_source_id_count": len(source_ids),
            "archive_member_index_records": len(member_rows),
            "non_test_archive_member_records": sum(
                row["split"] != "test" for row in member_rows
            ),
            "test_archive_member_records": sum(
                row["split"] == "test" for row in member_rows
            ),
            "archive_member_index_sha256": _sha256(member_index_path),
            "official_split_verified": True,
            "metadata_schema_verified": True,
            "non_test_member_index_ready": True,
            "real_source_recovery_ready": False,
            "test_exclusion_enforced": True,
            "official_schema_verified": False,
            "real_data_dry_run_ready": False,
            "formal_training_ready": False,
        }
    )

    plan = build_decof_small_sample_plan(
        records,
        repository_revision=str(registry["repository_revision"]),
        license_verified=registry["license"]["overall_status"] == "VERIFIED",
        real_source_recovery_ready=(
            registry["real_source_probe"]["recovery_status"] == "VERIFIED"
        ),
    )

    manifest_path = Path(args.manifest_output).expanduser().resolve()
    _write_jsonl(
        manifest_path,
        (record.to_formal_draft() for record in records),
    )
    summary["formal_manifest_sha256"] = _sha256(manifest_path)
    _write_json(
        Path(args.summary_output).expanduser().resolve(),
        summary,
    )
    _write_json(
        Path(args.plan_output).expanduser().resolve(),
        plan,
    )

    test_ids = sorted(
        record.source_id
        for record in records
        if record.is_real and record.split == "test"
    )
    checklist = _build_checklist(
        registry=registry,
        summary=summary,
        plan=plan,
        member_rows=member_rows,
        test_ids=test_ids,
    )
    checklist_path = Path(args.checklist_output).expanduser().resolve()
    checklist_path.parent.mkdir(parents=True, exist_ok=True)
    checklist_path.write_text(checklist, encoding="utf-8")

    print(
        json.dumps(
            {
                "records_total": summary["records_total"],
                "archive_member_index_records": len(member_rows),
                "media_download_bytes": 0,
                "binary_small_sample_plan_ready": plan[
                    "binary_small_sample_plan_ready"
                ],
                "fake_only_engineering_smoke_plan_ready": plan[
                    "fake_only_engineering_smoke_plan_ready"
                ],
                "small_sample_download_ready": plan[
                    "small_sample_download_ready"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
