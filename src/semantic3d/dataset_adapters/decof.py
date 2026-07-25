"""Metadata-only DeCoF adapter with explicit official-layout validation.

The adapter never opens video files and never performs network requests.  It
keeps unresolved media, license, and real-source state explicit until a later
stage supplies verified assets.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from semantic3d.dataset_builder.formal_schema import (
    FORMAL_SAMPLE_SCHEMA_VERSION,
    OPTIONAL_METADATA_FIELDS,
)
from semantic3d.dataset_builder.ids import stable_id

DECOF_DATASET_ID = "decof"
DECOF_DATASET_NAME = "DeCoF/GVF"
DECOF_ADAPTER_STATUS = "metadata_only"
DECOF_PRODUCTION_ADAPTER_READY = False
DECOF_MEDIA_RESOLUTION_READY = False
DECOF_DATA_DOWNLOADED = False
DECOF_OFFICIAL_SCHEMA_VERIFIED = False
DECOF_NETWORK_ACCESS_ALLOWED = False

DECOF_SPLIT_FILES = {
    "train": "datas/split/train.json",
    "validation": "datas/split/val.json",
    "test": "datas/split/test.json",
}

# These mappings are frozen from the official Google Drive archive names and
# the corresponding generator names in the official README/paper.  Unknown
# archives and member prefixes are rejected instead of inferred.
DECOF_ARCHIVE_LAYOUTS: Mapping[str, Mapping[str, str]] = {
    "ModelScopeT2V.zip": {
        "generator": "ModelScopeT2V",
        "member_prefix": "ModelScopeT2V/",
    },
    "Show1.zip": {
        "generator": "Show-1",
        "member_prefix": "Show1/",
    },
    "T2V.zip": {
        "generator": "Text2Video-Zero",
        "member_prefix": "T2V/",
    },
    "zeroscope.zip": {
        "generator": "ZeroScope",
        "member_prefix": "zeroscope/",
    },
}
DECOF_GENERATORS = tuple(
    layout["generator"] for layout in DECOF_ARCHIVE_LAYOUTS.values()
)

ZIP_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"
ZIP_CENTRAL_DIRECTORY_HEADER = struct.Struct("<4s6H3L5H2L")


class DeCoFMetadataError(ValueError):
    """Raised when official metadata or a frozen member index is inconsistent."""


@dataclass(frozen=True)
class DeCoFMetadataRecord:
    """One real or generated DeCoF media role before media acquisition."""

    sample_id: str
    source_id: str
    split: str
    label: int
    is_real: bool
    generator: str | None
    prompt: str
    source_dataset: str
    source_lineage: Mapping[str, Any]
    pair_source_id: str
    archive_name: str | None
    video_member: str | None
    video_path: None
    real_source_locator: str | None
    metadata_status: Mapping[str, str]
    temporal_annotation: None
    spatial_annotation: None
    official_split: bool
    archive_member_located: bool
    real_source_metadata_locator_present: bool
    media_downloaded: bool
    downloadable: bool
    blocked_reasons: tuple[str, ...]
    adapter_status: str = DECOF_ADAPTER_STATUS

    def __post_init__(self) -> None:
        if not self.sample_id or not self.source_id or not self.prompt:
            raise DeCoFMetadataError("sample_id, source_id, and prompt must be non-empty")
        if self.split not in DECOF_SPLIT_FILES:
            raise DeCoFMetadataError(f"unsupported official split: {self.split!r}")
        if self.label not in {0, 1} or self.is_real != (self.label == 0):
            raise DeCoFMetadataError("label and is_real must use the project convention")
        if self.is_real and self.generator is not None:
            raise DeCoFMetadataError("real records cannot have a generator")
        if not self.is_real and self.generator not in DECOF_GENERATORS:
            raise DeCoFMetadataError("generated records require a frozen generator mapping")
        if self.video_path is not None:
            raise DeCoFMetadataError("metadata-only records must keep video_path null")
        if self.media_downloaded or self.downloadable:
            raise DeCoFMetadataError("B0.2 records cannot be marked downloaded or downloadable")
        if not self.blocked_reasons:
            raise DeCoFMetadataError("blocked metadata-only records need explicit reasons")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable record."""

        payload = asdict(self)
        payload["source_lineage"] = dict(self.source_lineage)
        payload["metadata_status"] = dict(self.metadata_status)
        payload["blocked_reasons"] = list(self.blocked_reasons)
        return payload

    def to_formal_draft(self) -> dict[str, Any]:
        """Map verified metadata to the A1 field names without forging media fields."""

        return {
            "sample_id": self.sample_id,
            "video_path": None,
            "label": self.label,
            "split": self.split,
            "source_dataset": self.source_dataset,
            "source_id": self.source_id,
            "source_lineage": dict(self.source_lineage),
            "generator": self.generator,
            "is_real": self.is_real,
            "duration": None,
            "fps": None,
            "frame_count": None,
            "width": None,
            "height": None,
            "file_size": None,
            "sha256": None,
            "temporal_annotation": None,
            "spatial_annotation": None,
            "metadata_status": dict(self.metadata_status),
            "source_path": None,
            "path_mode": None,
            "official_split": True,
            "schema_version": FORMAL_SAMPLE_SCHEMA_VERSION,
            "formal_schema_ready": False,
            "formal_schema_blockers": [
                "video_path_missing",
                "file_size_missing",
                "sha256_missing",
                *self.blocked_reasons,
            ],
            "archive_name": self.archive_name,
            "video_member": self.video_member,
            "pair_source_id": self.pair_source_id,
            "adapter_status": self.adapter_status,
            "media_downloaded": False,
        }


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise DeCoFMetadataError(f"{path}: row {line_number} is not an object")
        records.append(value)
    return records


def _read_split_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if any(not value for value in values):
        raise DeCoFMetadataError(f"{path}: empty official split ID")
    duplicates = sorted(value for value in set(values) if values.count(value) > 1)
    if duplicates:
        raise DeCoFMetadataError(f"{path}: duplicate official split IDs: {duplicates[:3]}")
    return values


def load_decof_member_index(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Load an explicit JSONL member-to-source mapping produced by the audit."""

    return _read_jsonl_objects(Path(path).expanduser().resolve())


def parse_decof_zip_central_directory(
    path: str | Path,
    *,
    archive_name: str,
    archive_size: int,
    source_ids: Iterable[str],
    split_by_source_id: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Parse a ZIP central-directory range without reading archive media payload."""

    if archive_name not in DECOF_ARCHIVE_LAYOUTS:
        raise DeCoFMetadataError(f"unknown DeCoF archive: {archive_name}")
    if archive_size <= 0:
        raise DeCoFMetadataError("archive_size must be positive")

    layout = DECOF_ARCHIVE_LAYOUTS[archive_name]
    prefix = layout["member_prefix"]
    generator = layout["generator"]
    expected_ids = set(source_ids)
    data = Path(path).expanduser().resolve().read_bytes()
    central_directory_sha256 = hashlib.sha256(data).hexdigest()
    rows: list[dict[str, Any]] = []
    offset = 0

    while offset < len(data):
        if len(data) - offset < ZIP_CENTRAL_DIRECTORY_HEADER.size:
            raise DeCoFMetadataError(
                f"{archive_name}: truncated central-directory header at byte {offset}"
            )
        fields = ZIP_CENTRAL_DIRECTORY_HEADER.unpack_from(data, offset)
        (
            signature,
            _version_made,
            _version_needed,
            flags,
            compression_method,
            _modified_time,
            _modified_date,
            crc32,
            compressed_size,
            uncompressed_size,
            filename_length,
            extra_length,
            comment_length,
            _disk_number,
            _internal_attributes,
            _external_attributes,
            local_header_offset,
        ) = fields
        if signature != ZIP_CENTRAL_DIRECTORY_SIGNATURE:
            raise DeCoFMetadataError(
                f"{archive_name}: unexpected central-directory signature at byte {offset}"
            )
        variable_start = offset + ZIP_CENTRAL_DIRECTORY_HEADER.size
        variable_end = (
            variable_start + filename_length + extra_length + comment_length
        )
        if variable_end > len(data):
            raise DeCoFMetadataError(
                f"{archive_name}: truncated central-directory entry at byte {offset}"
            )
        encoding = "utf-8" if flags & 0x800 else "cp437"
        member_path = data[
            variable_start : variable_start + filename_length
        ].decode(encoding)
        pure_path = PurePosixPath(member_path)
        if (
            not member_path.startswith(prefix)
            or pure_path.parent.as_posix() != prefix.rstrip("/")
            or pure_path.suffix.lower() != ".mp4"
        ):
            raise DeCoFMetadataError(
                f"{archive_name}: member violates frozen layout: {member_path!r}"
            )
        source_id = pure_path.stem
        if source_id not in expected_ids:
            raise DeCoFMetadataError(
                f"{archive_name}: unknown member source_id: {source_id!r}"
            )
        rows.append(
            {
                "archive_name": archive_name,
                "archive_size": archive_size,
                "archive_checksum": None,
                "archive_checksum_status": "NOT_AVAILABLE",
                "central_directory_sha256": central_directory_sha256,
                "member_path": member_path,
                "member_compressed_size": compressed_size,
                "member_uncompressed_size": uncompressed_size,
                "member_crc32": f"{crc32:08x}",
                "member_compression_method": compression_method,
                "member_local_header_offset": local_header_offset,
                "source_id": source_id,
                "generator": generator,
                "split": split_by_source_id[source_id],
                "mapping_status": "LAYOUT_DEFINED_MAPPING",
                "media_payload_bytes": 0,
            }
        )
        offset = variable_end

    indexed_ids = [row["source_id"] for row in rows]
    if len(indexed_ids) != len(set(indexed_ids)):
        raise DeCoFMetadataError(f"{archive_name}: duplicate source ID in member index")
    missing_ids = sorted(expected_ids - set(indexed_ids))
    if missing_ids:
        raise DeCoFMetadataError(
            f"{archive_name}: central directory misses official IDs: {missing_ids[:3]}"
        )
    if set(indexed_ids) != expected_ids:
        raise DeCoFMetadataError(f"{archive_name}: member ID set is not exhaustive")
    return rows


def _formal_metadata_status(*, is_real: bool) -> dict[str, str]:
    status = {
        name: "unresolved_schema" for name in OPTIONAL_METADATA_FIELDS
    }
    status.update(
        {
            "label": "derived",
            "split": "provided",
            "source_id": "provided",
            "source_lineage": "provided",
            "generator": "missing" if is_real else "provided",
            "source_domain": "provided",
            "is_real": "derived",
            "temporal_annotation": "unresolved_schema",
            "spatial_annotation": "unresolved_schema",
        }
    )
    return status


class DeCoFMetadataAdapter:
    """Join official prompts, official splits, and an explicit member index."""

    dataset_id = DECOF_DATASET_ID
    dataset_name = DECOF_DATASET_NAME
    adapter_status = DECOF_ADAPTER_STATUS
    production_adapter_ready = DECOF_PRODUCTION_ADAPTER_READY
    media_resolution_ready = DECOF_MEDIA_RESOLUTION_READY
    data_downloaded = DECOF_DATA_DOWNLOADED
    official_schema_verified = DECOF_OFFICIAL_SCHEMA_VERIFIED
    network_access_allowed = DECOF_NETWORK_ACCESS_ALLOWED

    def read_records(
        self,
        metadata_root: str | Path,
        *,
        member_index: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[DeCoFMetadataRecord]:
        """Read metadata only; no directory scan, network request, or video read."""

        root = Path(metadata_root).expanduser().resolve()
        prompts_path = root / "datas/prompts/data.json"
        if not prompts_path.is_file():
            raise FileNotFoundError(f"DeCoF prompts metadata is missing: {prompts_path}")
        prompts = _read_jsonl_objects(prompts_path)

        prompt_by_id: dict[str, dict[str, Any]] = {}
        for row_number, prompt in enumerate(prompts, start=1):
            source_id = prompt.get("video_id")
            if not isinstance(source_id, str) or not source_id.strip():
                raise DeCoFMetadataError(
                    f"prompts row {row_number} has an empty video_id"
                )
            if source_id in prompt_by_id:
                raise DeCoFMetadataError(f"duplicate prompt video_id: {source_id}")
            if not isinstance(prompt.get("prompt"), str) or not prompt["prompt"].strip():
                raise DeCoFMetadataError(f"{source_id}: prompt is missing")
            if prompt.get("source") not in {"MSVD", "MSRVTT"}:
                raise DeCoFMetadataError(f"{source_id}: unsupported real source")
            prompt_by_id[source_id] = prompt

        split_by_source_id: dict[str, str] = {}
        for split, relative_path in DECOF_SPLIT_FILES.items():
            split_path = root / relative_path
            if not split_path.is_file():
                raise FileNotFoundError(f"DeCoF official split is missing: {split_path}")
            for source_id in _read_split_ids(split_path):
                if source_id in split_by_source_id:
                    raise DeCoFMetadataError(
                        f"source ID appears in multiple official splits: {source_id}"
                    )
                split_by_source_id[source_id] = split

        prompt_ids = set(prompt_by_id)
        split_ids = set(split_by_source_id)
        unknown = sorted(split_ids - prompt_ids)
        unassigned = sorted(prompt_ids - split_ids)
        if unknown:
            raise DeCoFMetadataError(f"official splits contain unknown IDs: {unknown[:3]}")
        if unassigned:
            raise DeCoFMetadataError(
                f"prompt IDs are absent from official splits: {unassigned[:3]}"
            )

        member_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for raw_member in member_index or ():
            member = dict(raw_member)
            archive_name = member.get("archive_name")
            if archive_name not in DECOF_ARCHIVE_LAYOUTS:
                raise DeCoFMetadataError(f"unknown member archive: {archive_name!r}")
            layout = DECOF_ARCHIVE_LAYOUTS[str(archive_name)]
            generator = member.get("generator")
            if generator != layout["generator"]:
                raise DeCoFMetadataError(
                    f"{archive_name}: generator does not match frozen mapping"
                )
            source_id = member.get("source_id")
            if source_id not in prompt_by_id:
                raise DeCoFMetadataError(f"member references unknown source_id: {source_id!r}")
            if member.get("split") != split_by_source_id[source_id]:
                raise DeCoFMetadataError(f"{source_id}: member split mismatch")
            member_path = member.get("member_path")
            if not isinstance(member_path, str) or not member_path:
                raise DeCoFMetadataError(f"{source_id}: member_path is missing")
            expected_path = (
                f"{layout['member_prefix']}{source_id}.mp4"
            )
            if member_path != expected_path:
                raise DeCoFMetadataError(
                    f"{source_id}: member path is not the frozen explicit mapping"
                )
            pair = (source_id, str(generator))
            if pair in member_by_pair:
                raise DeCoFMetadataError(
                    f"duplicate generated member mapping: {pair}"
                )
            member_by_pair[pair] = member

        records: list[DeCoFMetadataRecord] = []
        for source_id in sorted(prompt_by_id):
            prompt = prompt_by_id[source_id]
            split = split_by_source_id[source_id]
            source_domain = str(prompt["source"])
            locator = prompt.get("video_url")
            if locator in {"", None}:
                locator = None
            lineage = {
                "original_dataset": source_domain,
                "original_video_id": source_id,
                "prompt_video_url": locator,
                "pair_source_id": source_id,
                "official_repository_dataset_id": DECOF_DATASET_ID,
            }
            real_blockers = [
                "decof_data_license_unverified",
                "real_source_terms_unverified",
                "media_not_downloaded",
            ]
            if locator is None:
                real_blockers.append("real_source_locator_missing")
            records.append(
                DeCoFMetadataRecord(
                    sample_id=stable_id(
                        "decof_metadata_record",
                        source_id,
                        "real",
                        prefix="decof",
                    ),
                    source_id=source_id,
                    split=split,
                    label=0,
                    is_real=True,
                    generator=None,
                    prompt=str(prompt["prompt"]),
                    source_dataset=DECOF_DATASET_ID,
                    source_lineage=lineage,
                    pair_source_id=source_id,
                    archive_name=None,
                    video_member=None,
                    video_path=None,
                    real_source_locator=locator,
                    metadata_status=_formal_metadata_status(is_real=True),
                    temporal_annotation=None,
                    spatial_annotation=None,
                    official_split=True,
                    archive_member_located=False,
                    real_source_metadata_locator_present=locator is not None,
                    media_downloaded=False,
                    downloadable=False,
                    blocked_reasons=tuple(real_blockers),
                )
            )

            for archive_name, layout in DECOF_ARCHIVE_LAYOUTS.items():
                generator = layout["generator"]
                member = member_by_pair.get((source_id, generator))
                video_member = (
                    None if member is None else str(member["member_path"])
                )
                fake_blockers = [
                    "decof_data_license_unverified",
                    "archive_checksum_unavailable",
                    "media_not_downloaded",
                ]
                if member is None:
                    fake_blockers.append("archive_member_missing")
                generated_lineage = {
                    **lineage,
                    "generator": generator,
                    "archive_name": archive_name,
                    "paired_real_source_id": source_id,
                }
                records.append(
                    DeCoFMetadataRecord(
                        sample_id=stable_id(
                            "decof_metadata_record",
                            source_id,
                            "fake",
                            generator,
                            prefix="decof",
                        ),
                        source_id=source_id,
                        split=split,
                        label=1,
                        is_real=False,
                        generator=generator,
                        prompt=str(prompt["prompt"]),
                        source_dataset=DECOF_DATASET_ID,
                        source_lineage=generated_lineage,
                        pair_source_id=source_id,
                        archive_name=archive_name,
                        video_member=video_member,
                        video_path=None,
                        real_source_locator=locator,
                        metadata_status=_formal_metadata_status(is_real=False),
                        temporal_annotation=None,
                        spatial_annotation=None,
                        official_split=True,
                        archive_member_located=member is not None,
                        real_source_metadata_locator_present=locator is not None,
                        media_downloaded=False,
                        downloadable=False,
                        blocked_reasons=tuple(fake_blockers),
                    )
                )

        sample_ids = [record.sample_id for record in records]
        if len(sample_ids) != len(set(sample_ids)):
            raise DeCoFMetadataError("duplicate metadata-only sample_id")
        return records


def summarize_decof_records(
    records: Sequence[DeCoFMetadataRecord],
) -> dict[str, Any]:
    """Summarize one metadata-only manifest without treating null as negative."""

    sample_ids = [record.sample_id for record in records]
    split_by_source: dict[str, str] = {}
    split_overlap_count = 0
    for record in records:
        previous = split_by_source.setdefault(record.source_id, record.split)
        if previous != record.split:
            split_overlap_count += 1
    generator_counts = {
        generator: sum(record.generator == generator for record in records)
        for generator in DECOF_GENERATORS
    }
    return {
        "records_total": len(records),
        "train_records": sum(record.split == "train" for record in records),
        "validation_records": sum(
            record.split == "validation" for record in records
        ),
        "test_records": sum(record.split == "test" for record in records),
        "real_records": sum(record.is_real for record in records),
        "fake_records": sum(not record.is_real for record in records),
        "generator_counts": generator_counts,
        "missing_generator": sum(
            not record.is_real and record.generator is None for record in records
        ),
        "generator_not_applicable_real": sum(record.is_real for record in records),
        "missing_real_source": sum(
            record.is_real and not record.real_source_metadata_locator_present
            for record in records
        ),
        "missing_archive_member": sum(
            not record.is_real and not record.archive_member_located
            for record in records
        ),
        "downloadable_non_test_records": sum(
            record.split != "test" and record.downloadable for record in records
        ),
        "blocked_non_test_records": sum(
            record.split != "test" and not record.downloadable for record in records
        ),
        "duplicate_sample_ids": len(sample_ids) - len(set(sample_ids)),
        "split_overlap_count": split_overlap_count,
        "media_download_bytes": 0,
        "adapter_status": DECOF_ADAPTER_STATUS,
        "production_adapter_ready": False,
        "data_downloaded": False,
    }


def build_decof_small_sample_plan(
    records: Sequence[DeCoFMetadataRecord],
    *,
    repository_revision: str,
    license_verified: bool,
    real_source_recovery_ready: bool,
    train_limit: int = 32,
    validation_limit: int = 16,
) -> dict[str, Any]:
    """Build a deterministic balanced plan only when all download gates are open."""

    if train_limit < 0 or validation_limit < 0:
        raise ValueError("sample limits must be non-negative")

    selected: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    global_blockers = []
    if not license_verified:
        global_blockers.append("decof_data_license_unverified")
    if not real_source_recovery_ready:
        global_blockers.append("real_source_recovery_not_ready")

    def source_sort_key(split: str, source_id: str) -> str:
        payload = f"{repository_revision}:{split}:{source_id}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    records_by_source: dict[tuple[str, str], list[DeCoFMetadataRecord]] = {}
    for record in records:
        records_by_source.setdefault((record.split, record.source_id), []).append(record)

    if not global_blockers:
        for split, limit in (("train", train_limit), ("validation", validation_limit)):
            pair_limit = limit // 2
            candidates: list[
                tuple[str, DeCoFMetadataRecord, list[DeCoFMetadataRecord]]
            ] = []
            for (candidate_split, source_id), grouped in records_by_source.items():
                if candidate_split != split:
                    continue
                real = next((record for record in grouped if record.is_real), None)
                fakes = sorted(
                    (
                        record
                        for record in grouped
                        if not record.is_real and record.archive_member_located
                    ),
                    key=lambda record: (
                        record.generator or "",
                        record.sample_id,
                    ),
                )
                if (
                    real is None
                    or not real.real_source_metadata_locator_present
                    or not fakes
                ):
                    continue
                candidates.append((source_id, real, fakes))
            candidates.sort(key=lambda item: source_sort_key(split, item[0]))

            generator_use = {generator: 0 for generator in DECOF_GENERATORS}
            for source_id, real, fakes in candidates[:pair_limit]:
                fake = min(
                    fakes,
                    key=lambda record: (
                        generator_use[record.generator or ""],
                        source_sort_key(split, record.sample_id),
                    ),
                )
                assert fake.generator is not None
                generator_use[fake.generator] += 1
                selected[split].extend(
                    [
                        {
                            "sample_id": real.sample_id,
                            "source_id": source_id,
                            "split": split,
                            "label": 0,
                            "is_real": True,
                            "generator": None,
                            "archive_name": None,
                            "video_member": None,
                        },
                        {
                            "sample_id": fake.sample_id,
                            "source_id": source_id,
                            "split": split,
                            "label": 1,
                            "is_real": False,
                            "generator": fake.generator,
                            "archive_name": fake.archive_name,
                            "video_member": fake.video_member,
                        },
                    ]
                )

    binary_ready = (
        not global_blockers
        and bool(selected["train"])
        and bool(selected["validation"])
        and all(len(selected[split]) % 2 == 0 for split in ("train", "validation"))
    )
    fake_only_ready = license_verified and any(
        record.split != "test"
        and not record.is_real
        and record.archive_member_located
        for record in records
    )
    return {
        "schema_version": "semantic3d_decof_small_sample_plan_v1",
        "selection_rule": (
            "Sort source IDs by SHA-256(repository_revision:split:source_id); "
            "select balanced real/fake pairs and choose the least-used available "
            "generator with a SHA-256 tie-break."
        ),
        "seed": None,
        "repository_revision": repository_revision,
        "target_limits": {"train": train_limit, "validation": validation_limit, "test": 0},
        "selected": selected,
        "counts": {
            split: len(values) for split, values in selected.items()
        },
        "binary_small_sample_plan_ready": binary_ready,
        "fake_only_engineering_smoke_plan_ready": fake_only_ready,
        "small_sample_download_ready": binary_ready,
        "blocking_items": global_blockers,
        "test_exclusion_enforced": not selected["test"],
        "media_download_bytes": 0,
    }
