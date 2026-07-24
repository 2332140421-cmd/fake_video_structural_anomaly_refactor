"""GenVideo-100K adapter skeleton with no assumed official field names."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from semantic3d.dataset_builder.formal_schema import FormalVideoSample
from semantic3d.dataset_builder.manifest import build_formal_video_sample

from .base import DatasetAdapter, UnresolvedDatasetSchemaError

GENVIDEO100K_SCHEMA_STATUS = "TODO/UNRESOLVED_SCHEMA"


@dataclass(frozen=True)
class GenVideo100KFieldMapping:
    """Caller-supplied mapping to be frozen after official schema verification."""

    video_path: str
    label: str | None = None
    split: str | None = None
    source_id: str | None = None
    source_lineage: str | None = None
    generator: str | None = None
    source_domain: str | None = None
    checksum: str | None = None
    temporal_annotation: str | None = None
    spatial_annotation: str | None = None
    label_values: Mapping[str, int] = field(default_factory=dict)
    split_values: Mapping[str, str] = field(default_factory=dict)
    json_records_key: str | None = None

    def __post_init__(self) -> None:
        if not self.video_path.strip():
            raise ValueError("An explicit video_path field mapping is required")
        invalid_labels = {
            value for value in self.label_values.values() if value not in {0, 1}
        }
        if invalid_labels:
            raise ValueError("label_values may map only to canonical labels 0 or 1")


def _records_from_file(
    metadata_path: Path,
    *,
    json_records_key: str | None,
) -> list[Mapping[str, Any]]:
    suffix = metadata_path.suffix.lower()
    if suffix == ".csv":
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".jsonl":
        records = []
        for line_number, line in enumerate(
            metadata_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL row {line_number} is not an object")
            records.append(value)
        return records
    if suffix == ".json":
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        if json_records_key is not None:
            if not isinstance(payload, Mapping) or json_records_key not in payload:
                raise ValueError(f"Configured JSON records key is missing: {json_records_key}")
            payload = payload[json_records_key]
        if not isinstance(payload, list) or not all(
            isinstance(value, Mapping) for value in payload
        ):
            raise ValueError("JSON metadata must be a list of objects")
        return payload
    raise ValueError("Metadata must be .csv, .jsonl, or .json")


def _mapped_value(
    record: Mapping[str, Any],
    field_name: str | None,
) -> tuple[Any | None, str]:
    if field_name is None:
        return None, "unresolved_schema"
    if field_name not in record:
        return None, "missing"
    value = record[field_name]
    if value is None or value == "":
        return None, "missing"
    return value, "provided"


class GenVideo100KAdapter(DatasetAdapter):
    """Map future verified metadata; never infer today's unknown official layout."""

    dataset_name = "GenVideo-100K"
    adapter_status = "skeleton"
    official_schema_verified = False

    def __init__(self, field_mapping: GenVideo100KFieldMapping | None = None) -> None:
        self.field_mapping = field_mapping

    def read_samples(
        self,
        metadata_path: str | Path,
        *,
        data_root: str | Path,
    ) -> list[FormalVideoSample]:
        if self.field_mapping is None:
            raise UnresolvedDatasetSchemaError(
                f"{GENVIDEO100K_SCHEMA_STATUS}: supply an explicit verified field mapping"
            )
        metadata = Path(metadata_path).expanduser().resolve()
        if not metadata.is_file():
            raise FileNotFoundError(f"Metadata file does not exist: {metadata}")
        mapping = self.field_mapping
        records = _records_from_file(
            metadata,
            json_records_key=mapping.json_records_key,
        )
        return [
            self._map_record(record, data_root=data_root, row_number=index)
            for index, record in enumerate(records, start=1)
        ]

    def _map_record(
        self,
        record: Mapping[str, Any],
        *,
        data_root: str | Path,
        row_number: int,
    ) -> FormalVideoSample:
        mapping = self.field_mapping
        assert mapping is not None
        if mapping.video_path not in record:
            raise ValueError(f"Metadata row {row_number} has no mapped video path")
        video_path = record[mapping.video_path]
        if video_path is None or video_path == "":
            raise ValueError(f"Metadata row {row_number} has no mapped video path")

        raw_label, label_status = _mapped_value(record, mapping.label)
        if raw_label is None:
            label = None
        else:
            key = str(raw_label)
            if key not in mapping.label_values:
                raise ValueError(f"Unmapped label value in metadata row {row_number}: {key!r}")
            label = int(mapping.label_values[key])

        raw_split, split_status = _mapped_value(record, mapping.split)
        if raw_split is None:
            split = None
        else:
            key = str(raw_split)
            split = mapping.split_values.get(key, key)

        source_id, source_id_status = _mapped_value(record, mapping.source_id)
        lineage, lineage_status = _mapped_value(record, mapping.source_lineage)
        if lineage is not None and not isinstance(lineage, Mapping):
            raise ValueError(f"source_lineage must be an object in metadata row {row_number}")
        generator, generator_status = _mapped_value(record, mapping.generator)
        source_domain, source_domain_status = _mapped_value(record, mapping.source_domain)
        checksum, checksum_status = _mapped_value(record, mapping.checksum)
        temporal, temporal_status = _mapped_value(record, mapping.temporal_annotation)
        spatial, spatial_status = _mapped_value(record, mapping.spatial_annotation)

        status = {
            "label": label_status,
            "split": split_status,
            "source_id": source_id_status if source_id is not None else "derived",
            "source_lineage": lineage_status,
            "generator": generator_status,
            "source_domain": source_domain_status,
            "is_real": "derived" if label is not None else "unresolved_schema",
            "temporal_annotation": temporal_status,
            "spatial_annotation": spatial_status,
            "official_checksum": checksum_status,
            "official_schema": "unresolved_schema",
        }
        return build_formal_video_sample(
            str(video_path),
            data_root=data_root,
            source_dataset=self.dataset_name,
            split=split,
            label=label,
            source_id=None if source_id is None else str(source_id),
            source_lineage=None if lineage is None else dict(lineage),
            generator=None if generator is None else str(generator),
            source_domain=None if source_domain is None else str(source_domain),
            temporal_annotation=temporal,
            spatial_annotation=spatial,
            official_split=split is not None,
            expected_sha256=None if checksum is None else str(checksum),
            path_mode="data_root_relative",
            metadata_status=status,
        )
