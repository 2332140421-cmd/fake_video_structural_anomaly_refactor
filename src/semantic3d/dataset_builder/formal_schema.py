"""Canonical sample contract for formal video-dataset onboarding.

This contract is intentionally independent of any one public dataset.  Dataset
adapters must map their verified metadata into these fields without guessing
unknown source columns or directory layouts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

FORMAL_SAMPLE_SCHEMA_VERSION = "semantic3d_formal_video_sample_v1"
FORMAL_SPLITS = frozenset({"train", "validation", "test"})
METADATA_STATES = frozenset({"provided", "derived", "missing", "unresolved_schema"})
OPTIONAL_METADATA_FIELDS = (
    "label",
    "split",
    "source_id",
    "source_lineage",
    "generator",
    "source_domain",
    "is_real",
    "duration",
    "fps",
    "frame_count",
    "width",
    "height",
    "temporal_annotation",
    "spatial_annotation",
)


@dataclass(frozen=True)
class FormalVideoSample:
    """One source video and its explicit metadata/missingness state."""

    sample_id: str
    video_path: str
    label: int | None
    split: str | None
    source_dataset: str
    source_id: str | None
    source_lineage: Mapping[str, Any] | None
    generator: str | None
    is_real: bool | None
    duration: float | None
    fps: float | None
    frame_count: int | None
    width: int | None
    height: int | None
    file_size: int
    sha256: str
    temporal_annotation: Any | None
    spatial_annotation: Any | None
    metadata_status: Mapping[str, str]
    source_path: str
    path_mode: str
    source_domain: str | None = None
    official_split: bool = False
    schema_version: str = FORMAL_SAMPLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        if not self.video_path.strip():
            raise ValueError("video_path must be non-empty")
        if not self.source_dataset.strip():
            raise ValueError("source_dataset must be non-empty")
        if self.label not in {0, 1, None}:
            raise ValueError("label must be 0, 1, or None")
        if self.split is not None and self.split not in FORMAL_SPLITS:
            raise ValueError(f"split must be one of {sorted(FORMAL_SPLITS)} or None")
        if self.path_mode not in {"absolute", "data_root_relative"}:
            raise ValueError("path_mode must be absolute or data_root_relative")
        if self.file_size < 0:
            raise ValueError("file_size must be non-negative")
        digest = self.sha256.lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("sha256 must be a 64-character hexadecimal digest")
        if self.duration is not None and self.duration < 0:
            raise ValueError("duration must be non-negative or None")
        if self.fps is not None and self.fps <= 0:
            raise ValueError("fps must be positive or None")
        for name in ("frame_count", "width", "height"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative or None")
        if self.label is not None and self.is_real is not None:
            expected_is_real = self.label == 0
            if self.is_real != expected_is_real:
                raise ValueError("label and is_real disagree with the project label convention")
        for name in ("source_id", "generator", "source_domain"):
            value = getattr(self, name)
            if value == "":
                raise ValueError(f"{name} must use None, not an empty string, for missing data")

        statuses = dict(self.metadata_status)
        missing_statuses = sorted(set(statuses.values()) - METADATA_STATES)
        if missing_statuses:
            raise ValueError(f"Unsupported metadata_status values: {missing_statuses}")
        absent = sorted(set(OPTIONAL_METADATA_FIELDS) - set(statuses))
        if absent:
            raise ValueError(f"metadata_status is missing required fields: {absent}")
        for name in OPTIONAL_METADATA_FIELDS:
            if getattr(self, name) is None and statuses[name] not in {
                "missing",
                "unresolved_schema",
            }:
                raise ValueError(f"{name}=None requires an explicit missing status")

        object.__setattr__(self, "sha256", digest)
        object.__setattr__(
            self,
            "source_lineage",
            None if self.source_lineage is None else dict(self.source_lineage),
        )
        object.__setattr__(self, "metadata_status", statuses)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV-friendly deterministic mapping."""

        return asdict(self)


FORMAL_SAMPLE_FIELDS = tuple(FormalVideoSample.__dataclass_fields__)
