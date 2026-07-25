"""Dataset-specific adapters for the unified formal video schema."""

from .base import DatasetAdapter, UnresolvedDatasetSchemaError
from .decof import (
    DECOF_ADAPTER_STATUS,
    DECOF_ARCHIVE_LAYOUTS,
    DECOF_DATA_DOWNLOADED,
    DECOF_MEDIA_RESOLUTION_READY,
    DECOF_NETWORK_ACCESS_ALLOWED,
    DECOF_OFFICIAL_SCHEMA_VERIFIED,
    DECOF_PRODUCTION_ADAPTER_READY,
    DeCoFMetadataAdapter,
    DeCoFMetadataError,
    DeCoFMetadataRecord,
    build_decof_small_sample_plan,
    load_decof_member_index,
    parse_decof_zip_central_directory,
    summarize_decof_records,
)
from .genvideo100k import (
    GENVIDEO100K_SCHEMA_STATUS,
    GenVideo100KAdapter,
    GenVideo100KFieldMapping,
)

__all__ = [
    "DECOF_ADAPTER_STATUS",
    "DECOF_ARCHIVE_LAYOUTS",
    "DECOF_DATA_DOWNLOADED",
    "DECOF_MEDIA_RESOLUTION_READY",
    "DECOF_NETWORK_ACCESS_ALLOWED",
    "DECOF_OFFICIAL_SCHEMA_VERIFIED",
    "DECOF_PRODUCTION_ADAPTER_READY",
    "DatasetAdapter",
    "DeCoFMetadataAdapter",
    "DeCoFMetadataError",
    "DeCoFMetadataRecord",
    "GENVIDEO100K_SCHEMA_STATUS",
    "GenVideo100KAdapter",
    "GenVideo100KFieldMapping",
    "UnresolvedDatasetSchemaError",
    "build_decof_small_sample_plan",
    "load_decof_member_index",
    "parse_decof_zip_central_directory",
    "summarize_decof_records",
]
