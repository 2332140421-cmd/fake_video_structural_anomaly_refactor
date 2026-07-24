"""Abstract interfaces for dataset-specific metadata adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from semantic3d.dataset_builder.formal_schema import FormalVideoSample


class UnresolvedDatasetSchemaError(RuntimeError):
    """Raised when an adapter would otherwise have to guess source metadata."""


class DatasetAdapter(ABC):
    """Map verified dataset metadata into the project-wide sample contract."""

    @abstractmethod
    def read_samples(
        self,
        metadata_path: str | Path,
        *,
        data_root: str | Path,
    ) -> list[FormalVideoSample]:
        """Read source metadata without scanning or guessing its directory layout."""
