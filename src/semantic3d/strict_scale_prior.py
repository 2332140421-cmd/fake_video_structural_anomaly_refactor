"""Frozen physical scale-prior catalog with explicit audit-state resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Union

import yaml

from .scale_depth import ScalePrior
from .scale_prior import normalize_label


PathLike = Union[str, Path]
PriorResolution = Literal["exact", "alias", "unreliable", "missing"]


@dataclass(frozen=True)
class StrictPhysicalPriorEntry:
    """One sourced candidate stored in a frozen physical-prior version."""

    label: str
    min_size: Optional[float]
    max_size: Optional[float]
    characteristic_dimension: str
    dimension_definition: str
    unit: str
    estimation_method: str
    source_count: int
    sources: tuple[Mapping[str, Any], ...]
    audit_status: str
    reliable: bool
    reliability_reason: str
    pose_sensitivity: str
    multimodal_warning: bool
    reviewed_at: str
    prior_version: str

    @property
    def has_interval(self) -> bool:
        """Whether the entry has a positive ordered physical interval."""

        return bool(
            self.min_size is not None
            and self.max_size is not None
            and self.min_size > 0
            and self.max_size > self.min_size
        )


@dataclass(frozen=True)
class StrictResolvedPhysicalPrior:
    """Resolution result preserving strict audit and explanation metadata."""

    original_label: str
    resolved_label: str
    resolution: PriorResolution
    entry: Optional[StrictPhysicalPriorEntry]
    prior_source: Literal["physical"] = "physical"

    @property
    def reliable(self) -> bool:
        """Return True only for strict reliable_single entries."""

        return bool(
            self.entry is not None
            and self.entry.reliable
            and self.entry.audit_status == "reliable_single"
            and self.entry.has_interval
        )

    @property
    def low(self) -> float:
        """Return the lower bound or NaN when no interval exists."""

        if self.entry is None or self.entry.min_size is None:
            return float("nan")
        return float(self.entry.min_size)

    @property
    def high(self) -> float:
        """Return the upper bound or NaN when no interval exists."""

        if self.entry is None or self.entry.max_size is None:
            return float("nan")
        return float(self.entry.max_size)

    @property
    def audit_status(self) -> str:
        """Return the audit status, using missing for unknown labels."""

        return self.entry.audit_status if self.entry is not None else "missing"


class StrictPhysicalScalePriorResolver:
    """Resolve labels from a frozen physical prior without empirical fallback."""

    def __init__(
        self,
        entries: Mapping[str, StrictPhysicalPriorEntry],
        aliases: Optional[Mapping[str, str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.entries = {normalize_label(key): value for key, value in entries.items()}
        self.aliases = {
            normalize_label(key): normalize_label(value)
            for key, value in (aliases or {}).items()
        }
        self.metadata = dict(metadata or {})
        for alias, target in self.aliases.items():
            if target not in self.entries:
                raise ValueError(f"Strict alias '{alias}' points to unknown '{target}'.")

    def resolve(self, label: str) -> StrictResolvedPhysicalPrior:
        """Resolve exact/alias labels and preserve audited unreliable entries."""

        normalized = normalize_label(label)
        resolution: PriorResolution = "exact"
        target = normalized
        if normalized not in self.entries:
            target = self.aliases.get(normalized, normalized)
            resolution = "alias" if normalized in self.aliases else "missing"
        entry = self.entries.get(target)
        if entry is None:
            return StrictResolvedPhysicalPrior(label, normalized, "missing", None)
        if not (
            entry.reliable
            and entry.audit_status == "reliable_single"
            and entry.has_interval
        ):
            resolution = "unreliable"
        return StrictResolvedPhysicalPrior(label, target, resolution, entry)

    def to_scale_prior_map(self) -> dict[str, ScalePrior]:
        """Return only strict reliable_single physical intervals."""

        return {
            label: ScalePrior(entry.min_size, entry.max_size)  # type: ignore[arg-type]
            for label, entry in self.entries.items()
            if entry.reliable and entry.audit_status == "reliable_single" and entry.has_interval
        }


def _optional_float(value: Any) -> Optional[float]:
    """Convert a YAML value to an optional float."""

    return None if value is None else float(value)


def _entry(label: str, raw: Mapping[str, Any]) -> StrictPhysicalPriorEntry:
    """Build one strict entry from frozen YAML metadata."""

    status = str(raw.get("audit_status", "unsupported"))
    reliable = bool(raw.get("reliable", False))
    if reliable and status != "reliable_single":
        raise ValueError(
            f"Frozen prior '{label}' cannot be reliable with audit_status={status!r}."
        )
    return StrictPhysicalPriorEntry(
        label=normalize_label(label),
        min_size=_optional_float(raw.get("min")),
        max_size=_optional_float(raw.get("max")),
        characteristic_dimension=str(raw.get("characteristic_dimension", "")),
        dimension_definition=str(raw.get("dimension_definition", "")),
        unit=str(raw.get("unit", "")),
        estimation_method=str(raw.get("estimation_method", "")),
        source_count=int(raw.get("source_count", 0)),
        sources=tuple(raw.get("sources", [])),
        audit_status=status,
        reliable=reliable,
        reliability_reason=str(raw.get("reliability_reason", "")),
        pose_sensitivity=str(raw.get("pose_sensitivity", "")),
        multimodal_warning=bool(raw.get("multimodal_warning", False)),
        reviewed_at=str(raw.get("reviewed_at", "")),
        prior_version=str(raw.get("prior_version", "")),
    )


def load_strict_physical_prior_resolver(
    config_path: PathLike,
) -> StrictPhysicalScalePriorResolver:
    """Load a frozen strict physical prior and its excluded reviewed entries."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Strict prior config must be a mapping: {path}")
    metadata = data.get("metadata")
    if not isinstance(metadata, dict) or not metadata.get("prior_version"):
        raise ValueError("Strict prior config requires metadata.prior_version.")
    raw_priors = data.get("scale_priors", {})
    raw_excluded = data.get("excluded_priors", {})
    if not isinstance(raw_priors, dict) or not isinstance(raw_excluded, dict):
        raise ValueError("scale_priors and excluded_priors must be mappings.")
    entries = {
        str(label): _entry(str(label), raw)
        for section in (raw_priors, raw_excluded)
        for label, raw in section.items()
    }
    return StrictPhysicalScalePriorResolver(
        entries,
        aliases=data.get("aliases", {}),
        metadata=metadata,
    )

