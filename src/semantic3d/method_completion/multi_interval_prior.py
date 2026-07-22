"""Versionable multi-interval physical-size priors for metric object scale."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from ..scale_prior import normalize_label


@dataclass(frozen=True, order=True)
class SizeInterval:
    """One positive physical interval in meters."""

    low: float
    high: float

    def __post_init__(self) -> None:
        low, high = float(self.low), float(self.high)
        if not math.isfinite(low) or not math.isfinite(high) or low <= 0.0 or high <= low:
            raise ValueError("SizeInterval requires 0 < low < high.")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)


@dataclass(frozen=True)
class DimensionScalePrior:
    """Physical prior for one explicitly named object dimension."""

    dimension_name: str
    intervals: tuple[SizeInterval, ...]
    unit: str = "meter"
    subtype: str = ""
    pose_applicability: tuple[str, ...] = ()
    confidence: float = 1.0
    sources: tuple[Mapping[str, Any], ...] = ()
    license: str = "unresolved"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        intervals = tuple(sorted(self.intervals))
        if not self.dimension_name.strip() or not intervals:
            raise ValueError("DimensionScalePrior requires a name and intervals.")
        if self.unit not in {"meter", "m"}:
            raise ValueError("Metric physical priors must use meter units.")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Prior confidence must be in [0, 1].")
        object.__setattr__(self, "intervals", intervals)
        object.__setattr__(self, "unit", "meter")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "sources", tuple(dict(item) for item in self.sources))
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True)
class ObjectPhysicalScalePrior:
    """Dimension-specific prior registry entry for one class or subtype."""

    class_name: str
    dimensions: Mapping[str, DimensionScalePrior]
    subtype: str = ""
    prior_version: str = ""

    def __post_init__(self) -> None:
        if not self.class_name.strip() or not self.dimensions:
            raise ValueError("ObjectPhysicalScalePrior requires class_name and dimensions.")
        object.__setattr__(self, "class_name", normalize_label(self.class_name))
        object.__setattr__(self, "dimensions", dict(self.dimensions))


@dataclass(frozen=True)
class ResolvedMultiIntervalPrior:
    """Exact/alias/missing registry resolution."""

    original_label: str
    resolved_label: str
    resolution: str
    prior: Optional[ObjectPhysicalScalePrior]


class MultiIntervalScalePriorRegistry:
    """Resolve dimension-specific, possibly disjoint physical intervals."""

    def __init__(
        self,
        entries: Mapping[str, ObjectPhysicalScalePrior],
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
                raise ValueError(f"Alias {alias!r} points to missing prior {target!r}.")

    def resolve(self, label: str) -> ResolvedMultiIntervalPrior:
        """Resolve exact first, then alias, without empirical fallback."""

        normalized = normalize_label(label)
        if normalized in self.entries:
            return ResolvedMultiIntervalPrior(label, normalized, "exact", self.entries[normalized])
        target = self.aliases.get(normalized)
        if target is not None:
            return ResolvedMultiIntervalPrior(label, target, "alias", self.entries[target])
        return ResolvedMultiIntervalPrior(label, normalized, "missing", None)

    @classmethod
    def from_strict_v2(cls, resolver: Any) -> "MultiIntervalScalePriorRegistry":
        """Adapt frozen strict-v2 entries as one-interval dimensions, read-only.

        The adapter does not modify or reinterpret strict-v2 pair calculations;
        it only exposes reviewed physical metadata to the new metric unary route.
        """

        entries: dict[str, ObjectPhysicalScalePrior] = {}
        for label, entry in resolver.entries.items():
            if not entry.available:
                continue
            dimension = _canonical_dimension(entry.characteristic_dimension)
            if not dimension:
                continue
            interval = SizeInterval(float(entry.min_size), float(entry.max_size))
            entries[label] = ObjectPhysicalScalePrior(
                class_name=label,
                dimensions={
                    dimension: DimensionScalePrior(
                        dimension_name=dimension,
                        intervals=(interval,),
                        confidence=1.0 if entry.reliability_status == "strict_high" else 0.7,
                        pose_applicability=(entry.pose_sensitivity,),
                        sources=entry.sources,
                        license="see_source_metadata",
                        provenance={
                            "source_registry": "strict_v2_read_only_adapter",
                            "reliability_status": entry.reliability_status,
                            "dimension_definition": entry.dimension_definition,
                        },
                    )
                },
                prior_version=entry.prior_version,
            )
        aliases = {
            alias: target
            for alias, target in resolver.aliases.items()
            if target in entries
        }
        return cls(entries, aliases, {**resolver.metadata, "adapter": "strict_v2"})


def _canonical_dimension(name: str) -> str:
    normalized = normalize_label(name)
    if "height" in normalized:
        return "height_m"
    if "width" in normalized:
        return "width_m"
    if "length" in normalized:
        return "length_m"
    if normalized in {"diameter", "characteristic_linear_extent"}:
        return "extent_m"
    return ""


def log_distance_to_interval_union(
    value: float, intervals: Sequence[SizeInterval]
) -> float:
    """Return dimensionless log-distance to the nearest legal interval."""

    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError("Observed metric scale must be finite and positive.")
    legal = tuple(intervals)
    if not legal:
        raise ValueError("At least one prior interval is required.")
    log_value = math.log(number)
    return min(
        max(0.0, math.log(interval.low) - log_value, log_value - math.log(interval.high))
        for interval in legal
    )
