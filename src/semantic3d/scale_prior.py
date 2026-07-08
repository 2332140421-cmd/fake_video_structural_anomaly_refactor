"""Scale-prior loading and alias resolution.

This module lets many fine-grained detector labels share a smaller set of
coarse physical scale priors. Unknown labels resolve to None so downstream
pipelines can skip them safely instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Union

import yaml

from .scale_depth import ScalePrior

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ScalePriorRecord:
    """Physical scale prior with reliability metadata."""

    min_size: float
    max_size: float
    reliable: bool = True

    @property
    def prior(self) -> ScalePrior:
        """Return the legacy ScalePrior object consumed by R_sd functions."""

        return ScalePrior(min_size=self.min_size, max_size=self.max_size)


@dataclass(frozen=True)
class ResolvedScalePrior:
    """Resolved prior status for a raw object label."""

    original_label: str
    resolved_label: str
    min_size: float
    max_size: float
    reliable: bool
    source: Literal["exact", "alias", "missing", "unreliable"]

    @property
    def prior(self) -> ScalePrior:
        """Return the legacy ScalePrior object consumed by R_sd functions."""

        return ScalePrior(min_size=self.min_size, max_size=self.max_size)

    @property
    def resolution(self) -> str:
        """Backward-compatible alias for older tests and callers."""

        return self.source

    @property
    def has_prior(self) -> bool:
        """Whether this result points to an existing scale prior."""

        return self.source in {"exact", "alias", "unreliable"}


def normalize_label(label: str) -> str:
    """Normalize category labels for exact and alias lookup."""

    return "_".join(label.strip().lower().replace("_", " ").split())


class ScalePriorResolver:
    """Resolve exact labels or aliases into coarse scale priors."""

    def __init__(
        self,
        scale_priors: Mapping[str, ScalePrior | ScalePriorRecord],
        aliases: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Create a resolver from exact priors and alias mappings."""

        self.scale_priors = {}
        for label, prior in scale_priors.items():
            if isinstance(prior, ScalePriorRecord):
                record = prior
            else:
                record = ScalePriorRecord(
                    min_size=prior.min_size,
                    max_size=prior.max_size,
                    reliable=True,
                )
            self.scale_priors[normalize_label(label)] = record
        self.aliases = {
            normalize_label(alias): normalize_label(target)
            for alias, target in (aliases or {}).items()
        }
        self._validate()

    def _validate(self) -> None:
        """Validate scale priors and alias targets."""

        for label, prior in self.scale_priors.items():
            if prior.min_size <= 0:
                raise ValueError(
                    f"Scale prior '{label}' has min={prior.min_size}; min must be > 0."
                )
            if prior.max_size <= prior.min_size:
                raise ValueError(
                    f"Scale prior '{label}' has min={prior.min_size}, "
                    f"max={prior.max_size}; max must be > min."
                )
        for alias, target in self.aliases.items():
            if target not in self.scale_priors:
                raise ValueError(
                    f"Alias '{alias}' points to missing scale prior '{target}'."
                )

    def resolve(
        self,
        label: str,
        require_reliable: bool = True,
    ) -> ResolvedScalePrior:
        """Resolve a raw label to a scale prior or a missing/unreliable status."""

        normalized = normalize_label(label)
        if normalized in self.scale_priors:
            record = self.scale_priors[normalized]
            source = "exact" if record.reliable or not require_reliable else "unreliable"
            return ResolvedScalePrior(
                original_label=label,
                resolved_label=normalized,
                min_size=record.min_size,
                max_size=record.max_size,
                reliable=record.reliable,
                source=source,
            )

        target = self.aliases.get(normalized)
        if target is None:
            return ResolvedScalePrior(
                original_label=label,
                resolved_label=normalized,
                min_size=0.0,
                max_size=0.0,
                reliable=False,
                source="missing",
            )
        record = self.scale_priors[target]
        source = "alias" if record.reliable or not require_reliable else "unreliable"
        return ResolvedScalePrior(
            original_label=label,
            resolved_label=target,
            min_size=record.min_size,
            max_size=record.max_size,
            reliable=record.reliable,
            source=source,
        )

    def to_scale_prior_map(self) -> dict[str, ScalePrior]:
        """Return a copy of exact coarse scale-prior mapping."""

        return {label: record.prior for label, record in self.scale_priors.items()}


def load_scale_prior_resolver(config_path: PathLike) -> ScalePriorResolver:
    """Load scale priors and aliases from a YAML config file."""

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Scale prior config must be a mapping: {path}")

    raw_priors = data.get("scale_priors")
    if not isinstance(raw_priors, dict):
        raise ValueError("Scale prior config requires a 'scale_priors' mapping.")

    priors: dict[str, ScalePriorRecord] = {}
    for label, item in raw_priors.items():
        if not isinstance(item, dict):
            raise ValueError(f"Scale prior for '{label}' must be a mapping.")
        if "min" not in item or "max" not in item:
            raise ValueError(f"Scale prior for '{label}' requires min and max.")
        priors[str(label)] = ScalePriorRecord(
            min_size=float(item["min"]),
            max_size=float(item["max"]),
            reliable=bool(item.get("reliable", True)),
        )

    raw_aliases = data.get("aliases", {})
    if not isinstance(raw_aliases, dict):
        raise ValueError("Scale prior config field 'aliases' must be a mapping.")
    aliases = {str(alias): str(target) for alias, target in raw_aliases.items()}
    return ScalePriorResolver(priors, aliases)


def default_scale_prior_resolver(project_root: Optional[PathLike] = None) -> ScalePriorResolver:
    """Load the default project scale-prior resolver."""

    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    return load_scale_prior_resolver(root / "configs" / "scale_priors.yaml")
