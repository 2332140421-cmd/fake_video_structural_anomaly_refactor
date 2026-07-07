"""Scale-prior loading and alias resolution.

This module lets many fine-grained detector labels share a smaller set of
coarse physical scale priors. Unknown labels resolve to None so downstream
pipelines can skip them safely instead of crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import yaml

from .scale_depth import ScalePrior

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ResolvedScalePrior:
    """Resolved prior for a raw object label."""

    original_label: str
    resolved_label: str
    prior: ScalePrior
    resolution: str


def normalize_label(label: str) -> str:
    """Normalize category labels for exact and alias lookup."""

    return label.strip().lower()


class ScalePriorResolver:
    """Resolve exact labels or aliases into coarse scale priors."""

    def __init__(
        self,
        scale_priors: Mapping[str, ScalePrior],
        aliases: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Create a resolver from exact priors and alias mappings."""

        self.scale_priors = {
            normalize_label(label): prior for label, prior in scale_priors.items()
        }
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

    def resolve(self, label: str) -> Optional[ResolvedScalePrior]:
        """Resolve a raw label to a scale prior, or return None if unknown."""

        normalized = normalize_label(label)
        if normalized in self.scale_priors:
            return ResolvedScalePrior(
                original_label=label,
                resolved_label=normalized,
                prior=self.scale_priors[normalized],
                resolution="exact",
            )

        target = self.aliases.get(normalized)
        if target is None:
            return None
        return ResolvedScalePrior(
            original_label=label,
            resolved_label=target,
            prior=self.scale_priors[target],
            resolution="alias",
        )

    def to_scale_prior_map(self) -> dict[str, ScalePrior]:
        """Return a copy of exact coarse scale-prior mapping."""

        return dict(self.scale_priors)


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

    priors: dict[str, ScalePrior] = {}
    for label, item in raw_priors.items():
        if not isinstance(item, dict):
            raise ValueError(f"Scale prior for '{label}' must be a mapping.")
        if "min" not in item or "max" not in item:
            raise ValueError(f"Scale prior for '{label}' requires min and max.")
        priors[str(label)] = ScalePrior(
            min_size=float(item["min"]),
            max_size=float(item["max"]),
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
