"""Reserved interface for future empirical category-pair priors.

This module defines contracts only. The strict physical R_sd baseline never
instantiates or falls back to an empirical resolver.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class EmpiricalPairPrior:
    """Future mixture summary for one category pair."""

    category_pair: tuple[str, str]
    sample_count: int
    num_components: int
    component_means: Sequence[float]
    component_stds: Sequence[float]
    component_weights: Sequence[float]
    expected_intervals: Sequence[tuple[float, float]]
    support_confidence: float
    prior_source: str = "empirical_pair"


@dataclass(frozen=True)
class EmpiricalPairPriorMatch:
    """Future component match returned for an observed category pair."""

    prior: EmpiricalPairPrior
    matched_component: Optional[int]


class EmpiricalPairPriorResolver(ABC):
    """Abstract future resolver; no implementation is enabled in this project stage."""

    @abstractmethod
    def resolve(
        self,
        category_a: str,
        category_b: str,
        observed_log_ratio: float,
    ) -> Optional[EmpiricalPairPriorMatch]:
        """Resolve a category pair without mixing it with physical priors."""

