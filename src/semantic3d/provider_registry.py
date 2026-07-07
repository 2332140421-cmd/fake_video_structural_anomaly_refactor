"""Provider registry for selecting object providers by name."""

from __future__ import annotations

from typing import Any

from .providers import BaseObjectProvider, MockObjectProvider
from .real_object_provider import RealObjectProvider


def get_object_provider(provider_name: str, **kwargs: Any) -> BaseObjectProvider:
    """Return an object provider by registry name.

    Supported names:
        mock: Deterministic mock objects.
        real_detector: RealObjectProvider wrapper around a local detector.
    """

    normalized = provider_name.strip().lower()
    if normalized == "mock":
        mock_mode = kwargs.pop("mock_mode", "reasonable")
        return MockObjectProvider(mode=mock_mode)
    if normalized == "real_detector":
        return RealObjectProvider(**kwargs)
    raise ValueError(
        f"Unknown object provider '{provider_name}'. Use 'mock' or 'real_detector'."
    )
