"""Build the P4-C2 formal dataset and license registries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .formal_schema import FormalDatasetRecord


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def build_formal_dataset_registry(
    dataset_configs: Iterable[Mapping[str, Any]],
    *,
    project_root: str | Path,
) -> tuple[FormalDatasetRecord, ...]:
    """Validate configured datasets and record lightweight local-root inventory."""

    project = Path(project_root)
    output: list[FormalDatasetRecord] = []
    seen: set[tuple[str, str]] = set()
    for item in dataset_configs:
        key = (str(item["dataset_name"]), str(item["version"]))
        if key in seen:
            raise ValueError(f"Duplicate dataset registry entry: {key}")
        seen.add(key)
        local_root = _resolve(project, str(item["local_root"]))
        missing = list(str(value) for value in item.get("missing_requirements", ()))
        if not local_root.exists():
            missing.append("local_root_missing")
        role = str(item["dataset_role"])
        formal_flags = {
            name: bool(item.get(name, False))
            for name in (
                "eligible_for_training",
                "eligible_for_model_selection",
                "eligible_for_threshold_selection",
                "eligible_for_final_evaluation",
            )
        }
        if role == "geometry_validation_smoke" and any(formal_flags.values()):
            raise ValueError("geometry_validation_smoke cannot be eligible for formal modeling")
        output.append(
            FormalDatasetRecord(
                dataset_name=key[0],
                version=key[1],
                official_source=str(item["official_source"]),
                license=str(item["license"]),
                citation=str(item["citation"]),
                download_method=str(item["download_method"]),
                official_split=str(item["official_split"]),
                annotation_types=tuple(sorted(str(v) for v in item.get("annotation_types", ()))),
                expected_size=dict(item.get("expected_size", {})),
                checksum_policy=str(item["checksum_policy"]),
                local_root=str(item["local_root"]),
                dataset_role=role,
                registry_status=str(item.get("registry_status", "candidate")),
                missing_requirements=tuple(sorted(set(missing))),
                metadata={
                    **dict(item.get("metadata", {})),
                    "local_root_exists": local_root.exists(),
                    "metadata_inventory_only": True,
                },
                **formal_flags,
            )
        )
    return tuple(sorted(output, key=lambda row: (row.dataset_name, row.version)))


def license_registry_rows(
    datasets: Iterable[FormalDatasetRecord],
) -> list[dict[str, Any]]:
    """Return a separate, deterministic license review registry."""

    return [
        {
            "dataset_name": row.dataset_name,
            "version": row.version,
            "official_source": row.official_source,
            "license": row.license,
            "citation": row.citation,
            "download_method": row.download_method,
            "license_verification_status": row.metadata.get(
                "license_verification_status", "unverified"
            ),
            "redistribution_allowed": row.metadata.get("redistribution_allowed"),
            "review_required": row.metadata.get("license_verification_status") != "verified",
        }
        for row in datasets
    ]

