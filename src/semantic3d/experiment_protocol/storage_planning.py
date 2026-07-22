"""Estimate storage from P4-B.5 artifacts without copying formal data."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any


def _tree_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _disk_info(path: Path) -> dict[str, Any]:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return {
        "configured_path": str(path),
        "probed_existing_path": str(probe),
        "exists": path.exists(),
        "writable": os.access(probe, os.W_OK),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def estimate_storage(
    structural_dataset_root: str | Path,
    *,
    frame_count: int,
    source_video_bytes: int,
    planned_frame_count: int = 100_000,
    source_root: str | Path | None = None,
    dataset_output_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    archive_root: str | Path | None = None,
) -> dict[str, Any]:
    """Scale observed storage per frame and expose an explicit runtime assumption."""

    root = Path(structural_dataset_root)
    data_root = Path(os.environ.get("DATA_ROOT", "data"))
    output_root = Path(os.environ.get("OUTPUT_ROOT", "outputs"))
    runtime_cache_root = Path(os.environ.get("CACHE_ROOT", "cache"))
    source_root = Path(source_root) if source_root is not None else data_root / "formal_sources"
    dataset_output_root = (
        Path(dataset_output_root)
        if dataset_output_root is not None
        else output_root / "formal_datasets"
    )
    cache_root = Path(cache_root) if cache_root is not None else runtime_cache_root
    archive_root = (
        Path(archive_root) if archive_root is not None else data_root / "formal_archive"
    )
    categories = {
        "frames_bytes": _tree_size(root / "arrays/frames"),
        "depth_arrays_bytes": _tree_size(root / "arrays/depth"),
        "mask_arrays_bytes": _tree_size(root / "arrays/masks"),
        "shared_3d_arrays_bytes": _tree_size(root / "arrays/shared_3d_frames"),
        "point_and_3d_tables_bytes": sum(
            _tree_size(path)
            for path in [
                root / "observations/point_tracks_2d.parquet",
                root / "observations/point_tracks_3d.parquet",
                root / "observations/keypoints_3d.parquet",
            ]
        ),
        "cache_bytes": _tree_size(root / ".cache"),
        "final_parquet_and_reports_bytes": _tree_size(root / "observations")
        + _tree_size(root / "evidence")
        + _tree_size(root / "reports")
        + _tree_size(root / "manifests"),
        "source_video_bytes": int(source_video_bytes),
    }
    measured_total = _tree_size(root)
    scale = planned_frame_count / max(frame_count, 1)
    estimated_final = int(measured_total * scale)
    estimated_cache = int(categories["cache_bytes"] * scale)
    estimated_source = int(source_video_bytes * scale)
    temporary_peak = int(1.25 * (estimated_final + estimated_cache + estimated_source))
    return {
        "basis": {
            "structural_dataset_root": str(root),
            "measured_frame_count": frame_count,
            "measured_total_bytes": measured_total,
            "planned_frame_count": planned_frame_count,
            "linear_scaling_assumption": True,
        },
        "measured_categories": categories,
        "formal_build_estimate": {
            "source_video_bytes": estimated_source,
            "dataset_output_bytes": estimated_final,
            "cache_bytes": estimated_cache,
            "temporary_peak_bytes": temporary_peak,
            "runtime_hours_low": planned_frame_count * 1.0 / 3600.0,
            "runtime_hours_high": planned_frame_count * 5.0 / 3600.0,
            "runtime_assumption": "planning range 1-5 CPU seconds/frame; benchmark before acquisition freeze",
        },
        "paths": {
            "source_root": _disk_info(source_root),
            "dataset_output_root": _disk_info(dataset_output_root),
            "cache_root": _disk_info(cache_root),
            "archive_root": _disk_info(archive_root),
        },
        "ntfs_small_file_warning": (
            "Non-native or network filesystems may be slow for many small files; prefer sharded "
            "arrays/Parquet and a fast local active cache."
        ),
        "data_copied": False,
    }
