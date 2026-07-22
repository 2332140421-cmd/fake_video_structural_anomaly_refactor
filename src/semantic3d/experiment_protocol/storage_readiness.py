"""Deterministic storage batch planning for P4-C2 readiness."""

from __future__ import annotations

import math
from typing import Any, Mapping


def build_storage_batch_plan(
    storage_estimate: Mapping[str, Any],
    storage_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Plan batches from a frozen disk snapshot without downloading data."""

    formal = storage_estimate["formal_build_estimate"]
    basis = storage_estimate["basis"]
    planned_frames = int(basis["planned_frame_count"])
    batch_frames = int(storage_config["batch_frame_count"])
    safety = int(storage_config["safety_margin_bytes"])
    available = int(storage_config["audited_available_bytes"])
    if planned_frames <= 0 or batch_frames <= 0:
        raise ValueError("Storage frame counts must be positive")
    batch_count = math.ceil(planned_frames / batch_frames)

    def scaled(total: int, count: int) -> int:
        return math.ceil(int(total) * count / planned_frames)

    batches = []
    cumulative_permanent = 0
    feasible_batch_count = 0
    for index in range(batch_count):
        frames = min(batch_frames, planned_frames - index * batch_frames)
        download = scaled(int(formal["source_video_bytes"]), frames)
        permanent = scaled(int(formal["dataset_output_bytes"]), frames)
        cache = scaled(int(formal["cache_bytes"]), frames)
        temporary_peak = scaled(int(formal["temporary_peak_bytes"]), frames)
        required_before = cumulative_permanent + temporary_peak + safety
        fits = required_before <= available
        feasible_batch_count += int(fits)
        batches.append(
            {
                "batch_id": f"batch_{index + 1:03d}",
                "frame_start": index * batch_frames,
                "frame_end_exclusive": index * batch_frames + frames,
                "frame_count": frames,
                "expected_download_bytes": download,
                "temporary_peak_bytes": temporary_peak,
                "permanent_retained_bytes": permanent,
                "rebuildable_cache_bytes": cache,
                "deletable_after_validation_bytes": max(0, temporary_peak - permanent) + cache,
                "required_available_before_batch_bytes": required_before,
                "fits_audited_snapshot": fits,
                "validation_steps": [
                    "verify_official_checksums",
                    "validate_manifest_row_counts",
                    "validate_array_references_and_sha256",
                    "write_stage_status_atomically",
                ],
                "failure_recovery": "resume_from_last_validated_stage_status_without_deleting_verified_prior_batches",
            }
        )
        cumulative_permanent += permanent

    total_peak_with_safety = int(formal["temporary_peak_bytes"]) + safety
    full_build_fits = total_peak_with_safety <= available
    return {
        "planning_only": True,
        "downloads_performed": False,
        "formal_build_started": False,
        "storage_snapshot": {
            "path": str(storage_config["path"]),
            "audited_at": str(storage_config["audited_at"]),
            "total_bytes": int(storage_config["audited_total_bytes"]),
            "available_bytes": available,
            "safety_margin_bytes": safety,
            "snapshot_source": str(storage_config["snapshot_source"]),
        },
        "planned_frame_count": planned_frames,
        "batch_frame_count": batch_frames,
        "batch_count": batch_count,
        "feasible_batch_count_without_archiving": feasible_batch_count,
        "formal_total_source_bytes": int(formal["source_video_bytes"]),
        "formal_total_permanent_bytes": int(formal["dataset_output_bytes"]),
        "formal_total_temporary_peak_bytes": int(formal["temporary_peak_bytes"]),
        "full_build_required_with_safety_bytes": total_peak_with_safety,
        "full_build_fits_audited_snapshot": full_build_fits,
        "build_blocked_insufficient_storage": not full_build_fits,
        "safe_space_policy": "temporary_peak_plus_safety_margin_must_fit_before_start",
        "batches": batches,
    }

