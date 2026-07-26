#!/usr/bin/env python3
"""Run one reusable paper-core pipeline over a frozen local media manifest."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.residual_dataset import RESIDUAL_NAMES
from data.video import SAMPLER_VERSION
from inference.cli import _pipeline
from inference.outputs import save_analysis_outputs


SUMMARY_FIELDS = (
    "sample_id",
    "group_id",
    "split",
    "label_for_posthoc_reference",
    "analysis_ok",
    "source_video_sha256",
    "sampling_provenance_sha256",
    "residual_sha256",
    "frames",
    "clips",
    "objects",
    "object_tracks",
    "point_tracks",
    "semantic_prior_total",
    "semantic_prior_available",
    "semantic_temporal_total",
    "semantic_temporal_available",
    "d1_total",
    "d1_available",
    "d2_total",
    "d2_available",
    "d3_total",
    "d3_available",
    "overall_coverage",
    "runtime_seconds",
    "peak_gpu_memory_mb",
    "failure_reason",
    "source_commit",
    "source_config_sha256",
    "scale_prior_schema_version",
    "scale_prior_sha256",
    "scale_prior_source_table_sha256",
    "scale_prior_entry_id",
    "scale_prior_confidence",
    "sampler_version",
    "authenticity_label_used",
    "result_path",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_identity(value: str) -> str:
    encoded = base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii")
    return "sid_" + encoded.rstrip("=")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv_atomic(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Iterable[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _git_commit(project_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _clip_sequence(payload: Mapping[str, Any]) -> dict[str, Any]:
    clips = list(payload.get("clips", ()))
    values = np.full((len(clips), len(RESIDUAL_NAMES)), np.nan, dtype=np.float32)
    availability = np.zeros(values.shape, dtype=bool)
    confidence = np.zeros(values.shape, dtype=np.float32)
    unknown: set[str] = set()
    for clip_index, clip in enumerate(clips):
        by_name: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in clip.get("residuals", ()):
            name = str(row.get("name", ""))
            if name not in RESIDUAL_NAMES:
                unknown.add(name)
                continue
            if bool(row.get("valid_mask", False)):
                if row.get("availability") != "observed":
                    raise ValueError("Valid residual is not marked observed.")
                value = float(row["normalized_value"])
                quality = float(row["confidence"])
                if not math.isfinite(value) or not math.isfinite(quality):
                    raise ValueError("Valid residual contains NaN or Inf.")
                by_name[name].append(row)
        for channel_index, name in enumerate(RESIDUAL_NAMES):
            rows = by_name.get(name, ())
            if rows:
                values[clip_index, channel_index] = np.mean(
                    [float(row["normalized_value"]) for row in rows]
                )
                availability[clip_index, channel_index] = True
                confidence[clip_index, channel_index] = np.mean(
                    [float(row["confidence"]) for row in rows]
                )
    if unknown:
        raise ValueError(f"Unknown residual channels: {sorted(unknown)}.")
    if np.any(~np.isfinite(values[availability])):
        raise ValueError("Available residual position contains NaN or Inf.")
    if np.any(confidence[~availability] != 0):
        raise ValueError("Unavailable residual position has nonzero confidence.")
    return {
        "schema_version": 1,
        "channel_count": len(RESIDUAL_NAMES),
        "channel_names": list(RESIDUAL_NAMES),
        "clip_ids": [str(clip["clip_id"]) for clip in clips],
        "values": [
            [None if not math.isfinite(float(value)) else float(value) for value in row]
            for row in values
        ],
        "availability": availability.tolist(),
        "confidence": confidence.tolist(),
        "missing_value_policy": "null with availability=false and confidence=0",
        "authenticity_label_used": False,
    }


def _branch_counts(metadata: Mapping[str, Any], branch: str) -> tuple[int, int]:
    row = metadata.get("branch_evidence_counts", {}).get(branch, {})
    return int(row.get("total", 0)), int(row.get("available", 0))


def _sampling_provenance(
    *,
    row: Mapping[str, str],
    result_metadata: Mapping[str, Any],
    source_sha256: str,
    requested_frames: int,
    clip_length: int,
    clip_count: int,
) -> dict[str, Any]:
    selected_indices = [int(value) for value in result_metadata["selected_frame_indices"]]
    selected_timestamps = [
        float(value) for value in result_metadata["selected_timestamps"]
    ]
    if len(selected_indices) != requested_frames or len(set(selected_indices)) != requested_frames:
        raise ValueError("Uniform sampler did not provide unique requested frames.")
    if any(
        current >= following
        for current, following in zip(selected_indices, selected_indices[1:])
    ):
        raise ValueError("Selected frame indices are not strictly increasing.")
    if any(
        current > following
        for current, following in zip(selected_timestamps, selected_timestamps[1:])
    ):
        raise ValueError("Selected timestamps are decreasing.")
    return {
        "sample_id": row["sample_id"],
        "sampling_mode": "uniform_full_video",
        "source_frame_count": int(result_metadata["source_frame_count"]),
        "source_duration_seconds": float(result_metadata["source_duration_seconds"]),
        "source_fps": float(result_metadata["source_fps"]),
        "requested_frame_count": requested_frames,
        "selected_frame_count": len(selected_indices),
        "selected_frame_indices": selected_indices,
        "selected_timestamps": selected_timestamps,
        "clip_length": clip_length,
        "clip_count": clip_count,
        "sampler_version": SAMPLER_VERSION,
        "source_video_sha256": source_sha256,
        "label_used": False,
    }


def _summary_row(
    row: Mapping[str, str],
    payload: Mapping[str, Any],
    *,
    source_sha256: str,
    provenance_sha256: str,
    result_sha256: str,
    result_path: Path,
    source_commit: str,
    config_sha256: str,
) -> dict[str, Any]:
    metadata = payload["metadata"]
    semantic_prior = _branch_counts(metadata, "semantic_prior")
    semantic_temporal = _branch_counts(metadata, "semantic_temporal")
    d1 = _branch_counts(metadata, "d1")
    d2 = _branch_counts(metadata, "d2")
    d3 = _branch_counts(metadata, "d3")
    return {
        "sample_id": row["sample_id"],
        "group_id": row["group_id"],
        "split": row["split"],
        "label_for_posthoc_reference": row["label"],
        "analysis_ok": True,
        "source_video_sha256": source_sha256,
        "sampling_provenance_sha256": provenance_sha256,
        "residual_sha256": result_sha256,
        "frames": metadata["selected_frame_count"],
        "clips": metadata["clip_count"],
        "objects": metadata.get("objects_total", 0),
        "object_tracks": metadata.get("object_tracks", 0),
        "point_tracks": metadata.get("point_tracks", 0),
        "semantic_prior_total": semantic_prior[0],
        "semantic_prior_available": semantic_prior[1],
        "semantic_temporal_total": semantic_temporal[0],
        "semantic_temporal_available": semantic_temporal[1],
        "d1_total": d1[0],
        "d1_available": d1[1],
        "d2_total": d2[0],
        "d2_available": d2[1],
        "d3_total": d3[0],
        "d3_available": d3[1],
        "overall_coverage": metadata.get("overall_coverage", 0.0),
        "runtime_seconds": metadata.get("runtime_seconds", 0.0),
        "peak_gpu_memory_mb": metadata.get("peak_gpu_memory_mb", 0.0),
        "failure_reason": "",
        "source_commit": source_commit,
        "source_config_sha256": config_sha256,
        "scale_prior_schema_version": metadata["scale_prior_schema_version"],
        "scale_prior_sha256": metadata["scale_prior_sha256"],
        "scale_prior_source_table_sha256": metadata[
            "scale_prior_source_table_sha256"
        ],
        "scale_prior_entry_id": json.dumps(
            metadata.get("scale_prior_entry_id", []),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "scale_prior_confidence": json.dumps(
            metadata.get("scale_prior_confidence", {}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "sampler_version": SAMPLER_VERSION,
        "authenticity_label_used": bool(
            metadata.get("authenticity_label_used", True)
        ),
        "result_path": str(result_path),
    }


def _resume_valid(
    row: Mapping[str, str],
    sample_dir: Path,
    *,
    source_commit: str,
    config_sha256: str,
    source_sha256: str,
) -> tuple[bool, dict[str, Any] | None]:
    result_path = sample_dir / "result.json"
    provenance_path = sample_dir / "sampling_provenance.json"
    sequence_path = sample_dir / "residual_sequence.json"
    if not all(path.is_file() for path in (result_path, provenance_path, sequence_path)):
        return False, None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        sequence = json.loads(sequence_path.read_text(encoding="utf-8"))
    except Exception:
        return False, None
    metadata = payload.get("metadata", {})
    valid = (
        provenance.get("sample_id") == row["sample_id"]
        and provenance.get("source_video_sha256") == source_sha256
        and provenance.get("sampler_version") == SAMPLER_VERSION
        and provenance.get("label_used") is False
        and metadata.get("source_commit") == source_commit
        and metadata.get("source_config_sha256") == config_sha256
        and metadata.get("authenticity_label_used") is False
        and metadata.get("scale_prior_schema_version")
        == "paper_core_scale_priors_v1"
        and bool(metadata.get("scale_prior_sha256"))
        and bool(metadata.get("scale_prior_source_table_sha256"))
        and sequence.get("channel_names") == list(RESIDUAL_NAMES)
    )
    return bool(valid), payload if valid else None


def _compare_payloads(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    sequence_a = _clip_sequence(first)
    sequence_b = _clip_sequence(second)
    if sequence_a["channel_names"] != sequence_b["channel_names"]:
        raise ValueError("Batch/single channel order differs.")
    available_a = np.asarray(sequence_a["availability"], dtype=bool)
    available_b = np.asarray(sequence_b["availability"], dtype=bool)
    if not np.array_equal(available_a, available_b):
        raise ValueError("Batch/single availability differs.")
    values_a = np.asarray(
        [[np.nan if value is None else value for value in row] for row in sequence_a["values"]]
    )
    values_b = np.asarray(
        [[np.nan if value is None else value for value in row] for row in sequence_b["values"]]
    )
    confidence_a = np.asarray(sequence_a["confidence"], dtype=float)
    confidence_b = np.asarray(sequence_b["confidence"], dtype=float)
    valid_value_difference = (
        float(np.max(np.abs(values_a[available_a] - values_b[available_b])))
        if np.any(available_a)
        else 0.0
    )
    confidence_difference = (
        float(np.max(np.abs(confidence_a - confidence_b)))
        if confidence_a.size
        else 0.0
    )
    metadata_keys = (
        "clip_count",
        "objects_total",
        "object_tracks",
        "point_tracks",
        "branch_evidence_counts",
    )
    if any(first["metadata"].get(key) != second["metadata"].get(key) for key in metadata_keys):
        raise ValueError("Batch/single structural counts differ.")
    if valid_value_difference > 1e-6 or confidence_difference > 1e-6:
        raise ValueError("Batch/single residual values differ beyond 1e-6.")
    return {
        "channel_order_equal": True,
        "availability_equal": True,
        "max_valid_residual_difference": valid_value_difference,
        "max_confidence_difference": confidence_difference,
        "structural_counts_equal": True,
        "ready": True,
    }


def _quality_outputs(
    *,
    rows: list[dict[str, str]],
    summaries: list[dict[str, Any]],
    parent: Path,
    final_manifest_path: Path | None,
) -> dict[str, Any]:
    by_sample = {row["sample_id"]: row for row in summaries}
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    group_rows = []
    final_rows = []
    for group_id, members in groups.items():
        completed = [
            member
            for member in members
            if bool(by_sample.get(member["sample_id"], {}).get("analysis_ok"))
        ]
        schema_equal = len(completed) == 2
        sampling_equal = len(completed) == 2 and len(
            {
                (
                    by_sample[member["sample_id"]]["frames"],
                    by_sample[member["sample_id"]]["clips"],
                    by_sample[member["sample_id"]]["sampler_version"],
                )
                for member in completed
            }
        ) == 1
        complete = (
            len(members) == 2
            and len(completed) == 2
            and {member["label"] for member in members} == {"0", "1"}
            and schema_equal
            and sampling_equal
        )
        group_rows.append(
            {
                "group_id": group_id,
                "split": members[0]["split"],
                "real_status": (
                    "COMPLETE"
                    if any(
                        member["label"] == "0"
                        and member["sample_id"] in by_sample
                        and by_sample[member["sample_id"]]["analysis_ok"]
                        for member in members
                    )
                    else "FAILED"
                ),
                "fake_status": (
                    "COMPLETE"
                    if any(
                        member["label"] == "1"
                        and member["sample_id"] in by_sample
                        and by_sample[member["sample_id"]]["analysis_ok"]
                        for member in members
                    )
                    else "FAILED"
                ),
                "residual_schema_equal": schema_equal,
                "sampling_protocol_equal": sampling_equal,
                "group_status": (
                    "GROUP_RESIDUAL_COMPLETE"
                    if complete
                    else "GROUP_RESIDUAL_PARTIAL"
                ),
            }
        )
        if complete and final_manifest_path is not None:
            for member in members:
                summary = by_sample[member["sample_id"]]
                final_rows.append(
                    {
                        "sample_id": member["sample_id"],
                        "dataset_name": member["dataset_name"],
                        "source_video_id": member["source_video_id"],
                        "group_id": group_id,
                        "split": member["split"],
                        "label": member["label"],
                        "residual_sequence_path": summary["result_path"],
                        "video_id": member["video_id"],
                        "generator_name": member["generator_name"],
                        "prompt_sha256": member["prompt_sha256"],
                        "source_video_path": member["local_video_path"],
                        "source_video_sha256": summary["source_video_sha256"],
                        "residual_sha256": summary["residual_sha256"],
                        "sampling_mode": "uniform_full_video",
                        "selected_frame_count": summary["frames"],
                        "clip_length": 8,
                        "clip_count": summary["clips"],
                        "source_commit": summary["source_commit"],
                        "source_config_sha256": summary["source_config_sha256"],
                        "scale_prior_schema_version": summary[
                            "scale_prior_schema_version"
                        ],
                        "scale_prior_sha256": summary["scale_prior_sha256"],
                        "scale_prior_source_table_sha256": summary[
                            "scale_prior_source_table_sha256"
                        ],
                        "scale_prior_entry_id": summary["scale_prior_entry_id"],
                        "scale_prior_confidence": summary["scale_prior_confidence"],
                        "sampler_version": summary["sampler_version"],
                        "license_status": member["license_status"],
                    }
                )
    _write_csv_atomic(
        parent / "group_residual_status.csv",
        group_rows,
        (
            "group_id",
            "split",
            "real_status",
            "fake_status",
            "residual_schema_equal",
            "sampling_protocol_equal",
            "group_status",
        ),
    )
    if final_manifest_path is not None:
        _write_csv_atomic(
            final_manifest_path,
            final_rows,
            (
                "sample_id",
                "dataset_name",
                "source_video_id",
                "group_id",
                "split",
                "label",
                "residual_sequence_path",
                "video_id",
                "generator_name",
                "prompt_sha256",
                "source_video_path",
                "source_video_sha256",
                "residual_sha256",
                "sampling_mode",
                "selected_frame_count",
                "clip_length",
                "clip_count",
                "source_commit",
                "source_config_sha256",
                "scale_prior_schema_version",
                "scale_prior_sha256",
                "scale_prior_source_table_sha256",
                "scale_prior_entry_id",
                "scale_prior_confidence",
                "sampler_version",
                "license_status",
            ),
        )
    return {
        "group_count": len(group_rows),
        "complete_group_count": sum(
            row["group_status"] == "GROUP_RESIDUAL_COMPLETE" for row in group_rows
        ),
        "final_manifest_row_count": len(final_rows),
    }


def run(arguments: argparse.Namespace, pipeline_factory=_pipeline) -> int:
    manifest = Path(arguments.manifest).resolve()
    config_path = Path(arguments.config).resolve()
    output_root = Path(arguments.output).resolve()
    project_root = Path(__file__).resolve().parents[1]
    parent = output_root.parent
    rows = _read_csv(manifest)
    if not rows:
        raise ValueError("Local media manifest is empty.")
    if arguments.sampling_mode != "uniform_full_video":
        raise ValueError("Only uniform_full_video is supported in R5-B2.")
    if arguments.sampled_frames != arguments.clip_length * arguments.clip_count:
        raise ValueError("sampled-frames must equal clip-length * clip-count.")
    if any(row.get("media_inventory_status") != "COMPLETE" for row in rows):
        raise ValueError("Residual runner accepts only COMPLETE media inventory rows.")
    source_commit = _git_commit(project_root)
    config_sha256 = sha256_file(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = parent / "residual_extraction_summary.csv"
    existing = {
        row["sample_id"]: row
        for row in _read_csv(summary_path)
    } if arguments.resume and summary_path.is_file() else {}
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    coverage = {
        name: {"channel_name": name, "total_clips": 0, "available_clips": 0}
        for name in RESIDUAL_NAMES
    }
    unavailable_reasons: Counter[str] = Counter()
    pipeline, config = pipeline_factory(config_path)
    initialized_pipeline_count = 1
    equivalence_targets = {
        row["sample_id"]
        for row in sorted(rows, key=lambda item: (item["selection_key"], item["sample_id"]))[:2]
    }
    equivalence: dict[str, Any] = {}
    started_all = time.perf_counter()

    def persist() -> None:
        _write_csv_atomic(summary_path, summaries, SUMMARY_FIELDS)
        _write_json_atomic(
            parent / "residual_extraction_summary.json",
            {
                "source_commit": source_commit,
                "source_config_sha256": config_sha256,
                "sampler_version": SAMPLER_VERSION,
                "pipeline_initialization_count": initialized_pipeline_count,
                "completed": sum(bool(row["analysis_ok"]) for row in summaries),
                "failed": len(failures),
                "total": len(rows),
                "elapsed_seconds": time.perf_counter() - started_all,
                "authenticity_label_used": False,
            },
        )
        _write_csv_atomic(
            parent / "residual_failures.csv",
            failures,
            ("sample_id", "group_id", "split", "failure_reason"),
        )

    for index, row in enumerate(rows, 1):
        sample_started = time.perf_counter()
        sample_dir = output_root / safe_identity(row["sample_id"])
        video_path = Path(row["local_video_path"]).resolve()
        source_sha256 = row.get("source_video_sha256") or sha256_file(video_path)
        resumed, payload = _resume_valid(
            row,
            sample_dir,
            source_commit=source_commit,
            config_sha256=config_sha256,
            source_sha256=source_sha256,
        ) if arguments.resume else (False, None)
        try:
            if not resumed:
                result = pipeline.analyze_video(
                    video_path,
                    sampling_mode=arguments.sampling_mode,
                    sampled_frames=arguments.sampled_frames,
                    clip_length=arguments.clip_length,
                    clip_count=arguments.clip_count,
                )
                result.metadata.update(
                    {
                        "source_commit": source_commit,
                        "source_config_sha256": config_sha256,
                        "authenticity_label_used": False,
                        "batch_runner": "paper_core_r5b2",
                    }
                )
                provenance = _sampling_provenance(
                    row=row,
                    result_metadata=result.metadata,
                    source_sha256=source_sha256,
                    requested_frames=arguments.sampled_frames,
                    clip_length=arguments.clip_length,
                    clip_count=arguments.clip_count,
                )
                save_analysis_outputs(
                    result,
                    pipeline.last_observations,
                    sample_dir,
                    heatmap_sigma=float(config["localization"]["heatmap_sigma"]),
                )
                _write_json_atomic(sample_dir / "sampling_provenance.json", provenance)
                payload = json.loads((sample_dir / "result.json").read_text(encoding="utf-8"))
                sequence = _clip_sequence(payload)
                _write_json_atomic(sample_dir / "residual_sequence.json", sequence)
            else:
                provenance = json.loads(
                    (sample_dir / "sampling_provenance.json").read_text(encoding="utf-8")
                )
                sequence = json.loads(
                    (sample_dir / "residual_sequence.json").read_text(encoding="utf-8")
                )
            if payload is None:
                raise RuntimeError("Missing result payload.")
            metadata = payload["metadata"]
            if (
                int(metadata["selected_frame_count"]) != arguments.sampled_frames
                or int(metadata["clip_count"]) != arguments.clip_count
                or int(metadata.get("metric_depth_frames", 0)) != arguments.sampled_frames
                or int(metadata.get("intrinsics_frames", 0)) != arguments.sampled_frames
                or bool(metadata.get("authenticity_label_used", True))
                or sequence["channel_count"] != 12
                or sequence["channel_names"] != list(RESIDUAL_NAMES)
            ):
                raise ValueError("Residual smoke/output gate failed.")
            result_path = sample_dir / "result.json"
            provenance_path = sample_dir / "sampling_provenance.json"
            summary = _summary_row(
                row,
                payload,
                source_sha256=source_sha256,
                provenance_sha256=sha256_file(provenance_path),
                result_sha256=sha256_file(result_path),
                result_path=result_path,
                source_commit=source_commit,
                config_sha256=config_sha256,
            )
            summaries.append(summary)
            for clip in payload["clips"]:
                valid_names = {
                    residual["name"]
                    for residual in clip.get("residuals", ())
                    if residual.get("valid_mask", False)
                }
                for name in RESIDUAL_NAMES:
                    coverage[name]["total_clips"] += 1
                    coverage[name]["available_clips"] += int(name in valid_names)
                for residual in clip.get("residuals", ()):
                    if not residual.get("valid_mask", False):
                        unavailable_reasons[
                            str(residual.get("reason") or "unspecified")
                        ] += 1
            status = "OK_RESUME" if resumed else "OK"
        except KeyboardInterrupt:
            persist()
            raise
        except Exception as error:
            failure = f"{type(error).__name__}:{error}"
            failures.append(
                {
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "split": row["split"],
                    "failure_reason": failure,
                }
            )
            summaries.append(
                {
                    **{key: "" for key in SUMMARY_FIELDS},
                    "sample_id": row["sample_id"],
                    "group_id": row["group_id"],
                    "split": row["split"],
                    "label_for_posthoc_reference": row["label"],
                    "analysis_ok": False,
                    "source_video_sha256": source_sha256,
                    "failure_reason": failure,
                    "source_commit": source_commit,
                    "source_config_sha256": config_sha256,
                    "sampler_version": SAMPLER_VERSION,
                    "authenticity_label_used": False,
                }
            )
            status = "FAILED"
        persist()
        current = summaries[-1]
        print(
            f"[RESIDUAL] sample={row['sample_id']} group={row['group_id']} "
            f"split={row['split']} index={index}/{len(rows)} "
            f"frames={current.get('frames', 0) or 0} clips={current.get('clips', 0) or 0} "
            f"objects={current.get('objects', 0) or 0} "
            f"tracks={current.get('object_tracks', 0) or 0} "
            f"points={current.get('point_tracks', 0) or 0} "
            f"semantic_prior={current.get('semantic_prior_available', 0) or 0}/"
            f"{current.get('semantic_prior_total', 0) or 0} "
            f"semantic_temporal={current.get('semantic_temporal_available', 0) or 0}/"
            f"{current.get('semantic_temporal_total', 0) or 0} "
            f"d1={current.get('d1_available', 0) or 0}/{current.get('d1_total', 0) or 0} "
            f"d2={current.get('d2_available', 0) or 0}/{current.get('d2_total', 0) or 0} "
            f"d3={current.get('d3_available', 0) or 0}/{current.get('d3_total', 0) or 0} "
            f"coverage={current.get('overall_coverage', 0) or 0} "
            f"runtime={time.perf_counter()-sample_started:.3f} "
            f"gpu_peak={current.get('peak_gpu_memory_mb', 0) or 0} status={status}",
            flush=True,
        )
        if status == "FAILED":
            return 1
        if row["sample_id"] in equivalence_targets and not resumed:
            repeated = pipeline.analyze_video(
                video_path,
                sampling_mode=arguments.sampling_mode,
                sampled_frames=arguments.sampled_frames,
                clip_length=arguments.clip_length,
                clip_count=arguments.clip_count,
            )
            repeated.metadata.update(
                {
                    "source_commit": source_commit,
                    "source_config_sha256": config_sha256,
                    "authenticity_label_used": False,
                }
            )
            repeated_payload = {
                "clips": [
                    {
                        "clip_id": clip.clip_id,
                        "start_frame": clip.start_frame,
                        "residuals": [
                            {
                                "name": residual.name,
                                "normalized_value": (
                                    residual.normalized_value
                                    if residual.valid_mask
                                    else None
                                ),
                                "availability": residual.availability,
                                "valid_mask": residual.valid_mask,
                                "confidence": residual.confidence,
                            }
                            for residual in clip.residuals
                        ],
                    }
                    for clip in repeated.clip_results
                ],
                "metadata": repeated.metadata,
            }
            equivalence[row["sample_id"]] = _compare_payloads(payload, repeated_payload)
            _write_json_atomic(
                parent / "batch_single_equivalence.json",
                {
                    "samples": equivalence,
                    "tolerance": 1e-6,
                    "ready": len(equivalence) == len(equivalence_targets)
                    and all(value["ready"] for value in equivalence.values()),
                },
            )
        if index % 50 == 0:
            elapsed = time.perf_counter() - started_all
            average = elapsed / index
            print(
                f"[RESIDUAL_SUMMARY] completed={sum(bool(item['analysis_ok']) for item in summaries)} "
                f"failed={len(failures)} remaining={len(rows)-index} "
                f"success_rate={sum(bool(item['analysis_ok']) for item in summaries)/index:.6f} "
                f"average_runtime={average:.3f} "
                f"estimated_remaining_time={average*(len(rows)-index):.3f} "
                f"current_disk_usage={sum(path.stat().st_size for path in output_root.rglob('*') if path.is_file())}",
                flush=True,
            )
    _write_csv_atomic(
        parent / "residual_channel_coverage.csv",
        (
            {
                **row,
                "availability_rate": (
                    row["available_clips"] / row["total_clips"]
                    if row["total_clips"]
                    else 0.0
                ),
            }
            for row in coverage.values()
        ),
        ("channel_name", "total_clips", "available_clips", "availability_rate"),
    )
    _write_csv_atomic(
        parent / "unavailable_reason_counts.csv",
        (
            {"reason": reason, "count": count}
            for reason, count in sorted(unavailable_reasons.items())
        ),
        ("reason", "count"),
    )
    _write_csv_atomic(
        parent / "runtime_summary.csv",
        (
            {
                "sample_id": row["sample_id"],
                "runtime_seconds": row["runtime_seconds"],
                "peak_gpu_memory_mb": row["peak_gpu_memory_mb"],
            }
            for row in summaries
            if row["analysis_ok"]
        ),
        ("sample_id", "runtime_seconds", "peak_gpu_memory_mb"),
    )
    dataset_root = manifest.parents[2]
    final_manifest = (
        dataset_root
        / "manifests"
        / "aigvdbench_open_sora_paired_2k_residual_v1.csv"
        if len(rows) == 2000
        else None
    )
    group_summary = _quality_outputs(
        rows=rows,
        summaries=summaries,
        parent=parent,
        final_manifest_path=final_manifest,
    )
    _write_json_atomic(
        parent / "residual_extraction_summary.json",
        {
            "source_commit": source_commit,
            "source_config_sha256": config_sha256,
            "sampler_version": SAMPLER_VERSION,
            "pipeline_initialization_count": initialized_pipeline_count,
            "completed": sum(bool(row["analysis_ok"]) for row in summaries),
            "failed": len(failures),
            "total": len(rows),
            "elapsed_seconds": time.perf_counter() - started_all,
            "authenticity_label_used": False,
            "group_summary": group_summary,
            "batch_single_equivalence_ready": bool(equivalence)
            and all(value["ready"] for value in equivalence.values()),
            "final_manifest": "" if final_manifest is None else str(final_manifest),
            "final_manifest_sha256": (
                "" if final_manifest is None else sha256_file(final_manifest)
            ),
        },
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", required=True)
    result.add_argument("--config", default="configs/default.yaml")
    result.add_argument("--output", required=True)
    result.add_argument("--sampling-mode", default="uniform_full_video")
    result.add_argument("--sampled-frames", type=int, default=32)
    result.add_argument("--clip-length", type=int, default=8)
    result.add_argument("--clip-count", type=int, default=4)
    result.add_argument("--device", default="cuda")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--log-every", type=int, default=1)
    result.add_argument("--workers", type=int, default=1, choices=(1,))
    return result


def main() -> int:
    return run(parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
