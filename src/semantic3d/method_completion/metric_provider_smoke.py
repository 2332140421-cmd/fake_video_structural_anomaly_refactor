"""Small, label-blind quality smoke for a real metric-depth provider."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import cv2
import numpy as np
import pandas as pd
import yaml

from .metric_depth_adapters import (
    METRIC_DEPTH_ADAPTERS,
    BaseMetricDepthAdapter,
    MetricDepthFrameResult,
    MetricProviderRuntimeError,
    get_metric_depth_adapter,
    resolve_weight_file,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SmokeFrameSpec:
    """One deterministic video frame selected without using a class label."""

    source_name: str
    video_path: Path
    frame_index: int
    clip_id: str


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _project_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256_text(path: Path) -> str:
    return sha256_file(path)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def load_smoke_config(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the P4-C3B smoke configuration."""

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("stage") != "P4-C3B-M1":
        raise ValueError("Expected a P4-C3B-M1 mapping configuration.")
    if not payload.get("inputs"):
        raise ValueError("Smoke configuration requires at least one input video.")
    return payload


def build_smoke_frame_specs(config: Mapping[str, Any]) -> list[SmokeFrameSpec]:
    """Create stable frame specs in source-name/frame-index order."""

    specs: list[SmokeFrameSpec] = []
    for source in sorted(config["inputs"], key=lambda item: str(item["source_name"])):
        source_name = str(source["source_name"])
        video_path = PROJECT_ROOT / str(source["video_path"])
        for frame_index in sorted({int(item) for item in source["frame_indices"]}):
            specs.append(
                SmokeFrameSpec(
                    source_name=source_name,
                    video_path=video_path,
                    frame_index=frame_index,
                    clip_id=f"{source_name}_metric_smoke_clip_000",
                )
            )
    return specs


def _extract_frame(spec: SmokeFrameSpec, output_dir: Path) -> Path:
    if not spec.video_path.is_file():
        raise FileNotFoundError(f"Smoke video is missing: {spec.video_path}")
    frame_path = output_dir / "frames" / spec.source_name / f"frame_{spec.frame_index:06d}.jpg"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(spec.video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, spec.frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError(
            f"Could not decode frame {spec.frame_index} from {spec.video_path}."
        )
    if not cv2.imwrite(str(frame_path), frame):
        raise OSError(f"Could not save decoded smoke frame: {frame_path}")
    return frame_path


def _finite_stats(values: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)[valid_mask]
    if finite.size == 0:
        return {"min": None, "median": None, "max": None, "mean": None}
    return {
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _boundary_mask(shape: tuple[int, int], fraction: float) -> np.ndarray:
    height, width = shape
    border_y = max(1, int(round(height * fraction)))
    border_x = max(1, int(round(width * fraction)))
    mask = np.zeros(shape, dtype=bool)
    mask[:border_y] = True
    mask[-border_y:] = True
    mask[:, :border_x] = True
    mask[:, -border_x:] = True
    return mask


def _rank_array(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def _rank_correlation(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    *,
    stride: int,
) -> float:
    sampled = mask.copy()
    grid = np.zeros(mask.shape, dtype=bool)
    grid[:: max(1, stride), :: max(1, stride)] = True
    sampled &= grid
    a = np.asarray(first, dtype=np.float64)[sampled]
    b = np.asarray(second, dtype=np.float64)[sampled]
    if a.size < 3:
        return float("nan")
    rank_a = _rank_array(a)
    rank_b = _rank_array(b)
    if np.std(rank_a) <= 0.0 or np.std(rank_b) <= 0.0:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


class _ReferenceObservations:
    def __init__(self, dataset_root: Path) -> None:
        self.root = dataset_root
        self.video_ids: dict[str, str] = {}
        self.depth = pd.DataFrame()
        self.masks = pd.DataFrame()
        videos_path = dataset_root / "manifests" / "videos.parquet"
        if videos_path.is_file():
            videos = pd.read_parquet(videos_path)
            self.video_ids = {
                str(row.source_name): str(row.video_id) for row in videos.itertuples()
            }
        depth_path = dataset_root / "observations" / "depth.parquet"
        masks_path = dataset_root / "observations" / "masks.parquet"
        if depth_path.is_file():
            self.depth = pd.read_parquet(depth_path)
        if masks_path.is_file():
            self.masks = pd.read_parquet(masks_path)

    @staticmethod
    def _load_npz_array(path: str | Path, preferred: Iterable[str]) -> Optional[np.ndarray]:
        array_path = Path(path)
        if not array_path.is_file():
            return None
        with np.load(array_path, allow_pickle=False) as archive:
            for name in preferred:
                if name in archive:
                    return np.asarray(archive[name])
        return None

    def relative_depth(self, source_name: str, frame_index: int) -> Optional[np.ndarray]:
        video_id = self.video_ids.get(source_name)
        if not video_id or self.depth.empty:
            return None
        rows = self.depth[
            (self.depth["video_id"] == video_id)
            & (self.depth["frame_index"].astype(int) == int(frame_index))
            & (self.depth["valid"].astype(bool))
        ]
        if rows.empty:
            return None
        return self._load_npz_array(rows.iloc[0]["array_path"], ("depth_map", "relative_depth"))

    def masks_for_frame(self, source_name: str, frame_index: int) -> list[dict[str, Any]]:
        video_id = self.video_ids.get(source_name)
        if not video_id or self.masks.empty:
            return []
        rows = self.masks[
            (self.masks["video_id"] == video_id)
            & (self.masks["frame_index"].astype(int) == int(frame_index))
            & (self.masks["valid"].astype(bool))
            & (~self.masks["bbox_fallback"].astype(bool))
        ]
        output: list[dict[str, Any]] = []
        for row in rows.sort_values("object_track_id").to_dict("records"):
            mask = self._load_npz_array(row["array_path"], ("visible_mask", "mask"))
            if mask is not None:
                output.append({**row, "mask": np.asarray(mask, dtype=bool)})
        return output


def _save_frame_arrays(
    output_dir: Path,
    spec: SmokeFrameSpec,
    result: MetricDepthFrameResult,
) -> dict[str, str]:
    array_dir = output_dir / "arrays" / spec.source_name / f"frame_{spec.frame_index:06d}"
    array_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "depth_m_path": ("depth_m.npy", result.depth_observation.depth_map),
        "valid_mask_path": ("valid_mask.npy", result.depth_observation.valid_mask),
        "confidence_path": ("confidence_rank_quality.npy", result.depth_observation.confidence_map),
        "uncertainty_path": ("unidepth_log_error.npy", result.uncertainty_map),
        "raw_radius_path": ("raw_radius_m.npy", result.depth_observation.raw_model_output),
        "intrinsics_path": ("intrinsics_K.npy", result.camera_observation.K),
    }
    paths: dict[str, str] = {}
    for field, (filename, array) in arrays.items():
        path = array_dir / filename
        if array is not None:
            np.save(path, np.asarray(array), allow_pickle=False)
            paths[field] = _project_path(path)
        else:
            paths[field] = ""
    return paths


def _software_environment(adapter: BaseMetricDepthAdapter) -> dict[str, Any]:
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
    except ImportError:
        torch_version = "unavailable"
        cuda_available = False
        cuda_version = None
    versions: dict[str, str] = {}
    for package in ("unidepth", "torch", "torchvision", "timm", "opencv-python"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    adapters = []
    for name, adapter_class in sorted(METRIC_DEPTH_ADAPTERS.items()):
        if name == "unidepth_v2":
            descriptor = adapter.describe()
        else:
            descriptor = adapter_class().describe()
        adapters.append(descriptor.to_dict())
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": versions,
        "torch_version": torch_version,
        "torch_cuda_version": cuda_version,
        "cuda_available": cuda_available,
        "selected_adapter": adapter.describe().to_dict(),
        "all_metric_adapters": adapters,
        "offline_only": True,
    }


def run_metric_provider_smoke(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    adapter: Optional[BaseMetricDepthAdapter] = None,
) -> dict[str, Any]:
    """Run a small real-inference quality smoke without authenticity scoring."""

    config_file = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_smoke_config(config_file)
    provider_config = config["provider"]
    if adapter is None:
        adapter = get_metric_depth_adapter(
            provider_config["name"],
            weights_path=PROJECT_ROOT / provider_config["weights_path"],
            device=provider_config.get("device", "cpu"),
            precision=provider_config.get("precision", "fp32"),
            expected_weight_sha256=provider_config["expected_weight_sha256"],
            resolution_level=int(provider_config.get("resolution_level", 0)),
        )
    config_sha = _sha256_text(config_file)
    software_commit = _git_commit()
    reference_root = PROJECT_ROOT / config["reference_observations"]["dataset_root"]
    references = _ReferenceObservations(reference_root)
    boundary_fraction = float(config["quality_audit"].get("boundary_fraction", 0.05))
    rank_stride = int(config["quality_audit"].get("rank_sample_stride", 8))

    environment_path = output / "provider_environment.json"
    _write_json(environment_path, _software_environment(adapter))
    weight_file = resolve_weight_file(adapter.weights_path)
    weight_manifest = {
        "provider_name": adapter.provider_name,
        "model_family": adapter.expected_model_family,
        "weights_path": "" if weight_file is None else _project_path(weight_file),
        "file_size": None if weight_file is None else weight_file.stat().st_size,
        "sha256": "" if weight_file is None else sha256_file(weight_file),
        "expected_sha256": adapter.expected_weight_sha256,
        "sha256_verified": adapter.describe().weight_hash_verified,
        "automatic_download_allowed": False,
        "sensor_ground_truth": False,
        "license_registry_path": "configs/model_registry/unidepth_v2_vits14_v1.yaml",
    }
    weight_manifest_path = output / "provider_weight_manifest.json"
    _write_json(weight_manifest_path, weight_manifest)

    frame_rows: list[dict[str, Any]] = []
    numeric_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    frame_results: dict[tuple[str, int], MetricDepthFrameResult] = {}
    for spec in build_smoke_frame_specs(config):
        frame_id = f"{spec.source_name}_frame_{spec.frame_index:06d}"
        try:
            frame_path = _extract_frame(spec, output)
            result = adapter.predict_frame(frame_path, frame_index=spec.frame_index)
            frame_results[(spec.source_name, spec.frame_index)] = result
            depth = np.asarray(result.depth_observation.depth_map, dtype=np.float32)
            valid = np.asarray(result.depth_observation.valid_mask, dtype=bool)
            uncertainty = np.asarray(result.uncertainty_map, dtype=np.float32)
            paths = _save_frame_arrays(output, spec, result)
            K = np.asarray(result.camera_observation.K, dtype=float)
            frame_row = {
                "video_id": spec.source_name,
                "clip_id": spec.clip_id,
                "frame_id": frame_id,
                "frame_index": spec.frame_index,
                "provider_name": adapter.provider_name,
                "provider_version": result.provider_version,
                "depth_type": "metric",
                "depth_unit": "meter",
                "depth_definition": result.standardized_depth_definition,
                **paths,
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
                "cx": float(K[0, 2]),
                "cy": float(K[1, 2]),
                "intrinsics_source": result.camera_observation.intrinsics_source,
                "intrinsics_confidence": result.intrinsics_confidence,
                "metric_scale_status": "model_predicted",
                "provider_status": result.provider_status,
                "failure_reason": result.failure_reason,
                "runtime_seconds": result.runtime_seconds,
                "peak_gpu_memory": result.peak_gpu_memory_bytes,
                "weight_sha256": result.weight_sha256,
                "config_sha256": config_sha,
                "software_commit": software_commit,
            }
            frame_rows.append(frame_row)
            boundary = _boundary_mask(depth.shape, boundary_fraction)
            relative = references.relative_depth(spec.source_name, spec.frame_index)
            rank_valid = valid.copy()
            rank_correlation = float("nan")
            if relative is not None and relative.shape == depth.shape:
                rank_valid &= np.isfinite(relative) & (relative > 0.0)
                rank_correlation = _rank_correlation(
                    depth, relative, rank_valid, stride=rank_stride
                )
            numeric_rows.append(
                {
                    "video_id": spec.source_name,
                    "frame_index": spec.frame_index,
                    "valid_depth_ratio": float(np.mean(valid)),
                    "nan_count": int(np.isnan(depth).sum()),
                    "inf_count": int(np.isinf(depth).sum()),
                    **_finite_stats(depth, valid),
                    "uncertainty_median": float(np.median(uncertainty[valid])),
                    "boundary_valid_ratio": float(np.mean(valid[boundary])),
                    "boundary_depth_median": float(np.median(depth[valid & boundary])),
                    "relative_metric_rank_correlation": rank_correlation,
                    "z_not_greater_than_radius_ratio": float(
                        np.mean(
                            depth[valid]
                            <= np.asarray(result.depth_observation.raw_model_output)[valid] + 1e-5
                        )
                    ),
                }
            )
            masks = references.masks_for_frame(spec.source_name, spec.frame_index)
            if not masks:
                object_rows.append(
                    {
                        "video_id": spec.source_name,
                        "frame_index": spec.frame_index,
                        "object_track_id": "",
                        "class_name": "",
                        "formal_mask_available": False,
                        "mask_pixel_count": 0,
                        "valid_depth_ratio": float("nan"),
                        "depth_median": float("nan"),
                        "confidence_rank_quality_median": float("nan"),
                        "status": "blocked_by_input",
                        "failure_reason": "formal_instance_mask_unavailable",
                    }
                )
            for mask_record in masks:
                mask = mask_record.pop("mask")
                if mask.shape != depth.shape:
                    continue
                pixels = int(mask.sum())
                inside_valid = valid & mask
                object_rows.append(
                    {
                        "video_id": spec.source_name,
                        "frame_index": spec.frame_index,
                        "object_track_id": mask_record["object_track_id"],
                        "class_name": mask_record["class_name"],
                        "formal_mask_available": True,
                        "mask_pixel_count": pixels,
                        "valid_depth_ratio": (
                            float(inside_valid.sum() / pixels) if pixels else float("nan")
                        ),
                        "depth_median": (
                            float(np.median(depth[inside_valid]))
                            if np.any(inside_valid)
                            else float("nan")
                        ),
                        "confidence_rank_quality_median": (
                            float(
                                np.median(
                                    result.depth_observation.confidence_map[inside_valid]
                                )
                            )
                            if np.any(inside_valid)
                            else float("nan")
                        ),
                        "status": "executed_valid" if np.any(inside_valid) else "executed_invalid",
                        "failure_reason": "" if np.any(inside_valid) else "no_valid_depth_in_mask",
                    }
                )
        except (MetricProviderRuntimeError, FileNotFoundError, ValueError, OSError) as exc:
            reason = getattr(exc, "reason", "metric_provider_smoke_input_failed")
            failures.append(
                {
                    "video_id": spec.source_name,
                    "frame_index": spec.frame_index,
                    "provider_status": "provider_failed",
                    "failure_reason": reason,
                    "message": str(exc),
                }
            )
            frame_rows.append(
                {
                    "video_id": spec.source_name,
                    "clip_id": spec.clip_id,
                    "frame_id": frame_id,
                    "frame_index": spec.frame_index,
                    "provider_name": adapter.provider_name,
                    "provider_status": "provider_failed",
                    "failure_reason": reason,
                    "config_sha256": config_sha,
                    "software_commit": software_commit,
                }
            )

    repeatability: dict[str, Any] = {"attempted": False}
    if frame_results and bool(config["quality_audit"].get("repeat_first_frame", True)):
        first_spec = build_smoke_frame_specs(config)[0]
        key = (first_spec.source_name, first_spec.frame_index)
        if key in frame_results:
            repeatability["attempted"] = True
            try:
                frame_path = output / "frames" / first_spec.source_name / f"frame_{first_spec.frame_index:06d}.jpg"
                repeated = adapter.predict_frame(frame_path, frame_index=first_spec.frame_index)
                original = frame_results[key].depth_observation.depth_map
                comparison = repeated.depth_observation.depth_map
                delta = np.abs(np.asarray(original) - np.asarray(comparison))
                repeatability.update(
                    {
                        "status": "executed_valid",
                        "max_abs_depth_difference": float(np.nanmax(delta)),
                        "mean_abs_depth_difference": float(np.nanmean(delta)),
                        "deterministic_within_1e_5": bool(np.nanmax(delta) <= 1e-5),
                    }
                )
            except Exception as exc:  # repeat audit must not hide the main smoke
                repeatability.update(
                    {"status": "provider_failed", "failure_reason": str(exc)}
                )

    frame_manifest_path = output / "metric_depth_frame_manifest.csv"
    frame_fields = [
        "video_id", "clip_id", "frame_id", "frame_index", "provider_name",
        "provider_version", "depth_type", "depth_unit", "depth_definition",
        "depth_m_path", "valid_mask_path", "confidence_path", "uncertainty_path",
        "raw_radius_path", "intrinsics_path", "fx", "fy", "cx", "cy",
        "intrinsics_source", "metric_scale_status", "provider_status", "failure_reason",
        "intrinsics_confidence",
        "runtime_seconds", "peak_gpu_memory", "weight_sha256", "config_sha256",
        "software_commit",
    ]
    _write_csv(frame_manifest_path, frame_rows, frame_fields)
    object_path = output / "object_region_depth_audit.csv"
    object_fields = [
        "video_id", "frame_index", "object_track_id", "class_name",
        "formal_mask_available", "mask_pixel_count", "valid_depth_ratio",
        "depth_median", "confidence_rank_quality_median", "status", "failure_reason",
    ]
    _write_csv(object_path, object_rows, object_fields)

    temporal_by_video: dict[str, Any] = {}
    intrinsics_by_video: dict[str, Any] = {}
    for source_name in sorted({key[0] for key in frame_results}):
        ordered = [
            (index, frame_results[(name, index)])
            for name, index in sorted(frame_results)
            if name == source_name
        ]
        medians = [
            float(np.median(result.depth_observation.depth_map[result.depth_observation.valid_mask]))
            for _, result in ordered
        ]
        focal = [
            [
                float(result.camera_observation.K[0, 0]),
                float(result.camera_observation.K[1, 1]),
                float(result.camera_observation.K[0, 2]),
                float(result.camera_observation.K[1, 2]),
            ]
            for _, result in ordered
        ]
        log_drift = (
            [abs(math.log(value / medians[0])) for value in medians]
            if medians and medians[0] > 0.0
            else []
        )
        temporal_by_video[source_name] = {
            "frame_indices": [index for index, _ in ordered],
            "global_depth_medians_m": medians,
            "absolute_log_scale_drift_from_first": log_drift,
            "mean_absolute_log_scale_drift": (
                float(np.mean(log_drift)) if log_drift else None
            ),
            "max_absolute_log_scale_drift": (
                float(np.max(log_drift)) if log_drift else None
            ),
        }
        focal_array = np.asarray(focal, dtype=float)
        if focal_array.size:
            normalized = np.abs((focal_array - focal_array[0]) / np.maximum(np.abs(focal_array[0]), 1e-8))
            intrinsics_by_video[source_name] = {
                "frame_indices": [index for index, _ in ordered],
                "intrinsics_fx_fy_cx_cy": focal,
                "max_relative_parameter_drift": float(np.max(normalized)),
                "mean_relative_parameter_drift": float(np.mean(normalized)),
                "intrinsics_source": "model_predicted",
                "calibrated": False,
            }

    numeric_path = output / "metric_depth_numeric_audit.json"
    valid_numeric = [row for row in numeric_rows if row["valid_depth_ratio"] > 0.0]
    _write_json(
        numeric_path,
        {
            "frame_audits": numeric_rows,
            "repeatability": repeatability,
            "summary": {
                "frames_requested": len(build_smoke_frame_specs(config)),
                "frames_valid": len(valid_numeric),
                "mean_valid_depth_ratio": (
                    float(np.mean([row["valid_depth_ratio"] for row in valid_numeric]))
                    if valid_numeric
                    else None
                ),
                "total_nan": int(sum(row["nan_count"] for row in numeric_rows)),
                "total_inf": int(sum(row["inf_count"] for row in numeric_rows)),
                "depth_is_model_predicted_metric_not_sensor_truth": True,
            },
        },
    )
    intrinsics_path = output / "intrinsics_audit.json"
    _write_json(
        intrinsics_path,
        {
            "source": "model_predicted",
            "quality_tier": "model_predicted_unscored",
            "confidence_available": False,
            "confidence_reason": "provider_does_not_expose_intrinsics_confidence",
            "calibrated": False,
            "per_video": intrinsics_by_video,
        },
    )
    temporal_path = output / "temporal_scale_drift.json"
    _write_json(
        temporal_path,
        {
            "audit_only": True,
            "not_authenticity_evidence": True,
            "per_video": temporal_by_video,
        },
    )
    failure_path = output / "provider_failure_audit.json"
    _write_json(
        failure_path,
        {
            "attempted_frames": len(build_smoke_frame_specs(config)),
            "provider_failure_count": len(failures),
            "provider_failures_are_not_anomaly_evidence": True,
            "failures": failures,
        },
    )

    valid_count = sum(row.get("provider_status") == "executed_valid" for row in frame_rows)
    intrinsics_available = bool(valid_count) and all(
        row.get("intrinsics_source") == "model_predicted"
        for row in frame_rows
        if row.get("provider_status") == "executed_valid"
    )
    z_verified = bool(numeric_rows) and all(
        row["z_not_greater_than_radius_ratio"] >= 0.999 for row in numeric_rows
    )
    statuses = {
        "metric_provider_adapter_complete": True,
        "metric_provider_real_inference_executed": valid_count > 0,
        "metric_depth_output_verified": valid_count == len(build_smoke_frame_specs(config)),
        "depth_definition_verified": z_verified,
        "intrinsics_output_available": intrinsics_available,
        "intrinsics_quality": (
            "model_predicted_unscored" if intrinsics_available else "unavailable"
        ),
        "metric_depth_temporal_stability_audited": bool(temporal_by_video),
        "metric_single_object_ready_for_execution": bool(valid_count and intrinsics_available),
        "ready_for_metric_scene3d_build": bool(valid_count and intrinsics_available and z_verified),
        "ready_for_full_984_frame_build": False,
        "method_effectiveness_established": False,
    }
    report_path = output / "METRIC_PROVIDER_SMOKE_REPORT.md"
    report_lines = [
        "# P4-C3B-M1 Metric Provider Smoke Report",
        "",
        "## Scope",
        "",
        f"- Provider: `{adapter.provider_name}`",
        f"- Requested frames: {len(build_smoke_frame_specs(config))}",
        f"- Valid frames: {valid_count}",
        "- Depth is monocular model-predicted metric Z depth in meters, not sensor ground truth.",
        "- Predicted intrinsics are not calibrated camera metadata.",
        "- No authenticity metric, learned distribution, threshold, or model training was produced.",
        "",
        "## Status",
        "",
    ]
    report_lines.extend(f"- `{name}`: `{str(value).lower()}`" for name, value in statuses.items())
    report_lines.extend(
        [
            "",
            "## Quality Notes",
            "",
            "- UniDepth raw ray radius is retained; canonical geometry uses `z_depth`.",
            "- The model error map is saved as uncertainty. Confidence is a frame-relative rank quality, not a probability.",
            "- Temporal depth and focal drift are upstream quality diagnostics only.",
            "- Full 984-frame construction remains blocked pending runtime/storage review and broader stability checks.",
            "",
        ]
    )
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    artifact_paths = [
        environment_path,
        weight_manifest_path,
        frame_manifest_path,
        numeric_path,
        intrinsics_path,
        temporal_path,
        object_path,
        failure_path,
        report_path,
    ]
    validation = {
        **statuses,
        "stage": "P4-C3B-M1",
        "config_path": _project_path(config_file),
        "config_sha256": config_sha,
        "software_commit": software_commit,
        "requested_frame_count": len(build_smoke_frame_specs(config)),
        "executed_valid_frame_count": valid_count,
        "provider_failure_count": len(failures),
        "artifact_sha256": {
            _project_path(path): sha256_file(path) for path in sorted(artifact_paths)
        },
        "frozen_inputs_modified": False,
    }
    validation_path = output / "validation_report.json"
    _write_json(validation_path, validation)
    return validation
