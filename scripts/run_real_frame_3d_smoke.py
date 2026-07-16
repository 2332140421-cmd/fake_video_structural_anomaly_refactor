#!/usr/bin/env python3
"""Build one camera-frame relative sparse 3D observation from a real video frame.

The current-frame projection cycle is geometry QA only. It is not independent
evidence and must not be interpreted as a forged-video anomaly residual.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic3d.depth_provider import BaseDepthProvider, RealDepthProvider  # noqa: E402
from semantic3d.geometry.camera import CameraObservation  # noqa: E402
from semantic3d.geometry.projection import project_point  # noqa: E402
from semantic3d.keypoint_provider import (  # noqa: E402
    BaseKeypointProvider,
    RealHumanKeypointProvider,
)
from semantic3d.observations import (  # noqa: E402
    FrameObservationJSON,
    ObjectObservationJSON,
)
from semantic3d.providers import BaseObjectProvider  # noqa: E402
from semantic3d.real_object_provider import RealObjectProvider  # noqa: E402
from semantic3d.reconstruction import Shared3DFrameBuilder  # noqa: E402
from semantic3d.shared_3d_observation import (  # noqa: E402
    Object3DObservation,
    Point2DObservation,
    Point3DObservation,
    Shared3DFrameObservation,
)
from semantic3d.static_3d import (  # noqa: E402
    ReconstructionQualityEvidence,
    reprojection_cycle_evidence,
)


def _json_safe(value: Any) -> Any:
    """Convert nested geometry metadata into JSON-safe values."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _point_payload(point: Optional[Point3DObservation]) -> Optional[dict[str, Any]]:
    if point is None:
        return None
    return {
        "point_id": point.point_id,
        "x": point.x,
        "y": point.y,
        "z": point.z,
        "coordinate_frame": point.coordinate_frame,
        "scale_status": point.scale_status.value,
        "confidence": point.confidence,
        "valid": point.valid,
        "missing_reason": point.missing_reason,
        "source_point_2d_id": point.source_point_2d_id,
        "metadata": _json_safe(point.metadata),
    }


def _object_payload(obj: Object3DObservation) -> dict[str, Any]:
    return {
        "video_id": obj.video_id,
        "frame_index": obj.frame_index,
        "track_id": obj.track_id,
        "semantic_label": obj.semantic_label,
        "canonical_label": obj.canonical_label,
        "source_object_2d_id": obj.source_object_2d_id,
        "center_3d_camera": _point_payload(obj.center_3d_camera),
        "center_3d_world": _point_payload(obj.center_3d_world),
        "boundary_points_3d": [_point_payload(point) for point in obj.boundary_points_3d],
        "keypoints_3d": [_point_payload(point) for point in obj.keypoints_3d],
        "structure_points_3d": [_point_payload(point) for point in obj.structure_points_3d],
        "observed_scale_3d": obj.observed_scale_3d,
        "scale_status": obj.scale_status.value,
        "scale_unit": obj.scale_unit.value,
        "scale_method": obj.scale_method,
        "reconstruction_frame": obj.reconstruction_frame.value,
        "visibility": obj.visibility.value,
        "reconstruction_quality": obj.reconstruction_quality,
        "valid": obj.valid,
        "missing_reason": obj.missing_reason,
        "metadata": _json_safe(obj.metadata),
    }


def _extract_frame(video_path: Path, frame_index: int, output_path: Path) -> np.ndarray:
    """Decode one global frame index and save the exact image used downstream."""

    if frame_index < 0:
        raise ValueError("frame_index must be non-negative.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, image = capture.read()
    finally:
        capture.release()
    if not success or image is None:
        raise ValueError(
            f"Could not decode frame_index={frame_index} from video: {video_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not save decoded frame: {output_path}")
    return image


def _approximate_camera(width: int, height: int, focal_length_factor: float) -> CameraObservation:
    """Create explicitly approximate intrinsics without fabricating camera pose."""

    if focal_length_factor <= 0.0:
        raise ValueError("focal_length_factor must be positive.")
    focal = float(focal_length_factor * max(width, height))
    K = np.asarray(
        [
            [focal, 0.0, (width - 1) / 2.0],
            [0.0, focal, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return CameraObservation.from_parameters(
        K=K,
        image_width=width,
        image_height=height,
        intrinsics_source="approximate",
        quality=0.5,
        pose_source="",
        metadata={
            "focal_length_factor": focal_length_factor,
            "metric_calibration": False,
            "warning": "Approximate K supports smoke-test camera-frame geometry only.",
        },
    )


def _keypoints_by_object(
    frame_path: Path,
    objects: Sequence[ObjectObservationJSON],
    provider: Optional[BaseKeypointProvider],
) -> tuple[dict[str, tuple[Point2DObservation, ...]], dict[str, str]]:
    """Collect optional person keypoints while retaining unsupported statuses."""

    point_map: dict[str, tuple[Point2DObservation, ...]] = {}
    statuses: dict[str, str] = {}
    if provider is None:
        return point_map, statuses
    for obj in objects:
        if obj.bbox is None:
            statuses[obj.object_id] = "missing_bbox"
            continue
        try:
            prediction = provider.predict(frame_path, obj.bbox, obj.label)
            statuses[obj.object_id] = prediction.status
            converted: list[Point2DObservation] = []
            for keypoint in prediction.keypoints:
                if keypoint.valid:
                    converted.append(
                        Point2DObservation(
                            point_id=f"{obj.object_id}:keypoint:{keypoint.keypoint_name}",
                            x=keypoint.x,
                            y=keypoint.y,
                            confidence=float(np.clip(keypoint.confidence, 0.0, 1.0)),
                            valid=True,
                            source=keypoint.provider_name,
                            metadata={"keypoint_name": keypoint.keypoint_name},
                        )
                    )
                else:
                    converted.append(
                        Point2DObservation(
                            point_id=f"{obj.object_id}:keypoint:{keypoint.keypoint_name}",
                            x=None,
                            y=None,
                            confidence=0.0,
                            valid=False,
                            missing_reason="invalid_keypoint",
                            source=keypoint.provider_name,
                            metadata={"keypoint_name": keypoint.keypoint_name},
                        )
                    )
            point_map[obj.object_id] = tuple(converted)
        except Exception as error:
            statuses[obj.object_id] = f"keypoint_error:{error}"
    return point_map, statuses


def _reprojection_rows(
    shared: Shared3DFrameObservation,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Project reconstructed points back to their recorded source coordinates."""

    rows: list[dict[str, Any]] = []
    mean_errors: dict[str, float] = {}
    if shared.camera.K is None:
        return rows, mean_errors
    for obj in shared.objects:
        errors: list[float] = []
        keypoint_ids = {point.point_id for point in obj.keypoints_3d}
        for point in obj.structure_points_3d:
            source_xy = point.metadata.get("source_point_xy")
            projected = project_point(point, shared.camera.K)
            source_u = source_v = reprojected_u = reprojected_v = None
            error_px = math.nan
            if (
                point.valid
                and projected.valid
                and source_xy is not None
                and len(source_xy) == 2
                and projected.x is not None
                and projected.y is not None
            ):
                source_u, source_v = (float(value) for value in source_xy)
                reprojected_u, reprojected_v = projected.x, projected.y
                error_px = float(
                    np.hypot(reprojected_u - source_u, reprojected_v - source_v)
                )
                errors.append(error_px)
            rows.append(
                {
                    "object_id": obj.source_object_2d_id,
                    "semantic_label": obj.semantic_label,
                    "point_id": point.point_id,
                    "point_group": (
                        "keypoint" if point.point_id in keypoint_ids else "boundary"
                    ),
                    "x_camera": point.x,
                    "y_camera": point.y,
                    "z_camera": point.z,
                    "point_valid": point.valid,
                    "point_quality": point.confidence,
                    "missing_reason": point.missing_reason,
                    "source_u": source_u,
                    "source_v": source_v,
                    "reprojected_u": reprojected_u,
                    "reprojected_v": reprojected_v,
                    "reconstruction_cycle_error_px": error_px,
                }
            )
        mean_errors[obj.source_object_2d_id] = (
            float(np.mean(errors)) if errors else math.nan
        )
    return rows, mean_errors


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_reprojection_plot(
    image: np.ndarray,
    objects_2d: Sequence[ObjectObservationJSON],
    point_rows: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    """Save source/reprojected point overlay; this plot is QA, not anomaly output."""

    figure, axis = plt.subplots(figsize=(10, 7))
    axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for obj in objects_2d:
        if obj.bbox is None:
            continue
        x1, y1, x2, y2 = (float(value) for value in obj.bbox)
        axis.add_patch(
            Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="lime", linewidth=1.5)
        )
        axis.text(
            x1,
            max(0.0, y1 - 3.0),
            f"{obj.object_id} ({obj.label})",
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 1.5},
        )
    source_label_used = False
    reprojection_label_used = False
    for row in point_rows:
        source_u, source_v = row["source_u"], row["source_v"]
        projected_u, projected_v = row["reprojected_u"], row["reprojected_v"]
        if None in (source_u, source_v, projected_u, projected_v):
            continue
        axis.scatter(
            source_u,
            source_v,
            s=22,
            marker="o",
            facecolors="none",
            edgecolors="yellow",
            label="source 2D point" if not source_label_used else None,
        )
        source_label_used = True
        axis.scatter(
            projected_u,
            projected_v,
            s=22,
            marker="x",
            color="red",
            label="same-point reprojection" if not reprojection_label_used else None,
        )
        reprojection_label_used = True
        axis.plot(
            [source_u, projected_u],
            [source_v, projected_v],
            color="cyan",
            linewidth=0.6,
        )
    axis.set_xlim(0, image.shape[1])
    axis.set_ylim(image.shape[0], 0)
    axis.set_title("Current-frame Reconstruction Cycle QA (Not Anomaly Evidence)")
    axis.set_xlabel("u (pixels)")
    axis.set_ylabel("v (pixels)")
    if source_label_used:
        axis.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_real_frame_3d_smoke(
    *,
    video_path: Path,
    frame_index: int,
    output_dir: Path,
    object_provider: Optional[BaseObjectProvider] = None,
    depth_provider: Optional[BaseDepthProvider] = None,
    keypoint_provider: Optional[BaseKeypointProvider] = None,
    model_path: Path = PROJECT_ROOT / "checkpoints" / "yolov8n.pt",
    pose_model_path: Path = PROJECT_ROOT / "checkpoints" / "yolov8n-pose.pt",
    depth_model: str = "depth-anything/Depth-Anything-V2-Small",
    confidence_threshold: float = 0.3,
    device: str = "cpu",
    focal_length_factor: float = 1.2,
) -> Shared3DFrameObservation:
    """Run the P1.5 single-frame reconstruction and save all acceptance artifacts."""

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = output_dir / "source_frame.png"
    image = _extract_frame(video_path, frame_index, frame_path)
    height, width = image.shape[:2]

    detector = object_provider or RealObjectProvider(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        default_depth=1.0,
        device=device,
        skip_unknown_scale_prior=False,
    )
    depth_estimator = depth_provider or RealDepthProvider(
        model_name=depth_model,
        device=device,
        invert_depth=True,
    )
    pose_estimator = keypoint_provider
    if pose_estimator is None and pose_model_path.exists():
        try:
            pose_estimator = RealHumanKeypointProvider(
                model_path=pose_model_path,
                device=device,
            )
        except Exception as error:
            print(f"Human keypoint provider unavailable; continuing without keypoints: {error}")

    objects = detector.predict(frame_path, frame_index, width, height)
    depth = depth_estimator.predict_observation(frame_path, frame_index=frame_index)
    depth.require_geometry_depth()
    if bool(depth.metadata.get("legacy_normalized_depth", False)):
        raise ValueError("Legacy [1,10] normalized depth is forbidden in P1.5 geometry.")
    camera = _approximate_camera(width, height, focal_length_factor)
    keypoint_map, keypoint_status = _keypoints_by_object(
        frame_path, objects, pose_estimator
    )
    frame = FrameObservationJSON(
        frame_index=frame_index,
        frame_id=f"{video_path.stem}_frame_{frame_index:06d}",
        width=width,
        height=height,
        objects=list(objects),
        image_path=str(frame_path),
        depth_metadata={
            "provider_name": depth.provider_name,
            "depth_representation": depth.depth_representation.value,
            "scale_status": depth.scale_status.value,
            "larger_value_means": depth.larger_value_means.value,
            "canonical_geometry_depth": True,
        },
    )
    shared = Shared3DFrameBuilder().build(
        video_id=video_path.stem,
        frame=frame,
        depth=depth,
        camera=camera,
        keypoints_by_object=keypoint_map,
    )

    point_rows, cycle_errors = _reprojection_rows(shared)
    quality_module = ReconstructionQualityEvidence()
    quality_records: list[dict[str, Any]] = []
    for obj in shared.objects:
        cycle_error = cycle_errors.get(obj.source_object_2d_id, math.nan)
        cycle = reprojection_cycle_evidence(
            cycle_error,
            source_ids=(obj.source_object_2d_id,),
            independent_observation=False,
        )
        quality = quality_module.evaluate(
            shared,
            obj.source_object_2d_id,
            reprojection_cycle_error=(cycle_error if math.isfinite(cycle_error) else None),
        )
        quality_records.append(
            {
                "object_id": obj.source_object_2d_id,
                "semantic_label": obj.semantic_label,
                "object_valid": obj.valid,
                "object_reconstruction_quality": obj.reconstruction_quality,
                "object_missing_reason": obj.missing_reason,
                "keypoint_status": keypoint_status.get(obj.source_object_2d_id, "not_requested"),
                "reconstruction_cycle_error": _json_safe(cycle.value),
                "cycle_valid": cycle.valid,
                "cycle_evidence_role": cycle.metadata.get("evidence_role"),
                "cycle_is_anomaly_residual": cycle.metadata.get("anomaly_residual"),
                "quality_value": _json_safe(quality.value),
                "quality_valid": quality.valid,
                "quality_missing_reason": quality.missing_reason,
                "quality_details": _json_safe(quality.metadata),
            }
        )

    payload = {
        "schema": "Shared3DFrameObservation/P1.5",
        "reconstruction_description": "camera-frame relative sparse 3D",
        "metric_reconstruction": False,
        "world_frame_available": camera.pose_valid,
        "video_id": shared.video_id,
        "frame_index": shared.frame_index,
        "source_frame_id": shared.source_frame_id,
        "image_width": shared.image_width,
        "image_height": shared.image_height,
        "valid": shared.valid,
        "quality": shared.quality,
        "missing_reason": shared.missing_reason,
        "camera": {
            "K": _json_safe(camera.K),
            "intrinsics_source": camera.intrinsics_source,
            "intrinsics_quality": camera.quality,
            "approximate_intrinsics": camera.intrinsics_source == "approximate",
            "pose_valid": camera.pose_valid,
            "T_world_from_camera": _json_safe(camera.T_world_from_camera),
            "T_camera_from_world": _json_safe(camera.T_camera_from_world),
            "coordinate_convention": camera.coordinate_convention.value,
            "depth_definition": camera.depth_definition.value,
            "pixel_center_convention": camera.pixel_center_convention.value,
            "metadata": _json_safe(camera.metadata),
        },
        "depth": {
            "provider_name": depth.provider_name,
            "depth_representation": depth.depth_representation.value,
            "scale_status": depth.scale_status.value,
            "larger_value_means": depth.larger_value_means.value,
            "canonical_geometry_depth": True,
            "legacy_normalized_depth_used": False,
            "quality": depth.quality,
            "valid": depth.valid,
            "missing_reason": depth.missing_reason,
            "metadata": _json_safe(depth.metadata),
        },
        "objects": [_object_payload(obj) for obj in shared.objects],
        "metadata": _json_safe(shared.metadata),
    }
    (output_dir / "shared_3d_frame.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    object_rows = [
        {
            "object_id": obj.source_object_2d_id,
            "semantic_label": obj.semantic_label,
            "canonical_label": obj.canonical_label,
            "track_id": obj.track_id,
            "center_x_camera": None if obj.center_3d_camera is None else obj.center_3d_camera.x,
            "center_y_camera": None if obj.center_3d_camera is None else obj.center_3d_camera.y,
            "center_z_camera": None if obj.center_3d_camera is None else obj.center_3d_camera.z,
            "center_world_available": obj.center_3d_world is not None,
            "observed_scale_3d": obj.observed_scale_3d,
            "scale_status": obj.scale_status.value,
            "scale_unit": obj.scale_unit.value,
            "reconstruction_quality": obj.reconstruction_quality,
            "valid": obj.valid,
            "missing_reason": obj.missing_reason,
        }
        for obj in shared.objects
    ]
    _write_csv(
        output_dir / "object_3d_observations.csv",
        object_rows,
        (
            "object_id",
            "semantic_label",
            "canonical_label",
            "track_id",
            "center_x_camera",
            "center_y_camera",
            "center_z_camera",
            "center_world_available",
            "observed_scale_3d",
            "scale_status",
            "scale_unit",
            "reconstruction_quality",
            "valid",
            "missing_reason",
        ),
    )
    point_fields = (
        "object_id",
        "semantic_label",
        "point_id",
        "point_group",
        "x_camera",
        "y_camera",
        "z_camera",
        "point_valid",
        "point_quality",
        "missing_reason",
        "source_u",
        "source_v",
        "reprojected_u",
        "reprojected_v",
        "reconstruction_cycle_error_px",
    )
    _write_csv(output_dir / "reconstructed_points_3d.csv", point_rows, point_fields)
    _save_reprojection_plot(
        image,
        objects,
        point_rows,
        output_dir / "current_frame_reprojection.png",
    )
    report = {
        "reconstruction_description": "camera-frame relative sparse 3D",
        "metric_reconstruction": False,
        "frame_valid": shared.valid,
        "frame_quality": shared.quality,
        "frame_missing_reason": shared.missing_reason,
        "object_count": len(shared.objects),
        "valid_object_count": sum(obj.valid for obj in shared.objects),
        "invalid_object_count": sum(not obj.valid for obj in shared.objects),
        "approximate_K": True,
        "camera_pose_available": camera.pose_valid,
        "canonical_geometry_depth": True,
        "legacy_normalized_depth_used": False,
        "depth_representation": depth.depth_representation.value,
        "depth_scale_status": depth.scale_status.value,
        "quality_records": quality_records,
    }
    (output_dir / "reconstruction_quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return shared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video_path",
        type=Path,
        default=PROJECT_ROOT / "data" / "tests_videos" / "tests_real_videos" / "real_1.mp4",
    )
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "real_frame_3d_smoke",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "yolov8n.pt",
    )
    parser.add_argument(
        "--pose_model_path",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "yolov8n-pose.pt",
    )
    parser.add_argument(
        "--depth_model",
        default="depth-anything/Depth-Anything-V2-Small",
    )
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--focal_length_factor", type=float, default=1.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shared = run_real_frame_3d_smoke(
        video_path=args.video_path,
        frame_index=args.frame_index,
        output_dir=args.output_dir,
        model_path=args.model_path,
        pose_model_path=args.pose_model_path,
        depth_model=args.depth_model,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        focal_length_factor=args.focal_length_factor,
    )
    print(f"reconstruction_description=camera-frame relative sparse 3D")
    print(f"frame_valid={shared.valid}")
    print(f"frame_quality={shared.quality:.6f}")
    print(f"objects={len(shared.objects)}")
    print(f"valid_objects={sum(obj.valid for obj in shared.objects)}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
