#!/usr/bin/env python3
"""Demonstrate the P1 synthetic projection/back-projection geometry closure."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic3d.depth_provider import (  # noqa: E402
    DepthObservation,
    DepthRepresentation,
    DepthScaleStatus,
    LargerValueMeans,
)
from semantic3d.geometry.camera import CameraObservation  # noqa: E402
from semantic3d.geometry.projection import project_points  # noqa: E402
from semantic3d.observations import ObjectObservationJSON  # noqa: E402
from semantic3d.reconstruction.object_3d_reconstructor import (  # noqa: E402
    Object3DReconstructor,
)
from semantic3d.shared_3d_observation import (  # noqa: E402
    GeometryScaleStatus,
    Point2DObservation,
    Point3DObservation,
)


def _point_to_dict(point: Point3DObservation | None) -> dict[str, Any] | None:
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
        "metadata": dict(point.metadata),
    }


def _synthetic_cube() -> np.ndarray:
    center = np.asarray([0.0, 0.0, 8.0])
    half = np.asarray([1.0, 1.0, 1.0])
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    return center + signs * half


def run_demo(output_dir: Path) -> dict[str, Any]:
    """Run a deterministic metric synthetic reconstruction and save artifacts."""

    width, height = 640, 480
    K = np.asarray(
        [[500.0, 0.0, 319.5], [0.0, 500.0, 239.5], [0.0, 0.0, 1.0]]
    )
    camera = CameraObservation.from_parameters(
        K=K,
        image_width=width,
        image_height=height,
        intrinsics_source="synthetic_ground_truth",
        quality=1.0,
        T_world_from_camera=np.eye(4),
        pose_source="synthetic_ground_truth",
        metadata={"synthetic_only": True},
    )
    ground_truth_xyz = _synthetic_cube()
    ground_truth_points = tuple(
        Point3DObservation(
            point_id=f"corner_{index}",
            x=float(point[0]),
            y=float(point[1]),
            z=float(point[2]),
            coordinate_frame="camera",
            scale_status=GeometryScaleStatus.METRIC_3D,
            confidence=1.0,
            valid=True,
            metadata={"synthetic_ground_truth": True},
        )
        for index, point in enumerate(ground_truth_xyz)
    )
    projected = project_points(ground_truth_points, K)

    depth_map = np.full((height, width), np.nan, dtype=np.float32)
    for pixel, point in zip(projected, ground_truth_xyz, strict=True):
        assert pixel.x is not None and pixel.y is not None
        column = int(np.floor(pixel.x + 0.5))
        row = int(np.floor(pixel.y + 0.5))
        depth_map[row - 1 : row + 2, column - 1 : column + 2] = float(point[2])
    valid_mask = np.isfinite(depth_map) & (depth_map > 0.0)
    depth = DepthObservation(
        depth_map=depth_map,
        raw_model_output=depth_map.copy(),
        visualization_depth=np.where(valid_mask, depth_map, np.nan),
        depth_representation=DepthRepresentation.METRIC_DEPTH,
        scale_status=DepthScaleStatus.METRIC_CALIBRATED,
        larger_value_means=LargerValueMeans.FARTHER,
        valid_mask=valid_mask,
        confidence_map=np.where(valid_mask, 1.0, 0.0).astype(np.float32),
        provider_name="synthetic_ground_truth",
        frame_index=0,
        valid=True,
        quality=1.0,
        metadata={
            "synthetic_only": True,
            "metric_scale_source": "synthetic_ground_truth",
        },
    )
    pixel_array = np.asarray([[point.x, point.y] for point in projected], dtype=float)
    obj = ObjectObservationJSON(
        object_id="synthetic_cube",
        label="synthetic_cuboid",
        mask_area=10_000.0,
        frame_area=float(width * height),
        depth=8.0,
        confidence=1.0,
        bbox=[
            float(pixel_array[:, 0].min()),
            float(pixel_array[:, 1].min()),
            float(pixel_array[:, 0].max()),
            float(pixel_array[:, 1].max()),
        ],
        track_id="synthetic_track_0",
        provenance={"source": "synthetic_ground_truth"},
    )
    boundary_2d = tuple(
        Point2DObservation(
            point_id=point.point_id,
            x=point.x,
            y=point.y,
            confidence=1.0,
            valid=True,
            source="synthetic_projection",
        )
        for point in projected
    )
    reconstructed = Object3DReconstructor().reconstruct(
        video_id="synthetic_video",
        frame_index=0,
        obj=obj,
        depth=depth,
        camera=camera,
        boundary_points_2d=boundary_2d,
    )
    if not reconstructed.valid or reconstructed.center_3d_camera is None:
        raise RuntimeError(f"Synthetic reconstruction failed: {reconstructed.missing_reason}")

    reconstructed_points = tuple(
        point for point in reconstructed.boundary_points_3d if point.valid
    )
    reprojected = project_points(reconstructed_points, K)
    reconstructed_xyz = np.stack([point.as_array() for point in reconstructed_points])
    reconstruction_error = np.linalg.norm(reconstructed_xyz - ground_truth_xyz, axis=1)
    reprojected_pixels = np.asarray(
        [[point.x, point.y] for point in reprojected], dtype=float
    )
    reprojection_error = np.linalg.norm(reprojected_pixels - pixel_array, axis=1)
    true_center = np.median(ground_truth_xyz, axis=0)
    center_error = float(
        np.linalg.norm(reconstructed.center_3d_camera.as_array() - true_center)
    )
    true_scale = float(np.linalg.norm(np.ptp(ground_truth_xyz, axis=0)))
    assert reconstructed.observed_scale_3d is not None
    scale_error = abs(reconstructed.observed_scale_3d - true_scale)

    output_dir.mkdir(parents=True, exist_ok=True)
    points_csv = output_dir / "synthetic_points.csv"
    with points_csv.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "point_id",
            "ground_truth_x",
            "ground_truth_y",
            "ground_truth_z",
            "projected_u",
            "projected_v",
            "reconstructed_x",
            "reconstructed_y",
            "reconstructed_z",
            "reprojected_u",
            "reprojected_v",
            "reconstruction_error_3d",
            "reprojection_error_px",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(ground_truth_xyz)):
            writer.writerow(
                {
                    "point_id": f"corner_{index}",
                    "ground_truth_x": ground_truth_xyz[index, 0],
                    "ground_truth_y": ground_truth_xyz[index, 1],
                    "ground_truth_z": ground_truth_xyz[index, 2],
                    "projected_u": pixel_array[index, 0],
                    "projected_v": pixel_array[index, 1],
                    "reconstructed_x": reconstructed_xyz[index, 0],
                    "reconstructed_y": reconstructed_xyz[index, 1],
                    "reconstructed_z": reconstructed_xyz[index, 2],
                    "reprojected_u": reprojected_pixels[index, 0],
                    "reprojected_v": reprojected_pixels[index, 1],
                    "reconstruction_error_3d": reconstruction_error[index],
                    "reprojection_error_px": reprojection_error[index],
                }
            )

    reconstructed_payload = {
        "video_id": reconstructed.video_id,
        "frame_index": reconstructed.frame_index,
        "semantic_label": reconstructed.semantic_label,
        "center_3d_camera": _point_to_dict(reconstructed.center_3d_camera),
        "center_3d_world": _point_to_dict(reconstructed.center_3d_world),
        "boundary_points_3d": [
            _point_to_dict(point) for point in reconstructed.boundary_points_3d
        ],
        "normalized_structure_points": [
            _point_to_dict(point) for point in reconstructed.normalized_structure_points
        ],
        "observed_scale_3d": reconstructed.observed_scale_3d,
        "scale_method": reconstructed.scale_method,
        "scale_unit": reconstructed.scale_unit.value,
        "scale_status": reconstructed.scale_status.value,
        "depth_scale_status": reconstructed.depth_scale_status.value,
        "scale_descriptors": dict(reconstructed.scale_descriptors),
        "reconstruction_quality": reconstructed.reconstruction_quality,
        "metadata": dict(reconstructed.metadata),
    }
    (output_dir / "reconstructed_object.json").write_text(
        json.dumps(reconstructed_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(pixel_array[:, 0], pixel_array[:, 1], marker="o", s=70, label="Projected GT")
    axis.scatter(
        reprojected_pixels[:, 0],
        reprojected_pixels[:, 1],
        marker="x",
        s=80,
        label="Reprojected reconstruction",
    )
    for original, recovered in zip(pixel_array, reprojected_pixels, strict=True):
        axis.plot([original[0], recovered[0]], [original[1], recovered[1]], color="0.4", linewidth=0.8)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.set_aspect("equal")
    axis.set_xlabel("u (pixels, right)")
    axis.set_ylabel("v (pixels, down)")
    axis.set_title("Synthetic 3D Reconstruction Reprojection Closure")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "reprojection.png", dpi=180)
    plt.close(figure)

    report = {
        "mean_3d_reconstruction_error": float(np.mean(reconstruction_error)),
        "max_3d_reconstruction_error": float(np.max(reconstruction_error)),
        "mean_reprojection_error_px": float(np.mean(reprojection_error)),
        "max_reprojection_error_px": float(np.max(reprojection_error)),
        "center_error": center_error,
        "scale_error": scale_error,
        "coordinate_convention": camera.coordinate_convention.value,
        "depth_definition": camera.depth_definition.value,
        "pixel_center_convention": camera.pixel_center_convention.value,
        "transform_convention": camera.transform_convention.value,
        "scale_status": reconstructed.scale_status.value,
        "scale_unit": reconstructed.scale_unit.value,
        "synthetic_only": True,
    }
    (output_dir / "geometry_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    output_dir = PROJECT_ROOT / "outputs" / "geometry_demo"
    report = run_demo(output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved synthetic geometry artifacts to {output_dir}")


if __name__ == "__main__":
    main()
