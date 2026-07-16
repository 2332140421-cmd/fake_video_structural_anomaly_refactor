"""Build one shared 3D frame for future static and dynamic branches."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from ..depth_provider import DepthObservation
from ..geometry.camera import CameraObservation
from ..observations import FrameObservationJSON, ObjectObservationJSON
from ..shared_3d_observation import (
    Object3DObservation,
    Point2DObservation,
    Shared3DFrameObservation,
)
from .object_3d_reconstructor import Object3DReconstructor


def _keypoints_from_json(obj: ObjectObservationJSON) -> tuple[Point2DObservation, ...]:
    """Convert optional legacy JSON keypoint dictionaries into the P0 contract."""

    output: list[Point2DObservation] = []
    for index, record in enumerate(obj.keypoints_2d or []):
        name = str(record.get("keypoint_name", record.get("name", f"keypoint_{index}")))
        x = record.get("x")
        y = record.get("y")
        confidence = float(record.get("confidence", 1.0))
        valid = bool(record.get("valid", x is not None and y is not None))
        if valid and x is not None and y is not None:
            output.append(
                Point2DObservation(
                    point_id=f"{obj.object_id}:keypoint:{name}",
                    x=float(x),
                    y=float(y),
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    valid=True,
                    source=obj.pose_provider or "observation_json",
                    metadata={"keypoint_name": name},
                )
            )
        else:
            output.append(
                Point2DObservation(
                    point_id=f"{obj.object_id}:keypoint:{name}",
                    x=None,
                    y=None,
                    confidence=0.0,
                    valid=False,
                    missing_reason=str(record.get("missing_reason", "invalid_keypoint")),
                    source=obj.pose_provider or "observation_json",
                    metadata={"keypoint_name": name},
                )
            )
    return tuple(output)


class Shared3DFrameBuilder:
    """Use one immutable camera/depth pair for every object in a frame."""

    def __init__(self, reconstructor: Optional[Object3DReconstructor] = None) -> None:
        self.reconstructor = reconstructor or Object3DReconstructor()

    def build(
        self,
        *,
        video_id: str,
        frame: FrameObservationJSON,
        depth: DepthObservation,
        camera: CameraObservation,
        objects: Optional[Sequence[ObjectObservationJSON]] = None,
        keypoints_by_object: Optional[Mapping[str, Sequence[Point2DObservation]]] = None,
        boundary_points_by_object: Optional[
            Mapping[str, Sequence[Point2DObservation]]
        ] = None,
    ) -> Shared3DFrameObservation:
        """Build shared 3D evidence without letting one failed object abort the frame."""

        source_objects = tuple(frame.objects if objects is None else objects)
        if (frame.width, frame.height) != (camera.image_width, camera.image_height):
            raise ValueError("Camera dimensions must match FrameObservationJSON dimensions.")
        if depth.depth_map is not None and depth.depth_map.shape != (
            frame.height,
            frame.width,
        ):
            raise ValueError("DepthObservation shape must match frame dimensions.")
        if depth.frame_index is not None and depth.frame_index != frame.frame_index:
            raise ValueError("DepthObservation frame_index must match the source frame.")

        shared_metadata = {
            "source_frame_id": frame.frame_id,
            "source_image_path": frame.image_path,
            "camera_object_identity_shared": True,
            "depth_object_identity_shared": True,
            "static_dynamic_shared_contract": "Shared3DFrameObservation",
            "object_failures_are_retained": True,
        }
        try:
            depth.require_geometry_depth()
        except ValueError as error:
            return Shared3DFrameObservation.missing(
                video_id,
                frame.frame_index,
                frame.width,
                frame.height,
                camera,
                depth,
                reason="invalid_geometry_depth",
                source_frame_id=frame.frame_id,
                objects=(),
            )
        if not camera.valid or camera.K is None:
            return Shared3DFrameObservation.missing(
                video_id,
                frame.frame_index,
                frame.width,
                frame.height,
                camera,
                depth,
                reason="invalid_camera_intrinsics",
                source_frame_id=frame.frame_id,
                objects=(),
            )

        reconstructed: list[Object3DObservation] = []
        for obj in source_objects:
            try:
                keypoints = (
                    tuple(keypoints_by_object[obj.object_id])
                    if keypoints_by_object is not None
                    and obj.object_id in keypoints_by_object
                    else _keypoints_from_json(obj)
                )
                boundaries = (
                    tuple(boundary_points_by_object[obj.object_id])
                    if boundary_points_by_object is not None
                    and obj.object_id in boundary_points_by_object
                    else None
                )
                reconstructed.append(
                    self.reconstructor.reconstruct(
                        video_id=video_id,
                        frame_index=frame.frame_index,
                        obj=obj,
                        depth=depth,
                        camera=camera,
                        keypoints_2d=keypoints,
                        boundary_points_2d=boundaries,
                    )
                )
            except Exception as error:
                reconstructed.append(
                    Object3DObservation.missing(
                        video_id,
                        frame.frame_index,
                        obj.label,
                        obj.object_id,
                        track_id=obj.track_id or obj.person_track_id,
                        canonical_label=obj.canonical_label,
                        reason="object_reconstruction_error",
                        metadata={"error": str(error)},
                    )
                )

        valid_objects = [obj for obj in reconstructed if obj.valid]
        shared_metadata.update(
            {
                "object_count": len(reconstructed),
                "valid_object_count": len(valid_objects),
                "invalid_object_count": len(reconstructed) - len(valid_objects),
                "camera_pose_available": camera.pose_valid,
                "reconstruction_frame": "camera",
            }
        )
        if not valid_objects:
            return Shared3DFrameObservation.missing(
                video_id,
                frame.frame_index,
                frame.width,
                frame.height,
                camera,
                depth,
                reason="no_valid_object_reconstruction",
                source_frame_id=frame.frame_id,
                objects=tuple(reconstructed),
            )
        object_quality = float(
            np.mean([obj.reconstruction_quality for obj in valid_objects])
        )
        frame_quality = float(
            np.mean([camera.quality, depth.quality, object_quality])
        )
        return Shared3DFrameObservation(
            video_id=video_id,
            frame_index=frame.frame_index,
            image_width=frame.width,
            image_height=frame.height,
            camera=camera,
            depth=depth,
            objects=tuple(reconstructed),
            valid=True,
            quality=frame_quality,
            source_frame_id=frame.frame_id,
            metadata=shared_metadata,
        )


def build_shared_3d_frame_observation(
    *,
    video_id: str,
    frame: FrameObservationJSON,
    depth: DepthObservation,
    camera: CameraObservation,
    objects: Optional[Sequence[ObjectObservationJSON]] = None,
    keypoints_by_object: Optional[Mapping[str, Sequence[Point2DObservation]]] = None,
    boundary_points_by_object: Optional[Mapping[str, Sequence[Point2DObservation]]] = None,
    reconstructor: Optional[Object3DReconstructor] = None,
) -> Shared3DFrameObservation:
    """Functional wrapper around :class:`Shared3DFrameBuilder`."""

    return Shared3DFrameBuilder(reconstructor).build(
        video_id=video_id,
        frame=frame,
        depth=depth,
        camera=camera,
        objects=objects,
        keypoints_by_object=keypoints_by_object,
        boundary_points_by_object=boundary_points_by_object,
    )
