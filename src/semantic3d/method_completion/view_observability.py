"""Camera/object view contracts and dimension-wise observability gates.

The contracts in this module do not estimate authenticity. They describe
whether a physical object dimension is supported by the current visible
observation. Unknown viewpoint or incomplete visibility remains explicit and
never becomes a zero residual.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..geometry.camera import CameraObservation


class ViewpointClass(str, Enum):
    """Coarse object-to-camera viewpoint classes."""

    FRONTAL = "frontal"
    LATERAL = "lateral"
    OBLIQUE = "oblique"
    TOP_DOWN = "top_down"
    BOTTOM_UP = "bottom_up"
    UNKNOWN = "unknown"


class CameraMotionClass(str, Enum):
    """Camera/scene motion state used by the view-scale route."""

    STATIC = "static"
    LOW_MOTION = "low_motion"
    OBJECT_MOTION = "object_motion"
    CAMERA_MOTION = "camera_motion"
    MIXED_MOTION = "mixed_motion"
    MOTION_UNRELIABLE = "motion_unreliable"


class PoseEstimateStatus(str, Enum):
    """Reliability of an object orientation or pose estimate."""

    RESOLVED = "resolved"
    UPRIGHT_FULL_BODY = "upright_full_body"
    UPRIGHT_SHAPE_COMPATIBLE = "upright_shape_compatible"
    SITTING = "sitting"
    BENDING = "bending"
    KNEELING = "kneeling"
    UNAVAILABLE = "unavailable"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class CameraViewObservation:
    """Pinhole view information used by per-frame observability decisions."""

    frame_id: str
    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int
    fov_x: float
    fov_y: float
    distortion_status: str
    intrinsics_source: str
    intrinsics_confidence: float
    camera_motion_class: CameraMotionClass | str
    pose_status: str
    image_transform_chain: tuple[Mapping[str, Any], ...]
    valid: bool
    failure_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        motion = CameraMotionClass(self.camera_motion_class)
        confidence = float(self.intrinsics_confidence)
        if not math.isnan(confidence) and (
            not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("intrinsics_confidence must be NaN or in [0, 1].")
        finite = all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in (self.fx, self.fy, self.fov_x, self.fov_y)
        )
        if self.valid:
            if self.image_width <= 0 or self.image_height <= 0 or not finite:
                raise ValueError("Valid CameraViewObservation requires finite intrinsics.")
            if self.failure_reason:
                raise ValueError("Valid CameraViewObservation cannot have failure_reason.")
        elif not self.failure_reason:
            raise ValueError("Invalid CameraViewObservation requires failure_reason.")
        object.__setattr__(self, "camera_motion_class", motion)
        object.__setattr__(self, "intrinsics_confidence", confidence)
        object.__setattr__(
            self,
            "image_transform_chain",
            tuple(dict(item) for item in self.image_transform_chain),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_camera(
        cls,
        frame_id: str,
        camera: CameraObservation,
        *,
        camera_motion_class: CameraMotionClass | str,
        intrinsics_confidence: float = float("nan"),
        image_transform_chain: Sequence[Mapping[str, Any]] = (),
        distortion_status: str = "unreported",
    ) -> "CameraViewObservation":
        """Create a view contract from canonical camera intrinsics."""

        if not camera.valid or camera.K is None:
            return cls(
                frame_id=frame_id,
                fx=float("nan"),
                fy=float("nan"),
                cx=float("nan"),
                cy=float("nan"),
                image_width=camera.image_width,
                image_height=camera.image_height,
                fov_x=float("nan"),
                fov_y=float("nan"),
                distortion_status=distortion_status,
                intrinsics_source=camera.intrinsics_source,
                intrinsics_confidence=intrinsics_confidence,
                camera_motion_class=camera_motion_class,
                pose_status="unavailable",
                image_transform_chain=tuple(image_transform_chain),
                valid=False,
                failure_reason=camera.missing_reason or "camera_intrinsics_unavailable",
            )
        fx, fy = float(camera.K[0, 0]), float(camera.K[1, 1])
        cx, cy = float(camera.K[0, 2]), float(camera.K[1, 2])
        fov_x = math.degrees(2.0 * math.atan(camera.image_width / (2.0 * fx)))
        fov_y = math.degrees(2.0 * math.atan(camera.image_height / (2.0 * fy)))
        return cls(
            frame_id=frame_id,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            image_width=camera.image_width,
            image_height=camera.image_height,
            fov_x=fov_x,
            fov_y=fov_y,
            distortion_status=distortion_status,
            intrinsics_source=camera.intrinsics_source,
            intrinsics_confidence=intrinsics_confidence,
            camera_motion_class=camera_motion_class,
            pose_status="available" if camera.pose_valid else "unavailable",
            image_transform_chain=tuple(image_transform_chain),
            valid=True,
            metadata={
                "fov_unit": "degree",
                "intrinsics_are_calibrated": "calibrated"
                in camera.intrinsics_source.lower(),
                "pose_source": camera.pose_source,
            },
        )


@dataclass(frozen=True)
class ObjectViewObservation:
    """Object visibility and independent canonical-dimension observability."""

    object_id: str
    track_id: str
    class_name: str
    viewpoint_class: ViewpointClass | str
    orientation_estimate: Mapping[str, float]
    pose_estimate_status: PoseEstimateStatus | str
    foreshortening_risk: str
    border_contact_ratio: float
    visible_ratio: float
    occlusion_ratio: float
    mask_completeness: float
    height_observable: bool
    width_observable: bool
    length_observable: bool
    depth_extent_observable: bool
    view_confidence: float
    failure_reason: str = ""
    dimension_reasons: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    valid: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        viewpoint = ViewpointClass(self.viewpoint_class)
        pose = PoseEstimateStatus(self.pose_estimate_status)
        for name in (
            "border_contact_ratio",
            "visible_ratio",
            "occlusion_ratio",
            "mask_completeness",
            "view_confidence",
        ):
            value = float(getattr(self, name))
            if not math.isnan(value) and (
                not math.isfinite(value) or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be NaN or in [0, 1].")
            object.__setattr__(self, name, value)
        if self.valid and self.failure_reason:
            raise ValueError("Valid ObjectViewObservation cannot have failure_reason.")
        if not self.valid and not self.failure_reason:
            raise ValueError("Invalid ObjectViewObservation requires failure_reason.")
        object.__setattr__(self, "viewpoint_class", viewpoint)
        object.__setattr__(self, "pose_estimate_status", pose)
        object.__setattr__(
            self,
            "orientation_estimate",
            {str(key): float(value) for key, value in self.orientation_estimate.items()},
        )
        object.__setattr__(
            self,
            "dimension_reasons",
            {str(key): tuple(value) for key, value in self.dimension_reasons.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def observable_dimensions(self) -> tuple[str, ...]:
        """Return canonical dimensions accepted by this observation."""

        flags = {
            "height_m": self.height_observable,
            "width_m": self.width_observable,
            "length_m": self.length_observable,
            "depth_extent_m": self.depth_extent_observable,
        }
        return tuple(name for name, accepted in flags.items() if accepted)

    def metric_region_metadata(self) -> dict[str, Any]:
        """Return explicit gates understood by ``MetricObjectRegion``."""

        return {
            "height_observable": self.height_observable,
            "width_observable": self.width_observable,
            "length_observable": self.length_observable,
            "depth_extent_observable": self.depth_extent_observable,
            "viewpoint_class": self.viewpoint_class.value,
            "view_confidence": self.view_confidence,
            "foreshortening_risk": self.foreshortening_risk,
            "object_view_failure_reason": self.failure_reason,
            "dimension_observability_reasons": {
                key: list(value) for key, value in self.dimension_reasons.items()
            },
        }


@dataclass(frozen=True)
class ObjectViewInput:
    """Minimal 2D/visibility input for deterministic view gating."""

    object_id: str
    track_id: str
    class_name: str
    bbox: tuple[float, float, float, float]
    image_width: int
    image_height: int
    detection_confidence: float
    mask_area: float
    bbox_area: float
    occlusion_ratio: float = float("nan")
    mask_stability: float = float("nan")
    viewpoint_hint: ViewpointClass | str = ViewpointClass.UNKNOWN
    orientation_estimate: Mapping[str, float] = field(default_factory=dict)
    pose_estimate_status: PoseEstimateStatus | str = PoseEstimateStatus.UNAVAILABLE
    view_confidence: float = float("nan")
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _border_contacts(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    margin_ratio: float,
) -> frozenset[str]:
    x1, y1, x2, y2 = bbox
    margin_x = max(1.0, width * margin_ratio)
    margin_y = max(1.0, height * margin_ratio)
    contacts: set[str] = set()
    if x1 <= margin_x:
        contacts.add("left")
    if y1 <= margin_y:
        contacts.add("top")
    if x2 >= width - margin_x:
        contacts.add("right")
    if y2 >= height - margin_y:
        contacts.add("bottom")
    return frozenset(contacts)


def classify_viewpoint(
    *,
    viewpoint_hint: ViewpointClass | str = ViewpointClass.UNKNOWN,
    orientation_estimate: Optional[Mapping[str, float]] = None,
    confidence: float = float("nan"),
    minimum_confidence: float = 0.6,
) -> ViewpointClass:
    """Classify viewpoint only from an explicit, sufficiently reliable estimate."""

    hint = ViewpointClass(viewpoint_hint)
    if hint != ViewpointClass.UNKNOWN:
        if math.isfinite(confidence) and confidence >= minimum_confidence:
            return hint
        return ViewpointClass.UNKNOWN
    orientation = dict(orientation_estimate or {})
    if not (math.isfinite(confidence) and confidence >= minimum_confidence):
        return ViewpointClass.UNKNOWN
    yaw = orientation.get("yaw_degrees")
    pitch = orientation.get("pitch_degrees")
    if pitch is not None and math.isfinite(float(pitch)):
        if float(pitch) <= -60.0:
            return ViewpointClass.TOP_DOWN
        if float(pitch) >= 60.0:
            return ViewpointClass.BOTTOM_UP
    if yaw is None or not math.isfinite(float(yaw)):
        return ViewpointClass.UNKNOWN
    folded_yaw = abs(((float(yaw) + 90.0) % 180.0) - 90.0)
    if folded_yaw <= 30.0:
        return ViewpointClass.FRONTAL
    if folded_yaw >= 60.0:
        return ViewpointClass.LATERAL
    return ViewpointClass.OBLIQUE


def evaluate_object_view(
    value: ObjectViewInput,
    *,
    minimum_view_confidence: float = 0.6,
    minimum_visible_ratio: float = 0.2,
    maximum_occlusion_ratio: float = 0.5,
    border_margin_ratio: float = 0.01,
) -> ObjectViewObservation:
    """Evaluate viewpoint and canonical dimensions without guessing missing pose."""

    bbox = tuple(float(item) for item in value.bbox)
    if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return ObjectViewObservation(
            object_id=value.object_id,
            track_id=value.track_id,
            class_name=value.class_name,
            viewpoint_class=ViewpointClass.UNKNOWN,
            orientation_estimate={},
            pose_estimate_status=PoseEstimateStatus.UNAVAILABLE,
            foreshortening_risk="unresolved",
            border_contact_ratio=float("nan"),
            visible_ratio=float("nan"),
            occlusion_ratio=float("nan"),
            mask_completeness=float("nan"),
            height_observable=False,
            width_observable=False,
            length_observable=False,
            depth_extent_observable=False,
            view_confidence=float("nan"),
            failure_reason="invalid_bbox",
            valid=False,
        )
    contacts = _border_contacts(
        bbox,
        value.image_width,
        value.image_height,
        margin_ratio=border_margin_ratio,
    )
    border_ratio = len(contacts) / 4.0
    bbox_area = max(float(value.bbox_area), 1.0)
    visible_ratio = min(1.0, max(0.0, float(value.mask_area) / bbox_area))
    mask_completeness = (
        float(value.metadata["mask_completeness"])
        if "mask_completeness" in value.metadata
        else visible_ratio * (1.0 - border_ratio)
    )
    occlusion = float(value.occlusion_ratio)
    viewpoint = classify_viewpoint(
        viewpoint_hint=value.viewpoint_hint,
        orientation_estimate=value.orientation_estimate,
        confidence=float(value.view_confidence),
        minimum_confidence=minimum_view_confidence,
    )
    pose = PoseEstimateStatus(value.pose_estimate_status)
    heavy_occlusion = math.isfinite(occlusion) and occlusion > maximum_occlusion_ratio
    visible_enough = visible_ratio >= minimum_visible_ratio
    class_name = value.class_name.strip().lower().replace(" ", "_")

    reasons: dict[str, list[str]] = {
        "height_m": [],
        "width_m": [],
        "length_m": [],
        "depth_extent_m": [],
    }
    height = width = length = depth_extent = False
    if not visible_enough:
        for items in reasons.values():
            items.append("insufficient_visible_mask_ratio")
    if heavy_occlusion:
        for items in reasons.values():
            items.append("heavy_occlusion")
    if not math.isfinite(mask_completeness):
        for items in reasons.values():
            items.append("mask_completeness_unresolved")

    base_ready = visible_enough and not heavy_occlusion and math.isfinite(mask_completeness)
    vertical_complete = not bool({"top", "bottom"} & contacts)
    horizontal_complete = not bool({"left", "right"} & contacts)

    if class_name == "person":
        height = (
            base_ready
            and vertical_complete
            and pose == PoseEstimateStatus.UPRIGHT_FULL_BODY
        )
        if not height:
            reasons["height_m"].append("person_upright_full_body_not_verified")
        reasons["width_m"].append("person_width_not_a_stable_strict_dimension")
        reasons["length_m"].append("person_length_not_defined")
    elif class_name in {"cup", "bottle", "vase"}:
        height = (
            base_ready
            and vertical_complete
            and pose
            in {
                PoseEstimateStatus.RESOLVED,
                PoseEstimateStatus.UPRIGHT_SHAPE_COMPATIBLE,
            }
        )
        if not height:
            reasons["height_m"].append("upright_axis_not_verified")
        reasons["width_m"].append("container_width_not_selected_by_strict_prior")
        reasons["length_m"].append("container_length_not_defined")
    elif class_name in {"car", "bus", "truck", "bicycle", "motorcycle"}:
        height = base_ready and vertical_complete and viewpoint in {
            ViewpointClass.FRONTAL,
            ViewpointClass.LATERAL,
        }
        width = (
            base_ready and horizontal_complete and viewpoint == ViewpointClass.FRONTAL
        )
        length = (
            base_ready and horizontal_complete and viewpoint == ViewpointClass.LATERAL
        )
        if viewpoint == ViewpointClass.UNKNOWN:
            for name in ("height_m", "width_m", "length_m"):
                reasons[name].append("vehicle_viewpoint_unresolved")
        elif viewpoint == ViewpointClass.OBLIQUE:
            reasons["width_m"].append("oblique_width_foreshortening")
            reasons["length_m"].append("oblique_length_foreshortening")
    elif class_name in {"soccer_ball", "sports_ball", "ball"}:
        complete = base_ready and vertical_complete and horizontal_complete
        height = width = complete
        length = complete
        if not complete:
            for name in ("height_m", "width_m", "length_m"):
                reasons[name].append("ball_visible_diameter_incomplete")
    else:
        for name in ("height_m", "width_m", "length_m"):
            reasons[name].append("class_specific_view_model_unavailable")

    depth_extent = (
        base_ready
        and viewpoint in {ViewpointClass.LATERAL, ViewpointClass.OBLIQUE}
        and bool(value.metadata.get("depth_extent_supported", False))
    )
    if not depth_extent:
        reasons["depth_extent_m"].append("visible_depth_range_not_canonical_length")

    if not vertical_complete:
        reasons["height_m"].append("vertical_border_contact")
    if not horizontal_complete:
        reasons["width_m"].append("horizontal_border_contact")
        reasons["length_m"].append("horizontal_border_contact")

    if viewpoint == ViewpointClass.UNKNOWN:
        foreshortening = "unresolved"
    elif viewpoint in {
        ViewpointClass.OBLIQUE,
        ViewpointClass.TOP_DOWN,
        ViewpointClass.BOTTOM_UP,
    }:
        foreshortening = "high"
    else:
        foreshortening = "low"
    return ObjectViewObservation(
        object_id=value.object_id,
        track_id=value.track_id,
        class_name=value.class_name,
        viewpoint_class=viewpoint,
        orientation_estimate=value.orientation_estimate,
        pose_estimate_status=pose,
        foreshortening_risk=foreshortening,
        border_contact_ratio=border_ratio,
        visible_ratio=visible_ratio,
        occlusion_ratio=occlusion,
        mask_completeness=mask_completeness,
        height_observable=height,
        width_observable=width,
        length_observable=length,
        depth_extent_observable=depth_extent,
        view_confidence=float(value.view_confidence),
        dimension_reasons={key: tuple(items) for key, items in reasons.items()},
        metadata={
            **dict(value.metadata),
            "border_contacts": sorted(contacts),
            "mask_completeness_is_amodal": False,
            "mask_completeness_definition": (
                "visible_mask_bbox_fill_with_border_penalty_heuristic"
            ),
            "view_quality_is_probability": False,
        },
    )


def intrinsics_relative_change(
    previous: CameraViewObservation, current: CameraViewObservation
) -> float:
    """Return maximum relative change across fx, fy, cx, and cy."""

    if not previous.valid or not current.valid:
        return float("nan")
    prior = np.asarray([previous.fx, previous.fy, previous.cx, previous.cy], dtype=float)
    now = np.asarray([current.fx, current.fy, current.cx, current.cy], dtype=float)
    denominator = np.maximum(np.abs(prior), 1.0)
    return float(np.max(np.abs(now - prior) / denominator))
