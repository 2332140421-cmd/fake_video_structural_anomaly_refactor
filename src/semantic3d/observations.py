"""JSON-serializable observations for structural video anomaly experiments.

The dataclasses in this module define a stable boundary between upstream
vision providers and downstream 3D structural residual analysis. They do not
call segmentation, depth, flow, tracking, or correspondence models; they only
store their future outputs in a compact JSON-friendly form.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


def _require_fields(data: Mapping[str, Any], fields: List[str], context: str) -> None:
    """Raise a clear error when required JSON fields are absent."""

    missing = [field_name for field_name in fields if field_name not in data]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required field(s) in {context}: {joined}.")


def _optional_float_list(value: Any, field_name: str) -> Optional[List[float]]:
    """Convert an optional JSON list into a list of floats."""

    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"Field '{field_name}' must be a list of numbers or null.")
    return [float(item) for item in value]


@dataclass(frozen=True)
class ObjectObservationJSON:
    """JSON record for one observed object in a frame.

    Attributes:
        object_id: Object or instance identifier.
        label: Semantic class label used by downstream scale priors.
        mask_area: Object mask area in pixels.
        frame_area: Full image area in pixels.
        depth: Median or representative object depth.
        confidence: Optional detection or tracking confidence.
        bbox: Optional bounding box [x1, y1, x2, y2] in pixel coordinates.
        mask_path: Optional path to a saved binary mask file.
        track_id: Optional cross-frame track identifier within one video.
        canonical_label: Optional normalized/alias-resolved label for tracking.
    """

    object_id: str
    label: str
    mask_area: float
    frame_area: float
    depth: float
    confidence: float = 1.0
    bbox: Optional[List[float]] = None
    mask_path: Optional[str] = None
    track_id: Optional[str] = None
    canonical_label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary without dropping fields."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObjectObservationJSON":
        """Build an object observation from JSON data.

        Only presence and basic type conversion are handled here. Geometric
        validity, such as positive depth or positive mask area, is checked by
        the residual modules that consume these observations.
        """

        _require_fields(
            data,
            ["object_id", "label", "mask_area", "frame_area", "depth"],
            "ObjectObservationJSON",
        )
        return cls(
            object_id=str(data["object_id"]),
            label=str(data["label"]),
            mask_area=float(data["mask_area"]),
            frame_area=float(data["frame_area"]),
            depth=float(data["depth"]),
            confidence=float(data.get("confidence", 1.0)),
            bbox=_optional_float_list(data.get("bbox"), "bbox"),
            mask_path=None if data.get("mask_path") is None else str(data["mask_path"]),
            track_id=None if data.get("track_id") is None else str(data["track_id"]),
            canonical_label=(
                None
                if data.get("canonical_label") is None
                else str(data["canonical_label"])
            ),
        )

    def to_scale_depth_observation(self) -> "ObjectObservation":
        """Convert this JSON record into the scale-depth residual input type."""

        from .scale_depth import ObjectObservation

        return ObjectObservation(
            object_id=self.object_id,
            label=self.label,
            mask_area=self.mask_area,
            frame_area=self.frame_area,
            depth=self.depth,
            confidence=self.confidence,
        )


@dataclass(frozen=True)
class FrameObservationJSON:
    """JSON record for one video frame and its observed structural evidence."""

    frame_index: int
    frame_id: str
    width: int
    height: int
    objects: List[ObjectObservationJSON] = field(default_factory=list)
    image_path: Optional[str] = None
    depth_map_path: Optional[str] = None
    flow_residual_map_path: Optional[str] = None
    depth_residual_map_path: Optional[str] = None
    corr_residual_map_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary without dropping fields."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrameObservationJSON":
        """Build a frame observation from JSON data."""

        _require_fields(
            data,
            ["frame_index", "frame_id", "width", "height", "objects"],
            "FrameObservationJSON",
        )
        objects_value = data["objects"]
        if not isinstance(objects_value, list):
            raise ValueError("Field 'objects' in FrameObservationJSON must be a list.")

        return cls(
            frame_index=int(data["frame_index"]),
            frame_id=str(data["frame_id"]),
            width=int(data["width"]),
            height=int(data["height"]),
            objects=[
                ObjectObservationJSON.from_dict(object_data)
                for object_data in objects_value
            ],
            image_path=None if data.get("image_path") is None else str(data["image_path"]),
            depth_map_path=(
                None
                if data.get("depth_map_path") is None
                else str(data["depth_map_path"])
            ),
            flow_residual_map_path=(
                None
                if data.get("flow_residual_map_path") is None
                else str(data["flow_residual_map_path"])
            ),
            depth_residual_map_path=(
                None
                if data.get("depth_residual_map_path") is None
                else str(data["depth_residual_map_path"])
            ),
            corr_residual_map_path=(
                None
                if data.get("corr_residual_map_path") is None
                else str(data["corr_residual_map_path"])
            ),
        )


@dataclass(frozen=True)
class ClipObservationJSON:
    """JSON record for a clip consisting of multiple frame observations."""

    clip_id: str
    video_id: str
    frame_indices: List[int]
    frames: List[FrameObservationJSON]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary without dropping fields."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClipObservationJSON":
        """Build a clip observation from JSON data."""

        _require_fields(
            data,
            ["clip_id", "video_id", "frame_indices", "frames"],
            "ClipObservationJSON",
        )
        if not isinstance(data["frame_indices"], list):
            raise ValueError("Field 'frame_indices' in ClipObservationJSON must be a list.")
        if not isinstance(data["frames"], list):
            raise ValueError("Field 'frames' in ClipObservationJSON must be a list.")

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Field 'metadata' in ClipObservationJSON must be a dict.")

        return cls(
            clip_id=str(data["clip_id"]),
            video_id=str(data["video_id"]),
            frame_indices=[int(frame_index) for frame_index in data["frame_indices"]],
            frames=[
                FrameObservationJSON.from_dict(frame_data)
                for frame_data in data["frames"]
            ],
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class ClipResidualResultJSON:
    """JSON record for residual outputs computed on a clip."""

    clip_id: str
    object_residuals: List[Dict[str, Any]]
    pair_residuals: List[Dict[str, Any]]
    clip_score: float
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary without dropping fields."""

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ClipResidualResultJSON":
        """Build a clip residual result from JSON data."""

        _require_fields(
            data,
            ["clip_id", "object_residuals", "pair_residuals", "clip_score", "details"],
            "ClipResidualResultJSON",
        )
        if not isinstance(data["object_residuals"], list):
            raise ValueError(
                "Field 'object_residuals' in ClipResidualResultJSON must be a list."
            )
        if not isinstance(data["pair_residuals"], list):
            raise ValueError(
                "Field 'pair_residuals' in ClipResidualResultJSON must be a list."
            )
        if not isinstance(data["details"], dict):
            raise ValueError("Field 'details' in ClipResidualResultJSON must be a dict.")

        return cls(
            clip_id=str(data["clip_id"]),
            object_residuals=[dict(item) for item in data["object_residuals"]],
            pair_residuals=[dict(item) for item in data["pair_residuals"]],
            clip_score=float(data["clip_score"]),
            details=dict(data["details"]),
        )
