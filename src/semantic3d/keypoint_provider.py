"""Pluggable 2D human-keypoint providers for physical-prior applicability.

The keypoints in this module are image-plane observations only. They do not
perform 3D back-projection, camera-pose estimation, tracking, or R_track.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import cv2
import numpy as np


FrameInput = Union[str, Path, np.ndarray]

COCO_PERSON_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class Keypoint2D:
    """One named 2D keypoint with explicit validity and provenance."""

    keypoint_name: str
    x: float
    y: float
    confidence: float
    valid: bool
    provider_name: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable keypoint record."""

        return asdict(self)


@dataclass(frozen=True)
class KeypointPrediction:
    """Keypoints matched to one requested object bbox."""

    label: str
    keypoints: tuple[Keypoint2D, ...]
    status: str
    provider_name: str
    matched_bbox: Optional[tuple[float, float, float, float]] = None
    details: Mapping[str, object] = field(default_factory=dict)

    @property
    def supported(self) -> bool:
        """Return whether the requested semantic class is supported."""

        return self.status != "unsupported_label"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable prediction for future track modules."""

        return {
            "label": self.label,
            "keypoints": [keypoint.to_dict() for keypoint in self.keypoints],
            "status": self.status,
            "provider_name": self.provider_name,
            "matched_bbox": list(self.matched_bbox) if self.matched_bbox else None,
            "details": dict(self.details),
        }


class BaseKeypointProvider(ABC):
    """Interface for object-conditioned 2D keypoint inference."""

    @abstractmethod
    def predict(
        self,
        frame: FrameInput,
        bbox: Sequence[float],
        label: str,
    ) -> KeypointPrediction:
        """Return keypoints for the object matching ``bbox`` in ``frame``."""


class MockKeypointProvider(BaseKeypointProvider):
    """Deterministic provider used by unit tests and synthetic experiments."""

    def __init__(
        self,
        keypoints: Sequence[Keypoint2D] = (),
        provider_name: str = "mock_keypoints",
    ) -> None:
        self._keypoints = tuple(keypoints)
        self.provider_name = provider_name

    def predict(
        self,
        frame: FrameInput,
        bbox: Sequence[float],
        label: str,
    ) -> KeypointPrediction:
        """Return configured points for person and unsupported for other labels."""

        del frame
        if label.strip().lower().replace(" ", "_") != "person":
            return KeypointPrediction(label, (), "unsupported_label", self.provider_name)
        points = tuple(
            Keypoint2D(
                point.keypoint_name,
                float(point.x),
                float(point.y),
                float(point.confidence),
                bool(point.valid),
                self.provider_name,
            )
            for point in self._keypoints
        )
        matched = tuple(float(value) for value in bbox) if len(bbox) == 4 else None
        return KeypointPrediction(
            label,
            points,
            "ok" if points else "no_keypoints",
            self.provider_name,
            matched_bbox=matched,  # type: ignore[arg-type]
        )


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Return IoU for two xyxy boxes."""

    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(value) for value in a)
    bx1, by1, bx2, by2 = (float(value) for value in b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(
        0.0, bx2 - bx1
    ) * max(0.0, by2 - by1) - intersection
    return intersection / union if union > 0 else 0.0


class RealHumanKeypointProvider(BaseKeypointProvider):
    """Ultralytics human-pose provider matched to an existing person bbox.

    The model is loaded from a local weight file. Missing dependencies or
    weights raise a clear error; the provider never silently substitutes mock
    points. Full-frame pose inference is cached by image path so multiple
    person boxes in the same frame do not trigger duplicate model calls.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = "checkpoints/yolov8n-pose.pt",
        confidence_threshold: float = 0.25,
        keypoint_confidence_threshold: float = 0.25,
        minimum_match_iou: float = 0.05,
        device: str = "cpu",
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1].")
        if not 0 <= keypoint_confidence_threshold <= 1:
            raise ValueError("keypoint_confidence_threshold must be in [0, 1].")
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                "Human pose weights are unavailable. Place yolov8n-pose.pt at "
                f"{self.model_path} or use MockKeypointProvider."
            )
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "RealHumanKeypointProvider requires ultralytics in the project environment."
            ) from error
        self.provider_name = "ultralytics_yolov8_pose"
        self.confidence_threshold = float(confidence_threshold)
        self.keypoint_confidence_threshold = float(keypoint_confidence_threshold)
        self.minimum_match_iou = float(minimum_match_iou)
        self.device = device
        self._model = YOLO(str(self.model_path))
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @staticmethod
    def _read_frame(frame: FrameInput) -> tuple[np.ndarray, Optional[str]]:
        if isinstance(frame, np.ndarray):
            if frame.ndim != 3:
                raise ValueError("Frame ndarray must have shape [H, W, C].")
            return frame, None
        path = Path(frame)
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Unable to read frame image: {path}")
        return image, str(path.resolve())

    def _infer(self, image: np.ndarray, cache_key: Optional[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]
        results = self._model.predict(
            source=image,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        boxes = (
            result.boxes.xyxy.detach().cpu().numpy().astype(float)
            if result.boxes is not None and result.boxes.xyxy is not None
            else np.empty((0, 4), dtype=float)
        )
        xy = (
            result.keypoints.xy.detach().cpu().numpy().astype(float)
            if result.keypoints is not None and result.keypoints.xy is not None
            else np.empty((0, len(COCO_PERSON_KEYPOINT_NAMES), 2), dtype=float)
        )
        confidence_tensor = result.keypoints.conf if result.keypoints is not None else None
        confidence = (
            confidence_tensor.detach().cpu().numpy().astype(float)
            if confidence_tensor is not None
            else np.zeros(xy.shape[:2], dtype=float)
        )
        output = (boxes, xy, confidence)
        if cache_key:
            self._cache[cache_key] = output
        return output

    def predict(
        self,
        frame: FrameInput,
        bbox: Sequence[float],
        label: str,
    ) -> KeypointPrediction:
        """Infer full-frame poses and return the pose with greatest bbox IoU."""

        if label.strip().lower().replace(" ", "_") != "person":
            return KeypointPrediction(label, (), "unsupported_label", self.provider_name)
        image, cache_key = self._read_frame(frame)
        boxes, xy, confidence = self._infer(image, cache_key)
        if boxes.shape[0] == 0 or xy.shape[0] == 0:
            return KeypointPrediction(label, (), "no_pose_detection", self.provider_name)
        overlaps = np.asarray([_bbox_iou(candidate, bbox) for candidate in boxes])
        index = int(np.argmax(overlaps))
        if float(overlaps[index]) < self.minimum_match_iou:
            return KeypointPrediction(
                label,
                (),
                "no_matching_pose",
                self.provider_name,
                details={"best_iou": float(overlaps[index])},
            )
        points: list[Keypoint2D] = []
        height, width = image.shape[:2]
        for point_index, name in enumerate(COCO_PERSON_KEYPOINT_NAMES):
            x, y = (float(value) for value in xy[index, point_index])
            score = float(confidence[index, point_index])
            valid = bool(
                math.isfinite(x)
                and math.isfinite(y)
                and math.isfinite(score)
                and score >= self.keypoint_confidence_threshold
                and 0 <= x < width
                and 0 <= y < height
                and not (x == 0 and y == 0)
            )
            points.append(Keypoint2D(name, x, y, score, valid, self.provider_name))
        return KeypointPrediction(
            label,
            tuple(points),
            "ok",
            self.provider_name,
            matched_bbox=tuple(float(value) for value in boxes[index]),  # type: ignore[arg-type]
            details={"match_iou": float(overlaps[index])},
        )

