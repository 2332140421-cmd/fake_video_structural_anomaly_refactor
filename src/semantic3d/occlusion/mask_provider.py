"""Canonical instance-mask provider interfaces and non-downloading adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import cv2
import numpy as np

from ..observations import FrameObservationJSON, ObjectObservationJSON
from .mask_observation import InstanceMaskObservation


@dataclass(frozen=True)
class InstanceMaskCandidate:
    """One unassociated visible-mask prediction from a real segmenter."""

    candidate_id: str
    class_id: int
    class_name: str
    visible_mask: np.ndarray
    confidence: float
    source_detection_id: str
    source_provider: str
    model_name: str
    model_version: str
    inference_device: str
    weight_sha256: str = ""
    preprocessing_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mask = np.asarray(self.visible_mask, dtype=bool)
        confidence = float(self.confidence)
        if mask.ndim != 2 or not np.any(mask):
            raise ValueError("InstanceMaskCandidate requires a non-empty 2D mask.")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("InstanceMaskCandidate confidence must be in [0, 1].")
        if not self.candidate_id or not self.class_name or not self.source_provider:
            raise ValueError("Mask candidate identifiers and class/source must be present.")
        output = mask.copy()
        output.setflags(write=False)
        object.__setattr__(self, "visible_mask", output)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "preprocessing_metadata", dict(self.preprocessing_metadata))


@dataclass(frozen=True)
class InstanceSegmentationModelMetadata:
    """Auditable metadata for one frozen local segmentation model."""

    weights_path: str
    weights_readable: bool
    weights_size_bytes: int
    weight_sha256: str
    model_name: str
    model_version: str
    provider_name: str
    inference_device: str
    model_task: str
    output_contains_instance_masks: bool
    available: bool
    missing_reason: str = ""


def _sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for a local immutable artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BaseInstanceMaskProvider(ABC):
    """Canonical offline segmentation interface; it never receives truth labels."""

    provider_name = "base_instance_mask_provider"

    @abstractmethod
    def predict(
        self,
        *,
        video_id: str,
        frame: FrameObservationJSON,
    ) -> tuple[InstanceMaskObservation, ...]:
        """Return masks aligned with frame objects and their stable track IDs."""


class SyntheticInstanceMaskProvider(BaseInstanceMaskProvider):
    """Return supplied masks for deterministic synthetic tests."""

    provider_name = "synthetic_instance_mask_truth"

    def __init__(self, masks: Mapping[tuple[int, str], np.ndarray]) -> None:
        self.masks = {(int(frame), str(track)): np.asarray(mask, dtype=bool) for (frame, track), mask in masks.items()}

    def predict(self, *, video_id: str, frame: FrameObservationJSON) -> tuple[InstanceMaskObservation, ...]:
        output = []
        for obj in frame.objects:
            track_id = str(obj.track_id or obj.person_track_id or obj.object_id)
            mask = self.masks.get((frame.frame_index, track_id))
            if mask is None:
                output.append(InstanceMaskObservation.missing(video_id=video_id, frame_index=frame.frame_index, object_track_id=track_id, semantic_label=obj.label, image_shape=(frame.height, frame.width), reason="synthetic_mask_missing", source_provider=self.provider_name))
            else:
                expected_shape = (frame.height, frame.width)
                if mask.shape != expected_shape:
                    raise ValueError(
                        f"Synthetic mask image shape {mask.shape} does not match "
                        f"frame image shape {expected_shape}."
                    )
                output.append(InstanceMaskObservation.from_visible_mask(video_id=video_id, frame_index=frame.frame_index, object_track_id=track_id, semantic_label=obj.label, mask=mask, confidence=1.0, source_provider=self.provider_name, metadata={"model_version": "synthetic_truth_v1", "segmentation_confidence": 1.0, "source_detection_id": obj.object_id, "whether_bbox_prompted": False, "whether_temporally_propagated": False, "truth_label_used": False}))
        return tuple(output)


class MockInstanceMaskProvider(BaseInstanceMaskProvider):
    """Generate deterministic test masks, explicitly marked as mock evidence."""

    provider_name = "mock_instance_mask_provider"

    def predict(self, *, video_id: str, frame: FrameObservationJSON) -> tuple[InstanceMaskObservation, ...]:
        output = []
        for obj in frame.objects:
            track_id = str(obj.track_id or obj.person_track_id or obj.object_id)
            if obj.bbox is None:
                output.append(InstanceMaskObservation.missing(video_id=video_id, frame_index=frame.frame_index, object_track_id=track_id, semantic_label=obj.label, image_shape=(frame.height, frame.width), reason="mock_bbox_missing", source_provider=self.provider_name))
                continue
            x1, y1, x2, y2 = (int(round(value)) for value in obj.bbox)
            mask = np.zeros((frame.height, frame.width), dtype=np.uint8)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
            cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 1, -1)
            output.append(InstanceMaskObservation.from_visible_mask(video_id=video_id, frame_index=frame.frame_index, object_track_id=track_id, semantic_label=obj.label, mask=mask, confidence=0.8, source_provider=self.provider_name, metadata={"model_version": "mock_v1", "segmentation_confidence": 0.8, "source_detection_id": obj.object_id, "whether_bbox_prompted": True, "whether_temporally_propagated": False, "mock_only": True, "truth_label_used": False}))
        return tuple(output)


class ExistingDetectionMaskAdapter(BaseInstanceMaskProvider):
    """Load existing mask paths, with an optional low-quality bbox diagnostic."""

    provider_name = "existing_detection_mask_adapter"

    def __init__(self, *, allow_legacy_bbox_fallback: bool = False) -> None:
        self.allow_legacy_bbox_fallback = bool(allow_legacy_bbox_fallback)

    @staticmethod
    def _load(path: str, shape: tuple[int, int]) -> Optional[np.ndarray]:
        mask_path = Path(path)
        if not mask_path.exists():
            return None
        if mask_path.suffix.lower() == ".npy":
            mask = np.load(mask_path, allow_pickle=False)
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != shape:
            return None
        return np.asarray(mask, dtype=bool)

    def predict(self, *, video_id: str, frame: FrameObservationJSON) -> tuple[InstanceMaskObservation, ...]:
        output = []
        shape = (frame.height, frame.width)
        for obj in frame.objects:
            track_id = str(obj.track_id or obj.person_track_id or obj.object_id)
            mask = None if not obj.mask_path else self._load(obj.mask_path, shape)
            legacy = False
            if mask is None and self.allow_legacy_bbox_fallback and obj.bbox is not None:
                x1, y1, x2, y2 = (int(round(value)) for value in obj.bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.width, x2), min(frame.height, y2)
                mask = np.zeros(shape, dtype=bool)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = True
                    legacy = True
            if mask is None or not np.any(mask):
                output.append(InstanceMaskObservation.missing(video_id=video_id, frame_index=frame.frame_index, object_track_id=track_id, semantic_label=obj.label, image_shape=shape, reason="instance_mask_unavailable", source_provider=self.provider_name))
                continue
            confidence = min(float(obj.confidence), 0.25) if legacy else float(obj.confidence)
            output.append(InstanceMaskObservation.from_visible_mask(video_id=video_id, frame_index=frame.frame_index, object_track_id=track_id, semantic_label=obj.label, mask=mask, confidence=confidence, source_provider=self.provider_name, metadata={"model_version": "existing_artifact", "segmentation_confidence": confidence, "source_detection_id": obj.object_id, "whether_bbox_prompted": legacy, "whether_temporally_propagated": False, "legacy_bbox_fallback": legacy, "formal_mask_evidence": not legacy, "truth_label_used": False}))
        return tuple(output)


class RealInstanceMaskProvider(BaseInstanceMaskProvider):
    """Frozen local Ultralytics segmentation provider with no network fallback.

    A legacy callback is retained for deterministic tests. With a model path,
    full-frame visible masks are inferred first and then associated one-to-one
    with existing objects. The provider never returns amodal masks and never
    substitutes a filled bbox when segmentation is unavailable.
    """

    provider_name = "real_instance_mask_provider"

    def __init__(
        self,
        predictor: Optional[Callable[[FrameObservationJSON, ObjectObservationJSON], Optional[np.ndarray]]] = None,
        *,
        model_path: str | Path = "checkpoints/yolov8n-seg.pt",
        confidence_threshold: float = 0.25,
        device: str = "cpu",
        model_name: str = "yolov8n-seg",
        model_version: str = "local_frozen",
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")
        self.predictor = predictor
        self.model_path = Path(model_path)
        self.confidence_threshold = float(confidence_threshold)
        self.device = str(device)
        self.model_name = str(model_name)
        self.model_version = str(model_version)
        self._model: Any = None
        self._ultralytics_version = ""
        self._weight_sha256 = ""
        self._weights_size_bytes = 0
        self._model_task = "callback" if predictor is not None else "unknown"
        self._output_contains_instance_masks = False
        self.unavailable_reason = ""
        if predictor is not None:
            return
        if not self.model_path.exists():
            self.unavailable_reason = "instance_segmentation_weights_missing"
            return
        if not self.model_path.is_file() or not os.access(self.model_path, os.R_OK):
            self.unavailable_reason = "instance_segmentation_weights_unreadable"
            return
        self._weights_size_bytes = int(self.model_path.stat().st_size)
        if self._weights_size_bytes <= 0:
            self.unavailable_reason = "instance_segmentation_weights_empty"
            return
        self._weight_sha256 = _sha256(self.model_path)
        try:
            import ultralytics
            from ultralytics import YOLO
        except ImportError:
            self.unavailable_reason = "real_instance_segmentation_dependency_missing: install ultralytics"
            return
        self._ultralytics_version = str(ultralytics.__version__)
        try:
            self._model = YOLO(str(self.model_path))
            self._model_task = str(getattr(self._model, "task", "unknown"))
            if self._model_task != "segment":
                self._model = None
                self.unavailable_reason = "instance_segmentation_model_task_mismatch"
        except Exception as error:  # pragma: no cover - model backend specific
            self.unavailable_reason = f"real_instance_segmentation_load_failed:{type(error).__name__}"

    @property
    def available(self) -> bool:
        """Return whether a callback or frozen local segmentation model is ready."""

        return self.predictor is not None or self._model is not None

    @property
    def model_metadata(self) -> InstanceSegmentationModelMetadata:
        """Return immutable weight and runtime metadata without running inference."""

        readable = self.predictor is not None or (
            self.model_path.is_file() and os.access(self.model_path, os.R_OK)
        )
        version = self.model_version
        if self._ultralytics_version:
            version = f"{version};ultralytics={self._ultralytics_version}"
        return InstanceSegmentationModelMetadata(
            weights_path=str(self.model_path),
            weights_readable=readable,
            weights_size_bytes=self._weights_size_bytes,
            weight_sha256=self._weight_sha256,
            model_name=self.model_name,
            model_version=version,
            provider_name=self.provider_name,
            inference_device=self.device,
            model_task=self._model_task,
            output_contains_instance_masks=self._output_contains_instance_masks,
            available=self.available,
            missing_reason=self.unavailable_reason,
        )

    def predict_candidates(self, frame: FrameObservationJSON) -> tuple[InstanceMaskCandidate, ...]:
        """Infer unassociated visible masks without consuming truth labels."""

        if self._model is None:
            return ()
        if not frame.image_path:
            self.unavailable_reason = "frame_image_path_missing_for_instance_segmentation"
            return ()
        image_path = Path(frame.image_path)
        if not image_path.exists():
            self.unavailable_reason = "frame_image_missing_for_instance_segmentation"
            return ()
        results = self._model.predict(
            source=str(image_path),
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        if not results:
            return ()
        result = results[0]
        if result.boxes is None:
            return ()
        if result.masks is None:
            if len(result.boxes) > 0:
                self.unavailable_reason = "instance_segmentation_output_missing_masks"
            return ()
        masks = result.masks.data.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
        track_ids = None
        if result.boxes.id is not None:
            track_ids = result.boxes.id.detach().cpu().numpy().astype(int)
        names = result.names
        if len(masks) != len(classes):
            self.unavailable_reason = "instance_segmentation_mask_box_count_mismatch"
            return ()
        self._output_contains_instance_masks = True
        output = []
        for index, (raw_mask, class_id, confidence) in enumerate(zip(masks, classes, confidences)):
            resized = cv2.resize(
                raw_mask.astype(np.float32),
                (frame.width, frame.height),
                interpolation=cv2.INTER_NEAREST,
            ) >= 0.5
            if not np.any(resized):
                continue
            class_name = str(names[int(class_id)])
            source_id = str(track_ids[index]) if track_ids is not None else f"seg_{frame.frame_index}_{index}"
            output.append(InstanceMaskCandidate(
                candidate_id=f"{frame.frame_index}:{index}",
                class_id=int(class_id),
                class_name=class_name,
                visible_mask=resized,
                confidence=float(confidence),
                source_detection_id=source_id,
                source_provider=self.provider_name,
                model_name=self.model_name,
                model_version=self.model_version,
                inference_device=self.device,
                weight_sha256=self._weight_sha256,
                preprocessing_metadata={
                    "input_image_path": str(image_path),
                    "output_resized_to_original": True,
                    "original_shape": [frame.height, frame.width],
                    "truth_label_used": False,
                    "frozen_pretrained_model": True,
                    "weight_sha256": self._weight_sha256,
                    "weights_size_bytes": self._weights_size_bytes,
                    "model_task": self._model_task,
                    "ultralytics_version": self._ultralytics_version,
                },
            ))
        return tuple(output)

    def predict(self, *, video_id: str, frame: FrameObservationJSON) -> tuple[InstanceMaskObservation, ...]:
        if self.predictor is None:
            if not self.available:
                reason = self.unavailable_reason or "real_instance_mask_provider_unavailable"
                return tuple(InstanceMaskObservation.missing(video_id=video_id, frame_index=frame.frame_index, object_track_id=str(obj.track_id or obj.person_track_id or obj.object_id), semantic_label=obj.label, image_shape=(frame.height, frame.width), reason=reason, source_provider=self.provider_name) for obj in frame.objects)
            from .mask_object_association import associate_instance_masks

            result = associate_instance_masks(
                video_id=video_id,
                frame=frame,
                candidates=self.predict_candidates(frame),
            )
            return result.masks
        output = []
        for obj in frame.objects:
            track_id = str(obj.track_id or obj.person_track_id or obj.object_id)
            mask = self.predictor(frame, obj)
            if mask is None:
                output.append(InstanceMaskObservation.missing(video_id=video_id, frame_index=frame.frame_index, object_track_id=track_id, semantic_label=obj.label, image_shape=(frame.height, frame.width), reason="real_segmenter_returned_no_mask", source_provider=self.provider_name))
            else:
                output.append(InstanceMaskObservation.from_visible_mask(video_id=video_id, frame_index=frame.frame_index, object_track_id=track_id, semantic_label=obj.label, mask=mask, confidence=float(obj.confidence), source_provider=self.provider_name, metadata={"model_name": self.model_name, "model_version": self.model_version, "segmentation_confidence": float(obj.confidence), "source_detection_id": obj.object_id, "class_id": None, "class_name": obj.label, "inference_device": self.device, "preprocessing_metadata": {"callback": True}, "whether_bbox_prompted": True, "whether_temporally_propagated": False, "frozen_pretrained_model": True, "truth_label_used": False, "formal_mask_evidence": True}))
        return tuple(output)
