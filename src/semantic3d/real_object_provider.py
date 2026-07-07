"""Real-object-provider adapter for observation JSON construction.

This module defines a pluggable object detector wrapper. It can use an injected
detector in tests or a local detector backend in experiments. It never downloads
model weights by itself; real detector weights must already exist locally.
"""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union

from .observations import ObjectObservationJSON
from .providers import BaseObjectProvider

PathLike = Union[str, Path]
BBox = Tuple[float, float, float, float]

LABEL_MAPPING: Mapping[str, str] = {
    "sports ball": "soccer_ball",
    "sport ball": "soccer_ball",
    "ball": "soccer_ball",
    "soccer ball": "soccer_ball",
    "person": "person",
    "car": "car",
    "cup": "cup",
    "chair": "chair",
}

DEFAULT_SCALE_PRIOR_LABELS = {
    "soccer_ball",
    "person",
    "car",
    "cup",
    "chair",
    "elephant",
}


def normalize_label(label: str) -> str:
    """Normalize detector class names to project-level object labels."""

    normalized = label.strip().lower().replace("_", " ")
    return LABEL_MAPPING.get(normalized, normalized.replace(" ", "_"))


def clip_bbox(bbox: Sequence[float], width: int, height: int) -> BBox:
    """Clip an xyxy bbox to image coordinates."""

    if len(bbox) != 4:
        raise ValueError(f"bbox must contain 4 numbers, got {len(bbox)}.")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    x1 = min(max(0.0, x1), float(width - 1))
    y1 = min(max(0.0, y1), float(height - 1))
    x2 = min(max(x1 + 1.0, x2), float(width))
    y2 = min(max(y1 + 1.0, y2), float(height))
    return x1, y1, x2, y2


def bbox_area_to_mask_area(bbox: Sequence[float]) -> float:
    """Approximate mask_area with bbox area when no instance mask is available."""

    if len(bbox) != 4:
        raise ValueError(f"bbox must contain 4 numbers, got {len(bbox)}.")
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def estimate_mock_depth(
    label: str,
    bbox: Sequence[float],
    frame_area: float,
    default_depth: Optional[float] = None,
) -> float:
    """Estimate a temporary depth for detections before real depth is available."""

    if default_depth is not None:
        if default_depth <= 0:
            raise ValueError(f"default_depth must be > 0, got {default_depth}.")
        return float(default_depth)

    area_ratio = max(bbox_area_to_mask_area(bbox) / max(frame_area, 1.0), 1e-6)
    class_base_depth = {
        "soccer_ball": 3.0,
        "cup": 2.5,
        "person": 8.0,
        "car": 10.0,
        "chair": 6.0,
        "elephant": 12.0,
    }.get(label, 8.0)

    # Larger projected boxes are treated as closer. This is only a placeholder
    # until a real depth provider is connected.
    size_factor = min(max(area_ratio**0.5, 0.02), 0.8)
    return float(max(0.5, class_base_depth / (0.7 + 2.0 * size_factor)))


class RealObjectProvider(BaseObjectProvider):
    """Object provider backed by a real detector or an injected detector.

    Args:
        detector: Optional callable or object with ``predict``. It should return
            detection dicts containing bbox/xyxy, label/name/class_name, and
            confidence/score. This path is useful for tests and custom models.
        backend: Optional backend name. Currently ``ultralytics`` is supported
            when installed locally.
        model_path: Local model weights path for detector backends. No download
            is attempted if the file is missing.
        confidence_threshold: Minimum confidence for returned detections.
        default_depth: Optional fixed temporary depth for all detections.
        allowed_labels: Labels allowed to flow into scale-depth residual code.
            Unknown labels are skipped with a warning.
    """

    def __init__(
        self,
        detector: Optional[Any] = None,
        backend: str = "ultralytics",
        model_path: Optional[PathLike] = None,
        confidence_threshold: float = 0.3,
        default_depth: Optional[float] = None,
        allowed_labels: Optional[set[str]] = None,
    ) -> None:
        """Create a real object provider wrapper."""

        if confidence_threshold < 0 or confidence_threshold > 1:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}."
            )
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.default_depth = default_depth
        self.allowed_labels = allowed_labels or set(DEFAULT_SCALE_PRIOR_LABELS)

        if detector is not None:
            self.detector = detector
            return

        if backend != "ultralytics":
            raise RuntimeError(
                f"Unsupported real detector backend '{backend}'. "
                "Supported backend: ultralytics."
            )
        self.detector = self._load_ultralytics_detector(model_path)

    def predict(
        self, frame_path: PathLike, frame_index: int, width: int, height: int
    ) -> list[ObjectObservationJSON]:
        """Run detection and convert results into ObjectObservationJSON records."""

        raw_detections = self._run_detector(frame_path)
        frame_area = float(width * height)
        objects: list[ObjectObservationJSON] = []

        for detection_index, detection in enumerate(raw_detections):
            parsed = self._parse_detection(detection)
            if parsed is None:
                continue
            raw_label, confidence, bbox = parsed
            if confidence < self.confidence_threshold:
                continue

            label = normalize_label(raw_label)
            if label not in self.allowed_labels:
                warnings.warn(
                    f"Skipping detected label '{raw_label}' normalized to '{label}' "
                    "because no scale prior is available.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            clipped_bbox = clip_bbox(bbox, width, height)
            mask_area = bbox_area_to_mask_area(clipped_bbox)
            depth = estimate_mock_depth(
                label,
                clipped_bbox,
                frame_area,
                default_depth=self.default_depth,
            )
            objects.append(
                ObjectObservationJSON(
                    object_id=f"{label}_f{frame_index}_{detection_index}",
                    label=label,
                    mask_area=mask_area,
                    frame_area=frame_area,
                    depth=depth,
                    confidence=confidence,
                    bbox=list(clipped_bbox),
                )
            )

        return objects

    def _load_ultralytics_detector(self, model_path: Optional[PathLike]) -> Any:
        """Load a local Ultralytics model without downloading weights."""

        try:
            ultralytics = importlib.import_module("ultralytics")
        except ImportError as exc:
            raise RuntimeError(
                "RealObjectProvider backend 'ultralytics' requires the optional "
                "dependency 'ultralytics'. Install it and provide a local model "
                "weights path, or use --object_provider mock."
            ) from exc

        if model_path is None:
            raise RuntimeError(
                "RealObjectProvider with backend 'ultralytics' requires a local "
                "model_path. No model is downloaded automatically."
            )
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"Local detector model_path does not exist: {model_file}. "
                "Provide local weights or use --object_provider mock."
            )
        return ultralytics.YOLO(str(model_file))

    def _run_detector(self, frame_path: PathLike) -> list[Any]:
        """Run an injected detector or backend detector and return raw detections."""

        if callable(self.detector) and not hasattr(self.detector, "predict"):
            output = self.detector(frame_path)
        elif hasattr(self.detector, "predict"):
            output = self.detector.predict(frame_path)
        else:
            output = self.detector(str(frame_path), verbose=False)

        return self._flatten_detector_output(output)

    def _flatten_detector_output(self, output: Any) -> list[Any]:
        """Flatten detector-specific output into detection-like records."""

        if output is None:
            return []
        if isinstance(output, list) and output and isinstance(output[0], dict):
            return output
        if isinstance(output, tuple) and output and isinstance(output[0], dict):
            return list(output)

        # Ultralytics results: each result has boxes, names, xyxy/conf/cls arrays.
        detections: list[dict[str, Any]] = []
        results = output if isinstance(output, Iterable) else [output]
        for result in results:
            boxes = getattr(result, "boxes", None)
            names = getattr(result, "names", {})
            if boxes is None:
                continue
            xyxy = getattr(boxes, "xyxy", [])
            conf = getattr(boxes, "conf", [])
            cls = getattr(boxes, "cls", [])
            xyxy_values = xyxy.cpu().numpy().tolist() if hasattr(xyxy, "cpu") else xyxy
            conf_values = conf.cpu().numpy().tolist() if hasattr(conf, "cpu") else conf
            cls_values = cls.cpu().numpy().tolist() if hasattr(cls, "cpu") else cls
            for bbox, score, class_id in zip(xyxy_values, conf_values, cls_values):
                class_index = int(class_id)
                label = names.get(class_index, str(class_index))
                detections.append(
                    {"bbox": bbox, "label": label, "confidence": float(score)}
                )
        return detections

    def _parse_detection(self, detection: Any) -> Optional[tuple[str, float, BBox]]:
        """Parse one detection dict into label, confidence, and bbox."""

        if not isinstance(detection, Mapping):
            return None
        bbox_value = detection.get("bbox", detection.get("xyxy"))
        label_value = detection.get(
            "label", detection.get("name", detection.get("class_name"))
        )
        confidence_value = detection.get("confidence", detection.get("score", 1.0))

        if bbox_value is None or label_value is None:
            return None
        bbox = tuple(float(value) for value in bbox_value)
        if len(bbox) != 4:
            raise ValueError(f"Detection bbox must contain 4 numbers, got {bbox}.")
        return str(label_value), float(confidence_value), bbox  # type: ignore[return-value]
