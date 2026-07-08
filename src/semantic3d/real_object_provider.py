"""Real-object-provider adapter for observation JSON construction.

This module defines a pluggable object detector wrapper. It can use an injected
detector in tests or a local detector backend in experiments. It never downloads
model weights by itself; real detector weights must already exist locally.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union

from .observations import ObjectObservationJSON
from .providers import BaseObjectProvider
from .scale_prior import default_scale_prior_resolver

PathLike = Union[str, Path]
BBox = Tuple[float, float, float, float]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "yolov8n.pt"

LABEL_MAPPING: Mapping[str, str] = {
    "sports ball": "soccer_ball",
    "sport ball": "soccer_ball",
    "ball": "soccer_ball",
    "soccer ball": "soccer_ball",
    "person": "person",
    "car": "car",
    "bus": "bus",
    "truck": "truck",
    "bicycle": "bicycle",
    "motorcycle": "motorcycle",
    "cup": "cup",
    "chair": "chair",
    "bottle": "bottle",
    "backpack": "backpack",
    "handbag": "handbag",
    "suitcase": "suitcase",
    "dog": "dog",
    "cat": "cat",
}

def _default_scale_prior_labels() -> set[str]:
    """Return exact and alias labels that can be resolved downstream."""

    fallback = {
        "soccer_ball",
        "person",
        "car",
        "bus",
        "truck",
        "bicycle",
        "motorcycle",
        "cup",
        "chair",
        "bottle",
        "backpack",
        "handbag",
        "suitcase",
        "dog",
        "cat",
        "elephant",
    }
    try:
        resolver = default_scale_prior_resolver(PROJECT_ROOT)
    except Exception:
        return fallback
    labels = set(resolver.scale_priors)
    labels.update(resolver.aliases)
    return {label.replace(" ", "_") for label in labels}


DEFAULT_SCALE_PRIOR_LABELS = _default_scale_prior_labels()


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
        "bottle": 2.5,
        "backpack": 4.0,
        "handbag": 3.5,
        "suitcase": 4.5,
        "person": 8.0,
        "car": 10.0,
        "bus": 14.0,
        "truck": 14.0,
        "bicycle": 7.0,
        "motorcycle": 7.0,
        "chair": 6.0,
        "dog": 5.0,
        "cat": 4.0,
        "elephant": 12.0,
    }.get(label, 8.0)

    # Larger projected boxes are treated as closer. This is only a placeholder
    # until a real depth provider is connected.
    size_factor = min(max(area_ratio**0.5, 0.02), 0.8)
    return float(max(0.5, class_base_depth / (0.7 + 2.0 * size_factor)))


class RealObjectProvider(BaseObjectProvider):
    """Object provider backed by a real detector or an injected detector.

    Args:
        model_path: Local YOLO weights path. Defaults to checkpoints/yolov8n.pt.
        confidence_threshold: Minimum confidence for returned detections.
        default_depth: Temporary depth assigned to every detected object until a
            real depth provider is connected.
        device: Device passed to Ultralytics predict, such as cpu or cuda:0.
        allowed_labels: Labels with available coarse scale priors. If omitted,
            project default scale-prior labels are used.
        skip_unknown_scale_prior: If True, detections whose normalized labels
            are not in allowed_labels are skipped with a warning. If False, they
            are kept so downstream code can decide how to handle missing priors.
        detector: Optional callable or object with ``predict``. It should return
            detection dicts containing bbox/xyxy, label/name/class_name, and
            confidence/score. This path is useful for tests and custom models.
        backend: Optional backend name. Currently ultralytics is supported.
    """

    def __init__(
        self,
        model_path: PathLike = "checkpoints/yolov8n.pt",
        backend: str = "ultralytics",
        confidence_threshold: float = 0.3,
        default_depth: float = 5.0,
        device: str = "cpu",
        allowed_labels: Optional[Sequence[str]] = None,
        skip_unknown_scale_prior: bool = True,
        detector: Optional[Any] = None,
    ) -> None:
        """Create a real object provider wrapper."""

        if confidence_threshold < 0 or confidence_threshold > 1:
            raise ValueError(
                f"confidence_threshold must be in [0, 1], got {confidence_threshold}."
            )
        if default_depth <= 0:
            raise ValueError(f"default_depth must be > 0, got {default_depth}.")
        self.backend = backend
        self.confidence_threshold = confidence_threshold
        self.default_depth = default_depth
        self.device = device
        self.skip_unknown_scale_prior = skip_unknown_scale_prior
        self.allowed_labels = (
            set(DEFAULT_SCALE_PRIOR_LABELS)
            if allowed_labels is None
            else {normalize_label(label) for label in allowed_labels}
        )

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
                message = (
                    f"skipped object because missing scale prior: raw_label={raw_label!r}, "
                    f"normalized_label={label!r}"
                )
                if self.skip_unknown_scale_prior:
                    print(message, file=sys.stderr)
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                    continue
                print(
                    f"keeping object without scale prior: raw_label={raw_label!r}, "
                    f"normalized_label={label!r}",
                    file=sys.stderr,
                )

            clipped_bbox = clip_bbox(bbox, width, height)
            mask_area = bbox_area_to_mask_area(clipped_bbox)
            depth = float(self.default_depth)
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

    def _load_ultralytics_detector(self, model_path: PathLike) -> Any:
        """Load a local Ultralytics model without downloading weights."""

        try:
            ultralytics = importlib.import_module("ultralytics")
        except ImportError as exc:
            raise RuntimeError(
                "RealObjectProvider backend 'ultralytics' requires the optional "
                "dependency 'ultralytics'. Install it and provide a local model "
                "weights path, or use --object_provider mock."
            ) from exc

        model_file = _resolve_model_path(model_path)
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
            try:
                output = self.detector.predict(
                    source=str(frame_path),
                    conf=self.confidence_threshold,
                    device=self.device,
                    verbose=False,
                )
            except TypeError:
                output = self.detector.predict(frame_path)
        else:
            output = self.detector(str(frame_path), verbose=False)

        return self._flatten_detector_output(output)

    def _flatten_detector_output(self, output: Any) -> list[Any]:
        """Flatten detector-specific output into detection-like records."""

        if output is None:
            return []
        if isinstance(output, dict):
            return [output]
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


def _resolve_model_path(model_path: PathLike) -> Path:
    """Resolve model_path relative to cwd first, then project root."""

    path = Path(model_path)
    if path.is_absolute() or path.exists():
        return path
    project_path = PROJECT_ROOT / path
    return project_path
