"""Mock providers for building observation JSON without real vision models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Literal, Tuple, Union

from .observations import ObjectObservationJSON

PathLike = Union[str, Path]
MockMode = Literal["reasonable", "anomaly"]


class BaseObjectProvider(ABC):
    """Interface for future object detection or instance segmentation providers."""

    @abstractmethod
    def predict(
        self, frame_path: PathLike, frame_index: int, width: int, height: int
    ) -> List[ObjectObservationJSON]:
        """Return object observations for one frame.

        Future real implementations can call YOLO/SAM-style models and convert
        detected labels, masks, boxes, mask areas, depths, and confidences into
        ObjectObservationJSON records.
        """


class MockDepthProvider:
    """Legacy label-to-depth helper for synthetic object-provider tests.

    This class does not produce dense depth maps and is not the canonical depth
    provider. New frame-level code uses ``depth_provider.MockDepthProvider``.
    """

    def __init__(self, mode: MockMode = "reasonable") -> None:
        """Create a mock depth provider for reasonable or anomalous geometry."""

        if mode not in {"reasonable", "anomaly"}:
            raise ValueError("mode must be either 'reasonable' or 'anomaly'.")
        self.mode = mode

    def depth_for_label(self, label: str, frame_index: int) -> float:
        """Return a deterministic synthetic depth for an object label."""

        phase = (frame_index % 5) * 0.03
        if self.mode == "reasonable":
            if label == "soccer_ball":
                return 3.0 + phase
            if label == "elephant":
                return 12.0 + phase * 2.0
        else:
            if label == "soccer_ball":
                return 8.0 + phase
            if label == "elephant":
                return 8.2 + phase

        raise KeyError(f"MockDepthProvider has no depth rule for label '{label}'.")


LegacyObjectDepthProvider = MockDepthProvider


class MockObjectProvider(BaseObjectProvider):
    """Object provider that emits soccer_ball and elephant mock observations.

    In ``reasonable`` mode, the ball is closer and smaller while the elephant is
    farther and larger, producing low scale-depth residuals. In ``anomaly``
    mode, the elephant has an implausibly small projection at a similar depth,
    producing high scale-depth residuals.
    """

    def __init__(
        self,
        mode: MockMode = "reasonable",
        depth_provider: MockDepthProvider | None = None,
    ) -> None:
        """Create a deterministic mock object provider."""

        if mode not in {"reasonable", "anomaly"}:
            raise ValueError("mode must be either 'reasonable' or 'anomaly'.")
        self.mode = mode
        self.depth_provider = depth_provider or MockDepthProvider(mode=mode)

    def predict(
        self, frame_path: PathLike, frame_index: int, width: int, height: int
    ) -> List[ObjectObservationJSON]:
        """Return two mock objects for one frame."""

        del frame_path
        frame_area = float(width * height)
        soccer_bbox, elephant_bbox = self._boxes(width, height, frame_index)
        soccer_area = _bbox_area(soccer_bbox)
        elephant_area = _bbox_area(elephant_bbox)

        return [
            ObjectObservationJSON(
                object_id=f"soccer_ball_f{frame_index}",
                label="soccer_ball",
                mask_area=soccer_area,
                frame_area=frame_area,
                depth=self.depth_provider.depth_for_label("soccer_ball", frame_index),
                confidence=0.98,
                bbox=list(soccer_bbox),
            ),
            ObjectObservationJSON(
                object_id=f"elephant_f{frame_index}",
                label="elephant",
                mask_area=elephant_area,
                frame_area=frame_area,
                depth=self.depth_provider.depth_for_label("elephant", frame_index),
                confidence=0.97,
                bbox=list(elephant_bbox),
            ),
        ]

    def _boxes(
        self, width: int, height: int, frame_index: int
    ) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
        """Return synthetic boxes whose areas define object projection sizes."""

        drift = float(frame_index % 6) * 2.0
        soccer_size = max(16.0, min(width, height) * 0.12)
        soccer_x1 = width * 0.18 + drift
        soccer_y1 = height * 0.62
        soccer_box = (
            soccer_x1,
            soccer_y1,
            soccer_x1 + soccer_size,
            soccer_y1 + soccer_size,
        )

        if self.mode == "reasonable":
            elephant_w = width * 0.32
            elephant_h = height * 0.42
            elephant_x1 = width * 0.58 - drift
            elephant_y1 = height * 0.35
        else:
            elephant_w = width * 0.06
            elephant_h = height * 0.08
            elephant_x1 = width * 0.70 - drift
            elephant_y1 = height * 0.48

        elephant_box = (
            elephant_x1,
            elephant_y1,
            elephant_x1 + elephant_w,
            elephant_y1 + elephant_h,
        )
        return _clip_box(soccer_box, width, height), _clip_box(elephant_box, width, height)


def _clip_box(
    box: Tuple[float, float, float, float], width: int, height: int
) -> Tuple[float, float, float, float]:
    """Clip a bounding box to image coordinates."""

    x1, y1, x2, y2 = box
    x1 = min(max(0.0, x1), float(width - 1))
    y1 = min(max(0.0, y1), float(height - 1))
    x2 = min(max(x1 + 1.0, x2), float(width))
    y2 = min(max(y1 + 1.0, y2), float(height))
    return x1, y1, x2, y2


def _bbox_area(box: Tuple[float, float, float, float]) -> float:
    """Return rectangular mask-area proxy from a bounding box."""

    x1, y1, x2, y2 = box
    return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))
