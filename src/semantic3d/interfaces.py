"""Abstract interfaces for future visual model providers.

These classes define integration boundaries only. They do not implement or
instantiate YOLO/SAM, Video Depth Anything, RAFT, CoTracker, or any other large
model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .data_structures import FrameObservation, ObjectTrack
from .scale_depth import ObjectObservation


class SegmentationProvider(ABC):
    """Interface for object detection and instance segmentation providers.

    Implementations should return a list of ObjectObservation instances for one
    frame. The object observations should include labels, mask areas, frame
    areas, confidence scores, and a depth value if depth has already been
    attached by the pipeline. A future adapter may wrap YOLO/SAM behind this
    interface.
    """

    @abstractmethod
    def segment_frame(self, image: Any, frame_index: int) -> List[ObjectObservation]:
        """Segment one image and return object-level observations."""


class DepthProvider(ABC):
    """Legacy image/object-depth provider interface.

    Implementations should return a dense depth map as a numpy array with shape
    H x W, or an equivalent array-like output converted to numpy by the adapter.
    New geometry code must use ``depth_provider.BaseDepthProvider`` and
    ``DepthObservation``. This class remains for source compatibility and can be
    wrapped by ``LegacyDepthProviderAdapter`` when its value semantics are known.
    """

    @abstractmethod
    def estimate_depth(self, image: Any, frame_index: int) -> np.ndarray:
        """Estimate a dense depth map for one frame."""

    @abstractmethod
    def object_depths(
        self, depth_map: np.ndarray, objects: Sequence[ObjectObservation]
    ) -> Dict[str, float]:
        """Return per-object representative depths keyed by object_id."""


# Stable explicit name for callers that need to identify the old interface.
LegacyDepthProvider = DepthProvider


class FlowProvider(ABC):
    """Interface for optical-flow providers.

    Implementations should return a dense flow field as a numpy array with
    shape H x W x 2, where the last dimension stores horizontal and vertical
    displacement. A future adapter may wrap RAFT behind this interface.
    """

    @abstractmethod
    def estimate_flow(self, image_a: Any, image_b: Any) -> np.ndarray:
        """Estimate optical flow from image_a to image_b."""


class TrackerProvider(ABC):
    """Interface for point or object tracking providers.

    Implementations should return ObjectTrack instances with aligned frame
    indices, centers, depths, mask areas, and projection scales. A future
    adapter may wrap CoTracker or an object tracker behind this interface.
    """

    @abstractmethod
    def build_tracks(self, frames: Sequence[FrameObservation]) -> List[ObjectTrack]:
        """Build object tracks from a sequence of frame observations."""

    @abstractmethod
    def track_points(
        self, frames: Sequence[FrameObservation], points: Sequence[Tuple[float, float]]
    ) -> Dict[str, np.ndarray]:
        """Track points through frames and return named trajectory arrays."""
