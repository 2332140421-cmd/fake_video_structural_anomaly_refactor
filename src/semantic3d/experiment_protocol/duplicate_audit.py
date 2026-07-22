"""Exact and near-duplicate video auditing without anomaly residuals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .schema import SplitAssignment, VideoInventoryRecord


class VisualEmbeddingProvider(ABC):
    """Optional future interface for label-free visual duplicate embeddings."""

    @abstractmethod
    def embed_video(self, video_path: str | Path) -> np.ndarray:
        """Return a deterministic content embedding without reading labels."""


def _dhash(frame: np.ndarray, hash_size: int = 8) -> str:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    bits = resized[:, 1:] > resized[:, :-1]
    packed = np.packbits(bits.reshape(-1).astype(np.uint8))
    return packed.tobytes().hex()


def video_perceptual_hashes(video_path: str | Path, sample_count: int = 5) -> tuple[str, ...]:
    """Sample deterministic positions and return frame difference hashes."""

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return ()
    try:
        frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        indices = np.linspace(0, frame_count - 1, max(1, sample_count), dtype=int)
        hashes = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok and frame is not None:
                hashes.append(_dhash(frame))
        return tuple(hashes)
    finally:
        capture.release()


def _hash_similarity(first: Sequence[str], second: Sequence[str]) -> float:
    if not first or not second:
        return 0.0
    similarities = []
    for left, right in zip(first, second):
        left_bytes = bytes.fromhex(left)
        right_bytes = bytes.fromhex(right)
        total_bits = 8 * min(len(left_bytes), len(right_bytes))
        distance = sum((a ^ b).bit_count() for a, b in zip(left_bytes, right_bytes))
        similarities.append(1.0 - distance / total_bits if total_bits else 0.0)
    return float(np.mean(similarities)) if similarities else 0.0


def audit_duplicates(
    inventory: Iterable[VideoInventoryRecord],
    assignments: Iterable[SplitAssignment],
    *,
    near_duplicate_threshold: float = 0.92,
    sample_count: int = 5,
) -> list[dict[str, Any]]:
    """Compare hashes, metadata, and perceptual samples for every video pair."""

    rows = list(inventory)
    split_by_video = {row.video_id: row.split for row in assignments}
    hashes = {row.video_id: video_perceptual_hashes(row.source_path, sample_count) for row in rows}
    results = []
    for first, second in combinations(sorted(rows, key=lambda item: item.video_id), 2):
        exact = first.source_sha256 == second.source_sha256
        duration_denominator = max(first.duration_seconds, second.duration_seconds, 1e-8)
        duration_similarity = max(
            0.0, 1.0 - abs(first.duration_seconds - second.duration_seconds) / duration_denominator
        )
        metadata_similarity = float(
            np.mean(
                [
                    duration_similarity,
                    1.0 if (first.width, first.height) == (second.width, second.height) else 0.0,
                    max(0.0, 1.0 - abs(first.fps - second.fps) / max(first.fps, second.fps, 1e-8)),
                ]
            )
        )
        perceptual_similarity = _hash_similarity(hashes[first.video_id], hashes[second.video_id])
        near_score = 1.0 if exact else 0.75 * perceptual_similarity + 0.25 * metadata_similarity
        same_group = first.source_group_id == second.source_group_id
        split_conflict = split_by_video.get(first.video_id) != split_by_video.get(second.video_id)
        results.append(
            {
                "video_id_a": first.video_id,
                "video_id_b": second.video_id,
                "exact_duplicate": exact,
                "near_duplicate_score": near_score,
                "perceptual_hash_score": perceptual_similarity,
                "metadata_similarity": metadata_similarity,
                "same_source_group": same_group,
                "split_conflict": split_conflict,
                "review_required": bool(
                    (exact or near_score >= near_duplicate_threshold) and (split_conflict or not same_group)
                ),
                "hash_sample_count_a": len(hashes[first.video_id]),
                "hash_sample_count_b": len(hashes[second.video_id]),
                "embedding_provider": "not_configured",
                "residual_fields_used": False,
            }
        )
    return results
