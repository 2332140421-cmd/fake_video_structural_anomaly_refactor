"""Minimal paper manifest: path, label, split, and optional annotations."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestRow:
    video_path: str
    label: int | None
    split: str
    temporal_annotation: str = ""
    spatial_annotation: str = ""


def read_manifest(path: str | Path, *, split: str | None = None) -> list[ManifestRow]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = []
    for row in rows:
        row_split = str(row.get("split", "")).strip()
        if row_split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split {row_split!r}.")
        if split is not None and row_split != split:
            continue
        raw_label = str(row.get("label", "")).strip()
        label = None if raw_label == "" else int(raw_label)
        if label not in {None, 0, 1}:
            raise ValueError("Labels must be 0, 1, or empty.")
        video_path = str(row.get("video_path", "")).strip()
        if not video_path:
            raise ValueError("Every manifest row requires video_path.")
        output.append(
            ManifestRow(
                video_path=video_path,
                label=label,
                split=row_split,
                temporal_annotation=str(row.get("temporal_annotation", "") or ""),
                spatial_annotation=str(row.get("spatial_annotation", "") or ""),
            )
        )
    return output
