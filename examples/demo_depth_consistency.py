#!/usr/bin/env python3
"""Demonstrate R_depth_cons on synthetic tracked object sequences."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_environment() -> None:
    """Re-execute with the project .venv Python when available."""

    project_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()
    if project_python.exists() and current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])


_ensure_project_environment()

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic3d.depth_temporal_consistency import (  # noqa: E402
    aggregate_track_depth_residuals,
    compute_depth_temporal_residual,
    save_depth_consistency_tracks_plot,
    with_frame_indices,
)
from semantic3d.observations import ObjectObservationJSON  # noqa: E402


FRAME_AREA = 640.0 * 480.0
DEPTH_REFERENCE = 10.0


def _object(
    track_id: str,
    label: str,
    frame_index: int,
    depth: float,
    projection_scale: float,
) -> ObjectObservationJSON:
    """Create one tracked object with a target projection scale."""

    return ObjectObservationJSON(
        object_id=f"{track_id}_f{frame_index}",
        label=label,
        canonical_label=label,
        track_id=track_id,
        mask_area=(projection_scale**2) * FRAME_AREA,
        frame_area=FRAME_AREA,
        depth=depth,
        confidence=1.0,
        bbox=[100.0, 100.0, 100.0 + projection_scale * 500.0, 180.0],
    )


def _track_sequence(
    track_id: str,
    label: str,
    depths: list[float],
    projection_scales: list[float],
) -> list[ObjectObservationJSON]:
    """Create a synthetic object sequence."""

    return [
        _object(track_id, label, index, depth, projection)
        for index, (depth, projection) in enumerate(zip(depths, projection_scales))
    ]


def _compute_sequence_residuals(
    objects: list[ObjectObservationJSON],
    tolerance: float = 0.10,
) -> list:
    """Compute transition residuals for one synthetic sequence."""

    results = []
    for previous, current in zip(objects[:-1], objects[1:]):
        result = compute_depth_temporal_residual(
            previous,
            current,
            previous_depth_reference=DEPTH_REFERENCE,
            current_depth_reference=DEPTH_REFERENCE,
            tolerance=tolerance,
        )
        results.append(
            with_frame_indices(
                result,
                previous_frame_index=int(previous.object_id.rsplit("_f", 1)[-1]),
                current_frame_index=int(current.object_id.rsplit("_f", 1)[-1]),
            )
        )
    return results


def main() -> None:
    """Run the synthetic R_depth_cons demo."""

    sequences = {
        "stable_object": _track_sequence(
            "trk_stable",
            "person",
            depths=[5.0, 5.05, 4.95, 5.0],
            projection_scales=[0.20, 0.20, 0.202, 0.199],
        ),
        "approaching_object_consistent": _track_sequence(
            "trk_approach",
            "car",
            depths=[8.0, 6.0, 4.0, 3.0],
            projection_scales=[0.10, 0.1333, 0.20, 0.2667],
        ),
        "depth_jump_anomaly": _track_sequence(
            "trk_depth_jump",
            "cup",
            depths=[4.0, 4.1, 9.0, 9.2],
            projection_scales=[0.12, 0.12, 0.12, 0.12],
        ),
        "projection_jump_anomaly": _track_sequence(
            "trk_projection_jump",
            "bottle",
            depths=[6.0, 6.0, 6.0, 6.0],
            projection_scales=[0.08, 0.08, 0.22, 0.22],
        ),
    }

    all_results = []
    for name, objects in sequences.items():
        results = _compute_sequence_residuals(objects)
        all_results.extend(results)
        depths = [obj.depth for obj in objects]
        projections = [
            (obj.mask_area / obj.frame_area) ** 0.5
            for obj in objects
        ]
        residuals = [result.residual for result in results]
        print(f"\n{name}")
        print(f"  depth sequence: {[round(value, 4) for value in depths]}")
        print(f"  projection scale sequence: {[round(value, 4) for value in projections]}")
        print(f"  residual sequence: {[round(value, 4) for value in residuals]}")
        print(f"  track summary: {aggregate_track_depth_residuals(results)[0]}")

    output_path = (
        PROJECT_ROOT / "outputs" / "visualizations" / "depth_consistency_tracks.png"
    )
    save_depth_consistency_tracks_plot(all_results, output_path)
    print(f"\nSaved visualization: {output_path}")
    print("Reminder: depth values are monocular relative depths, not metric meters.")


if __name__ == "__main__":
    main()
