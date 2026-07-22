"""Official-split preserving and group-aware split planning."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Iterable

from .schema import SPLIT_ALGORITHM_VERSION, SplitAssignment, VideoInventoryRecord

SPLIT_INPUT_FIELDS = (
    "source_group_id",
    "binary_label",
    "manipulation_type",
    "source_dataset",
    "resolution",
    "compression",
    "duration",
    "geometry_mode",
    "branch_eligibility",
)


def normalize_split_name(value: str) -> str:
    """Normalize common split aliases."""

    normalized = value.strip().lower()
    return {"val": "validation", "valid": "validation", "dev": "validation"}.get(normalized, normalized)


def _stable_group_order(group_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()


def _record_features(row: VideoInventoryRecord) -> Counter[str]:
    """Return non-residual balancing strata for one inventory record."""

    if row.duration_seconds < 10:
        duration_bucket = "short"
    elif row.duration_seconds < 60:
        duration_bucket = "medium"
    else:
        duration_bucket = "long"
    metadata = row.metadata
    features = Counter(
        {
            "total": 1,
            f"label:{row.binary_label}": 1,
            f"manipulation:{row.manipulation_type}": 1,
            f"dataset:{row.dataset_name}": 1,
            f"resolution:{row.width}x{row.height}": 1,
            f"duration:{duration_bucket}": 1,
            f"compression:{metadata.get('compression', 'unknown')}": 1,
            f"geometry:{metadata.get('geometry_mode', 'unknown')}": 1,
        }
    )
    for tier in metadata.get("eligible_tiers", []):
        features[f"branch_tier:{tier}"] += 1
    return features


def plan_group_aware_split(
    records: Iterable[VideoInventoryRecord],
    *,
    ratios: dict[str, float] | None = None,
    random_seed: int = 20260722,
    official_split_by_video: dict[str, str] | None = None,
    preserve_declared_smoke_split: bool = False,
) -> list[SplitAssignment]:
    """Assign whole source groups; never split frames or clips independently."""

    rows = list(records)
    ratios = ratios or {"train": 0.7, "validation": 0.15, "test": 0.15}
    if not rows:
        return []
    if abs(sum(ratios.values()) - 1.0) > 1e-6 or any(value < 0 for value in ratios.values()):
        raise ValueError("split ratios must be non-negative and sum to one")
    official = official_split_by_video or {}
    group_rows: dict[str, list[VideoInventoryRecord]] = defaultdict(list)
    for row in rows:
        group_rows[row.source_group_id].append(row)

    assignment_by_group: dict[str, tuple[str, str, bool]] = {}
    for group_id, members in group_rows.items():
        official_values = {normalize_split_name(official[m.video_id]) for m in members if m.video_id in official}
        if len(official_values) > 1:
            # Preserve the conflict for audit instead of silently rewriting it.
            assignment_by_group[group_id] = ("official_conflict", "official_split_conflict", False)
        elif official_values:
            assignment_by_group[group_id] = (next(iter(official_values)), "official", True)
        elif preserve_declared_smoke_split:
            declared = {normalize_split_name(m.declared_split) for m in members if m.declared_split}
            split = next(iter(declared)) if len(declared) == 1 else "validation"
            assignment_by_group[group_id] = (split, "independent_manifest_protocol_smoke", False)

    unassigned = [group_id for group_id in group_rows if group_id not in assignment_by_group]
    global_features = sum((_record_features(row) for row in rows), Counter())
    targets = {
        split: {feature: ratios[split] * count for feature, count in global_features.items()}
        for split in ratios
    }
    current_features: dict[str, Counter[str]] = {name: Counter() for name in ratios}
    for group_id, (split, _, _) in assignment_by_group.items():
        if split in current_features:
            current_features[split].update(
                sum((_record_features(row) for row in group_rows[group_id]), Counter())
            )
    for group_id in sorted(unassigned, key=lambda value: _stable_group_order(value, random_seed)):
        members = group_rows[group_id]
        group_features = sum((_record_features(row) for row in members), Counter())

        def incremental_cost(split: str) -> tuple[float, str]:
            before = current_features[split]
            cost = 0.0
            for feature, addition in group_features.items():
                target = max(float(targets[split].get(feature, 0.0)), 1.0)
                old_error = (before[feature] - target) / target
                new_error = (before[feature] + addition - target) / target
                cost += new_error * new_error - old_error * old_error
            return cost, split

        split = min(
            ratios,
            key=incremental_cost,
        )
        assignment_by_group[group_id] = (split, "group_aware_planner", False)
        current_features[split].update(group_features)

    assignments = []
    for row in sorted(rows, key=lambda item: item.video_id):
        split, source, preserved = assignment_by_group[row.source_group_id]
        assignments.append(
            SplitAssignment(
                video_id=row.video_id,
                source_group_id=row.source_group_id,
                split=split,
                split_source=source,
                official_split_preserved=preserved,
                algorithm_version=SPLIT_ALGORITHM_VERSION,
                random_seed=random_seed,
                decision_inputs=SPLIT_INPUT_FIELDS,
            )
        )
    return assignments


def split_rows(records: Iterable[SplitAssignment]) -> list[dict[str, object]]:
    """Convert split dataclasses to table rows."""

    return [asdict(record) for record in records]
