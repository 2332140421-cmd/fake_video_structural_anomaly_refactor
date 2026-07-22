"""Stable grouping of originals, derivatives, transcodes, crops, and clips."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .schema import LabelRecord, SourceGroupRecord


def stable_source_group_id(dataset_name: str, source_identity: str) -> str:
    """Return an order-independent source-group ID from a canonical identity."""

    dataset = dataset_name.strip().lower()
    identity = source_identity.strip().lower()
    if not dataset or not identity:
        raise ValueError("dataset_name and source_identity must be non-empty")
    payload = json.dumps([dataset, identity], separators=(",", ":"), ensure_ascii=True)
    return f"srcgrp_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


def assign_source_groups(
    video_rows: list[Mapping[str, Any]],
    labels_by_source_name: Mapping[str, LabelRecord],
    *,
    dataset_name: str,
) -> dict[str, SourceGroupRecord]:
    """Assign known origins together and mark hash fallbacks for manual review."""

    result: dict[str, SourceGroupRecord] = {}
    for row in video_rows:
        source_name = str(row["source_name"])
        label = labels_by_source_name.get(source_name)
        explicit_group = label.source_group_id if label else ""
        original_identity = label.original_source_identity if label else ""
        if explicit_group:
            group_id = explicit_group
            basis = "explicit_manifest_source_group"
            review = False
            reason = ""
        elif original_identity:
            group_id = stable_source_group_id(dataset_name, original_identity)
            basis = "original_source_identity"
            review = False
            reason = ""
        else:
            group_id = stable_source_group_id(dataset_name, f"unresolved:{row['source_sha256']}")
            basis = "file_hash_fallback_unresolved_origin"
            review = True
            reason = "original_real_and_derivative_relationship_not_provided"
        result[str(row["video_id"])] = SourceGroupRecord(
            video_id=str(row["video_id"]),
            source_group_id=group_id,
            grouping_basis=basis,
            original_source_identity=original_identity,
            source_group_review_required=review,
            review_reason=reason,
        )
    return result
