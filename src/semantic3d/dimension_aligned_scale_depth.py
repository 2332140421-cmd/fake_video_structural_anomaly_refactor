"""Dimension-aligned strict physical scale-depth residuals (strict v2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import yaml

from .observations import FrameObservationJSON, ObjectObservationJSON
from .pose_applicability import ApplicabilityGateResult, applicability_skip_reason
from .physical_prior_gate import evaluate_physical_prior_gate
from .projected_measurement import (
    ProjectedMeasurementRules,
    compute_projected_measurement,
)
from .scale_prior import normalize_label


PathLike = Union[str, Path]
AVAILABLE_STATUSES = {"strict_high", "conditional_physical"}


@dataclass(frozen=True)
class DimensionAlignedPriorEntry:
    """One physical prior whose 3D dimension is aligned to a 2D measurement."""

    label: str
    characteristic_dimension: str
    dimension_definition: str
    unit: str
    min_size: Optional[float]
    max_size: Optional[float]
    projected_measurement: str
    compatibility_group: str
    reliability_status: str
    sources: tuple[Mapping[str, Any], ...]
    source_count: int
    pose_sensitivity: str
    observation_gate: Mapping[str, object]
    reliability_reason: str
    prior_version: str

    @property
    def available(self) -> bool:
        """Return whether this entry can potentially provide strict v2 evidence."""

        return bool(
            self.reliability_status in AVAILABLE_STATUSES
            and self.min_size is not None
            and self.max_size is not None
            and self.min_size > 0
            and self.max_size > self.min_size
        )


@dataclass(frozen=True)
class ResolvedDimensionAlignedPrior:
    """Exact/alias/missing v2 prior resolution with full physical metadata."""

    original_label: str
    resolved_label: str
    resolution: Literal["exact", "alias", "missing"]
    entry: Optional[DimensionAlignedPriorEntry]
    prior_source: Literal["physical"] = "physical"


class DimensionAlignedPriorResolver:
    """Resolve frozen strict v2 physical priors without empirical fallback."""

    def __init__(
        self,
        entries: Mapping[str, DimensionAlignedPriorEntry],
        aliases: Optional[Mapping[str, str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.entries = {normalize_label(label): entry for label, entry in entries.items()}
        self.aliases = {
            normalize_label(label): normalize_label(target)
            for label, target in (aliases or {}).items()
        }
        self.metadata = dict(metadata or {})
        for alias, target in self.aliases.items():
            if target not in self.entries:
                raise ValueError(f"v2 alias {alias!r} points to unknown prior {target!r}.")

    def resolve(self, label: str) -> ResolvedDimensionAlignedPrior:
        """Resolve exact first, then alias, otherwise return missing."""

        normalized = normalize_label(label)
        if normalized in self.entries:
            return ResolvedDimensionAlignedPrior(label, normalized, "exact", self.entries[normalized])
        target = self.aliases.get(normalized)
        if target is not None:
            return ResolvedDimensionAlignedPrior(label, target, "alias", self.entries[target])
        return ResolvedDimensionAlignedPrior(label, normalized, "missing", None)


def _optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def load_dimension_aligned_prior_resolver(path: PathLike) -> DimensionAlignedPriorResolver:
    """Load and validate a frozen strict_physical_v2 YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict) or not isinstance(data.get("scale_priors"), dict):
        raise ValueError(f"strict v2 config requires scale_priors mapping: {config_path}")
    metadata = data.get("metadata", {})
    if int(metadata.get("schema_version", 0)) != 2:
        raise ValueError("Dimension-aligned resolver requires metadata.schema_version=2.")
    entries: dict[str, DimensionAlignedPriorEntry] = {}
    for label, raw in data["scale_priors"].items():
        entry = DimensionAlignedPriorEntry(
            label=normalize_label(str(label)),
            characteristic_dimension=str(raw.get("characteristic_dimension", "")),
            dimension_definition=str(raw.get("dimension_definition", "")),
            unit=str(raw.get("unit", "")),
            min_size=_optional_float(raw.get("min")),
            max_size=_optional_float(raw.get("max")),
            projected_measurement=str(raw.get("projected_measurement", "")),
            compatibility_group=str(raw.get("compatibility_group", "")),
            reliability_status=str(raw.get("reliability_status", "unsupported")),
            sources=tuple(raw.get("sources", [])),
            source_count=int(raw.get("source_count", 0)),
            pose_sensitivity=str(raw.get("pose_sensitivity", "high")),
            observation_gate=dict(raw.get("observation_gate", {})),
            reliability_reason=str(raw.get("reliability_reason", "")),
            prior_version=str(raw.get("prior_version", metadata.get("prior_version", ""))),
        )
        if entry.source_count != len(entry.sources):
            raise ValueError(f"source_count mismatch for strict v2 prior {label!r}.")
        if entry.available and entry.unit != "m":
            raise ValueError(f"Available strict v2 prior {label!r} must use meters.")
        entries[str(label)] = entry
    return DimensionAlignedPriorResolver(entries, data.get("aliases", {}), metadata)


def _nan() -> float:
    return float("nan")


def _finite_positive(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0.0


def _distance(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0


def _base_result(frame: FrameObservationJSON, obj_a: ObjectObservationJSON, obj_b: ObjectObservationJSON) -> dict[str, object]:
    numeric_nan = {
        key: _nan()
        for key in (
            "object_a_prior_low", "object_a_prior_high", "object_b_prior_low", "object_b_prior_high",
            "projected_measurement_a", "projected_measurement_b", "gate_score_a", "gate_score_b",
            "observed_depth_ratio", "expected_ratio_low", "expected_ratio_high", "observed_log_ratio",
            "expected_log_low", "expected_log_high", "rsd_ratio", "rsd_log", "distance_to_interval",
        )
    }
    return {
        **numeric_nan,
        "frame_index": int(frame.frame_index),
        "object_a_id": obj_a.object_id,
        "object_b_id": obj_b.object_id,
        "object_a_label": obj_a.label,
        "object_b_label": obj_b.label,
        "valid": False,
        "skip_reason": "",
        "evidence_tier": "unavailable",
        "prior_source": "physical",
        "explanation_level": "insufficient_evidence",
        "explanation_text": "",
        "applicability_required": False,
        "applicability_status_a": "not_evaluated",
        "applicability_status_b": "not_evaluated",
        "applicability_score_a": _nan(),
        "applicability_score_b": _nan(),
        "applicability_passed_checks_a": "",
        "applicability_passed_checks_b": "",
        "applicability_failed_checks_a": "",
        "applicability_failed_checks_b": "",
    }


def _prior_skip_reason(entry: Optional[DimensionAlignedPriorEntry], side: str) -> str:
    if entry is None:
        return f"missing_prior_{side}"
    if entry.reliability_status == "pose_sensitive":
        return f"pose_sensitive_prior_{side}"
    if entry.reliability_status in {"unsupported", "insufficient_source"} or not entry.available:
        return f"unsupported_prior_{side}"
    return ""


def _skip_explanation(
    reason: str,
    obj_a: ObjectObservationJSON,
    obj_b: ObjectObservationJSON,
    prior_a: ResolvedDimensionAlignedPrior,
    prior_b: ResolvedDimensionAlignedPrior,
) -> str:
    side = "a" if reason.endswith("_a") else "b"
    obj = obj_a if side == "a" else obj_b
    prior = prior_a if side == "a" else prior_b
    if reason == "incompatible_projected_dimensions":
        return (
            f"{obj_a.label} 使用 {prior_a.entry.compatibility_group if prior_a.entry else 'unknown'} 投影维度，"
            f"{obj_b.label} 使用 {prior_b.entry.compatibility_group if prior_b.entry else 'unknown'} 投影维度；"
            "在缺少相机内参跨轴标定时不直接比较，本对象对被标记为维度不兼容。"
        )
    if reason.startswith("observation_gate_failed"):
        return f"{obj.label} 未通过完整性、检测质量或宽高比的通用观测门控，因此不计算 strict R_sd。"
    if reason.startswith("invalid_projected_measurement"):
        return f"{obj.label} 的二维投影量无效或过小，本对象对没有可用的维度对齐证据。"
    if reason.startswith("pose_sensitive_prior"):
        detail = prior.entry.reliability_reason if prior.entry else "姿态未解析"
        return f"{obj.label} 的二维投影高度或宽度高度依赖姿态（{detail}），当前不参与 strict R_sd。"
    if reason.startswith("unsupported_prior") or reason.startswith("missing_prior"):
        return f"{obj.label} 当前没有可用的 dimension-aligned 物理先验，本对象对证据不足而非残差为零。"
    if reason.startswith("invalid_depth"):
        return f"{obj.label} 的相对深度不是有限正值，无法计算尺度—深度比。"
    if reason.startswith("person_"):
        return "person 观测不满足完整直立身高先验的关键点或姿态适用条件，因此 R_sd 保持 NaN。"
    if reason.startswith("cup_"):
        return "cup 观测未通过直立高度的尺寸、边界、稳定性或深度质量门控，因此 R_sd 保持 NaN。"
    if reason == "invalid_depth_mode":
        return "strict v2 要求 real_depth_invert，且项目约定 depth 数值越大表示越远。"
    return "本对象对未形成有效的 dimension-aligned strict physical R_sd 证据。"


def compute_dimension_aligned_rsd(
    frame: FrameObservationJSON,
    obj_a: ObjectObservationJSON,
    obj_b: ObjectObservationJSON,
    resolver: DimensionAlignedPriorResolver,
    measurement_rules: ProjectedMeasurementRules,
    depth_mode: str = "real_depth_invert",
    history_a: Optional[Sequence[ObjectObservationJSON]] = None,
    history_b: Optional[Sequence[ObjectObservationJSON]] = None,
    applicability_a: Optional[ApplicabilityGateResult] = None,
    applicability_b: Optional[ApplicabilityGateResult] = None,
    require_applicability: bool = False,
) -> dict[str, object]:
    """Compute one strict v2 pair only after prior, gate, and dimension checks."""

    row = _base_result(frame, obj_a, obj_b)
    row["applicability_required"] = require_applicability
    for side, result in (("a", applicability_a), ("b", applicability_b)):
        if result is None:
            continue
        row.update(
            {
                f"applicability_status_{side}": result.applicability_status,
                f"applicability_score_{side}": result.applicability_score,
                f"applicability_passed_checks_{side}": "|".join(result.passed_checks),
                f"applicability_failed_checks_{side}": "|".join(result.failed_checks),
            }
        )
    prior_a = resolver.resolve(obj_a.label)
    prior_b = resolver.resolve(obj_b.label)
    entry_a, entry_b = prior_a.entry, prior_b.entry
    row.update(
        {
            "object_a_canonical_label": prior_a.resolved_label,
            "object_b_canonical_label": prior_b.resolved_label,
            "object_a_prior_status": entry_a.reliability_status if entry_a else "missing",
            "object_b_prior_status": entry_b.reliability_status if entry_b else "missing",
            "object_a_prior_resolution": prior_a.resolution,
            "object_b_prior_resolution": prior_b.resolution,
            "characteristic_dimension_a": entry_a.characteristic_dimension if entry_a else "",
            "characteristic_dimension_b": entry_b.characteristic_dimension if entry_b else "",
            "projected_measurement_type_a": entry_a.projected_measurement if entry_a else "",
            "projected_measurement_type_b": entry_b.projected_measurement if entry_b else "",
            "compatibility_group_a": entry_a.compatibility_group if entry_a else "",
            "compatibility_group_b": entry_b.compatibility_group if entry_b else "",
            "object_a_prior_low": entry_a.min_size if entry_a and entry_a.min_size is not None else _nan(),
            "object_a_prior_high": entry_a.max_size if entry_a and entry_a.max_size is not None else _nan(),
            "object_b_prior_low": entry_b.min_size if entry_b and entry_b.min_size is not None else _nan(),
            "object_b_prior_high": entry_b.max_size if entry_b and entry_b.max_size is not None else _nan(),
            "prior_version": resolver.metadata.get("prior_version", "strict_physical_v2"),
            "depth_mode": depth_mode,
        }
    )

    if _finite_positive(obj_a.depth) and _finite_positive(obj_b.depth):
        row["observed_depth_ratio"] = float(obj_a.depth) / float(obj_b.depth)
        row["observed_log_ratio"] = math.log(float(obj_a.depth)) - math.log(float(obj_b.depth))

    skip_reason = _prior_skip_reason(entry_a, "a") or _prior_skip_reason(entry_b, "b")
    if depth_mode != "real_depth_invert":
        skip_reason = "invalid_depth_mode"
    elif not _finite_positive(obj_a.depth):
        skip_reason = "invalid_depth_a"
    elif not _finite_positive(obj_b.depth):
        skip_reason = "invalid_depth_b"
    if skip_reason:
        row["skip_reason"] = skip_reason
        row["explanation_text"] = _skip_explanation(skip_reason, obj_a, obj_b, prior_a, prior_b)
        return row
    assert entry_a is not None and entry_b is not None

    measurement_a = compute_projected_measurement(
        obj_a, frame.width, frame.height, entry_a.projected_measurement, measurement_rules
    )
    measurement_b = compute_projected_measurement(
        obj_b, frame.width, frame.height, entry_b.projected_measurement, measurement_rules
    )
    row.update(
        {
            "projected_measurement_a": measurement_a.value,
            "projected_measurement_b": measurement_b.value,
            "measurement_quality_a": measurement_a.measurement_quality,
            "measurement_quality_b": measurement_b.measurement_quality,
            "measurement_invalid_reason_a": measurement_a.invalid_reason,
            "measurement_invalid_reason_b": measurement_b.invalid_reason,
        }
    )
    if not measurement_a.valid:
        skip_reason = "invalid_projected_measurement_a"
    elif not measurement_b.valid:
        skip_reason = "invalid_projected_measurement_b"
    elif entry_a.compatibility_group != entry_b.compatibility_group:
        skip_reason = "incompatible_projected_dimensions"
    else:
        skip_reason = ""
    if skip_reason:
        row["skip_reason"] = skip_reason
        row["explanation_text"] = _skip_explanation(skip_reason, obj_a, obj_b, prior_a, prior_b)
        return row

    gate_a = evaluate_physical_prior_gate(
        obj_a, frame.width, frame.height, measurement_a, entry_a.observation_gate, measurement_rules, history_a
    )
    gate_b = evaluate_physical_prior_gate(
        obj_b, frame.width, frame.height, measurement_b, entry_b.observation_gate, measurement_rules, history_b
    )
    row.update(
        {
            "gate_passed_a": gate_a.gate_passed,
            "gate_passed_b": gate_b.gate_passed,
            "gate_score_a": gate_a.gate_score,
            "gate_score_b": gate_b.gate_score,
            "gate_reasons_a": "|".join(gate_a.gate_reasons),
            "gate_reasons_b": "|".join(gate_b.gate_reasons),
            "failed_gate_reasons_a": "|".join(gate_a.failed_gate_reasons),
            "failed_gate_reasons_b": "|".join(gate_b.failed_gate_reasons),
        }
    )
    if not gate_a.gate_passed:
        skip_reason = "observation_gate_failed_a"
    elif not gate_b.gate_passed:
        skip_reason = "observation_gate_failed_b"
    else:
        skip_reason = ""
    if skip_reason:
        row["skip_reason"] = skip_reason
        row["explanation_text"] = _skip_explanation(skip_reason, obj_a, obj_b, prior_a, prior_b)
        return row

    if require_applicability:
        decisions = ((obj_a, applicability_a, "a"), (obj_b, applicability_b, "b"))
        # A person-height failure is reported before a cup-quality failure so
        # the physical-dimension applicability error remains explicit.
        ordered_decisions = sorted(
            decisions,
            key=lambda item: 0 if normalize_label(item[0].label) == "person" else 1,
        )
        for obj, result, _ in ordered_decisions:
            label = normalize_label(obj.label)
            if label not in {"person", "cup"}:
                continue
            if result is None:
                skip_reason = (
                    "person_insufficient_keypoints"
                    if label == "person"
                    else "cup_measurement_not_applicable"
                )
            else:
                skip_reason = applicability_skip_reason(label, result)
            if skip_reason:
                row["skip_reason"] = skip_reason
                row["explanation_text"] = _skip_explanation(
                    skip_reason, obj_a, obj_b, prior_a, prior_b
                )
                return row

    tier = (
        "strict_high"
        if entry_a.reliability_status == entry_b.reliability_status == "strict_high"
        else "conditional_physical"
    )
    p_a, p_b = measurement_a.value, measurement_b.value
    assert entry_a.min_size is not None and entry_a.max_size is not None
    assert entry_b.min_size is not None and entry_b.max_size is not None
    ratio = float(obj_a.depth) / float(obj_b.depth)
    low = entry_a.min_size / entry_b.max_size * p_b / p_a
    high = entry_a.max_size / entry_b.min_size * p_b / p_a
    observed_log = math.log(float(obj_a.depth)) - math.log(float(obj_b.depth))
    log_low = math.log(entry_a.min_size / entry_b.max_size) + math.log(p_b) - math.log(p_a)
    log_high = math.log(entry_a.max_size / entry_b.min_size) + math.log(p_b) - math.log(p_a)
    rsd_ratio = _distance(ratio, low, high)
    rsd_log = _distance(observed_log, log_low, log_high)
    normal = rsd_log == 0.0
    explanation = (
        f"{obj_a.label} 与 {obj_b.label} 均采用 {entry_a.compatibility_group} 投影尺度；"
        f"分别使用 {entry_a.characteristic_dimension} 与 {entry_b.characteristic_dimension} 物理先验。"
        f"两个对象通过通用观测门控，因此使用 {tier} 证据计算尺度—深度残差。"
    )
    if normal:
        explanation += "观测尺度—深度比位于物理参考区间内。"
    else:
        explanation += "观测尺度—深度比超出物理参考区间。"
    row.update(
        {
            "expected_ratio_low": low,
            "expected_ratio_high": high,
            "expected_log_low": log_low,
            "expected_log_high": log_high,
            "rsd_ratio": rsd_ratio,
            "rsd_log": rsd_log,
            "distance_to_interval": rsd_log,
            "valid": True,
            "skip_reason": "",
            "evidence_tier": tier,
            "combined_available": True,
            "explanation_level": "normal" if normal else "anomaly",
            "explanation_text": explanation,
        }
    )
    return row


def rsd_2d_dimension_aligned(
    frame: FrameObservationJSON,
    obj_a: ObjectObservationJSON,
    obj_b: ObjectObservationJSON,
    resolver: DimensionAlignedPriorResolver,
    measurement_rules: ProjectedMeasurementRules,
    depth_mode: str = "real_depth_invert",
    history_a: Optional[Sequence[ObjectObservationJSON]] = None,
    history_b: Optional[Sequence[ObjectObservationJSON]] = None,
    applicability_a: Optional[ApplicabilityGateResult] = None,
    applicability_b: Optional[ApplicabilityGateResult] = None,
    require_applicability: bool = False,
) -> dict[str, object]:
    """Explicit 2D baseline name for the frozen dimension-aligned R_sd."""

    return compute_dimension_aligned_rsd(
        frame,
        obj_a,
        obj_b,
        resolver,
        measurement_rules,
        depth_mode=depth_mode,
        history_a=history_a,
        history_b=history_b,
        applicability_a=applicability_a,
        applicability_b=applicability_b,
        require_applicability=require_applicability,
    )
