"""Hard guards preventing missing providers from becoming anomaly evidence."""

from __future__ import annotations

from enum import StrEnum


class EvidenceUse(StrEnum):
    """Purposes for which an eligibility state may be consumed."""

    QUALITY_CONTROL = "quality_control"
    MISSINGNESS_AUDIT = "missingness_audit"
    ANOMALY_SCORE = "anomaly_score"
    AUTHENTICITY_LABEL = "authenticity_label"
    NORMAL_REFERENCE_FIT = "normal_reference_fit"
    SUPERVISED_AGGREGATION = "supervised_aggregation"


_PROVIDER_FAILURE_ALLOWED_USES = {
    EvidenceUse.QUALITY_CONTROL,
    EvidenceUse.MISSINGNESS_AUDIT,
}


def validate_evidence_use(eligibility: str, purpose: EvidenceUse | str) -> None:
    """Reject any provider-failure use outside quality and missingness control."""

    use = EvidenceUse(purpose)
    if eligibility == "provider_failed" and use not in _PROVIDER_FAILURE_ALLOWED_USES:
        raise ValueError(
            f"provider_failed cannot be used for {use.value}; it is quality/missingness metadata only"
        )


def can_contribute_evidence(eligibility: str, purpose: EvidenceUse | str) -> bool:
    """Return whether a row may contribute, while enforcing provider-failure policy."""

    validate_evidence_use(eligibility, purpose)
    return eligibility == "eligible"

