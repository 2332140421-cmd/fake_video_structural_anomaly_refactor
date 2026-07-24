"""P4-C3C-A2 minimal, precomputed-evidence training interfaces."""

from .contracts import (
    EVIDENCE_MANIFEST_SCHEMA_VERSION,
    FEATURE_CONTRACT_SCHEMA_VERSION,
    TRAINING_CHECKPOINT_SCHEMA_VERSION,
    TRAINING_CONFIG_SCHEMA_VERSION,
    FeatureContract,
    load_feature_contract,
)
from .data import (
    EvidenceTrainingDataset,
    TrainingDataBundle,
    build_training_dataloaders,
    collate_evidence_samples,
)
from .loss import MaskedBinaryLoss, MaskedLossResult
from .model import MinimalMissingAwareEvidenceHead

__all__ = [
    "EVIDENCE_MANIFEST_SCHEMA_VERSION",
    "FEATURE_CONTRACT_SCHEMA_VERSION",
    "TRAINING_CHECKPOINT_SCHEMA_VERSION",
    "TRAINING_CONFIG_SCHEMA_VERSION",
    "EvidenceTrainingDataset",
    "FeatureContract",
    "MaskedBinaryLoss",
    "MaskedLossResult",
    "MinimalMissingAwareEvidenceHead",
    "TrainingDataBundle",
    "build_training_dataloaders",
    "collate_evidence_samples",
    "load_feature_contract",
]
