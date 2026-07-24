"""Small missing-aware evidence head for A2 engineering validation."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MinimalMissingAwareEvidenceHead(nn.Module):
    """Project fixed M6 branches, gate by masks/reliability, and emit one logit."""

    def __init__(
        self,
        *,
        branch_count: int,
        feature_dim: int,
        hidden_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if branch_count <= 0 or feature_dim <= 0 or hidden_dim <= 0:
            raise ValueError("branch_count, feature_dim, and hidden_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1).")
        self.branch_count = int(branch_count)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.branch_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.feature_dim, self.hidden_dim),
                    nn.GELU(),
                )
                for _ in range(self.branch_count)
            ]
        )
        self.branch_gates = nn.ModuleList(
            [nn.Linear(2, self.hidden_dim) for _ in range(self.branch_count)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def config_dict(self) -> dict[str, Any]:
        return {
            "class_name": type(self).__name__,
            "branch_count": self.branch_count,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }

    def forward(
        self,
        *,
        features: torch.Tensor,
        feature_mask: torch.Tensor,
        observability: torch.Tensor,
        reliability: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, branch, feature].")
        expected = (self.branch_count, self.feature_dim)
        if tuple(features.shape[1:]) != expected:
            raise ValueError(
                f"features trailing shape {tuple(features.shape[1:])} != {expected}."
            )
        if feature_mask.shape != features.shape or feature_mask.dtype != torch.bool:
            raise ValueError("feature_mask must be bool with the same shape as features.")
        if observability.shape != features.shape[:2] or observability.dtype != torch.bool:
            raise ValueError(
                "observability must be bool with shape [batch, branch]."
            )
        if not features.is_floating_point():
            raise ValueError("features must use a floating dtype.")
        if reliability is None:
            reliability = torch.ones(
                features.shape[:2],
                dtype=features.dtype,
                device=features.device,
            )
        if reliability.shape != features.shape[:2] or not reliability.is_floating_point():
            raise ValueError(
                "reliability must be floating with shape [batch, branch]."
            )
        if torch.any(feature_mask & ~torch.isfinite(features)):
            raise ValueError("Observed features contain NaN or Inf.")
        if not torch.isfinite(reliability).all():
            raise ValueError("reliability contains NaN or Inf.")
        if torch.any((reliability < 0) | (reliability > 1)):
            raise ValueError("reliability must be in [0,1].")

        safe_features = torch.where(feature_mask, features, torch.zeros_like(features))
        branch_valid = feature_mask.all(dim=-1) & observability
        embeddings = []
        gates = []
        for index in range(self.branch_count):
            projected = self.branch_projections[index](safe_features[:, index, :])
            gate_input = torch.stack(
                (
                    reliability[:, index],
                    observability[:, index].to(features.dtype),
                ),
                dim=-1,
            )
            gate = torch.sigmoid(self.branch_gates[index](gate_input))
            gate = gate * reliability[:, index, None]
            gate = gate * branch_valid[:, index, None].to(features.dtype)
            embeddings.append(projected)
            gates.append(gate)
        stacked_embeddings = torch.stack(embeddings, dim=1)
        stacked_gates = torch.stack(gates, dim=1)
        denominator = stacked_gates.sum(dim=1).clamp_min(
            torch.finfo(features.dtype).eps
        )
        pooled = (stacked_embeddings * stacked_gates).sum(dim=1) / denominator
        valid_sample_mask = branch_valid.any(dim=1)
        raw_logits = self.classifier(pooled).squeeze(-1)
        invalid_fill = torch.full_like(raw_logits, float("nan"))
        logits = torch.where(valid_sample_mask, raw_logits, invalid_fill)
        if torch.any(valid_sample_mask & ~torch.isfinite(logits)):
            raise FloatingPointError("Finite observed inputs produced NaN/Inf logits.")
        anomaly_probability = torch.sigmoid(logits)
        return {
            "logits": logits,
            "anomaly_probability": anomaly_probability,
            "valid_sample_mask": valid_sample_mask,
            "branch_valid_mask": branch_valid,
        }
