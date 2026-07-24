"""Masked binary loss for known labels and observable M6 evidence only."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MaskedLossResult:
    loss: torch.Tensor | None
    supervised_mask: torch.Tensor
    supervised_count: int
    missing_label_count: int
    no_evidence_count: int


class MaskedBinaryLoss(nn.Module):
    """BCEWithLogitsLoss that never treats missing labels/evidence as negatives."""

    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        if pos_weight is not None and pos_weight <= 0:
            raise ValueError("pos_weight must be positive.")
        tensor = None if pos_weight is None else torch.tensor(float(pos_weight))
        self.register_buffer("pos_weight", tensor)

    def forward(
        self,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        label_mask: torch.Tensor,
        valid_sample_mask: torch.Tensor,
    ) -> MaskedLossResult:
        if logits.ndim != 1 or labels.shape != logits.shape:
            raise ValueError("logits and labels must have identical [batch] shape.")
        if label_mask.shape != logits.shape or label_mask.dtype != torch.bool:
            raise ValueError("label_mask must be bool with [batch] shape.")
        if (
            valid_sample_mask.shape != logits.shape
            or valid_sample_mask.dtype != torch.bool
        ):
            raise ValueError("valid_sample_mask must be bool with [batch] shape.")
        if torch.any(label_mask & ~torch.isfinite(labels)):
            raise ValueError("Known labels contain NaN or Inf.")
        if torch.any(label_mask & ((labels != 0) & (labels != 1))):
            raise ValueError("Known labels must be binary 0/1.")
        supervised_mask = label_mask & valid_sample_mask
        count = int(supervised_mask.sum().item())
        if count == 0:
            loss = None
        else:
            selected_logits = logits[supervised_mask]
            if not torch.isfinite(selected_logits).all():
                raise FloatingPointError("Supervised logits contain NaN or Inf.")
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            loss = loss_fn(selected_logits, labels[supervised_mask])
        return MaskedLossResult(
            loss=loss,
            supervised_mask=supervised_mask,
            supervised_count=count,
            missing_label_count=int((~label_mask).sum().item()),
            no_evidence_count=int((~valid_sample_mask).sum().item()),
        )
