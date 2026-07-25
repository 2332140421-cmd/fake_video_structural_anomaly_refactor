"""A sub-100k parameter head that learns only residual-sequence fusion."""

from __future__ import annotations

import torch
from torch import nn


class ResidualTemporalHead(nn.Module):
    """Fuse residual values, availability, and confidence over time."""

    def __init__(self, residual_count: int, hidden_size: int = 32) -> None:
        super().__init__()
        if residual_count < 1 or hidden_size < 1:
            raise ValueError("residual_count and hidden_size must be positive.")
        self.residual_count = int(residual_count)
        input_size = 3 * self.residual_count
        self.normalization = nn.LayerNorm(input_size)
        self.temporal = nn.GRU(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        residuals: torch.Tensor,
        availability: torch.Tensor,
        confidence: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residuals.ndim != 3 or residuals.shape[-1] != self.residual_count:
            raise ValueError("residuals must have shape [B,T,R].")
        if availability.shape != residuals.shape or confidence.shape != residuals.shape:
            raise ValueError("availability and confidence must match residuals.")
        if padding_mask is None:
            padding_mask = availability.any(dim=-1)
        if padding_mask.shape != residuals.shape[:2]:
            raise ValueError("padding_mask must have shape [B,T].")
        mask = availability.to(dtype=residuals.dtype)
        values = torch.nan_to_num(residuals, nan=0.0) * mask
        quality = torch.clamp(confidence, 0.0, 1.0) * mask
        features = self.normalization(torch.cat((values, mask, quality), dim=-1))
        encoded, _ = self.temporal(features)
        valid_steps = padding_mask.to(dtype=torch.bool)
        step_indices = torch.arange(encoded.shape[1], device=encoded.device).expand_as(valid_steps)
        last_index = torch.where(valid_steps, step_indices, -1).max(dim=1).values.clamp(min=0)
        batch_index = torch.arange(encoded.shape[0], device=encoded.device)
        return self.classifier(encoded[batch_index, last_index]).squeeze(-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


__all__ = ["ResidualTemporalHead"]
