"""Trainable projection and prediction modules for enzyme MSA embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


DEFAULT_NUMERIC_FIELDS: tuple[str, ...] = (
    "kcat_1_per_s",
    "km_mM",
    "kcat_over_km_1_per_mM_s",
    "topt_C",
    "tm_C",
)


@dataclass(frozen=True)
class ModelShape:
    input_dim: int = 768
    d_model: int = 128
    output_dim: int = 1


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean over ``dim`` with zero output where the mask count is zero."""
    mask_f = mask.to(dtype=values.dtype)
    counts = mask_f.sum(dim=dim)
    safe_counts = counts.clamp_min(1.0)
    sums = (values * mask_f.unsqueeze(-1)).sum(dim=dim)
    means = sums / safe_counts.unsqueeze(-1)
    means = torch.where(counts.unsqueeze(-1) > 0, means, torch.zeros_like(means))
    return means, counts


class RowColumnProjector(nn.Module):
    """Project frozen MSA token embeddings from ``B x R x L x H`` to ``B x L x d``.

    Each token sees its original frozen embedding, a row summary, and a column summary.
    The fused token features are then mean-pooled over MSA rows per aligned column.
    """

    def __init__(self, input_dim: int = 768, d_model: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.fuse = nn.Sequential(
            nn.LayerNorm(input_dim * 3),
            nn.Linear(input_dim * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, token_embeddings: torch.Tensor, aa_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if token_embeddings.ndim != 4:
            raise ValueError("token_embeddings must have shape B x R x L x H")
        if aa_mask.shape != token_embeddings.shape[:3]:
            raise ValueError("aa_mask must have shape B x R x L matching token_embeddings")

        row_context, _ = masked_mean(token_embeddings, aa_mask, dim=2)  # B x R x H
        col_context, col_counts = masked_mean(token_embeddings, aa_mask, dim=1)  # B x L x H
        row_context = row_context.unsqueeze(2).expand_as(token_embeddings)
        col_context = col_context.unsqueeze(1).expand_as(token_embeddings)
        fused = self.fuse(torch.cat([token_embeddings, row_context, col_context], dim=-1))

        mask_f = aa_mask.to(dtype=fused.dtype)
        safe_counts = col_counts.clamp_min(1.0)
        projected = (fused * mask_f.unsqueeze(-1)).sum(dim=1) / safe_counts.unsqueeze(-1)
        projected = torch.where(col_counts.unsqueeze(-1) > 0, projected, torch.zeros_like(projected))
        column_mask = col_counts > 0
        return projected, column_mask


class NumericConditionTokenBank(nn.Module):
    """Per-property numeric embedding heads that produce appendable condition tokens."""

    def __init__(self, fields: Sequence[str] = DEFAULT_NUMERIC_FIELDS, d_model: int = 128, hidden_dim: int = 64) -> None:
        super().__init__()
        if not fields:
            raise ValueError("at least one condition field is required")
        self.fields = tuple(fields)
        self.d_model = d_model
        self.value_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, d_model),
                )
                for _ in self.fields
            ]
        )
        self.field_tokens = nn.Parameter(torch.empty(len(self.fields), d_model))
        self.missing_tokens = nn.Parameter(torch.empty(len(self.fields), d_model))
        self.output_norm = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.field_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.missing_tokens, mean=0.0, std=0.02)

    def forward(self, values: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        if values.shape != observed_mask.shape:
            raise ValueError("condition values and observed mask must have the same shape")
        if values.ndim != 2 or values.shape[1] != len(self.fields):
            raise ValueError(f"expected condition tensors with shape B x {len(self.fields)}")
        tokens: list[torch.Tensor] = []
        for idx, head in enumerate(self.value_heads):
            value_token = head(values[:, idx : idx + 1])
            missing_token = self.missing_tokens[idx].unsqueeze(0).expand_as(value_token)
            token = torch.where(observed_mask[:, idx : idx + 1], value_token, missing_token)
            token = token + self.field_tokens[idx]
            tokens.append(token)
        return self.output_norm(torch.stack(tokens, dim=1))


class EnzymeMSAPredictor(nn.Module):
    """Small predictor over frozen MSA embeddings plus numeric condition tokens."""

    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 128,
        condition_fields: Sequence[str] = DEFAULT_NUMERIC_FIELDS,
        output_dim: int = 1,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_positions: int = 2048,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.condition_fields = tuple(condition_fields)
        self.max_positions = max_positions
        self.projector = RowColumnProjector(input_dim=input_dim, d_model=d_model, dropout=dropout)
        self.condition_tokens = NumericConditionTokenBank(fields=self.condition_fields, d_model=d_model)
        self.position_embedding = nn.Embedding(max_positions, d_model)
        self.kind_embedding = nn.Embedding(2, d_model)  # 0 = MSA column token, 1 = numeric condition token
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        aa_mask: torch.Tensor,
        condition_values: torch.Tensor,
        condition_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        msa_tokens, column_mask = self.projector(token_embeddings, aa_mask)
        condition_tokens = self.condition_tokens(condition_values, condition_mask)
        sequence = torch.cat([msa_tokens, condition_tokens], dim=1)
        batch_size, total_length, _ = sequence.shape
        if total_length > self.max_positions:
            raise ValueError(f"combined token length {total_length} exceeds max_positions={self.max_positions}")

        positions = torch.arange(total_length, device=sequence.device).unsqueeze(0).expand(batch_size, -1)
        kind_ids = torch.cat(
            [
                torch.zeros((batch_size, msa_tokens.shape[1]), dtype=torch.long, device=sequence.device),
                torch.ones((batch_size, condition_tokens.shape[1]), dtype=torch.long, device=sequence.device),
            ],
            dim=1,
        )
        sequence = sequence + self.position_embedding(positions) + self.kind_embedding(kind_ids)

        condition_valid = torch.ones(
            (batch_size, condition_tokens.shape[1]), dtype=torch.bool, device=sequence.device
        )
        valid_mask = torch.cat([column_mask, condition_valid], dim=1)
        encoded = self.context_encoder(sequence, src_key_padding_mask=~valid_mask)
        encoded = self.final_norm(encoded)
        pooled, counts = masked_mean(encoded, valid_mask, dim=1)
        prediction = self.predictor(pooled)
        return {
            "prediction": prediction,
            "pooled": pooled,
            "projected_msa_tokens": msa_tokens,
            "condition_tokens": condition_tokens,
            "valid_token_counts": counts,
        }
