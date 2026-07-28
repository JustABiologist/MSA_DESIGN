"""Trainable projection and prediction modules for enzyme MSA embeddings."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_NUMERIC_FIELDS: tuple[str, ...] = (
    "kcat_1_per_s",
    "km_mM",
    "kcat_over_km_1_per_mM_s",
    "topt_C",
    "tm_C",
)
DEFAULT_CATEGORICAL_FIELDS: tuple[str, ...] = (
    "ec_numbers",
    "reaction_ids",
    "compound_ids",
)
STOP_TOKEN = "*"
MASK_TOKEN = "<MASK>"
SEQUENCE_TOKENS: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY") + (STOP_TOKEN, MASK_TOKEN)
TOKEN_TO_ID: dict[str, int] = {token: idx for idx, token in enumerate(SEQUENCE_TOKENS)}
ID_TO_TOKEN: dict[int, str] = {idx: token for token, idx in TOKEN_TO_ID.items()}
STOP_TOKEN_ID = TOKEN_TO_ID[STOP_TOKEN]
MASK_TOKEN_ID = TOKEN_TO_ID[MASK_TOKEN]
DEFAULT_MAX_SEQUENCE_LENGTH = 1280


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


def encode_sequence_with_stop(
    sequence: str,
    max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    tail_stop_weight: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one amino-acid sequence as fixed-length tokens padded with ``*``.

    Residues and the first ``*`` get full loss weight. Repeated trailing ``*``
    tokens get a small weight so padding does not dominate token training.
    """
    if max_length < 1:
        raise ValueError("max_length must be at least 1")
    if not 0.0 <= tail_stop_weight <= 1.0:
        raise ValueError("tail_stop_weight must be between 0 and 1")

    cleaned = "".join(sequence.strip().upper().split())
    if STOP_TOKEN in cleaned:
        cleaned = cleaned.split(STOP_TOKEN, 1)[0]
    cleaned = cleaned.replace("-", "")
    cleaned = cleaned.replace(".", "")
    if len(cleaned) >= max_length:
        cleaned = cleaned[: max_length - 1]

    unknown = sorted(set(cleaned) - set(TOKEN_TO_ID))
    if unknown:
        raise ValueError(f"sequence contains unknown token(s): {', '.join(unknown)}")

    token_ids = [TOKEN_TO_ID[token] for token in cleaned]
    token_ids.append(STOP_TOKEN_ID)
    trailing = max_length - len(token_ids)
    token_ids.extend([STOP_TOKEN_ID] * trailing)

    weights = [1.0] * (len(cleaned) + 1)
    weights.extend([float(tail_stop_weight)] * trailing)
    return torch.tensor(token_ids, dtype=torch.long), torch.tensor(weights, dtype=torch.float32)


def batch_encode_sequences_with_stop(
    sequences: Sequence[str],
    max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    tail_stop_weight: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("at least one sequence is required")
    encoded = [
        encode_sequence_with_stop(
            sequence,
            max_length=max_length,
            tail_stop_weight=tail_stop_weight,
        )
        for sequence in sequences
    ]
    tokens, weights = zip(*encoded)
    return torch.stack(list(tokens), dim=0), torch.stack(list(weights), dim=0)


def decode_tokens_until_stop(token_ids: Sequence[int]) -> str:
    residues: list[str] = []
    for token_id in token_ids:
        token = ID_TO_TOKEN[int(token_id)]
        if token == STOP_TOKEN:
            break
        if token == MASK_TOKEN:
            continue
        residues.append(token)
    return "".join(residues)


def weighted_token_cross_entropy(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape B x L x vocab_size")
    if target_tokens.shape != logits.shape[:2]:
        raise ValueError("target_tokens must have shape B x L matching logits")
    if loss_weights.shape != target_tokens.shape:
        raise ValueError("loss_weights must have shape B x L matching target_tokens")

    flat_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_tokens.reshape(-1),
        reduction="none",
    )
    token_loss = flat_loss.reshape_as(target_tokens)
    weights = loss_weights.to(dtype=token_loss.dtype)
    return (token_loss * weights).sum() / weights.sum().clamp_min(1.0)


def weighted_token_accuracy(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape B x L x vocab_size")
    if target_tokens.shape != logits.shape[:2]:
        raise ValueError("target_tokens must have shape B x L matching logits")
    if loss_weights.shape != target_tokens.shape:
        raise ValueError("loss_weights must have shape B x L matching target_tokens")
    predicted = torch.argmax(logits, dim=-1)
    correct = (predicted == target_tokens).to(dtype=logits.dtype)
    weights = loss_weights.to(dtype=logits.dtype)
    return (correct * weights).sum() / weights.sum().clamp_min(1.0)


def weighted_position_mse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    loss_weights: torch.Tensor,
) -> torch.Tensor:
    if predicted.shape != target.shape:
        raise ValueError("predicted and target tensors must have matching shapes")
    if predicted.ndim != 3:
        raise ValueError("predicted and target must have shape B x L x d_model")
    if loss_weights.shape != predicted.shape[:2]:
        raise ValueError("loss_weights must have shape B x L matching predicted")

    per_position = F.mse_loss(predicted, target, reduction="none").mean(dim=-1)
    weights = loss_weights.to(dtype=per_position.dtype)
    return (per_position * weights).sum() / weights.sum().clamp_min(1.0)


def sequence_latent_targets(
    token_embeddings: torch.Tensor,
    loss_weights: torch.Tensor,
    num_latent_tokens: int,
) -> torch.Tensor:
    if token_embeddings.ndim != 3:
        raise ValueError("token_embeddings must have shape B x L x d_model")
    if loss_weights.shape != token_embeddings.shape[:2]:
        raise ValueError("loss_weights must have shape B x L matching token_embeddings")
    if num_latent_tokens < 1:
        raise ValueError("num_latent_tokens must be at least 1")

    batch_size, sequence_length, _ = token_embeddings.shape
    weights = loss_weights.to(dtype=token_embeddings.dtype)
    global_counts = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    global_mean = (token_embeddings * weights.unsqueeze(-1)).sum(dim=1) / global_counts
    targets: list[torch.Tensor] = []
    for latent_index in range(num_latent_tokens):
        start = math.floor(latent_index * sequence_length / num_latent_tokens)
        stop = math.floor((latent_index + 1) * sequence_length / num_latent_tokens)
        stop = max(stop, start + 1)
        segment_embeddings = token_embeddings[:, start:stop]
        segment_weights = weights[:, start:stop]
        segment_counts = segment_weights.sum(dim=1, keepdim=True)
        segment_mean = (
            segment_embeddings * segment_weights.unsqueeze(-1)
        ).sum(dim=1) / segment_counts.clamp_min(1.0)
        segment_mean = torch.where(segment_counts > 0, segment_mean, global_mean)
        targets.append(segment_mean)
    return torch.stack(targets, dim=1)


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


class MSADepthScaler(nn.Module):
    """Compress MSA depth from ``B x R x L x H`` to ``B x L x d``.

    The output keeps the aligned-column length ``L`` while a learned attention
    distribution compresses the row/depth axis. Invalid all-gap columns are
    returned as zero vectors and marked false in the column mask.
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
        self.depth_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        self.output_norm = nn.LayerNorm(d_model)

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

        scores = self.depth_score(fused).squeeze(-1)
        scores = scores.masked_fill(~aa_mask, -1.0e4)
        depth_weights = torch.softmax(scores, dim=1) * aa_mask.to(dtype=fused.dtype)
        depth_weights = depth_weights / depth_weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)

        scaled = (fused * depth_weights.unsqueeze(-1)).sum(dim=1)
        column_mask = col_counts > 0
        scaled = torch.where(column_mask.unsqueeze(-1), scaled, torch.zeros_like(scaled))
        return self.output_norm(scaled), column_mask


class AxialMSAEncoder(nn.Module):
    """Encode a target-masked MSA tensor with explicit row and column attention.

    Input token embeddings stay in aligned MSA space as ``B x R x L x H``. The
    encoder first projects each cell, then alternates attention across columns
    within each row and across rows within each column. It returns both learned
    column summaries and learned row summaries so downstream decoders can
    cross-attend to both axes instead of seeing only depth-compressed columns.
    """

    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.input_dim = input_dim
        self.d_model = d_model
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.column_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.row_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=num_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.column_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        self.row_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        self.output_norm = nn.LayerNorm(d_model)

    @staticmethod
    def _safe_key_padding_mask(mask: torch.Tensor, dim: int) -> torch.Tensor:
        """Return a key-padding mask without all-masked rows.

        PyTorch attention can produce NaNs if every key in a sequence is masked.
        Invalid rows/columns are still zeroed after the layer, so allowing the
        all-padding sequence to attend to zeros is harmless and keeps the forward
        pass numerically defined.
        """

        key_padding = ~mask
        all_padded = key_padding.all(dim=dim, keepdim=True)
        return torch.where(all_padded, torch.zeros_like(key_padding), key_padding)

    def forward(
        self,
        token_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_embeddings.ndim != 4:
            raise ValueError("token_embeddings must have shape B x R x L x H")
        if valid_mask.shape != token_embeddings.shape[:3]:
            raise ValueError("valid_mask must have shape B x R x L matching token_embeddings")

        batch_size, row_count, col_count, _ = token_embeddings.shape
        x = self.input_proj(token_embeddings)
        mask_f = valid_mask.to(dtype=x.dtype).unsqueeze(-1)
        x = x * mask_f

        for column_layer, row_layer in zip(self.column_layers, self.row_layers):
            row_view = x.reshape(batch_size * row_count, col_count, self.d_model)
            row_mask = valid_mask.reshape(batch_size * row_count, col_count)
            row_padding = self._safe_key_padding_mask(row_mask, dim=1)
            row_view = column_layer(row_view, src_key_padding_mask=row_padding)
            x = row_view.reshape(batch_size, row_count, col_count, self.d_model) * mask_f

            col_view = x.transpose(1, 2).reshape(batch_size * col_count, row_count, self.d_model)
            col_mask = valid_mask.transpose(1, 2).reshape(batch_size * col_count, row_count)
            col_padding = self._safe_key_padding_mask(col_mask, dim=1)
            col_view = row_layer(col_view, src_key_padding_mask=col_padding)
            x = col_view.reshape(batch_size, col_count, row_count, self.d_model).transpose(1, 2) * mask_f

        column_counts = valid_mask.to(dtype=x.dtype).sum(dim=1)
        row_counts = valid_mask.to(dtype=x.dtype).sum(dim=2)

        column_scores = self.column_score(x).squeeze(-1).masked_fill(~valid_mask, -1.0e4)
        column_weights = torch.softmax(column_scores, dim=1) * valid_mask.to(dtype=x.dtype)
        column_weights = column_weights / column_weights.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        column_tokens = (x * column_weights.unsqueeze(-1)).sum(dim=1)
        column_mask = column_counts > 0
        column_tokens = torch.where(column_mask.unsqueeze(-1), column_tokens, torch.zeros_like(column_tokens))

        row_scores = self.row_score(x).squeeze(-1).masked_fill(~valid_mask, -1.0e4)
        row_weights = torch.softmax(row_scores, dim=2) * valid_mask.to(dtype=x.dtype)
        row_weights = row_weights / row_weights.sum(dim=2, keepdim=True).clamp_min(1.0e-6)
        row_tokens = (x * row_weights.unsqueeze(-1)).sum(dim=2)
        row_mask = row_counts > 0
        row_tokens = torch.where(row_mask.unsqueeze(-1), row_tokens, torch.zeros_like(row_tokens))

        return (
            self.output_norm(column_tokens),
            column_mask,
            self.output_norm(row_tokens),
            row_mask,
        )


class SequenceMSAAxialDecoderLayer(nn.Module):
    """Refine sequence latents with axial cross-attention into a static MSA grid."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        msa_axial_blocks: int = 1,
    ) -> None:
        super().__init__()
        if msa_axial_blocks < 1:
            raise ValueError("msa_axial_blocks must be at least 1")
        self.sequence_self_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.static_memory_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.column_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.row_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.row_fusion_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_norm = nn.LayerNorm(d_model)
        self.static_query_norm = nn.LayerNorm(d_model)
        self.static_key_norm = nn.LayerNorm(d_model)
        self.column_query_norm = nn.LayerNorm(d_model)
        self.column_key_norm = nn.LayerNorm(d_model)
        self.row_query_norm = nn.LayerNorm(d_model)
        self.row_key_norm = nn.LayerNorm(d_model)
        self.row_fusion_query_norm = nn.LayerNorm(d_model)
        self.row_fusion_key_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.output_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.msa_axial_blocks = int(msa_axial_blocks)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _safe_key_padding_mask(mask: torch.Tensor) -> torch.Tensor:
        key_padding = ~mask
        all_padded = key_padding.all(dim=1, keepdim=True)
        return torch.where(all_padded, torch.zeros_like(key_padding), key_padding)

    @staticmethod
    def _mask_attention_output(output: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return output * valid.to(dtype=output.dtype).view(output.shape[0], 1, 1)

    def _column_read_single_batch(
        self,
        sequence: torch.Tensor,
        msa_grid: torch.Tensor,
        msa_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, d_model = sequence.shape
        _, row_count, col_count, _ = msa_grid.shape
        active_cols = min(sequence_length, col_count)
        if active_cols < 1:
            return torch.zeros_like(sequence)

        column_queries = self.column_query_norm(sequence[:, :active_cols]).reshape(
            batch_size * active_cols,
            1,
            d_model,
        )
        column_cells = msa_grid[:, :, :active_cols].transpose(1, 2).reshape(
            batch_size * active_cols,
            row_count,
            d_model,
        )
        column_cell_mask = msa_mask[:, :, :active_cols].transpose(1, 2).reshape(
            batch_size * active_cols,
            row_count,
        )
        column_keys = self.column_key_norm(column_cells)
        column_update, _ = self.column_attn(
            column_queries,
            column_keys,
            column_keys,
            key_padding_mask=self._safe_key_padding_mask(column_cell_mask),
            need_weights=False,
        )
        column_update = self._mask_attention_output(column_update, column_cell_mask.any(dim=1))
        column_update = column_update.reshape(batch_size, active_cols, d_model)
        full_update = torch.zeros_like(sequence)
        full_update[:, :active_cols] = column_update
        return full_update

    def _column_read(
        self,
        sequence: torch.Tensor,
        msa_grid: torch.Tensor,
        msa_mask: torch.Tensor,
        target_group_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_group_indices is None:
            return self._column_read_single_batch(sequence, msa_grid, msa_mask)
        if target_group_indices.shape != (sequence.shape[0],):
            raise ValueError("target_group_indices must have shape B matching sequence")
        updates = []
        for target_index, group_index in enumerate(target_group_indices.detach().cpu().tolist()):
            updates.append(
                self._column_read_single_batch(
                    sequence[target_index : target_index + 1],
                    msa_grid[group_index : group_index + 1],
                    msa_mask[group_index : group_index + 1],
                )
            )
        return torch.cat(updates, dim=0)

    def _row_read_single_batch(
        self,
        sequence: torch.Tensor,
        msa_grid: torch.Tensor,
        msa_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, d_model = sequence.shape
        _, row_count, col_count, _ = msa_grid.shape
        if row_count < 1 or col_count < 1:
            return torch.zeros_like(sequence)

        row_queries = self.row_query_norm(sequence).unsqueeze(1).expand(
            batch_size,
            row_count,
            sequence_length,
            d_model,
        )
        row_queries = row_queries.reshape(batch_size * row_count, sequence_length, d_model)
        row_cells = msa_grid.reshape(batch_size * row_count, col_count, d_model)
        row_cell_mask = msa_mask.reshape(batch_size * row_count, col_count)
        row_keys = self.row_key_norm(row_cells)
        row_updates, _ = self.row_attn(
            row_queries,
            row_keys,
            row_keys,
            key_padding_mask=self._safe_key_padding_mask(row_cell_mask),
            need_weights=False,
        )
        row_valid = row_cell_mask.any(dim=1)
        row_updates = row_updates * row_valid.to(dtype=row_updates.dtype).view(-1, 1, 1)
        row_updates = row_updates.reshape(batch_size, row_count, sequence_length, d_model).transpose(1, 2)

        row_candidates = row_updates.reshape(batch_size * sequence_length, row_count, d_model)
        row_candidate_mask = row_valid.reshape(batch_size, row_count).unsqueeze(1).expand(
            batch_size,
            sequence_length,
            row_count,
        )
        row_candidate_mask = row_candidate_mask.reshape(batch_size * sequence_length, row_count)
        row_fusion_query = self.row_fusion_query_norm(sequence).reshape(batch_size * sequence_length, 1, d_model)
        row_fusion_keys = self.row_fusion_key_norm(row_candidates)
        row_update, _ = self.row_fusion_attn(
            row_fusion_query,
            row_fusion_keys,
            row_fusion_keys,
            key_padding_mask=self._safe_key_padding_mask(row_candidate_mask),
            need_weights=False,
        )
        row_update = self._mask_attention_output(row_update, row_candidate_mask.any(dim=1))
        return row_update.reshape(batch_size, sequence_length, d_model)

    def _row_read(
        self,
        sequence: torch.Tensor,
        msa_grid: torch.Tensor,
        msa_mask: torch.Tensor,
        target_group_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if target_group_indices is None:
            return self._row_read_single_batch(sequence, msa_grid, msa_mask)
        if target_group_indices.shape != (sequence.shape[0],):
            raise ValueError("target_group_indices must have shape B matching sequence")
        updates = []
        for target_index, group_index in enumerate(target_group_indices.detach().cpu().tolist()):
            updates.append(
                self._row_read_single_batch(
                    sequence[target_index : target_index + 1],
                    msa_grid[group_index : group_index + 1],
                    msa_mask[group_index : group_index + 1],
                )
            )
        return torch.cat(updates, dim=0)

    def forward(
        self,
        sequence: torch.Tensor,
        static_memory: torch.Tensor,
        static_memory_mask: torch.Tensor,
        msa_grid: torch.Tensor,
        msa_mask: torch.Tensor,
        target_group_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_group_indices is not None:
            if target_group_indices.ndim != 1 or target_group_indices.shape[0] != sequence.shape[0]:
                raise ValueError("target_group_indices must have shape B matching sequence")
            target_group_indices = target_group_indices.to(device=sequence.device, dtype=torch.long)
            if bool((target_group_indices < 0).any()) or bool((target_group_indices >= msa_grid.shape[0]).any()):
                raise ValueError("target_group_indices contains an out-of-range MSA grid index")
            if static_memory.shape[0] == sequence.shape[0]:
                target_static_memory = static_memory
                target_static_memory_mask = static_memory_mask
            elif static_memory.shape[0] == msa_grid.shape[0]:
                target_static_memory = static_memory.index_select(0, target_group_indices)
                target_static_memory_mask = static_memory_mask.index_select(0, target_group_indices)
            else:
                raise ValueError("static_memory batch must match sequence batch or shared MSA-grid batch")
        else:
            target_static_memory = static_memory
            target_static_memory_mask = static_memory_mask

        self_query = self.self_norm(sequence)
        self_update, _ = self.sequence_self_attn(
            self_query,
            self_query,
            self_query,
            need_weights=False,
        )
        sequence = sequence + self.dropout(self_update)

        static_keys = self.static_key_norm(target_static_memory)
        static_update, _ = self.static_memory_attn(
            self.static_query_norm(sequence),
            static_keys,
            static_keys,
            key_padding_mask=self._safe_key_padding_mask(target_static_memory_mask),
            need_weights=False,
        )
        static_valid = target_static_memory_mask.any(dim=1)
        sequence = sequence + self.dropout(self._mask_attention_output(static_update, static_valid))

        for _ in range(self.msa_axial_blocks):
            sequence = sequence + self.dropout(
                self._column_read(sequence, msa_grid, msa_mask, target_group_indices=target_group_indices)
            )
            sequence = sequence + self.dropout(
                self._row_read(sequence, msa_grid, msa_mask, target_group_indices=target_group_indices)
            )

        sequence = sequence + self.dropout(self.output_ffn(self.ffn_norm(sequence)))
        return sequence, msa_grid


class SequenceDiffusionDecoder(nn.Module):
    """Fixed-length conditional diffusion decoder for protein sequences."""

    def __init__(
        self,
        d_model: int = 128,
        vocab_size: int = len(SEQUENCE_TOKENS),
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_timesteps: int = 1000,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
        mask_token_id: int = MASK_TOKEN_ID,
        msa_grid_decoder: bool = False,
        msa_axial_blocks_per_layer: int = 1,
    ) -> None:
        super().__init__()
        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be at least 1")
        if num_timesteps < 2:
            raise ValueError("num_timesteps must be at least 2")
        if not 0 <= mask_token_id < vocab_size:
            raise ValueError("mask_token_id must be inside the decoder vocabulary")
        if msa_axial_blocks_per_layer < 1:
            raise ValueError("msa_axial_blocks_per_layer must be at least 1")
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.num_timesteps = num_timesteps
        self.mask_token_id = int(mask_token_id)
        self.msa_grid_decoder = bool(msa_grid_decoder)

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_sequence_length, d_model)
        self.time_embedding = nn.Sequential(
            nn.Embedding(num_timesteps, d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        if self.msa_grid_decoder:
            self.denoiser = None
            self.msa_grid_layers = nn.ModuleList(
                [
                    SequenceMSAAxialDecoderLayer(
                        d_model=d_model,
                        num_heads=num_heads,
                        dropout=dropout,
                        msa_axial_blocks=msa_axial_blocks_per_layer,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.denoiser = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
            self.msa_grid_layers = nn.ModuleList()
        self.output_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.continuous_input = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.continuous_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod), persistent=False)
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod), persistent=False)

    def _position_ids(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.arange(self.max_sequence_length, device=device).unsqueeze(0).expand(batch_size, -1)

    def q_sample(
        self,
        clean_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean_embeddings)
        scale_clean = self.sqrt_alpha_cumprod[timesteps].view(-1, 1, 1)
        scale_noise = self.sqrt_one_minus_alpha_cumprod[timesteps].view(-1, 1, 1)
        return scale_clean * clean_embeddings + scale_noise * noise

    def scaled_timesteps(self, timesteps: torch.Tensor, scale: float) -> torch.Tensor:
        if scale <= 0.0:
            raise ValueError("continuous timestep scale must be positive")
        return torch.round(timesteps.to(dtype=torch.float32) * scale).to(dtype=torch.long).clamp(
            min=0,
            max=self.num_timesteps - 1,
        )

    def discrete_corruption_probability(self, timesteps: torch.Tensor) -> torch.Tensor:
        return ((timesteps.to(dtype=torch.float32) + 1.0) / self.num_timesteps).clamp(0.0, 1.0)

    def discrete_corrupt_tokens(
        self,
        target_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        mode: str = "discrete_mask",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if target_tokens.ndim != 2:
            raise ValueError("target_tokens must have shape B x L")
        if timesteps.shape != (target_tokens.shape[0],):
            raise ValueError("timesteps must have shape B")
        if target_tokens.shape[1] != self.max_sequence_length:
            raise ValueError(
                f"target_tokens length {target_tokens.shape[1]} does not match "
                f"max_sequence_length={self.max_sequence_length}"
            )
        if mode not in {"discrete_mask", "discrete_random"}:
            raise ValueError(f"unknown discrete corruption mode: {mode}")

        probability = self.discrete_corruption_probability(timesteps).to(device=target_tokens.device)
        corruption_mask = (
            torch.rand(target_tokens.shape, dtype=torch.float32, device=target_tokens.device)
            < probability.view(-1, 1)
        )
        if mode == "discrete_mask":
            replacements = torch.full_like(target_tokens, self.mask_token_id)
        else:
            random_ids = torch.randint(
                self.vocab_size - 1,
                target_tokens.shape,
                dtype=torch.long,
                device=target_tokens.device,
            )
            replacements = torch.where(
                random_ids == target_tokens,
                (random_ids + 1) % (self.vocab_size - 1),
                random_ids,
            )
        corrupted = torch.where(corruption_mask, replacements, target_tokens)
        return corrupted, corruption_mask

    def mask_decoder_input_embeddings(
        self,
        clean_embeddings: torch.Tensor,
        token_dropout: float = 0.0,
        span_mask_fraction: float = 0.0,
        span_mask_length: int = 16,
    ) -> torch.Tensor:
        if token_dropout <= 0.0 and span_mask_fraction <= 0.0:
            return clean_embeddings
        if not 0.0 <= token_dropout <= 1.0:
            raise ValueError("token_dropout must be in [0, 1]")
        if not 0.0 <= span_mask_fraction <= 1.0:
            raise ValueError("span_mask_fraction must be in [0, 1]")
        if span_mask_length < 1:
            raise ValueError("span_mask_length must be at least 1")

        batch_size, sequence_length, _ = clean_embeddings.shape
        mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=clean_embeddings.device)
        if token_dropout > 0.0:
            mask |= torch.rand(mask.shape, dtype=torch.float32, device=clean_embeddings.device) < token_dropout
        if span_mask_fraction > 0.0:
            masked_positions = max(1, int(round(sequence_length * span_mask_fraction)))
            span_count = max(1, math.ceil(masked_positions / span_mask_length))
            max_start = max(sequence_length - span_mask_length + 1, 1)
            for batch_index in range(batch_size):
                starts = torch.randint(max_start, (span_count,), device=clean_embeddings.device)
                for start in starts.tolist():
                    stop = min(start + span_mask_length, sequence_length)
                    mask[batch_index, start:stop] = True

        mean_embedding = self.token_embedding.weight.mean(dim=0).view(1, 1, -1)
        return torch.where(mask.unsqueeze(-1), mean_embedding, clean_embeddings)

    def decoder_start_state(
        self,
        clean_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        mode: str = "mean",
        target_tokens: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        token_dropout: float = 0.0,
        span_mask_fraction: float = 0.0,
        span_mask_length: int = 16,
    ) -> dict[str, torch.Tensor]:
        if mode == "q_sample":
            start_embeddings = self.mask_decoder_input_embeddings(
                clean_embeddings=clean_embeddings,
                token_dropout=token_dropout,
                span_mask_fraction=span_mask_fraction,
                span_mask_length=span_mask_length,
            )
            return {"noisy_embeddings": self.q_sample(start_embeddings, timesteps, noise=noise)}
        if mode == "pure_noise":
            if noise is None:
                noise = torch.randn_like(clean_embeddings)
            return {"noisy_embeddings": noise}
        if mode == "mean":
            mean_embedding = self.token_embedding.weight.mean(dim=0).view(1, 1, -1)
            return {"noisy_embeddings": mean_embedding.expand_as(clean_embeddings)}
        if mode == "noisy_mean":
            mean_embedding = self.token_embedding.weight.mean(dim=0).view(1, 1, -1)
            start_embeddings = mean_embedding.expand_as(clean_embeddings)
            return {"noisy_embeddings": self.q_sample(start_embeddings, timesteps, noise=noise)}
        if mode in {"discrete_mask", "discrete_random"}:
            if target_tokens is None:
                raise ValueError(f"{mode} requires target_tokens")
            noisy_tokens, corruption_mask = self.discrete_corrupt_tokens(
                target_tokens=target_tokens,
                timesteps=timesteps,
                mode=mode,
            )
            return {
                "noisy_embeddings": self.token_embedding(noisy_tokens),
                "noisy_tokens": noisy_tokens,
                "corruption_mask": corruption_mask,
            }
        raise ValueError(f"unknown decoder start mode: {mode}")

    def decoder_start_embeddings(
        self,
        clean_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        mode: str = "mean",
        target_tokens: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        token_dropout: float = 0.0,
        span_mask_fraction: float = 0.0,
        span_mask_length: int = 16,
    ) -> torch.Tensor:
        return self.decoder_start_state(
            clean_embeddings=clean_embeddings,
            timesteps=timesteps,
            mode=mode,
            target_tokens=target_tokens,
            noise=noise,
            token_dropout=token_dropout,
            span_mask_fraction=span_mask_fraction,
            span_mask_length=span_mask_length,
        )["noisy_embeddings"]

    def denoise(
        self,
        noisy_embeddings: torch.Tensor,
        timesteps: torch.Tensor,
        latent_tokens: torch.Tensor,
        latent_mask: torch.Tensor,
        continuous_embeddings: torch.Tensor | None = None,
        msa_grid_tokens: torch.Tensor | None = None,
        msa_grid_mask: torch.Tensor | None = None,
        target_group_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noisy_embeddings.shape[1] != self.max_sequence_length:
            raise ValueError(
                f"noisy_embeddings length {noisy_embeddings.shape[1]} does not match "
                f"max_sequence_length={self.max_sequence_length}"
            )
        if latent_tokens.ndim != 3:
            raise ValueError("latent_tokens must have shape B x L x d_model")
        if latent_mask.shape != latent_tokens.shape[:2]:
            raise ValueError("latent_mask must have shape B x L matching latent_tokens")
        if not torch.all(latent_mask.any(dim=1)):
            raise ValueError("each sample must have at least one valid MSA latent token")
        batch_size = noisy_embeddings.shape[0]
        if timesteps.shape != (batch_size,):
            raise ValueError("timesteps must have shape B")
        if target_group_indices is not None:
            if target_group_indices.shape != (batch_size,):
                raise ValueError("target_group_indices must have shape B")
            target_group_indices = target_group_indices.to(device=noisy_embeddings.device, dtype=torch.long)
            if bool((target_group_indices < 0).any()):
                raise ValueError("target_group_indices contains a negative index")
            if self.msa_grid_decoder:
                if msa_grid_tokens is None:
                    raise ValueError("msa_grid_decoder requires msa_grid_tokens")
                if bool((target_group_indices >= msa_grid_tokens.shape[0]).any()):
                    raise ValueError("target_group_indices contains an out-of-range MSA grid index")
                if latent_tokens.shape[0] not in {batch_size, msa_grid_tokens.shape[0]}:
                    raise ValueError("latent_tokens batch must match sequence batch or shared MSA-grid batch")
            elif bool((target_group_indices >= latent_tokens.shape[0]).any()):
                raise ValueError("target_group_indices contains an out-of-range latent-memory index")
        elif latent_tokens.shape[0] != batch_size:
            raise ValueError("latent_tokens batch must match sequence batch when target_group_indices is absent")

        position_ids = self._position_ids(batch_size, noisy_embeddings.device)
        sequence = noisy_embeddings
        if continuous_embeddings is not None:
            if continuous_embeddings.shape != noisy_embeddings.shape:
                raise ValueError("continuous_embeddings must match noisy_embeddings shape")
            sequence = sequence + self.continuous_input(continuous_embeddings)
        sequence = sequence + self.position_embedding(position_ids)
        sequence = sequence + self.time_embedding(timesteps).unsqueeze(1)
        if self.msa_grid_decoder:
            if msa_grid_tokens is None or msa_grid_mask is None:
                raise ValueError("msa_grid_decoder requires msa_grid_tokens and msa_grid_mask")
            if msa_grid_tokens.ndim != 4:
                raise ValueError("msa_grid_tokens must have shape B x R x C x d_model")
            if msa_grid_tokens.shape[-1] != self.d_model:
                raise ValueError("msa_grid_tokens d_model dimension must match decoder")
            if target_group_indices is None and msa_grid_tokens.shape[0] != batch_size:
                raise ValueError("msa_grid_tokens batch must match decoder batch without target_group_indices")
            if msa_grid_mask.shape != msa_grid_tokens.shape[:3]:
                raise ValueError("msa_grid_mask must have shape B x R x C matching msa_grid_tokens")
            decoded = sequence
            grid = msa_grid_tokens
            for layer in self.msa_grid_layers:
                decoded, grid = layer(
                    sequence=decoded,
                    static_memory=latent_tokens,
                    static_memory_mask=latent_mask,
                    msa_grid=grid,
                    msa_mask=msa_grid_mask,
                    target_group_indices=target_group_indices,
                )
        else:
            if self.denoiser is None:
                raise RuntimeError("missing Transformer denoiser")
            decoder_latent_tokens = latent_tokens
            decoder_latent_mask = latent_mask
            if target_group_indices is not None:
                decoder_latent_tokens = latent_tokens.index_select(0, target_group_indices)
                decoder_latent_mask = latent_mask.index_select(0, target_group_indices)
            decoded = self.denoiser(
                tgt=sequence,
                memory=decoder_latent_tokens,
                memory_key_padding_mask=~decoder_latent_mask,
            )
        return self.output_norm(decoded)

    def forward(
        self,
        latent_tokens: torch.Tensor,
        latent_mask: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
        loss_weights: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        noisy_embeddings: torch.Tensor | None = None,
        decoder_start_mode: str = "mean",
        decoder_token_dropout: float = 0.0,
        decoder_span_mask_fraction: float = 0.0,
        decoder_span_mask_length: int = 16,
        discrete_loss_corrupted_only: bool = True,
        ccdd_continuous_targets: torch.Tensor | None = None,
        ccdd_continuous_mask: torch.Tensor | None = None,
        ccdd_continuous_noise: torch.Tensor | None = None,
        ccdd_continuous_timestep_scale: float = 0.75,
        ccdd_continuous_dropout: float = 0.0,
        msa_grid_tokens: torch.Tensor | None = None,
        msa_grid_mask: torch.Tensor | None = None,
        target_group_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = latent_tokens.shape[0]
        device = latent_tokens.device
        if timesteps is None:
            timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        if target_tokens is not None:
            if target_tokens.shape != (batch_size, self.max_sequence_length):
                raise ValueError(
                    f"target_tokens must have shape B x {self.max_sequence_length}; "
                    f"got {tuple(target_tokens.shape)}"
            )
            clean_embeddings = self.token_embedding(target_tokens)
            if noisy_embeddings is None:
                start_state = self.decoder_start_state(
                    clean_embeddings=clean_embeddings,
                    timesteps=timesteps,
                    mode=decoder_start_mode,
                    target_tokens=target_tokens,
                    noise=noise,
                    token_dropout=decoder_token_dropout,
                    span_mask_fraction=decoder_span_mask_fraction,
                    span_mask_length=decoder_span_mask_length,
                )
                noisy_embeddings = start_state["noisy_embeddings"]
            else:
                start_state = {}
        elif noisy_embeddings is None:
            raise ValueError("provide target_tokens for training or noisy_embeddings for denoising")
        else:
            clean_embeddings = None
            start_state = {}

        ccdd_outputs: dict[str, torch.Tensor] = {}
        continuous_embeddings = None
        if ccdd_continuous_targets is not None:
            if clean_embeddings is None or target_tokens is None:
                raise ValueError("CCDD continuous targets require target_tokens")
            if ccdd_continuous_targets.shape != clean_embeddings.shape:
                raise ValueError("ccdd_continuous_targets must have shape B x L x d_model")
            if not 0.0 <= ccdd_continuous_dropout <= 1.0:
                raise ValueError("ccdd_continuous_dropout must be in [0, 1]")
            corruption_mask = start_state.get("corruption_mask")
            if corruption_mask is None:
                raise ValueError("CCDD continuous stream requires a discrete corruption start mode")
            if ccdd_continuous_mask is None:
                ccdd_continuous_mask = torch.ones(
                    target_tokens.shape,
                    dtype=torch.bool,
                    device=target_tokens.device,
                )
            if ccdd_continuous_mask.shape != target_tokens.shape:
                raise ValueError("ccdd_continuous_mask must have shape B x L")

            zero_continuous = torch.zeros_like(ccdd_continuous_targets)
            leakage_safe_targets = torch.where(
                corruption_mask.unsqueeze(-1),
                zero_continuous,
                ccdd_continuous_targets,
            )
            continuous_timesteps = self.scaled_timesteps(timesteps, ccdd_continuous_timestep_scale)
            continuous_embeddings = self.q_sample(
                leakage_safe_targets,
                continuous_timesteps,
                noise=ccdd_continuous_noise,
            )
            if self.training and ccdd_continuous_dropout > 0.0:
                continuous_drop_mask = (
                    torch.rand(batch_size, dtype=torch.float32, device=device) < ccdd_continuous_dropout
                )
                continuous_embeddings = torch.where(
                    continuous_drop_mask.view(-1, 1, 1),
                    torch.zeros_like(continuous_embeddings),
                    continuous_embeddings,
                )
            else:
                continuous_drop_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            ccdd_outputs.update(
                {
                    "ccdd_clean_continuous_targets": ccdd_continuous_targets,
                    "ccdd_leakage_safe_continuous_targets": leakage_safe_targets,
                    "ccdd_noisy_continuous": continuous_embeddings,
                    "ccdd_continuous_timesteps": continuous_timesteps,
                    "ccdd_continuous_mask": ccdd_continuous_mask,
                    "ccdd_continuous_drop_mask": continuous_drop_mask,
                }
            )

        predicted_embeddings = self.denoise(
            noisy_embeddings,
            timesteps,
            latent_tokens,
            latent_mask,
            continuous_embeddings=continuous_embeddings,
            msa_grid_tokens=msa_grid_tokens,
            msa_grid_mask=msa_grid_mask,
            target_group_indices=target_group_indices,
        )
        logits = self.lm_head(predicted_embeddings)
        outputs = {
            "logits": logits,
            "predicted_embeddings": predicted_embeddings,
            "timesteps": timesteps,
            "noisy_embeddings": noisy_embeddings,
        }
        outputs.update(start_state)
        outputs.update(ccdd_outputs)
        if clean_embeddings is not None:
            mse = F.mse_loss(predicted_embeddings, clean_embeddings, reduction="none").mean(dim=(1, 2))
            outputs["diffusion_loss"] = mse.mean()
            if loss_weights is None:
                loss_weights = torch.ones_like(target_tokens, dtype=predicted_embeddings.dtype)
            token_loss_weights = loss_weights
            corruption_mask = start_state.get("corruption_mask")
            if discrete_loss_corrupted_only and corruption_mask is not None:
                token_loss_weights = loss_weights * corruption_mask.to(dtype=loss_weights.dtype)
                if torch.sum(token_loss_weights) <= 0:
                    token_loss_weights = loss_weights
            outputs["token_loss_weights"] = token_loss_weights
            outputs["token_loss"] = weighted_token_cross_entropy(logits, target_tokens, token_loss_weights)
            outputs["token_accuracy"] = weighted_token_accuracy(logits, target_tokens, token_loss_weights)
            outputs["full_token_accuracy"] = weighted_token_accuracy(logits, target_tokens, loss_weights)
            if corruption_mask is not None:
                outputs["corruption_fraction"] = corruption_mask.to(dtype=predicted_embeddings.dtype).mean()
                outputs["corrupted_token_accuracy"] = weighted_token_accuracy(
                    logits,
                    target_tokens,
                    token_loss_weights,
                )
            if ccdd_continuous_targets is not None:
                predicted_continuous = self.continuous_head(predicted_embeddings)
                continuous_loss_weights = loss_weights * ccdd_continuous_mask.to(dtype=loss_weights.dtype)
                outputs["ccdd_predicted_continuous"] = predicted_continuous
                outputs["ccdd_continuous_loss_weights"] = continuous_loss_weights
                outputs["ccdd_continuous_loss"] = weighted_position_mse(
                    predicted_continuous,
                    ccdd_continuous_targets,
                    continuous_loss_weights,
                )
            else:
                outputs["ccdd_continuous_loss"] = torch.zeros(
                    (),
                    dtype=predicted_embeddings.dtype,
                    device=predicted_embeddings.device,
                )
            outputs["loss"] = outputs["diffusion_loss"] + outputs["token_loss"] + outputs["ccdd_continuous_loss"]
        return outputs

    @torch.no_grad()
    def sample(
        self,
        latent_tokens: torch.Tensor,
        latent_mask: torch.Tensor,
        msa_grid_tokens: torch.Tensor | None = None,
        msa_grid_mask: torch.Tensor | None = None,
        steps: int | None = None,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        steps = steps or min(self.num_timesteps, 50)
        if steps < 1:
            raise ValueError("steps must be at least 1")
        batch_size = latent_tokens.shape[0]
        device = latent_tokens.device
        noisy = torch.randn(batch_size, self.max_sequence_length, self.d_model, device=device) * temperature
        schedule = torch.linspace(self.num_timesteps - 1, 0, steps, device=device).round().long()

        for idx, timestep in enumerate(schedule):
            timestep_batch = timestep.expand(batch_size)
            predicted = self.denoise(
                noisy,
                timestep_batch,
                latent_tokens,
                latent_mask,
                msa_grid_tokens=msa_grid_tokens,
                msa_grid_mask=msa_grid_mask,
            )
            if idx + 1 < len(schedule):
                next_timestep = schedule[idx + 1].expand(batch_size)
                noisy = self.q_sample(predicted, next_timestep)
            else:
                noisy = predicted

        logits = self.lm_head(noisy)
        tokens = torch.argmax(logits, dim=-1)
        return {"tokens": tokens, "logits": logits, "embeddings": noisy}

    @torch.no_grad()
    def sample_discrete(
        self,
        latent_tokens: torch.Tensor,
        latent_mask: torch.Tensor,
        steps: int | None = None,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        unconditional_latent_tokens: torch.Tensor | None = None,
        unconditional_latent_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        steps = steps or min(self.num_timesteps, 50)
        if steps < 1:
            raise ValueError("steps must be at least 1")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if (unconditional_latent_tokens is None) != (unconditional_latent_mask is None):
            raise ValueError("provide both unconditional latent tensors or neither")

        batch_size = latent_tokens.shape[0]
        device = latent_tokens.device
        tokens = torch.full(
            (batch_size, self.max_sequence_length),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        schedule = torch.linspace(self.num_timesteps - 1, 0, steps, device=device).round().long()
        confidence = torch.zeros(tokens.shape, dtype=torch.float32, device=device)
        logits = torch.empty(batch_size, self.max_sequence_length, self.vocab_size, device=device)

        for idx, timestep in enumerate(schedule):
            timestep_batch = timestep.expand(batch_size)
            embeddings = self.token_embedding(tokens)
            predicted = self.denoise(embeddings, timestep_batch, latent_tokens, latent_mask)
            logits = self.lm_head(predicted)
            if (
                guidance_scale != 1.0
                and unconditional_latent_tokens is not None
                and unconditional_latent_mask is not None
            ):
                uncond_predicted = self.denoise(
                    embeddings,
                    timestep_batch,
                    unconditional_latent_tokens,
                    unconditional_latent_mask,
                )
                uncond_logits = self.lm_head(uncond_predicted)
                logits = uncond_logits + guidance_scale * (logits - uncond_logits)

            logits[..., self.mask_token_id] = -torch.inf
            if temperature == 0.0:
                probabilities = torch.softmax(logits, dim=-1)
                sampled = torch.argmax(probabilities, dim=-1)
            else:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                sampled = torch.multinomial(
                    probabilities.reshape(-1, self.vocab_size),
                    num_samples=1,
                ).reshape(batch_size, self.max_sequence_length)
            confidence = probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            if idx + 1 == len(schedule):
                tokens = sampled
                break

            next_timestep = schedule[idx + 1]
            next_mask_fraction = float(self.discrete_corruption_probability(next_timestep).item())
            next_mask_count = int(round(self.max_sequence_length * next_mask_fraction))
            tokens = sampled.clone()
            if next_mask_count > 0:
                next_mask_count = min(next_mask_count, self.max_sequence_length)
                for batch_index in range(batch_size):
                    remask_indices = torch.topk(
                        confidence[batch_index],
                        k=next_mask_count,
                        largest=False,
                    ).indices
                    tokens[batch_index, remask_indices] = self.mask_token_id

        return {
            "tokens": tokens,
            "logits": logits,
            "confidence": confidence,
            "mask_fraction": (tokens == self.mask_token_id).to(dtype=torch.float32).mean(),
        }


class LatentDiffusionDenoiser(nn.Module):
    """Target-free denoiser for sequence summary latents conditioned on clean memory."""

    def __init__(
        self,
        d_model: int = 128,
        num_latent_tokens: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_timesteps: int = 1000,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
    ) -> None:
        super().__init__()
        if num_latent_tokens < 1:
            raise ValueError("num_latent_tokens must be at least 1")
        if num_timesteps < 2:
            raise ValueError("num_timesteps must be at least 2")
        self.d_model = d_model
        self.num_latent_tokens = int(num_latent_tokens)
        self.num_timesteps = int(num_timesteps)

        self.latent_start = nn.Parameter(torch.zeros(1, self.num_latent_tokens, d_model))
        self.position_embedding = nn.Embedding(self.num_latent_tokens, d_model)
        self.time_embedding = nn.Sequential(
            nn.Embedding(num_timesteps, d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.denoiser = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)

        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod), persistent=False)
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod), persistent=False)

    def q_sample(
        self,
        clean_latents: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(clean_latents)
        scale_clean = self.sqrt_alpha_cumprod[timesteps].view(-1, 1, 1)
        scale_noise = self.sqrt_one_minus_alpha_cumprod[timesteps].view(-1, 1, 1)
        return scale_clean * clean_latents + scale_noise * noise

    def start_state(
        self,
        batch_size: int,
        timesteps: torch.Tensor,
        device: torch.device,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        clean_start = self.latent_start.to(device=device).expand(batch_size, -1, -1)
        return self.q_sample(clean_start, timesteps, noise=noise)

    def denoise(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_latents.shape[1:] != (self.num_latent_tokens, self.d_model):
            raise ValueError(
                f"noisy_latents must have shape B x {self.num_latent_tokens} x {self.d_model}"
            )
        if memory_mask.shape != memory_tokens.shape[:2]:
            raise ValueError("memory_mask must have shape B x L matching memory_tokens")
        batch_size = noisy_latents.shape[0]
        if timesteps.shape != (batch_size,):
            raise ValueError("timesteps must have shape B")
        positions = torch.arange(self.num_latent_tokens, device=noisy_latents.device)
        latents = noisy_latents + self.position_embedding(positions).unsqueeze(0)
        latents = latents + self.time_embedding(timesteps).unsqueeze(1)
        decoded = self.denoiser(
            tgt=latents,
            memory=memory_tokens,
            memory_key_padding_mask=~memory_mask,
        )
        return self.output_norm(decoded)

    def forward(
        self,
        memory_tokens: torch.Tensor,
        memory_mask: torch.Tensor,
        target_latents: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        noisy_latents = self.start_state(
            batch_size=target_latents.shape[0],
            timesteps=timesteps,
            device=target_latents.device,
        )
        predicted_latents = self.denoise(noisy_latents, timesteps, memory_tokens, memory_mask)
        latent_loss = F.mse_loss(predicted_latents, target_latents, reduction="mean")
        return {
            "clean_sequence_latents": target_latents,
            "noisy_sequence_latents": noisy_latents,
            "predicted_sequence_latents": predicted_latents,
            "latent_loss": latent_loss,
        }

    @torch.no_grad()
    def sample(
        self,
        memory_tokens: torch.Tensor,
        memory_mask: torch.Tensor,
        steps: int | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        steps = steps or min(self.num_timesteps, 50)
        if steps < 1:
            raise ValueError("steps must be at least 1")
        batch_size = memory_tokens.shape[0]
        device = memory_tokens.device
        schedule = torch.linspace(self.num_timesteps - 1, 0, steps, device=device).round().long()
        latents = self.start_state(
            batch_size=batch_size,
            timesteps=schedule[0].expand(batch_size),
            device=device,
            noise=torch.randn(batch_size, self.num_latent_tokens, self.d_model, device=device) * temperature,
        )
        for idx, timestep in enumerate(schedule):
            timestep_batch = timestep.expand(batch_size)
            predicted = self.denoise(latents, timestep_batch, memory_tokens, memory_mask)
            if idx + 1 < len(schedule):
                next_timestep = schedule[idx + 1].expand(batch_size)
                latents = self.q_sample(predicted, next_timestep)
            else:
                latents = predicted
        return latents


class MSASequenceDiffusionModel(nn.Module):
    """End-to-end trainable boundary from frozen MSA embeddings to sequence diffusion."""

    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 128,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_timesteps: int = 1000,
        numeric_condition_fields: Sequence[str] = (),
        categorical_condition_fields: Sequence[str] = (),
        categorical_vocab_sizes: Sequence[int] | None = None,
        condition_layers: int = 1,
        latent_codiffusion_tokens: int = 0,
        ccdd_mode: str = "off",
    ) -> None:
        super().__init__()
        self.numeric_condition_fields = tuple(numeric_condition_fields)
        self.categorical_condition_fields = tuple(categorical_condition_fields)
        if latent_codiffusion_tokens < 0:
            raise ValueError("latent_codiffusion_tokens must be non-negative")
        if ccdd_mode not in {"off", "mdit"}:
            raise ValueError("ccdd_mode must be 'off' or 'mdit'")
        self.latent_codiffusion_tokens = int(latent_codiffusion_tokens)
        self.ccdd_mode = ccdd_mode
        self.depth_scaler = MSADepthScaler(input_dim=input_dim, d_model=d_model, dropout=dropout)
        self.condition_tokens = (
            MetadataConditionTokenBank(
                numeric_fields=self.numeric_condition_fields,
                categorical_fields=self.categorical_condition_fields,
                categorical_vocab_sizes=categorical_vocab_sizes or (),
                d_model=d_model,
                num_heads=num_heads,
                num_layers=condition_layers,
                dropout=dropout,
            )
            if self.numeric_condition_fields or self.categorical_condition_fields
            else None
        )
        self.decoder = SequenceDiffusionDecoder(
            d_model=d_model,
            max_sequence_length=max_sequence_length,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_timesteps=num_timesteps,
        )
        self.latent_denoiser = (
            LatentDiffusionDenoiser(
                d_model=d_model,
                num_latent_tokens=self.latent_codiffusion_tokens,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=dropout,
                num_timesteps=num_timesteps,
            )
            if self.latent_codiffusion_tokens
            else None
        )
        self.sequence_continuous_projector = (
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
            if self.ccdd_mode != "off"
            else None
        )
        self.null_condition_token = nn.Parameter(torch.zeros(1, 1, d_model))

    def encode_latent_memory(
        self,
        token_embeddings: torch.Tensor,
        aa_mask: torch.Tensor,
        condition_values: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
        categorical_condition_ids: torch.Tensor | None = None,
        categorical_condition_mask: torch.Tensor | None = None,
        condition_drop_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        msa_tokens, msa_mask = self.depth_scaler(token_embeddings, aa_mask)
        if self.condition_tokens is None:
            if any(
                item is not None
                for item in (
                    condition_values,
                    condition_mask,
                    categorical_condition_ids,
                    categorical_condition_mask,
                )
            ):
                raise ValueError("condition tensors were provided but the model has no condition fields")
            latent_tokens = msa_tokens
            latent_mask = msa_mask
            outputs: dict[str, torch.Tensor] = {}
        else:
            condition_tokens, condition_token_mask = self.condition_tokens(
                numeric_values=condition_values,
                numeric_mask=condition_mask,
                categorical_ids=categorical_condition_ids,
                categorical_mask=categorical_condition_mask,
            )
            latent_tokens = torch.cat([condition_tokens, msa_tokens], dim=1)
            latent_mask = torch.cat([condition_token_mask, msa_mask], dim=1)
            outputs = {
                "condition_tokens": condition_tokens,
                "condition_token_mask": condition_token_mask,
            }
        outputs.update(
            {
                "latent_tokens": latent_tokens,
                "latent_mask": latent_mask,
                "msa_latent_tokens": msa_tokens,
                "msa_latent_mask": msa_mask,
            }
        )
        if condition_drop_mask is not None:
            if condition_drop_mask.shape != (latent_tokens.shape[0],):
                raise ValueError("condition_drop_mask must have shape B")
            condition_drop_mask = condition_drop_mask.to(dtype=torch.bool, device=latent_tokens.device)
            dropped_tokens = torch.zeros_like(latent_tokens)
            dropped_tokens[:, :1] = self.null_condition_token.to(dtype=latent_tokens.dtype)
            dropped_mask = torch.zeros_like(latent_mask)
            dropped_mask[:, :1] = True
            outputs["latent_tokens"] = torch.where(
                condition_drop_mask.view(-1, 1, 1),
                dropped_tokens,
                latent_tokens,
            )
            outputs["latent_mask"] = torch.where(
                condition_drop_mask.view(-1, 1),
                dropped_mask,
                latent_mask,
            )
            outputs["condition_drop_mask"] = condition_drop_mask
        return outputs

    def _sequence_decoder_memory(
        self,
        memory: dict[str, torch.Tensor],
        target_tokens: torch.Tensor | None,
        loss_weights: torch.Tensor | None,
        timesteps: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.latent_denoiser is None or target_tokens is None:
            return memory["latent_tokens"], memory["latent_mask"], {}
        if timesteps is None:
            raise ValueError("timesteps are required for latent codiffusion")
        if loss_weights is None:
            loss_weights = torch.ones_like(target_tokens, dtype=memory["latent_tokens"].dtype)
        clean_embeddings = self.decoder.token_embedding(target_tokens)
        target_latents = sequence_latent_targets(
            token_embeddings=clean_embeddings,
            loss_weights=loss_weights,
            num_latent_tokens=self.latent_codiffusion_tokens,
        ).detach()
        latent_outputs = self.latent_denoiser(
            memory_tokens=memory["latent_tokens"],
            memory_mask=memory["latent_mask"],
            target_latents=target_latents,
            timesteps=timesteps,
        )
        predicted_latents = latent_outputs["predicted_sequence_latents"]
        latent_mask = torch.ones(
            predicted_latents.shape[:2],
            dtype=torch.bool,
            device=predicted_latents.device,
        )
        sequence_memory_tokens = torch.cat([predicted_latents, memory["latent_tokens"]], dim=1)
        sequence_memory_mask = torch.cat([latent_mask, memory["latent_mask"]], dim=1)
        latent_outputs["sequence_memory_tokens"] = sequence_memory_tokens
        latent_outputs["sequence_memory_mask"] = sequence_memory_mask
        return sequence_memory_tokens, sequence_memory_mask, latent_outputs

    def forward(
        self,
        token_embeddings: torch.Tensor,
        aa_mask: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
        loss_weights: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        condition_values: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
        categorical_condition_ids: torch.Tensor | None = None,
        categorical_condition_mask: torch.Tensor | None = None,
        decoder_start_mode: str = "mean",
        decoder_token_dropout: float = 0.0,
        decoder_span_mask_fraction: float = 0.0,
        decoder_span_mask_length: int = 16,
        discrete_loss_corrupted_only: bool = True,
        condition_dropout: float = 0.0,
        target_continuous_embeddings: torch.Tensor | None = None,
        target_continuous_mask: torch.Tensor | None = None,
        ccdd_continuous_timestep_scale: float = 0.75,
        ccdd_continuous_dropout: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if not 0.0 <= condition_dropout <= 1.0:
            raise ValueError("condition_dropout must be in [0, 1]")
        condition_drop_mask = None
        if self.training and condition_dropout > 0.0:
            condition_drop_mask = torch.rand(
                token_embeddings.shape[0],
                dtype=torch.float32,
                device=token_embeddings.device,
            ) < condition_dropout
        memory = self.encode_latent_memory(
            token_embeddings=token_embeddings,
            aa_mask=aa_mask,
            condition_values=condition_values,
            condition_mask=condition_mask,
            categorical_condition_ids=categorical_condition_ids,
            categorical_condition_mask=categorical_condition_mask,
            condition_drop_mask=condition_drop_mask,
        )
        sequence_memory_tokens, sequence_memory_mask, latent_outputs = self._sequence_decoder_memory(
            memory=memory,
            target_tokens=target_tokens,
            loss_weights=loss_weights,
            timesteps=timesteps,
        )
        ccdd_targets = None
        ccdd_mask = None
        if self.ccdd_mode != "off":
            if target_tokens is not None and target_continuous_embeddings is None:
                raise ValueError("target_continuous_embeddings are required when ccdd_mode is enabled")
            if target_continuous_embeddings is not None:
                if self.sequence_continuous_projector is None:
                    raise ValueError("received target_continuous_embeddings but ccdd_mode is off")
                if target_continuous_embeddings.shape[:2] != target_tokens.shape:
                    raise ValueError("target_continuous_embeddings must have shape B x L x input_dim")
                if target_continuous_mask is None:
                    target_continuous_mask = torch.ones(
                        target_tokens.shape,
                        dtype=torch.bool,
                        device=target_tokens.device,
                    )
                if target_continuous_mask.shape != target_tokens.shape:
                    raise ValueError("target_continuous_mask must have shape B x L")
                ccdd_targets = self.sequence_continuous_projector(target_continuous_embeddings)
                ccdd_targets = torch.where(
                    target_continuous_mask.unsqueeze(-1),
                    ccdd_targets,
                    torch.zeros_like(ccdd_targets),
                )
                ccdd_mask = target_continuous_mask
        outputs = self.decoder(
            latent_tokens=sequence_memory_tokens,
            latent_mask=sequence_memory_mask,
            target_tokens=target_tokens,
            loss_weights=loss_weights,
            timesteps=timesteps,
            decoder_start_mode=decoder_start_mode,
            decoder_token_dropout=decoder_token_dropout,
            decoder_span_mask_fraction=decoder_span_mask_fraction,
            decoder_span_mask_length=decoder_span_mask_length,
            discrete_loss_corrupted_only=discrete_loss_corrupted_only,
            ccdd_continuous_targets=ccdd_targets,
            ccdd_continuous_mask=ccdd_mask,
            ccdd_continuous_timestep_scale=ccdd_continuous_timestep_scale,
            ccdd_continuous_dropout=ccdd_continuous_dropout,
        )
        if "latent_loss" not in latent_outputs:
            outputs["latent_loss"] = torch.zeros((), dtype=outputs["logits"].dtype, device=outputs["logits"].device)
        outputs.update(latent_outputs)
        outputs.update(memory)
        return outputs

    @torch.no_grad()
    def sample(
        self,
        token_embeddings: torch.Tensor,
        aa_mask: torch.Tensor,
        condition_values: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
        categorical_condition_ids: torch.Tensor | None = None,
        categorical_condition_mask: torch.Tensor | None = None,
        steps: int | None = None,
        temperature: float = 1.0,
        sample_mode: str = "continuous",
        guidance_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        memory = self.encode_latent_memory(
            token_embeddings=token_embeddings,
            aa_mask=aa_mask,
            condition_values=condition_values,
            condition_mask=condition_mask,
            categorical_condition_ids=categorical_condition_ids,
            categorical_condition_mask=categorical_condition_mask,
        )
        if sample_mode == "continuous":
            latent_tokens = memory["latent_tokens"]
            latent_mask = memory["latent_mask"]
            if self.latent_denoiser is not None:
                sampled_latents = self.latent_denoiser.sample(
                    memory_tokens=latent_tokens,
                    memory_mask=latent_mask,
                    steps=steps,
                    temperature=temperature,
                )
                sampled_mask = torch.ones(
                    sampled_latents.shape[:2],
                    dtype=torch.bool,
                    device=sampled_latents.device,
                )
                latent_tokens = torch.cat([sampled_latents, latent_tokens], dim=1)
                latent_mask = torch.cat([sampled_mask, latent_mask], dim=1)
            outputs = self.decoder.sample(
                latent_tokens=latent_tokens,
                latent_mask=latent_mask,
                steps=steps,
                temperature=temperature,
            )
        elif sample_mode == "discrete":
            unconditional_memory = None
            if guidance_scale != 1.0:
                unconditional_memory = self.encode_latent_memory(
                    token_embeddings=token_embeddings,
                    aa_mask=aa_mask,
                    condition_values=condition_values,
                    condition_mask=condition_mask,
                    categorical_condition_ids=categorical_condition_ids,
                    categorical_condition_mask=categorical_condition_mask,
                    condition_drop_mask=torch.ones(
                        token_embeddings.shape[0],
                        dtype=torch.bool,
                        device=token_embeddings.device,
                    ),
                )
            outputs = self.decoder.sample_discrete(
                latent_tokens=memory["latent_tokens"],
                latent_mask=memory["latent_mask"],
                steps=steps,
                temperature=temperature,
                guidance_scale=guidance_scale,
                unconditional_latent_tokens=(
                    unconditional_memory["latent_tokens"] if unconditional_memory is not None else None
                ),
                unconditional_latent_mask=(
                    unconditional_memory["latent_mask"] if unconditional_memory is not None else None
                ),
            )
        else:
            raise ValueError(f"unknown sample_mode: {sample_mode}")
        outputs.update(memory)
        return outputs


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
        observed_mask = observed_mask.to(dtype=torch.bool)
        tokens: list[torch.Tensor] = []
        for idx, head in enumerate(self.value_heads):
            value_token = head(values[:, idx : idx + 1])
            missing_token = self.missing_tokens[idx].unsqueeze(0).expand_as(value_token)
            token = torch.where(observed_mask[:, idx : idx + 1], value_token, missing_token)
            token = token + self.field_tokens[idx]
            tokens.append(token)
        return self.output_norm(torch.stack(tokens, dim=1))


class CategoricalConditionTokenBank(nn.Module):
    """Per-field categorical condition tokens with support for multi-value fields."""

    def __init__(self, fields: Sequence[str], vocab_sizes: Sequence[int], d_model: int = 128) -> None:
        super().__init__()
        if not fields:
            raise ValueError("at least one categorical condition field is required")
        if len(fields) != len(vocab_sizes):
            raise ValueError("categorical fields and vocab sizes must have the same length")
        if any(size < 1 for size in vocab_sizes):
            raise ValueError("each categorical vocabulary must contain at least one token")
        self.fields = tuple(fields)
        self.vocab_sizes = tuple(int(size) for size in vocab_sizes)
        self.d_model = d_model
        self.embeddings = nn.ModuleList([nn.Embedding(size, d_model) for size in self.vocab_sizes])
        self.field_tokens = nn.Parameter(torch.empty(len(self.fields), d_model))
        self.missing_tokens = nn.Parameter(torch.empty(len(self.fields), d_model))
        self.output_norm = nn.LayerNorm(d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.field_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.missing_tokens, mean=0.0, std=0.02)

    def forward(self, category_ids: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        if category_ids.shape != observed_mask.shape:
            raise ValueError("categorical ids and observed mask must have the same shape")
        if category_ids.ndim != 3 or category_ids.shape[1] != len(self.fields):
            raise ValueError(f"expected categorical tensors with shape B x {len(self.fields)} x K")
        observed_mask = observed_mask.to(dtype=torch.bool)
        tokens: list[torch.Tensor] = []
        for idx, embedding in enumerate(self.embeddings):
            field_ids = category_ids[:, idx, :]
            field_mask = observed_mask[:, idx, :] & (field_ids >= 0)
            if torch.any(field_mask & (field_ids >= self.vocab_sizes[idx])):
                raise ValueError(f"categorical ids for {self.fields[idx]} exceed vocabulary size {self.vocab_sizes[idx]}")
            safe_ids = field_ids.clamp_min(0)
            embedded = embedding(safe_ids)
            mask_f = field_mask.to(dtype=embedded.dtype)
            counts = mask_f.sum(dim=1, keepdim=True)
            value_token = (embedded * mask_f.unsqueeze(-1)).sum(dim=1) / counts.clamp_min(1.0)
            missing_token = self.missing_tokens[idx].unsqueeze(0).expand_as(value_token)
            token = torch.where(counts > 0, value_token, missing_token)
            token = token + self.field_tokens[idx]
            tokens.append(token)
        return self.output_norm(torch.stack(tokens, dim=1))


class MetadataConditionTokenBank(nn.Module):
    """Mixed numeric and categorical metadata tokens, optionally self-attended before use."""

    def __init__(
        self,
        numeric_fields: Sequence[str] = (),
        categorical_fields: Sequence[str] = (),
        categorical_vocab_sizes: Sequence[int] = (),
        d_model: int = 128,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not numeric_fields and not categorical_fields:
            raise ValueError("at least one numeric or categorical condition field is required")
        self.numeric_fields = tuple(numeric_fields)
        self.categorical_fields = tuple(categorical_fields)
        self.numeric_tokens = (
            NumericConditionTokenBank(fields=self.numeric_fields, d_model=d_model, hidden_dim=hidden_dim)
            if self.numeric_fields
            else None
        )
        self.categorical_tokens = (
            CategoricalConditionTokenBank(
                fields=self.categorical_fields,
                vocab_sizes=categorical_vocab_sizes,
                d_model=d_model,
            )
            if self.categorical_fields
            else None
        )
        self.kind_embedding = nn.Embedding(2, d_model)
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        self.mixer: nn.TransformerEncoder | None
        if num_layers:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.mixer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:
            self.mixer = None
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        numeric_values: torch.Tensor | None = None,
        numeric_mask: torch.Tensor | None = None,
        categorical_ids: torch.Tensor | None = None,
        categorical_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pieces: list[torch.Tensor] = []
        if self.numeric_tokens is not None:
            if numeric_values is None or numeric_mask is None:
                raise ValueError("numeric condition tensors are required by this model")
            numeric = self.numeric_tokens(numeric_values, numeric_mask)
            numeric_kind = self.kind_embedding.weight[0].view(1, 1, -1)
            pieces.append(numeric + numeric_kind)
        if self.categorical_tokens is not None:
            if categorical_ids is None or categorical_mask is None:
                raise ValueError("categorical condition tensors are required by this model")
            categorical = self.categorical_tokens(categorical_ids, categorical_mask)
            categorical_kind = self.kind_embedding.weight[1].view(1, 1, -1)
            pieces.append(categorical + categorical_kind)
        tokens = torch.cat(pieces, dim=1)
        token_mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if self.mixer is not None:
            tokens = self.mixer(tokens, src_key_padding_mask=~token_mask)
        return self.output_norm(tokens), token_mask


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
