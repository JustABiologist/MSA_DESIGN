#!/usr/bin/env python3
"""Diagnostic attention maps and profile-channel ablations for mean-start CCDD.

This is deliberately an analysis script rather than a training dependency. It
monkeypatches the decoder's cross-attention modules at runtime so PyTorch
returns per-head memory attention weights, then pairs those maps with simple
profile-feature ablations and gradient saliency.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_mean_start_ccdd_from_cached_msas import (  # noqa: E402
    AA_TOKENS,
    CATEGORICAL_FIELDS,
    DEFAULT_EMBEDDING_MANIFEST,
    DEFAULT_LABEL_SUMMARY,
    NUMERIC_FIELDS,
    CachedMSAMaskedRowsDataset,
    CachedMSARowDataset,
    MeanStartCCDDModel,
    RowReconstructionCollator,
    build_examples,
    build_msa_groups,
    load_label_summary,
    masked_residue_accuracy_value,
    move_batch,
    read_embedding_manifest,
    uses_gap_inclusive_msa_mask,
    uses_msa_embedding_memory,
    weighted_residue_accuracy,
)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_val_batch(args: argparse.Namespace, checkpoint: dict[str, Any]) -> tuple[dict[str, Any], int]:
    config = checkpoint.get("config", {})
    labels = load_label_summary(Path(args.label_summary))
    rows = read_embedding_manifest(Path(args.embedding_manifest), split=args.split)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_msas:
        rows = rows[: args.max_msas]
    examples = build_examples(rows, max_rows_per_msa=config.get("max_rows_per_msa"))
    rng.shuffle(examples)
    if not examples:
        raise SystemExit("No examples selected for attention diagnostic")

    dataset_kwargs = {
        "labels": labels,
        "numeric_means": checkpoint["numeric_means"],
        "numeric_stds": checkpoint["numeric_stds"],
        "category_buckets": int(config.get("category_buckets", 4096)),
        "cache_size": args.cache_size,
        "consensus_loss_mode": str(config.get("consensus_loss_mode", "none")),
        "consensus_match_weight": float(config.get("consensus_match_weight", 0.35)),
        "nonconsensus_weight": float(config.get("nonconsensus_weight", 2.5)),
        "unobserved_nonconsensus_weight": float(config.get("unobserved_nonconsensus_weight", 1.0)),
        "max_sequence_loss_weight": float(config.get("max_sequence_loss_weight", 3.0)),
        "variable_column_min_entropy": float(config.get("variable_column_min_entropy", 0.05)),
        "variable_column_max_consensus": float(config.get("variable_column_max_consensus", 0.92)),
        "require_msa_embeddings": uses_msa_embedding_memory(str(config.get("memory_mode", "profile_row"))),
        "msa_embedding_dtype": str(config.get("msa_embedding_dtype", "float32")),
        "max_msa_context_rows": config.get("max_msa_context_rows"),
        "gap_inclusive_msa_mask": uses_gap_inclusive_msa_mask(str(config.get("memory_mode", "profile_row"))),
        "require_target_continuous_embeddings": str(config.get("continuous_target_mode", "token_embedding"))
        == "target_row_embedding",
    }
    masked_min = int(config.get("masked_rows_per_msa_min") or 1)
    masked_max = int(config.get("masked_rows_per_msa_max") or 1)
    if args.force_single_row:
        masked_min = masked_max = 1
    grouped = masked_max > 1
    if grouped:
        groups = build_msa_groups(examples)
        rng.shuffle(groups)
        dataset = CachedMSAMaskedRowsDataset(
            groups,
            masked_rows_per_msa_min=masked_min,
            masked_rows_per_msa_max=masked_max,
            **dataset_kwargs,
        )
        batch_size = args.msa_group_batch_size
    else:
        dataset = CachedMSARowDataset(examples, **dataset_kwargs)
        batch_size = args.example_batch_size

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    collator = RowReconstructionCollator(
        max_sequence_length=int(config.get("max_sequence_length", 1024)),
        tail_stop_weight=float(config.get("tail_stop_weight", 0.05)),
        profile_feature_mode=str(config.get("profile_feature_mode", "full")),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collator)
    batch = next(iter(loader))
    if args.max_flat_examples and len(batch["target_sequences"]) > args.max_flat_examples:
        keep = args.max_flat_examples
        trimmed: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (len(batch["target_sequences"]),):
                trimmed[key] = value[:keep]
            elif isinstance(value, list) and len(value) == len(batch["target_sequences"]):
                trimmed[key] = value[:keep]
            else:
                trimmed[key] = value
        batch = trimmed
    first_item = dataset[0]
    if isinstance(first_item, list):
        first_item = first_item[0]
    row_embedding_dim = int(first_item["row_embeddings"].shape[-1])
    msa_embedding_dim = int(first_item["msa_embeddings"].shape[-1])
    target_continuous_dim = int(first_item["target_continuous_embeddings"].shape[-1])
    return batch, row_embedding_dim, msa_embedding_dim, target_continuous_dim


def build_model(
    checkpoint: dict[str, Any],
    row_embedding_dim: int,
    msa_embedding_dim: int,
    target_continuous_dim: int,
    device: torch.device,
) -> MeanStartCCDDModel:
    config = checkpoint.get("config", {})
    model = MeanStartCCDDModel(
        row_embedding_dim=row_embedding_dim,
        d_model=int(config.get("d_model", 192)),
        layers=int(config.get("layers", 4)),
        heads=int(config.get("heads", 6)),
        dropout=float(config.get("dropout", 0.1)),
        max_sequence_length=int(config.get("max_sequence_length", 1024)),
        diffusion_timesteps=int(config.get("diffusion_timesteps", 250)),
        category_buckets=int(config.get("category_buckets", 4096)),
        memory_mode=str(config.get("memory_mode", "profile_row")),
        profile_feature_mode=str(config.get("profile_feature_mode", "full")),
        msa_embedding_dim=msa_embedding_dim,
        continuous_target_mode=str(config.get("continuous_target_mode", "token_embedding")),
        target_continuous_dim=target_continuous_dim,
        msa_axial_layers=int(config.get("msa_axial_layers", 1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


class CrossAttentionCapture:
    def __init__(self, model: MeanStartCCDDModel) -> None:
        self.records: list[tuple[int, torch.Tensor]] = []
        self._originals: list[tuple[Any, Any]] = []
        if model.decoder.denoiser is not None:
            modules = [
                (layer_index, layer.multihead_attn)
                for layer_index, layer in enumerate(model.decoder.denoiser.layers)
            ]
        else:
            modules = [
                (layer_index, layer.static_memory_attn)
                for layer_index, layer in enumerate(model.decoder.msa_grid_layers)
            ]
        for layer_index, module in modules:
            original = module.forward

            def wrapped_forward(query, key, value, *args, _layer_index=layer_index, _original=original, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                output, weights = _original(query, key, value, *args, **kwargs)
                self.records.append((_layer_index, weights.detach().cpu()))
                return output, weights

            module.forward = wrapped_forward
            self._originals.append((module, original))

    def clear(self) -> None:
        self.records.clear()

    def restore(self) -> None:
        for module, original in self._originals:
            module.forward = original


def memory_masks(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    batch_size, profile_len = batch["profile_mask"].shape
    row_len = batch["row_mask"].shape[1]
    condition_len = 1 + len(CATEGORICAL_FIELDS)
    total_len = condition_len + profile_len + row_len
    masks = {
        "condition": torch.zeros((batch_size, total_len), dtype=torch.bool),
        "profile_all": torch.zeros((batch_size, total_len), dtype=torch.bool),
        "profile_variable": torch.zeros((batch_size, total_len), dtype=torch.bool),
        "profile_nonvariable": torch.zeros((batch_size, total_len), dtype=torch.bool),
        "row": torch.zeros((batch_size, total_len), dtype=torch.bool),
    }
    masks["condition"][:, :condition_len] = True
    profile_slice = slice(condition_len, condition_len + profile_len)
    row_slice = slice(condition_len + profile_len, total_len)
    masks["profile_all"][:, profile_slice] = batch["profile_mask"].cpu()
    masks["profile_variable"][:, profile_slice] = batch["profile_variable_mask"].cpu() & batch["profile_mask"].cpu()
    masks["profile_nonvariable"][:, profile_slice] = (~batch["profile_variable_mask"].cpu()) & batch["profile_mask"].cpu()
    masks["row"][:, row_slice] = batch["row_mask"].cpu()
    return masks


def target_position_masks(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    batch_size, max_len = batch["target_tokens"].shape
    lengths = torch.tensor([min(len(seq), max_len) for seq in batch["target_sequences"]], dtype=torch.long)
    arange = torch.arange(max_len).unsqueeze(0).expand(batch_size, -1)
    residue = arange < lengths.unsqueeze(1)
    return {
        "residue": residue,
        "consensus": batch["consensus_match_mask"].cpu() & batch["consensus_observed_mask"].cpu() & residue,
        "nonconsensus": batch["nonconsensus_mask"].cpu() & residue,
        "variable_nonconsensus": batch["variable_nonconsensus_mask"].cpu() & residue,
    }


def summarize_attention(
    records: list[tuple[int, torch.Tensor]],
    batch: dict[str, Any],
) -> tuple[list[dict[str, float | int | str]], dict[str, np.ndarray]]:
    mem_masks = memory_masks(batch)
    pos_masks = target_position_masks(batch)
    rows: list[dict[str, float | int | str]] = []
    layer_names = sorted({layer for layer, _ in records})
    categories = ["condition", "profile_nonvariable", "profile_variable", "row"]
    matrices: dict[str, np.ndarray] = {
        name: np.zeros((len(layer_names), len(categories)), dtype=np.float64) for name in pos_masks
    }
    for layer_index, weights in records:
        # weights: B x heads x target_len x memory_len
        layer_pos = layer_names.index(layer_index)
        memory_len = weights.shape[-1]
        mem_masks = {name: mask[:, :memory_len] for name, mask in mem_masks.items()}
        valid_mem = torch.zeros_like(next(iter(mem_masks.values())))
        for mask in mem_masks.values():
            valid_mem |= mask
        for pos_name, pos_mask in pos_masks.items():
            denom_positions = pos_mask.to(dtype=torch.float32).sum().item()
            for cat_index, cat_name in enumerate(categories):
                mem_mask = mem_masks[cat_name]
                mask = pos_mask[:, None, :, None] & mem_mask[:, None, None, :]
                if denom_positions <= 0:
                    mass = 0.0
                else:
                    mass = float((weights * mask.to(dtype=weights.dtype)).sum().item())
                    mass /= float(weights.shape[1] * denom_positions)
                rows.append(
                    {
                        "layer": int(layer_index),
                        "position_set": pos_name,
                        "memory_category": cat_name,
                        "attention_mass": mass,
                    }
                )
                matrices[pos_name][layer_pos, cat_index] = mass
    return rows, matrices


def clone_batch_with_variant(batch: dict[str, Any], variant: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.clone() if isinstance(value, torch.Tensor) else value
    if variant == "baseline":
        return out
    if variant == "uniform_profile_aa":
        profiles = out["profiles"].clone()
        aa_mass = profiles[:, :, : len(AA_TOKENS)].sum(dim=-1, keepdim=True)
        profiles[:, :, : len(AA_TOKENS)] = aa_mass / float(len(AA_TOKENS))
        out["profiles"] = profiles
        return out
    if variant == "zero_profile_aa":
        profiles = out["profiles"].clone()
        profiles[:, :, : len(AA_TOKENS)] = 0.0
        out["profiles"] = profiles
        return out
    if variant == "zero_gap_frequency":
        profiles = out["profiles"].clone()
        gap_index = len(AA_TOKENS) if profiles.shape[-1] > 2 else 0
        profiles[:, :, gap_index] = 0.0
        out["profiles"] = profiles
        return out
    if variant == "zero_coverage":
        profiles = out["profiles"].clone()
        coverage_index = len(AA_TOKENS) + 1 if profiles.shape[-1] > 2 else 1
        profiles[:, :, coverage_index] = 0.0
        out["profiles"] = profiles
        return out
    if variant == "remove_profile_tokens":
        out["profile_mask"] = torch.zeros_like(out["profile_mask"])
        return out
    if variant == "remove_row_tokens":
        out["row_mask"] = torch.zeros_like(out["row_mask"])
        return out
    if variant == "blank_conditions":
        out["numeric_values"] = torch.zeros_like(out["numeric_values"])
        out["numeric_mask"] = torch.zeros_like(out["numeric_mask"])
        out["category_ids"] = torch.full_like(out["category_ids"], -1)
        out["category_mask"] = torch.zeros_like(out["category_mask"])
        return out
    raise ValueError(f"unknown ablation variant: {variant}")


def forward_metrics(model: MeanStartCCDDModel, batch: dict[str, Any], device: torch.device) -> dict[str, float]:
    moved = move_batch(batch, device)
    batch_size = moved["target_tokens"].shape[0]
    timesteps = torch.zeros((batch_size,), dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(
            profiles=moved["profiles"],
            profile_mask=moved["profile_mask"],
            row_embeddings=moved["row_embeddings"],
            row_mask=moved["row_mask"],
            msa_embeddings=moved["msa_embeddings"],
            msa_embedding_mask=moved["msa_embedding_mask"],
            numeric_values=moved["numeric_values"],
            numeric_mask=moved["numeric_mask"],
            category_ids=moved["category_ids"],
            category_mask=moved["category_mask"],
            target_tokens=moved["target_tokens"],
            loss_weights=moved["loss_weights"],
            sequence_loss_weights=moved["sequence_loss_weights"],
            target_continuous_embeddings=moved.get("target_continuous_embeddings"),
            target_continuous_mask=moved.get("target_continuous_mask"),
            timesteps=timesteps,
            decoder_start_mode="mean",
            memory_dropout=0.0,
            condition_mask_prob=0.0,
            profile_variable_mask=moved["profile_variable_mask"],
            profile_variable_dropout=0.0,
            profile_variable_blur=0.0,
            profile_blur_alpha=0.5,
        )
    residue_mask = moved["loss_weights"] > 0.5
    return {
        "token_loss": float(outputs["token_loss"].item()),
        "continuous_loss": float(outputs["weighted_continuous_loss"].item()),
        "residue_accuracy": weighted_residue_accuracy(outputs["logits"], moved["target_tokens"], moved["loss_weights"]),
        "consensus_accuracy": masked_residue_accuracy_value(
            outputs["logits"],
            moved["target_tokens"],
            moved["consensus_match_mask"] & moved["consensus_observed_mask"] & residue_mask,
        ),
        "nonconsensus_accuracy": masked_residue_accuracy_value(
            outputs["logits"],
            moved["target_tokens"],
            moved["nonconsensus_mask"] & residue_mask,
        ),
        "variable_nonconsensus_accuracy": masked_residue_accuracy_value(
            outputs["logits"],
            moved["target_tokens"],
            moved["variable_nonconsensus_mask"] & residue_mask,
        ),
    }


def gradient_saliency(
    model: MeanStartCCDDModel,
    batch: dict[str, Any],
    device: torch.device,
    profile_feature_mode: str,
) -> list[dict[str, float | str]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    moved = move_batch(batch, device)
    profiles = moved["profiles"].detach().clone().requires_grad_(True)
    batch_size = moved["target_tokens"].shape[0]
    timesteps = torch.zeros((batch_size,), dtype=torch.long, device=device)
    outputs = model(
        profiles=profiles,
        profile_mask=moved["profile_mask"],
        row_embeddings=moved["row_embeddings"],
        row_mask=moved["row_mask"],
        msa_embeddings=moved["msa_embeddings"],
        msa_embedding_mask=moved["msa_embedding_mask"],
        numeric_values=moved["numeric_values"],
        numeric_mask=moved["numeric_mask"],
        category_ids=moved["category_ids"],
        category_mask=moved["category_mask"],
        target_tokens=moved["target_tokens"],
        loss_weights=moved["loss_weights"],
        sequence_loss_weights=moved["sequence_loss_weights"],
        target_continuous_embeddings=moved.get("target_continuous_embeddings"),
        target_continuous_mask=moved.get("target_continuous_mask"),
        timesteps=timesteps,
        decoder_start_mode="mean",
        memory_dropout=0.0,
        condition_mask_prob=0.0,
        profile_variable_mask=moved["profile_variable_mask"],
        profile_variable_dropout=0.0,
        profile_variable_blur=0.0,
        profile_blur_alpha=0.5,
    )
    outputs["token_loss"].backward()
    attribution = (profiles.grad.detach().abs() * profiles.detach().abs()).cpu()
    valid = moved["profile_mask"].detach().cpu().unsqueeze(-1).to(dtype=attribution.dtype)
    variable = moved["profile_variable_mask"].detach().cpu().unsqueeze(-1).to(dtype=attribution.dtype)
    if profile_feature_mode == "full":
        groups = {
            "aa_frequency": attribution[:, :, : len(AA_TOKENS)].sum(dim=-1, keepdim=True),
            "gap_frequency": attribution[:, :, len(AA_TOKENS) : len(AA_TOKENS) + 1],
            "coverage": attribution[:, :, len(AA_TOKENS) + 1 : len(AA_TOKENS) + 2],
        }
    elif profile_feature_mode == "no_aa_frequency":
        groups = {
            "gap_frequency": attribution[:, :, 0:1],
            "coverage": attribution[:, :, 1:2],
        }
    else:
        raise ValueError(f"unknown profile_feature_mode: {profile_feature_mode}")
    total = sum(float((value * valid).sum().item()) for value in groups.values())
    rows: list[dict[str, float | str]] = []
    for name, value in groups.items():
        all_sum = float((value * valid).sum().item())
        variable_sum = float((value * valid * variable).sum().item())
        nonvariable_sum = float((value * valid * (1.0 - variable)).sum().item())
        rows.append(
            {
                "feature_group": name,
                "saliency_sum": all_sum,
                "saliency_fraction": all_sum / max(total, 1.0e-12),
                "variable_column_sum": variable_sum,
                "nonvariable_column_sum": nonvariable_sum,
            }
        )
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def svg_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_color(value: float, vmax: float) -> tuple[str, str]:
    t = 0.0 if vmax <= 0.0 else max(0.0, min(1.0, value / vmax))
    stops = [
        (248, 250, 252),
        (203, 213, 225),
        (56, 189, 248),
        (37, 99, 235),
        (30, 64, 175),
    ]
    scaled = t * (len(stops) - 1)
    idx = min(int(scaled), len(stops) - 2)
    frac = scaled - idx
    rgb = tuple(round(stops[idx][c] * (1.0 - frac) + stops[idx + 1][c] * frac) for c in range(3))
    fill = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    text = "#0f172a" if t < 0.45 else "#ffffff"
    return fill, text


def write_svg_outputs(
    out_dir: Path,
    attention_matrices: dict[str, np.ndarray],
    ablation_rows: list[dict[str, Any]],
    saliency_rows: list[dict[str, Any]],
) -> None:
    categories = ["condition", "profile_nonvariable", "profile_variable", "row"]
    for position_set, matrix in attention_matrices.items():
        cell_w, cell_h = 154, 54
        left, top = 164, 72
        width = left + cell_w * len(categories) + 36
        height = top + cell_h * matrix.shape[0] + 52
        vmax = max(0.001, float(matrix.max()) * 1.08)
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            f'<text x="24" y="30" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#0f172a">Cross-attention mass: {svg_escape(position_set)}</text>',
            '<text x="24" y="51" font-family="Inter,Arial,sans-serif" font-size="12" fill="#475569">Cells are mean attention probability mass per target residue, averaged over heads.</text>',
        ]
        for x, category in enumerate(categories):
            x_pos = left + x * cell_w + cell_w / 2
            parts.append(
                f'<text x="{x_pos:.1f}" y="{top - 12}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="#334155">{svg_escape(category)}</text>'
            )
        for y in range(matrix.shape[0]):
            y_pos = top + y * cell_h + cell_h / 2 + 4
            parts.append(
                f'<text x="{left - 18}" y="{y_pos:.1f}" text-anchor="end" font-family="Inter,Arial,sans-serif" font-size="12" font-weight="700" fill="#334155">L{y}</text>'
            )
            for x in range(len(categories)):
                value = float(matrix[y, x])
                fill, text = svg_color(value, vmax)
                rect_x = left + x * cell_w
                rect_y = top + y * cell_h
                parts.append(
                    f'<rect x="{rect_x}" y="{rect_y}" width="{cell_w - 2}" height="{cell_h - 2}" rx="5" fill="{fill}"/>'
                )
                parts.append(
                    f'<text x="{rect_x + cell_w / 2:.1f}" y="{rect_y + cell_h / 2 + 5:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="15" font-weight="700" fill="{text}">{value:.3f}</text>'
                )
        parts.append("</svg>")
        (out_dir / f"attention_memory_type_{position_set}.svg").write_text("\n".join(parts), encoding="utf-8")

    baseline = next(row for row in ablation_rows if row["variant"] == "baseline")
    ablated = [row for row in ablation_rows if row["variant"] != "baseline"]
    labels = [str(row["variant"]) for row in ablated]
    deltas = [float(row["token_loss"]) - float(baseline["token_loss"]) for row in ablated]
    width, height = 900, 420
    left, right, top, bottom = 96, 32, 58, 96
    plot_w, plot_h = width - left - right, height - top - bottom
    ymin = min(0.0, min(deltas) if deltas else 0.0)
    ymax = max(0.001, max(deltas) if deltas else 0.001)
    if ymax == ymin:
        ymax = ymin + 0.001
    zero_y = top + plot_h * (ymax / (ymax - ymin))
    bar_w = plot_w / max(len(deltas), 1) * 0.68
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="32" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#0f172a">Input ablation: token loss delta</text>',
        '<line x1="96" x2="868" y1="{:.1f}" y2="{:.1f}" stroke="#64748b" stroke-width="1"/>'.format(zero_y, zero_y),
    ]
    for i, (label, delta) in enumerate(zip(labels, deltas)):
        center = left + (i + 0.5) * plot_w / max(len(deltas), 1)
        y = top + plot_h * ((ymax - delta) / (ymax - ymin))
        rect_y = min(y, zero_y)
        rect_h = max(2.0, abs(zero_y - y))
        color = "#2563eb" if "profile_aa" in label else "#0f766e" if "profile" in label else "#b45309"
        parts.append(f'<rect x="{center - bar_w / 2:.1f}" y="{rect_y:.1f}" width="{bar_w:.1f}" height="{rect_h:.1f}" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{center:.1f}" y="{rect_y - 8:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" font-weight="700" fill="#0f172a">{delta:+.4f}</text>')
        parts.append(f'<text x="{center:.1f}" y="{height - 52}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="11" fill="#334155" transform="rotate(-18 {center:.1f} {height - 52})">{svg_escape(label)}</text>')
    parts.append('<text x="24" y="218" transform="rotate(-90 24 218)" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="#475569">loss increase vs baseline</text>')
    parts.append("</svg>")
    (out_dir / "profile_feature_ablation_loss_delta.svg").write_text("\n".join(parts), encoding="utf-8")

    saliency_labels = [str(row["feature_group"]) for row in saliency_rows]
    saliency_values = [float(row["saliency_fraction"]) for row in saliency_rows]
    width, height = 680, 360
    left, right, top, bottom = 78, 30, 58, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    vmax = max(1.0, max(saliency_values) * 1.1 if saliency_values else 1.0)
    colors = ["#2563eb", "#64748b", "#0f766e"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="32" font-family="Inter,Arial,sans-serif" font-size="18" font-weight="700" fill="#0f172a">Profile-channel saliency</text>',
        f'<line x1="{left}" x2="{width - right}" y1="{top + plot_h}" y2="{top + plot_h}" stroke="#64748b" stroke-width="1"/>',
    ]
    bar_w = plot_w / max(len(saliency_values), 1) * 0.54
    for i, (label, value) in enumerate(zip(saliency_labels, saliency_values)):
        center = left + (i + 0.5) * plot_w / max(len(saliency_values), 1)
        bar_h = plot_h * value / vmax
        rect_y = top + plot_h - bar_h
        parts.append(f'<rect x="{center - bar_w / 2:.1f}" y="{rect_y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="4" fill="{colors[i % len(colors)]}"/>')
        parts.append(f'<text x="{center:.1f}" y="{rect_y - 8:.1f}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="13" font-weight="700" fill="#0f172a">{value:.1%}</text>')
        parts.append(f'<text x="{center:.1f}" y="{height - 34}" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="#334155">{svg_escape(label)}</text>')
    parts.append('<text x="20" y="190" transform="rotate(-90 20 190)" text-anchor="middle" font-family="Inter,Arial,sans-serif" font-size="12" fill="#475569">fraction of |grad x input|</text>')
    parts.append("</svg>")
    (out_dir / "profile_channel_gradient_saliency.svg").write_text("\n".join(parts), encoding="utf-8")


def plot_outputs(
    out_dir: Path,
    attention_matrices: dict[str, np.ndarray],
    ablation_rows: list[dict[str, Any]],
    saliency_rows: list[dict[str, Any]],
    batch: dict[str, Any],
) -> None:
    meta = {
        "target_count": len(batch["target_sequences"]),
        "targets": [
            {
                "rank": idx + 1,
                "cluster_index": batch["cluster_indices"][idx],
                "kegg_entry": batch["kegg_entries"][idx],
                "row_index": int(batch["row_indices"][idx]),
                "target_length": len(batch["target_sequences"][idx]),
            }
            for idx in range(len(batch["target_sequences"]))
        ],
    }
    (out_dir / "batch_examples.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_svg_outputs(out_dir, attention_matrices, ablation_rows, saliency_rows)

    try:
        import matplotlib
    except ModuleNotFoundError:
        (out_dir / "plot_note.txt").write_text(
            "matplotlib is not installed in this environment, so SVG plots were written instead of PNG plots.\n",
            encoding="utf-8",
        )
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = ["condition", "profile_nonvariable", "profile_variable", "row"]
    for position_set, matrix in attention_matrices.items():
        fig, ax = plt.subplots(figsize=(7.4, 3.8))
        im = ax.imshow(matrix, vmin=0.0, vmax=max(0.001, float(matrix.max()) * 1.1), cmap="viridis")
        ax.set_xticks(range(len(categories)), labels=categories, rotation=30, ha="right")
        ax.set_yticks(range(matrix.shape[0]), labels=[f"L{i}" for i in range(matrix.shape[0])])
        ax.set_title(f"Cross-attention mass by memory type: {position_set}")
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                ax.text(x, y, f"{matrix[y, x]:.2f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / f"attention_memory_type_{position_set}.png", dpi=180)
        plt.close(fig)

    baseline = next(row for row in ablation_rows if row["variant"] == "baseline")
    labels = [row["variant"] for row in ablation_rows if row["variant"] != "baseline"]
    deltas = [float(row["token_loss"]) - float(baseline["token_loss"]) for row in ablation_rows if row["variant"] != "baseline"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = ["#2563eb" if "profile_aa" in label else "#0f766e" if "profile" in label else "#b45309" for label in labels]
    ax.bar(labels, deltas, color=colors)
    ax.axhline(0, color="#475569", linewidth=1)
    ax.set_ylabel("token loss increase vs baseline")
    ax.set_title("Input ablations on same held-out batch")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / "profile_feature_ablation_loss_delta.png", dpi=180)
    plt.close(fig)

    saliency_labels = [str(row["feature_group"]) for row in saliency_rows]
    saliency_values = [float(row["saliency_fraction"]) for row in saliency_rows]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.bar(saliency_labels, saliency_values, color=["#2563eb", "#64748b", "#0f766e"])
    ax.set_ylim(0.0, max(1.0, max(saliency_values) * 1.15))
    ax.set_ylabel("fraction of |grad x input|")
    ax.set_title("Profile-channel saliency")
    fig.tight_layout()
    fig.savefig(out_dir / "profile_channel_gradient_saliency.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding-manifest", default=str(DEFAULT_EMBEDDING_MANIFEST))
    parser.add_argument("--label-summary", default=str(DEFAULT_LABEL_SUMMARY))
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-msas", type=int, default=256)
    parser.add_argument("--cache-size", type=int, default=64)
    parser.add_argument("--msa-group-batch-size", type=int, default=1)
    parser.add_argument("--example-batch-size", type=int, default=4)
    parser.add_argument("--max-flat-examples", type=int, default=5)
    parser.add_argument("--force-single-row", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    step = checkpoint.get("step", "unknown")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else checkpoint_path.parent / f"attention_feature_diagnostics_step{step}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    batch, row_embedding_dim, msa_embedding_dim, target_continuous_dim = load_val_batch(args, checkpoint)
    model = build_model(
        checkpoint,
        row_embedding_dim=row_embedding_dim,
        msa_embedding_dim=msa_embedding_dim,
        target_continuous_dim=target_continuous_dim,
        device=device,
    )
    moved_batch = move_batch(batch, device)
    capture = CrossAttentionCapture(model)
    capture.clear()
    _ = forward_metrics(model, moved_batch, device)
    capture.restore()
    attention_rows, attention_matrices = summarize_attention(capture.records, batch)

    profile_feature_mode = str(checkpoint.get("config", {}).get("profile_feature_mode", "full"))
    variants = ["baseline"]
    if profile_feature_mode == "full":
        variants.extend(["uniform_profile_aa", "zero_profile_aa"])
    else:
        variants.extend(["zero_gap_frequency", "zero_coverage"])
    variants.extend(["remove_profile_tokens", "remove_row_tokens", "blank_conditions"])
    ablation_rows: list[dict[str, Any]] = []
    for variant in variants:
        variant_batch = clone_batch_with_variant(batch, variant)
        metrics = forward_metrics(model, variant_batch, device)
        ablation_rows.append({"variant": variant, **metrics})

    saliency_rows = gradient_saliency(model, batch, device, profile_feature_mode)

    write_tsv(out_dir / "cross_attention_memory_type.tsv", attention_rows)
    write_tsv(out_dir / "input_ablation_metrics.tsv", ablation_rows)
    write_tsv(out_dir / "profile_channel_gradient_saliency.tsv", saliency_rows)
    plot_outputs(out_dir, attention_matrices, ablation_rows, saliency_rows, batch)

    print(f"checkpoint={checkpoint_path}")
    print(f"checkpoint_step={step}")
    print(f"device={device}")
    print(f"examples={len(batch['target_sequences'])}")
    print(f"out_dir={out_dir}")
    print("ablation_metrics=" + str(out_dir / "input_ablation_metrics.tsv"))
    print("attention_metrics=" + str(out_dir / "cross_attention_memory_type.tsv"))
    print("saliency_metrics=" + str(out_dir / "profile_channel_gradient_saliency.tsv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
