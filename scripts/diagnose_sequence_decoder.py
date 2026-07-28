#!/usr/bin/env python3
"""Run residue-only diagnostics and length-forced samples for a sequence decoder."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msa_design_model import (  # noqa: E402
    ID_TO_TOKEN,
    MASK_TOKEN_ID,
    MSASequenceDiffusionModel,
    STOP_TOKEN_ID,
)
from train_sequence_decoder import (  # noqa: E402
    SequenceCollator,
    SequenceEmbeddingDataset,
    find_examples,
    parse_categorical_fields,
    parse_numeric_fields,
)


GAP_CHARS = {"-", ".", " ", "\n", "\r", "\t"}
DEFAULT_CASES = (
    "good R03817 row21|ec_7_1_1_6__rxn_R03817|21",
    "bad R07092 row5|ec_2_5_1_18__rxn_R07092|5",
    "bad R07003 row4|ec_2_5_1_18__rxn_R07003|4",
    "bad R07094 row38|ec_2_5_1_18__rxn_R07094|38",
)


@dataclass(frozen=True)
class FamilyAlignment:
    family_id: str
    aligned_sequences: tuple[str, ...]
    aa_mask: np.ndarray
    gap_mask: np.ndarray
    rows: int
    cols: int
    overall_gap_frac: float
    mean_col_conservation: float


@dataclass(frozen=True)
class TargetColumnStats:
    columns: tuple[int, ...]
    residues: tuple[str, ...]
    other_gap: np.ndarray
    support: np.ndarray
    other_conservation: np.ndarray
    consensus: tuple[str | None, ...]
    consensus_match: np.ndarray


class IndexedSubset(Dataset[dict[str, Any]]):
    def __init__(self, dataset: Dataset[dict[str, Any]], indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset_index = int(self.indices[index])
        item = dict(self.dataset[dataset_index])
        item["dataset_index"] = dataset_index
        return item


class DiagnosticCollator:
    def __init__(self, base: SequenceCollator) -> None:
        self.base = base

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        dataset_indices = [int(item["dataset_index"]) for item in batch]
        stripped = []
        for item in batch:
            copied = dict(item)
            copied.pop("dataset_index", None)
            stripped.append(copied)
        collated = self.base(stripped)
        collated["dataset_indices"] = dataset_indices
        return collated


def finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_family_alignment(embedding_path: Path) -> FamilyAlignment:
    arrays = np.load(embedding_path)
    aa_mask = arrays["aa_mask"].astype(np.bool_)
    gap_mask = arrays["gap_mask"].astype(np.bool_) if "gap_mask" in arrays.files else ~aa_mask
    sidecar = embedding_path.with_suffix(".metadata.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    aligned_sequences = tuple(str(sequence).upper() for sequence in metadata["cleaned_sequences"])
    rows, cols = aa_mask.shape
    col_conservations: list[float] = []
    for col in range(cols):
        residues = [
            aligned_sequences[row][col]
            for row in range(min(rows, len(aligned_sequences)))
            if col < len(aligned_sequences[row]) and aligned_sequences[row][col] not in GAP_CHARS
        ]
        if residues:
            counts = Counter(residues)
            col_conservations.append(max(counts.values()) / len(residues))
    return FamilyAlignment(
        family_id=embedding_path.stem,
        aligned_sequences=aligned_sequences,
        aa_mask=aa_mask,
        gap_mask=gap_mask,
        rows=rows,
        cols=cols,
        overall_gap_frac=float(gap_mask.mean()),
        mean_col_conservation=finite_mean(col_conservations),
    )


def target_column_stats(family: FamilyAlignment, row_index: int, target_len: int) -> TargetColumnStats:
    aligned = family.aligned_sequences[row_index]
    columns: list[int] = []
    residues: list[str] = []
    for col, residue in enumerate(aligned):
        if residue in GAP_CHARS:
            continue
        columns.append(col)
        residues.append(residue)
        if len(columns) >= target_len:
            break

    other_gap: list[float] = []
    support: list[float] = []
    other_conservation: list[float] = []
    consensus: list[str | None] = []
    consensus_match: list[float] = []
    denom = max(family.rows - 1, 1)
    for col, target_residue in zip(columns, residues):
        others = []
        for other_row, sequence in enumerate(family.aligned_sequences[: family.rows]):
            if other_row == row_index or col >= len(sequence):
                continue
            residue = sequence[col]
            if residue not in GAP_CHARS:
                others.append(residue)
        support_count = len(others)
        support.append(float(support_count))
        other_gap.append(1.0 - support_count / denom)
        if support_count:
            counts = Counter(others)
            residue, count = counts.most_common(1)[0]
            consensus.append(residue)
            other_conservation.append(count / support_count)
            consensus_match.append(1.0 if target_residue == residue else 0.0)
        else:
            consensus.append(None)
            other_conservation.append(float("nan"))
            consensus_match.append(float("nan"))

    return TargetColumnStats(
        columns=tuple(columns),
        residues=tuple(residues),
        other_gap=np.array(other_gap, dtype=np.float32),
        support=np.array(support, dtype=np.float32),
        other_conservation=np.array(other_conservation, dtype=np.float32),
        consensus=tuple(consensus),
        consensus_match=np.array(consensus_match, dtype=np.float32),
    )


def build_model(config: dict[str, Any], device: torch.device) -> MSASequenceDiffusionModel:
    model = MSASequenceDiffusionModel(
        input_dim=int(config["input_dim"]),
        d_model=int(config["d_model"]),
        max_sequence_length=int(config["max_sequence_length"]),
        num_layers=int(config["layers"]),
        num_heads=int(config["heads"]),
        dropout=float(config["dropout"]),
        num_timesteps=int(config["diffusion_timesteps"]),
        numeric_condition_fields=tuple(config["numeric_condition_fields"]),
        categorical_condition_fields=tuple(config["categorical_condition_fields"]),
        categorical_vocab_sizes=[len(vocab) for vocab in config["categorical_vocabs"]],
        condition_layers=int(config["condition_layers"]),
        latent_codiffusion_tokens=int(config["latent_codiffusion_tokens"]),
        ccdd_mode=str(config["ccdd_mode"]),
    )
    return model.to(device)


def make_dataset(args: argparse.Namespace, config: dict[str, Any]) -> tuple[SequenceEmbeddingDataset, list[Any]]:
    numeric_fields = parse_numeric_fields(",".join(config["numeric_condition_fields"]))
    categorical_fields = parse_categorical_fields(",".join(config["categorical_condition_fields"]))
    examples = find_examples(
        embeddings_dir=Path(args.embeddings_dir),
        metadata_dir=Path(args.metadata_dir),
        embedding_glob=args.embedding_glob,
        max_examples=args.max_examples,
        numeric_fields=numeric_fields,
        categorical_fields=categorical_fields,
        value_aggregation=str(config.get("value_aggregation", "mean")),
        condition_transform=str(config.get("condition_transform", "auto")),
        require_status_ok=not args.allow_non_ok_status,
    )
    if not examples:
        raise SystemExit("No examples found for diagnostics")
    dataset = SequenceEmbeddingDataset(
        examples=examples,
        numeric_means=np.array(config["numeric_condition_mean"], dtype=np.float32),
        numeric_stds=np.array(config["numeric_condition_std"], dtype=np.float32),
        categorical_vocabs=list(config["categorical_vocabs"]),
    )
    return dataset, examples


def batch_model_inputs(batch: dict[str, Any], config: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    inputs: dict[str, torch.Tensor] = {
        "token_embeddings": batch["token_embeddings"].to(device),
        "aa_mask": batch["aa_mask"].to(device),
        "target_tokens": batch["target_tokens"].to(device),
        "loss_weights": batch["loss_weights"].to(device),
    }
    if config["numeric_condition_fields"]:
        inputs["condition_values"] = batch["condition_values"].to(device)
        inputs["condition_mask"] = batch["condition_mask"].to(device)
    if config["categorical_condition_fields"]:
        inputs["categorical_condition_ids"] = batch["categorical_condition_ids"].to(device)
        inputs["categorical_condition_mask"] = batch["categorical_condition_mask"].to(device)
    if config["ccdd_mode"] != "off":
        inputs["target_continuous_embeddings"] = batch["target_continuous_embeddings"].to(device)
        inputs["target_continuous_mask"] = batch["target_continuous_mask"].to(device)
    return inputs


def evaluate_timestep(
    model: MSASequenceDiffusionModel,
    loader: DataLoader[dict[str, Any]],
    examples: list[Any],
    config: dict[str, Any],
    device: torch.device,
    timestep: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    torch.manual_seed(seed + timestep)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + timestep)
    family_cache: dict[Path, FamilyAlignment] = {}
    target_cache: dict[tuple[Path, int, int], TargetColumnStats] = {}
    rows: list[dict[str, Any]] = []
    bin_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    model.eval()
    local_index = 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch_model_inputs(batch, config, device)
            target_tokens = inputs["target_tokens"]
            timesteps = torch.full((target_tokens.shape[0],), timestep, dtype=torch.long, device=device)
            outputs = model(
                **inputs,
                timesteps=timesteps,
                decoder_start_mode=str(config["decoder_start_mode"]),
                decoder_token_dropout=0.0,
                decoder_span_mask_fraction=0.0,
                decoder_span_mask_length=int(config["decoder_span_mask_length"]),
                discrete_loss_corrupted_only=True,
                condition_dropout=0.0,
                ccdd_continuous_timestep_scale=float(config["ccdd_continuous_timestep_scale"]),
                ccdd_continuous_dropout=0.0,
            )
            predicted = torch.argmax(outputs["logits"], dim=-1).detach().cpu().numpy()
            target = target_tokens.detach().cpu().numpy()
            corruption = outputs["corruption_mask"].detach().cpu().numpy().astype(bool)

            for batch_index, dataset_index in enumerate(batch["dataset_indices"]):
                example = examples[dataset_index]
                target_len = min(len(example.target_sequence), int(config["max_sequence_length"]) - 1)
                residue_slice = slice(0, target_len)
                residue_correct = predicted[batch_index, residue_slice] == target[batch_index, residue_slice]
                residue_corruption = corruption[batch_index, residue_slice]
                masked_count = int(residue_corruption.sum())
                masked_correct = int((residue_correct & residue_corruption).sum())
                full_correct = int(residue_correct.sum())
                family = family_cache.get(example.embedding_path)
                if family is None:
                    family = load_family_alignment(example.embedding_path)
                    family_cache[example.embedding_path] = family
                target_key = (example.embedding_path, int(example.row_index), target_len)
                stats = target_cache.get(target_key)
                if stats is None:
                    stats = target_column_stats(family, int(example.row_index), target_len)
                    target_cache[target_key] = stats
                for pos, is_masked in enumerate(residue_corruption):
                    if not is_masked:
                        continue
                    correct = bool(residue_correct[pos])
                    gap_value = float(stats.other_gap[pos]) if pos < len(stats.other_gap) else float("nan")
                    cons_value = (
                        float(stats.other_conservation[pos])
                        if pos < len(stats.other_conservation)
                        else float("nan")
                    )
                    for kind, value, bins in (
                        ("gap", gap_value, ((0.0, 0.25), (0.25, 0.50), (0.50, 0.70), (0.70, 0.90), (0.90, 1.01))),
                        ("cons", cons_value, ((0.0, 0.40), (0.40, 0.60), (0.60, 0.80), (0.80, 1.01))),
                    ):
                        label = "nan"
                        if math.isfinite(value):
                            for low, high in bins:
                                if low <= value < high:
                                    label = f"{low:.2f}-{high:.2f}"
                                    break
                        bucket = bin_counts[f"{kind}\t{label}"]
                        bucket["n"] += 1
                        bucket["correct"] += int(correct)

                rows.append(
                    {
                        "local_index": local_index,
                        "dataset_index": dataset_index,
                        "family_id": family.family_id,
                        "row_index": int(example.row_index),
                        "target_len": target_len,
                        "family_rows": family.rows,
                        "family_cols": family.cols,
                        "family_overall_gap_frac": family.overall_gap_frac,
                        "family_mean_col_conservation": family.mean_col_conservation,
                        "target_mean_other_gap": finite_mean(stats.other_gap.tolist()),
                        "target_high_gap_frac": finite_mean((stats.other_gap > 0.50).astype(float).tolist()),
                        "target_low_support_frac": finite_mean((stats.support < 10.0).astype(float).tolist()),
                        "target_mean_support": finite_mean(stats.support.tolist()),
                        "target_mean_other_conservation": finite_mean(stats.other_conservation.tolist()),
                        "target_consensus_match": finite_mean(stats.consensus_match.tolist()),
                        "eval_timestep": timestep,
                        "residue_corruption_fraction": masked_count / max(target_len, 1),
                        "masked_residue_accuracy": masked_correct / masked_count if masked_count else float("nan"),
                        "full_residue_accuracy": full_correct / max(target_len, 1),
                    }
                )
                local_index += 1

    summary: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["family_id"])].append(row)
    for family_id, family_rows in grouped.items():
        summary[family_id] = {
            "family_id": family_id,
            "n_val": len(family_rows),
            "mean_masked_residue_accuracy": finite_mean([row["masked_residue_accuracy"] for row in family_rows]),
            "max_masked_residue_accuracy": max(float(row["masked_residue_accuracy"]) for row in family_rows),
            "min_masked_residue_accuracy": min(float(row["masked_residue_accuracy"]) for row in family_rows),
            "mean_full_residue_accuracy": finite_mean([row["full_residue_accuracy"] for row in family_rows]),
            "mean_residue_corruption_fraction": finite_mean([row["residue_corruption_fraction"] for row in family_rows]),
            "family_rows": family_rows[0]["family_rows"],
            "family_cols": family_rows[0]["family_cols"],
            "family_overall_gap_frac": family_rows[0]["family_overall_gap_frac"],
            "family_mean_col_conservation": family_rows[0]["family_mean_col_conservation"],
            "mean_target_other_gap": finite_mean([row["target_mean_other_gap"] for row in family_rows]),
            "mean_target_high_gap_frac": finite_mean([row["target_high_gap_frac"] for row in family_rows]),
            "mean_target_low_support_frac": finite_mean([row["target_low_support_frac"] for row in family_rows]),
            "mean_target_support": finite_mean([row["target_mean_support"] for row in family_rows]),
            "mean_target_other_conservation": finite_mean(
                [row["target_mean_other_conservation"] for row in family_rows]
            ),
            "mean_target_consensus_match": finite_mean([row["target_consensus_match"] for row in family_rows]),
        }
    return rows, summary, bin_counts


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_case(text: str) -> tuple[str, str, int]:
    parts = text.split("|")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("cases must be LABEL|FAMILY_ID|ROW_INDEX")
    return parts[0], parts[1], int(parts[2])


def encode_sample_string(tokens: np.ndarray, length: int) -> str:
    residues: list[str] = []
    for token_id in tokens[:length]:
        token = ID_TO_TOKEN[int(token_id)]
        if token in {"<MASK>", "*"}:
            continue
        residues.append(token)
    return "".join(residues)


@torch.no_grad()
def length_forced_sample(
    model: MSASequenceDiffusionModel,
    batch: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    target_len: int,
    steps: int,
    temperature: float,
) -> np.ndarray:
    inputs = batch_model_inputs(batch, config, device)
    memory = model.encode_latent_memory(
        token_embeddings=inputs["token_embeddings"],
        aa_mask=inputs["aa_mask"],
        condition_values=inputs.get("condition_values"),
        condition_mask=inputs.get("condition_mask"),
        categorical_condition_ids=inputs.get("categorical_condition_ids"),
        categorical_condition_mask=inputs.get("categorical_condition_mask"),
    )
    batch_size = inputs["token_embeddings"].shape[0]
    if batch_size != 1:
        raise ValueError("length_forced_sample expects batch size 1")
    tokens = torch.full(
        (1, int(config["max_sequence_length"])),
        MASK_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    if target_len < tokens.shape[1]:
        tokens[:, target_len:] = STOP_TOKEN_ID
    schedule = torch.linspace(
        int(config["diffusion_timesteps"]) - 1,
        0,
        steps,
        device=device,
    ).round().long()
    logits = torch.empty(1, tokens.shape[1], len(ID_TO_TOKEN), device=device)
    confidence = torch.zeros(tokens.shape, dtype=torch.float32, device=device)
    for step_index, timestep in enumerate(schedule):
        timestep_batch = timestep.expand(1)
        embeddings = model.decoder.token_embedding(tokens)
        predicted = model.decoder.denoise(
            embeddings,
            timestep_batch,
            memory["latent_tokens"],
            memory["latent_mask"],
        )
        logits = model.decoder.lm_head(predicted)
        logits[..., MASK_TOKEN_ID] = -torch.inf
        logits[:, :target_len, STOP_TOKEN_ID] = -torch.inf
        if target_len < tokens.shape[1]:
            logits[:, target_len:, :] = -torch.inf
            logits[:, target_len:, STOP_TOKEN_ID] = 0.0
        if temperature == 0.0:
            probabilities = torch.softmax(logits, dim=-1)
            sampled = torch.argmax(probabilities, dim=-1)
        else:
            probabilities = torch.softmax(logits / temperature, dim=-1)
            sampled = torch.multinomial(probabilities.reshape(-1, probabilities.shape[-1]), 1).reshape_as(tokens)
        confidence = probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        if step_index + 1 == len(schedule):
            tokens = sampled
            break
        next_timestep = schedule[step_index + 1]
        mask_fraction = float(model.decoder.discrete_corruption_probability(next_timestep).item())
        next_mask_count = min(target_len, int(round(target_len * mask_fraction)))
        tokens = sampled.clone()
        if target_len < tokens.shape[1]:
            tokens[:, target_len:] = STOP_TOKEN_ID
        if next_mask_count > 0:
            remask_indices = torch.topk(confidence[0, :target_len], k=next_mask_count, largest=False).indices
            tokens[0, remask_indices] = MASK_TOKEN_ID
    return tokens[0].detach().cpu().numpy()


def run_length_forced_cases(
    model: MSASequenceDiffusionModel,
    dataset: SequenceEmbeddingDataset,
    examples: list[Any],
    config: dict[str, Any],
    device: torch.device,
    cases: list[tuple[str, str, int]],
    steps: int,
    temperature: float,
    seed: int,
) -> list[dict[str, Any]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    collator = DiagnosticCollator(
        SequenceCollator(
            max_sequence_length=int(config["max_sequence_length"]),
            tail_stop_weight=float(config["tail_stop_weight"]),
            mask_target_row_in_msa=bool(config["mask_target_row_in_msa"]),
        )
    )
    family_cache: dict[Path, FamilyAlignment] = {}
    rows: list[dict[str, Any]] = []
    for label, family_id, row_index in cases:
        dataset_index = next(
            (
                idx
                for idx, example in enumerate(examples)
                if example.embedding_path.stem == family_id and int(example.row_index) == row_index
            ),
            None,
        )
        if dataset_index is None:
            raise SystemExit(f"Could not find case {label}: {family_id} row {row_index}")
        subset = IndexedSubset(dataset, [dataset_index])
        batch = collator([subset[0]])
        example = examples[dataset_index]
        target_len = min(len(example.target_sequence), int(config["max_sequence_length"]) - 1)
        sampled_tokens = length_forced_sample(
            model=model,
            batch=batch,
            config=config,
            device=device,
            target_len=target_len,
            steps=steps,
            temperature=temperature,
        )
        generated = encode_sample_string(sampled_tokens, target_len)
        target = example.target_sequence[:target_len]
        compare_len = min(len(generated), target_len)
        identity = (
            sum(1 for idx in range(compare_len) if generated[idx] == target[idx]) / target_len
            if target_len
            else float("nan")
        )
        ag_fraction = (
            sum(1 for residue in generated[:target_len] if residue in {"A", "G"}) / max(len(generated[:target_len]), 1)
        )
        family = family_cache.get(example.embedding_path)
        if family is None:
            family = load_family_alignment(example.embedding_path)
            family_cache[example.embedding_path] = family
        stats = target_column_stats(family, int(example.row_index), target_len)
        rows.append(
            {
                "label": label,
                "family_id": family_id,
                "row_index": row_index,
                "target_len": target_len,
                "generated_identity": identity,
                "generated_ag_fraction": ag_fraction,
                "mean_gap": finite_mean(stats.other_gap.tolist()),
                "high_gap_fraction": finite_mean((stats.other_gap > 0.50).astype(float).tolist()),
                "mean_conservation": finite_mean(stats.other_conservation.tolist()),
                "target_consensus_match": finite_mean(stats.consensus_match.tolist()),
                "target": target,
                "generated": generated,
            }
        )
    return rows


def plot_alignment_cases(rows: list[dict[str, Any]], path_png: Path, path_svg: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"warning: could not import matplotlib for alignment plot: {exc}", file=sys.stderr)
        write_alignment_svg(rows, path_svg)
        return
    fig, axes = plt.subplots(len(rows), 1, figsize=(14, max(2.0, 1.9 * len(rows))), sharex=False)
    if len(rows) == 1:
        axes = [axes]
    for axis, row in zip(axes, rows):
        target = row["target"]
        generated = row["generated"]
        xs = np.arange(len(target))
        correct = np.array([idx < len(generated) and generated[idx] == residue for idx, residue in enumerate(target)])
        colors = np.where(correct, "#2f9e44", "#c92a2a")
        axis.scatter(xs, np.zeros_like(xs), c=colors, s=8, linewidths=0)
        axis.set_yticks([])
        axis.set_ylabel(row["label"], rotation=0, ha="right", va="center", fontsize=8)
        axis.set_title(
            f"identity={float(row['generated_identity']):.3f} A/G={float(row['generated_ag_fraction']):.3f} "
            f"gap={float(row['mean_gap']):.3f} consensus={float(row['target_consensus_match']):.3f}",
            fontsize=9,
        )
        axis.set_xlim(-1, max(len(target), 1))
    axes[-1].set_xlabel("target residue position")
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=160)
    fig.savefig(path_svg)
    plt.close(fig)


def write_alignment_svg(rows: list[dict[str, Any]], path_svg: Path) -> None:
    width = 1280
    left = 220
    right = 40
    plot_width = width - left - right
    row_height = 86
    height = 34 + row_height * len(rows)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;font-size:13px}.small{font-size:11px;fill:#444}</style>',
    ]
    for row_idx, row in enumerate(rows):
        y = 36 + row_idx * row_height
        target = str(row["target"])
        generated = str(row["generated"])
        title = (
            f"{row['label']}  identity={float(row['generated_identity']):.3f} "
            f"A/G={float(row['generated_ag_fraction']):.3f} "
            f"gap={float(row['mean_gap']):.3f} consensus={float(row['target_consensus_match']):.3f}"
        )
        parts.append(f'<text x="12" y="{y}" class="small">{escape(title)}</text>')
        parts.append(f'<line x1="{left}" y1="{y + 20}" x2="{width - right}" y2="{y + 20}" stroke="#ddd"/>')
        bar_width = max(1.0, plot_width / max(len(target), 1))
        for pos, residue in enumerate(target):
            x = left + (pos / max(len(target), 1)) * plot_width
            correct = pos < len(generated) and generated[pos] == residue
            color = "#2f9e44" if correct else "#c92a2a"
            parts.append(
                f'<rect x="{x:.2f}" y="{y + 10}" width="{bar_width:.2f}" height="20" fill="{color}" opacity="0.9"/>'
            )
    parts.append("</svg>")
    path_svg.parent.mkdir(parents=True, exist_ok=True)
    path_svg.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main(args: argparse.Namespace) -> int:
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = Path(args.checkpoint)
    payload = load_checkpoint(checkpoint, device)
    config = payload["config"]
    dataset, examples = make_dataset(args, config)
    val_indices = [int(index) for index in config["val_indices"]]
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint.parent.parent / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config, device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    collator = DiagnosticCollator(
        SequenceCollator(
            max_sequence_length=int(config["max_sequence_length"]),
            tail_stop_weight=float(config["tail_stop_weight"]),
            mask_target_row_in_msa=bool(config["mask_target_row_in_msa"]),
        )
    )
    loader = DataLoader(
        IndexedSubset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    for timestep in args.eval_timestep:
        example_rows, summary, bin_counts = evaluate_timestep(
            model=model,
            loader=loader,
            examples=examples,
            config=config,
            device=device,
            timestep=timestep,
            seed=args.seed,
        )
        example_fields = [
            "local_index",
            "dataset_index",
            "family_id",
            "row_index",
            "target_len",
            "family_rows",
            "family_cols",
            "family_overall_gap_frac",
            "family_mean_col_conservation",
            "target_mean_other_gap",
            "target_high_gap_frac",
            "target_low_support_frac",
            "target_mean_support",
            "target_mean_other_conservation",
            "target_consensus_match",
            "eval_timestep",
            "residue_corruption_fraction",
            "masked_residue_accuracy",
            "full_residue_accuracy",
        ]
        write_tsv(
            output_dir / f"prediction_family_residue_examples_t{timestep}.tsv",
            example_rows,
            example_fields,
        )
        summary_fields = [
            "family_id",
            "n_val",
            "mean_masked_residue_accuracy",
            "max_masked_residue_accuracy",
            "min_masked_residue_accuracy",
            "mean_full_residue_accuracy",
            "mean_residue_corruption_fraction",
            "family_rows",
            "family_cols",
            "family_overall_gap_frac",
            "family_mean_col_conservation",
            "mean_target_other_gap",
            "mean_target_high_gap_frac",
            "mean_target_low_support_frac",
            "mean_target_support",
            "mean_target_other_conservation",
            "mean_target_consensus_match",
        ]
        write_tsv(
            output_dir / f"prediction_family_residue_summary_t{timestep}.tsv",
            [summary[key] for key in sorted(summary)],
            summary_fields,
        )
        if timestep == args.primary_timestep:
            bin_rows = []
            for combined_key in sorted(bin_counts):
                kind, label = combined_key.split("\t", 1)
                counts = bin_counts[combined_key]
                bin_rows.append(
                    {
                        "kind": kind,
                        "bin": label,
                        "n": counts["n"],
                        "correct": counts["correct"],
                        "accuracy": counts["correct"] / counts["n"] if counts["n"] else float("nan"),
                    }
                )
            write_tsv(
                output_dir / f"prediction_gap_conservation_bins_t{timestep}.tsv",
                bin_rows,
                ["kind", "bin", "n", "correct", "accuracy"],
            )
        overall_acc = finite_mean([row["masked_residue_accuracy"] for row in example_rows])
        print(
            f"timestep={timestep} examples={len(example_rows)} "
            f"mean_masked_residue_accuracy={overall_acc:.6f}",
            flush=True,
        )

    cases = [parse_case(case) for case in args.case]
    sample_rows = run_length_forced_cases(
        model=model,
        dataset=dataset,
        examples=examples,
        config=config,
        device=device,
        cases=cases,
        steps=args.sample_steps,
        temperature=args.sample_temperature,
        seed=args.seed + 1000,
    )
    sample_path = output_dir / "length_forced_alignment_error_cases.tsv"
    write_tsv(
        sample_path,
        sample_rows,
        [
            "label",
            "family_id",
            "row_index",
            "target_len",
            "generated_identity",
            "generated_ag_fraction",
            "mean_gap",
            "high_gap_fraction",
            "mean_conservation",
            "target_consensus_match",
            "target",
            "generated",
        ],
    )
    plot_alignment_cases(
        sample_rows,
        output_dir / "ccdd_lite_alignment_error_cases.png",
        output_dir / "ccdd_lite_alignment_error_cases.svg",
    )
    for row in sample_rows:
        print(
            f"sample={row['label']} identity={float(row['generated_identity']):.6f} "
            f"ag_fraction={float(row['generated_ag_fraction']):.6f}",
            flush=True,
        )
    print(f"wrote_diagnostics={output_dir}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embeddings-dir", default="outputs/training/okay24_20260713_233827/embeddings")
    parser.add_argument("--metadata-dir", default="outputs/training/okay24_20260713_233827/metadata")
    parser.add_argument("--embedding-glob", default="ec_*.npz")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--allow-non-ok-status", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--eval-timestep", type=int, action="append", default=[224, 124])
    parser.add_argument("--primary-timestep", type=int, default=224)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--sample-temperature", type=float, default=0.0)
    parser.add_argument("--case", action="append", default=list(DEFAULT_CASES))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
