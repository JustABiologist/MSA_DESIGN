#!/usr/bin/env python3
"""Smoke-train a fixed-length sequence diffusion decoder from frozen MSA embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msa_design_model import (  # noqa: E402
    DEFAULT_CATEGORICAL_FIELDS,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_NUMERIC_FIELDS,
    MASK_TOKEN,
    MASK_TOKEN_ID,
    MSASequenceDiffusionModel,
    SEQUENCE_TOKENS,
    STOP_TOKEN,
    batch_encode_sequences_with_stop,
    decode_tokens_until_stop,
)


FIELD_VALUE_COLUMNS: dict[str, tuple[str, ...]] = {
    "kcat_1_per_s": ("kcat_1_per_s_values", "numeric_col_7_unlabeled_values"),
    "km_mM": ("km_mM_values", "numeric_col_8_unlabeled_values"),
    "kcat_over_km_1_per_mM_s": (
        "kcat_over_km_1_per_mM_s_values",
        "numeric_col_9_unlabeled_values",
    ),
    "topt_C": ("topt_C_values", "numeric_col_10_unlabeled_values"),
    "tm_C": ("tm_C_values", "numeric_col_11_unlabeled_values"),
}
LOG_DEFAULT_FIELDS = {"kcat_1_per_s", "km_mM", "kcat_over_km_1_per_mM_s"}


@dataclass(frozen=True)
class SequenceExample:
    embedding_path: Path
    metadata_path: Path | None
    row_index: int
    target_sequence: str
    numeric_values: np.ndarray
    numeric_mask: np.ndarray
    categorical_values: tuple[tuple[str, ...], ...]


def ungap_sequence(sequence: str) -> str:
    return "".join(char for char in sequence.upper() if char not in {"-", ".", " ", "\n", "\r", "\t"})


def split_value_list(text: str) -> Iterable[str]:
    for part in str(text).replace(",", ";").split(";"):
        part = part.strip()
        if part:
            yield part


def parse_finite_values(text: str) -> list[float]:
    values: list[float] = []
    for part in split_value_list(text):
        try:
            value = float(part)
        except ValueError:
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def aggregate(values: list[float], method: str) -> float | None:
    if not values:
        return None
    if method == "first":
        return values[0]
    if method == "mean":
        return float(sum(values) / len(values))
    raise ValueError(f"unknown aggregation method: {method}")


def metadata_numeric_value(row: dict[str, str], field: str, method: str) -> float | None:
    for column in FIELD_VALUE_COLUMNS[field]:
        if column in row:
            value = aggregate(parse_finite_values(row.get(column, "")), method)
            if value is not None:
                return value
    return None


def transform_numeric_condition(value: float, field: str, mode: str) -> float:
    if mode == "identity" or (mode == "auto" and field not in LOG_DEFAULT_FIELDS):
        return value
    if mode == "log10" or mode == "auto":
        if value <= 0:
            raise ValueError(f"cannot log-transform non-positive condition value {value}")
        return math.log10(value)
    raise ValueError(f"unknown condition transform: {mode}")


def metadata_categories(row: dict[str, str], field: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(split_value_list(row.get(field, ""))))
    return values


def read_metadata_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_fasta_sequences(path: Path) -> list[str]:
    sequences: list[str] = []
    sequence_parts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sequence_parts:
                    sequences.append(ungap_sequence("".join(sequence_parts)))
                    sequence_parts = []
                continue
            sequence_parts.append(line)
    if sequence_parts:
        sequences.append(ungap_sequence("".join(sequence_parts)))
    return sequences


def target_sequences_from_sidecar(embedding_path: Path) -> list[str]:
    sidecar = embedding_path.with_suffix(".metadata.json")
    if not sidecar.exists():
        raise FileNotFoundError(f"no metadata sidecar found next to {embedding_path}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    cleaned_sequences = metadata.get("cleaned_sequences") or []
    if cleaned_sequences:
        return [ungap_sequence(str(sequence)) for sequence in cleaned_sequences]

    source_msa = metadata.get("source_msa")
    if source_msa:
        source_path = Path(source_msa)
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        return read_fasta_sequences(source_path)
    raise ValueError(f"{sidecar} does not contain cleaned_sequences or source_msa")


def find_examples(
    embeddings_dir: Path,
    metadata_dir: Path,
    embedding_glob: str,
    max_examples: int | None,
    numeric_fields: tuple[str, ...],
    categorical_fields: tuple[str, ...],
    value_aggregation: str,
    condition_transform: str,
    require_status_ok: bool,
) -> list[SequenceExample]:
    examples: list[SequenceExample] = []
    allowed_tokens = set(SEQUENCE_TOKENS)
    needs_metadata = bool(numeric_fields or categorical_fields)
    for embedding_path in sorted(embeddings_dir.glob(embedding_glob)):
        try:
            target_sequences = target_sequences_from_sidecar(embedding_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"warning: skipping {embedding_path}: {exc}", file=sys.stderr)
            continue
        metadata_path = metadata_dir / f"{embedding_path.stem}.metadata.tsv"
        if needs_metadata:
            if not metadata_path.exists():
                print(f"warning: skipping {embedding_path}: no metadata TSV at {metadata_path}", file=sys.stderr)
                continue
            rows = read_metadata_rows(metadata_path)
        else:
            metadata_path = None
            rows = [{} for _ in target_sequences]
        if len(rows) != len(target_sequences):
            print(
                f"warning: {embedding_path} has {len(target_sequences)} embedded sequences "
                f"but {len(rows)} metadata rows; using the aligned prefix",
                file=sys.stderr,
            )

        for row_index, (target_sequence, row) in enumerate(zip(target_sequences, rows)):
            if require_status_ok and row.get("status") not in {"ok", "dry_run", ""}:
                continue
            if STOP_TOKEN in target_sequence:
                target_sequence = target_sequence.split(STOP_TOKEN, 1)[0]
            unknown = sorted(set(target_sequence) - allowed_tokens)
            if unknown:
                print(
                    f"warning: skipping {embedding_path} row {row_index}: "
                    f"sequence contains unsupported token(s) {', '.join(unknown)}",
                    file=sys.stderr,
                )
                continue
            if not target_sequence:
                print(f"warning: skipping {embedding_path} row {row_index}: empty target sequence", file=sys.stderr)
                continue

            numeric_values: list[float] = []
            numeric_mask: list[bool] = []
            for field in numeric_fields:
                value = metadata_numeric_value(row, field, value_aggregation)
                if value is None:
                    numeric_values.append(0.0)
                    numeric_mask.append(False)
                    continue
                try:
                    value = transform_numeric_condition(value, field, condition_transform)
                except ValueError:
                    numeric_values.append(0.0)
                    numeric_mask.append(False)
                    continue
                numeric_values.append(float(value))
                numeric_mask.append(True)

            categorical_values = tuple(metadata_categories(row, field) for field in categorical_fields)
            examples.append(
                SequenceExample(
                    embedding_path=embedding_path,
                    metadata_path=metadata_path,
                    row_index=row_index,
                    target_sequence=target_sequence,
                    numeric_values=np.array(numeric_values, dtype=np.float32),
                    numeric_mask=np.array(numeric_mask, dtype=np.bool_),
                    categorical_values=categorical_values,
                )
            )
            if max_examples is not None and len(examples) >= max_examples:
                return examples
    return examples


def compute_numeric_normalization(examples: list[SequenceExample], num_fields: int) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros(num_fields, dtype=np.float32)
    stds = np.ones(num_fields, dtype=np.float32)
    if num_fields == 0:
        return means, stds
    values = np.stack([example.numeric_values for example in examples], axis=0)
    masks = np.stack([example.numeric_mask for example in examples], axis=0)
    for idx in range(num_fields):
        observed = values[masks[:, idx], idx]
        if observed.size == 0:
            continue
        means[idx] = float(observed.mean())
        std = float(observed.std()) if observed.size > 1 else 1.0
        stds[idx] = std if std > 1.0e-6 else 1.0
    return means, stds


def numeric_observation_counts(examples: list[SequenceExample], num_fields: int) -> list[int]:
    if num_fields == 0:
        return []
    counts = [0 for _ in range(num_fields)]
    for example in examples:
        for idx, observed in enumerate(example.numeric_mask):
            if bool(observed):
                counts[idx] += 1
    return counts


def build_categorical_vocabs(
    examples: list[SequenceExample],
    categorical_fields: tuple[str, ...],
) -> list[dict[str, int]]:
    vocabs: list[dict[str, int]] = []
    for field_index, field in enumerate(categorical_fields):
        values = sorted(
            {
                value
                for example in examples
                for value in example.categorical_values[field_index]
            }
        )
        if not values:
            raise SystemExit(f"No observed values for categorical condition field {field!r}")
        vocabs.append({value: idx for idx, value in enumerate(values)})
    return vocabs


class SequenceEmbeddingDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        examples: list[SequenceExample],
        numeric_means: np.ndarray,
        numeric_stds: np.ndarray,
        categorical_vocabs: list[dict[str, int]],
    ) -> None:
        self.examples = examples
        self.numeric_means = numeric_means.astype(np.float32)
        self.numeric_stds = np.maximum(numeric_stds.astype(np.float32), 1.0e-6)
        self.categorical_vocabs = categorical_vocabs
        self._embedding_cache: dict[Path, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.examples)

    def load_embedding(self, path: Path) -> dict[str, np.ndarray]:
        if path not in self._embedding_cache:
            arrays = np.load(path)
            if "token_embeddings" not in arrays:
                raise RuntimeError(f"{path} does not contain token_embeddings; rerun embed_msas.py without --pool-only")
            self._embedding_cache[path] = {
                "token_embeddings": arrays["token_embeddings"].astype(np.float32),
                "aa_mask": arrays["aa_mask"].astype(np.bool_),
            }
        return self._embedding_cache[path]

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        embedding = self.load_embedding(example.embedding_path)
        condition_values = example.numeric_values.astype(np.float32, copy=True)
        condition_values = np.where(
            example.numeric_mask,
            (condition_values - self.numeric_means) / self.numeric_stds,
            0.0,
        ).astype(np.float32)
        categorical_ids = tuple(
            tuple(self.categorical_vocabs[field_idx][value] for value in values)
            for field_idx, values in enumerate(example.categorical_values)
        )
        return {
            "token_embeddings": embedding["token_embeddings"],
            "aa_mask": embedding["aa_mask"],
            "target_sequence": example.target_sequence,
            "condition_values": condition_values,
            "condition_mask": example.numeric_mask,
            "categorical_condition_ids": categorical_ids,
            "embedding_path": str(example.embedding_path),
            "metadata_path": str(example.metadata_path) if example.metadata_path is not None else "",
            "row_index": example.row_index,
        }


class SequenceCollator:
    def __init__(
        self,
        max_sequence_length: int,
        tail_stop_weight: float,
        mask_target_row_in_msa: bool = False,
    ) -> None:
        self.max_sequence_length = max_sequence_length
        self.tail_stop_weight = tail_stop_weight
        self.mask_target_row_in_msa = mask_target_row_in_msa

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        batch_size = len(batch)
        max_rows = max(item["token_embeddings"].shape[0] for item in batch)
        max_cols = max(item["token_embeddings"].shape[1] for item in batch)
        hidden_dim = batch[0]["token_embeddings"].shape[2]

        tokens = np.zeros((batch_size, max_rows, max_cols, hidden_dim), dtype=np.float32)
        aa_mask = np.zeros((batch_size, max_rows, max_cols), dtype=np.bool_)
        num_numeric_fields = batch[0]["condition_values"].shape[0]
        num_categorical_fields = len(batch[0]["categorical_condition_ids"])
        max_category_values = 1
        if num_categorical_fields:
            max_category_values = max(
                1,
                max(
                    len(field_values)
                    for item in batch
                    for field_values in item["categorical_condition_ids"]
                ),
            )
        condition_values = np.zeros((batch_size, num_numeric_fields), dtype=np.float32)
        condition_mask = np.zeros((batch_size, num_numeric_fields), dtype=np.bool_)
        categorical_ids = np.full(
            (batch_size, num_categorical_fields, max_category_values),
            -1,
            dtype=np.int64,
        )
        categorical_mask = np.zeros(
            (batch_size, num_categorical_fields, max_category_values),
            dtype=np.bool_,
        )
        target_continuous_embeddings = np.zeros(
            (batch_size, self.max_sequence_length, hidden_dim),
            dtype=np.float32,
        )
        target_continuous_mask = np.zeros(
            (batch_size, self.max_sequence_length),
            dtype=np.bool_,
        )
        target_sequences: list[str] = []
        embedding_paths: list[str] = []
        metadata_paths: list[str] = []
        row_indexes: list[int] = []

        for idx, item in enumerate(batch):
            row_count, col_count, _ = item["token_embeddings"].shape
            tokens[idx, :row_count, :col_count] = item["token_embeddings"]
            aa_mask[idx, :row_count, :col_count] = item["aa_mask"]
            row_index = int(item["row_index"])
            if not 0 <= row_index < row_count:
                raise ValueError(f"target row index {row_index} is outside MSA row count {row_count}")
            target_length = min(len(item["target_sequence"]), self.max_sequence_length - 1)
            residue_embeddings = item["token_embeddings"][row_index, item["aa_mask"][row_index]]
            target_continuous_length = min(target_length, residue_embeddings.shape[0])
            if target_continuous_length > 0:
                target_continuous_embeddings[idx, :target_continuous_length] = residue_embeddings[
                    :target_continuous_length
                ]
                target_continuous_mask[idx, :target_continuous_length] = True
            if self.mask_target_row_in_msa:
                if row_count <= 1:
                    raise ValueError("cannot mask the target row when the MSA has only one row")
                tokens[idx, row_index, :col_count] = 0.0
                aa_mask[idx, row_index, :col_count] = False
            condition_values[idx] = item["condition_values"]
            condition_mask[idx] = item["condition_mask"]
            for field_idx, field_values in enumerate(item["categorical_condition_ids"]):
                width = len(field_values)
                if width:
                    categorical_ids[idx, field_idx, :width] = field_values
                    categorical_mask[idx, field_idx, :width] = True
            target_sequences.append(item["target_sequence"])
            embedding_paths.append(item["embedding_path"])
            metadata_paths.append(item["metadata_path"])
            row_indexes.append(item["row_index"])

        target_tokens, loss_weights = batch_encode_sequences_with_stop(
            target_sequences,
            max_length=self.max_sequence_length,
            tail_stop_weight=self.tail_stop_weight,
        )
        return {
            "token_embeddings": torch.from_numpy(tokens),
            "aa_mask": torch.from_numpy(aa_mask),
            "target_tokens": target_tokens,
            "loss_weights": loss_weights,
            "condition_values": torch.from_numpy(condition_values),
            "condition_mask": torch.from_numpy(condition_mask),
            "categorical_condition_ids": torch.from_numpy(categorical_ids),
            "categorical_condition_mask": torch.from_numpy(categorical_mask),
            "target_continuous_embeddings": torch.from_numpy(target_continuous_embeddings),
            "target_continuous_mask": torch.from_numpy(target_continuous_mask),
            "target_sequences": target_sequences,
            "embedding_paths": embedding_paths,
            "metadata_paths": metadata_paths,
            "row_indexes": row_indexes,
        }


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def resolve_timestep_range(args: argparse.Namespace) -> tuple[int, int]:
    max_timestep = args.max_diffusion_timestep
    if max_timestep < 0:
        max_timestep = args.diffusion_timesteps - 1
    if not 0 <= args.min_diffusion_timestep < args.diffusion_timesteps:
        raise SystemExit("--min-diffusion-timestep must be in [0, diffusion_timesteps)")
    if not 0 <= max_timestep < args.diffusion_timesteps:
        raise SystemExit("--max-diffusion-timestep must be in [0, diffusion_timesteps)")
    if args.min_diffusion_timestep > max_timestep:
        raise SystemExit("--min-diffusion-timestep cannot exceed --max-diffusion-timestep")
    return args.min_diffusion_timestep, max_timestep


def validate_decoder_input_hardening(args: argparse.Namespace) -> None:
    if not 0.0 <= args.decoder_token_dropout <= 1.0:
        raise SystemExit("--decoder-token-dropout must be in [0, 1]")
    if not 0.0 <= args.decoder_span_mask_fraction <= 1.0:
        raise SystemExit("--decoder-span-mask-fraction must be in [0, 1]")
    if args.decoder_span_mask_length < 1:
        raise SystemExit("--decoder-span-mask-length must be at least 1")
    if not 0.0 <= args.condition_dropout <= 1.0:
        raise SystemExit("--condition-dropout must be in [0, 1]")
    if args.timestep_curriculum_epochs < 0:
        raise SystemExit("--timestep-curriculum-epochs must be non-negative")
    if args.latent_codiffusion_tokens < 0:
        raise SystemExit("--latent-codiffusion-tokens must be non-negative")
    if args.ccdd_continuous_loss_weight < 0.0:
        raise SystemExit("--ccdd-continuous-loss-weight must be non-negative")
    if args.ccdd_mode != "off":
        if args.decoder_start_mode != "discrete_mask":
            raise SystemExit("--ccdd-mode requires --decoder-start-mode discrete_mask")
        if not args.mask_target_row_in_msa:
            raise SystemExit("--ccdd-mode requires --mask-target-row-in-msa to avoid conditioning leakage")
        if args.latent_codiffusion_tokens:
            raise SystemExit("--ccdd-mode is mutually exclusive with --latent-codiffusion-tokens")
    if args.ccdd_continuous_timestep_scale <= 0.0:
        raise SystemExit("--ccdd-continuous-timestep-scale must be positive")
    if not 0.0 <= args.ccdd_continuous_dropout <= 1.0:
        raise SystemExit("--ccdd-continuous-dropout must be in [0, 1]")


def resolve_curriculum_start_range(args: argparse.Namespace, final_range: tuple[int, int]) -> tuple[int, int]:
    start_min = args.curriculum_start_min_diffusion_timestep
    start_max = args.curriculum_start_max_diffusion_timestep
    if start_min < 0:
        start_min = 0
    if start_max < 0:
        start_max = min(final_range[1], max(start_min, int(round(args.diffusion_timesteps * 0.2)) - 1))
    if not 0 <= start_min < args.diffusion_timesteps:
        raise SystemExit("--curriculum-start-min-diffusion-timestep must be in [0, diffusion_timesteps)")
    if not 0 <= start_max < args.diffusion_timesteps:
        raise SystemExit("--curriculum-start-max-diffusion-timestep must be in [0, diffusion_timesteps)")
    if start_min > start_max:
        raise SystemExit("--curriculum-start-min-diffusion-timestep cannot exceed --curriculum-start-max-diffusion-timestep")
    return start_min, start_max


def curriculum_timestep_range(
    epoch: int,
    final_range: tuple[int, int],
    start_range: tuple[int, int],
    curriculum_epochs: int,
) -> tuple[int, int]:
    if epoch < 1:
        raise ValueError("epoch must be at least 1")
    if curriculum_epochs <= 0:
        return final_range
    if curriculum_epochs == 1:
        return final_range
    progress = min(max((epoch - 1) / (curriculum_epochs - 1), 0.0), 1.0)
    current_min = int(round(start_range[0] + progress * (final_range[0] - start_range[0])))
    current_max = int(round(start_range[1] + progress * (final_range[1] - start_range[1])))
    if current_min > current_max:
        current_min = current_max
    return current_min, current_max


def compatible_checkpoint_state_dict(
    model: torch.nn.Module,
    checkpoint_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Resize old decoder heads when a checkpoint predates the MASK token."""
    model_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    notes: list[str] = []
    for key, value in checkpoint_state.items():
        target = model_state.get(key)
        if target is None:
            compatible[key] = value
            continue
        if value.shape == target.shape:
            compatible[key] = value
            continue
        can_expand_vocab = (
            value.ndim == target.ndim
            and value.ndim >= 1
            and value.shape[1:] == target.shape[1:]
            and target.shape[0] == value.shape[0] + 1
        )
        if can_expand_vocab and (
            key.endswith("decoder.token_embedding.weight")
            or key.endswith("decoder.lm_head.weight")
            or key.endswith("decoder.lm_head.bias")
        ):
            expanded = target.detach().clone()
            expanded[: value.shape[0]] = value
            if value.ndim == 1:
                expanded[value.shape[0] :] = value.mean()
            else:
                expanded[value.shape[0] :] = value.mean(dim=0, keepdim=True)
            compatible[key] = expanded
            notes.append(f"expanded {key} from {tuple(value.shape)} to {tuple(target.shape)}")
            continue
        notes.append(f"skipped incompatible {key}: checkpoint {tuple(value.shape)} model {tuple(target.shape)}")
    return compatible, notes


def parse_numeric_fields(text: str) -> tuple[str, ...]:
    fields = tuple(field.strip() for field in text.split(",") if field.strip())
    unknown = sorted(set(fields) - set(DEFAULT_NUMERIC_FIELDS))
    if unknown:
        raise SystemExit(f"Unknown numeric condition field(s): {', '.join(unknown)}")
    return fields


def parse_categorical_fields(text: str) -> tuple[str, ...]:
    return tuple(field.strip() for field in text.split(",") if field.strip())


def train(args: argparse.Namespace) -> int:
    if not 0.0 <= args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be in [0, 1)")
    timestep_range = resolve_timestep_range(args)
    validate_decoder_input_hardening(args)
    curriculum_start_range = resolve_curriculum_start_range(args, timestep_range)
    numeric_condition_fields = parse_numeric_fields(args.numeric_condition_fields)
    categorical_condition_fields = parse_categorical_fields(args.categorical_condition_fields)
    examples = find_examples(
        embeddings_dir=Path(args.embeddings_dir),
        metadata_dir=Path(args.metadata_dir),
        embedding_glob=args.embedding_glob,
        max_examples=args.max_examples,
        numeric_fields=numeric_condition_fields,
        categorical_fields=categorical_condition_fields,
        value_aggregation=args.value_aggregation,
        condition_transform=args.condition_transform,
        require_status_ok=not args.allow_non_ok_status,
    )
    if not examples:
        raise SystemExit("No sequence examples found. Generate embeddings with token_embeddings and metadata sidecars first.")

    numeric_means, numeric_stds = compute_numeric_normalization(examples, len(numeric_condition_fields))
    numeric_counts = numeric_observation_counts(examples, len(numeric_condition_fields))
    if numeric_condition_fields:
        empty_fields = [
            field
            for field, observed_count in zip(numeric_condition_fields, numeric_counts)
            if observed_count == 0
        ]
        if empty_fields and not args.allow_empty_numeric_condition_fields:
            raise SystemExit(
                "Requested numeric condition field(s) have no observed values: "
                + ", ".join(empty_fields)
                + ". Enrich metadata first or pass --allow-empty-numeric-condition-fields."
            )
    categorical_vocabs = build_categorical_vocabs(examples, categorical_condition_fields)
    dataset = SequenceEmbeddingDataset(
        examples=examples,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
        categorical_vocabs=categorical_vocabs,
    )
    collator = SequenceCollator(
        max_sequence_length=args.max_sequence_length,
        tail_stop_weight=args.tail_stop_weight,
        mask_target_row_in_msa=args.mask_target_row_in_msa,
    )
    split_generator = torch.Generator().manual_seed(args.seed)
    if len(dataset) > 1 and args.val_fraction > 0.0:
        val_count = max(1, int(round(len(dataset) * args.val_fraction)))
        val_count = min(val_count, len(dataset) - 1)
        permutation = torch.randperm(len(dataset), generator=split_generator).tolist()
        val_indices = permutation[:val_count]
        train_indices = permutation[val_count:]
        train_dataset: Dataset[dict[str, Any]] = Subset(dataset, train_indices)
        val_dataset: Dataset[dict[str, Any]] | None = Subset(dataset, val_indices)
    else:
        train_indices = list(range(len(dataset)))
        val_indices: list[int] = []
        train_dataset = dataset
        val_dataset = None

    loader_generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        collate_fn=collator,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
        )
        if val_dataset is not None
        else None
    )

    first_embedding = dataset.load_embedding(examples[0].embedding_path)["token_embeddings"]
    model = MSASequenceDiffusionModel(
        input_dim=first_embedding.shape[-1],
        d_model=args.d_model,
        max_sequence_length=args.max_sequence_length,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        num_timesteps=args.diffusion_timesteps,
        numeric_condition_fields=numeric_condition_fields,
        categorical_condition_fields=categorical_condition_fields,
        categorical_vocab_sizes=[len(vocab) for vocab in categorical_vocabs],
        condition_layers=args.condition_layers,
        latent_codiffusion_tokens=args.latent_codiffusion_tokens,
        ccdd_mode=args.ccdd_mode,
    )
    device = choose_device(args.device)
    model.to(device)
    if args.init_checkpoint:
        init_checkpoint_path = Path(args.init_checkpoint)
        init_payload = torch.load(init_checkpoint_path, map_location=device)
        compatible_state, compatibility_notes = compatible_checkpoint_state_dict(
            model,
            init_payload["model_state_dict"],
        )
        load_result = model.load_state_dict(compatible_state, strict=False)
        print(f"Initialized model from {init_checkpoint_path}", flush=True)
        for note in compatibility_notes:
            print(f"checkpoint_compatibility={note}", flush=True)
        if load_result.missing_keys:
            print(f"checkpoint_missing_keys={','.join(load_result.missing_keys)}", flush=True)
        if load_result.unexpected_keys:
            print(f"checkpoint_unexpected_keys={','.join(load_result.unexpected_keys)}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    checkpoint_path = Path(args.out_checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_tsv) if args.metrics_tsv else checkpoint_path.with_suffix(".metrics.tsv")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_path = (
        Path(args.latest_checkpoint)
        if args.latest_checkpoint
        else checkpoint_path.with_name(f"{checkpoint_path.stem}.latest{checkpoint_path.suffix}")
    )
    latest_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_fields = [
        "epoch",
        "split",
        "examples",
        "loss",
        "diffusion_loss",
        "latent_loss",
        "ccdd_continuous_loss",
        "token_loss",
        "token_accuracy",
        "full_token_accuracy",
        "corruption_fraction",
        "condition_drop_fraction",
        "ccdd_continuous_drop_fraction",
        "timestep_min",
        "timestep_max",
        "timestep_mean",
        "elapsed_seconds",
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=metrics_fields)
        writer.writeheader()

    print(
        f"Training sequence decoder examples={len(dataset)} train={len(train_dataset)} "
        f"val={len(val_dataset) if val_dataset is not None else 0} "
        f"max_sequence_length={args.max_sequence_length} "
        f"device={device} stop_token={STOP_TOKEN!r} mask_token={MASK_TOKEN!r} "
        f"mask_token_id={MASK_TOKEN_ID} vocab_size={len(SEQUENCE_TOKENS)} "
        f"mask_target_row_in_msa={args.mask_target_row_in_msa} "
        f"timestep_range={timestep_range[0]}:{timestep_range[1]} "
        f"timestep_curriculum_epochs={args.timestep_curriculum_epochs} "
        f"curriculum_start_range={curriculum_start_range[0]}:{curriculum_start_range[1]} "
        f"decoder_start_mode={args.decoder_start_mode} "
        f"latent_codiffusion_tokens={args.latent_codiffusion_tokens} "
        f"latent_loss_weight={args.latent_loss_weight} "
        f"ccdd_mode={args.ccdd_mode} "
        f"ccdd_continuous_loss_weight={args.ccdd_continuous_loss_weight} "
        f"ccdd_continuous_timestep_scale={args.ccdd_continuous_timestep_scale} "
        f"ccdd_continuous_dropout={args.ccdd_continuous_dropout} "
        f"discrete_loss_corrupted_only={args.discrete_loss_corrupted_only} "
        f"condition_dropout={args.condition_dropout} "
        f"decoder_token_dropout={args.decoder_token_dropout} "
        f"decoder_span_mask_fraction={args.decoder_span_mask_fraction} "
        f"numeric_conditions={','.join(numeric_condition_fields) or 'none'} "
        f"categorical_conditions={','.join(categorical_condition_fields) or 'none'} "
        f"metrics={metrics_path}",
        flush=True,
    )
    if numeric_condition_fields:
        print(
            "numeric_condition_coverage="
            + ",".join(
                f"{field}:{observed_count}/{len(dataset)}"
                for field, observed_count in zip(numeric_condition_fields, numeric_counts)
            ),
            flush=True,
        )
    if len(dataset) < 8:
        print("warning: tiny dataset; this is a wiring smoke test, not a meaningful generation run", flush=True)

    last_outputs: dict[str, torch.Tensor] | None = None
    last_batch: dict[str, Any] | None = None
    start_time = time.monotonic()

    def run_loader(
        loader: DataLoader[dict[str, Any]],
        training: bool,
        epoch_timestep_range: tuple[int, int],
    ) -> dict[str, float]:
        nonlocal last_outputs, last_batch
        if training:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        total_diffusion = 0.0
        total_latent = 0.0
        total_ccdd_continuous = 0.0
        total_token = 0.0
        total_accuracy = 0.0
        total_full_accuracy = 0.0
        total_corruption_fraction = 0.0
        total_condition_drop_fraction = 0.0
        total_ccdd_continuous_drop_fraction = 0.0
        timestep_sum = 0.0
        timestep_count = 0
        sampled_timestep_min: int | None = None
        sampled_timestep_max: int | None = None
        total_count = 0
        for batch in loader:
            token_embeddings = batch["token_embeddings"].to(device)
            aa_mask = batch["aa_mask"].to(device)
            target_tokens = batch["target_tokens"].to(device)
            loss_weights = batch["loss_weights"].to(device)
            model_inputs: dict[str, torch.Tensor] = {}
            if numeric_condition_fields:
                model_inputs["condition_values"] = batch["condition_values"].to(device)
                model_inputs["condition_mask"] = batch["condition_mask"].to(device)
            if categorical_condition_fields:
                model_inputs["categorical_condition_ids"] = batch["categorical_condition_ids"].to(device)
                model_inputs["categorical_condition_mask"] = batch["categorical_condition_mask"].to(device)
            if args.ccdd_mode != "off":
                model_inputs["target_continuous_embeddings"] = batch["target_continuous_embeddings"].to(device)
                model_inputs["target_continuous_mask"] = batch["target_continuous_mask"].to(device)
            timesteps = torch.randint(
                epoch_timestep_range[0],
                epoch_timestep_range[1] + 1,
                (target_tokens.shape[0],),
                device=device,
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                outputs = model(
                    token_embeddings=token_embeddings,
                    aa_mask=aa_mask,
                    target_tokens=target_tokens,
                    loss_weights=loss_weights,
                    timesteps=timesteps,
                    decoder_start_mode=args.decoder_start_mode,
                    decoder_token_dropout=args.decoder_token_dropout,
                    decoder_span_mask_fraction=args.decoder_span_mask_fraction,
                    decoder_span_mask_length=args.decoder_span_mask_length,
                    discrete_loss_corrupted_only=args.discrete_loss_corrupted_only,
                    condition_dropout=args.condition_dropout,
                    ccdd_continuous_timestep_scale=args.ccdd_continuous_timestep_scale,
                    ccdd_continuous_dropout=args.ccdd_continuous_dropout,
                    **model_inputs,
                )
                loss = (
                    args.diffusion_loss_weight * outputs["diffusion_loss"]
                    + args.latent_loss_weight * outputs.get(
                        "latent_loss",
                        torch.zeros((), dtype=torch.float32, device=device),
                    )
                    + args.ccdd_continuous_loss_weight
                    * outputs.get(
                        "ccdd_continuous_loss",
                        torch.zeros((), dtype=torch.float32, device=device),
                    )
                    + args.token_loss_weight * outputs["token_loss"]
                )
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

            batch_count = target_tokens.shape[0]
            total_loss += float(loss.item()) * batch_count
            total_diffusion += float(outputs["diffusion_loss"].item()) * batch_count
            total_latent += float(
                outputs.get(
                    "latent_loss",
                    torch.zeros((), dtype=torch.float32, device=device),
                ).item()
            ) * batch_count
            total_ccdd_continuous += float(
                outputs.get(
                    "ccdd_continuous_loss",
                    torch.zeros((), dtype=torch.float32, device=device),
                ).item()
            ) * batch_count
            total_token += float(outputs["token_loss"].item()) * batch_count
            total_accuracy += float(outputs["token_accuracy"].item()) * batch_count
            total_full_accuracy += float(
                outputs.get("full_token_accuracy", outputs["token_accuracy"]).item()
            ) * batch_count
            total_corruption_fraction += float(
                outputs.get(
                    "corruption_fraction",
                    torch.zeros((), dtype=torch.float32, device=device),
                ).item()
            ) * batch_count
            condition_drop_mask = outputs.get("condition_drop_mask")
            if condition_drop_mask is not None:
                condition_drop_fraction = condition_drop_mask.to(dtype=torch.float32).mean()
            else:
                condition_drop_fraction = torch.zeros((), dtype=torch.float32, device=device)
            total_condition_drop_fraction += float(condition_drop_fraction.item()) * batch_count
            ccdd_continuous_drop_mask = outputs.get("ccdd_continuous_drop_mask")
            if ccdd_continuous_drop_mask is not None:
                ccdd_continuous_drop_fraction = ccdd_continuous_drop_mask.to(dtype=torch.float32).mean()
            else:
                ccdd_continuous_drop_fraction = torch.zeros((), dtype=torch.float32, device=device)
            total_ccdd_continuous_drop_fraction += float(ccdd_continuous_drop_fraction.item()) * batch_count
            timestep_sum += float(timesteps.to(dtype=torch.float32).sum().item())
            timestep_count += int(timesteps.numel())
            batch_timestep_min = int(timesteps.min().item())
            batch_timestep_max = int(timesteps.max().item())
            sampled_timestep_min = (
                batch_timestep_min if sampled_timestep_min is None else min(sampled_timestep_min, batch_timestep_min)
            )
            sampled_timestep_max = (
                batch_timestep_max if sampled_timestep_max is None else max(sampled_timestep_max, batch_timestep_max)
            )
            total_count += batch_count
            if training or last_outputs is None:
                last_outputs = {key: value.detach().cpu() for key, value in outputs.items() if isinstance(value, torch.Tensor)}
                last_batch = batch
        denom = max(total_count, 1)
        return {
            "examples": float(total_count),
            "loss": total_loss / denom,
            "diffusion_loss": total_diffusion / denom,
            "latent_loss": total_latent / denom,
            "ccdd_continuous_loss": total_ccdd_continuous / denom,
            "token_loss": total_token / denom,
            "token_accuracy": total_accuracy / denom,
            "full_token_accuracy": total_full_accuracy / denom,
            "corruption_fraction": total_corruption_fraction / denom,
            "condition_drop_fraction": total_condition_drop_fraction / denom,
            "ccdd_continuous_drop_fraction": total_ccdd_continuous_drop_fraction / denom,
            "timestep_min": float(sampled_timestep_min if sampled_timestep_min is not None else epoch_timestep_range[0]),
            "timestep_max": float(sampled_timestep_max if sampled_timestep_max is not None else epoch_timestep_range[1]),
            "timestep_mean": timestep_sum / max(timestep_count, 1),
        }

    def checkpoint_payload(epoch: int, epoch_metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
        return {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "metrics": epoch_metrics,
            "config": {
                "input_dim": int(first_embedding.shape[-1]),
                "d_model": args.d_model,
                "max_sequence_length": args.max_sequence_length,
                "layers": args.layers,
                "heads": args.heads,
                "dropout": args.dropout,
                "diffusion_timesteps": args.diffusion_timesteps,
                "diffusion_loss_weight": args.diffusion_loss_weight,
                "latent_loss_weight": args.latent_loss_weight,
                "ccdd_continuous_loss_weight": args.ccdd_continuous_loss_weight,
                "token_loss_weight": args.token_loss_weight,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "init_checkpoint": args.init_checkpoint,
                "sequence_tokens": SEQUENCE_TOKENS,
                "stop_token": STOP_TOKEN,
                "mask_token": MASK_TOKEN,
                "mask_token_id": MASK_TOKEN_ID,
                "tail_stop_weight": args.tail_stop_weight,
                "mask_target_row_in_msa": args.mask_target_row_in_msa,
                "min_diffusion_timestep": timestep_range[0],
                "max_diffusion_timestep": timestep_range[1],
                "timestep_curriculum_epochs": args.timestep_curriculum_epochs,
                "curriculum_start_min_diffusion_timestep": curriculum_start_range[0],
                "curriculum_start_max_diffusion_timestep": curriculum_start_range[1],
                "decoder_start_mode": args.decoder_start_mode,
                "latent_codiffusion_tokens": args.latent_codiffusion_tokens,
                "ccdd_mode": args.ccdd_mode,
                "ccdd_continuous_timestep_scale": args.ccdd_continuous_timestep_scale,
                "ccdd_continuous_dropout": args.ccdd_continuous_dropout,
                "discrete_loss_corrupted_only": args.discrete_loss_corrupted_only,
                "condition_dropout": args.condition_dropout,
                "decoder_token_dropout": args.decoder_token_dropout,
                "decoder_span_mask_fraction": args.decoder_span_mask_fraction,
                "decoder_span_mask_length": args.decoder_span_mask_length,
                "numeric_condition_fields": numeric_condition_fields,
                "numeric_condition_observed_counts": dict(zip(numeric_condition_fields, numeric_counts)),
                "numeric_condition_observed_fraction": {
                    field: observed_count / max(len(dataset), 1)
                    for field, observed_count in zip(numeric_condition_fields, numeric_counts)
                },
                "categorical_condition_fields": categorical_condition_fields,
                "categorical_vocabs": categorical_vocabs,
                "numeric_condition_mean": numeric_means.tolist(),
                "numeric_condition_std": numeric_stds.tolist(),
                "condition_transform": args.condition_transform,
                "value_aggregation": args.value_aggregation,
                "condition_layers": args.condition_layers,
                "train_indices": train_indices,
                "val_indices": val_indices,
            },
        }

    for epoch in range(1, args.epochs + 1):
        epoch_timestep_range = curriculum_timestep_range(
            epoch,
            final_range=timestep_range,
            start_range=curriculum_start_range,
            curriculum_epochs=args.timestep_curriculum_epochs,
        )
        train_metrics = run_loader(train_loader, training=True, epoch_timestep_range=epoch_timestep_range)
        val_metrics = (
            run_loader(val_loader, training=False, epoch_timestep_range=epoch_timestep_range)
            if val_loader is not None
            else {}
        )
        elapsed = time.monotonic() - start_time
        epoch_metrics = {"train": train_metrics}
        if val_metrics:
            epoch_metrics["val"] = val_metrics
        with metrics_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=metrics_fields)
            for split, metrics in epoch_metrics.items():
                writer.writerow(
                    {
                        "epoch": epoch,
                        "split": split,
                        "examples": int(metrics["examples"]),
                        "loss": f"{metrics['loss']:.8f}",
                        "diffusion_loss": f"{metrics['diffusion_loss']:.8f}",
                        "latent_loss": f"{metrics['latent_loss']:.8f}",
                        "ccdd_continuous_loss": f"{metrics['ccdd_continuous_loss']:.8f}",
                        "token_loss": f"{metrics['token_loss']:.8f}",
                        "token_accuracy": f"{metrics['token_accuracy']:.8f}",
                        "full_token_accuracy": f"{metrics['full_token_accuracy']:.8f}",
                        "corruption_fraction": f"{metrics['corruption_fraction']:.8f}",
                        "condition_drop_fraction": f"{metrics['condition_drop_fraction']:.8f}",
                        "ccdd_continuous_drop_fraction": f"{metrics['ccdd_continuous_drop_fraction']:.8f}",
                        "timestep_min": f"{metrics['timestep_min']:.0f}",
                        "timestep_max": f"{metrics['timestep_max']:.0f}",
                        "timestep_mean": f"{metrics['timestep_mean']:.4f}",
                        "elapsed_seconds": f"{elapsed:.3f}",
                    }
                )
        if val_metrics:
            val_text = (
                f" val_loss={val_metrics['loss']:.6f} "
                f"val_token_acc={val_metrics['token_accuracy']:.4f} "
                f"val_full_token_acc={val_metrics['full_token_accuracy']:.4f}"
            )
        else:
            val_text = ""
        print(
            f"epoch={epoch} loss={train_metrics['loss']:.6f} "
            f"diffusion={train_metrics['diffusion_loss']:.6f} "
            f"latent={train_metrics['latent_loss']:.6f} "
            f"ccdd_cont={train_metrics['ccdd_continuous_loss']:.6f} "
            f"token={train_metrics['token_loss']:.6f} "
            f"token_acc={train_metrics['token_accuracy']:.4f} "
            f"full_token_acc={train_metrics['full_token_accuracy']:.4f} "
            f"corruption={train_metrics['corruption_fraction']:.3f} "
            f"timestep={epoch_timestep_range[0]}:{epoch_timestep_range[1]} "
            f"condition_drop={train_metrics['condition_drop_fraction']:.3f}{val_text}",
            flush=True,
        )
        torch.save(checkpoint_payload(epoch, epoch_metrics), latest_checkpoint_path)

    if last_outputs is not None and last_batch is not None:
        predicted_ids = torch.argmax(last_outputs["logits"][0], dim=-1).tolist()
        preview = decode_tokens_until_stop(predicted_ids)
        target = last_batch["target_sequences"][0]
        print(f"preview target_len={len(target)} decoded_len={len(preview)} decoded_prefix={preview[:80]}", flush=True)

    torch.save(checkpoint_payload(args.epochs, {"final": train_metrics}), checkpoint_path)
    sidecar = checkpoint_path.with_suffix(".metadata.json")
    sidecar.write_text(
        json.dumps(
            {
                "examples": len(dataset),
                "train_examples": len(train_dataset),
                "val_examples": len(val_dataset) if val_dataset is not None else 0,
                "checkpoint": str(checkpoint_path),
                "latest_checkpoint": str(latest_checkpoint_path),
                "metrics_tsv": str(metrics_path),
                "max_sequence_length": args.max_sequence_length,
                "sequence_tokens": SEQUENCE_TOKENS,
                "stop_token": STOP_TOKEN,
                "mask_token": MASK_TOKEN,
                "mask_token_id": MASK_TOKEN_ID,
                "diffusion_loss_weight": args.diffusion_loss_weight,
                "latent_loss_weight": args.latent_loss_weight,
                "ccdd_continuous_loss_weight": args.ccdd_continuous_loss_weight,
                "token_loss_weight": args.token_loss_weight,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "init_checkpoint": args.init_checkpoint,
                "mask_target_row_in_msa": args.mask_target_row_in_msa,
                "min_diffusion_timestep": timestep_range[0],
                "max_diffusion_timestep": timestep_range[1],
                "timestep_curriculum_epochs": args.timestep_curriculum_epochs,
                "curriculum_start_min_diffusion_timestep": curriculum_start_range[0],
                "curriculum_start_max_diffusion_timestep": curriculum_start_range[1],
                "decoder_start_mode": args.decoder_start_mode,
                "latent_codiffusion_tokens": args.latent_codiffusion_tokens,
                "ccdd_mode": args.ccdd_mode,
                "ccdd_continuous_timestep_scale": args.ccdd_continuous_timestep_scale,
                "ccdd_continuous_dropout": args.ccdd_continuous_dropout,
                "discrete_loss_corrupted_only": args.discrete_loss_corrupted_only,
                "condition_dropout": args.condition_dropout,
                "decoder_token_dropout": args.decoder_token_dropout,
                "decoder_span_mask_fraction": args.decoder_span_mask_fraction,
                "decoder_span_mask_length": args.decoder_span_mask_length,
                "numeric_condition_fields": numeric_condition_fields,
                "numeric_condition_observed_counts": dict(zip(numeric_condition_fields, numeric_counts)),
                "numeric_condition_observed_fraction": {
                    field: observed_count / max(len(dataset), 1)
                    for field, observed_count in zip(numeric_condition_fields, numeric_counts)
                },
                "categorical_condition_fields": categorical_condition_fields,
                "categorical_vocabs": categorical_vocabs,
                "numeric_condition_mean": numeric_means.tolist(),
                "numeric_condition_std": numeric_stds.tolist(),
                "condition_transform": args.condition_transform,
                "value_aggregation": args.value_aggregation,
                "condition_layers": args.condition_layers,
                "note": "MSA Transformer embeddings are frozen/precomputed; condition tokens are prepended to decoder memory, and the loss reconstructs the protein sequence.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved checkpoint to {checkpoint_path}", flush=True)
    print(f"Saved latest checkpoint to {latest_checkpoint_path}", flush=True)
    print(f"Saved metrics to {metrics_path}", flush=True)
    print(f"Saved metadata to {sidecar}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", default="outputs/embeddings", help="Directory containing embedding NPZ files.")
    parser.add_argument("--metadata-dir", default="outputs/pilot_msas", help="Directory containing metadata TSV files.")
    parser.add_argument("--embedding-glob", default="ec_*.npz", help="Embedding filename glob relative to --embeddings-dir.")
    parser.add_argument("--max-examples", type=int, default=None, help="Optional cap for quick smoke runs.")
    parser.add_argument("--max-sequence-length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    parser.add_argument("--tail-stop-weight", type=float, default=0.05)
    parser.add_argument(
        "--mask-target-row-in-msa",
        action="store_true",
        help="For each reconstruction example, remove the target row from the MSA memory.",
    )
    parser.add_argument(
        "--numeric-condition-fields",
        default=",".join(DEFAULT_NUMERIC_FIELDS),
        help="Comma-separated numeric metadata fields encoded as condition tokens.",
    )
    parser.add_argument(
        "--categorical-condition-fields",
        default=",".join(DEFAULT_CATEGORICAL_FIELDS),
        help="Comma-separated metadata columns encoded with categorical embeddings.",
    )
    parser.add_argument("--condition-transform", choices=["auto", "identity", "log10"], default="auto")
    parser.add_argument("--value-aggregation", choices=["mean", "first"], default="mean")
    parser.add_argument("--condition-layers", type=int, default=1, help="Self-attention layers over condition tokens.")
    parser.add_argument(
        "--allow-empty-numeric-condition-fields",
        action="store_true",
        help="Allow requested numeric condition fields to be present only as learned missing-value tokens.",
    )
    parser.add_argument("--allow-non-ok-status", action="store_true", help="Do not filter metadata rows by status.")
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Held-out fraction for validation metrics.")
    parser.add_argument("--d-model", type=int, default=128, help="Trainable projection and decoder width.")
    parser.add_argument("--layers", type=int, default=2, help="Transformer decoder layers.")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads in the trainable decoder.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--diffusion-timesteps", type=int, default=1000)
    parser.add_argument(
        "--min-diffusion-timestep",
        type=int,
        default=0,
        help="Lowest diffusion timestep sampled during training and validation.",
    )
    parser.add_argument(
        "--max-diffusion-timestep",
        type=int,
        default=-1,
        help="Highest diffusion timestep sampled during training and validation. -1 means diffusion_timesteps - 1.",
    )
    parser.add_argument(
        "--timestep-curriculum-epochs",
        type=int,
        default=0,
        help="Ramp sampled timestep range from a low-corruption start range to --min/--max-diffusion-timestep.",
    )
    parser.add_argument(
        "--curriculum-start-min-diffusion-timestep",
        type=int,
        default=-1,
        help="Initial curriculum minimum timestep. -1 defaults to 0.",
    )
    parser.add_argument(
        "--curriculum-start-max-diffusion-timestep",
        type=int,
        default=-1,
        help="Initial curriculum maximum timestep. -1 defaults to roughly 20%% of diffusion timesteps.",
    )
    parser.add_argument(
        "--decoder-start-mode",
        choices=["mean", "noisy_mean", "discrete_mask", "discrete_random", "q_sample", "pure_noise"],
        default="mean",
        help="Decoder-side starting embeddings used while training against the target sequence.",
    )
    parser.add_argument(
        "--latent-codiffusion-tokens",
        type=int,
        default=0,
        help="Enable target-free latent + sequence codiffusion with this many denoised sequence-latent tokens.",
    )
    parser.add_argument(
        "--ccdd-mode",
        choices=["off", "mdit"],
        default="off",
        help="Enable CCDD-style per-position continuous/discrete co-denoising.",
    )
    parser.add_argument(
        "--ccdd-continuous-timestep-scale",
        type=float,
        default=0.75,
        help="Scale discrete timesteps before noising the continuous CCDD stream; values below 1 keep z_t cleaner.",
    )
    parser.add_argument(
        "--ccdd-continuous-dropout",
        type=float,
        default=0.0,
        help="Drop the whole continuous sequence stream for a fraction of training examples.",
    )
    parser.add_argument(
        "--discrete-loss-corrupted-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For discrete corruption modes, apply token CE only to corrupted positions.",
    )
    parser.add_argument(
        "--condition-dropout",
        type=float,
        default=0.0,
        help="Randomly replace the MSA/metadata memory with a learned null token for classifier-free guidance.",
    )
    parser.add_argument(
        "--decoder-token-dropout",
        type=float,
        default=0.0,
        help="For q_sample starts, replace this fraction of target input embeddings with the mean token embedding before noising.",
    )
    parser.add_argument(
        "--decoder-span-mask-fraction",
        type=float,
        default=0.0,
        help="For q_sample starts, replace contiguous spans covering this fraction of positions before noising.",
    )
    parser.add_argument(
        "--decoder-span-mask-length",
        type=int,
        default=16,
        help="Approximate span length for --decoder-span-mask-fraction.",
    )
    parser.add_argument("--diffusion-loss-weight", type=float, default=1.0)
    parser.add_argument("--latent-loss-weight", type=float, default=0.0)
    parser.add_argument("--ccdd-continuous-loss-weight", type=float, default=0.0)
    parser.add_argument("--token-loss-weight", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--init-checkpoint", default="", help="Optional checkpoint to initialize model weights from.")
    parser.add_argument("--out-checkpoint", default="outputs/checkpoints/sequence_diffusion_smoke.pt")
    parser.add_argument("--latest-checkpoint", default="", help="Checkpoint path overwritten after each epoch.")
    parser.add_argument("--metrics-tsv", default="", help="Per-epoch train/validation metrics TSV.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(train(parse_args()))
