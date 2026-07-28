#!/usr/bin/env python3
"""Train mean-start sequence reconstruction from cached MSA features.

This trainer deliberately avoids cached column embeddings because those include
the target row. For each example it removes the target row on the fly, builds a
leave-one-row-out profile, optionally appends cached non-target row embeddings,
and reconstructs the target sequence from a mean/noisy-mean decoder start.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import json
import math
import random
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msa_design_model import (  # noqa: E402
    MASK_TOKEN,
    MSADepthScaler,
    SEQUENCE_TOKENS,
    STOP_TOKEN,
    SequenceDiffusionDecoder,
    batch_encode_sequences_with_stop,
    decode_tokens_until_stop,
)
from msa_design_model.model import weighted_position_mse  # noqa: E402


DEFAULT_TRAINING_ROOT = Path("/mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim")
DEFAULT_EMBEDDING_MANIFEST = DEFAULT_TRAINING_ROOT / "esm_msa_embeddings_col" / "embedding_manifest.tsv"
DEFAULT_LABEL_SUMMARY = DEFAULT_TRAINING_ROOT / "sequence_label_summary.tsv.gz"
DEFAULT_OUT_DIR = DEFAULT_TRAINING_ROOT / "mean_start_ccdd_cached"

AA_TOKENS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_COL = {aa: idx for idx, aa in enumerate(AA_TOKENS)}
ALLOWED_TARGET_TOKENS = set(AA_TOKENS)
NUMERIC_FIELDS = ("kcat_1_per_s", "km_mM", "kcat_over_km_1_per_mM_s", "topt_C", "tm_C")
CATEGORICAL_FIELDS = ("domain", "reaction_id", "ec_numbers", "compound_id")
LOG_NUMERIC_FIELDS = {"kcat_1_per_s", "km_mM", "kcat_over_km_1_per_mM_s"}
RAW_PROFILE_DIM = len(AA_TOKENS) + 2
PROFILE_DIM = RAW_PROFILE_DIM
PROFILE_FEATURE_MODES = ("full", "no_aa_frequency")
MEMORY_MODES = ("profile_only", "profile_row", "profile_msa", "profile_msa_row", "profile_msa_axial")
MSA_EMBEDDING_DTYPES = ("float32", "float16", "native")
AMP_MODES = ("off", "fp16", "bf16")
CONTINUOUS_TARGET_MODES = ("token_embedding", "target_row_embedding")

CONSENSUS_METRIC_FIELDS = (
    "sequence_loss_weight_mean",
    "consensus_residue_accuracy",
    "nonconsensus_residue_accuracy",
    "nonconsensus_fraction",
    "variable_nonconsensus_fraction",
    "profile_variable_fraction",
    "profile_drop_fraction",
    "profile_blur_fraction",
)
CONDITION_METRIC_FIELDS = (
    "numeric_value_loss",
    "numeric_value_mae",
    "numeric_presence_loss",
    "numeric_presence_accuracy",
    "category_value_loss",
    "category_value_accuracy",
    "category_presence_loss",
    "category_presence_accuracy",
    "condition_mask_fraction",
)


def profile_input_dim(profile_feature_mode: str) -> int:
    if profile_feature_mode == "full":
        return RAW_PROFILE_DIM
    if profile_feature_mode == "no_aa_frequency":
        return 2
    raise ValueError(f"unknown profile_feature_mode: {profile_feature_mode}")


def select_profile_features(profile: np.ndarray, profile_feature_mode: str) -> np.ndarray:
    if profile_feature_mode == "full":
        return profile
    if profile_feature_mode == "no_aa_frequency":
        return profile[:, len(AA_TOKENS) : len(AA_TOKENS) + 2]
    raise ValueError(f"unknown profile_feature_mode: {profile_feature_mode}")


def uses_row_memory(memory_mode: str) -> bool:
    return memory_mode in {"profile_row", "profile_msa_row"}


def uses_msa_embedding_memory(memory_mode: str) -> bool:
    return memory_mode in {"profile_msa", "profile_msa_row", "profile_msa_axial"}


def uses_axial_msa_memory(memory_mode: str) -> bool:
    return memory_mode == "profile_msa_axial"


def uses_gap_inclusive_msa_mask(memory_mode: str) -> bool:
    return uses_axial_msa_memory(memory_mode)


@dataclass(frozen=True)
class RowExample:
    cluster_index: str
    split: str
    npz_path: Path
    metadata_path: Path
    row_index: int
    kegg_entry: str
    aligned_sequence: str
    target_sequence: str


def open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def split_values(text: str) -> tuple[str, ...]:
    values: list[str] = []
    for part in str(text).replace(",", ";").split(";"):
        value = part.strip()
        if value and value.lower() != "nan":
            values.append(value)
    return tuple(dict.fromkeys(values))


def stable_hash(text: str, buckets: int) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % buckets


def parse_float(text: str) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def transform_numeric(field: str, value: float) -> float | None:
    if field in LOG_NUMERIC_FIELDS:
        if value <= 0:
            return None
        return math.log10(value)
    return value


def ungap(sequence: str) -> str:
    return "".join(char for char in sequence.upper() if char not in {"-", ".", " ", "\n", "\r", "\t"})


def parse_path_rewrites(rewrites: Iterable[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for rewrite in rewrites:
        if "=" not in rewrite:
            raise ValueError(f"path rewrite must have OLD=NEW form: {rewrite}")
        old, new = rewrite.split("=", 1)
        if not old:
            raise ValueError(f"path rewrite has empty OLD prefix: {rewrite}")
        parsed.append((old, new))
    return parsed


def rewrite_manifest_path(text: str, path_rewrites: list[tuple[str, str]]) -> str:
    for old, new in path_rewrites:
        if text == old:
            return new
        if text.startswith(old.rstrip("/") + "/"):
            return new.rstrip("/") + text[len(old.rstrip("/")) :]
    return text


def read_embedding_manifest(
    path: Path,
    split: str | None = None,
    path_rewrites: list[tuple[str, str]] | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    rewrites = path_rewrites or []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") != "embedded":
                continue
            if split and row.get("split") != split:
                continue
            cluster_index = row.get("cluster_index", "")
            if not cluster_index or cluster_index in seen:
                continue
            seen.add(cluster_index)
            row = dict(row)
            row["npz_path"] = rewrite_manifest_path(row.get("npz_path", ""), rewrites)
            row["metadata_path"] = rewrite_manifest_path(row.get("metadata_path", ""), rewrites)
            row["source_msa"] = rewrite_manifest_path(row.get("source_msa", ""), rewrites)
            npz_path = Path(row["npz_path"])
            metadata_path = Path(row["metadata_path"])
            if not npz_path.exists() or not metadata_path.exists():
                continue
            rows.append(row)
    return rows


def load_label_summary(path: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    started = time.monotonic()
    with open_text(path, "r") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            kegg_entry = row.get("kegg_entry")
            if kegg_entry:
                labels[kegg_entry] = row
    print(f"Loaded {len(labels):,} label-summary rows from {path} in {time.monotonic() - started:.1f}s", flush=True)
    return labels


def build_examples(
    rows: Iterable[dict[str, str]],
    max_rows_per_msa: int | None,
) -> list[RowExample]:
    examples: list[RowExample] = []
    for row in rows:
        metadata_path = Path(row["metadata_path"])
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping metadata {metadata_path}: {exc}", file=sys.stderr)
            continue
        headers = [str(header).split()[0] for header in metadata.get("headers", [])]
        sequences = [str(sequence).upper() for sequence in metadata.get("cleaned_sequences", [])]
        if not headers or len(headers) != len(sequences):
            print(f"warning: skipping {metadata_path}: missing aligned headers/sequences", file=sys.stderr)
            continue
        if len(sequences) <= 1:
            continue
        row_count = len(sequences)
        if max_rows_per_msa is not None:
            row_count = min(row_count, max_rows_per_msa)
        for row_index in range(row_count):
            aligned = sequences[row_index]
            if not aligned or any(char not in ALLOWED_TARGET_TOKENS | {"-"} for char in aligned):
                continue
            target = ungap(aligned)
            if not target or any(char not in ALLOWED_TARGET_TOKENS for char in target):
                continue
            examples.append(
                RowExample(
                    cluster_index=row["cluster_index"],
                    split=row.get("split", "train"),
                    npz_path=Path(row["npz_path"]),
                    metadata_path=metadata_path,
                    row_index=row_index,
                    kegg_entry=headers[row_index],
                    aligned_sequence=aligned,
                    target_sequence=target,
                )
            )
    return examples


def build_msa_groups(examples: list[RowExample]) -> list[list[RowExample]]:
    groups: OrderedDict[tuple[str, Path, Path], list[RowExample]] = OrderedDict()
    for example in examples:
        key = (example.cluster_index, example.npz_path, example.metadata_path)
        groups.setdefault(key, []).append(example)
    return [group for group in groups.values() if group]


def numeric_normalization(
    examples: Iterable[RowExample],
    labels: dict[str, dict[str, str]],
) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    values_by_field: dict[str, list[float]] = {field: [] for field in NUMERIC_FIELDS}
    for example in examples:
        row = labels.get(example.kegg_entry)
        if not row:
            continue
        for field in NUMERIC_FIELDS:
            value = parse_float(row.get(f"{field}_mean", ""))
            if value is None:
                continue
            transformed = transform_numeric(field, value)
            if transformed is not None:
                values_by_field[field].append(transformed)
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    counts: dict[str, int] = {}
    for field, values in values_by_field.items():
        counts[field] = len(values)
        if not values:
            means[field] = 0.0
            stds[field] = 1.0
            continue
        array = np.array(values, dtype=np.float32)
        means[field] = float(array.mean())
        std = float(array.std())
        stds[field] = std if std > 1.0e-6 else 1.0
    return means, stds, counts


def condition_features(
    labels: dict[str, dict[str, str]],
    kegg_entry: str,
    numeric_means: dict[str, float],
    numeric_stds: dict[str, float],
    category_buckets: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    row = labels.get(kegg_entry, {})
    numeric_values = np.zeros((len(NUMERIC_FIELDS),), dtype=np.float32)
    numeric_mask = np.zeros((len(NUMERIC_FIELDS),), dtype=np.bool_)
    for idx, field in enumerate(NUMERIC_FIELDS):
        value = parse_float(row.get(f"{field}_mean", ""))
        if value is None:
            continue
        transformed = transform_numeric(field, value)
        if transformed is None:
            continue
        numeric_values[idx] = (transformed - numeric_means[field]) / numeric_stds[field]
        numeric_mask[idx] = True

    category_ids = np.full((len(CATEGORICAL_FIELDS),), -1, dtype=np.int64)
    category_mask = np.zeros((len(CATEGORICAL_FIELDS),), dtype=np.bool_)
    for idx, field in enumerate(CATEGORICAL_FIELDS):
        values = split_values(row.get(f"{field}_values", ""))
        if values:
            joined = "|".join(values)
            category_ids[idx] = stable_hash(f"{field}:{joined}", category_buckets)
            category_mask[idx] = True
    return numeric_values, numeric_mask, category_ids, category_mask


def profile_excluding_rows(sequences: list[str], row_indices: set[int], length: int) -> np.ndarray:
    features = np.zeros((length, RAW_PROFILE_DIM), dtype=np.float32)
    valid_targets = {idx for idx in row_indices if 0 <= idx < len(sequences)}
    total_other = max(len(sequences) - len(valid_targets), 1)
    for idx, sequence in enumerate(sequences):
        if idx in valid_targets:
            continue
        for col, char in enumerate(sequence[:length]):
            if char == "-":
                features[col, len(AA_TOKENS)] += 1.0
            elif char in AA_TO_COL:
                features[col, AA_TO_COL[char]] += 1.0
    counts = features[:, : len(AA_TOKENS)].sum(axis=1) + features[:, len(AA_TOKENS)]
    nonzero = counts > 0
    features[nonzero, : len(AA_TOKENS) + 1] /= counts[nonzero, None]
    features[:, -1] = counts / float(total_other)
    return features


def profile_excluding_row(sequences: list[str], row_index: int, length: int) -> np.ndarray:
    return profile_excluding_rows(sequences, {row_index}, length)


def msa_embeddings_excluding_rows(
    item: dict[str, Any],
    row_indices: set[int],
    length: int,
    gap_inclusive_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if "token_embeddings" not in item:
        raise RuntimeError(
            f"{item.get('npz_path', 'embedding NPZ')} does not contain token_embeddings; "
            "precompute ESM-MSA embeddings with --store-token-embeddings for profile_msa modes"
        )
    token_embeddings = item["token_embeddings"]
    aa_mask = item["aa_mask"]
    sequences = item["sequences"]
    row_count = min(token_embeddings.shape[0], len(sequences))
    col_count = min(length, token_embeddings.shape[1])
    valid_targets = {idx for idx in row_indices if 0 <= idx < row_count}
    row_mask = np.ones((row_count,), dtype=np.bool_)
    for row_index in valid_targets:
        row_mask[row_index] = False
    embeddings = token_embeddings[:row_count, :col_count][row_mask]
    if gap_inclusive_mask:
        mask = np.ones((embeddings.shape[0], embeddings.shape[1]), dtype=np.bool_)
    else:
        mask = aa_mask[:row_count, :col_count][row_mask]
    return embeddings, mask


def target_row_residue_embeddings(
    item: dict[str, Any],
    row_index: int,
    target_sequence: str,
) -> tuple[np.ndarray, np.ndarray]:
    if "token_embeddings" not in item:
        raise RuntimeError(
            f"{item.get('npz_path', 'embedding NPZ')} does not contain token_embeddings; "
            "target-row continuous targets require precomputed per-token ESM-MSA embeddings"
        )
    token_embeddings = item["token_embeddings"]
    aa_mask = item["aa_mask"]
    sequences = item["sequences"]
    row_count = min(token_embeddings.shape[0], len(sequences))
    if not 0 <= row_index < row_count:
        raise IndexError(f"row_index={row_index} outside token embedding rows={row_count}")
    col_count = min(len(sequences[row_index]), token_embeddings.shape[1], aa_mask.shape[1])
    residue_embeddings = token_embeddings[row_index, :col_count][aa_mask[row_index, :col_count]]
    target_length = min(len(target_sequence), residue_embeddings.shape[0])
    embeddings = residue_embeddings[:target_length]
    mask = np.ones((target_length,), dtype=np.bool_)
    return embeddings, mask


def limit_msa_context_rows(
    msa_embeddings: np.ndarray,
    msa_embedding_mask: np.ndarray,
    max_msa_context_rows: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if max_msa_context_rows is None or max_msa_context_rows <= 0:
        return msa_embeddings, msa_embedding_mask
    row_count = msa_embeddings.shape[0]
    if row_count <= max_msa_context_rows:
        return msa_embeddings, msa_embedding_mask
    indices = np.random.choice(row_count, size=max_msa_context_rows, replace=False)
    indices.sort()
    return msa_embeddings[indices], msa_embedding_mask[indices]


def msa_embeddings_excluding_row(
    item: dict[str, Any],
    row_index: int,
    length: int,
    gap_inclusive_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    return msa_embeddings_excluding_rows(
        item,
        {row_index},
        length,
        gap_inclusive_mask=gap_inclusive_mask,
    )


def consensus_training_features(
    profile: np.ndarray,
    aligned_target: str,
    consensus_loss_mode: str,
    consensus_match_weight: float,
    nonconsensus_weight: float,
    unobserved_nonconsensus_weight: float,
    max_sequence_loss_weight: float,
    variable_column_min_entropy: float,
    variable_column_max_consensus: float,
) -> dict[str, np.ndarray]:
    target_chars = [char for char in aligned_target.upper() if char in AA_TO_COL]
    target_len = len(target_chars)
    position_loss_weights = np.ones((target_len,), dtype=np.float32)
    consensus_observed_mask = np.zeros((target_len,), dtype=np.bool_)
    consensus_match_mask = np.zeros((target_len,), dtype=np.bool_)
    nonconsensus_mask = np.zeros((target_len,), dtype=np.bool_)
    variable_nonconsensus_mask = np.zeros((target_len,), dtype=np.bool_)
    profile_variable_mask = np.zeros((profile.shape[0],), dtype=np.bool_)

    residue_pos = 0
    for col, target_char in enumerate(aligned_target.upper()[: profile.shape[0]]):
        aa_col = AA_TO_COL.get(target_char)
        aa_mass = float(profile[col, : len(AA_TOKENS)].sum())
        aa_observed = aa_mass > 1.0e-8
        entropy = 0.0
        max_freq = 0.0
        target_freq = 0.0
        consensus_char = ""
        variable_col = False
        if aa_observed:
            aa_probs = profile[col, : len(AA_TOKENS)] / aa_mass
            max_index = int(np.argmax(aa_probs))
            max_freq = float(aa_probs[max_index])
            target_freq = float(aa_probs[aa_col]) if aa_col is not None else 0.0
            nonzero = aa_probs > 0.0
            entropy = float(-(aa_probs[nonzero] * np.log(aa_probs[nonzero])).sum() / math.log(len(AA_TOKENS)))
            consensus_char = AA_TOKENS[max_index]
            variable_col = entropy >= variable_column_min_entropy and max_freq <= variable_column_max_consensus
            profile_variable_mask[col] = variable_col

        if aa_col is None:
            continue

        if aa_observed:
            consensus_observed_mask[residue_pos] = True
            if consensus_char == target_char:
                consensus_match_mask[residue_pos] = True
                if consensus_loss_mode == "residual":
                    position_loss_weights[residue_pos] = consensus_match_weight
            else:
                nonconsensus_mask[residue_pos] = True
                if variable_col and target_freq > 0.0:
                    variable_nonconsensus_mask[residue_pos] = True
                    if consensus_loss_mode == "residual":
                        position_loss_weights[residue_pos] = nonconsensus_weight
                elif consensus_loss_mode == "residual":
                    position_loss_weights[residue_pos] = unobserved_nonconsensus_weight
        residue_pos += 1

    position_loss_weights = np.clip(position_loss_weights, 0.0, max_sequence_loss_weight).astype(np.float32)
    return {
        "position_loss_weights": position_loss_weights,
        "consensus_observed_mask": consensus_observed_mask,
        "consensus_match_mask": consensus_match_mask,
        "nonconsensus_mask": nonconsensus_mask,
        "variable_nonconsensus_mask": variable_nonconsensus_mask,
        "profile_variable_mask": profile_variable_mask,
    }


def masked_scalar_loss(losses: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=losses.dtype)
    total = weights.sum()
    if torch.sum(total.detach()) <= 0:
        return losses.sum() * 0.0
    return (losses * weights).sum() / total.clamp_min(1.0)


def masked_mse(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_scalar_loss(F.mse_loss(predicted, target, reduction="none"), mask)


def masked_mae(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_scalar_loss(torch.abs(predicted - target), mask)


def masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_scalar_loss(F.binary_cross_entropy_with_logits(logits, target, reduction="none"), mask)


def masked_binary_accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    predictions = logits > 0
    correct = predictions == target.to(dtype=torch.bool)
    return masked_scalar_loss(correct.to(dtype=logits.dtype), mask)


def masked_category_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if torch.sum(mask.detach()) <= 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target[mask], reduction="mean")


def masked_category_accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if torch.sum(mask.detach()) <= 0:
        return logits.sum() * 0.0
    predictions = torch.argmax(logits, dim=-1)
    return (predictions[mask] == target[mask]).to(dtype=logits.dtype).mean()


class CachedMSAStore:
    def __init__(self, cache_size: int, msa_embedding_dtype: str) -> None:
        self.cache_size = max(cache_size, 0)
        if msa_embedding_dtype not in MSA_EMBEDDING_DTYPES:
            raise ValueError(f"msa_embedding_dtype must be one of {MSA_EMBEDDING_DTYPES}")
        self.msa_embedding_dtype = msa_embedding_dtype
        self.cache: OrderedDict[Path, dict[str, Any]] = OrderedDict()

    def load(self, npz_path: Path, metadata_path: Path) -> dict[str, Any]:
        if npz_path in self.cache:
            item = self.cache.pop(npz_path)
            self.cache[npz_path] = item
            return item
        arrays = np.load(npz_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        item = {
            "npz_path": str(npz_path),
            "row_embeddings": arrays["row_embeddings"].astype(np.float32),
            "aa_mask": arrays["aa_mask"].astype(np.bool_),
            "headers": [str(header).split()[0] for header in metadata["headers"]],
            "sequences": [str(sequence).upper() for sequence in metadata["cleaned_sequences"]],
        }
        if "token_embeddings" in arrays:
            token_embeddings = arrays["token_embeddings"]
            if self.msa_embedding_dtype == "float32":
                token_embeddings = token_embeddings.astype(np.float32)
            elif self.msa_embedding_dtype == "float16":
                token_embeddings = token_embeddings.astype(np.float16)
            item["token_embeddings"] = token_embeddings
        if self.cache_size:
            self.cache[npz_path] = item
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return item


class CachedMSARowDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        examples: list[RowExample],
        labels: dict[str, dict[str, str]],
        numeric_means: dict[str, float],
        numeric_stds: dict[str, float],
        category_buckets: int,
        cache_size: int,
        consensus_loss_mode: str,
        consensus_match_weight: float,
        nonconsensus_weight: float,
        unobserved_nonconsensus_weight: float,
        max_sequence_loss_weight: float,
        variable_column_min_entropy: float,
        variable_column_max_consensus: float,
        require_msa_embeddings: bool,
        msa_embedding_dtype: str,
        max_msa_context_rows: int | None,
        gap_inclusive_msa_mask: bool,
        require_target_continuous_embeddings: bool = False,
    ) -> None:
        self.examples = examples
        self.labels = labels
        self.numeric_means = numeric_means
        self.numeric_stds = numeric_stds
        self.category_buckets = category_buckets
        self.store = CachedMSAStore(cache_size=cache_size, msa_embedding_dtype=msa_embedding_dtype)
        self.consensus_loss_mode = consensus_loss_mode
        self.consensus_match_weight = consensus_match_weight
        self.nonconsensus_weight = nonconsensus_weight
        self.unobserved_nonconsensus_weight = unobserved_nonconsensus_weight
        self.max_sequence_loss_weight = max_sequence_loss_weight
        self.variable_column_min_entropy = variable_column_min_entropy
        self.variable_column_max_consensus = variable_column_max_consensus
        self.require_msa_embeddings = require_msa_embeddings
        self.require_target_continuous_embeddings = require_target_continuous_embeddings
        self.max_msa_context_rows = max_msa_context_rows
        self.gap_inclusive_msa_mask = gap_inclusive_msa_mask

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        item = self.store.load(example.npz_path, example.metadata_path)
        sequences = item["sequences"]
        if example.row_index >= len(sequences):
            raise IndexError(f"row_index={example.row_index} outside MSA with {len(sequences)} rows")
        length = len(sequences[example.row_index])
        profile = profile_excluding_row(sequences, example.row_index, length)
        if self.require_msa_embeddings:
            msa_embeddings, msa_embedding_mask = msa_embeddings_excluding_row(
                item,
                example.row_index,
                length,
                gap_inclusive_mask=self.gap_inclusive_msa_mask,
            )
            msa_embeddings, msa_embedding_mask = limit_msa_context_rows(
                msa_embeddings,
                msa_embedding_mask,
                self.max_msa_context_rows,
            )
        else:
            msa_embeddings = np.zeros((0, length, 1), dtype=np.float32)
            msa_embedding_mask = np.zeros((0, length), dtype=np.bool_)
        if self.require_target_continuous_embeddings:
            target_continuous_embeddings, target_continuous_mask = target_row_residue_embeddings(
                item,
                example.row_index,
                example.target_sequence,
            )
        else:
            target_continuous_embeddings = np.zeros((0, 1), dtype=np.float32)
            target_continuous_mask = np.zeros((0,), dtype=np.bool_)
        consensus_features = consensus_training_features(
            profile=profile,
            aligned_target=sequences[example.row_index],
            consensus_loss_mode=self.consensus_loss_mode,
            consensus_match_weight=self.consensus_match_weight,
            nonconsensus_weight=self.nonconsensus_weight,
            unobserved_nonconsensus_weight=self.unobserved_nonconsensus_weight,
            max_sequence_loss_weight=self.max_sequence_loss_weight,
            variable_column_min_entropy=self.variable_column_min_entropy,
            variable_column_max_consensus=self.variable_column_max_consensus,
        )

        row_embeddings = item["row_embeddings"]
        row_count = min(row_embeddings.shape[0], len(sequences))
        row_mask = np.ones((row_count,), dtype=np.bool_)
        row_mask[example.row_index] = False
        non_target_rows = row_embeddings[:row_count][row_mask]

        numeric_values, numeric_mask, category_ids, category_mask = condition_features(
            self.labels,
            example.kegg_entry,
            self.numeric_means,
            self.numeric_stds,
            self.category_buckets,
        )

        return {
            "profile": profile,
            "msa_embeddings": msa_embeddings,
            "msa_embedding_mask": msa_embedding_mask,
            **consensus_features,
            "row_embeddings": non_target_rows,
            "target_continuous_embeddings": target_continuous_embeddings,
            "target_continuous_mask": target_continuous_mask,
            "target_sequence": example.target_sequence,
            "numeric_values": numeric_values,
            "numeric_mask": numeric_mask,
            "category_ids": category_ids,
            "category_mask": category_mask,
            "cluster_index": example.cluster_index,
            "kegg_entry": example.kegg_entry,
            "row_index": example.row_index,
        }


class CachedMSAMaskedRowsDataset(Dataset[list[dict[str, Any]]]):
    def __init__(
        self,
        groups: list[list[RowExample]],
        labels: dict[str, dict[str, str]],
        numeric_means: dict[str, float],
        numeric_stds: dict[str, float],
        category_buckets: int,
        cache_size: int,
        consensus_loss_mode: str,
        consensus_match_weight: float,
        nonconsensus_weight: float,
        unobserved_nonconsensus_weight: float,
        max_sequence_loss_weight: float,
        variable_column_min_entropy: float,
        variable_column_max_consensus: float,
        masked_rows_per_msa_min: int,
        masked_rows_per_msa_max: int,
        require_msa_embeddings: bool,
        msa_embedding_dtype: str,
        max_msa_context_rows: int | None,
        gap_inclusive_msa_mask: bool,
        require_target_continuous_embeddings: bool = False,
    ) -> None:
        self.groups = groups
        self.labels = labels
        self.numeric_means = numeric_means
        self.numeric_stds = numeric_stds
        self.category_buckets = category_buckets
        self.store = CachedMSAStore(cache_size=cache_size, msa_embedding_dtype=msa_embedding_dtype)
        self.consensus_loss_mode = consensus_loss_mode
        self.consensus_match_weight = consensus_match_weight
        self.nonconsensus_weight = nonconsensus_weight
        self.unobserved_nonconsensus_weight = unobserved_nonconsensus_weight
        self.max_sequence_loss_weight = max_sequence_loss_weight
        self.variable_column_min_entropy = variable_column_min_entropy
        self.variable_column_max_consensus = variable_column_max_consensus
        self.masked_rows_per_msa_min = masked_rows_per_msa_min
        self.masked_rows_per_msa_max = masked_rows_per_msa_max
        self.require_msa_embeddings = require_msa_embeddings
        self.require_target_continuous_embeddings = require_target_continuous_embeddings
        self.max_msa_context_rows = max_msa_context_rows
        self.gap_inclusive_msa_mask = gap_inclusive_msa_mask

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> list[dict[str, Any]]:
        group = self.groups[index]
        if not group:
            raise IndexError("empty MSA group")
        item = self.store.load(group[0].npz_path, group[0].metadata_path)
        sequences = item["sequences"]
        eligible = [example for example in group if example.row_index < len(sequences)]
        if not eligible:
            raise IndexError(f"no eligible rows in MSA group index={index}")
        max_targets = min(self.masked_rows_per_msa_max, max(len(eligible) - 1, 1))
        min_targets = min(self.masked_rows_per_msa_min, max_targets)
        target_count = random.randint(min_targets, max_targets)
        selected = random.sample(eligible, target_count)
        selected_indices = {example.row_index for example in selected}
        length = len(sequences[selected[0].row_index])
        profile = profile_excluding_rows(sequences, selected_indices, length)
        if self.require_msa_embeddings:
            msa_embeddings, msa_embedding_mask = msa_embeddings_excluding_rows(
                item,
                selected_indices,
                length,
                gap_inclusive_mask=self.gap_inclusive_msa_mask,
            )
            msa_embeddings, msa_embedding_mask = limit_msa_context_rows(
                msa_embeddings,
                msa_embedding_mask,
                self.max_msa_context_rows,
            )
        else:
            msa_embeddings = np.zeros((0, length, 1), dtype=np.float32)
            msa_embedding_mask = np.zeros((0, length), dtype=np.bool_)

        row_embeddings = item["row_embeddings"]
        row_count = min(row_embeddings.shape[0], len(sequences))
        row_mask = np.ones((row_count,), dtype=np.bool_)
        for row_index in selected_indices:
            if row_index < row_count:
                row_mask[row_index] = False
        non_target_rows = row_embeddings[:row_count][row_mask]

        outputs: list[dict[str, Any]] = []
        for example in selected:
            if self.require_target_continuous_embeddings:
                target_continuous_embeddings, target_continuous_mask = target_row_residue_embeddings(
                    item,
                    example.row_index,
                    example.target_sequence,
                )
            else:
                target_continuous_embeddings = np.zeros((0, 1), dtype=np.float32)
                target_continuous_mask = np.zeros((0,), dtype=np.bool_)
            consensus_features = consensus_training_features(
                profile=profile,
                aligned_target=sequences[example.row_index],
                consensus_loss_mode=self.consensus_loss_mode,
                consensus_match_weight=self.consensus_match_weight,
                nonconsensus_weight=self.nonconsensus_weight,
                unobserved_nonconsensus_weight=self.unobserved_nonconsensus_weight,
                max_sequence_loss_weight=self.max_sequence_loss_weight,
                variable_column_min_entropy=self.variable_column_min_entropy,
                variable_column_max_consensus=self.variable_column_max_consensus,
            )
            numeric_values, numeric_mask, category_ids, category_mask = condition_features(
                self.labels,
                example.kegg_entry,
                self.numeric_means,
                self.numeric_stds,
                self.category_buckets,
            )
            outputs.append(
                {
                    "profile": profile,
                    "msa_embeddings": msa_embeddings,
                    "msa_embedding_mask": msa_embedding_mask,
                    **consensus_features,
                    "row_embeddings": non_target_rows,
                    "target_continuous_embeddings": target_continuous_embeddings,
                    "target_continuous_mask": target_continuous_mask,
                    "target_sequence": example.target_sequence,
                    "numeric_values": numeric_values,
                    "numeric_mask": numeric_mask,
                    "category_ids": category_ids,
                    "category_mask": category_mask,
                    "cluster_index": example.cluster_index,
                    "kegg_entry": example.kegg_entry,
                    "row_index": example.row_index,
                }
            )
        return outputs


class RowReconstructionCollator:
    def __init__(
        self,
        max_sequence_length: int,
        tail_stop_weight: float,
        profile_feature_mode: str,
        share_msa_embeddings: bool = False,
    ) -> None:
        self.max_sequence_length = max_sequence_length
        self.tail_stop_weight = tail_stop_weight
        self.profile_feature_mode = profile_feature_mode
        self.profile_dim = profile_input_dim(profile_feature_mode)
        self.share_msa_embeddings = bool(share_msa_embeddings)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        flattened: list[dict[str, Any]] = []
        msa_batch: list[dict[str, Any]] = []
        target_msa_group_indices: list[int] = []
        for item in batch:
            group_items = item if isinstance(item, list) else [item]
            if not group_items:
                continue
            if self.share_msa_embeddings:
                msa_group_index = len(msa_batch)
                msa_batch.append(group_items[0])
            for group_item in group_items:
                if not self.share_msa_embeddings:
                    msa_group_index = len(msa_batch)
                    msa_batch.append(group_item)
                flattened.append(group_item)
                target_msa_group_indices.append(msa_group_index)
        target_batch = flattened
        batch_size = len(target_batch)
        msa_batch_size = len(msa_batch)
        max_cols = max(item["profile"].shape[0] for item in target_batch)
        max_rows = max(item["row_embeddings"].shape[0] for item in target_batch)
        row_dim = target_batch[0]["row_embeddings"].shape[-1]
        max_msa_rows = max(1, max(item["msa_embeddings"].shape[0] for item in msa_batch))
        msa_embedding_dim = target_batch[0]["msa_embeddings"].shape[-1]
        msa_embedding_dtype = (
            target_batch[0]["msa_embeddings"].dtype
            if np.issubdtype(target_batch[0]["msa_embeddings"].dtype, np.floating)
            else np.float32
        )
        target_continuous_dim = target_batch[0]["target_continuous_embeddings"].shape[-1]
        target_continuous_dtype = (
            target_batch[0]["target_continuous_embeddings"].dtype
            if np.issubdtype(target_batch[0]["target_continuous_embeddings"].dtype, np.floating)
            else np.float32
        )

        profiles = np.zeros((batch_size, max_cols, self.profile_dim), dtype=np.float32)
        profile_mask = np.zeros((batch_size, max_cols), dtype=np.bool_)
        row_embeddings = np.zeros((batch_size, max_rows, row_dim), dtype=np.float32)
        row_mask = np.zeros((batch_size, max_rows), dtype=np.bool_)
        msa_embeddings = np.zeros((msa_batch_size, max_msa_rows, max_cols, msa_embedding_dim), dtype=msa_embedding_dtype)
        msa_embedding_mask = np.zeros((msa_batch_size, max_msa_rows, max_cols), dtype=np.bool_)
        target_continuous_embeddings = np.zeros(
            (batch_size, self.max_sequence_length, target_continuous_dim),
            dtype=target_continuous_dtype,
        )
        target_continuous_mask = np.zeros((batch_size, self.max_sequence_length), dtype=np.bool_)
        numeric_values = np.zeros((batch_size, len(NUMERIC_FIELDS)), dtype=np.float32)
        numeric_mask = np.zeros((batch_size, len(NUMERIC_FIELDS)), dtype=np.bool_)
        category_ids = np.full((batch_size, len(CATEGORICAL_FIELDS)), -1, dtype=np.int64)
        category_mask = np.zeros((batch_size, len(CATEGORICAL_FIELDS)), dtype=np.bool_)
        profile_variable_mask = np.zeros((batch_size, max_cols), dtype=np.bool_)
        target_sequences: list[str] = []
        cluster_indices: list[str] = []
        kegg_entries: list[str] = []
        row_indices: list[int] = []

        for idx, item in enumerate(target_batch):
            col_count = item["profile"].shape[0]
            row_count = item["row_embeddings"].shape[0]
            profiles[idx, :col_count] = select_profile_features(item["profile"], self.profile_feature_mode)
            profile_mask[idx, :col_count] = True
            profile_variable_mask[idx, :col_count] = item["profile_variable_mask"]
            if row_count:
                row_embeddings[idx, :row_count] = item["row_embeddings"]
                row_mask[idx, :row_count] = True
            target_continuous_length = min(item["target_continuous_embeddings"].shape[0], self.max_sequence_length)
            if target_continuous_length:
                target_continuous_embeddings[idx, :target_continuous_length] = item[
                    "target_continuous_embeddings"
                ][:target_continuous_length]
                target_continuous_mask[idx, :target_continuous_length] = item["target_continuous_mask"][
                    :target_continuous_length
                ]
            numeric_values[idx] = item["numeric_values"]
            numeric_mask[idx] = item["numeric_mask"]
            category_ids[idx] = item["category_ids"]
            category_mask[idx] = item["category_mask"]
            target_sequences.append(item["target_sequence"])
            cluster_indices.append(item["cluster_index"])
            kegg_entries.append(item["kegg_entry"])
            row_indices.append(int(item["row_index"]))

        for idx, item in enumerate(msa_batch):
            col_count = item["profile"].shape[0]
            msa_row_count = item["msa_embeddings"].shape[0]
            if msa_row_count:
                msa_embeddings[idx, :msa_row_count, :col_count] = item["msa_embeddings"]
                msa_embedding_mask[idx, :msa_row_count, :col_count] = item["msa_embedding_mask"]

        target_tokens, loss_weights = batch_encode_sequences_with_stop(
            target_sequences,
            max_length=self.max_sequence_length,
            tail_stop_weight=self.tail_stop_weight,
        )
        sequence_loss_weights = loss_weights.clone()
        consensus_observed_mask = torch.zeros_like(loss_weights, dtype=torch.bool)
        consensus_match_mask = torch.zeros_like(loss_weights, dtype=torch.bool)
        nonconsensus_mask = torch.zeros_like(loss_weights, dtype=torch.bool)
        variable_nonconsensus_mask = torch.zeros_like(loss_weights, dtype=torch.bool)
        for idx, item in enumerate(target_batch):
            residue_count = min(len(item["target_sequence"]), self.max_sequence_length - 1)
            if residue_count <= 0:
                continue
            sequence_loss_weights[idx, :residue_count] = torch.from_numpy(
                item["position_loss_weights"][:residue_count]
            )
            consensus_observed_mask[idx, :residue_count] = torch.from_numpy(
                item["consensus_observed_mask"][:residue_count]
            )
            consensus_match_mask[idx, :residue_count] = torch.from_numpy(
                item["consensus_match_mask"][:residue_count]
            )
            nonconsensus_mask[idx, :residue_count] = torch.from_numpy(
                item["nonconsensus_mask"][:residue_count]
            )
            variable_nonconsensus_mask[idx, :residue_count] = torch.from_numpy(
                item["variable_nonconsensus_mask"][:residue_count]
            )
        return {
            "profiles": torch.from_numpy(profiles),
            "profile_mask": torch.from_numpy(profile_mask),
            "profile_variable_mask": torch.from_numpy(profile_variable_mask),
            "row_embeddings": torch.from_numpy(row_embeddings),
            "row_mask": torch.from_numpy(row_mask),
            "msa_embeddings": torch.from_numpy(msa_embeddings),
            "msa_embedding_mask": torch.from_numpy(msa_embedding_mask),
            "target_msa_group_indices": torch.tensor(target_msa_group_indices, dtype=torch.long),
            "target_continuous_embeddings": torch.from_numpy(target_continuous_embeddings),
            "target_continuous_mask": torch.from_numpy(target_continuous_mask),
            "target_tokens": target_tokens,
            "loss_weights": loss_weights,
            "sequence_loss_weights": sequence_loss_weights,
            "consensus_observed_mask": consensus_observed_mask,
            "consensus_match_mask": consensus_match_mask,
            "nonconsensus_mask": nonconsensus_mask,
            "variable_nonconsensus_mask": variable_nonconsensus_mask,
            "numeric_values": torch.from_numpy(numeric_values),
            "numeric_mask": torch.from_numpy(numeric_mask),
            "category_ids": torch.from_numpy(category_ids),
            "category_mask": torch.from_numpy(category_mask),
            "target_sequences": target_sequences,
            "cluster_indices": cluster_indices,
            "kegg_entries": kegg_entries,
            "row_indices": row_indices,
        }


class MeanStartCCDDModel(nn.Module):
    def __init__(
        self,
        row_embedding_dim: int,
        d_model: int,
        layers: int,
        heads: int,
        dropout: float,
        max_sequence_length: int,
        diffusion_timesteps: int,
        category_buckets: int,
        memory_mode: str,
        profile_feature_mode: str = "full",
        msa_embedding_dim: int = 1,
        continuous_target_mode: str = "token_embedding",
        target_continuous_dim: int = 1,
        msa_axial_layers: int = 1,
        max_profile_cols: int = 1024,
        max_rows: int = 128,
    ) -> None:
        super().__init__()
        if memory_mode not in MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {MEMORY_MODES}")
        if profile_feature_mode not in PROFILE_FEATURE_MODES:
            raise ValueError(f"profile_feature_mode must be one of {PROFILE_FEATURE_MODES}")
        if continuous_target_mode not in CONTINUOUS_TARGET_MODES:
            raise ValueError(f"continuous_target_mode must be one of {CONTINUOUS_TARGET_MODES}")
        if target_continuous_dim < 1:
            raise ValueError("target_continuous_dim must be positive")
        self.memory_mode = memory_mode
        self.profile_feature_mode = profile_feature_mode
        self.continuous_target_mode = continuous_target_mode
        self.profile_dim = profile_input_dim(profile_feature_mode)
        self.msa_embedding_dim = int(msa_embedding_dim)
        self.target_continuous_dim = int(target_continuous_dim)
        self.profile_proj = nn.Sequential(
            nn.LayerNorm(self.profile_dim),
            nn.Linear(self.profile_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.row_proj = nn.Sequential(
            nn.LayerNorm(row_embedding_dim),
            nn.Linear(row_embedding_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.msa_grid_proj = None
        if uses_axial_msa_memory(memory_mode):
            self.msa_embedding_proj = None
            self.msa_grid_proj = nn.Sequential(
                nn.LayerNorm(self.msa_embedding_dim),
                nn.Linear(self.msa_embedding_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
            )
        elif uses_msa_embedding_memory(memory_mode):
            self.msa_embedding_proj = MSADepthScaler(
                input_dim=self.msa_embedding_dim,
                d_model=d_model,
                dropout=dropout,
            )
        else:
            self.msa_embedding_proj = None
        self.numeric_proj = nn.Sequential(
            nn.Linear(len(NUMERIC_FIELDS) * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(category_buckets, d_model) for _ in CATEGORICAL_FIELDS]
        )
        self.condition_pool_norm = nn.LayerNorm(d_model)
        self.numeric_value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(NUMERIC_FIELDS)),
        )
        self.numeric_presence_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(NUMERIC_FIELDS)),
        )
        self.category_value_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(d_model),
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, category_buckets),
                )
                for _ in CATEGORICAL_FIELDS
            ]
        )
        self.category_presence_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, len(CATEGORICAL_FIELDS)),
        )
        self.target_continuous_head = None
        if self.continuous_target_mode == "target_row_embedding" and self.target_continuous_dim != d_model:
            self.target_continuous_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, self.target_continuous_dim),
            )
        self.profile_pos = nn.Embedding(max_profile_cols, d_model)
        self.row_pos = nn.Embedding(max_rows, d_model)
        self.type_embedding = nn.Embedding(3, d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.null_memory = nn.Parameter(torch.zeros(1, 1, d_model))
        self.decoder = SequenceDiffusionDecoder(
            d_model=d_model,
            max_sequence_length=max_sequence_length,
            num_layers=layers,
            num_heads=heads,
            dropout=dropout,
            num_timesteps=diffusion_timesteps,
            msa_grid_decoder=uses_axial_msa_memory(memory_mode),
            msa_axial_blocks_per_layer=msa_axial_layers,
        )

    def apply_profile_variable_dropout(
        self,
        profiles: torch.Tensor,
        profile_mask: torch.Tensor,
        profile_variable_mask: torch.Tensor | None,
        profile_variable_dropout: float,
        profile_variable_blur: float,
        profile_blur_alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zero = torch.zeros((), dtype=profiles.dtype, device=profiles.device)
        if not self.training or (profile_variable_dropout <= 0.0 and profile_variable_blur <= 0.0):
            return profiles, zero, zero
        if self.profile_feature_mode != "full":
            return profiles, zero, zero

        eligible = profile_mask
        if profile_variable_mask is not None:
            eligible = eligible & profile_variable_mask
        eligible_count = eligible.to(dtype=profiles.dtype).sum().clamp_min(1.0)

        masked_profiles = profiles
        blur_fraction = zero
        if profile_variable_blur > 0.0:
            blur_mask = eligible & (
                torch.rand(eligible.shape, dtype=profiles.dtype, device=profiles.device) < profile_variable_blur
            )
            uniform_profile = profiles.clone()
            uniform_profile[:, :, : len(AA_TOKENS) + 1] = 0.0
            uniform_profile[:, :, : len(AA_TOKENS)] = 1.0 / float(len(AA_TOKENS))
            blended = masked_profiles * (1.0 - profile_blur_alpha) + uniform_profile * profile_blur_alpha
            masked_profiles = torch.where(blur_mask.unsqueeze(-1), blended, masked_profiles)
            blur_fraction = blur_mask.to(dtype=profiles.dtype).sum() / eligible_count

        drop_fraction = zero
        if profile_variable_dropout > 0.0:
            drop_mask = eligible & (
                torch.rand(eligible.shape, dtype=profiles.dtype, device=profiles.device) < profile_variable_dropout
            )
            dropped_profile = masked_profiles.clone()
            dropped_profile[:, :, : len(AA_TOKENS) + 1] = 0.0
            masked_profiles = torch.where(drop_mask.unsqueeze(-1), dropped_profile, masked_profiles)
            drop_fraction = drop_mask.to(dtype=profiles.dtype).sum() / eligible_count

        return masked_profiles, drop_fraction, blur_fraction

    def encode_memory(
        self,
        profiles: torch.Tensor,
        profile_mask: torch.Tensor,
        row_embeddings: torch.Tensor,
        row_mask: torch.Tensor,
        msa_embeddings: torch.Tensor,
        msa_embedding_mask: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        category_ids: torch.Tensor,
        category_mask: torch.Tensor,
        memory_drop_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, profile_len, _ = profiles.shape
        device = profiles.device
        profile_positions = torch.arange(profile_len, device=device).clamp_max(self.profile_pos.num_embeddings - 1)
        profile_tokens = (
            self.profile_proj(profiles)
            + self.profile_pos(profile_positions).unsqueeze(0)
            + self.type_embedding(torch.zeros((), dtype=torch.long, device=device))
        )

        numeric_input = torch.cat([numeric_values, numeric_mask.to(dtype=numeric_values.dtype)], dim=-1)
        condition_tokens = [self.numeric_proj(numeric_input).unsqueeze(1)]
        for idx, embedding in enumerate(self.category_embeddings):
            ids = category_ids[:, idx].clamp_min(0)
            token = embedding(ids)
            token = torch.where(category_mask[:, idx].view(-1, 1), token, torch.zeros_like(token))
            token = token + self.type_embedding(torch.full((), 2, dtype=torch.long, device=device))
            condition_tokens.append(token.unsqueeze(1))
        condition = torch.cat(condition_tokens, dim=1)
        condition_mask = torch.ones((batch_size, condition.shape[1]), dtype=torch.bool, device=device)

        tokens = [condition, profile_tokens]
        masks = [condition_mask, profile_mask]
        if uses_axial_msa_memory(self.memory_mode):
            pass
        elif uses_msa_embedding_memory(self.memory_mode):
            if self.msa_embedding_proj is None:
                raise RuntimeError(f"memory_mode={self.memory_mode} has no MSA embedding projector")
            msa_tokens, msa_mask = self.msa_embedding_proj(msa_embeddings, msa_embedding_mask)
            msa_positions = torch.arange(msa_tokens.shape[1], device=device).clamp_max(
                self.profile_pos.num_embeddings - 1
            )
            msa_tokens = (
                msa_tokens
                + self.profile_pos(msa_positions).unsqueeze(0)
                + self.type_embedding(torch.zeros((), dtype=torch.long, device=device))
            )
            tokens.append(msa_tokens)
            masks.append(msa_mask)
        if uses_row_memory(self.memory_mode):
            row_len = row_embeddings.shape[1]
            row_positions = torch.arange(row_len, device=device).clamp_max(self.row_pos.num_embeddings - 1)
            row_tokens = (
                self.row_proj(row_embeddings)
                + self.row_pos(row_positions).unsqueeze(0)
                + self.type_embedding(torch.ones((), dtype=torch.long, device=device))
            )
            tokens.append(row_tokens)
            masks.append(row_mask)
        memory_tokens = self.memory_norm(torch.cat(tokens, dim=1))
        memory_mask = torch.cat(masks, dim=1)

        if memory_drop_mask is not None:
            dropped_tokens = torch.zeros_like(memory_tokens)
            dropped_mask = torch.zeros_like(memory_mask)
            dropped_tokens[:, :1] = self.null_memory.to(dtype=memory_tokens.dtype)
            dropped_mask[:, :1] = True
            memory_tokens = torch.where(memory_drop_mask.view(-1, 1, 1), dropped_tokens, memory_tokens)
            memory_mask = torch.where(memory_drop_mask.view(-1, 1), dropped_mask, memory_mask)
        return memory_tokens, memory_mask

    def encode_msa_grid(
        self,
        msa_embeddings: torch.Tensor,
        msa_embedding_mask: torch.Tensor,
        memory_drop_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not uses_axial_msa_memory(self.memory_mode):
            return None, None
        if self.msa_grid_proj is None:
            raise RuntimeError(f"memory_mode={self.memory_mode} has no MSA grid projector")
        if msa_embeddings.ndim != 4:
            raise ValueError("msa_embeddings must have shape B x R x C x H")
        if msa_embedding_mask.shape != msa_embeddings.shape[:3]:
            raise ValueError("msa_embedding_mask must have shape B x R x C matching msa_embeddings")

        _, row_count, col_count, _ = msa_embeddings.shape
        device = msa_embeddings.device
        row_positions = torch.arange(row_count, device=device).clamp_max(self.row_pos.num_embeddings - 1)
        col_positions = torch.arange(col_count, device=device).clamp_max(self.profile_pos.num_embeddings - 1)
        grid = (
            self.msa_grid_proj(msa_embeddings)
            + self.row_pos(row_positions).view(1, row_count, 1, -1)
            + self.profile_pos(col_positions).view(1, 1, col_count, -1)
        )
        grid = grid * msa_embedding_mask.to(dtype=grid.dtype).unsqueeze(-1)
        grid_mask = msa_embedding_mask
        if memory_drop_mask is not None:
            grid = torch.where(memory_drop_mask.view(-1, 1, 1, 1), torch.zeros_like(grid), grid)
            grid_mask = torch.where(memory_drop_mask.view(-1, 1, 1), torch.zeros_like(grid_mask), grid_mask)
        return grid, grid_mask

    def condition_prediction_losses(
        self,
        predicted_embeddings: torch.Tensor,
        loss_weights: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        category_ids: torch.Tensor,
        category_mask: torch.Tensor,
        numeric_recovery_mask: torch.Tensor,
        category_recovery_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        sequence_mask = loss_weights > 0.5
        sequence_weights = sequence_mask.to(dtype=predicted_embeddings.dtype)
        pooled = (predicted_embeddings * sequence_weights.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / sequence_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = self.condition_pool_norm(pooled)

        numeric_predictions = self.numeric_value_head(pooled)
        numeric_presence_logits = self.numeric_presence_head(pooled)
        numeric_value_mask = numeric_recovery_mask & numeric_mask
        numeric_presence_target = numeric_mask.to(dtype=predicted_embeddings.dtype)
        numeric_presence_loss = masked_bce_with_logits(
            numeric_presence_logits,
            numeric_presence_target,
            numeric_recovery_mask,
        )

        category_presence_logits = self.category_presence_head(pooled)
        category_presence_target = category_mask.to(dtype=predicted_embeddings.dtype)
        category_presence_loss = masked_bce_with_logits(
            category_presence_logits,
            category_presence_target,
            category_recovery_mask,
        )

        category_losses: list[torch.Tensor] = []
        category_accuracies: list[torch.Tensor] = []
        for idx, head in enumerate(self.category_value_heads):
            logits = head(pooled)
            field_mask = category_recovery_mask[:, idx] & category_mask[:, idx]
            targets = category_ids[:, idx].clamp_min(0)
            category_losses.append(masked_category_cross_entropy(logits, targets, field_mask))
            category_accuracies.append(masked_category_accuracy(logits, targets, field_mask))

        category_value_loss = torch.stack(category_losses).mean()
        category_value_accuracy = torch.stack(category_accuracies).mean()
        condition_mask_fraction = torch.cat(
            [
                numeric_recovery_mask.reshape(numeric_recovery_mask.shape[0], -1),
                category_recovery_mask.reshape(category_recovery_mask.shape[0], -1),
            ],
            dim=1,
        ).to(dtype=predicted_embeddings.dtype).mean()

        return {
            "numeric_predictions": numeric_predictions,
            "numeric_presence_logits": numeric_presence_logits,
            "numeric_value_loss": masked_mse(numeric_predictions, numeric_values, numeric_value_mask),
            "numeric_value_mae": masked_mae(numeric_predictions, numeric_values, numeric_value_mask),
            "numeric_presence_loss": numeric_presence_loss,
            "numeric_presence_accuracy": masked_binary_accuracy(
                numeric_presence_logits,
                numeric_mask,
                numeric_recovery_mask,
            ),
            "category_presence_logits": category_presence_logits,
            "category_value_loss": category_value_loss,
            "category_value_accuracy": category_value_accuracy,
            "category_presence_loss": category_presence_loss,
            "category_presence_accuracy": masked_binary_accuracy(
                category_presence_logits,
                category_mask,
                category_recovery_mask,
            ),
            "condition_mask_fraction": condition_mask_fraction,
        }

    def forward(
        self,
        profiles: torch.Tensor,
        profile_mask: torch.Tensor,
        row_embeddings: torch.Tensor,
        row_mask: torch.Tensor,
        msa_embeddings: torch.Tensor,
        msa_embedding_mask: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        category_ids: torch.Tensor,
        category_mask: torch.Tensor,
        target_tokens: torch.Tensor,
        loss_weights: torch.Tensor,
        sequence_loss_weights: torch.Tensor | None,
        timesteps: torch.Tensor,
        decoder_start_mode: str,
        target_continuous_embeddings: torch.Tensor | None = None,
        target_continuous_mask: torch.Tensor | None = None,
        memory_dropout: float = 0.0,
        condition_mask_prob: float = 0.0,
        profile_variable_mask: torch.Tensor | None = None,
        profile_variable_dropout: float = 0.0,
        profile_variable_blur: float = 0.0,
        profile_blur_alpha: float = 0.5,
        target_group_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if sequence_loss_weights is None:
            sequence_loss_weights = loss_weights
        memory_drop_mask = None
        if self.training and memory_dropout > 0.0:
            memory_drop_mask = torch.rand(
                target_tokens.shape[0],
                dtype=torch.float32,
                device=target_tokens.device,
            ) < memory_dropout
        if not 0.0 <= condition_mask_prob <= 1.0:
            raise ValueError("condition_mask_prob must be in [0, 1]")
        if not 0.0 <= profile_variable_dropout <= 1.0:
            raise ValueError("profile_variable_dropout must be in [0, 1]")
        if not 0.0 <= profile_variable_blur <= 1.0:
            raise ValueError("profile_variable_blur must be in [0, 1]")
        if not 0.0 <= profile_blur_alpha <= 1.0:
            raise ValueError("profile_blur_alpha must be in [0, 1]")
        numeric_recovery_mask = torch.zeros_like(numeric_mask, dtype=torch.bool)
        category_recovery_mask = torch.zeros_like(category_mask, dtype=torch.bool)
        if condition_mask_prob > 0.0:
            numeric_recovery_mask = (
                torch.rand(numeric_mask.shape, dtype=torch.float32, device=numeric_mask.device) < condition_mask_prob
            )
            category_recovery_mask = (
                torch.rand(category_mask.shape, dtype=torch.float32, device=category_mask.device) < condition_mask_prob
            )
        masked_numeric_values = torch.where(numeric_recovery_mask, torch.zeros_like(numeric_values), numeric_values)
        masked_numeric_mask = numeric_mask & ~numeric_recovery_mask
        masked_category_ids = torch.where(category_recovery_mask, torch.full_like(category_ids, -1), category_ids)
        masked_category_mask = category_mask & ~category_recovery_mask
        masked_profiles, profile_drop_fraction, profile_blur_fraction = self.apply_profile_variable_dropout(
            profiles=profiles,
            profile_mask=profile_mask,
            profile_variable_mask=profile_variable_mask,
            profile_variable_dropout=profile_variable_dropout,
            profile_variable_blur=profile_variable_blur,
            profile_blur_alpha=profile_blur_alpha,
        )
        memory_tokens, memory_mask = self.encode_memory(
            profiles=masked_profiles,
            profile_mask=profile_mask,
            row_embeddings=row_embeddings,
            row_mask=row_mask,
            msa_embeddings=msa_embeddings,
            msa_embedding_mask=msa_embedding_mask,
            numeric_values=masked_numeric_values,
            numeric_mask=masked_numeric_mask,
            category_ids=masked_category_ids,
            category_mask=masked_category_mask,
            memory_drop_mask=memory_drop_mask,
        )
        msa_memory_drop_mask = None
        if target_group_indices is None:
            msa_memory_drop_mask = memory_drop_mask
        msa_grid_tokens, msa_grid_mask = self.encode_msa_grid(
            msa_embeddings=msa_embeddings,
            msa_embedding_mask=msa_embedding_mask,
            memory_drop_mask=msa_memory_drop_mask,
        )
        outputs = self.decoder(
            latent_tokens=memory_tokens,
            latent_mask=memory_mask,
            target_tokens=target_tokens,
            loss_weights=sequence_loss_weights,
            timesteps=timesteps,
            decoder_start_mode=decoder_start_mode,
            discrete_loss_corrupted_only=False,
            msa_grid_tokens=msa_grid_tokens,
            msa_grid_mask=msa_grid_mask,
            target_group_indices=target_group_indices,
        )
        if self.continuous_target_mode == "target_row_embedding":
            if target_continuous_embeddings is None:
                raise ValueError("target_continuous_embeddings are required for target_row_embedding mode")
            if target_continuous_embeddings.shape[:2] != target_tokens.shape:
                raise ValueError("target_continuous_embeddings must have shape B x L x H")
            if target_continuous_embeddings.shape[-1] != self.target_continuous_dim:
                raise ValueError(
                    "target_continuous_embeddings hidden dimension does not match "
                    f"target_continuous_dim={self.target_continuous_dim}"
                )
            if target_continuous_mask is None:
                target_continuous_mask = torch.ones_like(target_tokens, dtype=torch.bool)
            if target_continuous_mask.shape != target_tokens.shape:
                raise ValueError("target_continuous_mask must have shape B x L")
            continuous_mask = target_continuous_mask & (loss_weights > 0.5)
            continuous_loss_weights = sequence_loss_weights * continuous_mask.to(dtype=sequence_loss_weights.dtype)
            if self.target_continuous_head is None:
                predicted_continuous = outputs["predicted_embeddings"]
            else:
                predicted_continuous = self.target_continuous_head(outputs["predicted_embeddings"])
            continuous_targets = torch.where(
                target_continuous_mask.unsqueeze(-1),
                target_continuous_embeddings.to(dtype=predicted_continuous.dtype),
                torch.zeros_like(predicted_continuous),
            )
            outputs["predicted_continuous_embeddings"] = predicted_continuous
            outputs["target_continuous_mask_fraction"] = continuous_mask.to(dtype=torch.float32).mean()
            outputs["weighted_continuous_loss"] = weighted_position_mse(
                predicted_continuous,
                continuous_targets,
                continuous_loss_weights,
            )
        else:
            clean_embeddings = self.decoder.token_embedding(target_tokens)
            outputs["weighted_continuous_loss"] = weighted_position_mse(
                outputs["predicted_embeddings"],
                clean_embeddings,
                sequence_loss_weights,
            )
            outputs["target_continuous_mask_fraction"] = torch.ones(
                (),
                dtype=torch.float32,
                device=target_tokens.device,
            )
        outputs["sequence_loss_weight_mean"] = masked_scalar_loss(sequence_loss_weights, loss_weights > 0.5)
        outputs.update(
            self.condition_prediction_losses(
                predicted_embeddings=outputs["predicted_embeddings"],
                loss_weights=loss_weights,
                numeric_values=numeric_values,
                numeric_mask=numeric_mask,
                category_ids=category_ids,
                category_mask=category_mask,
                numeric_recovery_mask=numeric_recovery_mask,
                category_recovery_mask=category_recovery_mask,
            )
        )
        outputs["memory_tokens"] = memory_tokens
        outputs["memory_mask"] = memory_mask
        if msa_grid_mask is not None:
            outputs["msa_grid_valid_fraction"] = msa_grid_mask.to(dtype=torch.float32).mean()
        if memory_drop_mask is not None:
            outputs["memory_drop_fraction"] = memory_drop_mask.to(dtype=torch.float32).mean()
        else:
            outputs["memory_drop_fraction"] = torch.zeros((), dtype=target_tokens.dtype, device=target_tokens.device)
        outputs["profile_drop_fraction"] = profile_drop_fraction
        outputs["profile_blur_fraction"] = profile_blur_fraction
        return outputs


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def sampled_timesteps(batch_size: int, timestep_range: tuple[int, int], device: torch.device) -> torch.Tensor:
    return torch.randint(timestep_range[0], timestep_range[1] + 1, (batch_size,), device=device)


def weighted_residue_accuracy(logits: torch.Tensor, targets: torch.Tensor, loss_weights: torch.Tensor) -> float:
    predicted = torch.argmax(logits, dim=-1)
    correct = predicted == targets
    weights = (loss_weights > 0.5).to(dtype=logits.dtype)
    return float((correct.to(dtype=logits.dtype) * weights).sum().div(weights.sum().clamp_min(1.0)).item())


def masked_residue_accuracy_value(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> float:
    if torch.sum(mask.detach()) <= 0:
        return 0.0
    predicted = torch.argmax(logits, dim=-1)
    correct = (predicted == targets).to(dtype=logits.dtype)
    weights = mask.to(dtype=logits.dtype)
    return float((correct * weights).sum().div(weights.sum().clamp_min(1.0)).item())


def masked_fraction_value(mask: torch.Tensor, denominator_mask: torch.Tensor) -> float:
    denominator = denominator_mask.to(dtype=torch.float32).sum().clamp_min(1.0)
    numerator = (mask & denominator_mask).to(dtype=torch.float32).sum()
    return float((numerator / denominator).item())


def combined_training_loss(
    outputs: dict[str, torch.Tensor],
    continuous_loss_weight: float,
    token_loss_weight: float,
    numeric_condition_loss_weight: float,
    category_condition_loss_weight: float,
    condition_presence_loss_weight: float,
) -> torch.Tensor:
    return (
        continuous_loss_weight * outputs["weighted_continuous_loss"]
        + token_loss_weight * outputs["token_loss"]
        + numeric_condition_loss_weight * outputs["numeric_value_loss"]
        + category_condition_loss_weight * outputs["category_value_loss"]
        + condition_presence_loss_weight
        * (outputs["numeric_presence_loss"] + outputs["category_presence_loss"])
    )


def amp_dtype(amp_mode: str) -> torch.dtype:
    if amp_mode == "fp16":
        return torch.float16
    if amp_mode == "bf16":
        return torch.bfloat16
    raise ValueError(f"amp mode has no autocast dtype: {amp_mode}")


def amp_is_enabled(amp_mode: str, device: torch.device) -> bool:
    return amp_mode != "off" and device.type == "cuda"


def autocast_context(device: torch.device, amp_mode: str) -> contextlib.AbstractContextManager[None]:
    if not amp_is_enabled(amp_mode, device):
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=amp_dtype(amp_mode))


def decode_panel(
    model: MeanStartCCDDModel,
    batch: dict[str, Any],
    device: torch.device,
    amp_mode: str,
    max_sequences: int,
    out_path: Path,
) -> None:
    model.eval()
    moved = move_batch(batch, device)
    timesteps = torch.zeros((moved["target_tokens"].shape[0],), dtype=torch.long, device=device)
    with torch.no_grad(), autocast_context(device, amp_mode):
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
            sequence_loss_weights=moved.get("sequence_loss_weights"),
            target_continuous_embeddings=moved.get("target_continuous_embeddings"),
            target_continuous_mask=moved.get("target_continuous_mask"),
            timesteps=timesteps,
            decoder_start_mode="mean",
            memory_dropout=0.0,
            condition_mask_prob=0.0,
            profile_variable_mask=moved.get("profile_variable_mask"),
            profile_variable_dropout=0.0,
            profile_variable_blur=0.0,
            profile_blur_alpha=0.5,
            target_group_indices=moved.get("target_msa_group_indices"),
        )
        predicted = torch.argmax(outputs["logits"], dim=-1).detach().cpu()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for idx in range(min(max_sequences, predicted.shape[0])):
            decoded = decode_tokens_until_stop(predicted[idx].tolist())
            target = batch["target_sequences"][idx]
            compare = min(len(decoded), len(target))
            identity = (
                sum(1 for pos in range(compare) if decoded[pos] == target[pos]) / max(len(target), 1)
                if target
                else 0.0
            )
            handle.write(
                f">decoded rank={idx + 1} kegg={batch['kegg_entries'][idx]} "
                f"cluster={batch['cluster_indices'][idx]} identity={identity:.4f}\n{decoded}\n"
            )
            handle.write(
                f">target rank={idx + 1} kegg={batch['kegg_entries'][idx]} "
                f"cluster={batch['cluster_indices'][idx]}\n{target}\n"
            )


def evaluate(
    model: MeanStartCCDDModel,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    timestep_range: tuple[int, int],
    decoder_start_mode: str,
    continuous_loss_weight: float,
    token_loss_weight: float,
    numeric_condition_loss_weight: float,
    category_condition_loss_weight: float,
    condition_presence_loss_weight: float,
    condition_mask_prob: float,
    memory_dropout: float,
    profile_variable_dropout: float,
    profile_variable_blur: float,
    profile_blur_alpha: float,
    amp_mode: str,
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    totals = {
        "examples": 0.0,
        "loss": 0.0,
        "continuous_loss": 0.0,
        "token_loss": 0.0,
        "token_accuracy": 0.0,
        "residue_accuracy": 0.0,
        "timestep_mean": 0.0,
        "memory_drop_fraction": 0.0,
    }
    for field in CONSENSUS_METRIC_FIELDS:
        totals[field] = 0.0
    for field in CONDITION_METRIC_FIELDS:
        totals[field] = 0.0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > max_batches:
                break
            moved = move_batch(batch, device)
            batch_size = moved["target_tokens"].shape[0]
            timesteps = sampled_timesteps(batch_size, timestep_range, device)
            with autocast_context(device, amp_mode):
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
                    decoder_start_mode=decoder_start_mode,
                    memory_dropout=memory_dropout,
                    condition_mask_prob=condition_mask_prob,
                    profile_variable_mask=moved["profile_variable_mask"],
                    profile_variable_dropout=profile_variable_dropout,
                    profile_variable_blur=profile_variable_blur,
                    profile_blur_alpha=profile_blur_alpha,
                    target_group_indices=moved.get("target_msa_group_indices"),
                )
                loss = combined_training_loss(
                    outputs,
                    continuous_loss_weight,
                    token_loss_weight,
                    numeric_condition_loss_weight,
                    category_condition_loss_weight,
                    condition_presence_loss_weight,
                )
            totals["examples"] += batch_size
            totals["loss"] += float(loss.item()) * batch_size
            totals["continuous_loss"] += float(outputs["weighted_continuous_loss"].item()) * batch_size
            totals["token_loss"] += float(outputs["token_loss"].item()) * batch_size
            totals["token_accuracy"] += float(outputs["token_accuracy"].item()) * batch_size
            totals["residue_accuracy"] += weighted_residue_accuracy(
                outputs["logits"], moved["target_tokens"], moved["loss_weights"]
            ) * batch_size
            residue_mask = moved["loss_weights"] > 0.5
            totals["sequence_loss_weight_mean"] += float(outputs["sequence_loss_weight_mean"].item()) * batch_size
            totals["consensus_residue_accuracy"] += masked_residue_accuracy_value(
                outputs["logits"],
                moved["target_tokens"],
                moved["consensus_match_mask"] & moved["consensus_observed_mask"] & residue_mask,
            ) * batch_size
            totals["nonconsensus_residue_accuracy"] += masked_residue_accuracy_value(
                outputs["logits"],
                moved["target_tokens"],
                moved["nonconsensus_mask"] & residue_mask,
            ) * batch_size
            totals["nonconsensus_fraction"] += masked_fraction_value(
                moved["nonconsensus_mask"],
                residue_mask,
            ) * batch_size
            totals["variable_nonconsensus_fraction"] += masked_fraction_value(
                moved["variable_nonconsensus_mask"],
                residue_mask,
            ) * batch_size
            totals["profile_variable_fraction"] += masked_fraction_value(
                moved["profile_variable_mask"],
                moved["profile_mask"],
            ) * batch_size
            totals["profile_drop_fraction"] += float(outputs["profile_drop_fraction"].item()) * batch_size
            totals["profile_blur_fraction"] += float(outputs["profile_blur_fraction"].item()) * batch_size
            totals["timestep_mean"] += float(timesteps.to(dtype=torch.float32).mean().item()) * batch_size
            totals["memory_drop_fraction"] += float(outputs["memory_drop_fraction"].item()) * batch_size
            for field in CONDITION_METRIC_FIELDS:
                totals[field] += float(outputs[field].item()) * batch_size
    denom = max(totals["examples"], 1.0)
    for key in list(totals):
        if key != "examples":
            totals[key] /= denom
    model.train()
    return totals


def save_checkpoint(
    path: Path,
    model: MeanStartCCDDModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    numeric_means: dict[str, float],
    numeric_stds: dict[str, float],
    numeric_counts: dict[str, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
            "numeric_means": numeric_means,
            "numeric_stds": numeric_stds,
            "numeric_counts": numeric_counts,
            "numeric_fields": NUMERIC_FIELDS,
            "categorical_fields": CATEGORICAL_FIELDS,
            "sequence_tokens": SEQUENCE_TOKENS,
            "stop_token": STOP_TOKEN,
            "mask_token": MASK_TOKEN,
        },
        path,
    )


def parse_timestep_range(args: argparse.Namespace) -> tuple[int, int]:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-manifest", default=str(DEFAULT_EMBEDDING_MANIFEST))
    parser.add_argument("--label-summary", default=str(DEFAULT_LABEL_SUMMARY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--path-rewrite",
        action="append",
        default=[],
        help="Rewrite manifest paths with OLD=NEW prefixes before opening cached files.",
    )
    parser.add_argument("--memory-mode", choices=MEMORY_MODES, default="profile_row")
    parser.add_argument(
        "--profile-feature-mode",
        choices=PROFILE_FEATURE_MODES,
        default="full",
        help=(
            "Profile features passed to the model. 'full' uses AA frequencies plus gap/coverage; "
            "'no_aa_frequency' removes AA-frequency inputs and keeps only gap/coverage profile channels."
        ),
    )
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-train-msas", type=int, default=None)
    parser.add_argument("--max-val-msas", type=int, default=None)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-val-examples", type=int, default=None)
    parser.add_argument("--max-rows-per-msa", type=int, default=None)
    parser.add_argument(
        "--max-msa-context-rows",
        type=int,
        default=None,
        help="Randomly subsample non-target MSA token-memory context rows to this depth.",
    )
    parser.add_argument(
        "--masked-rows-per-msa-min",
        type=int,
        default=1,
        help="Minimum target rows to mask together from one MSA item.",
    )
    parser.add_argument(
        "--masked-rows-per-msa-max",
        type=int,
        default=1,
        help="Maximum target rows to mask together from one MSA item.",
    )
    parser.add_argument("--cache-size", type=int, default=64)
    parser.add_argument(
        "--msa-embedding-dtype",
        choices=MSA_EMBEDDING_DTYPES,
        default="float32",
        help="Dtype used when loading cached token_embeddings into the data pipeline.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument(
        "--msa-axial-layers",
        type=int,
        default=1,
        help=(
            "Number of static MSA row/column read blocks inside each decoder layer "
            "for memory_mode=profile_msa_axial."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--category-buckets", type=int, default=4096)
    parser.add_argument("--max-sequence-length", type=int, default=1024)
    parser.add_argument("--tail-stop-weight", type=float, default=0.05)
    parser.add_argument("--diffusion-timesteps", type=int, default=250)
    parser.add_argument("--min-diffusion-timestep", type=int, default=0)
    parser.add_argument("--max-diffusion-timestep", type=int, default=-1)
    parser.add_argument("--decoder-start-mode", choices=["mean", "noisy_mean"], default="noisy_mean")
    parser.add_argument(
        "--continuous-target-mode",
        choices=CONTINUOUS_TARGET_MODES,
        default="token_embedding",
        help=(
            "Continuous latent target. 'token_embedding' keeps the historical learned amino-acid embedding target; "
            "'target_row_embedding' predicts the cached target-row ESM-MSA residue embeddings from decoder states."
        ),
    )
    parser.add_argument("--continuous-loss-weight", type=float, default=0.5)
    parser.add_argument("--token-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--consensus-loss-mode",
        choices=["none", "residual"],
        default="none",
        help="Use leave-one-row-out consensus metadata to reweight sequence loss.",
    )
    parser.add_argument(
        "--consensus-match-weight",
        type=float,
        default=0.35,
        help="Residual mode weight for residue positions matching the leave-one-row-out consensus.",
    )
    parser.add_argument(
        "--nonconsensus-weight",
        type=float,
        default=2.5,
        help="Residual mode weight for observed non-consensus residues in variable columns.",
    )
    parser.add_argument(
        "--unobserved-nonconsensus-weight",
        type=float,
        default=1.0,
        help="Residual mode weight for non-consensus residues that are not supported by a variable family column.",
    )
    parser.add_argument(
        "--max-sequence-loss-weight",
        type=float,
        default=3.0,
        help="Clamp per-residue sequence loss weights to this value.",
    )
    parser.add_argument(
        "--variable-column-min-entropy",
        type=float,
        default=0.05,
        help="Minimum normalized AA entropy for a profile column to count as variable.",
    )
    parser.add_argument(
        "--variable-column-max-consensus",
        type=float,
        default=0.92,
        help="Maximum top-AA frequency for a profile column to count as variable.",
    )
    parser.add_argument(
        "--profile-variable-dropout",
        type=float,
        default=0.0,
        help="During training, drop AA/gap profile frequencies at this fraction of variable columns.",
    )
    parser.add_argument(
        "--profile-variable-blur",
        type=float,
        default=0.0,
        help="During training, mix this fraction of variable profile columns toward a uniform AA distribution.",
    )
    parser.add_argument(
        "--profile-blur-alpha",
        type=float,
        default=0.5,
        help="Uniform-profile mix strength for --profile-variable-blur.",
    )
    parser.add_argument(
        "--condition-mask-prob",
        type=float,
        default=0.25,
        help="Randomly hide this fraction of numeric/category condition fields before reconstructing them.",
    )
    parser.add_argument("--numeric-condition-loss-weight", type=float, default=0.2)
    parser.add_argument("--category-condition-loss-weight", type=float, default=0.02)
    parser.add_argument("--condition-presence-loss-weight", type=float, default=0.05)
    parser.add_argument("--memory-dropout", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=64)
    parser.add_argument("--decode-every-steps", type=int, default=500)
    parser.add_argument("--decode-examples", type=int, default=8)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--amp",
        choices=AMP_MODES,
        default="off",
        help="Enable CUDA autocast for model forward/backward. fp16 uses GradScaler.",
    )
    parser.add_argument("--resume-checkpoint", default=None, help="Resume model/optimizer state from this checkpoint.")
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="When resuming, load model weights but restart AdamW state.",
    )
    parser.add_argument(
        "--allow-partial-resume",
        action="store_true",
        help="Allow missing/unexpected checkpoint keys when adding new heads.",
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        path_rewrites = parse_path_rewrites(args.path_rewrite)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not 0.0 <= args.memory_dropout <= 1.0:
        raise SystemExit("--memory-dropout must be in [0, 1]")
    if not 0.0 <= args.condition_mask_prob <= 1.0:
        raise SystemExit("--condition-mask-prob must be in [0, 1]")
    if args.numeric_condition_loss_weight < 0.0:
        raise SystemExit("--numeric-condition-loss-weight must be non-negative")
    if args.category_condition_loss_weight < 0.0:
        raise SystemExit("--category-condition-loss-weight must be non-negative")
    if args.condition_presence_loss_weight < 0.0:
        raise SystemExit("--condition-presence-loss-weight must be non-negative")
    if args.consensus_match_weight < 0.0:
        raise SystemExit("--consensus-match-weight must be non-negative")
    if args.nonconsensus_weight < 0.0:
        raise SystemExit("--nonconsensus-weight must be non-negative")
    if args.unobserved_nonconsensus_weight < 0.0:
        raise SystemExit("--unobserved-nonconsensus-weight must be non-negative")
    if args.max_sequence_loss_weight <= 0.0:
        raise SystemExit("--max-sequence-loss-weight must be positive")
    if not 0.0 <= args.variable_column_min_entropy <= 1.0:
        raise SystemExit("--variable-column-min-entropy must be in [0, 1]")
    if not 0.0 <= args.variable_column_max_consensus <= 1.0:
        raise SystemExit("--variable-column-max-consensus must be in [0, 1]")
    if not 0.0 <= args.profile_variable_dropout <= 1.0:
        raise SystemExit("--profile-variable-dropout must be in [0, 1]")
    if not 0.0 <= args.profile_variable_blur <= 1.0:
        raise SystemExit("--profile-variable-blur must be in [0, 1]")
    if not 0.0 <= args.profile_blur_alpha <= 1.0:
        raise SystemExit("--profile-blur-alpha must be in [0, 1]")
    if args.masked_rows_per_msa_min < 1:
        raise SystemExit("--masked-rows-per-msa-min must be >= 1")
    if args.masked_rows_per_msa_max < args.masked_rows_per_msa_min:
        raise SystemExit("--masked-rows-per-msa-max must be >= --masked-rows-per-msa-min")
    if args.max_msa_context_rows is not None and args.max_msa_context_rows < 1:
        raise SystemExit("--max-msa-context-rows must be >= 1")
    if args.msa_axial_layers < 1:
        raise SystemExit("--msa-axial-layers must be >= 1")
    if args.profile_feature_mode != "full" and (args.profile_variable_dropout > 0.0 or args.profile_variable_blur > 0.0):
        print(
            "warning: profile AA-frequency dropout/blur is inactive when "
            f"profile_feature_mode={args.profile_feature_mode}",
            flush=True,
        )
    timestep_range = parse_timestep_range(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = load_label_summary(Path(args.label_summary))
    train_rows = read_embedding_manifest(Path(args.embedding_manifest), split="train", path_rewrites=path_rewrites)
    val_rows = read_embedding_manifest(Path(args.embedding_manifest), split="val", path_rewrites=path_rewrites)
    if not train_rows:
        raise SystemExit("No train rows found in embedding manifest")
    if not val_rows:
        raise SystemExit("No val rows found in embedding manifest")

    rng = random.Random(args.seed)
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    if args.max_train_msas is not None:
        train_rows = train_rows[: args.max_train_msas]
    if args.max_val_msas is not None:
        val_rows = val_rows[: args.max_val_msas]

    train_examples = build_examples(train_rows, args.max_rows_per_msa)
    val_examples = build_examples(val_rows, args.max_rows_per_msa)
    rng.shuffle(train_examples)
    rng.shuffle(val_examples)
    if args.max_examples is not None:
        train_examples = train_examples[: args.max_examples]
        val_examples = val_examples[: max(1, min(len(val_examples), args.max_examples // 10))]
    if args.max_train_examples is not None:
        train_examples = train_examples[: args.max_train_examples]
    if args.max_val_examples is not None:
        val_examples = val_examples[: args.max_val_examples]
    if not train_examples or not val_examples:
        raise SystemExit("No train/val examples selected")

    numeric_means, numeric_stds, numeric_counts = numeric_normalization(train_examples, labels)
    dataset_kwargs = {
        "labels": labels,
        "numeric_means": numeric_means,
        "numeric_stds": numeric_stds,
        "category_buckets": args.category_buckets,
        "cache_size": args.cache_size,
        "consensus_loss_mode": args.consensus_loss_mode,
        "consensus_match_weight": args.consensus_match_weight,
        "nonconsensus_weight": args.nonconsensus_weight,
        "unobserved_nonconsensus_weight": args.unobserved_nonconsensus_weight,
        "max_sequence_loss_weight": args.max_sequence_loss_weight,
        "variable_column_min_entropy": args.variable_column_min_entropy,
        "variable_column_max_consensus": args.variable_column_max_consensus,
        "require_msa_embeddings": uses_msa_embedding_memory(args.memory_mode),
        "require_target_continuous_embeddings": args.continuous_target_mode == "target_row_embedding",
        "msa_embedding_dtype": args.msa_embedding_dtype,
        "max_msa_context_rows": args.max_msa_context_rows,
        "gap_inclusive_msa_mask": uses_gap_inclusive_msa_mask(args.memory_mode),
    }
    grouped_target_mode = args.masked_rows_per_msa_max > 1
    if grouped_target_mode:
        train_groups = build_msa_groups(train_examples)
        val_groups = build_msa_groups(val_examples)
        train_dataset = CachedMSAMaskedRowsDataset(
            train_groups,
            masked_rows_per_msa_min=args.masked_rows_per_msa_min,
            masked_rows_per_msa_max=args.masked_rows_per_msa_max,
            **dataset_kwargs,
        )
        val_dataset = CachedMSAMaskedRowsDataset(
            val_groups,
            masked_rows_per_msa_min=args.masked_rows_per_msa_min,
            masked_rows_per_msa_max=args.masked_rows_per_msa_max,
            **dataset_kwargs,
        )
    else:
        train_groups = []
        val_groups = []
        train_dataset = CachedMSARowDataset(train_examples, **dataset_kwargs)
        val_dataset = CachedMSARowDataset(val_examples, **dataset_kwargs)
    shared_msa_grid_mode = grouped_target_mode and uses_axial_msa_memory(args.memory_mode)
    collator = RowReconstructionCollator(
        args.max_sequence_length,
        args.tail_stop_weight,
        args.profile_feature_mode,
        share_msa_embeddings=shared_msa_grid_mode,
    )
    loader_generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=args.workers,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collator,
    )

    first_sample = train_dataset[0]
    first_item = first_sample[0] if isinstance(first_sample, list) else first_sample
    row_embedding_dim = int(first_item["row_embeddings"].shape[-1])
    msa_embedding_dim = int(first_item["msa_embeddings"].shape[-1])
    target_continuous_dim = int(first_item["target_continuous_embeddings"].shape[-1])
    model = MeanStartCCDDModel(
        row_embedding_dim=row_embedding_dim,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        max_sequence_length=args.max_sequence_length,
        diffusion_timesteps=args.diffusion_timesteps,
        category_buckets=args.category_buckets,
        memory_mode=args.memory_mode,
        profile_feature_mode=args.profile_feature_mode,
        msa_embedding_dim=msa_embedding_dim,
        continuous_target_mode=args.continuous_target_mode,
        target_continuous_dim=target_continuous_dim,
        msa_axial_layers=args.msa_axial_layers,
    )
    device = choose_device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16" and device.type == "cuda"))
    start_step = 0
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.exists():
            raise SystemExit(f"--resume-checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location="cpu")
        checkpoint_state = checkpoint["model_state_dict"]
        skipped_incompatible: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
        if args.allow_partial_resume:
            model_state = model.state_dict()
            filtered_state = {}
            for key, value in checkpoint_state.items():
                if key in model_state and tuple(value.shape) != tuple(model_state[key].shape):
                    skipped_incompatible.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                    continue
                filtered_state[key] = value
            checkpoint_state = filtered_state
        load_result = model.load_state_dict(
            checkpoint_state,
            strict=not args.allow_partial_resume,
        )
        if args.allow_partial_resume:
            missing = list(load_result.missing_keys)
            unexpected = list(load_result.unexpected_keys)
            print(
                f"partial_resume missing_keys={len(missing)} unexpected_keys={len(unexpected)} "
                f"skipped_incompatible={len(skipped_incompatible)} "
                f"missing_sample={missing[:12]} unexpected_sample={unexpected[:12]} "
                f"skipped_sample={skipped_incompatible[:6]}",
                flush=True,
            )
        if skipped_incompatible and not args.reset_optimizer:
            print(
                "warning: skipped optimizer resume because partial model load skipped incompatible tensors",
                flush=True,
            )
        elif not args.reset_optimizer and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                move_optimizer_state(optimizer, device)
            except ValueError as exc:
                if not args.allow_partial_resume:
                    raise
                print(
                    f"warning: skipped optimizer resume after partial model load: {exc}",
                    flush=True,
                )
        start_step = int(checkpoint.get("step", 0))
        print(
            f"Resumed mean-start CCDD checkpoint={resume_path} step={start_step} "
            f"reset_optimizer={args.reset_optimizer}",
            flush=True,
        )

    metrics_path = out_dir / "metrics.tsv"
    checkpoint_path = out_dir / "mean_start_ccdd.latest.pt"
    final_path = out_dir / "mean_start_ccdd.final.pt"
    best_path = out_dir / "mean_start_ccdd.best.pt"
    best_metadata_path = out_dir / "mean_start_ccdd.best.json"
    metrics_fields = [
        "step",
        "split",
        "examples",
        "loss",
        "continuous_loss",
        "token_loss",
        "token_accuracy",
        "residue_accuracy",
        "timestep_mean",
        "memory_drop_fraction",
        *CONSENSUS_METRIC_FIELDS,
        *CONDITION_METRIC_FIELDS,
        "elapsed_seconds",
    ]
    if not (args.resume_checkpoint and metrics_path.exists() and metrics_path.stat().st_size > 0):
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=metrics_fields)
            writer.writeheader()

    print(
        f"Mean-start CCDD cached-MSA training train_examples={len(train_examples):,} "
        f"val_examples={len(val_examples):,} memory_mode={args.memory_mode} "
        f"profile_feature_mode={args.profile_feature_mode} profile_dim={model.profile_dim} "
        f"msa_embedding_dim={model.msa_embedding_dim} "
        f"continuous_target_mode={args.continuous_target_mode} "
        f"target_continuous_dim={model.target_continuous_dim} "
        f"decoder_start_mode={args.decoder_start_mode} max_sequence_length={args.max_sequence_length} "
        f"device={device} timestep_range={timestep_range[0]}:{timestep_range[1]} "
        f"condition_mask_prob={args.condition_mask_prob} amp={args.amp} "
        f"msa_embedding_dtype={args.msa_embedding_dtype} "
        f"msa_axial_layers={args.msa_axial_layers} "
        f"max_msa_context_rows={args.max_msa_context_rows} metrics={metrics_path}",
        flush=True,
    )
    if path_rewrites:
        print(
            "path_rewrites=" + ",".join(f"{old}=>{new}" for old, new in path_rewrites),
            flush=True,
        )
    if grouped_target_mode:
        print(
            f"grouped_target_mode=1 train_msa_groups={len(train_groups):,} val_msa_groups={len(val_groups):,} "
            f"masked_rows_per_msa={args.masked_rows_per_msa_min}:{args.masked_rows_per_msa_max} "
            f"shared_msa_grid={int(shared_msa_grid_mode)} "
            "all selected rows are removed from profile, ESM-MSA token memory, and row-memory together",
            flush=True,
        )
    else:
        print("grouped_target_mode=0 masked_rows_per_msa=1:1", flush=True)
    print(
        f"loss_weights=continuous:{args.continuous_loss_weight} token:{args.token_loss_weight} "
        f"numeric_condition:{args.numeric_condition_loss_weight} "
        f"category_condition:{args.category_condition_loss_weight} "
        f"condition_presence:{args.condition_presence_loss_weight}",
        flush=True,
    )
    print(
        f"consensus_objective mode={args.consensus_loss_mode} match_weight={args.consensus_match_weight} "
        f"nonconsensus_weight={args.nonconsensus_weight} "
        f"unobserved_nonconsensus_weight={args.unobserved_nonconsensus_weight} "
        f"max_sequence_loss_weight={args.max_sequence_loss_weight} "
        f"variable_entropy_min={args.variable_column_min_entropy} "
        f"variable_consensus_max={args.variable_column_max_consensus}",
        flush=True,
    )
    print(
        f"profile_regularization variable_dropout={args.profile_variable_dropout} "
        f"variable_blur={args.profile_variable_blur} blur_alpha={args.profile_blur_alpha} "
        f"active={args.profile_feature_mode == 'full'}",
        flush=True,
    )
    print(
        "numeric_condition_coverage="
        + ",".join(f"{field}:{numeric_counts[field]}/{len(train_examples)}" for field in NUMERIC_FIELDS),
        flush=True,
    )
    print(
        "leakage_control=target row(s) are excluded from profile, cached ESM-MSA token memory, and row-memory; "
        "target-row residue embeddings are used only as continuous targets when requested; "
        "cached col_embeddings are not used; profile_msa_axial keeps a gap-inclusive target-masked MSA grid static "
        "and lets only decoder latents cross-attend directly over static MSA column cells and row cells every layer",
        flush=True,
    )

    started = time.monotonic()
    step = start_step
    best_val_loss = float("inf")
    rolling = {
        "examples": 0.0,
        "loss": 0.0,
        "continuous_loss": 0.0,
        "token_loss": 0.0,
        "token_accuracy": 0.0,
        "residue_accuracy": 0.0,
        "timestep_mean": 0.0,
        "memory_drop_fraction": 0.0,
    }
    for field in CONSENSUS_METRIC_FIELDS:
        rolling[field] = 0.0
    for field in CONDITION_METRIC_FIELDS:
        rolling[field] = 0.0
    first_val_batch = next(iter(val_loader))
    model.train()
    while step < args.max_steps:
        for batch in train_loader:
            step += 1
            moved = move_batch(batch, device)
            batch_size = moved["target_tokens"].shape[0]
            timesteps = sampled_timesteps(batch_size, timestep_range, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp):
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
                    decoder_start_mode=args.decoder_start_mode,
                    memory_dropout=args.memory_dropout,
                    condition_mask_prob=args.condition_mask_prob,
                    profile_variable_mask=moved["profile_variable_mask"],
                    profile_variable_dropout=args.profile_variable_dropout,
                    profile_variable_blur=args.profile_variable_blur,
                    profile_blur_alpha=args.profile_blur_alpha,
                    target_group_indices=moved.get("target_msa_group_indices"),
                )
                loss = combined_training_loss(
                    outputs,
                    args.continuous_loss_weight,
                    args.token_loss_weight,
                    args.numeric_condition_loss_weight,
                    args.category_condition_loss_weight,
                    args.condition_presence_loss_weight,
                )
            if grad_scaler.is_enabled():
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            metrics = {
                "examples": float(batch_size),
                "loss": float(loss.item()),
                "continuous_loss": float(outputs["weighted_continuous_loss"].item()),
                "token_loss": float(outputs["token_loss"].item()),
                "token_accuracy": float(outputs["token_accuracy"].item()),
                "residue_accuracy": weighted_residue_accuracy(
                    outputs["logits"], moved["target_tokens"], moved["loss_weights"]
                ),
                "timestep_mean": float(timesteps.to(dtype=torch.float32).mean().item()),
                "memory_drop_fraction": float(outputs["memory_drop_fraction"].item()),
            }
            residue_mask = moved["loss_weights"] > 0.5
            metrics["sequence_loss_weight_mean"] = float(outputs["sequence_loss_weight_mean"].item())
            metrics["consensus_residue_accuracy"] = masked_residue_accuracy_value(
                outputs["logits"],
                moved["target_tokens"],
                moved["consensus_match_mask"] & moved["consensus_observed_mask"] & residue_mask,
            )
            metrics["nonconsensus_residue_accuracy"] = masked_residue_accuracy_value(
                outputs["logits"],
                moved["target_tokens"],
                moved["nonconsensus_mask"] & residue_mask,
            )
            metrics["nonconsensus_fraction"] = masked_fraction_value(moved["nonconsensus_mask"], residue_mask)
            metrics["variable_nonconsensus_fraction"] = masked_fraction_value(
                moved["variable_nonconsensus_mask"],
                residue_mask,
            )
            metrics["profile_variable_fraction"] = masked_fraction_value(
                moved["profile_variable_mask"],
                moved["profile_mask"],
            )
            metrics["profile_drop_fraction"] = float(outputs["profile_drop_fraction"].item())
            metrics["profile_blur_fraction"] = float(outputs["profile_blur_fraction"].item())
            for field in CONDITION_METRIC_FIELDS:
                metrics[field] = float(outputs[field].item())
            for key, value in metrics.items():
                rolling[key] += value * batch_size if key != "examples" else value

            elapsed = time.monotonic() - started
            if step % args.log_every_steps == 0:
                denom = max(rolling["examples"], 1.0)
                train_metrics = {key: (value / denom if key != "examples" else value) for key, value in rolling.items()}
                with metrics_path.open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, delimiter="\t", fieldnames=metrics_fields)
                    writer.writerow(
                        {
                            "step": step,
                            "split": "train_window",
                            "examples": int(train_metrics["examples"]),
                            "loss": f"{train_metrics['loss']:.8f}",
                            "continuous_loss": f"{train_metrics['continuous_loss']:.8f}",
                            "token_loss": f"{train_metrics['token_loss']:.8f}",
                            "token_accuracy": f"{train_metrics['token_accuracy']:.8f}",
                            "residue_accuracy": f"{train_metrics['residue_accuracy']:.8f}",
                            "timestep_mean": f"{train_metrics['timestep_mean']:.4f}",
                            "memory_drop_fraction": f"{train_metrics['memory_drop_fraction']:.8f}",
                            **{field: f"{train_metrics[field]:.8f}" for field in CONSENSUS_METRIC_FIELDS},
                            **{field: f"{train_metrics[field]:.8f}" for field in CONDITION_METRIC_FIELDS},
                            "elapsed_seconds": f"{elapsed:.3f}",
                        }
                    )
                print(
                    f"step={step} train_loss={train_metrics['loss']:.5f} "
                    f"cont={train_metrics['continuous_loss']:.5f} token={train_metrics['token_loss']:.5f} "
                    f"token_acc={train_metrics['token_accuracy']:.4f} residue_acc={train_metrics['residue_accuracy']:.4f} "
                    f"noncons_acc={train_metrics['nonconsensus_residue_accuracy']:.4f} "
                    f"seq_w={train_metrics['sequence_loss_weight_mean']:.3f} "
                    f"num_cond={train_metrics['numeric_value_loss']:.4f} cat_cond={train_metrics['category_value_loss']:.4f} "
                    f"examples={int(train_metrics['examples'])} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                rolling = {key: 0.0 for key in rolling}

            if step % args.eval_every_steps == 0:
                val_metrics = evaluate(
                    model,
                    val_loader,
                    device,
                    timestep_range,
                    args.decoder_start_mode,
                    args.continuous_loss_weight,
                    args.token_loss_weight,
                    args.numeric_condition_loss_weight,
                    args.category_condition_loss_weight,
                    args.condition_presence_loss_weight,
                    args.condition_mask_prob,
                    args.memory_dropout,
                    args.profile_variable_dropout,
                    args.profile_variable_blur,
                    args.profile_blur_alpha,
                    args.amp,
                    args.val_batches,
                )
                with metrics_path.open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, delimiter="\t", fieldnames=metrics_fields)
                    writer.writerow(
                        {
                            "step": step,
                            "split": "val",
                            "examples": int(val_metrics["examples"]),
                            "loss": f"{val_metrics['loss']:.8f}",
                            "continuous_loss": f"{val_metrics['continuous_loss']:.8f}",
                            "token_loss": f"{val_metrics['token_loss']:.8f}",
                            "token_accuracy": f"{val_metrics['token_accuracy']:.8f}",
                            "residue_accuracy": f"{val_metrics['residue_accuracy']:.8f}",
                            "timestep_mean": f"{val_metrics['timestep_mean']:.4f}",
                            "memory_drop_fraction": f"{val_metrics['memory_drop_fraction']:.8f}",
                            **{field: f"{val_metrics[field]:.8f}" for field in CONSENSUS_METRIC_FIELDS},
                            **{field: f"{val_metrics[field]:.8f}" for field in CONDITION_METRIC_FIELDS},
                            "elapsed_seconds": f"{elapsed:.3f}",
                        }
                    )
                print(
                    f"step={step} val_loss={val_metrics['loss']:.5f} "
                    f"val_cont={val_metrics['continuous_loss']:.5f} val_token={val_metrics['token_loss']:.5f} "
                    f"val_token_acc={val_metrics['token_accuracy']:.4f} val_residue_acc={val_metrics['residue_accuracy']:.4f} "
                    f"val_noncons_acc={val_metrics['nonconsensus_residue_accuracy']:.4f} "
                    f"val_seq_w={val_metrics['sequence_loss_weight_mean']:.3f} "
                    f"val_num_cond={val_metrics['numeric_value_loss']:.4f} "
                    f"val_cat_cond={val_metrics['category_value_loss']:.4f}",
                    flush=True,
                )
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    save_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        step,
                        args,
                        numeric_means,
                        numeric_stds,
                        numeric_counts,
                    )
                    best_metadata_path.write_text(
                        json.dumps(
                            {
                                "step": step,
                                "metric": "val_loss",
                                "val_loss": best_val_loss,
                                "val_metrics": val_metrics,
                                "checkpoint": str(best_path),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    print(f"saved_best_checkpoint={best_path} step={step} val_loss={best_val_loss:.5f}", flush=True)

            if args.decode_every_steps > 0 and step % args.decode_every_steps == 0:
                decode_path = out_dir / "decodes" / f"decode_step_{step:07d}.fasta"
                decode_panel(model, first_val_batch, device, args.amp, args.decode_examples, decode_path)
                print(f"saved_decode={decode_path}", flush=True)
                model.train()

            if step % args.checkpoint_every_steps == 0:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    step,
                    args,
                    numeric_means,
                    numeric_stds,
                    numeric_counts,
                )
                print(f"saved_checkpoint={checkpoint_path} step={step}", flush=True)

            if step >= args.max_steps:
                break

    save_checkpoint(final_path, model, optimizer, step, args, numeric_means, numeric_stds, numeric_counts)
    save_checkpoint(checkpoint_path, model, optimizer, step, args, numeric_means, numeric_stds, numeric_counts)
    print(f"Finished mean-start CCDD cached-MSA training step={step} final_checkpoint={final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
