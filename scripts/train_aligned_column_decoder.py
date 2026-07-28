#!/usr/bin/env python3
"""Train an aligned-column decoder from cached ESM-MSA column embeddings."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset


DEFAULT_TRAINING_ROOT = Path("/mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim")
DEFAULT_EMBEDDING_MANIFEST = DEFAULT_TRAINING_ROOT / "esm_msa_embeddings_col" / "embedding_manifest.tsv"
DEFAULT_LABEL_SUMMARY = DEFAULT_TRAINING_ROOT / "sequence_label_summary.tsv.gz"
DEFAULT_OUT_DIR = DEFAULT_TRAINING_ROOT / "aligned_column_training"
TOKENS = "-ACDEFGHIKLMNPQRSTVWY"
TOKEN_TO_ID = {token: idx for idx, token in enumerate(TOKENS)}
AA_TOKENS = TOKENS[1:]
NUMERIC_FIELDS = ("kcat_1_per_s", "km_mM", "kcat_over_km_1_per_mM_s", "topt_C", "tm_C")
CATEGORICAL_FIELDS = ("domain", "reaction_id", "ec_numbers", "compound_id")
LOG_NUMERIC_FIELDS = {"kcat_1_per_s", "km_mM", "kcat_over_km_1_per_mM_s"}


@dataclass(frozen=True)
class ColumnExample:
    cluster_index: str
    split: str
    npz_path: Path
    metadata_path: Path
    row_index: int
    kegg_entry: str


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
    if math.isfinite(value):
        return value
    return None


def transform_numeric(field: str, value: float) -> float | None:
    if field in LOG_NUMERIC_FIELDS:
        if value <= 0:
            return None
        return math.log10(value)
    return value


def read_embedding_manifest(path: Path, split: str | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
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
            npz_path = Path(row.get("npz_path", ""))
            metadata_path = Path(row.get("metadata_path", ""))
            if not npz_path.exists() or not metadata_path.exists():
                continue
            rows.append(row)
    return rows


def examples_from_manifest_rows(rows: list[dict[str, str]], max_rows_per_msa: int | None) -> list[ColumnExample]:
    examples: list[ColumnExample] = []
    for row in rows:
        metadata_path = Path(row["metadata_path"])
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipping metadata {metadata_path}: {exc}", file=sys.stderr)
            continue
        headers = [str(header) for header in metadata.get("headers", [])]
        sequences = [str(sequence) for sequence in metadata.get("cleaned_sequences", [])]
        if not headers or len(headers) != len(sequences):
            print(f"warning: skipping {metadata_path}: missing aligned headers/sequences", file=sys.stderr)
            continue
        row_count = len(headers)
        if max_rows_per_msa is not None:
            row_count = min(row_count, max_rows_per_msa)
        for row_index in range(row_count):
            sequence = sequences[row_index].upper()
            if not sequence or any(char not in TOKEN_TO_ID for char in sequence):
                continue
            examples.append(
                ColumnExample(
                    cluster_index=row["cluster_index"],
                    split=row.get("split", "train"),
                    npz_path=Path(row["npz_path"]),
                    metadata_path=metadata_path,
                    row_index=row_index,
                    kegg_entry=headers[row_index].split()[0],
                )
            )
    return examples


def load_label_summary(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        print(f"warning: label summary not found yet: {path}; training will use missing condition tokens", file=sys.stderr)
        return {}
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


def numeric_arrays(
    examples: list[ColumnExample],
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


class CachedMSAStore:
    def __init__(self, cache_size: int = 64) -> None:
        self.cache_size = max(cache_size, 0)
        self.cache: dict[Path, dict[str, Any]] = {}
        self.order: list[Path] = []

    def load(self, npz_path: Path, metadata_path: Path) -> dict[str, Any]:
        if npz_path in self.cache:
            return self.cache[npz_path]
        arrays = np.load(npz_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        item = {
            "col_embeddings": arrays["col_embeddings"].astype(np.float32),
            "headers": [str(header).split()[0] for header in metadata["headers"]],
            "sequences": [str(sequence).upper() for sequence in metadata["cleaned_sequences"]],
        }
        if self.cache_size:
            self.cache[npz_path] = item
            self.order.append(npz_path)
            if len(self.order) > self.cache_size:
                old = self.order.pop(0)
                self.cache.pop(old, None)
        return item


class AlignedColumnDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        examples: list[ColumnExample],
        labels: dict[str, dict[str, str]],
        numeric_means: dict[str, float],
        numeric_stds: dict[str, float],
        category_buckets: int,
        cache_size: int,
    ) -> None:
        self.examples = examples
        self.labels = labels
        self.numeric_means = numeric_means
        self.numeric_stds = numeric_stds
        self.category_buckets = category_buckets
        self.store = CachedMSAStore(cache_size=cache_size)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        item = self.store.load(example.npz_path, example.metadata_path)
        sequences = item["sequences"]
        target = sequences[example.row_index]
        col_embeddings = item["col_embeddings"]
        length = min(len(target), col_embeddings.shape[0])
        target = target[:length]
        col_embeddings = col_embeddings[:length]

        profile = profile_excluding_row(sequences, example.row_index, length)
        token_ids = np.array([TOKEN_TO_ID[char] for char in target], dtype=np.int64)
        row = self.labels.get(example.kegg_entry, {})
        numeric_values = np.zeros((len(NUMERIC_FIELDS),), dtype=np.float32)
        numeric_mask = np.zeros((len(NUMERIC_FIELDS),), dtype=np.bool_)
        for idx, field in enumerate(NUMERIC_FIELDS):
            value = parse_float(row.get(f"{field}_mean", ""))
            if value is None:
                continue
            transformed = transform_numeric(field, value)
            if transformed is None:
                continue
            numeric_values[idx] = (transformed - self.numeric_means[field]) / self.numeric_stds[field]
            numeric_mask[idx] = True
        category_ids = np.full((len(CATEGORICAL_FIELDS),), -1, dtype=np.int64)
        category_mask = np.zeros((len(CATEGORICAL_FIELDS),), dtype=np.bool_)
        for idx, field in enumerate(CATEGORICAL_FIELDS):
            values = split_values(row.get(f"{field}_values", ""))
            if values:
                joined = "|".join(values)
                category_ids[idx] = stable_hash(f"{field}:{joined}", self.category_buckets)
                category_mask[idx] = True
        return {
            "col_embeddings": col_embeddings,
            "profile": profile,
            "target_tokens": token_ids,
            "numeric_values": numeric_values,
            "numeric_mask": numeric_mask,
            "category_ids": category_ids,
            "category_mask": category_mask,
            "cluster_index": example.cluster_index,
            "kegg_entry": example.kegg_entry,
        }


def profile_excluding_row(sequences: list[str], row_index: int, length: int) -> np.ndarray:
    features = np.zeros((length, len(AA_TOKENS) + 2), dtype=np.float32)
    total_other = max(len(sequences) - 1, 1)
    aa_to_col = {aa: idx for idx, aa in enumerate(AA_TOKENS)}
    for idx, sequence in enumerate(sequences):
        if idx == row_index:
            continue
        for col, char in enumerate(sequence[:length]):
            if char == "-":
                features[col, len(AA_TOKENS)] += 1.0
            elif char in aa_to_col:
                features[col, aa_to_col[char]] += 1.0
    counts = features[:, : len(AA_TOKENS)].sum(axis=1) + features[:, len(AA_TOKENS)]
    nonzero = counts > 0
    features[nonzero, : len(AA_TOKENS) + 1] /= counts[nonzero, None]
    features[:, -1] = counts / float(total_other)
    return features


class ColumnCollator:
    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        batch_size = len(batch)
        max_len = max(item["target_tokens"].shape[0] for item in batch)
        hidden = batch[0]["col_embeddings"].shape[-1]
        profile_dim = batch[0]["profile"].shape[-1]
        col_embeddings = np.zeros((batch_size, max_len, hidden), dtype=np.float32)
        profiles = np.zeros((batch_size, max_len, profile_dim), dtype=np.float32)
        targets = np.zeros((batch_size, max_len), dtype=np.int64)
        mask = np.zeros((batch_size, max_len), dtype=np.bool_)
        numeric_values = np.zeros((batch_size, len(NUMERIC_FIELDS)), dtype=np.float32)
        numeric_mask = np.zeros((batch_size, len(NUMERIC_FIELDS)), dtype=np.bool_)
        category_ids = np.full((batch_size, len(CATEGORICAL_FIELDS)), -1, dtype=np.int64)
        category_mask = np.zeros((batch_size, len(CATEGORICAL_FIELDS)), dtype=np.bool_)
        cluster_indices: list[str] = []
        kegg_entries: list[str] = []
        for idx, item in enumerate(batch):
            length = item["target_tokens"].shape[0]
            col_embeddings[idx, :length] = item["col_embeddings"]
            profiles[idx, :length] = item["profile"]
            targets[idx, :length] = item["target_tokens"]
            mask[idx, :length] = True
            numeric_values[idx] = item["numeric_values"]
            numeric_mask[idx] = item["numeric_mask"]
            category_ids[idx] = item["category_ids"]
            category_mask[idx] = item["category_mask"]
            cluster_indices.append(item["cluster_index"])
            kegg_entries.append(item["kegg_entry"])
        return {
            "col_embeddings": torch.from_numpy(col_embeddings),
            "profiles": torch.from_numpy(profiles),
            "target_tokens": torch.from_numpy(targets),
            "mask": torch.from_numpy(mask),
            "numeric_values": torch.from_numpy(numeric_values),
            "numeric_mask": torch.from_numpy(numeric_mask),
            "category_ids": torch.from_numpy(category_ids),
            "category_mask": torch.from_numpy(category_mask),
            "cluster_indices": cluster_indices,
            "kegg_entries": kegg_entries,
        }


class AlignedColumnDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        profile_dim: int = len(AA_TOKENS) + 2,
        d_model: int = 192,
        layers: int = 4,
        heads: int = 6,
        dropout: float = 0.1,
        max_cols: int = 1024,
        category_buckets: int = 4096,
    ) -> None:
        super().__init__()
        self.max_cols = max_cols
        self.col_proj = nn.Linear(input_dim, d_model)
        self.profile_proj = nn.Linear(profile_dim, d_model)
        self.pos_embedding = nn.Embedding(max_cols, d_model)
        self.numeric_proj = nn.Sequential(
            nn.Linear(len(NUMERIC_FIELDS) * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(category_buckets, d_model) for _ in CATEGORICAL_FIELDS]
        )
        self.condition_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, len(TOKENS))

    def condition_token(
        self,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        category_ids: torch.Tensor,
        category_mask: torch.Tensor,
    ) -> torch.Tensor:
        numeric_input = torch.cat([numeric_values, numeric_mask.to(dtype=numeric_values.dtype)], dim=-1)
        token = self.numeric_proj(numeric_input)
        for idx, embedding in enumerate(self.category_embeddings):
            ids = category_ids[:, idx].clamp_min(0)
            cat = embedding(ids)
            cat = torch.where(category_mask[:, idx].view(-1, 1), cat, torch.zeros_like(cat))
            token = token + cat
        return self.condition_norm(token)

    def forward(
        self,
        col_embeddings: torch.Tensor,
        profiles: torch.Tensor,
        mask: torch.Tensor,
        numeric_values: torch.Tensor,
        numeric_mask: torch.Tensor,
        category_ids: torch.Tensor,
        category_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length, _ = col_embeddings.shape
        if length > self.max_cols:
            raise ValueError(f"batch has {length} columns but model max_cols={self.max_cols}")
        positions = torch.arange(length, device=col_embeddings.device).unsqueeze(0)
        x = self.col_proj(col_embeddings) + self.profile_proj(profiles) + self.pos_embedding(positions)
        condition = self.condition_token(numeric_values, numeric_mask, category_ids, category_mask).unsqueeze(1)
        x = torch.cat([condition, x], dim=1)
        full_mask = torch.cat([torch.ones((batch_size, 1), dtype=torch.bool, device=mask.device), mask], dim=1)
        encoded = self.encoder(x, src_key_padding_mask=~full_mask)
        return self.head(self.out_norm(encoded[:, 1:]))


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def masked_loss_and_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    gap_weight: float,
) -> tuple[torch.Tensor, float, float]:
    weights = mask.to(dtype=logits.dtype)
    if gap_weight != 1.0:
        weights = torch.where(targets == TOKEN_TO_ID["-"], weights * gap_weight, weights)
    flat_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none")
    loss = (flat_loss.reshape_as(targets) * weights).sum() / weights.sum().clamp_min(1.0)
    predicted = torch.argmax(logits, dim=-1)
    correct = (predicted == targets) & mask
    token_acc = correct.to(dtype=torch.float32).sum() / mask.to(dtype=torch.float32).sum().clamp_min(1.0)
    residue_mask = mask & (targets != TOKEN_TO_ID["-"])
    residue_acc = (correct & residue_mask).to(dtype=torch.float32).sum() / residue_mask.to(dtype=torch.float32).sum().clamp_min(1.0)
    return loss, float(token_acc.item()), float(residue_acc.item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-manifest", default=str(DEFAULT_EMBEDDING_MANIFEST))
    parser.add_argument("--label-summary", default=str(DEFAULT_LABEL_SUMMARY))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--max-rows-per-msa", type=int, default=None)
    parser.add_argument("--cache-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--category-buckets", type=int, default=4096)
    parser.add_argument("--max-cols", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--log-every-steps", type=int, default=25)
    parser.add_argument("--eval-every-steps", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=32)
    parser.add_argument("--checkpoint-every-steps", type=int, default=250)
    parser.add_argument("--gap-loss-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def evaluate(
    model: AlignedColumnDecoder,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    max_batches: int,
    gap_loss_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_token_acc = 0.0
    total_residue_acc = 0.0
    total_examples = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            if batch_index > max_batches:
                break
            moved = move_batch(batch, device)
            logits = model(**{key: moved[key] for key in ("col_embeddings", "profiles", "mask", "numeric_values", "numeric_mask", "category_ids", "category_mask")})
            loss, token_acc, residue_acc = masked_loss_and_accuracy(
                logits, moved["target_tokens"], moved["mask"], gap_loss_weight
            )
            batch_size = moved["target_tokens"].shape[0]
            total_loss += float(loss.item()) * batch_size
            total_token_acc += token_acc * batch_size
            total_residue_acc += residue_acc * batch_size
            total_examples += batch_size
    denom = max(total_examples, 1)
    model.train()
    return {
        "examples": float(total_examples),
        "loss": total_loss / denom,
        "token_accuracy": total_token_acc / denom,
        "residue_accuracy": total_residue_acc / denom,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def save_checkpoint(
    path: Path,
    model: nn.Module,
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
            "tokens": TOKENS,
            "numeric_fields": NUMERIC_FIELDS,
            "categorical_fields": CATEGORICAL_FIELDS,
            "numeric_means": numeric_means,
            "numeric_stds": numeric_stds,
            "numeric_counts": numeric_counts,
        },
        path,
    )


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embedding_manifest = Path(args.embedding_manifest)
    if not embedding_manifest.exists():
        raise SystemExit(f"Embedding manifest not found: {embedding_manifest}")
    labels = load_label_summary(Path(args.label_summary))
    manifest_rows = read_embedding_manifest(embedding_manifest)
    if not manifest_rows:
        raise SystemExit("No embedded MSA rows found in manifest")
    examples = examples_from_manifest_rows(manifest_rows, args.max_rows_per_msa)
    if args.max_examples is not None:
        rng = random.Random(args.seed)
        rng.shuffle(examples)
        examples = examples[: args.max_examples]
    train_examples = [example for example in examples if example.split == "train"]
    val_examples = [example for example in examples if example.split == "val"]
    if not val_examples:
        val_examples = [example for example in examples if example.split == "test"][: max(1, len(examples) // 20)]
    if not train_examples:
        raise SystemExit("No train examples selected")

    numeric_means, numeric_stds, numeric_counts = numeric_arrays(train_examples, labels)
    train_dataset = AlignedColumnDataset(
        train_examples,
        labels,
        numeric_means,
        numeric_stds,
        args.category_buckets,
        args.cache_size,
    )
    val_dataset = AlignedColumnDataset(
        val_examples,
        labels,
        numeric_means,
        numeric_stds,
        args.category_buckets,
        args.cache_size,
    ) if val_examples else None
    collator = ColumnCollator()
    generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=args.workers,
        collate_fn=collator,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collator,
        )
        if val_dataset is not None
        else None
    )
    device = choose_device(args.device)
    model = AlignedColumnDecoder(
        input_dim=768,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        max_cols=args.max_cols,
        category_buckets=args.category_buckets,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metrics_path = out_dir / "aligned_column_metrics.tsv"
    checkpoint_path = out_dir / "aligned_column_decoder.latest.pt"
    final_path = out_dir / "aligned_column_decoder.final.pt"
    fields = [
        "step",
        "split",
        "examples",
        "loss",
        "token_accuracy",
        "residue_accuracy",
        "elapsed_seconds",
    ]
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()

    print(
        f"Aligned-column training examples={len(examples):,} train={len(train_examples):,} "
        f"val={len(val_examples):,} embedded_msas={len(manifest_rows):,} "
        f"labels={len(labels):,} device={device} metrics={metrics_path}",
        flush=True,
    )
    print(
        "numeric_condition_coverage="
        + ",".join(f"{field}:{numeric_counts[field]}/{len(train_examples)}" for field in NUMERIC_FIELDS),
        flush=True,
    )
    started = time.monotonic()
    step = 0
    rolling_loss = 0.0
    rolling_token = 0.0
    rolling_residue = 0.0
    rolling_examples = 0
    model.train()
    while step < args.max_steps:
        for batch in train_loader:
            step += 1
            moved = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**{key: moved[key] for key in ("col_embeddings", "profiles", "mask", "numeric_values", "numeric_mask", "category_ids", "category_mask")})
            loss, token_acc, residue_acc = masked_loss_and_accuracy(
                logits,
                moved["target_tokens"],
                moved["mask"],
                args.gap_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            batch_size = moved["target_tokens"].shape[0]
            rolling_loss += float(loss.item()) * batch_size
            rolling_token += token_acc * batch_size
            rolling_residue += residue_acc * batch_size
            rolling_examples += batch_size
            elapsed = time.monotonic() - started

            if step % args.log_every_steps == 0:
                denom = max(rolling_examples, 1)
                train_metrics = {
                    "examples": float(rolling_examples),
                    "loss": rolling_loss / denom,
                    "token_accuracy": rolling_token / denom,
                    "residue_accuracy": rolling_residue / denom,
                }
                with metrics_path.open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
                    writer.writerow(
                        {
                            "step": step,
                            "split": "train_window",
                            "examples": int(train_metrics["examples"]),
                            "loss": f"{train_metrics['loss']:.8f}",
                            "token_accuracy": f"{train_metrics['token_accuracy']:.8f}",
                            "residue_accuracy": f"{train_metrics['residue_accuracy']:.8f}",
                            "elapsed_seconds": f"{elapsed:.3f}",
                        }
                    )
                print(
                    f"step={step} train_window_loss={train_metrics['loss']:.5f} "
                    f"token_acc={train_metrics['token_accuracy']:.4f} "
                    f"residue_acc={train_metrics['residue_accuracy']:.4f} "
                    f"examples={int(train_metrics['examples'])} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                rolling_loss = rolling_token = rolling_residue = 0.0
                rolling_examples = 0

            if val_loader is not None and step % args.eval_every_steps == 0:
                val_metrics = evaluate(model, val_loader, device, args.val_batches, args.gap_loss_weight)
                with metrics_path.open("a", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
                    writer.writerow(
                        {
                            "step": step,
                            "split": "val",
                            "examples": int(val_metrics["examples"]),
                            "loss": f"{val_metrics['loss']:.8f}",
                            "token_accuracy": f"{val_metrics['token_accuracy']:.8f}",
                            "residue_accuracy": f"{val_metrics['residue_accuracy']:.8f}",
                            "elapsed_seconds": f"{elapsed:.3f}",
                        }
                    )
                print(
                    f"step={step} val_loss={val_metrics['loss']:.5f} "
                    f"val_token_acc={val_metrics['token_accuracy']:.4f} "
                    f"val_residue_acc={val_metrics['residue_accuracy']:.4f}",
                    flush=True,
                )

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
    print(f"Finished aligned-column training step={step} final_checkpoint={final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
