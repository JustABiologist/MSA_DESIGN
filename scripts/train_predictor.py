#!/usr/bin/env python3
"""Smoke-train a predictor on frozen MSA Transformer embeddings plus enzyme metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from msa_design_model import DEFAULT_NUMERIC_FIELDS, EnzymeMSAPredictor  # noqa: E402


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


@dataclass
class Example:
    embedding_path: Path
    metadata_path: Path
    metadata_row: dict[str, str]
    target: float
    raw_target: float
    condition_values: np.ndarray
    condition_mask: np.ndarray


def split_csv_list(text: str) -> Iterable[str]:
    for part in text.replace(",", ";").split(";"):
        part = part.strip()
        if part:
            yield part


def parse_finite_values(text: str) -> list[float]:
    values: list[float] = []
    for part in split_csv_list(text):
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


def metadata_value(row: dict[str, str], field: str, method: str) -> float | None:
    for column in FIELD_VALUE_COLUMNS[field]:
        if column in row:
            value = aggregate(parse_finite_values(row.get(column, "")), method)
            if value is not None:
                return value
    return None


def transform_target(value: float, mode: str, field: str) -> float:
    if mode == "identity" or (mode == "auto" and field not in LOG_DEFAULT_FIELDS):
        return value
    if mode == "log10" or mode == "auto":
        if value <= 0:
            raise ValueError(f"cannot log-transform non-positive target value {value}")
        return math.log10(value)
    raise ValueError(f"unknown target transform: {mode}")


def read_metadata_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def find_examples(
    embeddings_dir: Path,
    metadata_dir: Path,
    embedding_glob: str,
    target_field: str,
    condition_fields: tuple[str, ...],
    aggregation: str,
    target_transform: str,
    include_target_as_condition: bool,
    require_status_ok: bool,
) -> list[Example]:
    examples: list[Example] = []
    for embedding_path in sorted(embeddings_dir.glob(embedding_glob)):
        stem = embedding_path.stem
        metadata_path = metadata_dir / f"{stem}.metadata.tsv"
        if not metadata_path.exists():
            print(f"warning: skipping {embedding_path}, no metadata TSV at {metadata_path}", file=sys.stderr)
            continue
        rows = read_metadata_rows(metadata_path)
        for row in rows:
            if require_status_ok and row.get("status") not in {"ok", "dry_run", ""}:
                continue
            raw_target = metadata_value(row, target_field, aggregation)
            if raw_target is None:
                continue
            try:
                target = transform_target(raw_target, target_transform, target_field)
            except ValueError as exc:
                print(f"warning: skipping row with invalid target in {metadata_path}: {exc}", file=sys.stderr)
                continue
            condition_values: list[float] = []
            condition_mask: list[bool] = []
            for field in condition_fields:
                if field == target_field and not include_target_as_condition:
                    condition_values.append(0.0)
                    condition_mask.append(False)
                    continue
                value = metadata_value(row, field, aggregation)
                if value is None:
                    condition_values.append(0.0)
                    condition_mask.append(False)
                else:
                    condition_values.append(float(value))
                    condition_mask.append(True)
            examples.append(
                Example(
                    embedding_path=embedding_path,
                    metadata_path=metadata_path,
                    metadata_row=row,
                    target=float(target),
                    raw_target=float(raw_target),
                    condition_values=np.array(condition_values, dtype=np.float32),
                    condition_mask=np.array(condition_mask, dtype=np.bool_),
                )
            )
    return examples


class FrozenEmbeddingDataset(Dataset[dict[str, Any]]):
    def __init__(self, examples: list[Example], target_mean: float, target_std: float) -> None:
        self.examples = examples
        self.target_mean = target_mean
        self.target_std = max(target_std, 1e-6)
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
        target = (example.target - self.target_mean) / self.target_std
        return {
            "token_embeddings": embedding["token_embeddings"],
            "aa_mask": embedding["aa_mask"],
            "condition_values": example.condition_values,
            "condition_mask": example.condition_mask,
            "target": np.array([target], dtype=np.float32),
            "raw_target": np.array([example.raw_target], dtype=np.float32),
            "embedding_path": str(example.embedding_path),
            "gene_id": example.metadata_row.get("gene_id", ""),
        }


def collate_examples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_rows = max(item["token_embeddings"].shape[0] for item in batch)
    max_cols = max(item["token_embeddings"].shape[1] for item in batch)
    hidden_dim = batch[0]["token_embeddings"].shape[2]
    num_fields = batch[0]["condition_values"].shape[0]

    tokens = np.zeros((batch_size, max_rows, max_cols, hidden_dim), dtype=np.float32)
    aa_mask = np.zeros((batch_size, max_rows, max_cols), dtype=np.bool_)
    condition_values = np.zeros((batch_size, num_fields), dtype=np.float32)
    condition_mask = np.zeros((batch_size, num_fields), dtype=np.bool_)
    targets = np.zeros((batch_size, 1), dtype=np.float32)
    raw_targets = np.zeros((batch_size, 1), dtype=np.float32)
    embedding_paths: list[str] = []
    gene_ids: list[str] = []

    for idx, item in enumerate(batch):
        row_count, col_count, _ = item["token_embeddings"].shape
        tokens[idx, :row_count, :col_count] = item["token_embeddings"]
        aa_mask[idx, :row_count, :col_count] = item["aa_mask"]
        condition_values[idx] = item["condition_values"]
        condition_mask[idx] = item["condition_mask"]
        targets[idx] = item["target"]
        raw_targets[idx] = item["raw_target"]
        embedding_paths.append(item["embedding_path"])
        gene_ids.append(item["gene_id"])

    return {
        "token_embeddings": torch.from_numpy(tokens),
        "aa_mask": torch.from_numpy(aa_mask),
        "condition_values": torch.from_numpy(condition_values),
        "condition_mask": torch.from_numpy(condition_mask),
        "target": torch.from_numpy(targets),
        "raw_target": torch.from_numpy(raw_targets),
        "embedding_paths": embedding_paths,
        "gene_ids": gene_ids,
    }


def parse_fields(text: str) -> tuple[str, ...]:
    fields = tuple(field.strip() for field in text.split(",") if field.strip())
    unknown = sorted(set(fields) - set(DEFAULT_NUMERIC_FIELDS))
    if unknown:
        raise SystemExit(f"Unknown condition field(s): {', '.join(unknown)}")
    return fields


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def train(args: argparse.Namespace) -> int:
    condition_fields = parse_fields(args.condition_fields)
    if args.target not in DEFAULT_NUMERIC_FIELDS:
        raise SystemExit(f"Unknown target field {args.target}; choose one of {', '.join(DEFAULT_NUMERIC_FIELDS)}")

    examples = find_examples(
        embeddings_dir=Path(args.embeddings_dir),
        metadata_dir=Path(args.metadata_dir),
        embedding_glob=args.embedding_glob,
        target_field=args.target,
        condition_fields=condition_fields,
        aggregation=args.value_aggregation,
        target_transform=args.target_transform,
        include_target_as_condition=args.include_target_as_condition,
        require_status_ok=not args.allow_non_ok_status,
    )
    if not examples:
        raise SystemExit(
            "No training examples found. Generate embeddings first, e.g. "
            "/home/florian/miniforge3/envs/msa_design/bin/python scripts/embed_msas.py "
            "--msa-glob 'outputs/pilot_msas/ec_*.msa.fasta' --out-dir outputs/embeddings"
        )

    transformed_targets = np.array([example.target for example in examples], dtype=np.float32)
    target_mean = float(transformed_targets.mean())
    target_std = float(transformed_targets.std()) if len(transformed_targets) > 1 else 1.0
    if target_std < 1e-6:
        target_std = 1.0

    dataset = FrozenEmbeddingDataset(examples, target_mean=target_mean, target_std=target_std)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_examples,
    )

    first_embedding = dataset.load_embedding(examples[0].embedding_path)["token_embeddings"]
    model = EnzymeMSAPredictor(
        input_dim=first_embedding.shape[-1],
        d_model=args.d_model,
        condition_fields=condition_fields,
        output_dim=1,
        num_layers=args.layers,
        num_heads=args.heads,
        dropout=args.dropout,
        max_positions=args.max_positions,
    )
    device = choose_device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()

    print(
        f"Training examples={len(dataset)} target={args.target} transform={args.target_transform} "
        f"conditions={','.join(condition_fields)} device={device} target_mean={target_mean:.4g} target_std={target_std:.4g}",
        flush=True,
    )
    if len(dataset) < 8:
        print("warning: tiny dataset; this is a wiring smoke test, not a meaningful validation run", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch in loader:
            token_embeddings = batch["token_embeddings"].to(device)
            aa_mask = batch["aa_mask"].to(device)
            condition_values = batch["condition_values"].to(device)
            condition_mask = batch["condition_mask"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(token_embeddings, aa_mask, condition_values, condition_mask)
            loss = loss_fn(outputs["prediction"], target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += float(loss.item()) * target.shape[0]
            total_count += target.shape[0]
        print(f"epoch={epoch} train_mse={total_loss / max(total_count, 1):.6f}", flush=True)

    checkpoint_path = Path(args.out_checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": int(first_embedding.shape[-1]),
                "d_model": args.d_model,
                "condition_fields": condition_fields,
                "target": args.target,
                "target_transform": args.target_transform,
                "target_mean": target_mean,
                "target_std": target_std,
                "value_aggregation": args.value_aggregation,
                "include_target_as_condition": args.include_target_as_condition,
            },
        },
        checkpoint_path,
    )
    sidecar = checkpoint_path.with_suffix(".metadata.json")
    sidecar.write_text(
        json.dumps(
            {
                "examples": len(dataset),
                "target": args.target,
                "condition_fields": condition_fields,
                "checkpoint": str(checkpoint_path),
                "note": "MSA Transformer embeddings are frozen/precomputed; checkpoint contains only trainable predictor weights.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved checkpoint to {checkpoint_path}", flush=True)
    print(f"Saved metadata to {sidecar}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings-dir", default="outputs/embeddings", help="Directory containing embedding NPZ files.")
    parser.add_argument("--metadata-dir", default="outputs/pilot_msas", help="Directory containing metadata TSV files.")
    parser.add_argument("--embedding-glob", default="ec_*.npz", help="Embedding filename glob relative to --embeddings-dir.")
    parser.add_argument("--target", default="kcat_1_per_s", help="Numeric target field to predict.")
    parser.add_argument(
        "--condition-fields",
        default=",".join(DEFAULT_NUMERIC_FIELDS),
        help="Comma-separated numeric fields that get condition-token heads.",
    )
    parser.add_argument(
        "--include-target-as-condition",
        action="store_true",
        help="Allow target leakage by exposing the target field token. Off by default; target token is marked missing.",
    )
    parser.add_argument("--target-transform", choices=["auto", "identity", "log10"], default="auto")
    parser.add_argument("--value-aggregation", choices=["mean", "first"], default="mean")
    parser.add_argument("--allow-non-ok-status", action="store_true", help="Do not filter metadata rows by status.")
    parser.add_argument("--d-model", type=int, default=128, help="Trainable projection width.")
    parser.add_argument("--layers", type=int, default=2, help="Transformer encoder layers after appending condition tokens.")
    parser.add_argument("--heads", type=int, default=4, help="Attention heads in the trainable predictor.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-positions", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-checkpoint", default="outputs/checkpoints/predictor_smoke.pt")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(train(parse_args()))
