#!/usr/bin/env python3
"""Decode held-out aligned-column examples from a trained checkpoint."""

from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from train_aligned_column_decoder import (
    AlignedColumnDataset,
    AlignedColumnDecoder,
    ColumnCollator,
    DEFAULT_EMBEDDING_MANIFEST,
    DEFAULT_LABEL_SUMMARY,
    TOKEN_TO_ID,
    TOKENS,
    choose_device,
    examples_from_manifest_rows,
    load_label_summary,
    masked_loss_and_accuracy,
    move_batch,
    read_embedding_manifest,
)


DEFAULT_CHECKPOINT = (
    Path("/mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim")
    / "aligned_column_training_full_20260717_220547"
    / "aligned_column_decoder.final.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--embedding-manifest", default=str(DEFAULT_EMBEDDING_MANIFEST))
    parser.add_argument("--label-summary", default=str(DEFAULT_LABEL_SUMMARY))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-examples", type=int, default=2048)
    parser.add_argument("--num-decodes", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cache-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def decode_tokens(token_ids: torch.Tensor, mask: torch.Tensor) -> str:
    ids = token_ids[mask].tolist()
    return "".join(TOKENS[int(idx)] for idx in ids)


def ungap(sequence: str) -> str:
    return sequence.replace("-", "")


def residue_identity(predicted: str, target: str) -> float:
    pairs = [(p, t) for p, t in zip(predicted, target) if t != "-"]
    if not pairs:
        return 0.0
    return sum(p == t for p, t in pairs) / len(pairs)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    return torch.load(path, map_location=device)


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    checkpoint_path = Path(args.checkpoint)
    embedding_manifest = Path(args.embedding_manifest)
    label_summary = Path(args.label_summary)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else checkpoint_path.parent / f"decode_eval_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    device = choose_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config", {})
    labels = load_label_summary(label_summary)
    manifest_rows = read_embedding_manifest(embedding_manifest, split=args.split)
    examples = examples_from_manifest_rows(manifest_rows, max_rows_per_msa=None)
    rng.shuffle(examples)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if not examples:
        raise SystemExit(f"No examples selected for split={args.split}")

    dataset = AlignedColumnDataset(
        examples,
        labels,
        checkpoint["numeric_means"],
        checkpoint["numeric_stds"],
        int(config.get("category_buckets", 4096)),
        args.cache_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=ColumnCollator(),
    )
    model = AlignedColumnDecoder(
        input_dim=768,
        d_model=int(config.get("d_model", 192)),
        layers=int(config.get("layers", 4)),
        heads=int(config.get("heads", 6)),
        dropout=0.0,
        max_cols=int(config.get("max_cols", 1024)),
        category_buckets=int(config.get("category_buckets", 4096)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    decode_rows: list[dict[str, Any]] = []
    total_loss = 0.0
    total_token_acc = 0.0
    total_residue_acc = 0.0
    total_examples = 0
    token_counts = {token: 0 for token in TOKENS}
    gap_token_id = TOKEN_TO_ID["-"]

    with torch.no_grad():
        for batch in loader:
            moved = move_batch(batch, device)
            logits = model(
                **{
                    key: moved[key]
                    for key in (
                        "col_embeddings",
                        "profiles",
                        "mask",
                        "numeric_values",
                        "numeric_mask",
                        "category_ids",
                        "category_mask",
                    )
                }
            )
            loss, token_acc, batch_residue_acc = masked_loss_and_accuracy(
                logits,
                moved["target_tokens"],
                moved["mask"],
                gap_weight=float(config.get("gap_loss_weight", 0.5)),
            )
            batch_size = moved["target_tokens"].shape[0]
            total_loss += float(loss.item()) * batch_size
            total_token_acc += token_acc * batch_size
            total_residue_acc += batch_residue_acc * batch_size
            total_examples += batch_size

            predicted_ids = torch.argmax(logits, dim=-1)
            probabilities = F.softmax(logits, dim=-1)
            confidence = probabilities.max(dim=-1).values
            for row_idx in range(batch_size):
                mask = moved["mask"][row_idx]
                predicted_aligned = decode_tokens(predicted_ids[row_idx].cpu(), mask.cpu())
                target_aligned = decode_tokens(moved["target_tokens"][row_idx].cpu(), mask.cpu())
                for char in predicted_aligned:
                    token_counts[char] += 1
                row_confidence = float(confidence[row_idx][mask].mean().item())
                exact = predicted_aligned == target_aligned
                decode_rows.append(
                    {
                        "rank": len(decode_rows) + 1,
                        "split": args.split,
                        "cluster_index": batch["cluster_indices"][row_idx],
                        "kegg_entry": batch["kegg_entries"][row_idx],
                        "length_aligned": len(target_aligned),
                        "target_gap_count": target_aligned.count("-"),
                        "decoded_gap_count": predicted_aligned.count("-"),
                        "token_accuracy": sum(
                            p == t for p, t in zip(predicted_aligned, target_aligned)
                        )
                        / max(len(target_aligned), 1),
                        "residue_identity": residue_identity(predicted_aligned, target_aligned),
                        "mean_confidence": row_confidence,
                        "exact_match": exact,
                        "decoded_aligned": predicted_aligned,
                        "decoded_ungapped": ungap(predicted_aligned),
                        "target_aligned": target_aligned,
                        "target_ungapped": ungap(target_aligned),
                    }
                )

    metrics = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "examples": total_examples,
        "loss": total_loss / max(total_examples, 1),
        "token_accuracy": total_token_acc / max(total_examples, 1),
        "residue_accuracy": total_residue_acc / max(total_examples, 1),
        "decoded_sequences": min(args.num_decodes, len(decode_rows)),
        "elapsed_seconds": time.monotonic() - started,
    }

    metrics_path = out_dir / "decode_metrics.tsv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)

    rows_path = out_dir / "decoded_sequences.tsv"
    row_fields = [
        "rank",
        "split",
        "cluster_index",
        "kegg_entry",
        "length_aligned",
        "target_gap_count",
        "decoded_gap_count",
        "token_accuracy",
        "residue_identity",
        "mean_confidence",
        "exact_match",
        "decoded_ungapped",
        "target_ungapped",
        "decoded_aligned",
        "target_aligned",
    ]
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=row_fields)
        writer.writeheader()
        for row in decode_rows:
            writer.writerow({field: row[field] for field in row_fields})

    fasta_path = out_dir / "decoded_sequences.fasta"
    with fasta_path.open("w", encoding="utf-8") as handle:
        for row in decode_rows[: args.num_decodes]:
            handle.write(
                f">decoded rank={row['rank']} split={row['split']} cluster={row['cluster_index']} "
                f"kegg={row['kegg_entry']} residue_identity={row['residue_identity']:.4f} "
                f"mean_confidence={row['mean_confidence']:.4f}\n"
            )
            handle.write(f"{row['decoded_ungapped']}\n")

    print(
        "decode_eval "
        f"split={metrics['split']} examples={metrics['examples']} "
        f"loss={metrics['loss']:.5f} token_acc={metrics['token_accuracy']:.4f} "
        f"residue_acc={metrics['residue_accuracy']:.4f} "
        f"elapsed={metrics['elapsed_seconds']:.1f}s"
    )
    print(f"outputs metrics={metrics_path} rows={rows_path} fasta={fasta_path}")
    print(
        "decoded_token_counts "
        + " ".join(f"{token}:{count}" for token, count in token_counts.items() if count)
    )
    for row in decode_rows[: args.num_decodes]:
        print(
            f">decoded_{row['rank']} kegg={row['kegg_entry']} cluster={row['cluster_index']} "
            f"len={len(row['decoded_ungapped'])} residue_identity={row['residue_identity']:.4f} "
            f"confidence={row['mean_confidence']:.4f}"
        )
        print(row["decoded_ungapped"])
        print(
            f">target_{row['rank']} kegg={row['kegg_entry']} len={len(row['target_ungapped'])}"
        )
        print(row["target_ungapped"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
