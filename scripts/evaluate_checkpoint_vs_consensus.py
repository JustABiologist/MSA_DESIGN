#!/usr/bin/env python3
"""Compare deterministic checkpoint decode with leave-one-row-out consensus."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, OrderedDict
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

from msa_design_model import decode_tokens_until_stop  # noqa: E402
from train_mean_start_ccdd_from_cached_msas import (  # noqa: E402
    AA_TOKENS,
    AMP_MODES,
    DEFAULT_EMBEDDING_MANIFEST,
    DEFAULT_LABEL_SUMMARY,
    CachedMSARowDataset,
    MeanStartCCDDModel,
    RowExample,
    RowReconstructionCollator,
    autocast_context,
    build_examples,
    move_batch,
    parse_path_rewrites,
    read_embedding_manifest,
    uses_msa_embedding_memory,
    uses_gap_inclusive_msa_mask,
    weighted_residue_accuracy,
)

AA_SET = set(AA_TOKENS)


class MetadataCache:
    def __init__(self, max_size: int = 128) -> None:
        self.max_size = max_size
        self.items: OrderedDict[Path, list[str]] = OrderedDict()

    def sequences(self, path: Path) -> list[str]:
        if path in self.items:
            value = self.items.pop(path)
            self.items[path] = value
            return value
        metadata = json.loads(path.read_text(encoding="utf-8"))
        sequences = [str(sequence).upper() for sequence in metadata["cleaned_sequences"]]
        self.items[path] = sequences
        while len(self.items) > self.max_size:
            self.items.popitem(last=False)
        return sequences


def consensus_for_example(example: RowExample, cache: MetadataCache) -> tuple[str, int, int]:
    sequences = cache.sequences(example.metadata_path)
    target_aligned = sequences[example.row_index]
    consensus: list[str] = []
    correct = 0
    total = 0
    for col, target_char in enumerate(target_aligned):
        if target_char not in AA_SET:
            continue
        counts: Counter[str] = Counter()
        for idx, sequence in enumerate(sequences):
            if idx == example.row_index or col >= len(sequence):
                continue
            char = sequence[col]
            if char in AA_SET:
                counts[char] += 1
        best = max(AA_TOKENS, key=lambda aa: (counts[aa], -AA_TOKENS.index(aa))) if counts else "X"
        consensus.append(best)
        correct += int(best == target_char)
        total += 1
    return "".join(consensus), correct, total


def sequence_identity(predicted: str, target: str) -> float:
    if not target:
        return 0.0
    compare = min(len(predicted), len(target))
    return sum(1 for idx in range(compare) if predicted[idx] == target[idx]) / len(target)


def add_masked_counts(
    totals: dict[str, float],
    prefix: str,
    predicted_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    mask_f = mask.to(dtype=torch.float32)
    correct = ((predicted_tokens == target_tokens) & mask).to(dtype=torch.float32)
    totals[f"{prefix}_correct"] += float(correct.sum().item())
    totals[f"{prefix}_total"] += float(mask_f.sum().item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding-manifest", default=str(DEFAULT_EMBEDDING_MANIFEST))
    parser.add_argument("--label-summary", default=str(DEFAULT_LABEL_SUMMARY))
    parser.add_argument(
        "--path-rewrite",
        action="append",
        default=[],
        help="Rewrite manifest paths with OLD=NEW prefixes before opening cached files.",
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--example-limit", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cache-size", type=int, default=128)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--amp",
        choices=("checkpoint", *AMP_MODES),
        default="checkpoint",
        help="Autocast mode for model forward. By default, reuse the checkpoint config.",
    )
    parser.add_argument("--progress-every", type=int, default=0, help="Print progress every N examples.")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--out-tsv", default=None)
    args = parser.parse_args()

    try:
        path_rewrites = parse_path_rewrites(args.path_rewrite)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config: dict[str, Any] = checkpoint.get("config", {})
    numeric_means = checkpoint["numeric_means"]
    numeric_stds = checkpoint["numeric_stds"]

    rng = random.Random(args.seed)
    rows = read_embedding_manifest(Path(args.embedding_manifest), split=args.split, path_rewrites=path_rewrites)
    rng.shuffle(rows)
    examples = build_examples(rows, max_rows_per_msa=config.get("max_rows_per_msa"))
    rng.shuffle(examples)
    examples = examples[: args.example_limit]

    labels: dict[str, dict[str, str]] = {}
    import gzip

    with gzip.open(args.label_summary, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            kegg_entry = row.get("kegg_entry")
            if kegg_entry:
                labels[kegg_entry] = row

    continuous_target_mode = str(config.get("continuous_target_mode", "token_embedding"))
    dataset = CachedMSARowDataset(
        examples=examples,
        labels=labels,
        numeric_means=numeric_means,
        numeric_stds=numeric_stds,
        category_buckets=int(config.get("category_buckets", 4096)),
        cache_size=args.cache_size,
        consensus_loss_mode=str(config.get("consensus_loss_mode", "none")),
        consensus_match_weight=float(config.get("consensus_match_weight", 0.35)),
        nonconsensus_weight=float(config.get("nonconsensus_weight", 2.5)),
        unobserved_nonconsensus_weight=float(config.get("unobserved_nonconsensus_weight", 1.0)),
        max_sequence_loss_weight=float(config.get("max_sequence_loss_weight", 3.0)),
        variable_column_min_entropy=float(config.get("variable_column_min_entropy", 0.05)),
        variable_column_max_consensus=float(config.get("variable_column_max_consensus", 0.92)),
        require_msa_embeddings=uses_msa_embedding_memory(str(config.get("memory_mode", "profile_row"))),
        msa_embedding_dtype=str(config.get("msa_embedding_dtype", "float32")),
        max_msa_context_rows=config.get("max_msa_context_rows"),
        gap_inclusive_msa_mask=uses_gap_inclusive_msa_mask(str(config.get("memory_mode", "profile_row"))),
        require_target_continuous_embeddings=continuous_target_mode == "target_row_embedding",
    )
    first_item = dataset[0]
    memory_mode = str(config.get("memory_mode", "profile_row"))
    row_embedding_dim = int(first_item["row_embeddings"].shape[-1])
    msa_embedding_dim = (
        int(first_item["msa_embeddings"].shape[-1]) if uses_msa_embedding_memory(memory_mode) else 1
    )
    target_continuous_dim = int(first_item["target_continuous_embeddings"].shape[-1])
    model = MeanStartCCDDModel(
        row_embedding_dim=row_embedding_dim,
        d_model=int(config.get("d_model", 192)),
        layers=int(config.get("layers", 4)),
        heads=int(config.get("heads", 6)),
        dropout=float(config.get("dropout", 0.1)),
        max_sequence_length=int(config.get("max_sequence_length", 1024)),
        diffusion_timesteps=int(config.get("diffusion_timesteps", 250)),
        category_buckets=int(config.get("category_buckets", 4096)),
        memory_mode=memory_mode,
        profile_feature_mode=str(config.get("profile_feature_mode", "full")),
        msa_embedding_dim=msa_embedding_dim,
        continuous_target_mode=continuous_target_mode,
        target_continuous_dim=target_continuous_dim,
        msa_axial_layers=int(config.get("msa_axial_layers", 1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    amp_mode = str(config.get("amp", "off")) if args.amp == "checkpoint" else args.amp

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=RowReconstructionCollator(
            max_sequence_length=int(config.get("max_sequence_length", 1024)),
            tail_stop_weight=float(config.get("tail_stop_weight", 0.05)),
            profile_feature_mode=str(config.get("profile_feature_mode", "full")),
        ),
    )

    cache = MetadataCache(max_size=args.cache_size)
    consensus_by_rank = [consensus_for_example(example, cache) for example in examples]
    consensus_correct = sum(item[1] for item in consensus_by_rank)
    consensus_total = sum(item[2] for item in consensus_by_rank)

    rows_out: list[dict[str, object]] = []
    model_residue_weighted_sum = 0.0
    model_residue_examples = 0
    model_identity_sum = 0.0
    consensus_identity_sum = 0.0
    model_wins = 0
    masked_totals = {
        "consensus_correct": 0.0,
        "consensus_total": 0.0,
        "nonconsensus_correct": 0.0,
        "nonconsensus_total": 0.0,
        "variable_nonconsensus_correct": 0.0,
        "variable_nonconsensus_total": 0.0,
        "residue_total": 0.0,
    }
    rank_offset = 0
    with torch.no_grad():
        for batch in loader:
            moved = move_batch(batch, device)
            batch_size = moved["target_tokens"].shape[0]
            timesteps = torch.zeros((batch_size,), dtype=torch.long, device=device)
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
                )
            residue_acc = weighted_residue_accuracy(outputs["logits"], moved["target_tokens"], moved["loss_weights"])
            model_residue_weighted_sum += residue_acc * batch_size
            model_residue_examples += batch_size
            predicted = torch.argmax(outputs["logits"], dim=-1).detach().cpu()
            predicted_device = torch.argmax(outputs["logits"], dim=-1)
            residue_mask = moved["loss_weights"] > 0.5
            consensus_mask = moved["consensus_match_mask"] & moved["consensus_observed_mask"] & residue_mask
            nonconsensus_mask = moved["nonconsensus_mask"] & residue_mask
            variable_nonconsensus_mask = moved["variable_nonconsensus_mask"] & residue_mask
            add_masked_counts(
                masked_totals,
                "consensus",
                predicted_device,
                moved["target_tokens"],
                consensus_mask,
            )
            add_masked_counts(
                masked_totals,
                "nonconsensus",
                predicted_device,
                moved["target_tokens"],
                nonconsensus_mask,
            )
            add_masked_counts(
                masked_totals,
                "variable_nonconsensus",
                predicted_device,
                moved["target_tokens"],
                variable_nonconsensus_mask,
            )
            masked_totals["residue_total"] += float(residue_mask.to(dtype=torch.float32).sum().item())
            for idx in range(batch_size):
                rank = rank_offset + idx
                decoded = decode_tokens_until_stop(predicted[idx].tolist())
                target = batch["target_sequences"][idx]
                consensus = consensus_by_rank[rank][0]
                model_identity = sequence_identity(decoded, target)
                consensus_identity = sequence_identity(consensus, target)
                model_identity_sum += model_identity
                consensus_identity_sum += consensus_identity
                model_wins += int(model_identity > consensus_identity)
                rows_out.append(
                    {
                        "rank": rank + 1,
                        "cluster_index": batch["cluster_indices"][idx],
                        "kegg_entry": batch["kegg_entries"][idx],
                        "target_length": len(target),
                        "model_identity": model_identity,
                        "consensus_identity": consensus_identity,
                        "model_minus_consensus": model_identity - consensus_identity,
                        "decoded": decoded,
                        "consensus": consensus,
                        "target": target,
                    }
                )
            rank_offset += batch_size
            if args.progress_every > 0 and rank_offset % args.progress_every == 0:
                print(f"progress_examples={rank_offset}/{len(examples)}", flush=True)

    n = max(len(rows_out), 1)
    summary = {
        "checkpoint": str(args.checkpoint),
        "embedding_manifest": str(args.embedding_manifest),
        "label_summary": str(args.label_summary),
        "split": args.split,
        "seed": args.seed,
        "example_limit": args.example_limit,
        "examples": len(rows_out),
        "batch_size": args.batch_size,
        "device": str(device),
        "amp": amp_mode,
        "model_t0_residue_accuracy": model_residue_weighted_sum / max(model_residue_examples, 1),
        "model_t0_mean_sequence_identity": model_identity_sum / n,
        "consensus_residue_accuracy": consensus_correct / max(consensus_total, 1),
        "consensus_mean_sequence_identity": consensus_identity_sum / n,
        "model_beats_consensus": model_wins,
        "model_beats_consensus_total": len(rows_out),
        "model_consensus_position_accuracy": (
            masked_totals["consensus_correct"] / max(masked_totals["consensus_total"], 1.0)
        ),
        "model_nonconsensus_position_accuracy": (
            masked_totals["nonconsensus_correct"] / max(masked_totals["nonconsensus_total"], 1.0)
        ),
        "model_variable_nonconsensus_position_accuracy": (
            masked_totals["variable_nonconsensus_correct"]
            / max(masked_totals["variable_nonconsensus_total"], 1.0)
        ),
        "nonconsensus_fraction": masked_totals["nonconsensus_total"] / max(masked_totals["residue_total"], 1.0),
        "variable_nonconsensus_fraction": (
            masked_totals["variable_nonconsensus_total"] / max(masked_totals["residue_total"], 1.0)
        ),
    }
    print(f"examples={summary['examples']}")
    print(f"model_t0_residue_accuracy={summary['model_t0_residue_accuracy']:.6f}")
    print(f"model_t0_mean_sequence_identity={summary['model_t0_mean_sequence_identity']:.6f}")
    print(f"consensus_residue_accuracy={summary['consensus_residue_accuracy']:.6f}")
    print(f"consensus_mean_sequence_identity={summary['consensus_mean_sequence_identity']:.6f}")
    print(f"model_beats_consensus={model_wins}/{len(rows_out)}")
    print(
        "model_consensus_position_accuracy="
        f"{summary['model_consensus_position_accuracy']:.6f}"
    )
    print(
        "model_nonconsensus_position_accuracy="
        f"{summary['model_nonconsensus_position_accuracy']:.6f}"
    )
    print(
        "model_variable_nonconsensus_position_accuracy="
        f"{summary['model_variable_nonconsensus_position_accuracy']:.6f}"
    )
    print(f"nonconsensus_fraction={summary['nonconsensus_fraction']:.6f}")
    print(f"variable_nonconsensus_fraction={summary['variable_nonconsensus_fraction']:.6f}")

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote_summary={summary_path}")

    if args.out_tsv:
        out_path = Path(args.out_tsv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"wrote={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
