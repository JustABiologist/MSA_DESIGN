#!/usr/bin/env python3
"""Evaluate leave-one-target-row-out consensus reconstruction baselines."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


DEFAULT_ROOT = Path("/mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim")
DEFAULT_MANIFEST = DEFAULT_ROOT / "esm_msa_embeddings_col" / "embedding_manifest.tsv"
AA_TOKENS = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA_TOKENS)


def ungap(sequence: str) -> str:
    return "".join(char for char in sequence.upper() if char in AA_SET)


def read_embedding_manifest(path: Path, split: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") != "embedded" or row.get("split") != split:
                continue
            cluster_index = row.get("cluster_index", "")
            if not cluster_index or cluster_index in seen:
                continue
            seen.add(cluster_index)
            metadata_path = Path(row.get("metadata_path", ""))
            if metadata_path.exists():
                rows.append(row)
    return rows


def consensus_for_row(sequences: list[str], row_index: int) -> tuple[str, int, int, int]:
    target_aligned = sequences[row_index]
    consensus: list[str] = []
    correct = 0
    total = 0
    resolved = 0
    for col, target_char in enumerate(target_aligned):
        if target_char not in AA_SET:
            continue
        counts: Counter[str] = Counter()
        for idx, sequence in enumerate(sequences):
            if idx == row_index or col >= len(sequence):
                continue
            char = sequence[col]
            if char in AA_SET:
                counts[char] += 1
        if counts:
            best = max(AA_TOKENS, key=lambda aa: (counts[aa], -AA_TOKENS.index(aa)))
            resolved += 1
        else:
            best = "X"
        consensus.append(best)
        correct += int(best == target_char)
        total += 1
    return "".join(consensus), correct, total, resolved


def iter_examples(rows: list[dict[str, str]], max_rows_per_msa: int | None):
    for row in rows:
        metadata_path = Path(row["metadata_path"])
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        headers = [str(header).split()[0] for header in metadata.get("headers", [])]
        sequences = [str(sequence).upper() for sequence in metadata.get("cleaned_sequences", [])]
        if not headers or len(headers) != len(sequences) or len(sequences) <= 1:
            continue
        row_count = len(sequences)
        if max_rows_per_msa is not None:
            row_count = min(row_count, max_rows_per_msa)
        for row_index in range(row_count):
            target = ungap(sequences[row_index])
            if not target:
                continue
            yield {
                "cluster_index": row["cluster_index"],
                "kegg_entry": headers[row_index],
                "row_index": row_index,
                "sequences": sequences,
                "target": target,
            }


def parse_decode_fasta(path: Path) -> list[dict[str, str]]:
    if not path:
        return []
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks)))

    pairs: list[dict[str, str]] = []
    for idx in range(0, len(records), 2):
        if idx + 1 >= len(records):
            break
        decoded_header, decoded = records[idx]
        target_header, target = records[idx + 1]
        if not decoded_header.startswith("decoded") or not target_header.startswith("target"):
            continue
        pairs.append({"decoded_header": decoded_header, "decoded": decoded, "target": target})
    return pairs


def sequence_identity(predicted: str, target: str) -> float:
    if not target:
        return 0.0
    compare = min(len(predicted), len(target))
    return sum(1 for idx in range(compare) if predicted[idx] == target[idx]) / len(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-rows-per-msa", type=int, default=None)
    parser.add_argument("--example-limit", type=int, default=2048)
    parser.add_argument("--decode-fasta", default=None)
    parser.add_argument("--out-tsv", default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_embedding_manifest(Path(args.embedding_manifest), args.split)
    rng.shuffle(rows)
    examples = list(iter_examples(rows, args.max_rows_per_msa))
    rng.shuffle(examples)
    if args.example_limit is not None:
        examples = examples[: args.example_limit]

    rows_out: list[dict[str, object]] = []
    total_correct = 0
    total_residues = 0
    total_resolved = 0
    identities: list[float] = []
    for example in examples:
        consensus, correct, total, resolved = consensus_for_row(example["sequences"], int(example["row_index"]))
        identity = correct / max(total, 1)
        identities.append(identity)
        total_correct += correct
        total_residues += total
        total_resolved += resolved
        rows_out.append(
            {
                "rank": len(rows_out) + 1,
                "cluster_index": example["cluster_index"],
                "kegg_entry": example["kegg_entry"],
                "row_index": example["row_index"],
                "target_length": len(example["target"]),
                "consensus_identity": identity,
                "consensus_resolved_fraction": resolved / max(total, 1),
                "consensus": consensus,
                "target": example["target"],
            }
        )

    identities_sorted = sorted(identities)
    def quantile(q: float) -> float:
        if not identities_sorted:
            return 0.0
        idx = min(len(identities_sorted) - 1, max(0, round(q * (len(identities_sorted) - 1))))
        return identities_sorted[idx]

    print(f"examples={len(examples)}")
    print(f"consensus_residue_accuracy={total_correct / max(total_residues, 1):.6f}")
    print(f"consensus_mean_sequence_identity={sum(identities) / max(len(identities), 1):.6f}")
    print(f"consensus_median_sequence_identity={quantile(0.5):.6f}")
    print(f"consensus_q25_sequence_identity={quantile(0.25):.6f}")
    print(f"consensus_q75_sequence_identity={quantile(0.75):.6f}")
    print(f"consensus_resolved_fraction={total_resolved / max(total_residues, 1):.6f}")

    if args.decode_fasta:
        pairs = parse_decode_fasta(Path(args.decode_fasta))
        limit = min(len(pairs), len(rows_out))
        model_ids: list[float] = []
        consensus_ids: list[float] = []
        wins = 0
        for idx in range(limit):
            model_id = sequence_identity(pairs[idx]["decoded"], pairs[idx]["target"])
            consensus_id = float(rows_out[idx]["consensus_identity"])
            model_ids.append(model_id)
            consensus_ids.append(consensus_id)
            wins += int(model_id > consensus_id)
            rows_out[idx]["decoded"] = pairs[idx]["decoded"]
            rows_out[idx]["model_identity"] = model_id
            rows_out[idx]["model_minus_consensus"] = model_id - consensus_id
        if limit:
            print(f"decode_examples={limit}")
            print(f"decode_model_mean_identity={sum(model_ids) / limit:.6f}")
            print(f"decode_consensus_mean_identity={sum(consensus_ids) / limit:.6f}")
            print(f"decode_model_beats_consensus={wins}/{limit}")

    if args.out_tsv:
        out_path = Path(args.out_tsv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "rank",
            "cluster_index",
            "kegg_entry",
            "row_index",
            "target_length",
            "consensus_identity",
            "consensus_resolved_fraction",
            "model_identity",
            "model_minus_consensus",
            "decoded",
            "consensus",
            "target",
        ]
        with out_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"wrote={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
