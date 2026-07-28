#!/usr/bin/env python3
"""Summarize MMseqs cluster TSVs for downstream MSA selection."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_BINS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]


def size_bin(size: int) -> str:
    previous = 1
    for boundary in DEFAULT_BINS:
        if size <= boundary:
            if boundary == 1:
                return "1"
            return f"{previous + 1}-{boundary}"
        previous = boundary
    return f">{DEFAULT_BINS[-1]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a MMseqs createtsv cluster file and list clusters suitable for MSA construction."
    )
    parser.add_argument("--clusters", required=True, help="MMseqs cluster TSV with representative and member columns.")
    parser.add_argument("--cluster-stats", required=True, help="Per-cluster output TSV.")
    parser.add_argument("--summary", required=True, help="Aggregate summary output TSV.")
    parser.add_argument("--good-clusters", required=True, help="Candidate cluster output TSV.")
    parser.add_argument("--good-members", default="", help="Optional candidate-cluster member TSV output.")
    parser.add_argument("--min-size", type=int, default=16, help="Minimum cluster size for MSA candidates.")
    parser.add_argument("--max-size", type=int, default=4096, help="Maximum cluster size for MSA candidates.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cluster_path = Path(args.clusters)
    sizes: defaultdict[str, int] = defaultdict(int)

    with cluster_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            representative = row[0]
            sizes[representative] += 1

    total_sequences = sum(sizes.values())
    total_clusters = len(sizes)
    good_reps = {rep for rep, size in sizes.items() if args.min_size <= size <= args.max_size}
    histogram = Counter(size_bin(size) for size in sizes.values())

    cluster_stats_path = Path(args.cluster_stats)
    cluster_stats_path.parent.mkdir(parents=True, exist_ok=True)
    with cluster_stats_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["representative", "cluster_size", "msa_candidate"])
        for representative, size in sorted(sizes.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([representative, size, "yes" if representative in good_reps else "no"])

    good_path = Path(args.good_clusters)
    good_path.parent.mkdir(parents=True, exist_ok=True)
    with good_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["representative", "cluster_size"])
        for representative, size in sorted(sizes.items(), key=lambda item: (-item[1], item[0])):
            if representative in good_reps:
                writer.writerow([representative, size])

    if args.good_members:
        good_members_path = Path(args.good_members)
        good_members_path.parent.mkdir(parents=True, exist_ok=True)
        with cluster_path.open("r", encoding="utf-8", errors="replace", newline="") as source, good_members_path.open(
            "w", encoding="utf-8", newline=""
        ) as target:
            reader = csv.reader(source, delimiter="\t")
            writer = csv.writer(target, delimiter="\t")
            writer.writerow(["representative", "member"])
            for row in reader:
                if len(row) >= 2 and row[0] in good_reps:
                    writer.writerow(row[:2])

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["metric", "value"])
        writer.writerow(["input_cluster_tsv", str(cluster_path)])
        writer.writerow(["total_sequences", total_sequences])
        writer.writerow(["total_clusters", total_clusters])
        writer.writerow(["singleton_clusters", histogram["1"]])
        writer.writerow(["msa_candidate_min_size", args.min_size])
        writer.writerow(["msa_candidate_max_size", args.max_size])
        writer.writerow(["msa_candidate_clusters", len(good_reps)])
        writer.writerow(["msa_candidate_sequences", sum(sizes[rep] for rep in good_reps)])
        for label in ["1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129-256", "257-512", "513-1024", "1025-2048", "2049-4096", "4097-8192", "8193-16384", ">16384"]:
            writer.writerow([f"cluster_size_{label}", histogram[label]])

    print(f"Wrote {cluster_stats_path}")
    print(f"Wrote {good_path}")
    if args.good_members:
        print(f"Wrote {args.good_members}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
