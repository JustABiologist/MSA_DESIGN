#!/usr/bin/env python3
"""Aggregate GotEnzymes reaction rows into one condition-label row per sequence."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import TextIO


DEFAULT_TRAINING_ROOT = Path("/mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim")
DEFAULT_REACTION_ROWS = DEFAULT_TRAINING_ROOT / "kept_reaction_parameters.tsv.gz"
DEFAULT_OUT = DEFAULT_TRAINING_ROOT / "sequence_label_summary.tsv.gz"
NUMERIC_FIELDS = ("kcat_1_per_s", "km_mM", "kcat_over_km_1_per_mM_s", "topt_C", "tm_C")
CATEGORICAL_FIELDS = (
    "domain",
    "reaction_id",
    "ec_numbers",
    "compound_id",
    "selected_uniprot_accession",
)
IDENTITY_FIELDS = (
    "gene_id",
    "organism_code",
    "sequence_id",
    "sequence_length",
    "uniprot_fasta_source",
    "uniprot_header",
)


class LabelAggregate:
    __slots__ = ("row_count", "numeric_sums", "numeric_counts", "categories", "identity")

    def __init__(self) -> None:
        self.row_count = 0
        self.numeric_sums = [0.0 for _ in NUMERIC_FIELDS]
        self.numeric_counts = [0 for _ in NUMERIC_FIELDS]
        self.categories = [set() for _ in CATEGORICAL_FIELDS]
        self.identity: dict[str, str] = {}

    def add(self, row: dict[str, str]) -> None:
        self.row_count += 1
        for idx, field in enumerate(NUMERIC_FIELDS):
            text = row.get(field, "")
            try:
                value = float(text)
            except ValueError:
                continue
            if math.isfinite(value):
                self.numeric_sums[idx] += value
                self.numeric_counts[idx] += 1
        for idx, field in enumerate(CATEGORICAL_FIELDS):
            for value in split_values(row.get(field, "")):
                self.categories[idx].add(value)
        for field in IDENTITY_FIELDS:
            if field not in self.identity and row.get(field):
                self.identity[field] = row[field]


def split_values(text: str) -> list[str]:
    values: list[str] = []
    for part in str(text).replace(",", ";").split(";"):
        value = part.strip()
        if value and value.lower() != "nan":
            values.append(value)
    return values


def open_text(path: Path, mode: str) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8", newline="")
    return path.open(mode, encoding="utf-8", newline="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reaction-rows", default=str(DEFAULT_REACTION_ROWS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--limit", type=int, default=None, help="Optional row cap for smoke tests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reaction_rows = Path(args.reaction_rows)
    out_path = Path(args.out)
    if not reaction_rows.exists():
        raise SystemExit(f"Reaction row file not found: {reaction_rows}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    aggregates: dict[str, LabelAggregate] = defaultdict(LabelAggregate)
    started = time.monotonic()
    read_rows = 0
    with open_text(reaction_rows, "r") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            kegg_entry = row.get("kegg_entry") or row.get("sequence_id")
            if not kegg_entry:
                continue
            aggregates[kegg_entry].add(row)
            read_rows += 1
            if args.limit is not None and read_rows >= args.limit:
                break
            if args.progress_every > 0 and read_rows % args.progress_every == 0:
                elapsed = time.monotonic() - started
                print(
                    f"read_rows={read_rows:,} sequences={len(aggregates):,} "
                    f"elapsed_seconds={elapsed:.1f}",
                    flush=True,
                )

    fieldnames = (
        ["kegg_entry", "reaction_row_count"]
        + list(IDENTITY_FIELDS)
        + [f"{field}_mean" for field in NUMERIC_FIELDS]
        + [f"{field}_count" for field in NUMERIC_FIELDS]
        + [f"{field}_values" for field in CATEGORICAL_FIELDS]
    )
    with open_text(out_path, "w") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for kegg_entry in sorted(aggregates):
            aggregate = aggregates[kegg_entry]
            out_row: dict[str, str] = {
                "kegg_entry": kegg_entry,
                "reaction_row_count": str(aggregate.row_count),
            }
            for field in IDENTITY_FIELDS:
                out_row[field] = aggregate.identity.get(field, "")
            for idx, field in enumerate(NUMERIC_FIELDS):
                count = aggregate.numeric_counts[idx]
                out_row[f"{field}_mean"] = (
                    f"{aggregate.numeric_sums[idx] / count:.8g}" if count else ""
                )
                out_row[f"{field}_count"] = str(count)
            for idx, field in enumerate(CATEGORICAL_FIELDS):
                out_row[f"{field}_values"] = ";".join(sorted(aggregate.categories[idx]))
            writer.writerow(out_row)

    elapsed = time.monotonic() - started
    print(
        f"Wrote {len(aggregates):,} sequence label summaries from {read_rows:,} rows to {out_path} "
        f"in {elapsed:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
