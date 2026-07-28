#!/usr/bin/env python3
"""Sample balanced GotEnzymes families and fetch source KEGG protein sequences."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
import time
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from remap_kegg_sequences import (
    ENZYME_PREFIX,
    EXPECTED_ENZYME_COLUMNS,
    KINETIC_COLUMN_VALUE_FIELDS,
    SequenceRecord,
    fetch_missing_from_kegg_rest,
    load_fasta_sequences,
    load_local_kegg_sequences,
    split_ec_numbers,
    unique_join,
    write_fasta,
    write_sequence_index,
)


def safe_family_id(ec_number: str, reaction_id: str) -> str:
    ec_part = ec_number.replace(".", "_").replace("-", "x")
    return f"ec_{ec_part}__rxn_{reaction_id}"


def ec_class(ec_number: str) -> str:
    first = ec_number.split(".", 1)[0]
    return first if first.isdigit() else "other"


def stable_score(family_key: str, entry: str) -> int:
    digest = hashlib.sha1(f"{family_key}\t{entry}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def add_capped(values: OrderedDict[str, None], value: str, limit: int = 25) -> None:
    if value in values:
        return
    if len(values) < limit:
        values[value] = None


def finite_text(value: str) -> str | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return value.strip()


def row_kinetic_values(row: list[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for column_index, field_name in KINETIC_COLUMN_VALUE_FIELDS.items():
        value = finite_text(row[column_index])
        if value is not None:
            values[field_name] = value
    return values


@dataclass
class EntryFamilyMetadata:
    row_count: int = 0
    domains: OrderedDict[str, None] = field(default_factory=OrderedDict)
    compounds: OrderedDict[str, None] = field(default_factory=OrderedDict)
    kinetic_values: dict[str, OrderedDict[str, None]] = field(
        default_factory=lambda: {field_name: OrderedDict() for field_name in KINETIC_COLUMN_VALUE_FIELDS.values()}
    )

    def observe(self, domain: str, compound_id: str, kinetic_values: dict[str, str]) -> None:
        self.row_count += 1
        add_capped(self.domains, domain)
        add_capped(self.compounds, compound_id)
        for field_name, value in kinetic_values.items():
            self.kinetic_values[field_name].setdefault(value, None)


@dataclass
class FamilySample:
    ec_number: str
    reaction_id: str
    rows_seen: int = 0
    entry_scores: dict[str, int] = field(default_factory=dict)
    entry_metadata: dict[str, EntryFamilyMetadata] = field(default_factory=dict)
    compounds: OrderedDict[str, None] = field(default_factory=OrderedDict)
    domains: OrderedDict[str, None] = field(default_factory=OrderedDict)

    @property
    def key(self) -> str:
        return f"{self.ec_number}\t{self.reaction_id}"

    @property
    def family_id(self) -> str:
        return safe_family_id(self.ec_number, self.reaction_id)

    @property
    def ec_class(self) -> str:
        return ec_class(self.ec_number)

    def observe(
        self,
        entry: str,
        domain: str,
        compound_id: str,
        kinetic_values: dict[str, str],
        max_entries: int,
    ) -> None:
        self.rows_seen += 1
        add_capped(self.domains, domain)
        add_capped(self.compounds, compound_id)
        metadata = self.entry_metadata.setdefault(entry, EntryFamilyMetadata())
        metadata.observe(domain=domain, compound_id=compound_id, kinetic_values=kinetic_values)
        if entry in self.entry_scores:
            return
        score = stable_score(self.key, entry)
        if len(self.entry_scores) < max_entries:
            self.entry_scores[entry] = score
            return
        worst_entry, worst_score = max(self.entry_scores.items(), key=lambda item: item[1])
        if score < worst_score:
            del self.entry_scores[worst_entry]
            self.entry_scores[entry] = score

    def sampled_entries(self) -> list[str]:
        return [
            entry
            for entry, _score in sorted(
                self.entry_scores.items(),
                key=lambda item: (item[1], item[0]),
            )
        ]


@dataclass
class SelectedFamily:
    selection_index: int
    family: FamilySample
    entries: list[str]


def iter_enzyme_rows(zip_path: Path, max_enzyme_files: int | None) -> tuple[int, list[str]]:
    with zipfile.ZipFile(zip_path) as zf:
        enzyme_names = sorted(
            name for name in zf.namelist() if name.startswith(ENZYME_PREFIX) and name.endswith(".txt")
        )
        if max_enzyme_files is not None:
            enzyme_names = enzyme_names[:max_enzyme_files]
        row_count = 0
        for file_index, name in enumerate(enzyme_names, start=1):
            with zf.open(name) as handle:
                for raw_line in handle:
                    line = raw_line.decode("utf-8", "replace").rstrip("\n\r")
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) != EXPECTED_ENZYME_COLUMNS:
                        continue
                    row_count += 1
                    yield row_count, parts
            if file_index % 1000 == 0:
                print(
                    f"scanned {file_index:,}/{len(enzyme_names):,} enzyme files; "
                    f"rows={row_count:,}",
                    flush=True,
                )


def scan_families(
    zip_path: Path,
    max_entries_per_family: int,
    max_enzyme_files: int | None,
) -> dict[str, FamilySample]:
    families: dict[str, FamilySample] = {}
    start = time.time()
    row_count = 0
    for row_count, row in iter_enzyme_rows(zip_path, max_enzyme_files=max_enzyme_files):
        gene_id = row[0]
        organism_code = row[1]
        domain = row[2]
        reaction_id = row[3]
        compound_id = row[5]
        kinetic_values = row_kinetic_values(row)
        entry = f"{organism_code}:{gene_id}"
        for ec_number in split_ec_numbers(row[4]):
            key = f"{ec_number}\t{reaction_id}"
            family = families.get(key)
            if family is None:
                family = FamilySample(ec_number=ec_number, reaction_id=reaction_id)
                families[key] = family
            family.observe(
                entry=entry,
                domain=domain,
                compound_id=compound_id,
                kinetic_values=kinetic_values,
                max_entries=max_entries_per_family,
            )
    print(
        f"Finished scan: rows={row_count:,} families={len(families):,} "
        f"elapsed={time.time() - start:.1f}s",
        flush=True,
    )
    return families


def choose_families(
    families: dict[str, FamilySample],
    target_sequences: int,
    seqs_per_family: int,
    min_seqs_per_family: int,
) -> list[SelectedFamily]:
    candidates = [
        family
        for family in families.values()
        if len(family.entry_scores) >= min_seqs_per_family
    ]
    by_class: OrderedDict[str, list[FamilySample]] = OrderedDict()
    for family in sorted(
        candidates,
        key=lambda item: (
            item.ec_class,
            -len(item.entry_scores),
            -item.rows_seen,
            item.ec_number,
            item.reaction_id,
        ),
    ):
        by_class.setdefault(family.ec_class, []).append(family)
    if not by_class:
        raise SystemExit("No families had enough sampled KEGG entries.")

    selected: list[SelectedFamily] = []
    used_entries: OrderedDict[str, None] = OrderedDict()
    class_names = list(by_class.keys())
    class_positions = {name: 0 for name in class_names}

    while len(used_entries) < target_sequences:
        progressed = False
        for class_name in class_names:
            families_for_class = by_class[class_name]
            while class_positions[class_name] < len(families_for_class):
                family = families_for_class[class_positions[class_name]]
                class_positions[class_name] += 1
                available = [entry for entry in family.sampled_entries() if entry not in used_entries]
                remaining = target_sequences - len(used_entries)
                take = min(seqs_per_family, remaining, len(available))
                if take < min_seqs_per_family and remaining > take:
                    continue
                if take <= 0:
                    continue
                chosen = available[:take]
                for entry in chosen:
                    used_entries[entry] = None
                selected.append(
                    SelectedFamily(
                        selection_index=len(selected) + 1,
                        family=family,
                        entries=chosen,
                    )
                )
                progressed = True
                break
            if len(used_entries) >= target_sequences:
                break
        if not progressed:
            break

    if len(used_entries) < target_sequences:
        raise SystemExit(
            f"Only selected {len(used_entries):,} unique entries; requested {target_sequences:,}. "
            "Lower --min-seqs-per-family or --seqs-per-family."
        )
    return selected


def selected_entries(selected: list[SelectedFamily]) -> OrderedDict[str, None]:
    entries: OrderedDict[str, None] = OrderedDict()
    for selected_family in selected:
        for entry in selected_family.entries:
            entries.setdefault(entry, None)
    return entries


def load_sequences(args: argparse.Namespace, entries: OrderedDict[str, None]) -> dict[str, SequenceRecord]:
    wanted = set(entries.keys())
    sequences: dict[str, SequenceRecord] = {}
    if args.sequence_fasta:
        sequences.update(load_fasta_sequences([Path(path) for path in args.sequence_fasta], wanted=wanted))
        print(f"Loaded {len(sequences):,} sequences from existing FASTA files.", flush=True)
    if args.kegg_root:
        before = len(sequences)
        local = load_local_kegg_sequences(Path(args.kegg_root), entries)
        sequences.update({entry: record for entry, record in local.items() if entry not in sequences})
        print(f"Loaded {len(sequences) - before:,} additional sequences from local KEGG.", flush=True)

    missing = [entry for entry in entries if entry not in sequences]
    if missing and args.fetch_missing == "rest":
        before = len(sequences)
        fetched = fetch_missing_from_kegg_rest(
            entries=missing,
            cache_dir=Path(args.cache_dir),
            batch_size=args.rest_batch_size,
            timeout=args.timeout,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
            max_rest_requests=args.max_rest_requests,
            allow_large_rest_run=args.allow_large_rest_run,
        )
        sequences.update({entry: record for entry, record in fetched.items() if entry not in sequences})
        print(f"Fetched/loaded {len(sequences) - before:,} additional sequences from KEGG REST.", flush=True)
    return sequences


def write_family_manifest(path: Path, selected: list[SelectedFamily], sequences: dict[str, SequenceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "selection_index",
        "family_id",
        "ec_class",
        "ec_number",
        "reaction_id",
        "selected_entries",
        "available_sequence_count",
        "sample_pool_size",
        "rows_seen",
        "domains",
        "compound_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for selected_family in selected:
            family = selected_family.family
            available = sum(1 for entry in selected_family.entries if entry in sequences)
            writer.writerow(
                {
                    "selection_index": selected_family.selection_index,
                    "family_id": family.family_id,
                    "ec_class": family.ec_class,
                    "ec_number": family.ec_number,
                    "reaction_id": family.reaction_id,
                    "selected_entries": len(selected_family.entries),
                    "available_sequence_count": available,
                    "sample_pool_size": len(family.entry_scores),
                    "rows_seen": family.rows_seen,
                    "domains": unique_join(family.domains.keys()),
                    "compound_ids": unique_join(family.compounds.keys()),
                }
            )


def write_entry_manifest(path: Path, selected: list[SelectedFamily], sequences: dict[str, SequenceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "family_id",
        "ec_number",
        "reaction_id",
        "family_rank",
        "kegg_entry",
        "organism_code",
        "gene_id",
        "sequence_status",
        "sequence_length",
        "sequence_source",
        "entry_row_count",
        "entry_domains",
        "entry_compound_ids",
        *KINETIC_COLUMN_VALUE_FIELDS.values(),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for selected_family in selected:
            family = selected_family.family
            for rank, entry in enumerate(selected_family.entries, start=1):
                org, gene = entry.split(":", 1)
                record = sequences.get(entry)
                entry_metadata = family.entry_metadata.get(entry)
                writer.writerow(
                    {
                        "family_id": family.family_id,
                        "ec_number": family.ec_number,
                        "reaction_id": family.reaction_id,
                        "family_rank": rank,
                        "kegg_entry": entry,
                        "organism_code": org,
                        "gene_id": gene,
                        "sequence_status": "ok" if record is not None and record.sequence else "missing",
                        "sequence_length": len(record.sequence) if record is not None else "",
                        "sequence_source": record.source if record is not None else "",
                        "entry_row_count": entry_metadata.row_count if entry_metadata is not None else "",
                        "entry_domains": unique_join(entry_metadata.domains.keys())
                        if entry_metadata is not None
                        else "",
                        "entry_compound_ids": unique_join(entry_metadata.compounds.keys())
                        if entry_metadata is not None
                        else "",
                        **{
                            field_name: unique_join(entry_metadata.kinetic_values[field_name].keys())
                            if entry_metadata is not None
                            else ""
                            for field_name in KINETIC_COLUMN_VALUE_FIELDS.values()
                        },
                    }
                )


def write_family_fastas(
    out_dir: Path,
    selected: list[SelectedFamily],
    sequences: dict[str, SequenceRecord],
) -> None:
    fasta_dir = out_dir / "families"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    for selected_family in selected:
        family = selected_family.family
        path = fasta_dir / f"{family.family_id}.fasta"
        write_fasta(path, selected_family.entries, sequences)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample balanced GotEnzymes EC/reaction families and fetch exact KEGG protein sequences."
    )
    parser.add_argument("--zip", default="data/input_data.zip", help="Path to input_data.zip")
    parser.add_argument("--out-dir", default="outputs/kegg_representative_5000", help="Output directory.")
    parser.add_argument("--target-sequences", type=int, default=5000, help="Total unique KEGG entries to select.")
    parser.add_argument("--seqs-per-family", type=int, default=50, help="Maximum selected entries per family.")
    parser.add_argument(
        "--min-seqs-per-family",
        type=int,
        default=50,
        help="Minimum selected entries required for a family, except possibly the final partial family.",
    )
    parser.add_argument(
        "--max-enzyme-files",
        type=int,
        default=None,
        help="Scan only the first N sorted enzyme files.",
    )
    parser.add_argument(
        "--sequence-fasta",
        action="append",
        default=[],
        help="Existing KEGG amino-acid FASTA to use before downloading. May be repeated.",
    )
    parser.add_argument(
        "--kegg-root",
        default="",
        help="Licensed local KEGG root containing genes/organisms/<org>/<org>.pep files.",
    )
    parser.add_argument(
        "--fetch-missing",
        choices=["none", "rest"],
        default="rest",
        help="How to resolve entries missing from local FASTA/local KEGG sources.",
    )
    parser.add_argument("--cache-dir", default="data/cache/kegg_aaseq", help="KEGG REST sequence cache directory.")
    parser.add_argument("--rest-batch-size", type=int, default=10, help="KEGG REST get/aaseq batch size, max 10.")
    parser.add_argument("--sleep-seconds", type=float, default=0.4, help="Delay after KEGG REST requests.")
    parser.add_argument("--timeout", type=float, default=30.0, help="KEGG REST request timeout.")
    parser.add_argument("--retries", type=int, default=3, help="KEGG REST retries per batch.")
    parser.add_argument("--max-rest-requests", type=int, default=1000, help="Safety cap for uncached REST batches.")
    parser.add_argument("--allow-large-rest-run", action="store_true", help="Allow exceeding --max-rest-requests.")
    parser.add_argument("--dry-run", action="store_true", help="Write selection manifests without fetching sequences.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target_sequences < 1:
        raise SystemExit("--target-sequences must be positive.")
    if args.seqs_per_family < 1:
        raise SystemExit("--seqs-per-family must be positive.")
    if args.min_seqs_per_family < 1 or args.min_seqs_per_family > args.seqs_per_family:
        raise SystemExit("--min-seqs-per-family must be between 1 and --seqs-per-family.")
    if args.rest_batch_size < 1 or args.rest_batch_size > 10:
        raise SystemExit("--rest-batch-size must be between 1 and 10.")

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"Archive not found: {zip_path}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep a larger pool than the final per-family size so duplicate genes across
    # related EC/reaction families do not reduce the combined sample.
    pool_size = max(args.seqs_per_family * 2, args.seqs_per_family + args.min_seqs_per_family)
    families = scan_families(
        zip_path=zip_path,
        max_entries_per_family=pool_size,
        max_enzyme_files=args.max_enzyme_files,
    )
    selected = choose_families(
        families=families,
        target_sequences=args.target_sequences,
        seqs_per_family=args.seqs_per_family,
        min_seqs_per_family=args.min_seqs_per_family,
    )
    entries = selected_entries(selected)
    expected_requests = math.ceil(len(entries) / args.rest_batch_size)
    print(
        f"Selected {len(selected):,} families and {len(entries):,} unique KEGG entries "
        f"({expected_requests:,} REST batches if uncached).",
        flush=True,
    )

    sequences: dict[str, SequenceRecord] = {}
    if not args.dry_run:
        sequences = load_sequences(args, entries)

    combined_fasta = out_dir / "combined.fasta"
    sequence_index = out_dir / "sequence_index.tsv"
    families_tsv = out_dir / "families.tsv"
    entries_tsv = out_dir / "entries.tsv"
    if not args.dry_run:
        write_fasta(combined_fasta, entries.keys(), sequences)
        write_sequence_index(sequence_index, entries.keys(), sequences)
        write_family_fastas(out_dir, selected, sequences)
    write_family_manifest(families_tsv, selected, sequences)
    write_entry_manifest(entries_tsv, selected, sequences)

    available = sum(1 for entry in entries if entry in sequences and sequences[entry].sequence)
    print(f"Wrote family manifest: {families_tsv}", flush=True)
    print(f"Wrote entry manifest: {entries_tsv}", flush=True)
    if not args.dry_run:
        print(f"Wrote combined FASTA: {combined_fasta}", flush=True)
        print(f"Wrote sequence index: {sequence_index}", flush=True)
        print(f"Wrote per-family FASTAs under: {out_dir / 'families'}", flush=True)
    print(f"Available sequences: {available:,}/{len(entries):,}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
