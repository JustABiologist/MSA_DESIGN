#!/usr/bin/env python3
"""Backfill GotEnzymes kinetic values into decoder metadata TSVs."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import zipfile
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.remap_kegg_sequences import (  # noqa: E402
    ENZYME_PREFIX,
    EXPECTED_ENZYME_COLUMNS,
    KINETIC_COLUMN_VALUE_FIELDS,
    split_ec_numbers,
    unique_join,
)


KINETIC_FIELDS = tuple(KINETIC_COLUMN_VALUE_FIELDS.values())


@dataclass(frozen=True)
class SourceKineticRow:
    ec_numbers: tuple[str, ...]
    reaction_id: str
    compound_id: str
    values: dict[str, tuple[str, ...]]


def split_metadata_values(value: str) -> set[str]:
    return {part.strip() for part in str(value).split(";") if part.strip()}


def metadata_value_set(row: dict[str, str], *field_names: str) -> set[str]:
    values: set[str] = set()
    for field_name in field_names:
        values.update(split_metadata_values(row.get(field_name, "")))
    return values


def finite_text(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return text


def parse_source_values(parts: list[str]) -> dict[str, tuple[str, ...]]:
    values: dict[str, tuple[str, ...]] = {}
    for column_index, field_name in KINETIC_COLUMN_VALUE_FIELDS.items():
        value = finite_text(parts[column_index])
        if value is not None:
            values[field_name] = (value,)
    return values


def read_metadata_files(paths: list[Path]) -> tuple[dict[Path, tuple[list[str], list[dict[str, str]]]], set[str]]:
    files: dict[Path, tuple[list[str], list[dict[str, str]]]] = {}
    entries: set[str] = set()
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        if "kegg_entry" not in fieldnames:
            raise SystemExit(f"{path} has no kegg_entry column")
        files[path] = (fieldnames, rows)
        entries.update(row["kegg_entry"] for row in rows if row.get("kegg_entry"))
    return files, entries


def enzyme_members(zip_path: Path) -> dict[str, str]:
    with zipfile.ZipFile(zip_path) as archive:
        return {
            Path(name).stem: name
            for name in archive.namelist()
            if name.startswith(ENZYME_PREFIX) and name.endswith(".txt")
        }


def load_source_rows(zip_path: Path, entries: set[str]) -> dict[str, list[SourceKineticRow]]:
    by_entry: dict[str, list[SourceKineticRow]] = defaultdict(list)
    organism_codes = {entry.split(":", 1)[0] for entry in entries if ":" in entry}
    members = enzyme_members(zip_path)
    missing = sorted(code for code in organism_codes if code not in members)
    if missing:
        print(
            f"warning: {len(missing)} organism enzyme file(s) not found in archive; "
            f"first missing: {', '.join(missing[:8])}",
            file=sys.stderr,
        )
    with zipfile.ZipFile(zip_path) as archive:
        for index, organism_code in enumerate(sorted(organism_codes), start=1):
            name = members.get(organism_code)
            if name is None:
                continue
            with archive.open(name) as handle:
                for raw_line in handle:
                    parts = raw_line.decode("utf-8", "replace").rstrip("\n\r").split("\t")
                    if len(parts) != EXPECTED_ENZYME_COLUMNS:
                        continue
                    entry = f"{parts[1]}:{parts[0]}"
                    if entry not in entries:
                        continue
                    values = parse_source_values(parts)
                    if not values:
                        continue
                    by_entry[entry].append(
                        SourceKineticRow(
                            ec_numbers=tuple(split_ec_numbers(parts[4])),
                            reaction_id=parts[3],
                            compound_id=parts[5],
                            values=values,
                        )
                    )
            if index % 500 == 0:
                print(f"scanned {index:,}/{len(organism_codes):,} organism files", flush=True)
    return by_entry


def matching_source_rows(
    metadata_row: dict[str, str],
    source_rows: list[SourceKineticRow],
    ignore_compound_filter: bool,
    fallback_entry_only: bool,
) -> list[SourceKineticRow]:
    ec_numbers = metadata_value_set(metadata_row, "ec_numbers", "ec_number")
    reaction_ids = metadata_value_set(metadata_row, "reaction_ids", "reaction_id")
    compound_ids = metadata_value_set(metadata_row, "compound_ids", "compound_id", "entry_compound_ids")
    matched: list[SourceKineticRow] = []
    for source_row in source_rows:
        if ec_numbers and not (set(source_row.ec_numbers) & ec_numbers):
            continue
        if reaction_ids and source_row.reaction_id not in reaction_ids:
            continue
        if not ignore_compound_filter and compound_ids and source_row.compound_id not in compound_ids:
            continue
        matched.append(source_row)
    if not matched and fallback_entry_only:
        return source_rows
    return matched


def aggregate_values(source_rows: list[SourceKineticRow]) -> dict[str, str]:
    aggregated: dict[str, OrderedDict[str, None]] = {field_name: OrderedDict() for field_name in KINETIC_FIELDS}
    for source_row in source_rows:
        for field_name, values in source_row.values.items():
            for value in values:
                aggregated[field_name].setdefault(value, None)
    return {field_name: unique_join(values.keys()) for field_name, values in aggregated.items()}


def with_kinetic_fields(fieldnames: list[str]) -> list[str]:
    existing = set(fieldnames)
    missing = [field_name for field_name in KINETIC_FIELDS if field_name not in existing]
    if not missing:
        return fieldnames
    output: list[str] = []
    inserted = False
    for field_name in fieldnames:
        output.append(field_name)
        if field_name == "compound_ids":
            output.extend(missing)
            inserted = True
    if not inserted:
        output.extend(missing)
    return output


def output_path_for(input_path: Path, metadata_dir: Path, out_dir: Path | None) -> Path:
    if out_dir is None:
        return input_path
    return out_dir / input_path.relative_to(metadata_dir)


def enrich_files(
    files: dict[Path, tuple[list[str], list[dict[str, str]]]],
    metadata_dir: Path,
    out_dir: Path | None,
    source_rows: dict[str, list[SourceKineticRow]],
    ignore_compound_filter: bool,
    fallback_entry_only: bool,
    dry_run: bool,
) -> dict[str, int]:
    stats = {field_name: 0 for field_name in KINETIC_FIELDS}
    stats["rows"] = 0
    stats["matched_rows"] = 0
    for input_path, (fieldnames, rows) in files.items():
        output_fieldnames = with_kinetic_fields(fieldnames)
        for row in rows:
            stats["rows"] += 1
            entry = row.get("kegg_entry", "")
            matches = matching_source_rows(
                metadata_row=row,
                source_rows=source_rows.get(entry, []),
                ignore_compound_filter=ignore_compound_filter,
                fallback_entry_only=fallback_entry_only,
            )
            if matches:
                stats["matched_rows"] += 1
            values = aggregate_values(matches)
            for field_name in KINETIC_FIELDS:
                row[field_name] = values[field_name]
                if values[field_name]:
                    stats[field_name] += 1
        if dry_run:
            continue
        output_path = output_path_for(input_path, metadata_dir, out_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fieldnames, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default="data/input_data.zip", help="Path to GotEnzymes input_data.zip.")
    parser.add_argument("--metadata-dir", required=True, help="Directory containing decoder .metadata.tsv files.")
    parser.add_argument("--metadata-glob", default="*.metadata.tsv", help="Glob inside --metadata-dir.")
    parser.add_argument("--out-dir", default="", help="Optional output directory. Omit with --in-place.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite metadata TSVs in place.")
    parser.add_argument(
        "--ignore-compound-filter",
        action="store_true",
        help="Match by KEGG entry, EC, and reaction only. Default also respects compound_ids when present.",
    )
    parser.add_argument(
        "--fallback-entry-only",
        action="store_true",
        help="If EC/reaction/compound matching finds nothing, aggregate all kinetic rows for the KEGG entry.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report coverage without writing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata_dir = Path(args.metadata_dir)
    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"Archive not found: {zip_path}")
    if not metadata_dir.exists():
        raise SystemExit(f"Metadata directory not found: {metadata_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else None
    if not args.dry_run and not args.in_place and out_dir is None:
        raise SystemExit("Pass --in-place, --out-dir, or --dry-run.")
    paths = sorted(metadata_dir.glob(args.metadata_glob))
    if not paths:
        raise SystemExit(f"No metadata files matched {metadata_dir / args.metadata_glob}")

    files, entries = read_metadata_files(paths)
    source_rows = load_source_rows(zip_path, entries)
    stats = enrich_files(
        files=files,
        metadata_dir=metadata_dir,
        out_dir=out_dir,
        source_rows=source_rows,
        ignore_compound_filter=args.ignore_compound_filter,
        fallback_entry_only=args.fallback_entry_only,
        dry_run=args.dry_run,
    )
    print(
        f"metadata_files={len(paths)} rows={stats['rows']} matched_rows={stats['matched_rows']} "
        + " ".join(f"{field}={stats[field]}" for field in KINETIC_FIELDS),
        flush=True,
    )
    if args.dry_run:
        print("dry_run=true; no files written", flush=True)
    elif out_dir is not None:
        print(f"wrote enriched metadata under {out_dir}", flush=True)
    else:
        print("updated metadata in place", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
