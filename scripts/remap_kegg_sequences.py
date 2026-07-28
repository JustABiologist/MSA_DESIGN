#!/usr/bin/env python3
"""Map GotEnzymes rows to their source KEGG protein sequences."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


ENZYME_PREFIX = "input_data/enzymes/"
EXPECTED_ENZYME_COLUMNS = 11
NUMERIC_COLUMN_INDEXES = range(6, 11)
KINETIC_COLUMN_VALUE_FIELDS = {
    6: "kcat_1_per_s_values",
    7: "km_mM_values",
    8: "kcat_over_km_1_per_mM_s_values",
    9: "topt_C_values",
    10: "tm_C_values",
}
ROW_FIELD_NAMES = [
    "gene_id",
    "organism_code",
    "domain",
    "reaction_id",
    "ec_numbers",
    "compound_id",
    "kcat_1_per_s",
    "km_mM",
    "kcat_over_km_1_per_mM_s",
    "topt_C",
    "tm_C",
]
FASTA_WRAP = 80
KEGG_GET_URL = "https://rest.kegg.jp/get/{entries}/aaseq"
USER_AGENT = "MSA_DESIGN KEGG sequence remapper"


@dataclass
class SequenceRecord:
    entry: str
    sequence: str
    header: str
    source: str
    cache_hit: bool = False


@dataclass
class GeneSummary:
    organism_code: str
    gene_id: str
    row_count: int = 0
    ec_numbers: OrderedDict[str, None] = field(default_factory=OrderedDict)
    reaction_ids: OrderedDict[str, None] = field(default_factory=OrderedDict)
    compound_ids: OrderedDict[str, None] = field(default_factory=OrderedDict)
    numeric_values: dict[int, OrderedDict[str, None]] = field(
        default_factory=lambda: {idx: OrderedDict() for idx in NUMERIC_COLUMN_INDEXES}
    )


def split_ec_numbers(raw_ecs: str) -> list[str]:
    return [part.strip() for part in raw_ecs.split(";") if part.strip()]


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug or "value"


def unique_join(values: Iterable[str], max_values: int = 25) -> str:
    seen: OrderedDict[str, None] = OrderedDict()
    for value in values:
        if value and value not in seen:
            seen[value] = None
    items = list(seen.keys())
    if len(items) <= max_values:
        return ";".join(items)
    return ";".join(items[:max_values]) + f";...(+{len(items) - max_values})"


def write_wrapped(handle: Any, sequence: str) -> None:
    for start in range(0, len(sequence), FASTA_WRAP):
        handle.write(sequence[start : start + FASTA_WRAP] + "\n")


def iter_enzyme_rows(
    zip_path: Path,
    ec_filters: set[str],
    organism_filters: set[str],
    limit: int | None,
    max_enzyme_files: int | None,
) -> Iterator[list[str]]:
    seen_entries: set[str] = set()
    with zipfile.ZipFile(zip_path) as zf:
        available = {
            Path(name).stem: name
            for name in zf.namelist()
            if name.startswith(ENZYME_PREFIX) and name.endswith(".txt")
        }
        if organism_filters:
            missing = sorted(code for code in organism_filters if code not in available)
            if missing:
                raise SystemExit(f"Organism enzyme file(s) not found in zip: {', '.join(missing)}")
            enzyme_names = [available[code] for code in sorted(organism_filters)]
        else:
            enzyme_names = sorted(available.values())
        if max_enzyme_files is not None:
            enzyme_names = enzyme_names[:max_enzyme_files]

        for name in enzyme_names:
            with zf.open(name) as handle:
                for raw_line in handle:
                    line = raw_line.decode("utf-8", "replace").rstrip("\n\r")
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) != EXPECTED_ENZYME_COLUMNS:
                        continue
                    organism_code = parts[1]
                    gene_id = parts[0]
                    if organism_filters and organism_code not in organism_filters:
                        continue
                    if ec_filters and not (set(split_ec_numbers(parts[4])) & ec_filters):
                        continue
                    if limit is not None:
                        entry = f"{organism_code}:{gene_id}"
                        if entry not in seen_entries:
                            if len(seen_entries) >= limit:
                                return
                            seen_entries.add(entry)
                    yield parts


def collect_selection(
    zip_path: Path,
    ec_filters: set[str],
    organism_filters: set[str],
    limit: int | None,
    max_enzyme_files: int | None,
    collect_summaries: bool,
) -> tuple[OrderedDict[str, None], OrderedDict[str, GeneSummary]]:
    entries: OrderedDict[str, None] = OrderedDict()
    summaries: OrderedDict[str, GeneSummary] = OrderedDict()
    for row in iter_enzyme_rows(
        zip_path=zip_path,
        ec_filters=ec_filters,
        organism_filters=organism_filters,
        limit=limit,
        max_enzyme_files=max_enzyme_files,
    ):
        organism_code = row[1]
        gene_id = row[0]
        entry = f"{organism_code}:{gene_id}"
        entries.setdefault(entry, None)
        if not collect_summaries:
            continue
        summary = summaries.get(entry)
        if summary is None:
            summary = GeneSummary(organism_code=organism_code, gene_id=gene_id)
            summaries[entry] = summary
        summary.row_count += 1
        for ec_number in split_ec_numbers(row[4]):
            summary.ec_numbers.setdefault(ec_number, None)
        summary.reaction_ids.setdefault(row[3], None)
        summary.compound_ids.setdefault(row[5], None)
        for idx in NUMERIC_COLUMN_INDEXES:
            summary.numeric_values[idx].setdefault(row[idx], None)
    return entries, summaries


def parse_fasta_records(handle: Iterable[str], source: str) -> Iterator[SequenceRecord]:
    header = ""
    parts: list[str] = []
    for raw_line in handle:
        line = raw_line.rstrip("\n\r")
        if not line:
            continue
        if line.startswith(">"):
            if header:
                entry = header[1:].split(None, 1)[0]
                yield SequenceRecord(entry=entry, sequence="".join(parts), header=header[1:], source=source)
            header = line
            parts = []
        else:
            parts.append("".join(line.split()).upper())
    if header:
        entry = header[1:].split(None, 1)[0]
        yield SequenceRecord(entry=entry, sequence="".join(parts), header=header[1:], source=source)


def load_fasta_sequences(paths: list[Path], wanted: set[str]) -> dict[str, SequenceRecord]:
    records: dict[str, SequenceRecord] = {}
    for path in paths:
        if path.suffix == ".gz":
            opener = lambda: gzip.open(path, "rt", encoding="utf-8", errors="replace")
        else:
            opener = lambda: path.open("r", encoding="utf-8", errors="replace")
        with opener() as handle:
            for record in parse_fasta_records(handle, source=str(path)):
                if record.entry in wanted and record.entry not in records:
                    records[record.entry] = record
    return records


def possible_kegg_pep_paths(kegg_root: Path, organism_code: str) -> list[Path]:
    roots = [kegg_root, kegg_root / "kegg"]
    suffixes = [".pep", ".pep.gz", ".faa", ".faa.gz", ".fasta", ".fasta.gz", ".fa", ".fa.gz"]
    paths: list[Path] = []
    for root in roots:
        base_dirs = [
            root / "genes" / "organisms" / organism_code,
            root / "genes" / "organisms",
        ]
        for base_dir in base_dirs:
            for suffix in suffixes:
                paths.append(base_dir / f"{organism_code}{suffix}")
        tar_path = root / "genes" / "organisms" / organism_code / f"{organism_code}.tar.gz"
        paths.append(tar_path)
    return paths


def iter_local_kegg_org_records(kegg_root: Path, organism_code: str) -> Iterator[SequenceRecord]:
    for path in possible_kegg_pep_paths(kegg_root, organism_code):
        if not path.exists():
            continue
        if path.suffixes[-2:] == [".tar", ".gz"]:
            with tarfile.open(path, "r:gz") as tar:
                for member in tar.getmembers():
                    if not member.isfile() or not member.name.endswith((".pep", ".faa", ".fa", ".fasta")):
                        continue
                    extracted = tar.extractfile(member)
                    if extracted is None:
                        continue
                    text = io.TextIOWrapper(extracted, encoding="utf-8", errors="replace")
                    yield from parse_fasta_records(text, source=f"{path}:{member.name}")
            return
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
                yield from parse_fasta_records(handle, source=str(path))
        else:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                yield from parse_fasta_records(handle, source=str(path))
        return


def load_local_kegg_sequences(kegg_root: Path, entries: OrderedDict[str, None]) -> dict[str, SequenceRecord]:
    by_org: OrderedDict[str, set[str]] = OrderedDict()
    for entry in entries:
        org, _ = entry.split(":", 1)
        by_org.setdefault(org, set()).add(entry)

    records: dict[str, SequenceRecord] = {}
    for org, wanted in by_org.items():
        for record in iter_local_kegg_org_records(kegg_root, org):
            if record.entry in wanted and record.entry not in records:
                records[record.entry] = record
                if len(records) % 100000 == 0:
                    print(f"loaded {len(records):,} local KEGG sequences", file=sys.stderr, flush=True)
            if wanted <= records.keys():
                break
    return records


def cache_path(cache_dir: Path, entry: str) -> Path:
    org, gene = entry.split(":", 1)
    digest = hashlib.sha1(entry.encode("utf-8")).hexdigest()[:12]
    return cache_dir / safe_slug(org) / f"{safe_slug(gene)}__{digest}.fasta"


def fetch_kegg_batch(entries: list[str], timeout: float, retries: int, sleep_seconds: float) -> str:
    encoded_entries = "+".join(urllib.parse.quote(entry, safe=":") for entry in entries)
    url = KEGG_GET_URL.format(entries=encoded_entries)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8", "replace")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            return payload
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            last_error = RuntimeError(f"HTTP {exc.code}: {body}")
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries:
                break
            time.sleep(max(sleep_seconds * attempt, 1.0))
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(max(sleep_seconds * attempt, 1.0))
    raise RuntimeError(str(last_error))


def read_cached_sequence(path: Path) -> SequenceRecord | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        records = list(parse_fasta_records(handle, source=str(path)))
    if not records:
        return None
    record = records[0]
    record.cache_hit = True
    return record


def write_cache_record(path: Path, record: SequenceRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f">{record.header}\n")
        write_wrapped(handle, record.sequence)


def fetch_missing_from_kegg_rest(
    entries: list[str],
    cache_dir: Path,
    batch_size: int,
    timeout: float,
    retries: int,
    sleep_seconds: float,
    max_rest_requests: int | None,
    allow_large_rest_run: bool,
) -> dict[str, SequenceRecord]:
    if batch_size < 1 or batch_size > 10:
        raise SystemExit("KEGG REST get/aaseq supports --rest-batch-size from 1 to 10.")
    records: dict[str, SequenceRecord] = {}
    uncached: list[str] = []
    for entry in entries:
        cached = read_cached_sequence(cache_path(cache_dir, entry))
        if cached is None:
            uncached.append(entry)
        else:
            records[entry] = cached

    request_count = (len(uncached) + batch_size - 1) // batch_size
    if max_rest_requests is not None and request_count > max_rest_requests and not allow_large_rest_run:
        raise SystemExit(
            f"Refusing {request_count:,} KEGG REST requests for {len(uncached):,} uncached entries. "
            "Use a licensed local KEGG dump with --kegg-root, lower the selection, raise "
            "--max-rest-requests, or pass --allow-large-rest-run intentionally."
        )

    for start in range(0, len(uncached), batch_size):
        batch = uncached[start : start + batch_size]
        payload = fetch_kegg_batch(batch, timeout=timeout, retries=retries, sleep_seconds=sleep_seconds)
        found = {record.entry: record for record in parse_fasta_records(payload.splitlines(), source="kegg_rest")}
        for entry, record in found.items():
            if entry in batch:
                write_cache_record(cache_path(cache_dir, entry), record)
                records[entry] = record
        if (start // batch_size + 1) % 100 == 0:
            print(
                f"fetched {start + len(batch):,}/{len(uncached):,} uncached KEGG entries",
                file=sys.stderr,
                flush=True,
            )
    return records


def write_fasta(path: Path, entries: Iterable[str], sequences: dict[str, SequenceRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            record = sequences.get(entry)
            if record is None or not record.sequence:
                continue
            header_tail = record.header.split(None, 1)[1] if " " in record.header else ""
            handle.write(f">{entry}")
            if header_tail:
                handle.write(f" {header_tail}")
            handle.write("\n")
            write_wrapped(handle, record.sequence)
            count += 1
    return count


def write_sequence_index(path: Path, entries: Iterable[str], sequences: dict[str, SequenceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kegg_entry",
                "organism_code",
                "gene_id",
                "sequence_status",
                "sequence_length",
                "sequence_source",
                "sequence_header",
                "cache_hit",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for entry in entries:
            org, gene = entry.split(":", 1)
            record = sequences.get(entry)
            writer.writerow(
                {
                    "kegg_entry": entry,
                    "organism_code": org,
                    "gene_id": gene,
                    "sequence_status": "ok" if record is not None and record.sequence else "missing",
                    "sequence_length": len(record.sequence) if record is not None else "",
                    "sequence_source": record.source if record is not None else "",
                    "sequence_header": record.header if record is not None else "",
                    "cache_hit": str(bool(record.cache_hit)) if record is not None else "",
                }
            )


def write_metadata(
    path: Path,
    summaries: OrderedDict[str, GeneSummary],
    sequences: dict[str, SequenceRecord],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "organism_code",
        "gene_id",
        "kegg_entry",
        "selected_row_count",
        "ec_numbers",
        "reaction_ids",
        "compound_ids",
        "kcat_1_per_s_values",
        "km_mM_values",
        "kcat_over_km_1_per_mM_s_values",
        "topt_C_values",
        "tm_C_values",
        "sequence_status",
        "sequence_length",
        "sequence_source",
        "sequence_header",
        "cache_hit",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for entry, summary in summaries.items():
            record = sequences.get(entry)
            row = {
                "organism_code": summary.organism_code,
                "gene_id": summary.gene_id,
                "kegg_entry": entry,
                "selected_row_count": str(summary.row_count),
                "ec_numbers": unique_join(summary.ec_numbers.keys()),
                "reaction_ids": unique_join(summary.reaction_ids.keys()),
                "compound_ids": unique_join(summary.compound_ids.keys()),
                "sequence_status": "ok" if record is not None and record.sequence else "missing",
                "sequence_length": len(record.sequence) if record is not None else "",
                "sequence_source": record.source if record is not None else "",
                "sequence_header": record.header if record is not None else "",
                "cache_hit": str(bool(record.cache_hit)) if record is not None else "",
            }
            for idx, field_name in KINETIC_COLUMN_VALUE_FIELDS.items():
                row[field_name] = unique_join(summary.numeric_values[idx].keys())
            writer.writerow(row)


def write_row_map(
    path: Path,
    zip_path: Path,
    ec_filters: set[str],
    organism_filters: set[str],
    limit: int | None,
    max_enzyme_files: int | None,
    sequences: dict[str, SequenceRecord],
    include_sequence: bool,
) -> int:
    fields = [
        "kegg_entry",
        *ROW_FIELD_NAMES,
        "sequence_status",
        "sequence_id",
        "sequence_length",
        "sequence_source",
    ]
    if include_sequence:
        fields.append("sequence")
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in iter_enzyme_rows(
            zip_path=zip_path,
            ec_filters=ec_filters,
            organism_filters=organism_filters,
            limit=limit,
            max_enzyme_files=max_enzyme_files,
        ):
            entry = f"{row[1]}:{row[0]}"
            record = sequences.get(entry)
            out = {
                "kegg_entry": entry,
                **{field: value for field, value in zip(ROW_FIELD_NAMES, row)},
                "sequence_status": "ok" if record is not None and record.sequence else "missing",
                "sequence_id": entry if record is not None and record.sequence else "",
                "sequence_length": len(record.sequence) if record is not None else "",
                "sequence_source": record.source if record is not None else "",
            }
            if include_sequence:
                out["sequence"] = record.sequence if record is not None else ""
            writer.writerow(out)
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select GotEnzymes rows, resolve their source KEGG gene protein sequences, "
            "and write normalized sequence/remap artifacts."
        )
    )
    parser.add_argument("--zip", default="data/input_data.zip", help="Path to input_data.zip")
    parser.add_argument("--ec", action="append", default=[], help="Exact EC number to select. May be repeated.")
    parser.add_argument(
        "--organism-code",
        action="append",
        default=[],
        help="KEGG organism code to select. May be repeated.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Allow selecting the whole GotEnzymes archive when no --ec/--organism-code filter is given.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum unique KEGG gene entries to select.")
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
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Polite delay after KEGG REST requests.")
    parser.add_argument("--timeout", type=float, default=30.0, help="KEGG REST request timeout.")
    parser.add_argument("--retries", type=int, default=3, help="KEGG REST retries per batch.")
    parser.add_argument(
        "--max-rest-requests",
        type=int,
        default=1000,
        help="Safety cap for uncached KEGG REST batch requests.",
    )
    parser.add_argument(
        "--allow-large-rest-run",
        action="store_true",
        help="Allow exceeding --max-rest-requests. Prefer --kegg-root for full archive remaps.",
    )
    parser.add_argument("--out-fasta", required=True, help="Output FASTA path for unique KEGG gene sequences.")
    parser.add_argument("--out-index", required=True, help="Output TSV sequence index path.")
    parser.add_argument("--out-metadata", default="", help="Optional per-gene metadata summary TSV path.")
    parser.add_argument("--out-row-map", default="", help="Optional per-property-row remap TSV path.")
    parser.add_argument(
        "--include-sequence-in-row-map",
        action="store_true",
        help="Duplicate protein sequences into --out-row-map. Usually avoid this for large runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"Archive not found: {zip_path}")
    ec_filters = set(args.ec)
    organism_filters = set(args.organism_code)
    if not ec_filters and not organism_filters and not args.all:
        raise SystemExit("Provide --ec/--organism-code or pass --all intentionally.")
    if args.include_sequence_in_row_map and not args.out_row_map:
        raise SystemExit("--include-sequence-in-row-map requires --out-row-map.")

    collect_summaries = bool(args.out_metadata)
    entries, summaries = collect_selection(
        zip_path=zip_path,
        ec_filters=ec_filters,
        organism_filters=organism_filters,
        limit=args.limit,
        max_enzyme_files=args.max_enzyme_files,
        collect_summaries=collect_summaries,
    )
    if not entries:
        raise SystemExit("No KEGG gene entries matched the requested selection.")
    print(f"Selected {len(entries):,} unique KEGG gene entries.", flush=True)

    sequences: dict[str, SequenceRecord] = {}
    wanted = set(entries.keys())
    fasta_paths = [Path(path) for path in args.sequence_fasta]
    if fasta_paths:
        sequences.update(load_fasta_sequences(fasta_paths, wanted=wanted))
        print(f"Loaded {len(sequences):,} sequences from existing FASTA files.", flush=True)

    if args.kegg_root:
        before = len(sequences)
        local_records = load_local_kegg_sequences(Path(args.kegg_root), entries)
        sequences.update({entry: record for entry, record in local_records.items() if entry not in sequences})
        print(f"Loaded {len(sequences) - before:,} additional sequences from local KEGG.", flush=True)

    missing = [entry for entry in entries if entry not in sequences]
    if missing and args.fetch_missing == "rest":
        before = len(sequences)
        rest_records = fetch_missing_from_kegg_rest(
            entries=missing,
            cache_dir=Path(args.cache_dir),
            batch_size=args.rest_batch_size,
            timeout=args.timeout,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
            max_rest_requests=args.max_rest_requests,
            allow_large_rest_run=args.allow_large_rest_run,
        )
        sequences.update({entry: record for entry, record in rest_records.items() if entry not in sequences})
        print(f"Fetched/loaded {len(sequences) - before:,} additional sequences from KEGG REST.", flush=True)

    fasta_count = write_fasta(Path(args.out_fasta), entries.keys(), sequences)
    write_sequence_index(Path(args.out_index), entries.keys(), sequences)
    if args.out_metadata:
        write_metadata(Path(args.out_metadata), summaries, sequences)
    row_count = 0
    if args.out_row_map:
        row_count = write_row_map(
            path=Path(args.out_row_map),
            zip_path=zip_path,
            ec_filters=ec_filters,
            organism_filters=organism_filters,
            limit=args.limit,
            max_enzyme_files=args.max_enzyme_files,
            sequences=sequences,
            include_sequence=args.include_sequence_in_row_map,
        )

    missing_count = len(entries) - fasta_count
    print(
        f"Wrote {fasta_count:,} FASTA records to {args.out_fasta}; "
        f"{missing_count:,} selected entries are missing sequences.",
        flush=True,
    )
    print(f"Wrote sequence index to {args.out_index}.", flush=True)
    if args.out_metadata:
        print(f"Wrote per-gene metadata to {args.out_metadata}.", flush=True)
    if args.out_row_map:
        print(f"Wrote {row_count:,} property remap rows to {args.out_row_map}.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
