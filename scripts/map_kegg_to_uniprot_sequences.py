#!/usr/bin/env python3
"""One-shot KEGG gene ID to UniProt sequence export."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Iterator


ENZYME_PREFIX = "input_data/enzymes/"
EXPECTED_ENZYME_COLUMNS = 11
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
UNIPROT_MAPPING_RUN_URL = "https://rest.uniprot.org/idmapping/run"
UNIPROT_MAPPING_STATUS_URL = "https://rest.uniprot.org/idmapping/status/{job_id}"
UNIPROT_MAPPING_RESULTS_URL = "https://rest.uniprot.org/idmapping/results/{job_id}"
UNIPROT_SPROT_FASTA_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/complete/uniprot_sprot.fasta.gz"
)
UNIPROT_TREMBL_FASTA_URL = (
    "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
    "knowledgebase/complete/uniprot_trembl.fasta.gz"
)
USER_AGENT = "MSA_DESIGN standalone KEGG-to-UniProt sequence mapper"


def split_ec_numbers(raw_ecs: str) -> list[str]:
    return [part.strip() for part in raw_ecs.split(";") if part.strip()]


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


def open_output_text(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def looks_like_kegg_entry(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_]+:[A-Za-z0-9_.-]+", value.strip()))


def iter_kegg_id_file(path: Path, input_column: str) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        delimiter = "\t" if "\t" in sample else ","
        reader = csv.reader(handle, delimiter=delimiter)
        first = next(reader, None)
        if first is None:
            return
        header_index: int | None = None
        if input_column:
            stripped_first = [cell.strip() for cell in first]
            if input_column in stripped_first:
                header_index = stripped_first.index(input_column)
            elif input_column.isdigit():
                header_index = int(input_column)
        if header_index is None:
            candidates = [cell.strip() for cell in first]
            for candidate in candidates:
                if looks_like_kegg_entry(candidate):
                    yield candidate
                    break
            rows = reader
            header_index = 0
        else:
            rows = reader
        for row in rows:
            if not row:
                continue
            if header_index >= len(row):
                continue
            value = row[header_index].strip()
            if value and not value.startswith("#"):
                yield value


def iter_gotenzymes_kegg_entries(
    zip_path: Path,
    ec_filters: set[str],
    organism_filters: set[str],
    max_enzyme_files: int | None,
) -> Iterator[str]:
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

        for file_index, name in enumerate(enzyme_names, start=1):
            with zf.open(name) as handle:
                for raw_line in handle:
                    line = raw_line.decode("utf-8", "replace").rstrip("\n\r")
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) != EXPECTED_ENZYME_COLUMNS:
                        continue
                    if ec_filters and not (set(split_ec_numbers(parts[4])) & ec_filters):
                        continue
                    yield f"{parts[1]}:{parts[0]}"
            if file_index % 1000 == 0:
                print(f"scanned {file_index:,}/{len(enzyme_names):,} enzyme files", flush=True)


def iter_gotenzymes_rows(
    zip_path: Path,
    ec_filters: set[str],
    organism_filters: set[str],
    max_enzyme_files: int | None,
) -> Iterator[list[str]]:
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
                    if ec_filters and not (set(split_ec_numbers(parts[4])) & ec_filters):
                        continue
                    row_count += 1
                    yield parts
            if file_index % 1000 == 0:
                print(
                    f"wrote through {file_index:,}/{len(enzyme_names):,} enzyme files; "
                    f"rows={row_count:,}",
                    flush=True,
                )


def collect_kegg_entries(args: argparse.Namespace) -> OrderedDict[str, None]:
    entries: OrderedDict[str, None] = OrderedDict()
    for entry in args.kegg_id:
        entries.setdefault(entry, None)
    for path_text in args.kegg_id_file:
        for entry in iter_kegg_id_file(Path(path_text), input_column=args.input_column):
            entries.setdefault(entry, None)

    if args.zip:
        ec_filters = set(args.ec)
        organism_filters = set(args.organism_code)
        if not ec_filters and not organism_filters and not args.all:
            raise SystemExit("When using --zip, provide --ec/--organism-code or pass --all intentionally.")
        zip_path = Path(args.zip)
        if not zip_path.exists():
            raise SystemExit(f"Archive not found: {zip_path}")
        for entry in iter_gotenzymes_kegg_entries(
            zip_path=zip_path,
            ec_filters=ec_filters,
            organism_filters=organism_filters,
            max_enzyme_files=args.max_enzyme_files,
        ):
            entries.setdefault(entry, None)
            if args.limit is not None and len(entries) >= args.limit:
                break
    elif args.limit is not None:
        limited = OrderedDict()
        for entry in entries:
            limited[entry] = None
            if len(limited) >= args.limit:
                break
        entries = limited

    return entries


def read_url_text(
    request: urllib.request.Request | str,
    timeout: float,
    retries: int,
    sleep_seconds: float,
) -> tuple[str, dict[str, str]]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8", "replace")
                headers = dict(response.headers.items())
            return payload, headers
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


def remote_content_length(url: str, timeout: float) -> int | None:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_length = response.headers.get("Content-Length")
    except (urllib.error.URLError, TimeoutError):
        return None
    if raw_length is None:
        return None
    try:
        return int(raw_length)
    except ValueError:
        return None


def download_file(url: str, path: Path, timeout: float) -> Path:
    expected_size = remote_content_length(url, timeout=timeout)
    if path.exists() and expected_size is not None and path.stat().st_size == expected_size:
        print(f"Using existing UniProt download: {path}", flush=True)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    print(f"Downloading {url} -> {path}", flush=True)
    with urllib.request.urlopen(request, timeout=timeout) as response, temp_path.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        last_reported_gib = -1
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            reported_gib = downloaded // (1024**3)
            if reported_gib != last_reported_gib:
                last_reported_gib = reported_gib
                if total:
                    percent = downloaded * 100 / total
                    print(
                        f"downloaded {downloaded / (1024**3):.1f}/{total / (1024**3):.1f} GiB "
                        f"({percent:.1f}%)",
                        flush=True,
                    )
                else:
                    print(f"downloaded {downloaded / (1024**3):.1f} GiB", flush=True)
    temp_path.replace(path)
    return path


def resolve_uniprot_fasta_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(path) for path in args.uniprot_fasta]
    if args.download_uniprot_dir:
        download_dir = Path(args.download_uniprot_dir)
        paths.extend(
            [
                download_file(
                    UNIPROT_SPROT_FASTA_URL,
                    download_dir / "uniprot_sprot.fasta.gz",
                    timeout=args.download_timeout,
                ),
                download_file(
                    UNIPROT_TREMBL_FASTA_URL,
                    download_dir / "uniprot_trembl.fasta.gz",
                    timeout=args.download_timeout,
                ),
            ]
        )
    for path in paths:
        if not path.exists():
            raise SystemExit(f"UniProt FASTA database not found: {path}")
    return paths


def parse_next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.strip()
        if 'rel="next"' not in section:
            continue
        match = re.match(r"<([^>]+)>", section)
        if match:
            return match.group(1)
    return None


def add_mapping_value(mapping: dict[str, OrderedDict[str, None]], kegg_entry: str, accession: str) -> None:
    accessions = mapping.setdefault(kegg_entry, OrderedDict())
    if accession:
        accessions.setdefault(accession, None)


def submit_mapping_job(entries: list[str], timeout: float, retries: int, sleep_seconds: float) -> str:
    payload = urllib.parse.urlencode(
        {
            "from": "KEGG",
            "to": "UniProtKB",
            "ids": ",".join(entries),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        UNIPROT_MAPPING_RUN_URL,
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": USER_AGENT,
        },
    )
    text, _headers = read_url_text(request, timeout=timeout, retries=retries, sleep_seconds=sleep_seconds)
    data = json.loads(text)
    job_id = data.get("jobId")
    if not job_id:
        raise RuntimeError(f"UniProt ID mapping did not return a jobId: {text[:500]}")
    return str(job_id)


def wait_for_mapping_job(
    job_id: str,
    timeout: float,
    retries: int,
    sleep_seconds: float,
    poll_seconds: float,
    max_polls: int,
) -> None:
    url = UNIPROT_MAPPING_STATUS_URL.format(job_id=urllib.parse.quote(job_id))
    for _poll in range(max_polls):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        text, _headers = read_url_text(request, timeout=timeout, retries=retries, sleep_seconds=sleep_seconds)
        data = json.loads(text)
        status = data.get("jobStatus")
        if status in (None, "FINISHED"):
            return
        if status in {"FAILED", "ERROR"}:
            raise RuntimeError(f"UniProt ID mapping job {job_id} failed: {text[:500]}")
        time.sleep(poll_seconds)
    raise RuntimeError(f"Timed out waiting for UniProt ID mapping job {job_id}.")


def fetch_mapping_results(
    job_id: str,
    timeout: float,
    retries: int,
    sleep_seconds: float,
) -> dict[str, OrderedDict[str, None]]:
    mapping: dict[str, OrderedDict[str, None]] = {}
    url = UNIPROT_MAPPING_RESULTS_URL.format(job_id=urllib.parse.quote(job_id)) + "?format=tsv&size=500"
    while url:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        text, headers = read_url_text(request, timeout=timeout, retries=retries, sleep_seconds=sleep_seconds)
        reader = csv.DictReader(text.splitlines(), delimiter="\t")
        for row in reader:
            kegg_entry = row.get("From", "").strip()
            accession = row.get("To", "").strip()
            if kegg_entry:
                add_mapping_value(mapping, kegg_entry, accession)
        url = parse_next_link(headers.get("Link") or headers.get("link"))
    return mapping


def map_entries(args: argparse.Namespace, entries: OrderedDict[str, None]) -> dict[str, OrderedDict[str, None]]:
    selected = list(entries.keys())
    mapping: dict[str, OrderedDict[str, None]] = {}
    job_count = math.ceil(len(selected) / args.batch_size) if selected else 0
    if args.max_mapping_jobs is not None and job_count > args.max_mapping_jobs and not args.allow_large_mapping_run:
        raise SystemExit(
            f"Refusing {job_count:,} UniProt ID mapping jobs for {len(selected):,} KEGG IDs. "
            "Raise --max-mapping-jobs or pass --allow-large-mapping-run intentionally."
        )

    api_log_handle = open_output_text(Path(args.api_log)) if args.api_log else None
    api_log_writer = None
    if api_log_handle:
        api_log_writer = csv.DictWriter(
            api_log_handle,
            fieldnames=[
                "batch_index",
                "start_offset",
                "submitted_ids",
                "job_id",
                "status",
                "mapped_ids",
                "mapping_rows",
                "error",
                "finished_at",
            ],
            delimiter="\t",
        )
        api_log_writer.writeheader()

    try:
        for batch_index, start in enumerate(range(0, len(selected), args.batch_size), start=1):
            batch = selected[start : start + args.batch_size]
            job_id = ""
            batch_mapping: dict[str, OrderedDict[str, None]] = {}
            try:
                job_id = submit_mapping_job(
                    entries=batch,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep_seconds=args.sleep_seconds,
                )
                wait_for_mapping_job(
                    job_id=job_id,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep_seconds=args.sleep_seconds,
                    poll_seconds=args.poll_seconds,
                    max_polls=args.max_polls,
                )
                batch_mapping = fetch_mapping_results(
                    job_id=job_id,
                    timeout=args.timeout,
                    retries=args.retries,
                    sleep_seconds=args.sleep_seconds,
                )
            except Exception as exc:
                if api_log_writer:
                    api_log_writer.writerow(
                        {
                            "batch_index": batch_index,
                            "start_offset": start,
                            "submitted_ids": len(batch),
                            "job_id": job_id,
                            "status": "error",
                            "mapped_ids": 0,
                            "mapping_rows": 0,
                            "error": str(exc)[:1000],
                            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                    )
                    api_log_handle.flush()
                raise

            mapped_ids = 0
            mapping_rows = 0
            for entry in batch:
                mapping.setdefault(entry, OrderedDict())
                accessions = batch_mapping.get(entry, OrderedDict())
                if accessions:
                    mapped_ids += 1
                for accession in accessions.keys():
                    mapping[entry].setdefault(accession, None)
                    mapping_rows += 1
            if api_log_writer:
                api_log_writer.writerow(
                    {
                        "batch_index": batch_index,
                        "start_offset": start,
                        "submitted_ids": len(batch),
                        "job_id": job_id,
                        "status": "ok",
                        "mapped_ids": mapped_ids,
                        "mapping_rows": mapping_rows,
                        "error": "",
                        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                )
                api_log_handle.flush()
            print(
                f"mapped {start + len(batch):,}/{len(selected):,} KEGG IDs through UniProt",
                flush=True,
            )
    finally:
        if api_log_handle:
            api_log_handle.close()

    return {entry: mapping.get(entry, OrderedDict()) for entry in entries}


def open_text_maybe_gzip(path: Path) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def uniprot_accession_from_header(header: str) -> str:
    token = header.split(None, 1)[0]
    parts = token.split("|")
    if len(parts) >= 2 and parts[0] in {"sp", "tr"}:
        return parts[1]
    return token


def iter_fasta_records(handle: Iterable[str]) -> Iterator[tuple[str, str]]:
    header = ""
    chunks: list[str] = []
    for raw_line in handle:
        line = raw_line.rstrip("\n\r")
        if not line:
            continue
        if line.startswith(">"):
            if header:
                yield header[1:], "".join(chunks)
            header = line
            chunks = []
        else:
            chunks.append("".join(line.split()).upper())
    if header:
        yield header[1:], "".join(chunks)


def load_uniprot_sequences(
    fasta_paths: list[Path],
    mapping: dict[str, OrderedDict[str, None]],
) -> dict[str, tuple[str, str, str]]:
    wanted: OrderedDict[str, None] = OrderedDict()
    for accessions in mapping.values():
        for accession in accessions:
            wanted.setdefault(accession, None)
    wanted_set = set(wanted.keys())
    records: dict[str, tuple[str, str, str]] = {}
    if not wanted_set:
        return records

    for path in fasta_paths:
        print(f"scanning UniProt FASTA: {path}", flush=True)
        with open_text_maybe_gzip(path) as handle:
            for header, sequence in iter_fasta_records(handle):
                accession = uniprot_accession_from_header(header)
                if accession in wanted_set and accession not in records:
                    records[accession] = (sequence, header, str(path))
        print(f"found {len(records):,}/{len(wanted_set):,} mapped UniProt sequences", flush=True)
        if len(records) == len(wanted_set):
            break
    return records


def write_mapping(path: Path, entries: OrderedDict[str, None], mapping: dict[str, OrderedDict[str, None]]) -> None:
    with open_output_text(path) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["kegg_entry", "uniprot_accession"])
        for entry in entries:
            accessions = list(mapping.get(entry, OrderedDict()).keys())
            if not accessions:
                writer.writerow([entry, ""])
                continue
            for accession in accessions:
                writer.writerow([entry, accession])


def first_available_accession(
    accessions: Iterable[str],
    sequences_by_accession: dict[str, tuple[str, str, str]],
) -> str:
    for accession in accessions:
        if accession in sequences_by_accession:
            return accession
    return ""


def sequence_status_for_entry(
    entry: str,
    mapping: dict[str, OrderedDict[str, None]],
    sequences_by_accession: dict[str, tuple[str, str, str]],
) -> tuple[str, str, list[str], str, str, str, str]:
    accessions = list(mapping.get(entry, OrderedDict()).keys())
    accession = first_available_accession(accessions, sequences_by_accession)
    if accession:
        sequence, uniprot_header, source = sequences_by_accession[accession]
        return accession, unique_join(accessions), accessions, "ok", str(len(sequence)), source, uniprot_header
    status = "missing_sequence" if accessions else "unmapped"
    return "", unique_join(accessions), accessions, status, "", "", ""


def write_gotenzymes_row_map(
    path: Path,
    args: argparse.Namespace,
    entries: OrderedDict[str, None],
    mapping: dict[str, OrderedDict[str, None]],
    sequences_by_accession: dict[str, tuple[str, str, str]],
) -> int:
    if not args.zip:
        raise SystemExit("--out-rows requires --zip so GotEnzymes reaction parameter rows can be streamed.")
    zip_path = Path(args.zip)
    fields = [
        "kegg_entry",
        *ROW_FIELD_NAMES,
        "selected_uniprot_accession",
        "mapped_uniprot_accessions",
        "sequence_status",
        "sequence_id",
        "sequence_length",
        "uniprot_fasta_source",
        "uniprot_header",
    ]
    ec_filters = set(args.ec)
    organism_filters = set(args.organism_code)
    row_count = 0
    with open_output_text(path) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in iter_gotenzymes_rows(
            zip_path=zip_path,
            ec_filters=ec_filters,
            organism_filters=organism_filters,
            max_enzyme_files=args.max_enzyme_files,
        ):
            entry = f"{row[1]}:{row[0]}"
            if entry not in entries:
                continue
            accession, mapped, _accessions, status, length, source, uniprot_header = sequence_status_for_entry(
                entry=entry,
                mapping=mapping,
                sequences_by_accession=sequences_by_accession,
            )
            writer.writerow(
                {
                    "kegg_entry": entry,
                    **{field_name: value for field_name, value in zip(ROW_FIELD_NAMES, row)},
                    "selected_uniprot_accession": accession,
                    "mapped_uniprot_accessions": mapped,
                    "sequence_status": status,
                    "sequence_id": entry if status == "ok" else "",
                    "sequence_length": length,
                    "uniprot_fasta_source": source,
                    "uniprot_header": uniprot_header,
                }
            )
            row_count += 1
    return row_count


def write_outputs(
    args: argparse.Namespace,
    entries: OrderedDict[str, None],
    mapping: dict[str, OrderedDict[str, None]],
    sequences_by_accession: dict[str, tuple[str, str, str]],
) -> None:
    fasta_count = 0
    if args.out_fasta:
        out_fasta = Path(args.out_fasta)
        with open_output_text(out_fasta) as handle:
            for entry in entries:
                accession, mapped, accessions, status, _length, _source, uniprot_header = sequence_status_for_entry(
                    entry=entry,
                    mapping=mapping,
                    sequences_by_accession=sequences_by_accession,
                )
                if status != "ok":
                    continue
                sequence, uniprot_header, _source = sequences_by_accession[accession]
                handle.write(
                    f">{entry} uniprot_accession={accession} "
                    f"mapped_uniprot_accessions={mapped} "
                    f"uniprot_header={uniprot_header}\n"
                )
                write_wrapped(handle, sequence)
                fasta_count += 1
        print(f"Wrote {fasta_count:,} FASTA records to {out_fasta}.", flush=True)

    if args.out_index:
        out_index = Path(args.out_index)
        with open_output_text(out_index) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "kegg_entry",
                    "selected_uniprot_accession",
                    "mapped_uniprot_accessions",
                    "sequence_status",
                    "sequence_length",
                    "uniprot_fasta_source",
                    "uniprot_header",
                ],
                delimiter="\t",
            )
            writer.writeheader()
            for entry in entries:
                accession, mapped, _accessions, status, length, source, uniprot_header = sequence_status_for_entry(
                    entry=entry,
                    mapping=mapping,
                    sequences_by_accession=sequences_by_accession,
                )
                writer.writerow(
                    {
                        "kegg_entry": entry,
                        "selected_uniprot_accession": accession,
                        "mapped_uniprot_accessions": mapped,
                        "sequence_status": status,
                        "sequence_length": length,
                        "uniprot_fasta_source": source,
                        "uniprot_header": uniprot_header,
                    }
                )
        print(f"Wrote sequence index to {out_index}.", flush=True)

    if args.out_map:
        out_map = Path(args.out_map)
        write_mapping(out_map, entries, mapping)
        print(f"Wrote selected KEGG-to-UniProt map to {out_map}.", flush=True)

    if args.out_rows:
        out_rows = Path(args.out_rows)
        row_count = write_gotenzymes_row_map(
            path=out_rows,
            args=args,
            entries=entries,
            mapping=mapping,
            sequences_by_accession=sequences_by_accession,
        )
        print(f"Wrote {row_count:,} GotEnzymes reaction parameter rows to {out_rows}.", flush=True)

    missing_count = len(entries) - fasta_count if args.out_fasta else 0
    if args.out_fasta:
        print(f"{missing_count:,} selected KEGG IDs did not produce a FASTA record.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot KEGG gene ID to UniProt sequence exporter. It uses UniProt's "
            "ID mapping API for KEGG->UniProtKB accessions, then scans local "
            "UniProtKB FASTA database files for the mapped sequences."
        )
    )
    parser.add_argument("--zip", default="", help="Optional GotEnzymes input_data.zip to collect KEGG IDs from.")
    parser.add_argument("--all", action="store_true", help="With --zip, select all GotEnzymes KEGG IDs.")
    parser.add_argument("--ec", action="append", default=[], help="With --zip, exact EC number to select.")
    parser.add_argument(
        "--organism-code",
        action="append",
        default=[],
        help="With --zip, KEGG organism code to select.",
    )
    parser.add_argument("--max-enzyme-files", type=int, default=None, help="With --zip, scan only first N files.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum unique KEGG IDs to process.")
    parser.add_argument("--kegg-id", action="append", default=[], help="Single KEGG gene ID, e.g. aaa:Acav_0021.")
    parser.add_argument(
        "--kegg-id-file",
        action="append",
        default=[],
        help="File containing KEGG gene IDs, one per line or in a TSV/CSV column.",
    )
    parser.add_argument(
        "--input-column",
        default="kegg_entry",
        help="Column name or zero-based column index for --kegg-id-file tables.",
    )
    parser.add_argument(
        "--uniprot-fasta",
        action="append",
        default=[],
        help=(
            "Local UniProtKB FASTA database file, e.g. uniprot_sprot.fasta.gz or "
            "uniprot_trembl.fasta.gz. May be repeated."
        ),
    )
    parser.add_argument(
        "--download-uniprot-dir",
        default="",
        help="Download current UniProtKB Swiss-Prot and TrEMBL FASTA.gz files into this directory.",
    )
    parser.add_argument("--out-fasta", default="", help="Output FASTA path.")
    parser.add_argument("--out-index", default="", help="Output sequence index TSV path.")
    parser.add_argument("--out-map", default="", help="Output selected KEGG-to-UniProt mapping TSV path.")
    parser.add_argument("--api-log", default="", help="Optional UniProt ID mapping API job audit TSV path.")
    parser.add_argument(
        "--out-rows",
        default="",
        help="Output GotEnzymes reaction parameter row-map TSV path. Use .gz for the full archive.",
    )
    parser.add_argument("--batch-size", type=int, default=50000, help="KEGG IDs per UniProt ID mapping job.")
    parser.add_argument("--max-mapping-jobs", type=int, default=200, help="Safety cap for UniProt mapping jobs.")
    parser.add_argument("--allow-large-mapping-run", action="store_true", help="Allow exceeding --max-mapping-jobs.")
    parser.add_argument("--timeout", type=float, default=60.0, help="UniProt request timeout in seconds.")
    parser.add_argument("--download-timeout", type=float, default=120.0, help="UniProt FASTA download timeout.")
    parser.add_argument("--retries", type=int, default=3, help="Retries per UniProt request.")
    parser.add_argument("--sleep-seconds", type=float, default=0.5, help="Delay between retries/backoff base.")
    parser.add_argument("--poll-seconds", type=float, default=3.0, help="ID mapping job poll interval.")
    parser.add_argument("--max-polls", type=int, default=120, help="Maximum polls per ID mapping job.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive.")
    if not args.zip and not args.kegg_id and not args.kegg_id_file:
        raise SystemExit("Provide --zip, --kegg-id, or --kegg-id-file.")
    if not args.uniprot_fasta and not args.download_uniprot_dir:
        raise SystemExit("Provide at least one --uniprot-fasta or --download-uniprot-dir.")
    if not args.out_fasta and not args.out_index and not args.out_map and not args.out_rows:
        raise SystemExit("Provide --out-fasta, --out-index, --out-map, and/or --out-rows.")
    if args.out_rows and not args.zip:
        raise SystemExit("--out-rows requires --zip.")

    fasta_paths = resolve_uniprot_fasta_paths(args)

    entries = collect_kegg_entries(args)
    if not entries:
        raise SystemExit("No KEGG gene IDs selected.")
    print(f"Selected {len(entries):,} unique KEGG gene IDs.", flush=True)

    mapping = map_entries(args, entries)
    mapped_count = sum(1 for accessions in mapping.values() if accessions)
    print(f"Mapped {mapped_count:,}/{len(entries):,} KEGG IDs to at least one UniProt accession.", flush=True)

    sequences_by_accession = load_uniprot_sequences(fasta_paths, mapping)
    write_outputs(args, entries, mapping, sequences_by_accession)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
