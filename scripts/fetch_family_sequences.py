#!/usr/bin/env python3
"""Fetch UniProt sequences for selected enzyme rows from input_data.zip."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ENZYME_PREFIX = "input_data/enzymes/"
EXPECTED_ENZYME_COLUMNS = 11
NUMERIC_COLUMN_INDEXES = range(6, 11)
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
USER_AGENT = "MSA_DESIGN pilot sequence fetcher (UniProt REST; contact unavailable)"
FASTA_WRAP = 80


def split_ec_numbers(raw_ecs: str) -> list[str]:
    return [part.strip() for part in raw_ecs.split(";") if part.strip()]


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return slug or "value"


def cache_path(cache_dir: Path, organism_code: str, gene_id: str) -> Path:
    key = f"{organism_code}:{gene_id}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{safe_slug(organism_code)}__{safe_slug(gene_id)}__{digest}.json"


def unique_join(values: list[str], max_values: int = 25) -> str:
    seen: OrderedDict[str, None] = OrderedDict()
    for value in values:
        if value not in seen:
            seen[value] = None
    items = list(seen.keys())
    if len(items) <= max_values:
        return ";".join(items)
    return ";".join(items[:max_values]) + f";...(+{len(items) - max_values})"


def iter_selected_rows(
    zip_path: Path,
    ec_filters: set[str],
    organism_filters: set[str],
    limit: int | None,
    max_enzyme_files: int | None,
) -> OrderedDict[tuple[str, str], list[list[str]]]:
    selected: OrderedDict[tuple[str, str], list[list[str]]] = OrderedDict()
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
                    key = (organism_code, gene_id)
                    if key not in selected:
                        if limit is not None and len(selected) >= limit:
                            return selected
                        selected[key] = []
                    selected[key].append(parts)
    return selected


def uniprot_escape_colon(value: str) -> str:
    return value.replace(":", r"\:")


def build_url(query: str, size: int) -> str:
    fields = ",".join(
        [
            "accession",
            "id",
            "protein_name",
            "gene_names",
            "organism_name",
            "xref_kegg",
            "sequence",
            "length",
            "ec",
            "reviewed",
        ]
    )
    return UNIPROT_SEARCH_URL + "?" + urllib.parse.urlencode(
        {"query": query, "format": "json", "size": str(size), "fields": fields}
    )


def fetch_json(url: str, sleep_seconds: float, timeout: float, retries: int = 3) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:1000]
            last_error = RuntimeError(f"HTTP {exc.code}: {body}")
            if exc.code not in {429, 500, 502, 503, 504} or attempt == retries:
                break
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else sleep_seconds * attempt
            time.sleep(max(delay, sleep_seconds, 1.0))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(max(sleep_seconds * attempt, 1.0))
    raise RuntimeError(str(last_error))


def extract_kegg_crossrefs(result: dict[str, Any]) -> list[str]:
    refs = []
    for ref in result.get("uniProtKBCrossReferences", []):
        if ref.get("database") == "KEGG" and ref.get("id"):
            refs.append(ref["id"])
    return refs


def extract_gene_names(result: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for gene in result.get("genes", []):
        for field in ("geneName",):
            value = gene.get(field, {}).get("value")
            if value:
                names.append(value)
        for field in ("synonyms", "orderedLocusNames", "orfNames"):
            for item in gene.get(field, []):
                value = item.get("value")
                if value:
                    names.append(value)
    return names


def extract_protein_name(result: dict[str, Any]) -> str:
    description = result.get("proteinDescription", {})
    recommended = description.get("recommendedName", {})
    full = recommended.get("fullName", {}).get("value")
    if full:
        return full
    for item in description.get("submissionNames", []):
        full = item.get("fullName", {}).get("value")
        if full:
            return full
    return ""


def extract_ec_numbers(result: dict[str, Any]) -> list[str]:
    ecs: list[str] = []
    description = result.get("proteinDescription", {})
    blocks: list[dict[str, Any]] = []
    if description.get("recommendedName"):
        blocks.append(description["recommendedName"])
    blocks.extend(description.get("alternativeNames", []))
    blocks.extend(description.get("submissionNames", []))
    for block in blocks:
        for item in block.get("ecNumbers", []):
            value = item.get("value")
            if value:
                ecs.append(value)
    return ecs


def normalize_result(result: dict[str, Any], expected_kegg: str) -> dict[str, Any]:
    sequence = result.get("sequence", {}) or {}
    organism = result.get("organism", {}) or {}
    kegg_crossrefs = extract_kegg_crossrefs(result)
    return {
        "accession": result.get("primaryAccession", ""),
        "entry_name": result.get("uniProtkbId", ""),
        "entry_type": result.get("entryType", ""),
        "protein_name": extract_protein_name(result),
        "gene_names": extract_gene_names(result),
        "organism_name": organism.get("scientificName", ""),
        "taxon_id": organism.get("taxonId", ""),
        "sequence": sequence.get("value", ""),
        "sequence_length": sequence.get("length", ""),
        "uniprot_ec_numbers": extract_ec_numbers(result),
        "kegg_crossrefs": kegg_crossrefs,
        "kegg_xref_verified": expected_kegg in kegg_crossrefs,
    }


def choose_candidate(
    results: list[dict[str, Any]],
    expected_kegg: str,
) -> tuple[dict[str, Any] | None, str]:
    with_sequence = [result for result in results if result.get("sequence", {}).get("value")]
    if not with_sequence:
        return None, "no_sequence"
    for result in with_sequence:
        if expected_kegg in extract_kegg_crossrefs(result):
            return result, "kegg_xref"
    return with_sequence[0], "gene_exact_unverified"


def query_uniprot_for_gene(
    organism_code: str,
    gene_id: str,
    cache_dir: Path,
    refresh: bool,
    sleep_seconds: float,
    timeout: float,
) -> dict[str, Any]:
    cache_file = cache_path(cache_dir, organism_code, gene_id)
    expected_kegg = f"{organism_code}:{gene_id}"
    if cache_file.exists() and not refresh:
        with cache_file.open("r", encoding="utf-8") as handle:
            cached = json.load(handle)
        cached["cache_file"] = str(cache_file)
        cached["cache_hit"] = True
        return cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "organism_code": organism_code,
        "gene_id": gene_id,
        "expected_kegg": expected_kegg,
        "queried_at_utc": datetime.now(timezone.utc).isoformat(),
        "queries": [],
        "selected": None,
        "status": "no_hit",
        "error": "",
    }

    try:
        exact_query = f"xref:KEGG-{organism_code}\\:{uniprot_escape_colon(gene_id)}"
        exact_url = build_url(exact_query, size=5)
        exact_data = fetch_json(exact_url, sleep_seconds=sleep_seconds, timeout=timeout)
        exact_results = exact_data.get("results", [])
        record["queries"].append(
            {
                "method": "kegg_xref",
                "query": exact_query,
                "result_count": len(exact_results),
                "results": exact_results,
            }
        )

        candidate, method = choose_candidate(exact_results, expected_kegg)
        if candidate is None:
            fallback_query = f"gene_exact:{gene_id}"
            fallback_url = build_url(fallback_query, size=5)
            fallback_data = fetch_json(fallback_url, sleep_seconds=sleep_seconds, timeout=timeout)
            fallback_results = fallback_data.get("results", [])
            record["queries"].append(
                {
                    "method": "gene_exact",
                    "query": fallback_query,
                    "result_count": len(fallback_results),
                    "results": fallback_results,
                }
            )
            candidate, method = choose_candidate(fallback_results, expected_kegg)

        if candidate is not None:
            selected = normalize_result(candidate, expected_kegg)
            selected["query_method"] = method
            record["selected"] = selected
            record["status"] = "ok"
    except Exception as exc:  # Network and API failures should still produce metadata.
        record["status"] = "error"
        record["error"] = str(exc)

    with cache_file.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    record["cache_file"] = str(cache_file)
    record["cache_hit"] = False
    return record


def summarize_rows(rows: list[list[str]]) -> dict[str, str]:
    summary = {
        "selected_row_count": str(len(rows)),
        "ec_numbers": unique_join([ec for row in rows for ec in split_ec_numbers(row[4])]),
        "reaction_ids": unique_join([row[3] for row in rows]),
        "compound_ids": unique_join([row[5] for row in rows]),
    }
    for idx in NUMERIC_COLUMN_INDEXES:
        summary[f"numeric_col_{idx + 1}_unlabeled_values"] = unique_join([row[idx] for row in rows])
    return summary


def fasta_header(
    organism_code: str,
    gene_id: str,
    selected: dict[str, Any],
    row_summary: dict[str, str],
) -> str:
    accession = selected.get("accession") or "uniprot_unknown"
    ec_numbers = row_summary.get("ec_numbers", "")
    verified = "yes" if selected.get("kegg_xref_verified") else "no"
    return (
        f"{accession}|{organism_code}|{gene_id} "
        f"ec={ec_numbers or 'NA'} rows={row_summary.get('selected_row_count', '0')} "
        f"kegg_verified={verified}"
    )


def write_wrapped(handle: Any, sequence: str) -> None:
    for start in range(0, len(sequence), FASTA_WRAP):
        handle.write(sequence[start : start + FASTA_WRAP] + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select enzyme rows and fetch matching UniProt sequences."
    )
    parser.add_argument("--zip", default="data/input_data.zip", help="Path to input_data.zip")
    parser.add_argument(
        "--ec",
        action="append",
        default=[],
        help="Exact EC number to select. May be repeated.",
    )
    parser.add_argument(
        "--organism-code",
        action="append",
        default=[],
        help="KEGG organism code to select. May be repeated.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of unique organism/gene IDs to fetch.",
    )
    parser.add_argument(
        "--max-enzyme-files",
        type=int,
        default=None,
        help="Scan only the first N sorted enzyme files.",
    )
    parser.add_argument("--out-fasta", required=True, help="Output FASTA path.")
    parser.add_argument("--out-metadata", required=True, help="Output metadata TSV path.")
    parser.add_argument(
        "--cache-dir",
        default="data/cache/uniprot",
        help="Directory for UniProt JSON cache files.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Polite delay after UniProt REST requests.",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="UniProt request timeout.")
    parser.add_argument("--refresh", action="store_true", help="Ignore existing cache files.")
    parser.add_argument(
        "--require-kegg-xref",
        action="store_true",
        help="Write FASTA only for UniProt records with the expected KEGG cross-reference.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and select rows, but do not contact UniProt.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"Archive not found: {zip_path}")
    ec_filters = set(args.ec)
    organism_filters = set(args.organism_code)
    if not ec_filters and not organism_filters:
        raise SystemExit("Provide --ec and/or --organism-code to avoid selecting the whole archive.")

    selected_rows = iter_selected_rows(
        zip_path=zip_path,
        ec_filters=ec_filters,
        organism_filters=organism_filters,
        limit=args.limit,
        max_enzyme_files=args.max_enzyme_files,
    )
    out_fasta = Path(args.out_fasta)
    out_metadata = Path(args.out_metadata)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)
    out_metadata.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    metadata_fields = [
        "organism_code",
        "gene_id",
        "kegg_expected",
        "selected_row_count",
        "ec_numbers",
        "reaction_ids",
        "compound_ids",
        "numeric_col_7_unlabeled_values",
        "numeric_col_8_unlabeled_values",
        "numeric_col_9_unlabeled_values",
        "numeric_col_10_unlabeled_values",
        "numeric_col_11_unlabeled_values",
        "status",
        "query_method",
        "uniprot_accession",
        "uniprot_entry_name",
        "entry_type",
        "protein_name",
        "gene_names",
        "organism_name",
        "taxon_id",
        "sequence_length",
        "uniprot_ec_numbers",
        "kegg_crossrefs",
        "kegg_xref_verified",
        "cache_hit",
        "cache_file",
        "error",
    ]

    fasta_count = 0
    with out_fasta.open("w", encoding="utf-8") as fasta_handle, out_metadata.open(
        "w", encoding="utf-8", newline=""
    ) as metadata_handle:
        writer = csv.DictWriter(metadata_handle, fieldnames=metadata_fields, delimiter="\t")
        writer.writeheader()

        for (organism_code, gene_id), rows in selected_rows.items():
            row_summary = summarize_rows(rows)
            expected_kegg = f"{organism_code}:{gene_id}"
            metadata = {
                "organism_code": organism_code,
                "gene_id": gene_id,
                "kegg_expected": expected_kegg,
                **row_summary,
                "status": "dry_run" if args.dry_run else "no_hit",
                "query_method": "",
                "uniprot_accession": "",
                "uniprot_entry_name": "",
                "entry_type": "",
                "protein_name": "",
                "gene_names": "",
                "organism_name": "",
                "taxon_id": "",
                "sequence_length": "",
                "uniprot_ec_numbers": "",
                "kegg_crossrefs": "",
                "kegg_xref_verified": "",
                "cache_hit": "",
                "cache_file": "",
                "error": "",
            }

            if not args.dry_run:
                record = query_uniprot_for_gene(
                    organism_code=organism_code,
                    gene_id=gene_id,
                    cache_dir=cache_dir,
                    refresh=args.refresh,
                    sleep_seconds=args.sleep_seconds,
                    timeout=args.timeout,
                )
                selected = record.get("selected") or {}
                metadata.update(
                    {
                        "status": record.get("status", ""),
                        "query_method": selected.get("query_method", ""),
                        "uniprot_accession": selected.get("accession", ""),
                        "uniprot_entry_name": selected.get("entry_name", ""),
                        "entry_type": selected.get("entry_type", ""),
                        "protein_name": selected.get("protein_name", ""),
                        "gene_names": unique_join(selected.get("gene_names", [])),
                        "organism_name": selected.get("organism_name", ""),
                        "taxon_id": selected.get("taxon_id", ""),
                        "sequence_length": selected.get("sequence_length", ""),
                        "uniprot_ec_numbers": unique_join(selected.get("uniprot_ec_numbers", [])),
                        "kegg_crossrefs": unique_join(selected.get("kegg_crossrefs", [])),
                        "kegg_xref_verified": str(bool(selected.get("kegg_xref_verified"))),
                        "cache_hit": str(bool(record.get("cache_hit"))),
                        "cache_file": record.get("cache_file", ""),
                        "error": record.get("error", ""),
                    }
                )
                sequence = selected.get("sequence", "")
                if args.require_kegg_xref and not selected.get("kegg_xref_verified"):
                    sequence = ""
                    if metadata["status"] == "ok":
                        metadata["status"] = "skipped_unverified_kegg_xref"
                if sequence:
                    fasta_handle.write(f">{fasta_header(organism_code, gene_id, selected, row_summary)}\n")
                    write_wrapped(fasta_handle, sequence)
                    fasta_count += 1

            writer.writerow(metadata)

    print(
        f"Selected {len(selected_rows)} unique organism/gene IDs; "
        f"wrote {fasta_count} FASTA records to {out_fasta}; metadata to {out_metadata}."
    )
    if args.dry_run:
        print("Dry run: UniProt was not queried.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
