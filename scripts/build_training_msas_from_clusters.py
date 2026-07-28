#!/usr/bin/env python3
"""Build aggressively trimmed training MSAs from clustered protein sequences."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


FASTA_WRAP = 80


@dataclass(frozen=True)
class ClusterJob:
    cluster_index: int
    representative: str
    size: int
    input_fasta: Path
    raw_alignment: Path
    trimmed_alignment: Path


@dataclass
class ClusterResult:
    cluster_index: int
    representative: str
    status: str
    message: str
    size: int
    kept_size: int
    dropped_size: int
    raw_alignment_length: int
    trimmed_alignment_length: int
    raw_gap_fraction: float
    trimmed_gap_fraction: float
    input_fasta: str
    raw_alignment: str
    trimmed_alignment: str
    elapsed_seconds: float
    sequence_rows: list[list[Any]]


def safe_name(value: str, max_len: int = 60) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    if len(clean) > max_len:
        clean = clean[:max_len].rstrip("_")
    return f"{clean}_{digest}" if clean else digest


def open_text(path: Path) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def open_output_text(path: Path) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return path.open("w", encoding="utf-8", newline="")


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_parts: list[str] = []
    with open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence_parts)))
                header = line[1:].split()[0]
                sequence_parts = []
            else:
                sequence_parts.append(line)
    if header is not None:
        records.append((header, "".join(sequence_parts)))
    return records


def write_fasta(records: Iterable[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_output_text(path) as handle:
        for header, sequence in records:
            handle.write(f">{header}\n")
            for start in range(0, len(sequence), FASTA_WRAP):
                handle.write(sequence[start : start + FASTA_WRAP] + "\n")


def parse_members(path: Path, max_clusters: int | None) -> list[tuple[str, list[str]]]:
    clusters: dict[str, list[str]] = {}
    order: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            representative = row["representative"]
            member = row["member"]
            if representative not in clusters:
                if max_clusters is not None and len(order) >= max_clusters:
                    continue
                clusters[representative] = []
                order.append(representative)
            clusters[representative].append(member)
    return [(representative, clusters[representative]) for representative in order]


def load_needed_sequences(fasta_path: Path, needed_ids: set[str]) -> dict[str, str]:
    sequences: dict[str, str] = {}
    header: str | None = None
    sequence_parts: list[str] = []
    with open_text(fasta_path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and header in needed_ids:
                    sequences[header] = "".join(sequence_parts)
                header = line[1:].split()[0]
                sequence_parts = []
            else:
                sequence_parts.append(line)
        if header is not None and header in needed_ids:
            sequences[header] = "".join(sequence_parts)
    return sequences


def alignment_gap_fraction(records: list[tuple[str, str]]) -> float:
    total = sum(len(sequence) for _header, sequence in records)
    if total == 0:
        return 0.0
    gaps = sum(sequence.count("-") for _header, sequence in records)
    return gaps / total


def remove_columns(records: list[tuple[str, str]], keep_columns: list[int]) -> list[tuple[str, str]]:
    return [(header, "".join(sequence[index] for index in keep_columns)) for header, sequence in records]


def trim_alignment(
    records: list[tuple[str, str]],
    max_column_gap: float,
    max_sequence_gap: float,
    min_sequences: int,
    min_columns: int,
    min_residues: int,
) -> tuple[str, str, list[tuple[str, str]], dict[str, str], dict[str, float]]:
    if not records:
        return "failed", "empty_alignment", [], {}, {}
    lengths = {len(sequence) for _header, sequence in records}
    if len(lengths) != 1:
        return "failed", "ragged_alignment", [], {}, {}

    raw_lengths = {header: len(sequence.replace("-", "")) for header, sequence in records}
    current = records
    dropped: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        alignment_length = len(current[0][1]) if current else 0
        if alignment_length == 0:
            return "failed", "no_columns_after_trimming", [], dropped, {}

        row_keep: list[tuple[str, str]] = []
        for header, sequence in current:
            non_gaps = alignment_length - sequence.count("-")
            gap_fraction = 1.0 - (non_gaps / alignment_length)
            if gap_fraction <= max_sequence_gap and non_gaps >= min_residues:
                row_keep.append((header, sequence))
            else:
                dropped[header] = "sequence_gap_or_short_after_trim"
                changed = True
        current = row_keep
        if len(current) < min_sequences:
            return "failed", "too_few_sequences_after_row_filter", current, dropped, {}

        alignment_length = len(current[0][1])
        keep_columns: list[int] = []
        for index in range(alignment_length):
            non_gaps = sum(1 for _header, sequence in current if sequence[index] != "-")
            gap_fraction = 1.0 - (non_gaps / len(current))
            if gap_fraction <= max_column_gap:
                keep_columns.append(index)
        if len(keep_columns) < alignment_length:
            current = remove_columns(current, keep_columns)
            changed = True
        if len(keep_columns) < min_columns:
            return "failed", "too_few_columns_after_column_filter", current, dropped, {}

    final_stats: dict[str, float] = {}
    for header, sequence in current:
        non_gaps = len(sequence) - sequence.count("-")
        final_stats[header] = non_gaps / len(sequence) if sequence else 0.0
        if raw_lengths.get(header, 0) < min_residues:
            dropped[header] = "sequence_too_short_raw"

    kept_headers = {header for header, _sequence in current}
    if len(kept_headers) < len(current):
        return "failed", "duplicate_headers", current, dropped, final_stats
    if len(current) < min_sequences:
        return "failed", "too_few_sequences_final", current, dropped, final_stats
    if len(current[0][1]) < min_columns:
        return "failed", "too_few_columns_final", current, dropped, final_stats
    return "ok", "", current, dropped, final_stats


def run_kalign(kalign: Path, input_fasta: Path, output_fasta: Path, threads: int, mode: str) -> None:
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(kalign),
        "-i",
        str(input_fasta),
        "-o",
        str(output_fasta),
        "--format",
        "fasta",
        "--type",
        "protein",
        "--mode",
        mode,
    ]
    if threads > 0:
        command.extend(["--nthreads", str(threads)])
    result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "").strip()[:2000])


def process_cluster(
    job: ClusterJob,
    kalign: Path,
    kalign_threads: int,
    kalign_mode: str,
    max_column_gap: float,
    max_sequence_gap: float,
    min_sequences: int,
    min_columns: int,
    min_residues: int,
    keep_raw_alignments: bool,
) -> ClusterResult:
    start = time.time()
    raw_records: list[tuple[str, str]] = []
    trimmed_records: list[tuple[str, str]] = []
    dropped: dict[str, str] = {}
    final_row_coverage: dict[str, float] = {}
    status = "ok"
    message = ""
    try:
        run_kalign(
            kalign=kalign,
            input_fasta=job.input_fasta,
            output_fasta=job.raw_alignment,
            threads=kalign_threads,
            mode=kalign_mode,
        )
        raw_records = read_fasta(job.raw_alignment)
        status, message, trimmed_records, dropped, final_row_coverage = trim_alignment(
            records=raw_records,
            max_column_gap=max_column_gap,
            max_sequence_gap=max_sequence_gap,
            min_sequences=min_sequences,
            min_columns=min_columns,
            min_residues=min_residues,
        )
        if status == "ok":
            write_fasta(trimmed_records, job.trimmed_alignment)
        elif job.trimmed_alignment.exists():
            job.trimmed_alignment.unlink()
    except Exception as exc:
        status = "failed"
        message = str(exc)[:2000]
    finally:
        if not keep_raw_alignments and job.raw_alignment.exists():
            job.raw_alignment.unlink()

    raw_by_header = {header: sequence for header, sequence in raw_records}
    kept_headers = {header for header, _sequence in trimmed_records}
    trimmed_by_header = {header: sequence for header, sequence in trimmed_records}
    sequence_rows: list[list[Any]] = []
    input_records = read_fasta(job.input_fasta)
    raw_alignment_length = len(raw_records[0][1]) if raw_records else 0
    trimmed_alignment_length = len(trimmed_records[0][1]) if trimmed_records else 0
    for header, original_sequence in input_records:
        raw_sequence = raw_by_header.get(header, "")
        trimmed_sequence = trimmed_by_header.get(header, "")
        raw_row_coverage = (
            (len(raw_sequence) - raw_sequence.count("-")) / len(raw_sequence) if raw_sequence else 0.0
        )
        kept = header in kept_headers and status == "ok"
        sequence_rows.append(
            [
                job.cluster_index,
                job.representative,
                header,
                "yes" if kept else "no",
                "" if kept else dropped.get(header, message or status),
                len(original_sequence),
                raw_row_coverage,
                final_row_coverage.get(header, 0.0),
                job.trimmed_alignment if kept else "",
            ]
        )

    return ClusterResult(
        cluster_index=job.cluster_index,
        representative=job.representative,
        status=status,
        message=message,
        size=job.size,
        kept_size=len(trimmed_records) if status == "ok" else 0,
        dropped_size=job.size - (len(trimmed_records) if status == "ok" else 0),
        raw_alignment_length=raw_alignment_length,
        trimmed_alignment_length=trimmed_alignment_length,
        raw_gap_fraction=alignment_gap_fraction(raw_records),
        trimmed_gap_fraction=alignment_gap_fraction(trimmed_records) if status == "ok" else 0.0,
        input_fasta=str(job.input_fasta),
        raw_alignment=str(job.raw_alignment) if keep_raw_alignments else "",
        trimmed_alignment=str(job.trimmed_alignment) if status == "ok" else "",
        elapsed_seconds=time.time() - start,
        sequence_rows=sequence_rows,
    )


def write_cluster_inputs(
    clusters: list[tuple[str, list[str]]],
    sequences: dict[str, str],
    out_dir: Path,
) -> list[ClusterJob]:
    jobs: list[ClusterJob] = []
    input_dir = out_dir / "cluster_fastas"
    raw_dir = out_dir / "raw_alignments"
    trimmed_dir = out_dir / "trimmed_alignments"
    for index, (representative, members) in enumerate(clusters, start=1):
        stem = f"{index:06d}_{safe_name(representative)}"
        input_fasta = input_dir / f"{stem}.fa"
        raw_alignment = raw_dir / f"{stem}.raw.afa"
        trimmed_alignment = trimmed_dir / f"{stem}.trimmed.afa.gz"
        if not input_fasta.exists():
            records = [(member, sequences[member]) for member in members if member in sequences]
            write_fasta(records, input_fasta)
        jobs.append(
            ClusterJob(
                cluster_index=index,
                representative=representative,
                size=len(members),
                input_fasta=input_fasta,
                raw_alignment=raw_alignment,
                trimmed_alignment=trimmed_alignment,
            )
        )
    return jobs


def filter_table_by_kept(
    source_path: Path,
    key_column: str,
    kept_ids: set[str],
    output_path: Path,
) -> int:
    count = 0
    with open_text(source_path) as source, open_output_text(output_path) as target:
        reader = csv.DictReader(source, delimiter="\t")
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames, delimiter="\t")
        writer.writeheader()
        for row in reader:
            if row.get(key_column, "") in kept_ids:
                writer.writerow(row)
                count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trimmed training MSAs from a cluster member TSV.")
    parser.add_argument("--input-fasta", required=True, help="FASTA containing all clustered sequences.")
    parser.add_argument("--members", required=True, help="good_msa_members.tsv from the chosen cluster threshold.")
    parser.add_argument("--out-root", required=True, help="Output directory, preferably on the large disk.")
    parser.add_argument("--kalign", default="tools/kalign/bin/kalign", help="Kalign executable.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel cluster alignments.")
    parser.add_argument("--kalign-threads", type=int, default=2, help="Threads per Kalign process.")
    parser.add_argument("--kalign-mode", default="fast", choices=["fast", "default", "recall", "accurate"])
    parser.add_argument("--max-clusters", type=int, default=None, help="Limit clusters for a smoke run.")
    parser.add_argument("--max-column-gap", type=float, default=0.20, help="Maximum final gap fraction per column.")
    parser.add_argument("--max-sequence-gap", type=float, default=0.30, help="Maximum final gap fraction per row.")
    parser.add_argument("--min-sequences", type=int, default=16, help="Minimum final MSA rows.")
    parser.add_argument("--min-columns", type=int, default=50, help="Minimum final MSA columns.")
    parser.add_argument("--min-residues", type=int, default=30, help="Minimum residues per retained sequence.")
    parser.add_argument("--keep-raw-alignments", action="store_true", help="Keep untrimmed Kalign outputs.")
    parser.add_argument("--sequence-index", default="", help="Optional all_sequence_index.tsv.gz for kept subset.")
    parser.add_argument("--reaction-rows", default="", help="Optional all_reaction_parameters_uniprot.tsv.gz.")
    parser.add_argument("--resume", action="store_true", help="Skip clusters with existing final trimmed MSA.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    kalign = Path(args.kalign)
    if not kalign.exists():
        raise SystemExit(f"Kalign executable not found: {kalign}")

    clusters = parse_members(Path(args.members), max_clusters=args.max_clusters)
    needed_ids = {member for _rep, members in clusters for member in members}
    print(f"Selected {len(clusters):,} clusters with {len(needed_ids):,} member sequences.", flush=True)
    sequences = load_needed_sequences(Path(args.input_fasta), needed_ids)
    missing = sorted(needed_ids - set(sequences))
    if missing:
        missing_path = out_root / "missing_cluster_members.txt"
        missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        print(f"WARNING: missing {len(missing):,} FASTA records; wrote {missing_path}", flush=True)
    print(f"Loaded {len(sequences):,} sequences from FASTA.", flush=True)

    jobs = write_cluster_inputs(clusters, sequences, out_root)
    if args.resume:
        jobs = [job for job in jobs if not job.trimmed_alignment.exists()]
        print(f"Resume mode: {len(jobs):,} clusters still need processing.", flush=True)

    msa_manifest_path = out_root / "msa_manifest.tsv"
    sequence_manifest_path = out_root / "sequence_manifest.tsv.gz"
    with msa_manifest_path.open("w", encoding="utf-8", newline="") as msa_handle, gzip.open(
        sequence_manifest_path, "wt", encoding="utf-8", newline=""
    ) as seq_handle:
        msa_writer = csv.writer(msa_handle, delimiter="\t")
        msa_writer.writerow(
            [
                "cluster_index",
                "representative",
                "status",
                "message",
                "input_size",
                "kept_size",
                "dropped_size",
                "raw_alignment_length",
                "trimmed_alignment_length",
                "raw_gap_fraction",
                "trimmed_gap_fraction",
                "input_fasta",
                "raw_alignment",
                "trimmed_alignment",
                "elapsed_seconds",
            ]
        )
        seq_writer = csv.writer(seq_handle, delimiter="\t")
        seq_writer.writerow(
            [
                "cluster_index",
                "representative",
                "kegg_entry",
                "kept",
                "drop_reason",
                "ungapped_length",
                "raw_row_coverage",
                "trimmed_row_coverage",
                "trimmed_alignment",
            ]
        )

        completed = 0
        ok_clusters = 0
        kept_sequences = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_job = {
                executor.submit(
                    process_cluster,
                    job,
                    kalign,
                    args.kalign_threads,
                    args.kalign_mode,
                    args.max_column_gap,
                    args.max_sequence_gap,
                    args.min_sequences,
                    args.min_columns,
                    args.min_residues,
                    args.keep_raw_alignments,
                ): job
                for job in jobs
            }
            for future in as_completed(future_to_job):
                result = future.result()
                completed += 1
                if result.status == "ok":
                    ok_clusters += 1
                    kept_sequences += result.kept_size
                msa_writer.writerow(
                    [
                        result.cluster_index,
                        result.representative,
                        result.status,
                        result.message,
                        result.size,
                        result.kept_size,
                        result.dropped_size,
                        result.raw_alignment_length,
                        result.trimmed_alignment_length,
                        f"{result.raw_gap_fraction:.6f}",
                        f"{result.trimmed_gap_fraction:.6f}",
                        result.input_fasta,
                        result.raw_alignment,
                        result.trimmed_alignment,
                        f"{result.elapsed_seconds:.3f}",
                    ]
                )
                seq_writer.writerows(result.sequence_rows)
                if completed % 100 == 0 or completed == len(jobs):
                    print(
                        f"processed {completed:,}/{len(jobs):,} clusters; "
                        f"ok={ok_clusters:,}; kept_sequences={kept_sequences:,}",
                        flush=True,
                    )

    kept_ids: set[str] = set()
    with gzip.open(sequence_manifest_path, "rt", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row["kept"] == "yes":
                kept_ids.add(row["kegg_entry"])

    if args.sequence_index:
        count = filter_table_by_kept(
            source_path=Path(args.sequence_index),
            key_column="kegg_entry",
            kept_ids=kept_ids,
            output_path=out_root / "kept_sequence_index.tsv.gz",
        )
        print(f"Wrote {count:,} kept sequence-index rows.", flush=True)
    if args.reaction_rows:
        count = filter_table_by_kept(
            source_path=Path(args.reaction_rows),
            key_column="kegg_entry",
            kept_ids=kept_ids,
            output_path=out_root / "kept_reaction_parameters.tsv.gz",
        )
        print(f"Wrote {count:,} kept reaction-parameter rows.", flush=True)

    report_path = out_root / "training_msa_report.txt"
    with msa_manifest_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    ok_rows = [row for row in rows if row["status"] == "ok"]
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("Training MSA build report\n")
        handle.write("=========================\n\n")
        handle.write(f"Input clusters: {len(rows):,}\n")
        handle.write(f"Successful MSAs: {len(ok_rows):,}\n")
        handle.write(f"Kept sequences: {len(kept_ids):,}\n")
        if ok_rows:
            raw_gap = sum(float(row["raw_gap_fraction"]) for row in ok_rows) / len(ok_rows)
            trimmed_gap = sum(float(row["trimmed_gap_fraction"]) for row in ok_rows) / len(ok_rows)
            handle.write(f"Mean raw gap fraction: {raw_gap:.6f}\n")
            handle.write(f"Mean trimmed gap fraction: {trimmed_gap:.6f}\n")
        handle.write(f"MSA manifest: {msa_manifest_path}\n")
        handle.write(f"Sequence manifest: {sequence_manifest_path}\n")
        handle.write("Trimmed alignment directory: " + str(out_root / "trimmed_alignments") + "\n")
    print(f"Wrote {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
