#!/usr/bin/env python3
"""Fetch a few early/small enzyme families and build pilot MSAs."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import zipfile
from collections import OrderedDict
from pathlib import Path


ENZYME_PREFIX = "input_data/enzymes/"
EXPECTED_ENZYME_COLUMNS = 11


def split_ec_numbers(raw_ecs: str) -> list[str]:
    return [part.strip() for part in raw_ecs.split(";") if part.strip()]


def safe_ec_name(ec_number: str) -> str:
    return "ec_" + ec_number.replace(".", "_").replace("-", "x")


def choose_families(
    zip_path: Path,
    scan_files: int,
    family_count: int,
    min_genes: int,
    max_genes: int,
) -> list[tuple[str, int]]:
    families: OrderedDict[str, OrderedDict[tuple[str, str], None]] = OrderedDict()
    with zipfile.ZipFile(zip_path) as zf:
        enzyme_names = sorted(
            name
            for name in zf.namelist()
            if name.startswith(ENZYME_PREFIX) and name.endswith(".txt")
        )[:scan_files]
        for name in enzyme_names:
            with zf.open(name) as handle:
                for raw_line in handle:
                    line = raw_line.decode("utf-8", "replace").rstrip("\n\r")
                    if not line:
                        continue
                    parts = line.split("\t")
                    if len(parts) != EXPECTED_ENZYME_COLUMNS:
                        continue
                    key = (parts[1], parts[0])
                    for ec_number in split_ec_numbers(parts[4]):
                        if ec_number not in families:
                            families[ec_number] = OrderedDict()
                        families[ec_number][key] = None

    candidates = [
        (ec_number, len(genes))
        for ec_number, genes in families.items()
        if min_genes <= len(genes) <= max_genes
    ]
    if len(candidates) < family_count:
        seen = {ec_number for ec_number, _ in candidates}
        for ec_number, genes in families.items():
            if ec_number not in seen and len(genes) >= min_genes:
                candidates.append((ec_number, len(genes)))
            if len(candidates) >= family_count:
                break
    return candidates[:family_count]


def count_fasta_records(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                count += 1
    return count


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Choose small early EC families, fetch KEGG source sequences, and build pilot MSAs."
    )
    parser.add_argument("--zip", default="data/input_data.zip", help="Path to input_data.zip")
    parser.add_argument("--out-dir", default="outputs/pilot_msas", help="Output directory.")
    parser.add_argument("--families", type=int, default=3, help="Number of EC families to pilot.")
    parser.add_argument(
        "--seqs-per-family",
        type=int,
        default=5,
        help="Maximum unique organism/gene IDs to fetch per family.",
    )
    parser.add_argument(
        "--scan-files",
        type=int,
        default=25,
        help="Select candidate families from the first N sorted enzyme files.",
    )
    parser.add_argument("--min-genes", type=int, default=2, help="Minimum genes per candidate family.")
    parser.add_argument("--max-genes", type=int, default=12, help="Preferred maximum genes per family.")
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="Polite delay after KEGG REST requests.",
    )
    parser.add_argument(
        "--kegg-root",
        default="",
        help="Licensed local KEGG root containing genes/organisms/<org>/<org>.pep files.",
    )
    parser.add_argument(
        "--sequence-fasta",
        action="append",
        default=[],
        help="Existing KEGG amino-acid FASTA to use before REST. May be repeated.",
    )
    parser.add_argument(
        "--fetch-missing",
        choices=["none", "rest"],
        default="rest",
        help="How to resolve entries missing from local FASTA/local KEGG sources.",
    )
    parser.add_argument(
        "--max-rest-requests",
        type=int,
        default=1000,
        help="Safety cap for uncached KEGG REST batch requests.",
    )
    parser.add_argument(
        "--allow-large-rest-run",
        action="store_true",
        help="Allow exceeding --max-rest-requests.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Select families but do not fetch sequences.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"Archive not found: {zip_path}")
    script_dir = Path(__file__).resolve().parent
    fetch_script = script_dir / "remap_kegg_sequences.py"
    build_script = script_dir / "build_msa.py"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    families = choose_families(
        zip_path=zip_path,
        scan_files=args.scan_files,
        family_count=args.families,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
    )
    if not families:
        raise SystemExit("No candidate families found with the requested thresholds.")

    print("Selected pilot EC families:", flush=True)
    for ec_number, gene_count in families:
        print(
            f"- {ec_number}: {gene_count} unique organism/gene IDs in first {args.scan_files} files",
            flush=True,
        )

    summary_path = out_dir / "pilot_summary.tsv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ec_number",
                "candidate_gene_count",
                "fasta",
                "sequence_index",
                "metadata",
                "msa",
                "fasta_records",
                "msa_built",
            ],
            delimiter="\t",
        )
        writer.writeheader()

        for ec_number, candidate_gene_count in families:
            stem = safe_ec_name(ec_number)
            fasta_path = out_dir / f"{stem}.fasta"
            metadata_path = out_dir / f"{stem}.metadata.tsv"
            index_path = out_dir / f"{stem}.sequence_index.tsv"
            msa_path = out_dir / f"{stem}.msa.fasta"

            fetch_command = [
                sys.executable,
                str(fetch_script),
                "--zip",
                str(zip_path),
                "--ec",
                ec_number,
                "--limit",
                str(args.seqs_per_family),
                "--max-enzyme-files",
                str(args.scan_files),
                "--sleep-seconds",
                str(args.sleep_seconds),
                "--out-fasta",
                str(fasta_path),
                "--out-index",
                str(index_path),
                "--out-metadata",
                str(metadata_path),
            ]
            for sequence_fasta in args.sequence_fasta:
                fetch_command.extend(["--sequence-fasta", sequence_fasta])
            if args.kegg_root:
                fetch_command.extend(["--kegg-root", args.kegg_root])
            fetch_missing = "none" if args.dry_run else args.fetch_missing
            fetch_command.extend(["--fetch-missing", fetch_missing])
            fetch_command.extend(["--max-rest-requests", str(args.max_rest_requests)])
            if args.allow_large_rest_run:
                fetch_command.append("--allow-large-rest-run")
            run_command(fetch_command)

            fasta_records = count_fasta_records(fasta_path)
            msa_built = "no"
            if fasta_records >= 1 and not args.dry_run:
                run_command([sys.executable, str(build_script), str(fasta_path), str(msa_path)])
                msa_built = "yes"
            elif args.dry_run:
                print(f"Dry run: skipped MSA build for {ec_number}.", flush=True)
            else:
                print(f"No FASTA records for {ec_number}; skipped MSA build.", flush=True)

            writer.writerow(
                {
                    "ec_number": ec_number,
                    "candidate_gene_count": candidate_gene_count,
                    "fasta": str(fasta_path),
                    "sequence_index": str(index_path),
                    "metadata": str(metadata_path),
                    "msa": str(msa_path) if msa_built == "yes" else "",
                    "fasta_records": fasta_records,
                    "msa_built": msa_built,
                }
            )

    print(f"Wrote pilot summary to {summary_path}.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
