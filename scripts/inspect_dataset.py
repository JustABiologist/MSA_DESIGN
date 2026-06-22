#!/usr/bin/env python3
"""Inspect the Zenodo enzyme dataset archive without extracting it."""

from __future__ import annotations

import argparse
import math
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Iterator


ENZYME_PREFIX = "input_data/enzymes/"
SUPPLEMENTARY_PREFIX = "input_data/supplementary/"
EXPECTED_ENZYME_COLUMNS = 11
ENZYME_COLUMN_NAMES = [
    "gene_id",
    "organism_code",
    "domain",
    "reaction_id",
    "ec_numbers",
    "compound_id",
    "numeric_col_7_unlabeled",
    "numeric_col_8_unlabeled",
    "numeric_col_9_unlabeled",
    "numeric_col_10_unlabeled",
    "numeric_col_11_unlabeled",
]
NUMERIC_COLUMN_INDEXES = range(6, 11)
REACTION_RE = re.compile(r"^R\d{5}$")
COMPOUND_RE = re.compile(r"^[CG]\d{5}$")
EC_RE = re.compile(r"^\d+\.(?:\d+|-)\.(?:\d+|-)\.(?:\d+|-)$")


class NumericStats:
    def __init__(self) -> None:
        self.total = 0
        self.missing = 0
        self.non_numeric = 0
        self.finite = 0
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.sum = 0.0

    def observe(self, raw_value: str) -> None:
        self.total += 1
        value = raw_value.strip()
        if not value or value.lower() == "nan":
            self.missing += 1
            return
        try:
            parsed = float(value)
        except ValueError:
            self.non_numeric += 1
            return
        if math.isnan(parsed):
            self.missing += 1
            return
        if not math.isfinite(parsed):
            self.non_numeric += 1
            return
        self.finite += 1
        self.sum += parsed
        self.minimum = parsed if self.minimum is None else min(self.minimum, parsed)
        self.maximum = parsed if self.maximum is None else max(self.maximum, parsed)

    @property
    def mean(self) -> float | None:
        if not self.finite:
            return None
        return self.sum / self.finite


def fmt_int(value: int) -> str:
    return f"{value:,}"


def fmt_size(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} GiB"


def fmt_float(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4g}"


def read_header_from_zip(zf: zipfile.ZipFile, name: str) -> list[str]:
    with zf.open(name) as handle:
        header = handle.readline().decode("utf-8", "replace").rstrip("\n\r")
    return header.split("\t") if header else []


def read_domain_codes(zf: zipfile.ZipFile) -> set[str]:
    name = f"{SUPPLEMENTARY_PREFIX}domain.txt"
    codes: set[str] = set()
    if name not in zf.namelist():
        return codes
    with zf.open(name) as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            if line_no == 1:
                continue
            line = raw_line.decode("utf-8", "replace").rstrip("\n\r")
            if not line:
                continue
            codes.add(line.split("\t", 1)[0])
    return codes


def split_ec_numbers(raw_ecs: str) -> list[str]:
    return [part.strip() for part in raw_ecs.split(";") if part.strip()]


def iter_enzyme_lines(
    zf: zipfile.ZipFile,
    enzyme_names: list[str],
) -> Iterator[tuple[str, int, list[str]]]:
    for name in enzyme_names:
        with zf.open(name) as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.decode("utf-8", "replace").rstrip("\n\r")
                if not line:
                    continue
                yield name, line_no, line.split("\t")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize input_data.zip without extracting the archive."
    )
    parser.add_argument("--zip", default="data/input_data.zip", help="Path to input_data.zip")
    parser.add_argument(
        "--max-enzyme-files",
        type=int,
        default=None,
        help="Inspect only the first N sorted enzyme files for a faster smoke run.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=10,
        help="Number of enzyme rows to print as examples.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"Archive not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        enzyme_infos = [
            info
            for info in infos
            if info.filename.startswith(ENZYME_PREFIX)
            and info.filename.endswith(".txt")
            and not info.is_dir()
        ]
        supplementary_infos = [
            info
            for info in infos
            if info.filename.startswith(SUPPLEMENTARY_PREFIX)
            and info.filename.endswith(".txt")
            and not info.is_dir()
        ]
        enzyme_names = sorted(info.filename for info in enzyme_infos)
        if args.max_enzyme_files is not None:
            enzyme_names = enzyme_names[: args.max_enzyme_files]

        print(f"Archive: {zip_path}")
        print(f"Zip entries: {fmt_int(len(infos))}")
        print(
            "Enzyme files: "
            f"{fmt_int(len(enzyme_infos))} "
            f"({fmt_size(sum(info.file_size for info in enzyme_infos))} uncompressed)"
        )
        print(
            "Supplementary files: "
            f"{fmt_int(len(supplementary_infos))} "
            f"({fmt_size(sum(info.file_size for info in supplementary_infos))} uncompressed)"
        )
        if args.max_enzyme_files is not None:
            print(f"Inspecting first sorted enzyme files only: {fmt_int(len(enzyme_names))}")

        print("\nSupplementary headers:")
        for info in sorted(supplementary_infos, key=lambda item: item.filename):
            header = read_header_from_zip(zf, info.filename)
            print(
                f"- {info.filename}: {fmt_size(info.file_size)}; "
                f"columns={len(header)}; header={header}"
            )

        valid_domains = read_domain_codes(zf)
        row_count = 0
        file_counts: Counter[str] = Counter()
        field_counts: Counter[int] = Counter()
        numeric_stats = {idx: NumericStats() for idx in NUMERIC_COLUMN_INDEXES}
        samples: list[tuple[str, int, list[str]]] = []
        file_code_mismatches: list[str] = []
        unknown_domains: list[str] = []
        bad_reactions: list[str] = []
        bad_compounds: list[str] = []
        bad_ecs: list[str] = []

        for filename, line_no, parts in iter_enzyme_lines(zf, enzyme_names):
            row_count += 1
            file_counts[filename] += 1
            field_counts[len(parts)] += 1
            if len(samples) < args.sample_rows:
                samples.append((filename, line_no, parts))
            if len(parts) != EXPECTED_ENZYME_COLUMNS:
                continue

            expected_code = Path(filename).stem
            if parts[1] != expected_code and len(file_code_mismatches) < 5:
                file_code_mismatches.append(
                    f"{filename}:{line_no} has organism_code={parts[1]!r}"
                )
            if valid_domains and parts[2] not in valid_domains and len(unknown_domains) < 5:
                unknown_domains.append(f"{filename}:{line_no} has domain={parts[2]!r}")
            if not REACTION_RE.match(parts[3]) and len(bad_reactions) < 5:
                bad_reactions.append(f"{filename}:{line_no} has reaction_id={parts[3]!r}")
            if not COMPOUND_RE.match(parts[5]) and len(bad_compounds) < 5:
                bad_compounds.append(f"{filename}:{line_no} has compound_id={parts[5]!r}")
            for ec_number in split_ec_numbers(parts[4]):
                if not EC_RE.match(ec_number) and len(bad_ecs) < 5:
                    bad_ecs.append(f"{filename}:{line_no} has ec_number={ec_number!r}")
                    break
            for idx in NUMERIC_COLUMN_INDEXES:
                numeric_stats[idx].observe(parts[idx])

        print("\nEnzyme row structure:")
        print(f"- Rows inspected: {fmt_int(row_count)}")
        print(f"- Expected tab-separated columns per row: {EXPECTED_ENZYME_COLUMNS}")
        print(f"- Observed field counts: {dict(sorted(field_counts.items()))}")
        if file_counts:
            counts = file_counts.values()
            print(
                "- Rows per inspected enzyme file: "
                f"min={fmt_int(min(counts))}, max={fmt_int(max(counts))}, "
                f"mean={sum(counts) / len(file_counts):.1f}"
            )

        print("\nSample enzyme rows:")
        for filename, line_no, parts in samples:
            print(f"- {filename}:{line_no}")
            for idx, value in enumerate(parts[:EXPECTED_ENZYME_COLUMNS]):
                print(f"  {idx + 1:02d} {ENZYME_COLUMN_NAMES[idx]} = {value}")
            if len(parts) > EXPECTED_ENZYME_COLUMNS:
                print(f"  extra_columns = {parts[EXPECTED_ENZYME_COLUMNS:]}")

        print("\nNumeric columns:")
        print("- Columns 7-11 are unlabeled in the enzyme files.")
        print(
            "- Current evidence only: columns 7-9 look kinetic-like because they vary by "
            "compound/reaction and have missing values; columns 10-11 look like temperature "
            "and pH*10 ranges, but this is unconfirmed."
        )
        for idx in NUMERIC_COLUMN_INDEXES:
            stats = numeric_stats[idx]
            missing_pct = (100.0 * stats.missing / stats.total) if stats.total else 0.0
            non_numeric_pct = (100.0 * stats.non_numeric / stats.total) if stats.total else 0.0
            print(
                f"- column {idx + 1}: total={fmt_int(stats.total)}, "
                f"finite={fmt_int(stats.finite)}, missing={fmt_int(stats.missing)} "
                f"({missing_pct:.2f}%), non_numeric={fmt_int(stats.non_numeric)} "
                f"({non_numeric_pct:.2f}%), min={fmt_float(stats.minimum)}, "
                f"max={fmt_float(stats.maximum)}, mean={fmt_float(stats.mean)}"
            )

        print("\nConsistency checks:")
        print(
            "- organism_code matches enzyme filename stem: "
            + ("PASS" if not file_code_mismatches else "CHECK")
        )
        for item in file_code_mismatches:
            print(f"  example: {item}")
        print(
            "- domain is present in supplementary/domain.txt: "
            + ("PASS" if not unknown_domains else "CHECK")
        )
        for item in unknown_domains:
            print(f"  example: {item}")
        print("- reaction IDs look like R00000: " + ("PASS" if not bad_reactions else "CHECK"))
        for item in bad_reactions:
            print(f"  example: {item}")
        print("- compound IDs look like KEGG C/G identifiers: " + ("PASS" if not bad_compounds else "CHECK"))
        for item in bad_compounds:
            print(f"  example: {item}")
        print("- EC fields parse as semicolon-separated EC numbers: " + ("PASS" if not bad_ecs else "CHECK"))
        for item in bad_ecs:
            print(f"  example: {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
