#!/usr/bin/env python3
"""Build an aligned FASTA from an input FASTA."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


MATCH_SCORE = 1
MISMATCH_SCORE = -1
GAP_SCORE = -1
FASTA_WRAP = 80


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_parts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence_parts)))
                header = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line)
    if header is not None:
        records.append((header, "".join(sequence_parts)))
    return records


def write_fasta(records: list[tuple[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n")
            for start in range(0, len(sequence), FASTA_WRAP):
                handle.write(sequence[start : start + FASTA_WRAP] + "\n")


def score_pair(left: str, right: str) -> int:
    if left == "-" or right == "-":
        return GAP_SCORE
    if left == right:
        return MATCH_SCORE
    return MISMATCH_SCORE


def needleman_wunsch(left: str, right: str) -> tuple[str, str]:
    """Global alignment with deterministic diag/up/left tie-breaking."""
    rows = len(left) + 1
    cols = len(right) + 1
    scores = [[0] * cols for _ in range(rows)]
    moves = [[""] * cols for _ in range(rows)]

    for i in range(1, rows):
        scores[i][0] = scores[i - 1][0] + GAP_SCORE
        moves[i][0] = "U"
    for j in range(1, cols):
        scores[0][j] = scores[0][j - 1] + GAP_SCORE
        moves[0][j] = "L"

    for i in range(1, rows):
        left_char = left[i - 1]
        for j in range(1, cols):
            right_char = right[j - 1]
            diag = scores[i - 1][j - 1] + score_pair(left_char, right_char)
            up = scores[i - 1][j] + GAP_SCORE
            left_score = scores[i][j - 1] + GAP_SCORE
            best = max(diag, up, left_score)
            scores[i][j] = best
            if diag == best:
                moves[i][j] = "D"
            elif up == best:
                moves[i][j] = "U"
            else:
                moves[i][j] = "L"

    aligned_left: list[str] = []
    aligned_right: list[str] = []
    i = len(left)
    j = len(right)
    while i > 0 or j > 0:
        move = moves[i][j]
        if move == "D":
            aligned_left.append(left[i - 1])
            aligned_right.append(right[j - 1])
            i -= 1
            j -= 1
        elif move == "U":
            aligned_left.append(left[i - 1])
            aligned_right.append("-")
            i -= 1
        else:
            aligned_left.append("-")
            aligned_right.append(right[j - 1])
            j -= 1

    return "".join(reversed(aligned_left)), "".join(reversed(aligned_right))


def alignment_identity(left: str, right: str) -> float:
    aligned_left, aligned_right = needleman_wunsch(left, right)
    comparable = 0
    matches = 0
    for left_char, right_char in zip(aligned_left, aligned_right):
        if left_char == "-" or right_char == "-":
            continue
        comparable += 1
        if left_char == right_char:
            matches += 1
    if comparable == 0:
        return 0.0
    return matches / comparable


def choose_center(records: list[tuple[str, str]]) -> int:
    if len(records) == 1:
        return 0
    scores: list[tuple[float, int, int]] = []
    for idx, (_, sequence) in enumerate(records):
        total = 0.0
        for other_idx, (_, other_sequence) in enumerate(records):
            if idx == other_idx:
                continue
            total += alignment_identity(sequence, other_sequence)
        scores.append((total, len(sequence), -idx))
    best = max(scores)
    return -best[2]


def decompose_pair_alignment(center_alignment: str, sequence_alignment: str) -> tuple[list[list[str]], list[str]]:
    center_length = len(center_alignment.replace("-", ""))
    insertions: list[list[str]] = [[] for _ in range(center_length + 1)]
    residue_chars = ["-"] * center_length
    center_pos = 0
    for center_char, sequence_char in zip(center_alignment, sequence_alignment):
        if center_char == "-":
            insertions[center_pos].append(sequence_char)
        else:
            if center_pos < center_length:
                residue_chars[center_pos] = sequence_char
            center_pos += 1
    return insertions, residue_chars


def center_star_align(records: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if len(records) <= 1:
        return records

    center_index = choose_center(records)
    center_header, center_sequence = records[center_index]
    decomposed: list[tuple[int, str, list[list[str]], list[str]]] = []
    max_insertions = [0] * (len(center_sequence) + 1)

    for original_index, (header, sequence) in enumerate(records):
        if original_index == center_index:
            insertions = [[] for _ in range(len(center_sequence) + 1)]
            residue_chars = list(center_sequence)
        else:
            aligned_center, aligned_sequence = needleman_wunsch(center_sequence, sequence)
            insertions, residue_chars = decompose_pair_alignment(aligned_center, aligned_sequence)
        for pos, chars in enumerate(insertions):
            max_insertions[pos] = max(max_insertions[pos], len(chars))
        decomposed.append((original_index, header, insertions, residue_chars))

    aligned_by_index: list[tuple[int, str, str]] = []
    for original_index, header, insertions, residue_chars in decomposed:
        aligned_parts: list[str] = []
        for pos in range(len(center_sequence) + 1):
            chars = insertions[pos]
            aligned_parts.extend(chars)
            aligned_parts.extend("-" for _ in range(max_insertions[pos] - len(chars)))
            if pos < len(center_sequence):
                aligned_parts.append(residue_chars[pos])
        aligned_by_index.append((original_index, header, "".join(aligned_parts)))

    return [(header, sequence) for _, header, sequence in sorted(aligned_by_index)]


def run_mafft(input_fasta: Path, output_fasta: Path) -> None:
    mafft = shutil.which("mafft")
    if mafft is None:
        raise RuntimeError("mafft executable was not found on PATH")
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [mafft, "--auto", str(input_fasta)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    if result.returncode != 0:
        raise RuntimeError(f"mafft failed with exit code {result.returncode}")
    output_fasta.write_text(result.stdout, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write an aligned FASTA from an input FASTA.")
    parser.add_argument("input_fasta", help="Input FASTA path.")
    parser.add_argument("output_fasta", help="Output aligned FASTA path.")
    parser.add_argument(
        "--method",
        choices=["auto", "mafft", "fallback"],
        default="auto",
        help="Alignment method. auto prefers mafft and falls back to pure Python.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_fasta = Path(args.input_fasta)
    output_fasta = Path(args.output_fasta)
    if not input_fasta.exists():
        raise SystemExit(f"Input FASTA not found: {input_fasta}")

    mafft_path = shutil.which("mafft")
    if args.method == "mafft" and mafft_path is None:
        raise SystemExit("mafft requested but not found on PATH")
    use_mafft = args.method == "mafft" or (args.method == "auto" and mafft_path is not None)
    if use_mafft:
        run_mafft(input_fasta, output_fasta)
        print(f"Wrote MAFFT alignment to {output_fasta}.")
        return 0

    records = read_fasta(input_fasta)
    if not records:
        raise SystemExit(f"No FASTA records found in {input_fasta}")
    print(
        "WARNING: mafft not available; using deterministic pure-Python center-star "
        "Needleman-Wunsch fallback. This is sufficient for small pilot MSAs only and is "
        "not production-quality.",
        file=sys.stderr,
    )
    aligned = center_star_align(records)
    write_fasta(aligned, output_fasta)
    print(f"Wrote fallback alignment with {len(aligned)} sequences to {output_fasta}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
