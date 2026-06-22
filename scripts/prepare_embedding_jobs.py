#!/usr/bin/env python3
"""Prepare simple embedding commands or a manifest for MSA Transformer jobs."""

from __future__ import annotations

import argparse
import csv
import glob
import shlex
import sys
from pathlib import Path


DEFAULT_MSA_GLOB = "outputs/pilot_msas/*.msa.fasta"
DEFAULT_OUT_DIR = "outputs/embeddings"
DEFAULT_WEIGHTS = "weights/esm_msa1b_t12_100M_UR50S.pt"


def output_stem(msa_path: Path) -> str:
    name = msa_path.name
    for suffix in (".msa.fasta", ".msa.fa", ".a3m", ".fasta", ".fa", ".fas"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return msa_path.stem


def find_msas(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(match) for match in glob.glob(pattern))
        if not matches:
            print(f"warning: no MSAs matched {pattern!r}", file=sys.stderr)
        paths.extend(matches)
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def build_command(msa_path: Path, args: argparse.Namespace) -> str:
    cmd = [
        sys.executable,
        "scripts/embed_msas.py",
        "--msa",
        str(msa_path),
        "--out-dir",
        args.out_dir,
        "--weights",
        args.weights,
        "--layer",
        str(args.layer),
        "--device",
        args.device,
        "--max-seqs",
        str(args.max_seqs),
        "--max-cols",
        str(args.max_cols),
        "--dtype",
        args.dtype,
    ]
    if args.pool_only:
        cmd.append("--pool-only")
    if not args.include_token_embeddings:
        cmd.append("--no-include-token-embeddings")
    if args.dry_run:
        cmd.append("--dry-run")
    return " ".join(shlex.quote(part) for part in cmd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--msa-glob",
        action="append",
        default=None,
        help=f"MSA glob to include. Repeatable. Default: {DEFAULT_MSA_GLOB}",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Embedding output directory.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Local ESM-MSA-1b weights path.")
    parser.add_argument("--layer", type=int, default=12, help="Representation layer.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Torch device.")
    parser.add_argument("--max-seqs", type=int, default=64, help="Maximum MSA rows.")
    parser.add_argument("--max-cols", type=int, default=1024, help="Maximum MSA columns.")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32", help="Output dtype.")
    parser.add_argument("--pool-only", action="store_true", help="Only store pooled embedding arrays.")
    parser.add_argument(
        "--include-token-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include full token embeddings unless --pool-only is used.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Include --dry-run in generated commands.")
    parser.add_argument("--out-manifest", help="Optional TSV manifest path to write.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    msa_patterns = args.msa_glob or [DEFAULT_MSA_GLOB]
    msa_paths = find_msas(msa_patterns)
    if not msa_paths:
        raise SystemExit("No MSA files found")

    rows: list[dict[str, str]] = []
    for msa_path in msa_paths:
        stem = output_stem(msa_path)
        rows.append(
            {
                "msa_path": str(msa_path),
                "output_stem": stem,
                "npz_path": str(Path(args.out_dir) / f"{stem}.npz"),
                "metadata_path": str(Path(args.out_dir) / f"{stem}.metadata.json"),
                "command": build_command(msa_path, args),
            }
        )

    if args.out_manifest:
        manifest_path = Path(args.out_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["msa_path", "output_stem", "npz_path", "metadata_path", "command"],
                delimiter="\t",
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} embedding jobs to {manifest_path}")
    else:
        for row in rows:
            print(row["command"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
