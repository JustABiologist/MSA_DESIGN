#!/usr/bin/env python3
"""Precompute reusable ESM-MSA embeddings for the large training-MSA manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from embed_msas import (  # noqa: E402
    DEFAULT_WEIGHTS,
    MSAFormatError,
    base_metadata,
    choose_device,
    contact_regression_path,
    crop_msa,
    dry_run_one,
    embed_one,
    load_and_clean_msa,
    output_stem,
    require_runtime_modules,
)


DEFAULT_TRAINING_ROOT = Path("/mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim")
DEFAULT_MSA_MANIFEST = DEFAULT_TRAINING_ROOT / "msa_manifest.tsv"
DEFAULT_OUT_DIR = DEFAULT_TRAINING_ROOT / "esm_msa_embeddings_col"


MANIFEST_FIELDS = [
    "cluster_index",
    "representative",
    "split",
    "status",
    "message",
    "source_msa",
    "npz_path",
    "metadata_path",
    "original_rows",
    "original_cols",
    "rows",
    "cols",
    "max_seqs",
    "max_cols",
    "dtype",
    "stores_token_embeddings",
    "elapsed_seconds",
]


def stable_split(cluster_index: str, seed: int, val_fraction: float, test_fraction: float) -> str:
    if val_fraction < 0.0 or test_fraction < 0.0 or val_fraction + test_fraction >= 1.0:
        raise ValueError("--val-fraction and --test-fraction must be non-negative and sum to < 1")
    digest = hashlib.sha256(f"{seed}:{cluster_index}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < test_fraction:
        return "test"
    if value < test_fraction + val_fraction:
        return "val"
    return "train"


def manifest_rows(path: Path, status: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if status and row.get("status") != status:
                continue
            trimmed = row.get("trimmed_alignment", "")
            if not trimmed:
                continue
            rows.append(row)
    return rows


def existing_manifest_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("cluster_index") and row.get("status") == "embedded":
                keys.add(row["cluster_index"])
    return keys


def metadata_shape(metadata_path: Path) -> dict[str, Any]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        "original_rows": payload.get("original_shape", {}).get("rows", ""),
        "original_cols": payload.get("original_shape", {}).get("cols", ""),
        "rows": payload.get("shape", {}).get("rows", ""),
        "cols": payload.get("shape", {}).get("cols", ""),
        "stores_token_embeddings": payload.get("stores_token_embeddings", ""),
    }


def planned_paths(msa_path: Path, out_dir: Path) -> tuple[Path, Path]:
    stem = output_stem(msa_path)
    return out_dir / f"{stem}.npz", out_dir / f"{stem}.metadata.json"


def append_manifest_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def embedding_args(args: argparse.Namespace, device_text: str) -> argparse.Namespace:
    return argparse.Namespace(
        weights=args.weights,
        layer=args.layer,
        device=device_text,
        dtype=args.dtype,
        max_seqs=args.max_seqs,
        max_cols=args.max_cols,
        include_token_embeddings=args.store_token_embeddings,
        pool_only=False,
    )


def dry_run_shape(msa_path: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    msa = load_and_clean_msa(msa_path)
    cropped, warnings = crop_msa(msa, args.max_seqs, args.max_cols)
    npz_path, metadata_path = planned_paths(msa_path, out_dir)
    metadata = base_metadata(msa_path, cropped, warnings, args, npz_path, metadata_path)
    return {
        "metadata": metadata,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msa-manifest", default=str(DEFAULT_MSA_MANIFEST), help="Training MSA manifest TSV.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Embedding output directory.")
    parser.add_argument(
        "--embedding-manifest",
        default=None,
        help="Output TSV manifest. Default: <out-dir>/embedding_manifest.tsv.",
    )
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Local ESM-MSA-1b weights path.")
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-seqs", type=int, default=64, help="Maximum rows per MSA sent through ESM-MSA.")
    parser.add_argument(
        "--max-cols",
        type=int,
        default=1023,
        help="Maximum cleaned MSA columns sent through ESM-MSA. ESM-MSA adds a special token, so 1023 keeps the model input length <= 1024.",
    )
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    parser.add_argument(
        "--store-token-embeddings",
        action="store_true",
        help="Store full rows x cols x hidden token embeddings. Default stores pooled row/column/global embeddings only.",
    )
    parser.add_argument("--status", default="ok", help="Input MSA manifest status to process.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of manifest rows to process.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based filtered manifest row offset.")
    parser.add_argument("--end-index", type=int, default=None, help="Exclusive filtered manifest row offset.")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    msa_manifest = Path(args.msa_manifest)
    out_dir = Path(args.out_dir)
    embedding_manifest = Path(args.embedding_manifest) if args.embedding_manifest else out_dir / "embedding_manifest.tsv"
    if not msa_manifest.exists():
        raise SystemExit(f"MSA manifest not found: {msa_manifest}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = manifest_rows(msa_manifest, args.status)
    if args.end_index is not None:
        rows = rows[: args.end_index]
    rows = rows[args.start_index :]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No input MSA manifest rows selected")

    known_done = existing_manifest_keys(embedding_manifest)
    run_args = embedding_args(args, args.device)

    np = torch = esm = model = alphabet = device = None
    if not args.dry_run:
        weights_path = Path(args.weights)
        if not weights_path.exists():
            raise SystemExit(f"Weights not found: {weights_path}")
        regression_path = contact_regression_path(weights_path)
        if not regression_path.exists():
            raise SystemExit(f"Missing contact-regression sidecar expected by fair-esm: {regression_path}")
        np, torch, esm = require_runtime_modules()
        device = choose_device(torch, args.device)
        run_args.device = str(device)
        print(f"Loading ESM-MSA from {weights_path} on {device}", flush=True)
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weights_path))
        model.eval().to(device)

    started = time.monotonic()
    embedded = 0
    skipped = 0
    failed = 0
    for selected_index, row in enumerate(rows, start=1):
        cluster_index = row["cluster_index"]
        representative = row.get("representative", "")
        split = stable_split(cluster_index, args.seed, args.val_fraction, args.test_fraction)
        msa_path = Path(row["trimmed_alignment"])
        npz_path, metadata_path = planned_paths(msa_path, out_dir)
        item_started = time.monotonic()
        if args.skip_existing and cluster_index in known_done and npz_path.exists() and metadata_path.exists():
            skipped += 1
            continue
        if args.skip_existing and npz_path.exists() and metadata_path.exists():
            shape = metadata_shape(metadata_path)
            append_manifest_row(
                embedding_manifest,
                {
                    "cluster_index": cluster_index,
                    "representative": representative,
                    "split": split,
                    "status": "embedded",
                    "message": "preexisting",
                    "source_msa": str(msa_path),
                    "npz_path": str(npz_path),
                    "metadata_path": str(metadata_path),
                    "max_seqs": args.max_seqs,
                    "max_cols": args.max_cols,
                    "dtype": args.dtype,
                    "elapsed_seconds": f"{time.monotonic() - item_started:.3f}",
                    **shape,
                },
            )
            known_done.add(cluster_index)
            skipped += 1
            continue

        try:
            if args.dry_run:
                dry_run_one(msa_path, out_dir, run_args)
                shape = metadata_shape(metadata_path)
                row_status = "dry_run"
            else:
                assert np is not None and torch is not None and model is not None and alphabet is not None and device is not None
                embed_one(msa_path, out_dir, run_args, np, torch, model, alphabet, device)
                shape = metadata_shape(metadata_path)
                row_status = "embedded"
            append_manifest_row(
                embedding_manifest,
                {
                    "cluster_index": cluster_index,
                    "representative": representative,
                    "split": split,
                    "status": row_status,
                    "source_msa": str(msa_path),
                    "npz_path": str(npz_path),
                    "metadata_path": str(metadata_path),
                    "max_seqs": args.max_seqs,
                    "max_cols": args.max_cols,
                    "dtype": args.dtype,
                    "elapsed_seconds": f"{time.monotonic() - item_started:.3f}",
                    **shape,
                },
            )
            known_done.add(cluster_index)
            embedded += 1
        except (MSAFormatError, RuntimeError, OSError, ValueError) as exc:
            failed += 1
            append_manifest_row(
                embedding_manifest,
                {
                    "cluster_index": cluster_index,
                    "representative": representative,
                    "split": split,
                    "status": "failed",
                    "message": str(exc),
                    "source_msa": str(msa_path),
                    "npz_path": str(npz_path),
                    "metadata_path": str(metadata_path),
                    "max_seqs": args.max_seqs,
                    "max_cols": args.max_cols,
                    "dtype": args.dtype,
                    "stores_token_embeddings": bool(args.store_token_embeddings),
                    "elapsed_seconds": f"{time.monotonic() - item_started:.3f}",
                },
            )
            print(f"FAILED cluster={cluster_index} msa={msa_path}: {exc}", file=sys.stderr, flush=True)

        if args.progress_every > 0 and selected_index % args.progress_every == 0:
            elapsed = time.monotonic() - started
            print(
                f"processed={selected_index}/{len(rows)} embedded={embedded} skipped={skipped} "
                f"failed={failed} elapsed_seconds={elapsed:.1f}",
                flush=True,
            )

    elapsed = time.monotonic() - started
    print(
        f"Done embedding manifest rows={len(rows)} embedded={embedded} skipped={skipped} "
        f"failed={failed} manifest={embedding_manifest} elapsed_seconds={elapsed:.1f}",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
