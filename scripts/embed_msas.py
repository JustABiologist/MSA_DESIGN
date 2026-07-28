#!/usr/bin/env python3
"""Embed aligned MSAs with ESM MSA Transformer (ESM-MSA-1b)."""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
from pathlib import Path
from typing import Any


AMINO_ACID_OR_GAP_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ-")
DEFAULT_WEIGHTS = "weights/esm_msa1b_t12_100M_UR50S.pt"
DEFAULT_OUT_DIR = "outputs/embeddings"
CONTACT_REGRESSION_URL_TEMPLATE = (
    "https://dl.fbaipublicfiles.com/fair-esm/regression/"
    "{model_name}-contact-regression.pt"
)
POOLING_DESCRIPTION = {
    "token_embeddings": "Per-MSA-token representation aligned to cleaned MSA columns, shape rows x cols x hidden_dim.",
    "aa_mask": "Boolean rows x cols mask, true for non-gap cleaned MSA positions.",
    "gap_mask": "Boolean rows x cols mask, true for '-' gap positions.",
    "row_embeddings": "Mean token embedding per MSA row over aa_mask positions; all-gap rows are zero.",
    "col_embeddings": "Mean token embedding per MSA column over aa_mask positions; all-gap columns are zero.",
    "query_embedding": "Mean embedding for row 0 over non-gap positions; zero if row 0 has no amino-acid positions.",
    "msa_embedding": "Mean embedding over every non-gap token in the cropped MSA; zero if no amino-acid positions exist.",
}


class MSAFormatError(ValueError):
    """Raised when an MSA cannot be parsed or validated."""


def read_fasta_like(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_parts: list[str] = []
    opener = gzip.open if path.suffix == ".gz" else Path.open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(sequence_parts)))
                header = line[1:].strip() or f"record_{len(records) + 1}"
                sequence_parts = []
                continue
            if header is None:
                raise MSAFormatError(f"{path}:{line_number}: sequence encountered before first FASTA header")
            sequence_parts.append("".join(line.split()))
    if header is not None:
        records.append((header, "".join(sequence_parts)))
    if not records:
        raise MSAFormatError(f"No FASTA/A3M records found in {path}")
    return records


def clean_a3m_sequence(sequence: str, path: Path, header: str) -> str:
    cleaned: list[str] = []
    for char in sequence:
        if char.islower() or char == ".":
            continue
        if char in AMINO_ACID_OR_GAP_CHARS:
            cleaned.append(char)
            continue
        if char.isspace():
            continue
        raise MSAFormatError(
            f"Unsupported character {char!r} in {path} record {header!r}. "
            "Expected uppercase residue letters, '-' gaps, lowercase A3M insertions, or dots."
        )
    return "".join(cleaned)


def load_and_clean_msa(path: Path) -> dict[str, Any]:
    raw_records = read_fasta_like(path)
    cleaned_records: list[tuple[str, str]] = []
    for header, sequence in raw_records:
        cleaned = clean_a3m_sequence(sequence, path, header)
        if not cleaned:
            raise MSAFormatError(f"Record {header!r} in {path} is empty after A3M cleanup")
        cleaned_records.append((header, cleaned))

    lengths = {len(sequence) for _, sequence in cleaned_records}
    if len(lengths) != 1:
        examples = ", ".join(str(length) for length in sorted(lengths)[:8])
        raise MSAFormatError(
            f"Cleaned sequences in {path} are not aligned to one length; observed lengths: {examples}"
        )

    rows = len(cleaned_records)
    cols = len(cleaned_records[0][1])
    if rows < 1 or cols < 1:
        raise MSAFormatError(f"MSA {path} has invalid shape rows={rows}, cols={cols}")
    return {
        "headers": [header for header, _ in cleaned_records],
        "sequences": [sequence for _, sequence in cleaned_records],
        "original_rows": rows,
        "original_cols": cols,
    }


def crop_msa(msa: dict[str, Any], max_seqs: int, max_cols: int) -> tuple[dict[str, Any], list[str]]:
    if max_seqs < 1:
        raise ValueError("--max-seqs must be >= 1")
    if max_cols < 1:
        raise ValueError("--max-cols must be >= 1")

    headers = list(msa["headers"])
    sequences = list(msa["sequences"])
    warnings: list[str] = []
    if len(sequences) > max_seqs:
        warnings.append(f"cropped rows from {len(sequences)} to first {max_seqs}")
        headers = headers[:max_seqs]
        sequences = sequences[:max_seqs]
    original_cols = len(sequences[0])
    if original_cols > max_cols:
        warnings.append(f"cropped columns from {original_cols} to first {max_cols}")
        sequences = [sequence[:max_cols] for sequence in sequences]

    return {
        "headers": headers,
        "sequences": sequences,
        "original_rows": msa["original_rows"],
        "original_cols": msa["original_cols"],
        "rows": len(sequences),
        "cols": len(sequences[0]),
    }, warnings


def contact_regression_path(weights_path: Path) -> Path:
    return Path(str(weights_path.with_suffix("")) + "-contact-regression.pt")


def output_stem(msa_path: Path) -> str:
    name = msa_path.name
    if name.endswith(".gz"):
        name = name[:-3]
    for suffix in (".trimmed.afa", ".msa.fasta", ".msa.fa", ".a3m", ".fasta", ".fa", ".fas", ".afa"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def find_msa_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for item in args.msa or []:
        paths.append(Path(item))
    for pattern in args.msa_glob or []:
        matches = sorted(Path(match) for match in glob.glob(pattern))
        if not matches:
            raise SystemExit(f"No MSAs matched --msa-glob pattern: {pattern}")
        paths.extend(matches)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    if not unique:
        raise SystemExit("Provide at least one --msa PATH or --msa-glob PATTERN")
    missing = [str(path) for path in unique if not path.exists()]
    if missing:
        raise SystemExit("MSA file(s) not found: " + ", ".join(missing))
    return unique


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def base_metadata(
    msa_path: Path,
    cropped: dict[str, Any],
    warnings: list[str],
    args: argparse.Namespace,
    npz_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    return {
        "source_msa": str(msa_path),
        "output_npz": str(npz_path),
        "output_metadata": str(metadata_path),
        "weights": str(Path(args.weights)),
        "contact_regression_weights": str(contact_regression_path(Path(args.weights))),
        "layer": args.layer,
        "requested_device": args.device,
        "dtype": args.dtype,
        "max_seqs": args.max_seqs,
        "max_cols": args.max_cols,
        "original_shape": {
            "rows": cropped["original_rows"],
            "cols": cropped["original_cols"],
        },
        "shape": {
            "rows": cropped["rows"],
            "cols": cropped["cols"],
        },
        "cropped": bool(warnings),
        "warnings": warnings,
        "headers": cropped["headers"],
        "cleaned_sequences": cropped["sequences"],
        "pooling": POOLING_DESCRIPTION,
        "stores_token_embeddings": bool(args.include_token_embeddings and not args.pool_only),
    }


def dry_run_one(msa_path: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    msa = load_and_clean_msa(msa_path)
    cropped, warnings = crop_msa(msa, args.max_seqs, args.max_cols)
    stem = output_stem(msa_path)
    npz_path = out_dir / f"{stem}.npz"
    metadata_path = out_dir / f"{stem}.metadata.json"
    metadata = base_metadata(msa_path, cropped, warnings, args, npz_path, metadata_path)
    metadata.update(
        {
            "status": "dry_run",
            "planned_outputs": ["metadata_json", "npz_when_not_dry_run"],
            "note": "Dry-run parsed, cleaned, validated, and cropped the MSA without importing torch or esm.",
        }
    )
    write_metadata(metadata_path, metadata)
    print(
        f"DRY-RUN {msa_path}: rows={cropped['rows']} cols={cropped['cols']} "
        f"metadata={metadata_path} planned_npz={npz_path}",
        flush=True,
    )
    for warning in warnings:
        print(f"  warning: {warning}", flush=True)
    return metadata


def require_runtime_modules() -> tuple[Any, Any, Any]:
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency: numpy. Install numpy in the embedding environment.") from exc
    try:
        import torch  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: torch. Install PyTorch in the embedding environment, then rerun without --dry-run."
        ) from exc
    try:
        import esm  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: esm/fair-esm. Install fair-esm in the embedding environment, then rerun without --dry-run."
        ) from exc
    return np, torch, esm


def choose_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def masks_from_sequences(np: Any, sequences: list[str]) -> tuple[Any, Any]:
    char_rows = [list(sequence) for sequence in sequences]
    chars = np.array(char_rows, dtype="U1")
    gap_mask = chars == "-"
    aa_mask = ~gap_mask
    return aa_mask, gap_mask


def masked_mean(np: Any, values: Any, mask: Any, axis: int) -> tuple[Any, Any]:
    counts = mask.sum(axis=axis).astype(np.int64)
    expanded_mask = np.expand_dims(mask, axis=-1)
    sums = (values * expanded_mask).sum(axis=axis)
    output = np.zeros_like(sums, dtype=values.dtype)
    nonzero = counts > 0
    if np.any(nonzero):
        output[nonzero] = sums[nonzero] / counts[nonzero, None]
    return output, counts


def global_masked_mean(np: Any, values: Any, mask: Any) -> tuple[Any, int]:
    count = int(mask.sum())
    if count == 0:
        return np.zeros((values.shape[-1],), dtype=values.dtype), 0
    return (values * mask[..., None]).sum(axis=(0, 1)) / count, count


def query_masked_mean(np: Any, values: Any, mask: Any) -> tuple[Any, int]:
    count = int(mask[0].sum())
    if count == 0:
        return np.zeros((values.shape[-1],), dtype=values.dtype), 0
    return (values[0] * mask[0, :, None]).sum(axis=0) / count, count


def aligned_representations(torch: Any, results: dict[str, Any], layer: int, rows: int, cols: int) -> Any:
    if "representations" not in results or layer not in results["representations"]:
        available = sorted(results.get("representations", {}).keys())
        raise RuntimeError(f"Layer {layer} not present in model results; available layers: {available}")
    representations = results["representations"][layer]
    if representations.ndim != 4 or representations.shape[0] != 1:
        raise RuntimeError(f"Unexpected representation shape: {tuple(representations.shape)}")
    token_repr = representations[0]
    if token_repr.shape[0] != rows:
        raise RuntimeError(f"Expected {rows} MSA rows in representation, got {token_repr.shape[0]}")
    if token_repr.shape[1] == cols + 1:
        return token_repr[:, 1:, :]
    if token_repr.shape[1] == cols + 2:
        return token_repr[:, 1:-1, :]
    if token_repr.shape[1] == cols:
        return token_repr
    raise RuntimeError(
        f"Cannot align representation length {token_repr.shape[1]} with cleaned MSA columns {cols}"
    )


def embed_one(
    msa_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
    np: Any,
    torch: Any,
    model: Any,
    alphabet: Any,
    device: Any,
) -> dict[str, Any]:
    msa = load_and_clean_msa(msa_path)
    cropped, warnings = crop_msa(msa, args.max_seqs, args.max_cols)
    rows = cropped["rows"]
    cols = cropped["cols"]
    stem = output_stem(msa_path)
    npz_path = out_dir / f"{stem}.npz"
    metadata_path = out_dir / f"{stem}.metadata.json"
    metadata = base_metadata(msa_path, cropped, warnings, args, npz_path, metadata_path)

    batch_converter = alphabet.get_batch_converter()
    msa_records = list(zip(cropped["headers"], cropped["sequences"]))
    _, _, tokens = batch_converter([msa_records])
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[args.layer], return_contacts=False)
        token_repr = aligned_representations(torch, results, args.layer, rows, cols)
        token_embeddings = token_repr.detach().cpu().numpy()

    if args.dtype == "float16":
        token_embeddings = token_embeddings.astype(np.float16)
    else:
        token_embeddings = token_embeddings.astype(np.float32)

    aa_mask, gap_mask = masks_from_sequences(np, cropped["sequences"])
    row_embeddings, row_counts = masked_mean(np, token_embeddings, aa_mask, axis=1)
    col_embeddings, col_counts = masked_mean(np, token_embeddings, aa_mask, axis=0)
    query_embedding, query_count = query_masked_mean(np, token_embeddings, aa_mask)
    msa_embedding, msa_count = global_masked_mean(np, token_embeddings, aa_mask)

    arrays: dict[str, Any] = {
        "query_embedding": query_embedding,
        "msa_embedding": msa_embedding,
        "row_embeddings": row_embeddings,
        "col_embeddings": col_embeddings,
        "aa_mask": aa_mask,
        "gap_mask": gap_mask,
        "row_aa_counts": row_counts,
        "col_aa_counts": col_counts,
    }
    if args.include_token_embeddings and not args.pool_only:
        arrays["token_embeddings"] = token_embeddings
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)

    metadata.update(
        {
            "status": "embedded",
            "device": str(device),
            "model_class": model.__class__.__name__,
            "hidden_dim": int(token_embeddings.shape[-1]),
            "token_embedding_shape": list(token_embeddings.shape),
            "row_aa_counts": row_counts.astype(int).tolist(),
            "col_aa_counts": col_counts.astype(int).tolist(),
            "query_aa_count": query_count,
            "msa_aa_count": msa_count,
            "npz_arrays": {name: list(value.shape) for name, value in arrays.items()},
        }
    )
    write_metadata(metadata_path, metadata)
    print(
        f"EMBEDDED {msa_path}: rows={rows} cols={cols} hidden={token_embeddings.shape[-1]} "
        f"npz={npz_path} metadata={metadata_path}",
        flush=True,
    )
    for warning in warnings:
        print(f"  warning: {warning}", flush=True)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msa", action="append", default=[], help="Aligned FASTA/A3M MSA path. Repeatable.")
    parser.add_argument(
        "--msa-glob",
        action="append",
        default=[],
        help="Glob for aligned FASTA/A3M MSA paths. Repeatable, e.g. 'outputs/pilot_msas/*.msa.fasta'.",
    )
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output embedding directory.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Local ESM-MSA-1b weights path.")
    parser.add_argument("--layer", type=int, default=12, help="Representation layer to extract.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Torch device.")
    parser.add_argument("--max-seqs", type=int, default=64, help="Maximum MSA rows to embed.")
    parser.add_argument("--max-cols", type=int, default=1024, help="Maximum aligned columns to embed.")
    parser.add_argument(
        "--include-token-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store full rows x cols x hidden token embeddings in the NPZ. Use --no-include-token-embeddings to omit.",
    )
    parser.add_argument("--pool-only", action="store_true", help="Only store pooled outputs and masks, not token_embeddings.")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32", help="Output embedding dtype.")
    parser.add_argument("--dry-run", action="store_true", help="Parse, clean, crop, and write metadata without torch/esm.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    msa_paths = find_msa_paths(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for msa_path in msa_paths:
            dry_run_one(msa_path, out_dir, args)
        return 0

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"Weights not found: {weights_path}. Expected local ESM-MSA-1b weights or pass --weights.")
    regression_path = contact_regression_path(weights_path)
    if not regression_path.exists():
        model_name = weights_path.stem
        url = CONTACT_REGRESSION_URL_TEMPLATE.format(model_name=model_name)
        raise SystemExit(
            "fair-esm local loading expects the contact-regression sidecar next to the model weights, "
            f"even for representation extraction. Missing: {regression_path}. Download: {url}"
        )

    np, torch, esm = require_runtime_modules()
    device = choose_device(torch, args.device)
    print(f"Loading ESM MSA Transformer weights from {weights_path} on {device}...", flush=True)
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weights_path))
    model.eval()
    model.to(device)

    for msa_path in msa_paths:
        embed_one(msa_path, out_dir, args, np, torch, model, alphabet, device)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MSAFormatError as exc:
        raise SystemExit(f"MSA format error: {exc}") from exc
