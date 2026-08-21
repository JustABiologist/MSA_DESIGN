#!/usr/bin/env python3
"""Generate baseline and thermostable variants for multiple MSA families."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_thermostability_contrast as contrast  # noqa: E402


DEFAULT_ROOT = Path("/mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim")
DEFAULT_CHECKPOINT = (
    DEFAULT_ROOT
    / "mean_start_ccdd_full_profile_row_bs32_from38000_20260718_161831"
    / "mean_start_ccdd.latest.pt"
)
DEFAULT_EMBEDDING_MANIFEST = DEFAULT_ROOT / "esm_msa_embeddings_col" / "embedding_manifest.tsv"
DEFAULT_LABEL_SUMMARY = DEFAULT_ROOT / "sequence_label_summary.tsv.gz"
DEFAULT_SEQUENCE_MANIFEST = DEFAULT_ROOT / "sequence_manifest.tsv.gz"
DEFAULT_ESM_MSA_WEIGHTS = Path("weights/esm_msa1b_t12_100M_UR50S.pt")
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    token = re.sub(r"_+", "_", token).strip("_")
    return token or "unknown"


def condition_label(topt: float, tm: float) -> str:
    def compact(value: float) -> str:
        return f"{value:g}".replace("-", "m").replace(".", "p")

    return f"thermo_topt{compact(topt)}_tm{compact(tm)}"


def condition_value(base: float, absolute: float, delta: float | None, cap: float | None) -> float:
    value = base + delta if delta is not None else absolute
    if cap is not None:
        value = min(value, cap)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--embedding-manifest", default=str(DEFAULT_EMBEDDING_MANIFEST))
    parser.add_argument("--label-summary", default=str(DEFAULT_LABEL_SUMMARY))
    parser.add_argument("--sequence-manifest", default=str(DEFAULT_SEQUENCE_MANIFEST))
    parser.add_argument(
        "--path-rewrite",
        action="append",
        default=[],
        help="Rewrite manifest paths with OLD=NEW prefixes before opening cached files.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-count", type=int, default=50)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--msa-embedding-dtype",
        choices=contrast.MSA_EMBEDDING_DTYPES,
        default=None,
        help="Dtype for cached token_embeddings; defaults to checkpoint config.",
    )
    parser.add_argument(
        "--amp",
        choices=("checkpoint", *contrast.AMP_MODES),
        default="checkpoint",
        help="Autocast mode for model forward. By default, reuse the checkpoint config.",
    )
    parser.add_argument(
        "--max-msa-context-rows",
        type=int,
        default=None,
        help="Optional context-row cap for profile_msa token memory.",
    )
    parser.add_argument("--min-kcat", type=float, default=50.0)
    parser.add_argument("--max-original-topt", type=float, default=45.0)
    parser.add_argument("--max-original-tm", type=float, default=62.0)
    parser.add_argument("--thermo-topt", type=float, default=60.0)
    parser.add_argument("--thermo-tm", type=float, default=75.0)
    parser.add_argument(
        "--thermo-topt-delta",
        type=float,
        default=None,
        help="If set, use baseline Topt plus this delta instead of absolute --thermo-topt.",
    )
    parser.add_argument(
        "--thermo-tm-delta",
        type=float,
        default=None,
        help="If set, use baseline Tm plus this delta instead of absolute --thermo-tm.",
    )
    parser.add_argument("--max-thermo-topt", type=float, default=None)
    parser.add_argument("--max-thermo-tm", type=float, default=None)
    parser.add_argument(
        "--inference-msa-row-selection",
        choices=("cached", "farthest_hamming"),
        default="cached",
        help=(
            "MSA rows to condition on at inference. 'cached' keeps the manifest NPZ; "
            "'farthest_hamming' rebuilds per-family ESM-MSA caches from farthest aligned rows."
        ),
    )
    parser.add_argument(
        "--farthest-msa-rows",
        type=int,
        default=64,
        help="Rows to embed for --inference-msa-row-selection farthest_hamming, including the target row.",
    )
    parser.add_argument(
        "--farthest-max-cols",
        type=int,
        default=1024,
        help="Aligned columns to keep when rebuilding farthest-Hamming ESM-MSA embeddings.",
    )
    parser.add_argument(
        "--farthest-embedding-out-dir",
        default=None,
        help="Optional output directory for rebuilt farthest-Hamming embedding NPZ/metadata files.",
    )
    parser.add_argument(
        "--reuse-farthest-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse compatible farthest-Hamming embedding caches when present.",
    )
    parser.add_argument(
        "--esm-msa-weights",
        default=str(DEFAULT_ESM_MSA_WEIGHTS),
        help="Local ESM-MSA-1b weights used to rebuild farthest-Hamming inference embeddings.",
    )
    parser.add_argument("--esm-msa-layer", type=int, default=12)
    parser.add_argument("--esm-msa-output-dtype", choices=("float32", "float16"), default="float16")
    parser.add_argument(
        "--esm-msa-device",
        choices=("same", "auto", "cpu", "cuda"),
        default="same",
        help="Device for rebuilding farthest-Hamming embeddings; 'same' reuses --device.",
    )
    parser.add_argument("--min-target-length", type=int, default=80)
    parser.add_argument("--max-target-length", type=int, default=700)
    parser.add_argument("--max-candidates", type=int, default=200000)
    parser.add_argument("--cache-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


def load_candidate_from_row(
    row: tuple[str, str, dict[str, str], float, float, float],
    embeddings: dict[str, dict[str, str]],
    min_target_length: int,
    max_target_length: int,
) -> contrast.Candidate | None:
    kegg_entry, cluster_index, label, kcat, topt, tm = row
    embed = embeddings.get(cluster_index)
    if not embed:
        return None
    npz_path = Path(embed["npz_path"])
    metadata_path = Path(embed["metadata_path"])
    if not npz_path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    headers = [str(header).split()[0] for header in metadata.get("headers", [])]
    sequences = [str(sequence).upper() for sequence in metadata.get("cleaned_sequences", [])]
    if kegg_entry not in headers:
        return None
    row_index = headers.index(kegg_entry)
    if row_index >= len(sequences):
        return None
    aligned = sequences[row_index]
    target = contrast.ungap(aligned)
    if not (min_target_length <= len(target) <= max_target_length):
        return None
    if set(target) - STANDARD_AA:
        return None
    return contrast.Candidate(
        kegg_entry=kegg_entry,
        cluster_index=cluster_index,
        label=label,
        kcat=kcat,
        topt=topt,
        tm=tm,
        row_index=row_index,
        npz_path=npz_path,
        metadata_path=metadata_path,
        target_sequence=target,
        aligned_sequence=aligned,
    )


def select_candidates(
    sequence_manifest: Path,
    labels: dict[str, dict[str, str]],
    embeddings: dict[str, dict[str, str]],
    min_kcat: float,
    max_original_topt: float,
    max_original_tm: float,
    min_target_length: int,
    max_target_length: int,
    target_count: int,
    max_candidates: int,
) -> list[contrast.Candidate]:
    rows = contrast.candidate_rows(sequence_manifest, labels, min_kcat=min_kcat)
    first_pass = [
        item for item in rows if item[4] <= max_original_topt and item[5] <= max_original_tm
    ]
    first_pass_keys = {(item[0], item[1]) for item in first_pass}
    search_rows = first_pass + [item for item in rows if (item[0], item[1]) not in first_pass_keys]
    accepted: list[contrast.Candidate] = []
    accepted_clusters: set[str] = set()
    inspected = 0
    for row in search_rows:
        inspected += 1
        if inspected > max_candidates:
            break
        cluster_index = row[1]
        if cluster_index in accepted_clusters:
            continue
        candidate = load_candidate_from_row(
            row,
            embeddings,
            min_target_length=min_target_length,
            max_target_length=max_target_length,
        )
        if not candidate:
            continue
        accepted.append(candidate)
        accepted_clusters.add(cluster_index)
        print(
            f"selected fam{len(accepted) - 1:03d} cluster={candidate.cluster_index} "
            f"kegg={candidate.kegg_entry} len={len(candidate.target_sequence)} "
            f"kcat={candidate.kcat:.6g} Topt={candidate.topt:.1f} Tm={candidate.tm:.1f}",
            flush=True,
        )
        if len(accepted) >= target_count:
            break
    if len(accepted) < target_count:
        raise SystemExit(f"Only selected {len(accepted)} candidates; requested {target_count}")
    return accepted


def header_token(header: str) -> str:
    return str(header).split()[0]


def source_msa_path_for_candidate(
    candidate: contrast.Candidate,
    embeddings: dict[str, dict[str, str]],
    path_rewrites: list[tuple[str, str]],
) -> Path:
    source_msa = embeddings.get(candidate.cluster_index, {}).get("source_msa", "")
    if not source_msa:
        try:
            metadata = json.loads(candidate.metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"Could not read metadata for cluster {candidate.cluster_index}: {candidate.metadata_path}"
            ) from exc
        source_msa = contrast.rewrite_manifest_path(str(metadata.get("source_msa", "")), path_rewrites)
    if not source_msa:
        raise SystemExit(f"No source_msa recorded for cluster {candidate.cluster_index}")
    path = Path(source_msa).expanduser()
    if not path.exists():
        raise SystemExit(f"source_msa not found for cluster {candidate.cluster_index}: {path}")
    return path


def hamming_distance_vector(np: Any, encoded: Any, query_index: int) -> Any:
    query = encoded[query_index]
    mismatches = encoded != query
    both_gaps = (encoded == ord("-")) & (query == ord("-"))
    return np.logical_and(mismatches, ~both_gaps).sum(axis=1).astype(np.int32)


def farthest_hamming_indices(
    np: Any,
    sequences: list[str],
    target_index: int,
    max_rows: int,
) -> tuple[list[int], dict[str, Any]]:
    if max_rows < 1:
        raise ValueError("max_rows must be >= 1")
    if not sequences:
        raise ValueError("cannot select rows from an empty MSA")
    rows = len(sequences)
    cols = len(sequences[0])
    encoded = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8).reshape(rows, cols)
    selected = [target_index]
    eligible = np.ones((rows,), dtype=np.bool_)
    eligible[target_index] = False
    min_distances = hamming_distance_vector(np, encoded, target_index)
    target_distances = min_distances.copy()
    min_distances[target_index] = -1

    while len(selected) < min(max_rows, rows) and np.any(eligible):
        eligible_indices = np.flatnonzero(eligible)
        best_min = int(min_distances[eligible_indices].max())
        tied = eligible_indices[min_distances[eligible_indices] == best_min]
        if tied.size > 1:
            best_target = int(target_distances[tied].max())
            tied = tied[target_distances[tied] == best_target]
        best_index = int(tied.min())
        selected.append(best_index)
        eligible[best_index] = False
        new_distances = hamming_distance_vector(np, encoded, best_index)
        min_distances = np.minimum(min_distances, new_distances)
        min_distances[~eligible] = -1

    selected_target_distances = target_distances[selected[1:]]
    stats = {
        "source_rows": rows,
        "source_cols": cols,
        "selected_rows": len(selected),
        "selected_source_indices": selected,
        "target_source_index": target_index,
        "target_distance_min": int(selected_target_distances.min()) if selected_target_distances.size else 0,
        "target_distance_median": float(np.median(selected_target_distances))
        if selected_target_distances.size
        else 0.0,
        "target_distance_max": int(selected_target_distances.max()) if selected_target_distances.size else 0,
    }
    return selected, stats


def compatible_farthest_embedding(
    npz_path: Path,
    metadata_path: Path,
    candidate: contrast.Candidate,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not args.reuse_farthest_embeddings or not npz_path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    selection = metadata.get("selection", {})
    headers = metadata.get("headers", [])
    sequences = metadata.get("cleaned_sequences", [])
    if metadata.get("status") != "embedded":
        return None
    if selection.get("mode") != "farthest_hamming":
        return None
    if int(selection.get("requested_rows", -1)) != int(args.farthest_msa_rows):
        return None
    if int(metadata.get("max_cols", -1)) != int(args.farthest_max_cols):
        return None
    if int(metadata.get("layer", -1)) != int(args.esm_msa_layer):
        return None
    if str(metadata.get("dtype")) != str(args.esm_msa_output_dtype):
        return None
    if not headers or header_token(str(headers[0])) != candidate.kegg_entry:
        return None
    if not sequences or contrast.ungap(str(sequences[0])) != candidate.target_sequence:
        return None
    return metadata


def load_esm_msa_context(args: argparse.Namespace, generation_device: torch.device) -> dict[str, Any]:
    try:
        import numpy as np  # type: ignore[import-not-found]
        import esm  # type: ignore[import-not-found]
        import embed_msas as msa_embed  # noqa: E402
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "farthest_hamming inference selection requires numpy and fair-esm in this environment"
        ) from exc

    weights_path = Path(args.esm_msa_weights)
    if not weights_path.exists():
        raise SystemExit(f"ESM-MSA weights not found: {weights_path}")
    regression_path = msa_embed.contact_regression_path(weights_path)
    if not regression_path.exists():
        raise SystemExit(f"ESM-MSA contact-regression sidecar not found: {regression_path}")

    requested_device = str(generation_device) if args.esm_msa_device == "same" else args.esm_msa_device
    if requested_device == "auto":
        esm_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif requested_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--esm-msa-device cuda requested but CUDA is unavailable")
    else:
        esm_device = torch.device(requested_device)

    print(f"loading ESM-MSA for farthest-Hamming inference embeddings on {esm_device}", flush=True)
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(weights_path))
    model.eval()
    model.to(esm_device)
    return {
        "np": np,
        "msa_embed": msa_embed,
        "weights_path": weights_path,
        "regression_path": regression_path,
        "device": esm_device,
        "model": model,
        "alphabet": alphabet,
    }


def embed_selected_msa(
    selected_msa: dict[str, Any],
    selection_stats: dict[str, Any],
    source_msa_path: Path,
    npz_path: Path,
    metadata_path: Path,
    args: argparse.Namespace,
    context: dict[str, Any],
) -> dict[str, Any]:
    np = context["np"]
    msa_embed = context["msa_embed"]
    device = context["device"]
    model = context["model"]
    alphabet = context["alphabet"]
    rows = len(selected_msa["sequences"])
    cols = len(selected_msa["sequences"][0])

    batch_converter = alphabet.get_batch_converter()
    msa_records = list(zip(selected_msa["headers"], selected_msa["sequences"]))
    _, _, tokens = batch_converter([msa_records])
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[args.esm_msa_layer], return_contacts=False)
        token_repr = msa_embed.aligned_representations(
            torch,
            results,
            args.esm_msa_layer,
            rows,
            cols,
        )
        token_embeddings = token_repr.detach().cpu().numpy()

    if args.esm_msa_output_dtype == "float16":
        token_embeddings = token_embeddings.astype(np.float16)
    else:
        token_embeddings = token_embeddings.astype(np.float32)

    aa_mask, gap_mask = msa_embed.masks_from_sequences(np, selected_msa["sequences"])
    row_embeddings, row_counts = msa_embed.masked_mean(np, token_embeddings, aa_mask, axis=1)
    col_embeddings, col_counts = msa_embed.masked_mean(np, token_embeddings, aa_mask, axis=0)
    query_embedding, query_count = msa_embed.query_masked_mean(np, token_embeddings, aa_mask)
    msa_embedding, msa_count = msa_embed.global_masked_mean(np, token_embeddings, aa_mask)

    arrays = {
        "query_embedding": query_embedding,
        "msa_embedding": msa_embedding,
        "row_embeddings": row_embeddings,
        "col_embeddings": col_embeddings,
        "aa_mask": aa_mask,
        "gap_mask": gap_mask,
        "row_aa_counts": row_counts,
        "col_aa_counts": col_counts,
        "token_embeddings": token_embeddings,
    }
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)

    metadata = {
        "source_msa": str(source_msa_path),
        "output_npz": str(npz_path),
        "output_metadata": str(metadata_path),
        "weights": str(context["weights_path"]),
        "contact_regression_weights": str(context["regression_path"]),
        "layer": args.esm_msa_layer,
        "requested_device": args.esm_msa_device,
        "device": str(device),
        "dtype": args.esm_msa_output_dtype,
        "max_seqs": args.farthest_msa_rows,
        "max_cols": args.farthest_max_cols,
        "original_shape": {
            "rows": int(selection_stats.get("source_original_rows", selection_stats["source_rows"])),
            "cols": int(selection_stats.get("source_original_cols", selection_stats["source_cols"])),
        },
        "shape": {
            "rows": rows,
            "cols": cols,
        },
        "cropped": bool(selected_msa["warnings"]),
        "warnings": selected_msa["warnings"],
        "headers": selected_msa["headers"],
        "cleaned_sequences": selected_msa["sequences"],
        "pooling": msa_embed.POOLING_DESCRIPTION,
        "stores_token_embeddings": True,
        "status": "embedded",
        "model_class": model.__class__.__name__,
        "hidden_dim": int(token_embeddings.shape[-1]),
        "token_embedding_shape": list(token_embeddings.shape),
        "row_aa_counts": row_counts.astype(int).tolist(),
        "col_aa_counts": col_counts.astype(int).tolist(),
        "query_aa_count": int(query_count),
        "msa_aa_count": int(msa_count),
        "npz_arrays": {name: list(value.shape) for name, value in arrays.items()},
        "selection": {
            "mode": "farthest_hamming",
            "requested_rows": int(args.farthest_msa_rows),
            **selection_stats,
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def rebuild_farthest_hamming_candidates(
    candidates: list[contrast.Candidate],
    embeddings: dict[str, dict[str, str]],
    path_rewrites: list[tuple[str, str]],
    args: argparse.Namespace,
    device: torch.device,
) -> list[contrast.Candidate]:
    if args.farthest_msa_rows < 2:
        raise SystemExit("--farthest-msa-rows must be >= 2 so at least one context row remains")
    if args.farthest_max_cols < 1:
        raise SystemExit("--farthest-max-cols must be >= 1")

    try:
        import numpy as np  # type: ignore[import-not-found]
        import embed_msas as msa_embed  # noqa: E402
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "farthest_hamming inference selection requires numpy and the local embed_msas module"
        ) from exc

    embedding_out_dir = (
        Path(args.farthest_embedding_out_dir)
        if args.farthest_embedding_out_dir
        else Path(args.out_dir) / "farthest_hamming_msa_embeddings"
    )
    embedding_out_dir.mkdir(parents=True, exist_ok=True)
    rebuilt: list[contrast.Candidate] = []
    context: dict[str, Any] | None = None
    for family_index, candidate in enumerate(candidates):
        family_id = f"fam{family_index:03d}"
        stem = (
            f"{family_id}_c{safe_token(candidate.cluster_index)}_"
            f"{safe_token(candidate.kegg_entry)}_hamming{args.farthest_msa_rows}"
        )
        npz_path = embedding_out_dir / f"{stem}.npz"
        metadata_path = embedding_out_dir / f"{stem}.metadata.json"
        metadata = compatible_farthest_embedding(npz_path, metadata_path, candidate, args)
        if metadata is None:
            source_msa_path = source_msa_path_for_candidate(candidate, embeddings, path_rewrites)
            msa = msa_embed.load_and_clean_msa(source_msa_path)
            headers = [header_token(header) for header in msa["headers"]]
            if candidate.kegg_entry not in headers:
                raise SystemExit(
                    f"Target {candidate.kegg_entry} not found in source MSA {source_msa_path}"
                )
            target_index = headers.index(candidate.kegg_entry)
            sequences = [str(sequence).upper()[: args.farthest_max_cols] for sequence in msa["sequences"]]
            selected_indices, selection_stats = farthest_hamming_indices(
                np,
                sequences,
                target_index,
                args.farthest_msa_rows,
            )
            selection_stats["source_original_rows"] = int(msa["original_rows"])
            selection_stats["source_original_cols"] = int(msa["original_cols"])
            selection_stats["hamming_cols"] = int(len(sequences[0]))
            selected_msa = {
                "headers": [msa["headers"][idx] for idx in selected_indices],
                "sequences": [sequences[idx] for idx in selected_indices],
                "warnings": [],
            }
            if msa["original_rows"] > len(selected_indices):
                selected_msa["warnings"].append(
                    f"selected farthest-Hamming rows from {msa['original_rows']} to {len(selected_indices)}"
                )
            if msa["original_cols"] > args.farthest_max_cols:
                selected_msa["warnings"].append(
                    f"cropped columns from {msa['original_cols']} to first {args.farthest_max_cols}"
                )
            target_sequence = contrast.ungap(selected_msa["sequences"][0])
            if target_sequence != candidate.target_sequence:
                raise SystemExit(
                    f"Target sequence mismatch after farthest-Hamming crop for {candidate.kegg_entry}: "
                    f"cached_len={len(candidate.target_sequence)} selected_len={len(target_sequence)}"
                )
            if context is None:
                context = load_esm_msa_context(args, device)
            metadata = embed_selected_msa(
                selected_msa=selected_msa,
                selection_stats=selection_stats,
                source_msa_path=source_msa_path,
                npz_path=npz_path,
                metadata_path=metadata_path,
                args=args,
                context=context,
            )
            print(
                f"embedded {family_id} farthest-Hamming MSA rows={metadata['shape']['rows']} "
                f"cols={metadata['shape']['cols']} target_dist="
                f"{metadata['selection']['target_distance_min']}/"
                f"{metadata['selection']['target_distance_median']:.1f}/"
                f"{metadata['selection']['target_distance_max']}",
                flush=True,
            )
        else:
            print(f"reused {family_id} farthest-Hamming MSA embedding {npz_path}", flush=True)

        rebuilt.append(
            contrast.Candidate(
                kegg_entry=candidate.kegg_entry,
                cluster_index=candidate.cluster_index,
                label=candidate.label,
                kcat=candidate.kcat,
                topt=candidate.topt,
                tm=candidate.tm,
                row_index=0,
                npz_path=npz_path,
                metadata_path=metadata_path,
                target_sequence=candidate.target_sequence,
                aligned_sequence=str(metadata["cleaned_sequences"][0]),
            )
        )

    if context is not None:
        del context["model"]
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rebuilt


def msa_selection_fields(metadata_path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    selection = metadata.get("selection")
    if not isinstance(selection, dict):
        return {}
    return {
        "msa_row_selection": selection.get("mode", ""),
        "msa_requested_rows": selection.get("requested_rows", ""),
        "msa_selected_rows": selection.get("selected_rows", ""),
        "msa_source_rows": selection.get("source_rows", ""),
        "msa_source_cols": selection.get("source_cols", ""),
        "msa_target_hamming_min": selection.get("target_distance_min", ""),
        "msa_target_hamming_median": selection.get("target_distance_median", ""),
        "msa_target_hamming_max": selection.get("target_distance_max", ""),
        "msa_source_msa": metadata.get("source_msa", ""),
    }


def make_dataset(
    candidates: list[contrast.Candidate],
    labels: dict[str, dict[str, str]],
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    args: argparse.Namespace,
) -> contrast.CachedMSARowDataset:
    examples = [
        contrast.RowExample(
            cluster_index=candidate.cluster_index,
            split="generate",
            npz_path=candidate.npz_path,
            metadata_path=candidate.metadata_path,
            row_index=candidate.row_index,
            kegg_entry=candidate.kegg_entry,
            aligned_sequence=candidate.aligned_sequence,
            target_sequence=candidate.target_sequence,
        )
        for candidate in candidates
    ]
    return contrast.CachedMSARowDataset(
        examples=examples,
        labels=labels,
        numeric_means=checkpoint["numeric_means"],
        numeric_stds=checkpoint["numeric_stds"],
        category_buckets=int(config["category_buckets"]),
        cache_size=max(1, min(args.cache_size, len(examples))),
        consensus_loss_mode=str(config.get("consensus_loss_mode", "none")),
        consensus_match_weight=float(config.get("consensus_match_weight", 0.35)),
        nonconsensus_weight=float(config.get("nonconsensus_weight", 2.5)),
        unobserved_nonconsensus_weight=float(config.get("unobserved_nonconsensus_weight", 1.0)),
        max_sequence_loss_weight=float(config.get("max_sequence_loss_weight", 3.0)),
        variable_column_min_entropy=float(config.get("variable_column_min_entropy", 0.05)),
        variable_column_max_consensus=float(config.get("variable_column_max_consensus", 0.92)),
        require_msa_embeddings=contrast.uses_msa_embedding_memory(str(config["memory_mode"])),
        msa_embedding_dtype=str(args.msa_embedding_dtype or config.get("msa_embedding_dtype", "float32")),
        max_msa_context_rows=(
            args.max_msa_context_rows
            if args.max_msa_context_rows is not None
            else config.get("max_msa_context_rows")
        ),
        gap_inclusive_msa_mask=contrast.uses_gap_inclusive_msa_mask(str(config["memory_mode"])),
        require_target_continuous_embeddings=str(config.get("continuous_target_mode", "token_embedding"))
        == "target_row_embedding",
    )


def build_model(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    first_item: dict[str, Any],
    device: torch.device,
) -> contrast.MeanStartCCDDModel:
    model = contrast.MeanStartCCDDModel(
        row_embedding_dim=int(first_item["row_embeddings"].shape[-1]),
        d_model=int(config["d_model"]),
        layers=int(config["layers"]),
        heads=int(config["heads"]),
        dropout=float(config["dropout"]),
        max_sequence_length=int(config["max_sequence_length"]),
        diffusion_timesteps=int(config["diffusion_timesteps"]),
        category_buckets=int(config["category_buckets"]),
        memory_mode=str(config["memory_mode"]),
        profile_feature_mode=str(config.get("profile_feature_mode", "full")),
        msa_embedding_dim=int(first_item["msa_embeddings"].shape[-1]),
        continuous_target_mode=str(config.get("continuous_target_mode", "token_embedding")),
        target_continuous_dim=int(first_item["target_continuous_embeddings"].shape[-1]),
        msa_axial_layers=int(config.get("msa_axial_layers", 1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    if args.max_target_length > int(config["max_sequence_length"]) - 1:
        raise SystemExit(
            f"--max-target-length {args.max_target_length} exceeds model max "
            f"{int(config['max_sequence_length']) - 1}"
        )
    amp_mode = str(config.get("amp", "off")) if args.amp == "checkpoint" else args.amp
    path_rewrites = contrast.parse_path_rewrites(args.path_rewrite)
    labels = contrast.load_labels(Path(args.label_summary))
    embeddings = contrast.read_embedding_manifest(Path(args.embedding_manifest), path_rewrites=path_rewrites)
    candidates = select_candidates(
        sequence_manifest=Path(args.sequence_manifest),
        labels=labels,
        embeddings=embeddings,
        min_kcat=args.min_kcat,
        max_original_topt=args.max_original_topt,
        max_original_tm=args.max_original_tm,
        min_target_length=args.min_target_length,
        max_target_length=args.max_target_length,
        target_count=args.target_count,
        max_candidates=args.max_candidates,
    )
    if args.inference_msa_row_selection == "farthest_hamming":
        candidates = rebuild_farthest_hamming_candidates(
            candidates,
            embeddings=embeddings,
            path_rewrites=path_rewrites,
            args=args,
            device=device,
        )

    dataset = make_dataset(candidates, labels, checkpoint, config, args)
    collator = contrast.RowReconstructionCollator(
        max_sequence_length=int(config["max_sequence_length"]),
        tail_stop_weight=float(config["tail_stop_weight"]),
        profile_feature_mode=str(config.get("profile_feature_mode", "full")),
    )
    first_item = dataset[0]
    model = build_model(checkpoint, config, first_item, device)

    metadata_rows: list[dict[str, Any]] = []
    fold_records: list[tuple[str, str]] = []
    target_records: list[tuple[str, str]] = []
    for family_index, candidate in enumerate(candidates):
        item = dataset[family_index]
        batch = collator([item])
        baseline_overrides = {
            "kcat_1_per_s": candidate.kcat,
            "topt_C": candidate.topt,
            "tm_C": candidate.tm,
        }
        thermo_topt = condition_value(
            candidate.topt,
            args.thermo_topt,
            args.thermo_topt_delta,
            args.max_thermo_topt,
        )
        thermo_tm = condition_value(
            candidate.tm,
            args.thermo_tm,
            args.thermo_tm_delta,
            args.max_thermo_tm,
        )
        thermo_overrides = {
            "kcat_1_per_s": candidate.kcat,
            "topt_C": thermo_topt,
            "tm_C": thermo_tm,
        }
        baseline_batch = contrast.set_numeric_override(
            batch,
            labels,
            candidate.kegg_entry,
            checkpoint["numeric_means"],
            checkpoint["numeric_stds"],
            baseline_overrides,
        )
        thermo_batch = contrast.set_numeric_override(
            batch,
            labels,
            candidate.kegg_entry,
            checkpoint["numeric_means"],
            checkpoint["numeric_stds"],
            thermo_overrides,
        )
        baseline_sequence = contrast.mean_decode(model, baseline_batch, device, amp_mode)
        thermo_sequence = contrast.mean_decode(model, thermo_batch, device, amp_mode)
        family_id = f"fam{family_index:03d}"
        entry_token = safe_token(candidate.kegg_entry)
        baseline_header = f"{family_id}_baseline_c{safe_token(candidate.cluster_index)}_{entry_token}"
        thermo_header = (
            f"{family_id}_{condition_label(thermo_topt, thermo_tm)}_"
            f"c{safe_token(candidate.cluster_index)}_{entry_token}"
        )
        target_header = f"{family_id}_target_c{safe_token(candidate.cluster_index)}_{entry_token}"
        fold_records.extend(
            [
                (baseline_header, baseline_sequence),
                (thermo_header, thermo_sequence),
            ]
        )
        target_records.append((target_header, candidate.target_sequence))
        row = {
            "family_id": family_id,
            "cluster_index": candidate.cluster_index,
            "kegg_entry": candidate.kegg_entry,
            "row_index": candidate.row_index,
            "kcat": f"{candidate.kcat:.8g}",
            "baseline_topt": f"{candidate.topt:.4g}",
            "baseline_tm": f"{candidate.tm:.4g}",
            "thermo_topt": f"{thermo_topt:.4g}",
            "thermo_tm": f"{thermo_tm:.4g}",
            "target_length": len(candidate.target_sequence),
            "baseline_length": len(baseline_sequence),
            "thermo_length": len(thermo_sequence),
            "variant_identity": f"{contrast.identity(baseline_sequence, thermo_sequence):.6f}",
            "target_baseline_identity": f"{contrast.identity(candidate.target_sequence, baseline_sequence):.6f}",
            "target_thermo_identity": f"{contrast.identity(candidate.target_sequence, thermo_sequence):.6f}",
            "baseline_header": baseline_header,
            "thermo_header": thermo_header,
            "target_header": target_header,
            "npz_path": str(candidate.npz_path),
            "metadata_path": str(candidate.metadata_path),
            "baseline_sequence": baseline_sequence,
            "thermo_sequence": thermo_sequence,
            "target_sequence": candidate.target_sequence,
        }
        row.update(msa_selection_fields(candidate.metadata_path))
        metadata_rows.append(row)
        print(
            f"decoded {family_id} baseline_len={len(baseline_sequence)} "
            f"thermo_len={len(thermo_sequence)} identity={row['variant_identity']}",
            flush=True,
        )

    contrast.write_fasta(out_dir / "batch_variants.fasta", fold_records)
    contrast.write_fasta(out_dir / "batch_targets.fasta", target_records)
    write_tsv(out_dir / "batch_metadata.tsv", metadata_rows)
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "settings": {
            "target_count": args.target_count,
            "min_kcat": args.min_kcat,
            "max_original_topt": args.max_original_topt,
            "max_original_tm": args.max_original_tm,
            "thermo_topt": args.thermo_topt,
            "thermo_tm": args.thermo_tm,
            "thermo_topt_delta": args.thermo_topt_delta,
            "thermo_tm_delta": args.thermo_tm_delta,
            "max_thermo_topt": args.max_thermo_topt,
            "max_thermo_tm": args.max_thermo_tm,
            "inference_msa_row_selection": args.inference_msa_row_selection,
            "farthest_msa_rows": args.farthest_msa_rows,
            "farthest_max_cols": args.farthest_max_cols,
            "farthest_embedding_out_dir": args.farthest_embedding_out_dir,
            "reuse_farthest_embeddings": args.reuse_farthest_embeddings,
            "esm_msa_weights": args.esm_msa_weights,
            "esm_msa_layer": args.esm_msa_layer,
            "esm_msa_output_dtype": args.esm_msa_output_dtype,
            "esm_msa_device": args.esm_msa_device,
            "min_target_length": args.min_target_length,
            "max_target_length": args.max_target_length,
            "amp": amp_mode,
            "device": str(device),
        },
        "rows": metadata_rows,
        "note": (
            "kcat is held fixed for baseline and thermostable decode. "
            "Current checkpoint has no pH optimum condition."
        ),
    }
    (out_dir / "batch_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(metadata_rows)} families to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
