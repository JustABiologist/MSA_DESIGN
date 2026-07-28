#!/usr/bin/env python3
"""Generate a baseline-vs-thermostable contrast from a mean-start CCDD checkpoint."""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from msa_design_model import batch_encode_sequences_with_stop, decode_tokens_until_stop  # noqa: E402
from train_mean_start_ccdd_from_cached_msas import (  # noqa: E402
    CATEGORICAL_FIELDS,
    MSA_EMBEDDING_DTYPES,
    NUMERIC_FIELDS,
    CachedMSARowDataset,
    MeanStartCCDDModel,
    RowExample,
    RowReconstructionCollator,
    open_text,
    parse_float,
    rewrite_manifest_path,
    transform_numeric,
    uses_msa_embedding_memory,
    uses_gap_inclusive_msa_mask,
)


DEFAULT_ROOT = Path("/mnt/backup4TB/MSA_DESIGN/training_msas_50_identity_core_gaptrim")
DEFAULT_CHECKPOINT = (
    DEFAULT_ROOT
    / "mean_start_ccdd_full_profile_row_bs32_from38000_20260718_161831"
    / "mean_start_ccdd.latest.pt"
)
DEFAULT_EMBEDDING_MANIFEST = DEFAULT_ROOT / "esm_msa_embeddings_col" / "embedding_manifest.tsv"
DEFAULT_LABEL_SUMMARY = DEFAULT_ROOT / "sequence_label_summary.tsv.gz"
DEFAULT_SEQUENCE_MANIFEST = DEFAULT_ROOT / "sequence_manifest.tsv.gz"


@dataclass(frozen=True)
class Candidate:
    kegg_entry: str
    cluster_index: str
    label: dict[str, str]
    kcat: float
    topt: float
    tm: float
    row_index: int
    npz_path: Path
    metadata_path: Path
    target_sequence: str
    aligned_sequence: str


def read_embedding_manifest(
    path: Path,
    path_rewrites: list[tuple[str, str]] | None = None,
) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    rewrites = path_rewrites or []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("status") != "embedded":
                continue
            row = dict(row)
            row["npz_path"] = rewrite_manifest_path(row.get("npz_path", ""), rewrites)
            row["metadata_path"] = rewrite_manifest_path(row.get("metadata_path", ""), rewrites)
            row["source_msa"] = rewrite_manifest_path(row.get("source_msa", ""), rewrites)
            rows.setdefault(row["cluster_index"], row)
    return rows


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    labels: dict[str, dict[str, str]] = {}
    with open_text(path, "r") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            kegg_entry = row.get("kegg_entry")
            if kegg_entry:
                labels[kegg_entry] = row
    return labels


def ungap(sequence: str) -> str:
    return "".join(char for char in sequence.upper() if char not in {"-", ".", " ", "\n", "\r", "\t"})


def candidate_rows(
    sequence_manifest: Path,
    labels: dict[str, dict[str, str]],
    min_kcat: float,
) -> list[tuple[str, str, dict[str, str], float, float, float]]:
    rows: list[tuple[str, str, dict[str, str], float, float, float]] = []
    with gzip.open(sequence_manifest, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("kept") not in {"1", "yes", "true", "True"}:
                continue
            kegg_entry = row["kegg_entry"]
            label = labels.get(kegg_entry)
            if not label:
                continue
            kcat = parse_float(label.get("kcat_1_per_s_mean", ""))
            topt = parse_float(label.get("topt_C_mean", ""))
            tm = parse_float(label.get("tm_C_mean", ""))
            if kcat is None or topt is None or tm is None:
                continue
            if kcat < min_kcat:
                continue
            rows.append((kegg_entry, row["cluster_index"], label, kcat, topt, tm))
    rows.sort(key=lambda item: item[3], reverse=True)
    return rows


def find_candidate(
    sequence_manifest: Path,
    labels: dict[str, dict[str, str]],
    embeddings: dict[str, dict[str, str]],
    min_kcat: float,
    max_original_topt: float,
    max_original_tm: float,
    max_candidates: int,
) -> Candidate:
    selected = candidate_rows(sequence_manifest, labels, min_kcat=min_kcat)
    first_pass = [
        item for item in selected if item[4] <= max_original_topt and item[5] <= max_original_tm
    ]
    first_pass_keys = {(item[0], item[1]) for item in first_pass}
    search_rows = first_pass + [item for item in selected if (item[0], item[1]) not in first_pass_keys]
    for kegg_entry, cluster_index, label, kcat, topt, tm in search_rows[:max_candidates]:
        embed = embeddings.get(cluster_index)
        if not embed:
            continue
        npz_path = Path(embed["npz_path"])
        metadata_path = Path(embed["metadata_path"])
        if not npz_path.exists() or not metadata_path.exists():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        headers = [str(header).split()[0] for header in metadata.get("headers", [])]
        sequences = [str(sequence).upper() for sequence in metadata.get("cleaned_sequences", [])]
        if kegg_entry not in headers:
            continue
        row_index = headers.index(kegg_entry)
        aligned = sequences[row_index]
        target = ungap(aligned)
        if target:
            return Candidate(
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
    raise SystemExit("No high-kcat candidate found inside cropped embedding metadata")


def normalize_numeric(
    field: str,
    value: float,
    means: dict[str, float],
    stds: dict[str, float],
) -> float:
    transformed = transform_numeric(field, value)
    if transformed is None:
        raise ValueError(f"invalid numeric value for {field}: {value}")
    return (transformed - means[field]) / stds[field]


def set_numeric_override(
    batch: dict[str, Any],
    labels: dict[str, dict[str, str]],
    kegg_entry: str,
    means: dict[str, float],
    stds: dict[str, float],
    overrides: dict[str, float],
) -> dict[str, Any]:
    output = dict(batch)
    numeric_values = batch["numeric_values"].clone()
    numeric_mask = batch["numeric_mask"].clone()
    for field, value in overrides.items():
        idx = NUMERIC_FIELDS.index(field)
        numeric_values[0, idx] = normalize_numeric(field, value, means, stds)
        numeric_mask[0, idx] = True
    output["numeric_values"] = numeric_values
    output["numeric_mask"] = numeric_mask
    return output


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


@torch.no_grad()
def mean_decode(model: MeanStartCCDDModel, batch: dict[str, Any], device: torch.device) -> str:
    model.eval()
    moved = move_batch(batch, device)
    timesteps = torch.zeros((moved["target_tokens"].shape[0],), dtype=torch.long, device=device)
    outputs = model(
        profiles=moved["profiles"],
        profile_mask=moved["profile_mask"],
        row_embeddings=moved["row_embeddings"],
        row_mask=moved["row_mask"],
        msa_embeddings=moved["msa_embeddings"],
        msa_embedding_mask=moved["msa_embedding_mask"],
        numeric_values=moved["numeric_values"],
        numeric_mask=moved["numeric_mask"],
        category_ids=moved["category_ids"],
        category_mask=moved["category_mask"],
        target_tokens=moved["target_tokens"],
        loss_weights=moved["loss_weights"],
        sequence_loss_weights=moved.get("sequence_loss_weights"),
        target_continuous_embeddings=moved.get("target_continuous_embeddings"),
        target_continuous_mask=moved.get("target_continuous_mask"),
        timesteps=timesteps,
        decoder_start_mode="mean",
        memory_dropout=0.0,
        profile_variable_mask=moved.get("profile_variable_mask"),
    )
    predicted = torch.argmax(outputs["logits"], dim=-1).detach().cpu()
    return decode_tokens_until_stop(predicted[0].tolist())


def needleman_wunsch(a: str, b: str) -> tuple[str, str]:
    match = 2
    mismatch = -1
    gap = -2
    rows = len(a) + 1
    cols = len(b) + 1
    score = [[0] * cols for _ in range(rows)]
    trace = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        score[i][0] = score[i - 1][0] + gap
        trace[i][0] = "U"
    for j in range(1, cols):
        score[0][j] = score[0][j - 1] + gap
        trace[0][j] = "L"
    for i in range(1, rows):
        for j in range(1, cols):
            diag = score[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch)
            up = score[i - 1][j] + gap
            left = score[i][j - 1] + gap
            best = max(diag, up, left)
            score[i][j] = best
            trace[i][j] = "D" if best == diag else "U" if best == up else "L"
    i, j = len(a), len(b)
    out_a: list[str] = []
    out_b: list[str] = []
    while i or j:
        direction = trace[i][j]
        if direction == "D":
            out_a.append(a[i - 1])
            out_b.append(b[j - 1])
            i -= 1
            j -= 1
        elif direction == "U":
            out_a.append(a[i - 1])
            out_b.append("-")
            i -= 1
        else:
            out_a.append("-")
            out_b.append(b[j - 1])
            j -= 1
    return "".join(reversed(out_a)), "".join(reversed(out_b))


def identity(a: str, b: str) -> float:
    aligned_a, aligned_b = needleman_wunsch(a, b)
    comparable = [(x, y) for x, y in zip(aligned_a, aligned_b) if x != "-" and y != "-"]
    if not comparable:
        return 0.0
    return sum(1 for x, y in comparable if x == y) / len(comparable)


def html_alignment(baseline: str, thermostable: str, metadata: dict[str, Any]) -> str:
    aligned_a, aligned_b = needleman_wunsch(baseline, thermostable)
    marker = "".join("|" if x == y else " " for x, y in zip(aligned_a, aligned_b))

    def row_html(text: str, other: str) -> str:
        spans: list[str] = []
        for residue, other_residue in zip(text, other):
            escaped = html.escape(residue)
            if residue == other_residue:
                spans.append(f"<span class='same'>{escaped}</span>")
            elif residue == "-":
                spans.append(f"<span class='gap'>{escaped}</span>")
            else:
                spans.append(f"<span class='diff'>{escaped}</span>")
        return "".join(spans)

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Thermostability Contrast Alignment</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #172026; }}
.meta {{ margin-bottom: 18px; line-height: 1.45; }}
.alignment {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; line-height: 1.75; white-space: pre-wrap; overflow-wrap: anywhere; }}
.label {{ display: inline-block; width: 14ch; color: #51606b; }}
.same {{ color: #172026; }}
.diff {{ background: #ffe08a; color: #111; border-radius: 2px; padding: 0 1px; }}
.gap {{ background: #dce4ea; color: #5c6870; border-radius: 2px; padding: 0 1px; }}
.legend span {{ padding: 1px 4px; border-radius: 2px; }}
</style>
</head>
<body>
<h1>Thermostability Contrast Alignment</h1>
<div class="meta">
<div><b>Family cluster:</b> {html.escape(str(metadata['cluster_index']))}</div>
<div><b>Target row:</b> {html.escape(str(metadata['kegg_entry']))}</div>
<div><b>Held kcat:</b> {metadata['kcat']:.6g} 1/s</div>
<div><b>Baseline conditions:</b> Topt {metadata['baseline_topt']:.1f} C, Tm {metadata['baseline_tm']:.1f} C</div>
<div><b>Thermostable conditions:</b> Topt {metadata['thermo_topt']:.1f} C, Tm {metadata['thermo_tm']:.1f} C</div>
<div><b>Variant identity:</b> {metadata['variant_identity']:.4f}</div>
</div>
<div class="legend">Legend: <span class="diff">different amino acid</span> <span class="gap">alignment gap</span></div>
<pre class="alignment"><span class="label">baseline</span>{row_html(aligned_a, aligned_b)}
<span class="label">match</span>{html.escape(marker)}
<span class="label">thermostable</span>{row_html(aligned_b, aligned_a)}</pre>
</body>
</html>
"""


def write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for header, sequence in records:
            handle.write(f">{header}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def parse_path_rewrites(values: list[str]) -> list[tuple[str, str]]:
    rewrites: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--path-rewrite must be OLD=NEW, got {value!r}")
        old, new = value.split("=", 1)
        if not old or not new:
            raise SystemExit(f"--path-rewrite must be OLD=NEW, got {value!r}")
        rewrites.append((old, new))
    return rewrites


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
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--msa-embedding-dtype",
        choices=MSA_EMBEDDING_DTYPES,
        default=None,
        help="Dtype for cached token_embeddings; defaults to checkpoint config.",
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
    parser.add_argument("--thermo-topt", type=float, default=68.0)
    parser.add_argument("--thermo-tm", type=float, default=78.0)
    parser.add_argument("--max-candidates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


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

    checkpoint_path = Path(args.checkpoint)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else checkpoint_path.parent / f"thermostability_contrast_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    path_rewrites = parse_path_rewrites(args.path_rewrite)
    labels = load_labels(Path(args.label_summary))
    embeddings = read_embedding_manifest(Path(args.embedding_manifest), path_rewrites=path_rewrites)
    candidate = find_candidate(
        sequence_manifest=Path(args.sequence_manifest),
        labels=labels,
        embeddings=embeddings,
        min_kcat=args.min_kcat,
        max_original_topt=args.max_original_topt,
        max_original_tm=args.max_original_tm,
        max_candidates=args.max_candidates,
    )
    example = RowExample(
        cluster_index=candidate.cluster_index,
        split="generate",
        npz_path=candidate.npz_path,
        metadata_path=candidate.metadata_path,
        row_index=candidate.row_index,
        kegg_entry=candidate.kegg_entry,
        aligned_sequence=candidate.aligned_sequence,
        target_sequence=candidate.target_sequence,
    )

    dataset = CachedMSARowDataset(
        examples=[example],
        labels=labels,
        numeric_means=checkpoint["numeric_means"],
        numeric_stds=checkpoint["numeric_stds"],
        category_buckets=int(config["category_buckets"]),
        cache_size=1,
        consensus_loss_mode=str(config.get("consensus_loss_mode", "none")),
        consensus_match_weight=float(config.get("consensus_match_weight", 0.35)),
        nonconsensus_weight=float(config.get("nonconsensus_weight", 2.5)),
        unobserved_nonconsensus_weight=float(config.get("unobserved_nonconsensus_weight", 1.0)),
        max_sequence_loss_weight=float(config.get("max_sequence_loss_weight", 3.0)),
        variable_column_min_entropy=float(config.get("variable_column_min_entropy", 0.05)),
        variable_column_max_consensus=float(config.get("variable_column_max_consensus", 0.92)),
        require_msa_embeddings=uses_msa_embedding_memory(str(config["memory_mode"])),
        msa_embedding_dtype=str(args.msa_embedding_dtype or config.get("msa_embedding_dtype", "float32")),
        max_msa_context_rows=(
            args.max_msa_context_rows
            if args.max_msa_context_rows is not None
            else config.get("max_msa_context_rows")
        ),
        gap_inclusive_msa_mask=uses_gap_inclusive_msa_mask(str(config["memory_mode"])),
        require_target_continuous_embeddings=str(config.get("continuous_target_mode", "token_embedding"))
        == "target_row_embedding",
    )
    collator = RowReconstructionCollator(
        max_sequence_length=int(config["max_sequence_length"]),
        tail_stop_weight=float(config["tail_stop_weight"]),
        profile_feature_mode=str(config.get("profile_feature_mode", "full")),
    )
    batch = collator([dataset[0]])

    first_item = dataset[0]
    target_continuous_dim = int(first_item["target_continuous_embeddings"].shape[-1])
    model = MeanStartCCDDModel(
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
        target_continuous_dim=target_continuous_dim,
        msa_axial_layers=int(config.get("msa_axial_layers", 1)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    baseline_overrides = {
        "kcat_1_per_s": candidate.kcat,
        "topt_C": candidate.topt,
        "tm_C": candidate.tm,
    }
    thermo_overrides = {
        "kcat_1_per_s": candidate.kcat,
        "topt_C": args.thermo_topt,
        "tm_C": args.thermo_tm,
    }
    baseline_batch = set_numeric_override(
        batch,
        labels,
        candidate.kegg_entry,
        checkpoint["numeric_means"],
        checkpoint["numeric_stds"],
        baseline_overrides,
    )
    thermo_batch = set_numeric_override(
        batch,
        labels,
        candidate.kegg_entry,
        checkpoint["numeric_means"],
        checkpoint["numeric_stds"],
        thermo_overrides,
    )
    baseline_sequence = mean_decode(model, baseline_batch, device)
    thermo_sequence = mean_decode(model, thermo_batch, device)
    variant_identity = identity(baseline_sequence, thermo_sequence)
    target_baseline_identity = identity(candidate.target_sequence, baseline_sequence)
    target_thermo_identity = identity(candidate.target_sequence, thermo_sequence)

    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": checkpoint.get("step"),
        "kegg_entry": candidate.kegg_entry,
        "cluster_index": candidate.cluster_index,
        "row_index": candidate.row_index,
        "kcat": candidate.kcat,
        "baseline_topt": candidate.topt,
        "baseline_tm": candidate.tm,
        "thermo_topt": args.thermo_topt,
        "thermo_tm": args.thermo_tm,
        "variant_identity": variant_identity,
        "target_baseline_identity": target_baseline_identity,
        "target_thermo_identity": target_thermo_identity,
        "note": "Current checkpoint has no pH optimum condition. kcat is held fixed by using the same normalized kcat condition for both variants.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_fasta(
        out_dir / "variants.fasta",
        [
            (
                f"baseline kegg={candidate.kegg_entry} cluster={candidate.cluster_index} "
                f"kcat={candidate.kcat:.6g} topt={candidate.topt:.1f} tm={candidate.tm:.1f}",
                baseline_sequence,
            ),
            (
                f"thermostable kegg={candidate.kegg_entry} cluster={candidate.cluster_index} "
                f"kcat={candidate.kcat:.6g} topt={args.thermo_topt:.1f} tm={args.thermo_tm:.1f}",
                thermo_sequence,
            ),
            (
                f"target kegg={candidate.kegg_entry} cluster={candidate.cluster_index}",
                candidate.target_sequence,
            ),
        ],
    )
    (out_dir / "alignment.html").write_text(html_alignment(baseline_sequence, thermo_sequence, metadata), encoding="utf-8")
    aligned_a, aligned_b = needleman_wunsch(baseline_sequence, thermo_sequence)
    marker = "".join("|" if x == y else " " for x, y in zip(aligned_a, aligned_b))
    with (out_dir / "alignment.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"baseline      {aligned_a}\n")
        handle.write(f"              {marker}\n")
        handle.write(f"thermostable  {aligned_b}\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
