#!/usr/bin/env python3
"""Score strict salt bridges in baseline and thermostable batch folds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
ACID_ATOMS = {
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
}
BASIC_ATOMS = {
    "LYS": ("NZ",),
    "ARG": ("NE", "NH1", "NH2"),
}


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


@dataclass
class Residue:
    chain: str
    resseq: int
    icode: str
    resname: str
    seq_pos: int
    atoms: dict[str, tuple[float, float, float]]

    @property
    def aa(self) -> str:
        return AA3_TO_1.get(self.resname, "X")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="batch_metadata.tsv from generate_thermostability_batch.py")
    parser.add_argument("--fold-dir", required=True, help="ColabFold output directory")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--distance-cutoff", type=float, default=4.0)
    parser.add_argument("--min-mean-plddt", type=float, default=0.0)
    return parser.parse_args()


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def find_one(fold_dir: Path, header: str, kind: str) -> Path | None:
    if kind == "pdb":
        patterns = [
            f"{header}_unrelaxed_rank_001_*.pdb",
            f"{header}_relaxed_rank_001_*.pdb",
            f"{header}*.pdb",
        ]
    elif kind == "scores":
        patterns = [f"{header}_scores_rank_001_*.json", f"{header}*_scores*.json"]
    elif kind == "a3m":
        patterns = [f"{header}.a3m", f"{header}*.a3m"]
    else:
        raise ValueError(kind)
    for pattern in patterns:
        matches = sorted(fold_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def read_scores(path: Path | None) -> dict[str, float | int | str]:
    if not path:
        return {"mean_plddt": math.nan, "ptm": math.nan, "score_path": ""}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"mean_plddt": math.nan, "ptm": math.nan, "score_path": str(path)}
    plddt = data.get("plddt", data.get("plddts"))
    if isinstance(plddt, list) and plddt:
        mean_plddt = float(sum(float(value) for value in plddt) / len(plddt))
    elif isinstance(plddt, (int, float)):
        mean_plddt = float(plddt)
    else:
        mean_plddt = math.nan
    ptm = data.get("ptm", data.get("ptm_score", math.nan))
    try:
        ptm_value = float(ptm)
    except (TypeError, ValueError):
        ptm_value = math.nan
    return {"mean_plddt": mean_plddt, "ptm": ptm_value, "score_path": str(path)}


def count_a3m_records(path: Path | None) -> int:
    if not path:
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for line in handle if line.startswith(">"))
    except OSError:
        return 0


def parse_pdb(path: Path) -> list[Residue]:
    residue_order: list[tuple[str, int, str]] = []
    residues: dict[tuple[str, int, str], Residue] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        altloc = line[16].strip()
        if altloc and altloc != "A":
            continue
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        chain = line[21].strip() or "_"
        try:
            resseq = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        icode = line[26].strip()
        key = (chain, resseq, icode)
        if key not in residues:
            residue_order.append(key)
            residues[key] = Residue(
                chain=chain,
                resseq=resseq,
                icode=icode,
                resname=resname,
                seq_pos=len(residue_order),
                atoms={},
            )
        residues[key].atoms[atom_name] = (x, y, z)
    return [residues[key] for key in residue_order]


def distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def ca_neighbor_counts(residues: list[Residue], radius: float = 10.0) -> dict[int, int]:
    ca_atoms = [(residue.seq_pos, residue.atoms["CA"]) for residue in residues if "CA" in residue.atoms]
    radius_sq = radius * radius
    counts = {seq_pos: 0 for seq_pos, _ in ca_atoms}
    for idx, (seq_pos_i, coord_i) in enumerate(ca_atoms):
        for seq_pos_j, coord_j in ca_atoms[idx + 1 :]:
            dx = coord_i[0] - coord_j[0]
            dy = coord_i[1] - coord_j[1]
            dz = coord_i[2] - coord_j[2]
            if dx * dx + dy * dy + dz * dz <= radius_sq:
                counts[seq_pos_i] += 1
                counts[seq_pos_j] += 1
    return counts


def estimate_kcal(distance_a: float, burial_neighbor_avg: float) -> float:
    if distance_a <= 2.8:
        base = 0.8
    elif distance_a <= 3.2:
        base = 0.6
    elif distance_a <= 3.6:
        base = 0.4
    else:
        base = 0.25
    if burial_neighbor_avg >= 22:
        multiplier = 1.4
    elif burial_neighbor_avg >= 16:
        multiplier = 1.0
    else:
        multiplier = 0.55
    return base * multiplier


def salt_bridges(residues: list[Residue], cutoff: float) -> list[dict[str, Any]]:
    burial = ca_neighbor_counts(residues)
    acids = [residue for residue in residues if residue.resname in ACID_ATOMS]
    basics = [residue for residue in residues if residue.resname in BASIC_ATOMS]
    bridges: list[dict[str, Any]] = []
    for acid in acids:
        acid_atoms = [(name, acid.atoms[name]) for name in ACID_ATOMS[acid.resname] if name in acid.atoms]
        if not acid_atoms:
            continue
        for basic in basics:
            if acid.chain != basic.chain:
                continue
            basic_atoms = [(name, basic.atoms[name]) for name in BASIC_ATOMS[basic.resname] if name in basic.atoms]
            if not basic_atoms:
                continue
            best_distance = math.inf
            best_atom_pair = ""
            for acid_atom, acid_coord in acid_atoms:
                for basic_atom, basic_coord in basic_atoms:
                    current = distance(acid_coord, basic_coord)
                    if current < best_distance:
                        best_distance = current
                        best_atom_pair = f"{acid_atom}-{basic_atom}"
            if best_distance > cutoff:
                continue
            burial_avg = (burial.get(acid.seq_pos, 0) + burial.get(basic.seq_pos, 0)) / 2.0
            score = estimate_kcal(best_distance, burial_avg)
            bridges.append(
                {
                    "acid_pos": acid.seq_pos,
                    "basic_pos": basic.seq_pos,
                    "acid_residue": acid.aa,
                    "basic_residue": basic.aa,
                    "pair": f"{acid.aa}{acid.seq_pos}-{basic.aa}{basic.seq_pos}",
                    "atom_pair": best_atom_pair,
                    "distance_A": best_distance,
                    "burial_neighbor_avg": burial_avg,
                    "estimated_kcal_mol": score,
                }
            )
    bridges.sort(key=lambda row: (row["distance_A"], row["pair"]))
    return bridges


def alignment_maps(
    baseline_sequence: str,
    thermo_sequence: str,
) -> tuple[dict[int, int], dict[int, int], str, str]:
    aligned_base, aligned_thermo = needleman_wunsch(baseline_sequence, thermo_sequence)
    base_map: dict[int, int] = {}
    thermo_map: dict[int, int] = {}
    base_pos = 0
    thermo_pos = 0
    for column, (base_residue, thermo_residue) in enumerate(zip(aligned_base, aligned_thermo)):
        if base_residue != "-":
            base_pos += 1
            base_map[base_pos] = column
        if thermo_residue != "-":
            thermo_pos += 1
            thermo_map[thermo_pos] = column
    return base_map, thermo_map, aligned_base, aligned_thermo


def bridge_key(bridge: dict[str, Any], position_map: dict[int, int]) -> tuple[int, int] | None:
    acid_column = position_map.get(int(bridge["acid_pos"]))
    basic_column = position_map.get(int(bridge["basic_pos"]))
    if acid_column is None or basic_column is None:
        return None
    return tuple(sorted((acid_column, basic_column)))


def summarize_pair(row: dict[str, str], fold_dir: Path, cutoff: float) -> dict[str, Any]:
    baseline_header = row["baseline_header"]
    thermo_header = row["thermo_header"]
    baseline_pdb = find_one(fold_dir, baseline_header, "pdb")
    thermo_pdb = find_one(fold_dir, thermo_header, "pdb")
    baseline_scores = read_scores(find_one(fold_dir, baseline_header, "scores"))
    thermo_scores = read_scores(find_one(fold_dir, thermo_header, "scores"))
    baseline_a3m_depth = count_a3m_records(find_one(fold_dir, baseline_header, "a3m"))
    thermo_a3m_depth = count_a3m_records(find_one(fold_dir, thermo_header, "a3m"))
    common: dict[str, Any] = {
        "family_id": row["family_id"],
        "cluster_index": row["cluster_index"],
        "kegg_entry": row["kegg_entry"],
        "baseline_header": baseline_header,
        "thermo_header": thermo_header,
        "target_length": int(row["target_length"]),
        "baseline_length": int(row["baseline_length"]),
        "thermo_length": int(row["thermo_length"]),
        "variant_identity": float(row["variant_identity"]),
        "target_baseline_identity": float(row["target_baseline_identity"]),
        "target_thermo_identity": float(row["target_thermo_identity"]),
        "baseline_mean_plddt": baseline_scores["mean_plddt"],
        "thermo_mean_plddt": thermo_scores["mean_plddt"],
        "baseline_ptm": baseline_scores["ptm"],
        "thermo_ptm": thermo_scores["ptm"],
        "baseline_a3m_depth": baseline_a3m_depth,
        "thermo_a3m_depth": thermo_a3m_depth,
        "baseline_pdb": str(baseline_pdb or ""),
        "thermo_pdb": str(thermo_pdb or ""),
    }
    if not baseline_pdb or not thermo_pdb:
        return {
            **common,
            "status": "missing_pdb",
            "baseline_strict_bridge_count": 0,
            "thermo_strict_bridge_count": 0,
            "delta_strict_bridge_count": 0,
            "new_strict_bridge_count": 0,
            "lost_strict_bridge_count": 0,
            "baseline_bridge_kcal_mid": math.nan,
            "thermo_bridge_kcal_mid": math.nan,
            "delta_bridge_kcal_mid": math.nan,
            "new_bridge_kcal_mid": math.nan,
            "lost_bridge_kcal_mid": math.nan,
            "mutation_associated_new_bridge_kcal_mid": math.nan,
            "geometry_only_new_bridge_kcal_mid": math.nan,
            "new_bridge_pairs": "",
            "lost_bridge_pairs": "",
        }
    baseline_bridges = salt_bridges(parse_pdb(baseline_pdb), cutoff=cutoff)
    thermo_bridges = salt_bridges(parse_pdb(thermo_pdb), cutoff=cutoff)
    base_map, thermo_map, aligned_base, aligned_thermo = alignment_maps(
        row["baseline_sequence"],
        row["thermo_sequence"],
    )
    baseline_by_key = {
        key: bridge
        for bridge in baseline_bridges
        if (key := bridge_key(bridge, base_map)) is not None
    }
    thermo_by_key = {
        key: bridge
        for bridge in thermo_bridges
        if (key := bridge_key(bridge, thermo_map)) is not None
    }
    new_keys = sorted(set(thermo_by_key) - set(baseline_by_key))
    lost_keys = sorted(set(baseline_by_key) - set(thermo_by_key))
    new_bridges = [thermo_by_key[key] for key in new_keys]
    lost_bridges = [baseline_by_key[key] for key in lost_keys]

    mutation_new_score = 0.0
    geometry_new_score = 0.0
    for key, bridge in zip(new_keys, new_bridges):
        base_chars = [aligned_base[column] if column < len(aligned_base) else "-" for column in key]
        thermo_chars = [aligned_thermo[column] if column < len(aligned_thermo) else "-" for column in key]
        if base_chars != thermo_chars:
            mutation_new_score += float(bridge["estimated_kcal_mol"])
        else:
            geometry_new_score += float(bridge["estimated_kcal_mol"])

    baseline_score = sum(float(bridge["estimated_kcal_mol"]) for bridge in baseline_bridges)
    thermo_score = sum(float(bridge["estimated_kcal_mol"]) for bridge in thermo_bridges)
    new_score = sum(float(bridge["estimated_kcal_mol"]) for bridge in new_bridges)
    lost_score = sum(float(bridge["estimated_kcal_mol"]) for bridge in lost_bridges)
    return {
        **common,
        "status": "ok",
        "baseline_strict_bridge_count": len(baseline_bridges),
        "thermo_strict_bridge_count": len(thermo_bridges),
        "delta_strict_bridge_count": len(thermo_bridges) - len(baseline_bridges),
        "new_strict_bridge_count": len(new_bridges),
        "lost_strict_bridge_count": len(lost_bridges),
        "baseline_bridge_kcal_mid": baseline_score,
        "thermo_bridge_kcal_mid": thermo_score,
        "delta_bridge_kcal_mid": thermo_score - baseline_score,
        "new_bridge_kcal_mid": new_score,
        "lost_bridge_kcal_mid": lost_score,
        "mutation_associated_new_bridge_kcal_mid": mutation_new_score,
        "geometry_only_new_bridge_kcal_mid": geometry_new_score,
        "new_bridge_pairs": ",".join(bridge["pair"] for bridge in new_bridges[:25]),
        "lost_bridge_pairs": ",".join(bridge["pair"] for bridge in lost_bridges[:25]),
        "baseline_bridges": baseline_bridges,
        "thermo_bridges": thermo_bridges,
        "new_bridges": new_bridges,
        "lost_bridges": lost_bridges,
    }


def value_for_tsv(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def write_summary_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    detail_fields = {"baseline_bridges", "thermo_bridges", "new_bridges", "lost_bridges"}
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key in detail_fields:
                continue
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value_for_tsv(row.get(key, "")) for key in fieldnames})


def make_plots(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    valid = [row for row in rows if row.get("status") == "ok"]
    if not valid:
        return
    try:
        import matplotlib
    except ModuleNotFoundError:
        (out_dir / "plotting_skipped.txt").write_text(
            "matplotlib is not installed in this Python environment; plots were skipped.\n",
            encoding="utf-8",
        )
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [float(row["baseline_bridge_kcal_mid"]) for row in valid]
    y = [float(row["thermo_bridge_kcal_mid"]) for row in valid]
    delta = [float(row["delta_bridge_kcal_mid"]) for row in valid]
    labels = [str(row["family_id"]) for row in valid]
    top_indices = sorted(range(len(valid)), key=lambda idx: delta[idx], reverse=True)[:8]

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    scatter = ax.scatter(x, y, c=delta, cmap="coolwarm", edgecolor="#222222", linewidth=0.4, s=54)
    low = min(x + y) - 0.5
    high = max(x + y) + 0.5
    ax.plot([low, high], [low, high], color="#444444", linestyle="--", linewidth=1)
    for idx in top_indices:
        ax.annotate(labels[idx], (x[idx], y[idx]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Baseline strict salt-bridge stability heuristic (kcal/mol)")
    ax.set_ylabel("Thermo variant strict salt-bridge stability heuristic (kcal/mol)")
    ax.set_title("MSA-folded thermostability batch: salt-bridge stability vs baseline")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Variant minus baseline (kcal/mol)")
    fig.tight_layout()
    fig.savefig(out_dir / "salt_bridge_stability_vs_baseline.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.scatter(x, delta, c=delta, cmap="coolwarm", edgecolor="#222222", linewidth=0.4, s=54)
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1)
    for idx in top_indices:
        ax.annotate(labels[idx], (x[idx], delta[idx]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Baseline strict salt-bridge stability heuristic (kcal/mol)")
    ax.set_ylabel("Thermo variant minus baseline (kcal/mol)")
    ax.set_title("Salt-bridge stability gain vs baseline")
    fig.tight_layout()
    fig.savefig(out_dir / "salt_bridge_delta_vs_baseline.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(delta, bins=min(18, max(6, len(delta) // 2)), color="#4C78A8", edgecolor="#222222")
    ax.axvline(0, color="#444444", linestyle="--", linewidth=1)
    ax.set_xlabel("Thermo variant minus baseline salt-bridge heuristic (kcal/mol)")
    ax.set_ylabel("Family count")
    ax.set_title("Distribution of salt-bridge stability deltas")
    fig.tight_layout()
    fig.savefig(out_dir / "salt_bridge_delta_histogram.png", dpi=170)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    valid = [row for row in rows if row.get("status") == "ok"]
    missing = len(rows) - len(valid)
    lines = [
        "# Thermostability Salt-Bridge Batch",
        "",
        (
            "Strict salt bridges are Asp/Glu sidechain oxygens to Lys/Arg sidechain nitrogens "
            "within 4.0 A. The stability score is the same distance plus CA-neighbor burial "
            "heuristic used for the single-family estimate, not a physical electrostatics calculation."
        ),
        "",
        f"- Total families in metadata: {len(rows)}",
        f"- Fold pairs scored: {len(valid)}",
        f"- Missing fold pairs: {missing}",
    ]
    if valid:
        deltas = [float(row["delta_bridge_kcal_mid"]) for row in valid]
        count_deltas = [int(row["delta_strict_bridge_count"]) for row in valid]
        lines.extend(
            [
                f"- Median heuristic delta: {statistics.median(deltas):.3f} kcal/mol",
                f"- Mean heuristic delta: {statistics.mean(deltas):.3f} kcal/mol",
                f"- Positive heuristic deltas: {sum(1 for value in deltas if value > 0)}/{len(deltas)}",
                f"- Median strict bridge-count delta: {statistics.median(count_deltas):.3g}",
                "",
                "Top positive deltas:",
            ]
        )
        for row in sorted(valid, key=lambda item: float(item["delta_bridge_kcal_mid"]), reverse=True)[:10]:
            lines.append(
                "- "
                f"{row['family_id']} {row['kegg_entry']} cluster {row['cluster_index']}: "
                f"delta {float(row['delta_bridge_kcal_mid']):.3f} kcal/mol, "
                f"count delta {int(row['delta_strict_bridge_count'])}, "
                f"mean pLDDT baseline/variant "
                f"{float(row['baseline_mean_plddt']):.1f}/{float(row['thermo_mean_plddt']):.1f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_metadata(Path(args.metadata))
    results = [summarize_pair(row, Path(args.fold_dir), cutoff=args.distance_cutoff) for row in rows]
    if args.min_mean_plddt > 0:
        for row in results:
            if row.get("status") != "ok":
                continue
            values = [float(row["baseline_mean_plddt"]), float(row["thermo_mean_plddt"])]
            if any(math.isnan(value) or value < args.min_mean_plddt for value in values):
                row["status"] = "low_plddt"
    write_summary_tsv(out_dir / "salt_bridge_batch_summary.tsv", results)
    (out_dir / "salt_bridge_batch_details.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    make_plots(out_dir, results)
    write_report(out_dir / "salt_bridge_batch_report.md", results)
    print(f"wrote salt-bridge batch analysis for {len(results)} families to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
