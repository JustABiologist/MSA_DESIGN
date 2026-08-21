#!/usr/bin/env python3
"""Score hydrophobic packing and disulfides in baseline/thermo batch folds."""

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
HYDROPHOBIC_RESNAMES = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR"}
BACKBONE_ATOMS = {"N", "CA", "C", "O", "OXT"}


@dataclass
class Residue:
    chain: str
    resseq: int
    icode: str
    resname: str
    seq_pos: int
    atoms: dict[str, tuple[float, float, float]]
    elements: dict[str, str]
    sidechain_sasa: float = math.nan

    @property
    def aa(self) -> str:
        return AA3_TO_1.get(self.resname, "X")

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.chain, self.resseq, self.icode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="batch_metadata.tsv")
    parser.add_argument("--fold-dir", required=True, help="directory with relaxed PDBs and ColabFold score JSONs")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--salt-bridge-summary", default="", help="optional salt_bridge_batch_summary.tsv to join deltas")
    parser.add_argument("--hydrophobic-contact-cutoff", type=float, default=4.8)
    parser.add_argument("--min-seq-separation", type=int, default=3)
    parser.add_argument("--disulfide-min-distance", type=float, default=1.8)
    parser.add_argument("--disulfide-max-distance", type=float, default=2.35)
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
    else:
        raise ValueError(kind)
    for pattern in patterns:
        matches = sorted(fold_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def read_scores(path: Path | None) -> dict[str, float | str]:
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
    try:
        ptm = float(data.get("ptm", data.get("ptm_score", math.nan)))
    except (TypeError, ValueError):
        ptm = math.nan
    return {"mean_plddt": mean_plddt, "ptm": ptm, "score_path": str(path)}


def atom_element(line: str, atom_name: str) -> str:
    element = line[76:78].strip() if len(line) >= 78 else ""
    if element:
        return element.upper()
    stripped = atom_name.strip()
    for char in stripped:
        if char.isalpha():
            return char.upper()
    return ""


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
                elements={},
            )
        residues[key].atoms[atom_name] = (x, y, z)
        residues[key].elements[atom_name] = atom_element(line, atom_name)
    return [residues[key] for key in residue_order]


def annotate_sidechain_sasa(path: Path, residues: list[Residue]) -> None:
    try:
        from Bio.PDB import PDBParser, ShrakeRupley  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        for residue in residues:
            residue.sidechain_sasa = math.nan
        return

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", str(path))
    ShrakeRupley(n_points=100).compute(structure, level="A")
    sasa_by_residue: dict[tuple[str, int, str], float] = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id.strip() or "_"
            for bio_residue in chain:
                hetfield, resseq, icode = bio_residue.id
                if hetfield.strip():
                    continue
                key = (chain_id, int(resseq), icode.strip())
                sidechain_sasa = 0.0
                for atom in bio_residue:
                    atom_name = atom.name.strip()
                    element = (getattr(atom, "element", "") or "").upper()
                    if atom_name in BACKBONE_ATOMS or element == "H":
                        continue
                    sidechain_sasa += float(getattr(atom, "sasa", 0.0))
                sasa_by_residue[key] = sidechain_sasa
    for residue in residues:
        residue.sidechain_sasa = sasa_by_residue.get(residue.key, math.nan)


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


def heavy_sidechain_atoms(residue: Residue) -> list[tuple[str, tuple[float, float, float]]]:
    atoms: list[tuple[str, tuple[float, float, float]]] = []
    for atom_name, coord in residue.atoms.items():
        if atom_name in BACKBONE_ATOMS:
            continue
        if residue.elements.get(atom_name, "").upper() == "H":
            continue
        atoms.append((atom_name, coord))
    return atoms


def burial_multiplier(neighbor_avg: float) -> float:
    if neighbor_avg >= 22:
        return 1.4
    if neighbor_avg >= 16:
        return 1.0
    return 0.55


def contact_closeness(min_distance: float, cutoff: float) -> float:
    if min_distance >= cutoff:
        return 0.0
    return 0.25 + 0.75 * max(0.0, min(1.0, (cutoff - min_distance) / max(0.1, cutoff - 3.2)))


def hydrophobic_contacts(
    residues: list[Residue],
    cutoff: float,
    min_seq_separation: int,
) -> list[dict[str, Any]]:
    burial = ca_neighbor_counts(residues)
    hydrophobic = [
        (residue, heavy_sidechain_atoms(residue))
        for residue in residues
        if residue.resname in HYDROPHOBIC_RESNAMES and heavy_sidechain_atoms(residue)
    ]
    contacts: list[dict[str, Any]] = []
    for idx, (left, left_atoms) in enumerate(hydrophobic):
        for right, right_atoms in hydrophobic[idx + 1 :]:
            if left.chain == right.chain and abs(left.seq_pos - right.seq_pos) <= min_seq_separation:
                continue
            if "CA" in left.atoms and "CA" in right.atoms and distance(left.atoms["CA"], right.atoms["CA"]) > 14.0:
                continue
            best_distance = math.inf
            best_atom_pair = ""
            for left_atom, left_coord in left_atoms:
                for right_atom, right_coord in right_atoms:
                    current = distance(left_coord, right_coord)
                    if current < best_distance:
                        best_distance = current
                        best_atom_pair = f"{left_atom}-{right_atom}"
            if best_distance > cutoff:
                continue
            neighbor_avg = (burial.get(left.seq_pos, 0) + burial.get(right.seq_pos, 0)) / 2.0
            score = contact_closeness(best_distance, cutoff) * burial_multiplier(neighbor_avg)
            left_sasa = left.sidechain_sasa
            right_sasa = right.sidechain_sasa
            both_sasa_buried = (
                math.isfinite(left_sasa)
                and math.isfinite(right_sasa)
                and left_sasa <= 15.0
                and right_sasa <= 15.0
            )
            contacts.append(
                {
                    "left_pos": left.seq_pos,
                    "right_pos": right.seq_pos,
                    "left_residue": left.aa,
                    "right_residue": right.aa,
                    "pair": f"{left.aa}{left.seq_pos}-{right.aa}{right.seq_pos}",
                    "atom_pair": best_atom_pair,
                    "distance_A": best_distance,
                    "burial_neighbor_avg": neighbor_avg,
                    "left_sidechain_sasa_A2": left_sasa,
                    "right_sidechain_sasa_A2": right_sasa,
                    "both_sasa_buried": both_sasa_buried,
                    "packing_score": score,
                }
            )
    contacts.sort(key=lambda row: (row["distance_A"], row["pair"]))
    return contacts


def disulfides(
    residues: list[Residue],
    min_distance: float,
    max_distance: float,
) -> list[dict[str, Any]]:
    cysteines = [residue for residue in residues if residue.resname == "CYS" and "SG" in residue.atoms]
    pairs: list[dict[str, Any]] = []
    for idx, left in enumerate(cysteines):
        for right in cysteines[idx + 1 :]:
            if left.chain == right.chain and abs(left.seq_pos - right.seq_pos) <= 1:
                continue
            current = distance(left.atoms["SG"], right.atoms["SG"])
            if min_distance <= current <= max_distance:
                pairs.append(
                    {
                        "left_pos": left.seq_pos,
                        "right_pos": right.seq_pos,
                        "pair": f"C{left.seq_pos}-C{right.seq_pos}",
                        "distance_A": current,
                    }
                )
    pairs.sort(key=lambda row: (row["distance_A"], row["pair"]))
    return pairs


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


def pair_key(pair: dict[str, Any], position_map: dict[int, int], left_key: str = "left_pos", right_key: str = "right_pos") -> tuple[int, int] | None:
    left_column = position_map.get(int(pair[left_key]))
    right_column = position_map.get(int(pair[right_key]))
    if left_column is None or right_column is None:
        return None
    return tuple(sorted((left_column, right_column)))


def structure_metrics(path: Path, hydrophobic_cutoff: float, min_seq_separation: int, disulfide_min: float, disulfide_max: float) -> dict[str, Any]:
    residues = parse_pdb(path)
    annotate_sidechain_sasa(path, residues)
    hydrophobic_residues = [residue for residue in residues if residue.resname in HYDROPHOBIC_RESNAMES]
    hydrophobic_contacts_list = hydrophobic_contacts(
        residues,
        cutoff=hydrophobic_cutoff,
        min_seq_separation=min_seq_separation,
    )
    disulfide_list = disulfides(residues, min_distance=disulfide_min, max_distance=disulfide_max)
    finite_hydro_sasa = [
        residue.sidechain_sasa
        for residue in hydrophobic_residues
        if math.isfinite(residue.sidechain_sasa)
    ]
    exposed_hydro_count = sum(1 for value in finite_hydro_sasa if value >= 30.0)
    buried_hydro_count = sum(1 for value in finite_hydro_sasa if value <= 15.0)
    contact_score = sum(float(contact["packing_score"]) for contact in hydrophobic_contacts_list)
    buried_contact_count = sum(1 for contact in hydrophobic_contacts_list if contact["both_sasa_buried"])
    hydrophobic_count = len(hydrophobic_residues)
    return {
        "residue_count": len(residues),
        "hydrophobic_residue_count": hydrophobic_count,
        "hydrophobic_sidechain_sasa_A2": sum(finite_hydro_sasa) if finite_hydro_sasa else math.nan,
        "hydrophobic_sidechain_sasa_per_residue_A2": (sum(finite_hydro_sasa) / hydrophobic_count)
        if hydrophobic_count and finite_hydro_sasa
        else math.nan,
        "exposed_hydrophobic_residue_count": exposed_hydro_count,
        "buried_hydrophobic_residue_count": buried_hydro_count,
        "hydrophobic_contact_count": len(hydrophobic_contacts_list),
        "buried_hydrophobic_contact_count": buried_contact_count,
        "hydrophobic_packing_score": contact_score,
        "hydrophobic_packing_density": contact_score / hydrophobic_count if hydrophobic_count else math.nan,
        "hydrophobic_contact_density": len(hydrophobic_contacts_list) / hydrophobic_count if hydrophobic_count else math.nan,
        "disulfide_count": len(disulfide_list),
        "hydrophobic_contacts": hydrophobic_contacts_list,
        "disulfides": disulfide_list,
    }


def high_confidence(row: dict[str, Any]) -> bool:
    values = [
        float(row["baseline_mean_plddt"]),
        float(row["thermo_mean_plddt"]),
        float(row["baseline_ptm"]),
        float(row["thermo_ptm"]),
    ]
    return values[0] >= 90.0 and values[1] >= 90.0 and values[2] >= 0.8 and values[3] >= 0.8


def summarize_pair(
    row: dict[str, str],
    fold_dir: Path,
    salt_by_family: dict[str, dict[str, str]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline_header = row["baseline_header"]
    thermo_header = row["thermo_header"]
    baseline_pdb = find_one(fold_dir, baseline_header, "pdb")
    thermo_pdb = find_one(fold_dir, thermo_header, "pdb")
    baseline_scores = read_scores(find_one(fold_dir, baseline_header, "scores"))
    thermo_scores = read_scores(find_one(fold_dir, thermo_header, "scores"))
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
        "baseline_pdb": str(baseline_pdb or ""),
        "thermo_pdb": str(thermo_pdb or ""),
    }
    salt = salt_by_family.get(row["family_id"], {})
    if salt:
        common["delta_bridge_kcal_mid"] = float(salt.get("delta_bridge_kcal_mid") or math.nan)
        common["delta_strict_bridge_count"] = float(salt.get("delta_strict_bridge_count") or math.nan)
    else:
        common["delta_bridge_kcal_mid"] = math.nan
        common["delta_strict_bridge_count"] = math.nan
    if not baseline_pdb or not thermo_pdb:
        return {**common, "status": "missing_pdb"}

    baseline = structure_metrics(
        baseline_pdb,
        args.hydrophobic_contact_cutoff,
        args.min_seq_separation,
        args.disulfide_min_distance,
        args.disulfide_max_distance,
    )
    thermo = structure_metrics(
        thermo_pdb,
        args.hydrophobic_contact_cutoff,
        args.min_seq_separation,
        args.disulfide_min_distance,
        args.disulfide_max_distance,
    )

    base_map, thermo_map, aligned_base, aligned_thermo = alignment_maps(
        row["baseline_sequence"],
        row["thermo_sequence"],
    )
    baseline_disulfides_by_key = {
        key: bridge
        for bridge in baseline["disulfides"]
        if (key := pair_key(bridge, base_map)) is not None
    }
    thermo_disulfides_by_key = {
        key: bridge
        for bridge in thermo["disulfides"]
        if (key := pair_key(bridge, thermo_map)) is not None
    }
    new_disulfide_keys = sorted(set(thermo_disulfides_by_key) - set(baseline_disulfides_by_key))
    lost_disulfide_keys = sorted(set(baseline_disulfides_by_key) - set(thermo_disulfides_by_key))
    new_disulfides = [thermo_disulfides_by_key[key] for key in new_disulfide_keys]
    lost_disulfides = [baseline_disulfides_by_key[key] for key in lost_disulfide_keys]
    mutation_new_disulfides = 0
    geometry_new_disulfides = 0
    for key in new_disulfide_keys:
        base_chars = [aligned_base[column] if column < len(aligned_base) else "-" for column in key]
        thermo_chars = [aligned_thermo[column] if column < len(aligned_thermo) else "-" for column in key]
        if base_chars != thermo_chars:
            mutation_new_disulfides += 1
        else:
            geometry_new_disulfides += 1

    result: dict[str, Any] = {
        **common,
        "status": "ok",
        "high_confidence": high_confidence(common),
        "new_disulfide_count": len(new_disulfides),
        "lost_disulfide_count": len(lost_disulfides),
        "mutation_associated_new_disulfide_count": mutation_new_disulfides,
        "geometry_only_new_disulfide_count": geometry_new_disulfides,
        "new_disulfide_pairs": ",".join(bridge["pair"] for bridge in new_disulfides),
        "lost_disulfide_pairs": ",".join(bridge["pair"] for bridge in lost_disulfides),
        "baseline_disulfides": baseline["disulfides"],
        "thermo_disulfides": thermo["disulfides"],
        "new_disulfides": new_disulfides,
        "lost_disulfides": lost_disulfides,
    }
    scalar_fields = [
        "residue_count",
        "hydrophobic_residue_count",
        "hydrophobic_sidechain_sasa_A2",
        "hydrophobic_sidechain_sasa_per_residue_A2",
        "exposed_hydrophobic_residue_count",
        "buried_hydrophobic_residue_count",
        "hydrophobic_contact_count",
        "buried_hydrophobic_contact_count",
        "hydrophobic_packing_score",
        "hydrophobic_packing_density",
        "hydrophobic_contact_density",
        "disulfide_count",
    ]
    for field in scalar_fields:
        result[f"baseline_{field}"] = baseline[field]
        result[f"thermo_{field}"] = thermo[field]
        if isinstance(baseline[field], (int, float)) and isinstance(thermo[field], (int, float)):
            result[f"delta_{field}"] = float(thermo[field]) - float(baseline[field])
    result["baseline_top_hydrophobic_contacts"] = baseline["hydrophobic_contacts"][:50]
    result["thermo_top_hydrophobic_contacts"] = thermo["hydrophobic_contacts"][:50]
    return result


def value_for_tsv(value: Any) -> Any:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def write_summary_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    detail_fields = {
        "baseline_disulfides",
        "thermo_disulfides",
        "new_disulfides",
        "lost_disulfides",
        "baseline_top_hydrophobic_contacts",
        "thermo_top_hydrophobic_contacts",
    }
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


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def correlation(xs: list[float], ys: list[float]) -> float:
    paired = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(paired) < 3:
        return math.nan
    x_values = [item[0] for item in paired]
    y_values = [item[1] for item in paired]
    mean_x = statistics.mean(x_values)
    mean_y = statistics.mean(y_values)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in paired)
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in x_values))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in y_values))
    if denom_x == 0 or denom_y == 0:
        return math.nan
    return numerator / (denom_x * denom_y)


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

    colors = ["#2f6fbb" if row.get("high_confidence") else "#a6a6a6" for row in valid]
    edges = ["#123a63" if row.get("high_confidence") else "#5f5f5f" for row in valid]
    labels = [str(row["family_id"]) for row in valid]

    x = [float(row["baseline_hydrophobic_packing_density"]) for row in valid]
    y = [float(row["thermo_hydrophobic_packing_density"]) for row in valid]
    delta_density = [float(row["delta_hydrophobic_packing_density"]) for row in valid]
    top_indices = sorted(range(len(valid)), key=lambda idx: abs(delta_density[idx]), reverse=True)[:8]
    fig, ax = plt.subplots(figsize=(8.2, 6.5), constrained_layout=True)
    scatter = ax.scatter(x, y, c=delta_density, cmap="coolwarm", edgecolor="#222222", linewidth=0.5, s=62)
    low = min(x + y) - 0.05
    high = max(x + y) + 0.05
    ax.plot([low, high], [low, high], color="#444444", linestyle="--", linewidth=1)
    for idx in top_indices:
        ax.annotate(labels[idx], (x[idx], y[idx]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Baseline burial-weighted hydrophobic packing density")
    ax.set_ylabel("Thermo variant burial-weighted hydrophobic packing density")
    ax.set_title("Hydrophobic packing density: thermo variant vs baseline")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Thermo minus baseline")
    fig.savefig(out_dir / "hydrophobic_packing_density_vs_baseline.png", dpi=170)
    plt.close(fig)

    salt_delta = [float(row.get("delta_bridge_kcal_mid", math.nan)) for row in valid]
    fig, ax = plt.subplots(figsize=(8.3, 6.3), constrained_layout=True)
    ax.scatter(delta_density, salt_delta, c=colors, edgecolors=edges, linewidth=0.7, s=70, alpha=0.9)
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1)
    ax.axvline(0, color="#444444", linestyle="--", linewidth=1)
    for idx in sorted(range(len(valid)), key=lambda idx: abs(delta_density[idx]) + abs(salt_delta[idx]), reverse=True)[:8]:
        ax.annotate(labels[idx], (delta_density[idx], salt_delta[idx]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    all_r = correlation(delta_density, salt_delta)
    hc_density = [float(row["delta_hydrophobic_packing_density"]) for row in valid if row.get("high_confidence")]
    hc_salt = [float(row.get("delta_bridge_kcal_mid", math.nan)) for row in valid if row.get("high_confidence")]
    hc_r = correlation(hc_density, hc_salt)
    ax.text(
        0.02,
        0.98,
        f"all r={all_r:+.2f}\nhigh-conf r={hc_r:+.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    ax.set_xlabel("Delta hydrophobic packing density")
    ax.set_ylabel("Delta salt-bridge heuristic (kcal/mol)")
    ax.set_title("Hydrophobic packing gain vs salt-bridge gain")
    ax.grid(True, color="#e4e4e4", linewidth=0.8)
    fig.savefig(out_dir / "hydrophobic_packing_delta_vs_salt_bridge_delta.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    ax.hist(delta_density, bins=min(18, max(6, len(delta_density) // 2)), color="#4C78A8", edgecolor="#222222")
    ax.axvline(0, color="#444444", linestyle="--", linewidth=1)
    ax.set_xlabel("Delta burial-weighted hydrophobic packing density")
    ax.set_ylabel("Family count")
    ax.set_title("Distribution of hydrophobic packing-density deltas")
    fig.savefig(out_dir / "hydrophobic_packing_delta_histogram.png", dpi=170)
    plt.close(fig)

    exposed_delta = [float(row["delta_hydrophobic_sidechain_sasa_per_residue_A2"]) for row in valid]
    fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    ax.hist(exposed_delta, bins=min(18, max(6, len(exposed_delta) // 2)), color="#59A14F", edgecolor="#222222")
    ax.axvline(0, color="#444444", linestyle="--", linewidth=1)
    ax.set_xlabel("Delta hydrophobic side-chain SASA per hydrophobic residue (A^2)")
    ax.set_ylabel("Family count")
    ax.set_title("Distribution of exposed hydrophobic side-chain SASA deltas")
    fig.savefig(out_dir / "hydrophobic_sasa_delta_histogram.png", dpi=170)
    plt.close(fig)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    valid = [row for row in rows if row.get("status") == "ok"]
    hc = [row for row in valid if row.get("high_confidence")]
    lines = [
        "# Hydrophobic Packing and Disulfide Batch",
        "",
        "Hydrophobic packing is a geometric heuristic on Amber-relaxed AF2 models, not a physical free energy.",
        "Hydrophobic contacts are side-chain heavy-atom contacts between A/V/I/L/M/F/W/Y residues within the configured cutoff, excluding near sequence neighbors.",
        "Packing score weights contact closeness by a CA-neighbor burial heuristic; packing density divides that score by hydrophobic residue count.",
        "Hydrophobic side-chain SASA is computed with Biopython Shrake-Rupley when available.",
        "",
        f"- Total families in metadata: {len(rows)}",
        f"- Fold pairs scored: {len(valid)}",
        f"- High-confidence pairs: {len(hc)}",
    ]
    if valid:
        delta_density = finite_values(valid, "delta_hydrophobic_packing_density")
        delta_contacts = finite_values(valid, "delta_hydrophobic_contact_count")
        delta_sasa = finite_values(valid, "delta_hydrophobic_sidechain_sasa_per_residue_A2")
        lines.extend(
            [
                f"- Median delta packing density: {statistics.median(delta_density):+.4f}",
                f"- Mean delta packing density: {statistics.mean(delta_density):+.4f}",
                f"- Positive delta packing density: {sum(value > 0 for value in delta_density)}/{len(delta_density)}",
                f"- Median delta hydrophobic contact count: {statistics.median(delta_contacts):+.3g}",
                f"- Median delta hydrophobic side-chain SASA per hydrophobic residue: {statistics.median(delta_sasa):+.3f} A^2",
                f"- Families with new disulfides: {sum(int(row.get('new_disulfide_count', 0)) > 0 for row in valid)}/{len(valid)}",
                f"- Families with lost disulfides: {sum(int(row.get('lost_disulfide_count', 0)) > 0 for row in valid)}/{len(valid)}",
            ]
        )
    if hc:
        delta_density = finite_values(hc, "delta_hydrophobic_packing_density")
        delta_sasa = finite_values(hc, "delta_hydrophobic_sidechain_sasa_per_residue_A2")
        lines.extend(
            [
                "",
                "High-confidence subset:",
                f"- Median delta packing density: {statistics.median(delta_density):+.4f}",
                f"- Mean delta packing density: {statistics.mean(delta_density):+.4f}",
                f"- Positive delta packing density: {sum(value > 0 for value in delta_density)}/{len(delta_density)}",
                f"- Median delta hydrophobic side-chain SASA per hydrophobic residue: {statistics.median(delta_sasa):+.3f} A^2",
            ]
        )
    if valid:
        lines.extend(["", "Top packing-density gains:"])
        for row in sorted(valid, key=lambda item: float(item["delta_hydrophobic_packing_density"]), reverse=True)[:10]:
            lines.append(
                "- "
                f"{row['family_id']} {row['kegg_entry']} cluster {row['cluster_index']}: "
                f"delta density {float(row['delta_hydrophobic_packing_density']):+.4f}, "
                f"delta contacts {float(row['delta_hydrophobic_contact_count']):+.0f}, "
                f"delta hydrophobic SASA/res {float(row['delta_hydrophobic_sidechain_sasa_per_residue_A2']):+.2f} A^2, "
                f"pLDDT {float(row['baseline_mean_plddt']):.1f}/{float(row['thermo_mean_plddt']):.1f}"
            )
        lines.extend(["", "Worst packing-density losses:"])
        for row in sorted(valid, key=lambda item: float(item["delta_hydrophobic_packing_density"]))[:10]:
            lines.append(
                "- "
                f"{row['family_id']} {row['kegg_entry']} cluster {row['cluster_index']}: "
                f"delta density {float(row['delta_hydrophobic_packing_density']):+.4f}, "
                f"delta contacts {float(row['delta_hydrophobic_contact_count']):+.0f}, "
                f"delta hydrophobic SASA/res {float(row['delta_hydrophobic_sidechain_sasa_per_residue_A2']):+.2f} A^2, "
                f"pLDDT {float(row['baseline_mean_plddt']):.1f}/{float(row['thermo_mean_plddt']):.1f}"
            )
        disulfide_rows = [
            row
            for row in valid
            if int(row.get("new_disulfide_count", 0)) > 0 or int(row.get("lost_disulfide_count", 0)) > 0
        ]
        lines.extend(["", "Disulfide changes:"])
        if disulfide_rows:
            for row in disulfide_rows:
                lines.append(
                    "- "
                    f"{row['family_id']} {row['kegg_entry']}: "
                    f"new {row.get('new_disulfide_pairs', '') or 'none'}, "
                    f"lost {row.get('lost_disulfide_pairs', '') or 'none'}"
                )
        else:
            lines.append("- No new or lost strict SG-SG disulfides detected.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_salt_summary(path: str) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    summary_path = Path(path)
    if not summary_path.exists():
        return {}
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return {row["family_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_metadata(Path(args.metadata))
    salt_by_family = read_salt_summary(args.salt_bridge_summary)
    results = [
        summarize_pair(row, Path(args.fold_dir), salt_by_family=salt_by_family, args=args)
        for row in rows
    ]
    write_summary_tsv(out_dir / "hydrophobic_packing_batch_summary.tsv", results)
    (out_dir / "hydrophobic_packing_batch_details.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    make_plots(out_dir, results)
    write_report(out_dir / "hydrophobic_packing_batch_report.md", results)
    print(f"wrote hydrophobic packing batch analysis for {len(results)} families to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
