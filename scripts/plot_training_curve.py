#!/usr/bin/env python3
"""Plot sequence decoder training metrics as a dependency-free SVG."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_metric_series(path: Path, metric: str) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                epoch = int(row["epoch"])
                value = float(row[metric])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                series[row.get("split", "train")].append((epoch, value))
    return {split: sorted(points) for split, points in series.items()}


def scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max <= src_min:
        return (dst_min + dst_max) / 2.0
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def polyline(points: list[tuple[float, float]], color: str) -> str:
    if not points:
        return ""
    coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{coords}" />'


def panel(
    title: str,
    series: dict[str, list[tuple[int, float]]],
    metric_label: str,
    x: float,
    y: float,
    width: float,
    height: float,
    colors: dict[str, str],
) -> list[str]:
    all_points = [point for points in series.values() for point in points]
    if not all_points:
        return []
    epochs = [epoch for epoch, _ in all_points]
    values = [value for _, value in all_points]
    x_min, x_max = min(epochs), max(epochs)
    y_min, y_max = min(values), max(values)
    y_pad = max((y_max - y_min) * 0.08, 1.0e-6)
    y_min -= y_pad
    y_max += y_pad
    plot_x = x + 58
    plot_y = y + 36
    plot_w = width - 82
    plot_h = height - 72
    baseline = plot_y + plot_h
    items = [
        f'<text x="{x:.0f}" y="{y + 18:.0f}" class="title">{title}</text>',
        f'<line x1="{plot_x:.0f}" y1="{plot_y:.0f}" x2="{plot_x:.0f}" y2="{baseline:.0f}" class="axis" />',
        f'<line x1="{plot_x:.0f}" y1="{baseline:.0f}" x2="{plot_x + plot_w:.0f}" y2="{baseline:.0f}" class="axis" />',
        f'<text x="{plot_x - 8:.0f}" y="{plot_y + 4:.0f}" text-anchor="end" class="tick">{y_max:.3g}</text>',
        f'<text x="{plot_x - 8:.0f}" y="{baseline + 4:.0f}" text-anchor="end" class="tick">{y_min:.3g}</text>',
        f'<text x="{plot_x:.0f}" y="{baseline + 24:.0f}" class="tick">epoch {x_min}</text>',
        f'<text x="{plot_x + plot_w:.0f}" y="{baseline + 24:.0f}" text-anchor="end" class="tick">epoch {x_max}</text>',
        f'<text x="{plot_x - 46:.0f}" y="{plot_y + plot_h / 2:.0f}" class="label" transform="rotate(-90 {plot_x - 46:.0f} {plot_y + plot_h / 2:.0f})">{metric_label}</text>',
    ]
    for split, points in sorted(series.items()):
        scaled = [
            (
                scale(epoch, x_min, x_max, plot_x, plot_x + plot_w),
                scale(value, y_min, y_max, baseline, plot_y),
            )
            for epoch, value in points
        ]
        items.append(polyline(scaled, colors.get(split, "#555555")))
    legend_x = plot_x + plot_w - 130
    legend_y = plot_y + 6
    for idx, split in enumerate(sorted(series)):
        row_y = legend_y + idx * 18
        color = colors.get(split, "#555555")
        items.append(f'<line x1="{legend_x:.0f}" y1="{row_y:.0f}" x2="{legend_x + 22:.0f}" y2="{row_y:.0f}" stroke="{color}" stroke-width="2.4" />')
        items.append(f'<text x="{legend_x + 28:.0f}" y="{row_y + 4:.0f}" class="tick">{split}</text>')
    return items


def build_svg(metrics_tsv: Path, out_svg: Path) -> None:
    colors = {"train": "#2563eb", "val": "#dc2626", "final": "#16a34a"}
    loss_series = read_metric_series(metrics_tsv, "loss")
    acc_series = read_metric_series(metrics_tsv, "token_accuracy")
    width, height = 920, 560
    items = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="920" height="560" viewBox="0 0 920 560">',
        "<style>",
        "text{font-family:Inter,Arial,sans-serif;fill:#111827}.title{font-size:18px;font-weight:700}.tick{font-size:12px;fill:#4b5563}.label{font-size:12px;fill:#374151}.axis{stroke:#9ca3af;stroke-width:1.2}.bg{fill:#ffffff}.panel{fill:#f9fafb;stroke:#e5e7eb;stroke-width:1}",
        "</style>",
        '<rect class="bg" width="920" height="560" />',
        '<text x="28" y="34" class="title">Sequence decoder training curve</text>',
        '<rect class="panel" x="24" y="54" width="872" height="232" rx="6" />',
        '<rect class="panel" x="24" y="304" width="872" height="232" rx="6" />',
    ]
    items.extend(panel("Loss", loss_series, "loss", 42, 76, 830, 188, colors))
    items.extend(panel("Token accuracy", acc_series, "accuracy", 42, 326, 830, 188, colors))
    items.append("</svg>")
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text("\n".join(items) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-tsv", required=True, help="Metrics TSV from train_sequence_decoder.py.")
    parser.add_argument("--out-svg", required=True, help="Output SVG path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_svg(Path(args.metrics_tsv), Path(args.out_svg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
