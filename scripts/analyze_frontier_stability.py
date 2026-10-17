#!/usr/bin/env python3
"""Summarize finite-grid frontier coverage and seed stability."""
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def f(row, key):
    return float(row[key])


def write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frontier_fraction(rows):
    ordered = sorted(rows, key=lambda row: (f(row, "gpu_pipeline_ms"), f(row, "exact_rank_residual")))
    best = math.inf
    frontier = 0
    for row in ordered:
        quality = f(row, "exact_rank_residual")
        if quality < best - 1e-12:
            frontier += 1
            best = quality
    return frontier / len(rows), frontier


def font(size):
    candidates = []
    if os.environ.get("WINDIR"):
        candidates.append(Path(os.environ["WINDIR"]) / "Fonts" / "arial.ttf")
    candidates.append(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def figure(frontier_rows, stability_rows, output):
    image = Image.new("RGB", (1750, 1020), "white")
    draw = ImageDraw.Draw(image)
    draw.text((875, 35), "Finite-grid frontier coverage and seed stability", anchor="mm", fill="black", font=font(29))
    left = (135, 145, 760, 720)
    right = (920, 145, 1600, 720)
    for box in (left, right):
        x0, y0, x1, y1 = box
        draw.line((x0, y1, x1, y1), fill=(25, 25, 25), width=2)
        draw.line((x0, y0, x0, y1), fill=(25, 25, 25), width=2)
    draw.text((450, 108), "frontier states / all states", anchor="mm", fill="black", font=font(22))
    draw.text((1260, 108), "high-state rho_k CV across seeds", anchor="mm", fill="black", font=font(22))
    labels = [r["matrix"] for r in frontier_rows]
    for i, row in enumerate(frontier_rows):
        group = (left[2] - left[0]) / len(frontier_rows)
        x = left[0] + (i + 0.5) * group
        value = float(row["mean_frontier_fraction"])
        y = left[3] - value * (left[3] - left[1])
        draw.rectangle((x - 24, y, x + 24, left[3]), fill=(42, 86, 160))
        draw.text((x, y - 14), f"{value:.2f}", anchor="ms", fill=(42, 86, 160), font=font(14))
        draw.text((x, left[3] + 18), labels[i], anchor="ma", fill="black", font=font(13))
    for i in range(6):
        value = i / 5
        y = left[3] - value * (left[3] - left[1])
        draw.text((left[0] - 12, y), f"{value:.1f}", anchor="rm", fill=(30, 30, 30), font=font(13))
        if i:
            draw.line((left[0], y, left[2], y), fill=(225, 230, 235), width=1)
    mats = list(dict.fromkeys(r["matrix"] for r in stability_rows))
    ranks = sorted({int(r["rank"]) for r in stability_rows})
    max_cv = max(float(r["cv"] or 0.0) for r in stability_rows) or 1.0
    cell_w = (right[2] - right[0]) / len(ranks)
    cell_h = (right[3] - right[1]) / len(mats)
    for mi, matrix in enumerate(mats):
        draw.text((right[0] - 10, right[1] + (mi + 0.5) * cell_h), matrix, anchor="rm", fill="black", font=font(13))
        for ri, rank in enumerate(ranks):
            row = next(r for r in stability_rows if r["matrix"] == matrix and int(r["rank"]) == rank)
            value = float(row["cv"] or 0.0)
            intensity = int(245 - 190 * min(value / max_cv, 1.0))
            x0 = right[0] + ri * cell_w
            y0 = right[1] + mi * cell_h
            draw.rectangle((x0, y0, x0 + cell_w, y0 + cell_h), fill=(intensity, min(250, intensity + 10), 255), outline=(255, 255, 255))
            draw.text((x0 + cell_w / 2, y0 + cell_h / 2), f"{value:.2g}", anchor="mm", fill="black", font=font(13))
            if mi == 0:
                draw.text((x0 + cell_w / 2, right[1] - 16), str(rank), anchor="ms", fill="black", font=font(13))
    draw.text((875, 960), "Frontier uses computed full-energy rho_k and GPU pipeline time; CV is computed at the maximum tested width and q=2 for each rank", anchor="mm", fill=(75, 75, 75), font=font(16))
    image.save(output, dpi=(300, 300))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read(args.input)
    by_matrix_seed = {}
    for row in rows:
        by_matrix_seed.setdefault((row["matrix"], row["seed"]), []).append(row)
    frontier = []
    for matrix in sorted({r["matrix"] for r in rows}):
        values = [frontier_fraction(group)[0] for (name, _), group in by_matrix_seed.items() if name == matrix]
        frontier.append({
            "matrix": matrix,
            "seeds": len(values),
            "mean_frontier_fraction": statistics.mean(values),
            "sd_frontier_fraction": statistics.stdev(values) if len(values) > 1 else 0.0,
        })
    stability = []
    for matrix in sorted({r["matrix"] for r in rows}):
        for rank in sorted({int(r["rank"]) for r in rows}):
            subset = [r for r in rows if r["matrix"] == matrix and int(r["rank"]) == rank]
            max_width = max(int(r["width"]) for r in subset)
            high_width = [r for r in subset if int(r["width"]) == max_width]
            max_power = max(int(r["power"]) for r in high_width)
            values = [f(r, "exact_rank_residual") for r in high_width if int(r["power"]) == max_power]
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            stability.append({"matrix": matrix, "rank": rank, "seeds": len(values), "mean_rho_k": mean, "sd": sd, "cv": sd / mean if mean else 0.0})
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "frontier_stability_summary.csv", frontier)
    write(args.output / "frontier_seed_stability.csv", stability)
    figure(frontier, stability, args.output / "fig15_frontier_stability.png")
    print(f"wrote {len(frontier)} frontier summaries and {len(stability)} stability rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
