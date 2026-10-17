#!/usr/bin/env python3
"""Render the non-LP structure-sensitivity quality/cost figure."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "ca-AstroPh": (42, 86, 160),
    "ca-HepPh": (214, 91, 73),
    "email-EuAll": (49, 126, 83),
    "p2p-Gnutella31": (154, 94, 166),
}


def rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def number(row, key):
    return float(row[key])


def font(size: int):
    windir = os.environ.get("WINDIR")
    candidates = []
    if windir:
        candidates += [Path(windir) / "Fonts" / "arial.ttf", Path(windir) / "Fonts" / "times.ttf"]
    candidates += [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def map_x(q, box):
    return box[0] + q / 2 * (box[2] - box[0])


def map_y(value, box, lo, hi):
    return box[3] - (value - lo) / (hi - lo) * (box[3] - box[1])


def axes(draw, box, xlabel, ylabel, title, lo, hi):
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=(30, 30, 30), width=2)
    draw.line((x0, y0, x0, y1), fill=(30, 30, 30), width=2)
    draw.text(((x0 + x1) / 2, 32), title, fill="black", anchor="mm", font=font(25))
    draw.text(((x0 + x1) / 2, y1 + 50), xlabel, fill="black", anchor="mm", font=font(20))
    # The panel title and tick values already identify the vertical quantity.
    # Omitting a long horizontal y-label avoids collision with tick labels
    # after the figure is reduced to manuscript width.
    for q in (0, 1, 2):
        x = map_x(q, box)
        draw.line((x, y1, x, y1 + 8), fill=(30, 30, 30), width=1)
        draw.text((x, y1 + 23), str(q), fill="black", anchor="ma", font=font(16))
    for frac in (0, .25, .5, .75, 1):
        y = y1 - frac * (y1 - y0)
        value = lo + frac * (hi - lo)
        draw.line((x0 - 8, y, x0, y), fill=(30, 30, 30), width=1)
        draw.text((x0 - 14, y), f"{value:.3g}", fill="black", anchor="rm", font=font(14))


def draw_gap_figure(data, output: Path) -> None:
    """Render the residual pair and its truncation gap for the same states."""
    im = Image.new("RGB", (1750, 980), "white")
    draw = ImageDraw.Draw(im)
    left, right = (145, 105, 805, 720), (945, 105, 1605, 720)
    all_residuals = [number(r, "mean_projection_residual") for r in data]
    all_residuals += [number(r, "mean_rank_k_residual") for r in data]
    lo = max(0.0, min(all_residuals) - 0.03)
    hi = min(1.05, max(all_residuals) + 0.03)
    axes(draw, left, "Power depth q", "residual", "Residual pair by power depth", lo, hi)
    gaps = [number(r, "mean_rank_k_residual") - number(r, "mean_projection_residual") for r in data]
    gap_hi = max(0.02, max(gaps) * 1.25)
    axes(draw, right, "Power depth q", "rho_k-rho_l", "Projected-factor truncation gap", 0.0, gap_hi)
    for matrix, color in COLORS.items():
        group = sorted((r for r in data if r["matrix"] == matrix and r["status"] == "PASS"), key=lambda r: int(r["power_steps"]))
        projection_points, rank_points, gap_points = [], [], []
        for row in group:
            q = int(row["power_steps"])
            x = map_x(q, left)
            projection_points.append((x, map_y(number(row, "mean_projection_residual"), left, lo, hi)))
            rank_points.append((x, map_y(number(row, "mean_rank_k_residual"), left, lo, hi)))
            gap_points.append((map_x(q, right), map_y(number(row, "mean_rank_k_residual") - number(row, "mean_projection_residual"), right, 0.0, gap_hi)))
        if len(group) > 1:
            draw.line(projection_points, fill=color, width=3)
            draw.line(rank_points, fill=color, width=3)
            draw.line(gap_points, fill=color, width=3)
        for point in projection_points:
            draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=color)
        for point in rank_points:
            draw.rectangle((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=color)
        for point in gap_points:
            draw.ellipse((point[0] - 6, point[1] - 6, point[0] + 6, point[1] + 6), fill=color)
    legend_y = 795
    for index, (matrix, color) in enumerate(COLORS.items()):
        x = 165 + index * 365
        draw.ellipse((x, legend_y - 7, x + 14, legend_y + 7), fill=color)
        draw.text((x + 24, legend_y), matrix, fill="black", anchor="lm", font=font(17))
    draw.text((400, 860), "circle = rho_l; square = rho_k", fill=(70, 70, 70), anchor="mm", font=font(17))
    draw.text((1230, 860), "k=16, ell=24; lower residual is better", fill=(70, 70, 70), anchor="mm", font=font(17))
    output.parent.mkdir(parents=True, exist_ok=True)
    im.save(output, dpi=(300, 300))
    print(f"wrote {output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gap-output", type=Path,
                        help="optional second figure showing the residual pair and rho_k-rho_l")
    args = parser.parse_args()
    data = rows(args.summary.resolve())
    im = Image.new("RGB", (1600, 950), "white")
    draw = ImageDraw.Draw(im)
    left, right = (130, 100, 750, 710), (850, 100, 1470, 710)
    axes(draw, left, "Power depth q", "rho_k", "Non-LP quality by power depth", .60, 1.02)
    axes(draw, right, "Power depth q", "GPU time (ms)", "Non-LP cost by power depth", 60, 150)
    for matrix, color in COLORS.items():
        group = sorted((r for r in data if r["matrix"] == matrix and r["status"] == "PASS"), key=lambda r: int(r["power_steps"]))
        quality_points, time_points = [], []
        for row in group:
            q = int(row["power_steps"])
            xq = map_x(q, left); xt = map_x(q, right)
            yq = map_y(number(row, "mean_rank_k_residual"), left, .60, 1.02)
            yt = map_y(number(row, "mean_gpu_pipeline_ms"), right, 60, 150)
            quality_points.append((xq, yq)); time_points.append((xt, yt))
            draw.ellipse((xq - 7, yq - 7, xq + 7, yq + 7), fill=color)
            draw.ellipse((xt - 7, yt - 7, xt + 7, yt + 7), fill=color)
        if len(quality_points) > 1:
            draw.line(quality_points, fill=color, width=3)
            draw.line(time_points, fill=color, width=3)
    legend_y = 795
    for index, (matrix, color) in enumerate(COLORS.items()):
        x = 165 + index * 340
        draw.ellipse((x, legend_y - 7, x + 14, legend_y + 7), fill=color)
        draw.text((x + 24, legend_y), matrix, fill="black", anchor="lm", font=font(17))
    draw.text((800, 880), "k=16, ell=24; two seeds per matrix; lower residual is better", fill=(70, 70, 70), anchor="mm", font=font(17))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    im.save(args.output, dpi=(300, 300))
    print(f"wrote {args.output}")
    if args.gap_output:
        draw_gap_figure(data, args.gap_output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
