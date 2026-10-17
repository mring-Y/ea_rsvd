#!/usr/bin/env python3
"""Summarize and plot the public non-LP rank/power sweep."""
from __future__ import annotations

import argparse
import csv
import os
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [row for row in csv.DictReader(handle) if row.get("status") == "ok"]


def font(size: int):
    for candidate in (Path(os.environ.get("WINDIR", "")) / "Fonts" / "arial.ttf",
                      Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups = {}
    for row in rows:
        key = (row["matrix"], row["rank"], row["power_steps"])
        groups.setdefault(key, []).append(row)
    output = []
    for (matrix, rank, power), subset in sorted(groups.items(), key=lambda item: (item[0][0], int(item[0][1]), int(item[0][2]))):
        output.append({
            "matrix": matrix, "rank": int(rank), "power_steps": int(power),
            "seeds": len(subset),
            "mean_rho_l": statistics.mean(float(r["exact_projection_residual"]) for r in subset),
            "sd_rho_l": statistics.stdev(float(r["exact_projection_residual"]) for r in subset) if len(subset) > 1 else 0.0,
            "mean_rho_k": statistics.mean(float(r["exact_rank_residual"]) for r in subset),
            "sd_rho_k": statistics.stdev(float(r["exact_rank_residual"]) for r in subset) if len(subset) > 1 else 0.0,
            "mean_gpu_ms": statistics.mean(float(r["gpu_pipeline_ms"]) for r in subset),
            "max_q_error": max(float(r["q_orthogonality_max_error"]) for r in subset),
        })
    return output


def draw(summary: list[dict[str, object]], output: Path) -> None:
    image = Image.new("RGB", (1800, 1000), "white")
    d = ImageDraw.Draw(image)
    title, label, tick = font(28), font(19), font(14)
    d.text((900, 35), "Non-LP rank and power sweep", anchor="mm", fill="black", font=title)
    colors = {"ca-GrQc": (43, 108, 176), "ca-HepPh": (194, 59, 34), "email-Enron": (82, 141, 78)}
    panels = [(140, 115, 830, 735, "mean computed rho_k at q=2", "rank"),
              (970, 115, 1660, 735, "mean computed rho_k at k=16", "power")]
    matrices = sorted({str(r["matrix"]) for r in summary})
    for left, top, right, bottom, heading, mode in panels:
        d.text(((left + right) / 2, top - 34), heading, anchor="mm", fill="black", font=label)
        d.line((left, bottom, right, bottom), fill="black", width=2)
        d.line((left, top, left, bottom), fill="black", width=2)
        ymax = 1.0
        for i in range(6):
            value = ymax * i / 5
            y = bottom - value * (bottom - top)
            d.line((left, y, right, y), fill=(225, 230, 235), width=1)
            d.text((left - 12, y), f"{value:.1f}", anchor="rm", fill="black", font=tick)
        category_count = 3
        group = (right - left) / max(len(matrices), 1)
        index = 0
        for matrix in matrices:
            rows = [r for r in summary if r["matrix"] == matrix and ((mode == "rank" and r["power_steps"] == 2) or (mode == "power" and r["rank"] == 16))]
            rows.sort(key=lambda r: int(r["rank"] if mode == "rank" else r["power_steps"]))
            offset = group * 0.17
            bar_width = group * 0.18
            for row_index, row in enumerate(rows):
                value = float(row["mean_rho_k"])
                x = left + index * group + offset + row_index * (bar_width + group * 0.04) + bar_width / 2
                y = bottom - value * (bottom - top)
                d.rectangle((x - bar_width / 2, y, x + bar_width / 2, bottom), fill=colors[matrix])
                d.text((x, y - 8), f"{value:.3f}", anchor="ms", fill=colors[matrix], font=tick)
                label_text = str(row["rank"]) if mode == "rank" else str(row["power_steps"])
                d.text((x, bottom + 16), label_text, anchor="ma", fill="black", font=tick)
            d.text((left + index * group + group / 2, bottom + 43), matrix, anchor="ma", fill="black", font=tick)
            index += 1
        d.text((left + 15, bottom + 54), "rank" if mode == "rank" else "q", fill=(70, 70, 70), font=tick)
    x = 170
    for matrix in matrices:
        d.rectangle((x, 850, x + 18, 868), fill=colors[matrix])
        d.text((x + 28, 859), matrix, anchor="lm", fill="black", font=tick)
        x += 250
    d.text((900, 945), "Two seeds per state; full-energy/no-probe CUDA residuals; bars are means", anchor="mm", fill=(75, 75, 75), font=tick)
    image.save(output, dpi=(300, 300))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = summarize(read(args.input.resolve()))
    with (args.output / "nonlp_rank_sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(summary[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    draw(summary, args.output / "fig22_nonlp_rank_sweep.png")
    print(f"wrote {len(summary)} summary rows and {args.output / 'fig22_nonlp_rank_sweep.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
