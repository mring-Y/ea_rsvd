#!/usr/bin/env python3
"""Render the controlled-spectrum stress figure from CSV output."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = [(42, 86, 160), (214, 91, 73), (49, 126, 83), (154, 94, 166), (210, 143, 44)]


def font(size: int):
    candidates = []
    if os.environ.get("WINDIR"):
        candidates.append(Path(os.environ["WINDIR"]) / "Fonts" / "arial.ttf")
    candidates.append(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def read(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def draw_axes(draw, box, title, xlabel, ylabel, ymax):
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=(25, 25, 25), width=2)
    draw.line((x0, y0, x0, y1), fill=(25, 25, 25), width=2)
    draw.text(((x0 + x1) / 2, y0 - 32), title, anchor="mm", fill="black", font=font(24))
    draw.text(((x0 + x1) / 2, y1 + 38), xlabel, anchor="mm", fill="black", font=font(18))
    draw.text((x0 - 58, (y0 + y1) / 2), ylabel, anchor="mm", fill="black", font=font(18))
    for i in range(6):
        value = ymax * i / 5.0
        y = y1 - (y1 - y0) * value / ymax
        draw.line((x0 - 6, y, x0, y), fill=(25, 25, 25), width=1)
        draw.text((x0 - 12, y), f"{value:.2f}", anchor="rm", fill=(30, 30, 30), font=font(13))
        if i:
            draw.line((x0, y, x1, y), fill=(225, 230, 235), width=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read(args.input)
    families = list(dict.fromkeys(row["family"] for row in rows))
    image = Image.new("RGB", (1700, 980), "white")
    draw = ImageDraw.Draw(image)
    draw.text((850, 36), "CPU spectral stress: action depends on singular-value decay", anchor="mm", fill="black", font=font(29))
    panels = [(135, 135, 755, 700), (900, 135, 1520, 700)]
    draw_axes(draw, panels[0], "q response at ell=64, k=16", "power steps q", "rho_k", 1.02)
    draw_axes(draw, panels[1], "rank response at ell=64, q=2", "target rank k", "rho_k", 1.02)
    for i, family in enumerate(families):
        color = COLORS[i % len(COLORS)]
        qrows = [r for r in rows if r["family"] == family and r["sketch_width"] == "64" and r["target_rank"] == "16"]
        qrows = sorted(qrows, key=lambda r: int(r["power_steps"]))
        points = []
        for r in qrows:
            x = panels[0][0] + (panels[0][2] - panels[0][0]) * int(r["power_steps"]) / 2.0
            y = panels[0][3] - (panels[0][3] - panels[0][1]) * float(r["rho_k"]) / 1.02
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        krows = [r for r in rows if r["family"] == family and r["sketch_width"] == "64" and r["power_steps"] == "2"]
        krows = sorted(krows, key=lambda r: int(r["target_rank"]))
        points = []
        for r in krows:
            x = panels[1][0] + (panels[1][2] - panels[1][0]) * (int(r["target_rank"]) - 8) / 24.0
            y = panels[1][3] - (panels[1][3] - panels[1][1]) * float(r["rho_k"]) / 1.02
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.rectangle((x - 5, y - 5, x + 5, y + 5), fill=color)
        legend_y = 760 + i * 31
        draw.line((170, legend_y, 210, legend_y), fill=color, width=4)
        draw.text((225, legend_y), family, anchor="lm", fill="black", font=font(17))
    draw.text((850, 930), "Five seeds per family; dense FP64 construction; quality mechanism only, no GPU timing", anchor="mm", fill=(75, 75, 75), font=font(16))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, dpi=(300, 300))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
