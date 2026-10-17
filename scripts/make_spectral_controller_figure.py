#!/usr/bin/env python3
"""Render the CPU controlled-spectrum controller summary."""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = [(42, 86, 160), (214, 91, 73), (49, 126, 83), (154, 94, 166), (210, 143, 44)]


def font(size: int):
    windir = os.environ.get("WINDIR")
    candidates = [Path(windir) / "Fonts" / "arial.ttf"] if windir else []
    candidates.append(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def read(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def bar_panel(draw, box, labels, values, title, ylabel, colors, ymax, log_y=False, scientific=False):
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=(25, 25, 25), width=2)
    draw.line((x0, y0, x0, y1), fill=(25, 25, 25), width=2)
    draw.text(((x0 + x1) / 2, y0 - 34), title, fill="black", anchor="mm", font=font(24))
    # The panel title already identifies the metric.  Keep the plot area clear
    # at manuscript width; long vertical labels collide with tick text after
    # Word scales the figure.
    for i in range(5):
        frac = i / 4
        if log_y:
            ymin = 0.02
            value = math.exp(math.log(ymin) + frac * (math.log(ymax) - math.log(ymin)))
            y = y1 - frac * (y1 - y0)
        else:
            value = frac * ymax
            y = y1 - frac * (y1 - y0)
        draw.line((x0 - 7, y, x0, y), fill=(25, 25, 25), width=1)
        label = f"{value:.1e}" if scientific else (f"{value:.2g}" if log_y else f"{value:.3g}")
        draw.text((x0 - 13, y), label, fill="black", anchor="rm", font=font(13))
        if i:
            draw.line((x0, y, x1, y), fill=(225, 230, 235), width=1)
    group = (x1 - x0) / len(labels)
    for i, (label, value) in enumerate(zip(labels, values)):
        center = x0 + (i + 0.5) * group
        if log_y:
            ymin = 0.02
            height = min(max((math.log(max(float(value), ymin)) - math.log(ymin)) / (math.log(ymax) - math.log(ymin)), 0.0), 1.0) * (y1 - y0)
        else:
            height = min(float(value) / ymax, 1.0) * (y1 - y0)
        draw.rectangle((center - 30, y1 - height, center + 30, y1), fill=colors[i])
        draw.text((center, y1 - height - 10), f"{float(value):.3g}", fill=colors[i], anchor="ms", font=font(13))
        draw.text((center, y1 + 20), label, fill="black", anchor="ma", font=font(13))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read(args.summary.resolve())
    labels = [row["family"] for row in rows]
    image = Image.new("RGB", (1600, 920), "white")
    draw = ImageDraw.Draw(image)
    max_ratio = max(float(r["mean_controller_cpu_ms"]) / max(float(r["mean_oracle_cpu_ms"]), 1e-9) for r in rows)
    ratios = [float(r["mean_controller_cpu_ms"]) / max(float(r["mean_oracle_cpu_ms"]), 1e-9) for r in rows]
    gaps = [max(0.0, float(r["mean_controller_rho_k"]) - float(r["mean_oracle_rho_k"])) for r in rows]
    bar_panel(draw, (115, 120, 740, 700), labels, ratios, "Action-path cost relative to best tested state", "cost ratio", COLORS, max(1.0, max_ratio * 1.12), log_y=True)
    bar_panel(draw, (860, 120, 1485, 700), labels, gaps, "Endpoint gap to best tested state", "rho_k gap", COLORS, max(1e-4, max(gaps) * 1.2), scientific=True)
    draw.text((800, 35), "Controlled-spectrum finite-grid action analysis", fill="black", anchor="mm", font=font(28))
    draw.text((800, 790), "Five families, five seeds, FP64 CPU states; lower is better. CPU timings are mechanism-study timings, not GPU results.", fill=(70, 70, 70), anchor="mm", font=font(16))
    draw.text((800, 830), "Reference: lowest-rho_k state evaluated for each family and seed.", fill=(70, 70, 70), anchor="mm", font=font(16))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output.resolve(), dpi=(300, 300))
    print(f"wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
