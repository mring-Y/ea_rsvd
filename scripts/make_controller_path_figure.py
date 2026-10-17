#!/usr/bin/env python3
"""Render the controller path replay summary as a publication figure."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def font(size: int):
    candidates = []
    windir = __import__("os").environ.get("WINDIR")
    if windir:
        candidates.append(Path(windir) / "Fonts" / "arial.ttf")
    candidates.append(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def axis(draw, box, heading, ymin, ymax, formatter):
    left, top, right, bottom = box
    draw.text(((left + right) / 2, top - 32), heading, anchor="mm", fill="black", font=font(20))
    draw.line((left, bottom, right, bottom), fill=(25, 25, 25), width=2)
    draw.line((left, top, left, bottom), fill=(25, 25, 25), width=2)
    for i in range(6):
        value = ymin + i * (ymax - ymin) / 5
        y = bottom - (value - ymin) / max(ymax - ymin, 1e-12) * (bottom - top)
        draw.line((left - 7, y, left, y), fill=(25, 25, 25), width=1)
        draw.text((left - 14, y), formatter(value), anchor="rm", fill="black", font=font(15))
        if i:
            draw.line((left, y, right, y), fill=(225, 230, 235), width=1)


def draw(summary: list[dict[str, str]], output: Path) -> None:
    image = Image.new("RGB", (1800, 980), "white")
    canvas = ImageDraw.Draw(image)
    canvas.text((900, 36), "Residual-pair finite-grid action path", anchor="mm", fill="black", font=font(30))
    labels = [row["matrix"] for row in summary]
    left_box = (135, 125, 820, 700)
    right_box = (980, 125, 1665, 700)
    max_rho = max(max(float(row["mean_final_rho_k"]), float(row["mean_oracle_rho_k"])) for row in summary)
    axis(canvas, left_box, "endpoint rho_k (lower is better)", 0.0, max(max_rho * 1.2, 0.1), lambda x: f"{x:.2f}")
    axis(canvas, right_box, "cumulative path cost / best-visited-state cost", 0.0, max(float(row["mean_cost_ratio_to_oracle"]) for row in summary) * 1.2, lambda x: f"{x:.1f}x")
    group = (left_box[2] - left_box[0]) / len(summary)
    for idx, row in enumerate(summary):
        center = left_box[0] + (idx + 0.5) * group
        for offset, key, color, label in [(-20, "mean_final_rho_k", (43, 108, 176), "path"), (20, "mean_oracle_rho_k", (110, 110, 110), "best visited")]:
            value = float(row[key])
            y = left_box[3] - value / max(max_rho * 1.2, 0.1) * (left_box[3] - left_box[1])
            canvas.rectangle((center + offset - 15, y, center + offset + 15, left_box[3]), fill=color)
            canvas.text((center + offset, y - 10), f"{value:.3f}", anchor="ma", fill=color, font=font(14))
        canvas.text((center, left_box[3] + 20), labels[idx], anchor="ma", fill="black", font=font(15))
    group = (right_box[2] - right_box[0]) / len(summary)
    max_cost = max(float(row["mean_cost_ratio_to_oracle"]) for row in summary) * 1.2
    for idx, row in enumerate(summary):
        center = right_box[0] + (idx + 0.5) * group
        value = float(row["mean_cost_ratio_to_oracle"])
        y = right_box[3] - value / max(max_cost, 1e-12) * (right_box[3] - right_box[1])
        canvas.rectangle((center - 18, y, center + 18, right_box[3]), fill=(194, 59, 34))
        canvas.text((center, y - 10), f"{value:.2f}x", anchor="ma", fill=(194, 59, 34), font=font(14))
        canvas.text((center, right_box[3] + 20), labels[idx], anchor="ma", fill="black", font=font(15))
    canvas.text((170, 805), "blue: action-path endpoint", fill=(43, 108, 176), font=font(18))
    canvas.text((470, 805), "gray: best state visited on the path", fill=(110, 110, 110), font=font(18))
    canvas.text((900, 900), "Computed full-energy state table", anchor="mm", fill=(70, 70, 70), font=font(16))
    image.save(output, dpi=(300, 300))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(read(args.summary.resolve()), args.output.resolve())
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
