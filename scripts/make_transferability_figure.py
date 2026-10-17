#!/usr/bin/env python3
"""Render the leave-one-matrix-out transfer replay."""
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


def draw(rows: list[dict[str, str]], output: Path) -> None:
    image = Image.new("RGB", (1900, 1000), "white")
    canvas = ImageDraw.Draw(image)
    canvas.text((950, 38), "Leave-one-matrix-out finite-grid transfer", anchor="mm", fill="black", font=font(30))
    left, top, right, bottom = 150, 130, 870, 725
    canvas.text(((left + right) / 2, top - 36), "held-out computed rho_k", anchor="mm", fill="black", font=font(21))
    canvas.line((left, bottom, right, bottom), fill=(25, 25, 25), width=2)
    canvas.line((left, top, left, bottom), fill=(25, 25, 25), width=2)
    max_rho = max(max(float(row["transfer_rho_k"]), float(row["oracle_rho_k"])) for row in rows) * 1.15
    for i in range(6):
        value = i * max_rho / 5
        y = bottom - value / max_rho * (bottom - top)
        canvas.line((left - 7, y, left, y), fill=(25, 25, 25), width=1)
        canvas.text((left - 14, y), f"{value:.2f}", anchor="rm", fill="black", font=font(15))
        if i:
            canvas.line((left, y, right, y), fill=(225, 230, 235), width=1)
    group = (right - left) / len(rows)
    for idx, row in enumerate(rows):
        center = left + (idx + 0.5) * group
        for offset, key, color in [(-19, "transfer_rho_k", (194, 59, 34)), (19, "oracle_rho_k", (43, 108, 176))]:
            value = float(row[key])
            y = bottom - value / max_rho * (bottom - top)
            canvas.rectangle((center + offset - 14, y, center + offset + 14, bottom), fill=color)
            canvas.text((center + offset, y - 9), f"{value:.2f}", anchor="ma", fill=color, font=font(13))
        canvas.text((center, bottom + 20), row["matrix"], anchor="ma", fill="black", font=font(13))

    left, top, right, bottom = 1030, 130, 1745, 725
    canvas.text(((left + right) / 2, top - 36), "transfer quality / held-out reference", anchor="mm", fill="black", font=font(21))
    canvas.line((left, bottom, right, bottom), fill=(25, 25, 25), width=2)
    canvas.line((left, top, left, bottom), fill=(25, 25, 25), width=2)
    max_ratio = max(float(row["transfer_quality_ratio"]) for row in rows) * 1.12
    for i in range(6):
        value = i * max_ratio / 5
        y = bottom - value / max_ratio * (bottom - top)
        canvas.line((left - 7, y, left, y), fill=(25, 25, 25), width=1)
        canvas.text((left - 14, y), f"{value:.2f}x", anchor="rm", fill="black", font=font(15))
        if i:
            canvas.line((left, y, right, y), fill=(225, 230, 235), width=1)
    group = (right - left) / len(rows)
    for idx, row in enumerate(rows):
        center = left + (idx + 0.5) * group
        value = float(row["transfer_quality_ratio"])
        y = bottom - value / max_ratio * (bottom - top)
        canvas.rectangle((center - 16, y, center + 16, bottom), fill=(110, 110, 110))
        canvas.text((center, y - 9), f"{value:.2f}x", anchor="ma", fill=(80, 80, 80), font=font(13))
        canvas.text((center, bottom + 20), row["matrix"], anchor="ma", fill="black", font=font(13))
    canvas.text((190, 835), "red: leave-one-out transfer", fill=(194, 59, 34), font=font(18))
    canvas.text((475, 835), "blue: held-out best-tested state", fill=(43, 108, 176), font=font(18))
    canvas.text((950, 930), "Computed state table; setting selection uses the other matrices", anchor="mm", fill=(70, 70, 70), font=font(16))
    image.save(output, dpi=(300, 300))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draw(read(args.input.resolve()), args.output.resolve())
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
