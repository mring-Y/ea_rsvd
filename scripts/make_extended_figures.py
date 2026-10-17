#!/usr/bin/env python3
"""Render the two extended-evidence figures from summarized CSV files."""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "pve_only": (42, 86, 160),
    "residual_only": (214, 91, 73),
    "dual_monitor": (49, 126, 83),
    "amazon0312": (154, 94, 166),
    "webbase-1M": (210, 143, 44),
}


def rows(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def number(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def font(size):
    candidates = []
    windir = os.environ.get("WINDIR")
    if windir:
        candidates.extend([Path(windir) / "Fonts" / "arial.ttf", Path(windir) / "Fonts" / "times.ttf"])
    candidates.append(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def axes(draw, box, xlabel, ylabel, title, xlabels=None, y_min=0.0, y_max=1.0):
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=(25, 25, 25), width=2)
    draw.line((x0, y0, x0, y1), fill=(25, 25, 25), width=2)
    draw.text(((x0 + x1) / 2, 30), title, fill="black", anchor="mm", font=font(25))
    draw.text(((x0 + x1) / 2, y1 + 48), xlabel, fill="black", anchor="mm", font=font(20))
    y_label = Image.new("RGBA", (260, 38), (255, 255, 255, 0))
    ImageDraw.Draw(y_label).text((130, 19), ylabel, fill="black", anchor="mm", font=font(18))
    y_label = y_label.rotate(90, expand=True)
    im = draw._image
    im.paste(y_label, (max(0, x0 - y_label.width - 24), int((y0 + y1 - y_label.height) / 2)), y_label)
    for frac in (0, .25, .5, .75, 1):
        x = x0 + frac * (x1 - x0)
        draw.line((x, y1, x, y1 + 8), fill=(25, 25, 25), width=1)
        if xlabels:
            i = min(int(round(frac * (len(xlabels) - 1))), len(xlabels) - 1)
            draw.text((x, y1 + 20), xlabels[i], fill="black", anchor="ma", font=font(14))
        y = y1 - frac * (y1 - y0)
        draw.line((x0 - 8, y, x0, y), fill=(25, 25, 25), width=1)
        value = y_min + frac * (y_max - y_min)
        label = f"{value:.3g}"
        draw.text((x0 - 14, y), label, fill="black", anchor="rm", font=font(14))


def legend(draw, items, x, y):
    for i, (label, color) in enumerate(items):
        xx = x + i * 205
        draw.rectangle((xx, y - 7, xx + 14, y + 7), fill=color)
        draw.text((xx + 22, y), label, fill="black", anchor="lm", font=font(15))


def save_controller(rows_in, output):
    matrices = list(dict.fromkeys(r["matrix"] for r in rows_in))
    policies = ["pve_only", "residual_only", "dual_monitor"]
    im = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(im)
    boxes = [(115, 95, 700, 685), (805, 95, 1390, 685)]
    labels = [m.replace("lp_", "") for m in matrices]
    max_time = max(number(r, "mean_cumulative_gpu_ms") for r in rows_in)
    max_res = max(number(r, "mean_final_probe_residual") for r in rows_in)
    axes(draw, boxes[0], "Matrix", "Mean cumulative GPU time (ms)", "Action-policy ablation: search cost", labels, 0, max_time)
    axes(draw, boxes[1], "Matrix", "Mean final probe residual", "Action-policy ablation: terminal quality", labels, 0, max_res)
    for panel, key, maximum in ((0, "mean_cumulative_gpu_ms", max_time), (1, "mean_final_probe_residual", max_res)):
        x0, y0, x1, y1 = boxes[panel]
        step = (x1 - x0) / len(matrices)
        for i, matrix in enumerate(matrices):
            base = x0 + i * step
            for j, policy in enumerate(policies):
                match = next(r for r in rows_in if r["matrix"] == matrix and r["policy"] == policy)
                value = number(match, key)
                bar_w = step * .18
                x = base + step * .18 + j * (bar_w + step * .03)
                height = value / maximum * (y1 - y0) if maximum else 0
                draw.rectangle((x, y1 - height, x + bar_w, y1), fill=COLORS[policy])
    legend(draw, [(p, COLORS[p]) for p in policies], 180, 775)
    draw.text((750, 845), "Five seeds per matrix; lower is better", fill=(70, 70, 70), anchor="mm", font=font(17))
    im.save(output, dpi=(300, 300))


def save_scale(rows_in, output):
    im = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(im)
    boxes = [(115, 95, 700, 685), (805, 95, 1390, 685)]
    ranks = sorted({int(float(r["target_rank"])) for r in rows_in if r["target_rank"]})
    axes(draw, boxes[0], "Target rank k", "Rank-k residual", "Large sparse matrices: quality", [str(k) for k in ranks], .94, 1.0)
    axes(draw, boxes[1], "Target rank k", "GPU pipeline time (ms)", "Large sparse matrices: cost", [str(k) for k in ranks], 0, 5500)
    families = ["amazon0312", "webbase-1M"]
    xlim = (min(ranks), max(ranks))
    y_quality = (0, 1)
    max_time = max(number(r, "gpu_pipeline_ms") for r in rows_in)
    for family in families:
        # Keep the rank comparison at a common power depth. q=2 is available
        # for both families; warning rows are excluded from the plotted mean.
        family_rows = [r for r in rows_in if r["family"] == family and r["power_steps"] == "2" and not r["status"].startswith("WARN")]
        by_rank = {}
        for r in family_rows:
            by_rank.setdefault(int(float(r["target_rank"])), []).append(r)
        points_q, points_t = [], []
        for rank in sorted(by_rank):
            group = by_rank[rank]
            x = boxes[0][0] + (rank - xlim[0]) / (xlim[1] - xlim[0]) * (boxes[0][2] - boxes[0][0])
            x_t = boxes[1][0] + (rank - xlim[0]) / (xlim[1] - xlim[0]) * (boxes[1][2] - boxes[1][0])
            q = sum(number(r, "rank_k_residual") for r in group) / len(group)
            t = sum(number(r, "gpu_pipeline_ms") for r in group) / len(group)
            yq = boxes[0][3] - (q - .94) / .06 * (boxes[0][3] - boxes[0][1])
            yt = boxes[1][3] - t / 5500 * (boxes[1][3] - boxes[1][1])
            points_q.append((x, yq)); points_t.append((x_t, yt))
            draw.ellipse((x - 7, yq - 7, x + 7, yq + 7), fill=COLORS[family])
            draw.ellipse((x_t - 7, yt - 7, x_t + 7, yt + 7), fill=COLORS[family])
        if len(points_q) > 1:
            draw.line(points_q, fill=COLORS[family], width=3)
            draw.line(points_t, fill=COLORS[family], width=3)
    legend(draw, [(f, COLORS[f]) for f in families], 320, 775)
    draw.text((750, 845), "q=2; Amazon0312: one seed; Webbase-1M: two seeds where available; rows above the Q reference excluded", fill=(70, 70, 70), anchor="mm", font=font(16))
    im.save(output, dpi=(300, 300))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    assets = args.assets.resolve()
    save_controller(rows(assets / "controller_ablation_plot.csv"), assets / "fig9_controller_ablation.png")
    save_scale(rows(assets / "extended_scale_rows.csv"), assets / "fig10_extended_scale.png")
    print(f"wrote {assets / 'fig9_controller_ablation.png'}")
    print(f"wrote {assets / 'fig10_extended_scale.png'}")


if __name__ == "__main__":
    main()
