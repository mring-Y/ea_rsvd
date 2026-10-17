#!/usr/bin/env python3
"""Derive controller-action evidence from saved rank-sweep CSV files.

The script does not rerun CUDA kernels.  It compares the measured residual
change and the measured pipeline cost of two actions: more sketch work and a
larger target rank at a fixed sketch.  It also summarizes label stability over
the supplied threshold grid.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "sketch work": (43, 108, 176),
    "target rank": (194, 59, 34),
    "rank-limited": (43, 108, 176),
    "projection-limited": (194, 59, 34),
    "adequate": (82, 141, 78),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def font(size: int):
    candidates = []
    windir = Path(__import__("os").environ["WINDIR"]) if __import__("os").environ.get("WINDIR") else None
    if windir:
        candidates.append(windir / "Fonts" / "arial.ttf")
    candidates.append(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def build_action_rows(grid: list[dict[str, str]], summary: list[dict[str, str]], transitions: list[dict[str, str]]):
    matrices = list(dict.fromkeys(row["matrix"] for row in summary if row["rank"] == "16"))
    rows = []
    for matrix in matrices:
        rank16 = next(row for row in summary if row["matrix"] == matrix and row["rank"] == "16")
        baseline = [r for r in grid if r["matrix"] == matrix and r["rank"] == "16" and r["width"] == "24" and r["power"] == "0"]
        high = [r for r in grid if r["matrix"] == matrix and r["rank"] == "16" and r["width"] == "40" and r["power"] == "2"]
        if not baseline or not high:
            raise ValueError(f"missing baseline/high state for {matrix}")
        baseline_time = sum(f(r, "gpu_pipeline_ms") for r in baseline) / len(baseline)
        high_time = sum(f(r, "gpu_pipeline_ms") for r in high) / len(high)
        sketch_gain = f(rank16, "delta_rho_k_baseline_to_high_mean")
        sketch_extra = max(high_time - baseline_time, 0.0)
        transition = next(r for r in transitions if r["matrix"] == matrix and r["transition"] == "16->32")
        rank_gain = f(transition, "rho_k_reduction_mean")
        rank_extra = max(f(transition, "gpu_ms_higher_mean") - f(transition, "gpu_ms_lower_mean"), 0.0)
        rows.append({
            "matrix": matrix,
            "sketch_rho_k_reduction": f"{sketch_gain:.8f}",
            "sketch_delta_ms": f"{high_time - baseline_time:.4f}",
            "rank_rho_k_reduction": f"{rank_gain:.8f}",
            "rank_delta_ms": f"{f(transition, 'gpu_ms_higher_mean') - f(transition, 'gpu_ms_lower_mean'):.4f}",
            "rank16_terminal_rho_k": f(rank16, "best_rho_k_mean"),
        })
        rows[-1]["rank16_terminal_rho_k"] = f"{float(rows[-1]['rank16_terminal_rho_k']):.8f}"
    return rows


def build_threshold_rows(thresholds: list[dict[str, str]]):
    rows = []
    for matrix in list(dict.fromkeys(row["matrix"] for row in thresholds)):
        subset = [row for row in thresholds if row["matrix"] == matrix]
        labels = sorted(set(row["label"] for row in subset))
        same = all(row["all_seeds_same_label"].lower() == "true" for row in subset)
        rows.append({
            "matrix": matrix,
            "threshold_pairs": str(len(subset)),
            "labels_observed": ", ".join(labels),
            "all_seed_labels_stable": str(same),
        })
    return rows


def draw_action(rows: list[dict[str, str]], out: Path) -> None:
    image = Image.new("RGB", (1500, 980), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font = font(36), font(27), font(23)
    draw.text((750, 38), "Action evidence from matched rank-sweep states", anchor="mm", fill="black", font=title_font)
    panels = [(150, 105, 1370, 420, "(a) reduction in rho_k", "residual reduction"), (150, 520, 1370, 835, "(b) change in GPU pipeline time", "delta time (ms)")]
    for left, top, right, bottom, panel_title, ylabel in panels:
        draw.text(((left + right) / 2, top - 38), panel_title, anchor="mm", fill="black", font=label_font)
        draw.line((left, bottom, right, bottom), fill=(25, 25, 25), width=3)
        draw.line((left, top, left, bottom), fill=(25, 25, 25), width=3)
        if panel_title.startswith("(a)"):
            values = [float(row["sketch_rho_k_reduction"]) for row in rows] + [float(row["rank_rho_k_reduction"]) for row in rows]
            ymin, ymax = 0.0, max(values) * 1.22
        else:
            values = [float(row["sketch_delta_ms"]) for row in rows] + [float(row["rank_delta_ms"]) for row in rows]
            span = max(abs(v) for v in values) * 1.3
            ymin, ymax = -span, span
        ymax = max(ymax, 0.01) if ymin >= 0 else max(ymax, 1.0)
        def ycoord(value):
            return bottom - (value - ymin) / (ymax - ymin) * (bottom - top)
        for i in range(6):
            value = ymin + i * (ymax - ymin) / 5
            y = ycoord(value)
            draw.line((left - 8, y, left, y), fill=(25, 25, 25), width=2)
            draw.text((left - 18, y), f"{value:.3g}", anchor="rm", fill="black", font=tick_font)
            if i > 0:
                draw.line((left, y, right, y), fill=(225, 230, 235), width=1)
        if ymin < 0 < ymax:
            draw.line((left, ycoord(0), right, ycoord(0)), fill=(80, 80, 80), width=2)
        group = (right - left) / len(rows)
        for index, row in enumerate(rows):
            center = left + (index + 0.5) * group
            keys = [("sketch_rho_k_reduction", COLORS["sketch work"]), ("rank_rho_k_reduction", COLORS["target rank"])] if panel_title.startswith("(a)") else [("sketch_delta_ms", COLORS["sketch work"]), ("rank_delta_ms", COLORS["target rank"])]
            for shift, (key, color) in zip((-38, 38), keys):
                value = float(row[key])
                y0, y1 = ycoord(0), ycoord(value)
                draw.rectangle((center + shift - 18, min(y0, y1), center + shift + 18, max(y0, y1)), fill=color)
                label_y = y1 - 25 if value >= 0 else y1 + 5
                draw.text((center + shift, label_y), f"{value:.3g}", anchor="ma", fill=color, font=tick_font)
            draw.text((center, bottom + 18), row["matrix"], anchor="ma", fill="black", font=tick_font)
    draw.text((180, 900), "blue: increase sketch width/power", fill=COLORS["sketch work"], font=label_font)
    draw.text((720, 900), "red: increase target rank at fixed sketch", fill=COLORS["target rank"], font=label_font)
    draw.text((750, 950), "Negative values indicate paired timing variability.", anchor="mm", fill=(70, 70, 70), font=tick_font)
    image.save(out, dpi=(300, 300))


def draw_threshold(rows: list[dict[str, str]], out: Path) -> None:
    image = Image.new("RGB", (1500, 720), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font = font(30), font(20), font(17)
    draw.text((750, 38), "Threshold-grid stability of the rank diagnosis", anchor="mm", fill="black", font=title_font)
    left, top, col_w, row_h = 180, 120, 245, 82
    headers = ["Matrix", "Threshold pairs", "Labels observed", "Five-seed label stability"]
    widths = [300, 230, 330, 470]
    x = left
    for header, width in zip(headers, widths):
        draw.rectangle((x, top, x + width, top + row_h), fill=(31, 78, 121), outline="white")
        draw.text((x + width / 2, top + row_h / 2), header, anchor="mm", fill="white", font=label_font)
        x += width
    for ri, row in enumerate(rows):
        y = top + row_h * (ri + 1)
        x = left
        values = [row["matrix"], row["threshold_pairs"], row["labels_observed"], row["all_seed_labels_stable"]]
        for value, width in zip(values, widths):
            fill = (242, 244, 247) if ri % 2 else (255, 255, 255)
            draw.rectangle((x, y, x + width, y + row_h), fill=fill, outline=(183, 195, 208))
            display_value = value.replace("rank_limited", "Rank-limited").replace("projection_limited", "Projection-limited")
            display_value = {
                "True": "Stable across seeds",
                "False": "Varies across seeds",
            }.get(display_value, display_value)
            color = (49, 126, 83) if value == "True" else (0, 0, 0)
            draw.text((x + width / 2, y + row_h / 2), display_value, anchor="mm", fill=color, font=label_font)
            x += width
    draw.text((750, 650), "Each row summarizes nine (tau_l, tau_k) pairs: 0.15, 0.20, and 0.25.", anchor="mm", fill=(70, 70, 70), font=tick_font)
    image.save(out, dpi=(300, 300))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank_root = args.work_root / "review_experiments/round20_rank_adequacy_multiseed"
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    action_rows = build_action_rows(
        read_csv(rank_root / "rank_adequacy_grid.csv"),
        read_csv(rank_root / "rank_adequacy_summary.csv"),
        read_csv(rank_root / "rank_transition_summary.csv"),
    )
    threshold_rows = build_threshold_rows(read_csv(rank_root / "rank_diagnostic_threshold_sensitivity.csv"))
    for name, rows in [("action_tradeoff.csv", action_rows), ("threshold_stability.csv", threshold_rows)]:
        with (out / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    draw_action(action_rows, out / "fig7_action_tradeoff.png")
    draw_threshold(threshold_rows, out / "fig8_threshold_stability.png")
    print(f"generated {out / 'action_tradeoff.csv'} and {out / 'threshold_stability.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
