#!/usr/bin/env python3
"""Summarize quality and cumulative cost along the stored controller path.

This is deterministic post-processing of exact rank-sweep states.  It exposes
the calibration budget at each visited state and does not launch CUDA.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from replay_controller_path import classify, read_csv, value
from budget_cohort import summarize


def path_rows(rows: list[dict[str, str]], tau_l: float, tau_k: float) -> list[dict[str, object]]:
    states = {(r["matrix"], r["seed"], r["rank"], r["width"], r["power"]): r for r in rows}
    output: list[dict[str, object]] = []
    for matrix in sorted({r["matrix"] for r in rows}):
        for seed in sorted({r["seed"] for r in rows if r["matrix"] == matrix}):
            current = states.get((matrix, seed, "16", "24", "0"))
            if current is None:
                continue
            chosen = [current]
            for width, q_values in ((24, (1, 2)), (32, (2,)), (40, (2,))):
                for q in q_values:
                    candidate = states.get((matrix, seed, "16", str(width), str(q)))
                    if candidate is None:
                        continue
                    chosen.append(candidate)
                    current = candidate
                decision = classify(current, tau_l, tau_k)
                if decision == "stop":
                    break
                if decision == "target-rank":
                    candidate = states.get((matrix, seed, "32", str(width), str(current["power"])))
                    if candidate is not None:
                        chosen.append(candidate)
                        break
            cumulative = 0.0
            best = float("inf")
            for step, state in enumerate(chosen, start=1):
                rho = value(state, "exact_rank_residual")
                cumulative += value(state, "gpu_pipeline_ms")
                best = min(best, rho)
                output.append({
                    "matrix": matrix,
                    "seed": seed,
                    "budget_step": step,
                    "endpoint_rho_k": rho,
                    "best_seen_rho_k": best,
                    "cumulative_gpu_ms": cumulative,
                })
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw(summary: list[dict[str, object]], output: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    import os
    candidates = [Path(os.environ.get("WINDIR", "")) / "Fonts" / "arial.ttf", Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    def font(size: int):
        for candidate in candidates:
            if candidate.exists():
                return ImageFont.truetype(str(candidate), size)
        return ImageFont.load_default()
    image = Image.new("RGB", (1800, 900), "white")
    canvas = ImageDraw.Draw(image)
    canvas.text((900, 36), "Finite-grid calibration budget curve", anchor="mm", fill="black", font=font(30))
    panels = [(135, 125, 820, 690), (980, 125, 1665, 690)]
    series = [("mean endpoint rho_k", "mean_endpoint_rho_k", (194, 59, 34)),
              ("best seen rho_k", "mean_best_seen_rho_k", (43, 108, 176))]
    for index, box in enumerate(panels):
        left, top, right, bottom = box
        heading = "quality after each visited state" if index == 0 else "cumulative GPU state cost (ms)"
        canvas.text(((left + right) / 2, top - 34), heading, anchor="mm", fill="black", font=font(21))
        canvas.line((left, bottom, right, bottom), fill=(25, 25, 25), width=2)
        canvas.line((left, top, left, bottom), fill=(25, 25, 25), width=2)
        values = [float(r["mean_cumulative_gpu_ms"] if index else r["mean_endpoint_rho_k"]) for r in summary]
        if index == 0:
            values += [float(r["mean_best_seen_rho_k"]) for r in summary]
        ymax = max(values) * 1.18 if values else 1.0
        ymax = max(ymax, 1e-8)
        for tick in range(6):
            value = ymax * tick / 5
            y = bottom - value / ymax * (bottom - top)
            canvas.line((left - 7, y, left, y), fill=(25, 25, 25), width=1)
            canvas.text((left - 14, y), f"{value:.2g}", anchor="rm", fill="black", font=font(15))
            if tick:
                canvas.line((left, y, right, y), fill=(225, 230, 235), width=1)
        xstep = (right - left) / max(len(summary) - 1, 1)
        if index == 0:
            for label, key, color in series:
                points = []
                for pos, row in enumerate(summary):
                    x = left + pos * xstep
                    y = bottom - float(row[key]) / ymax * (bottom - top)
                    points.append((x, y))
                    canvas.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
                for a, b in zip(points, points[1:]):
                    canvas.line((*a, *b), fill=color, width=3)
                canvas.text((left + 25 + series.index((label, key, color)) * 230, bottom + 58), label, fill=color, font=font(16))
        else:
            points = []
            for pos, row in enumerate(summary):
                x = left + pos * xstep
                y = bottom - float(row["mean_cumulative_gpu_ms"]) / ymax * (bottom - top)
                points.append((x, y))
                canvas.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(82, 141, 78))
            for a, b in zip(points, points[1:]):
                canvas.line((*a, *b), fill=(82, 141, 78), width=3)
            canvas.text((left + 25, bottom + 58), "mean cumulative state cost", fill=(82, 141, 78), font=font(16))
        for pos, row in enumerate(summary):
            canvas.text((left + pos * xstep, bottom + 19), str(row["budget_step"]), anchor="ma", fill="black", font=font(14))
        canvas.text(((left + right) / 2, bottom + 40), "visited-state budget", anchor="ma", fill=(70, 70, 70), font=font(14))
    canvas.text((900, 845), "Fixed cohort; ended paths retain their measured result and cost", anchor="mm", fill=(70, 70, 70), font=font(16))
    image.save(output, dpi=(300, 300))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tau-l", type=float, default=0.20)
    parser.add_argument("--tau-k", type=float, default=0.20)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = path_rows(read_csv(args.input.resolve()), args.tau_l, args.tau_k)
    summary = summarize(rows)
    write_csv(output / "budget_curve_rows.csv", rows)
    write_csv(output / "budget_curve_summary.csv", summary)
    draw(summary, output / "fig20_budget_curve.png")
    print(f"wrote {len(rows)} budget rows and {len(summary)} budget summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
