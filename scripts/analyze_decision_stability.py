#!/usr/bin/env python3
"""Measure residual-pair decision stability under controlled perturbations.

The input is an exact state table produced by the CUDA rank sweep.  The
analysis perturbs the two stored residuals with deterministic relative Gaussian
noise and replays the same three-way controller label.  It is a sensitivity
analysis of the recorded diagnostic, not a replacement for repeated GPU runs.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def classify(rho_l: np.ndarray, rho_k: np.ndarray, tau_l: float, tau_k: float) -> np.ndarray:
    labels = np.full(rho_l.shape, "projection", dtype=object)
    labels[(rho_l <= tau_l) & (rho_k > tau_k)] = "target-rank"
    labels[rho_k <= tau_k] = "stop"
    return labels


def font(size: int):
    import os
    candidates = []
    windir = os.environ.get("WINDIR")
    if windir:
        candidates.append(Path(windir) / "Fonts" / "arial.ttf")
    candidates.append(Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(rows: list[dict[str, str]], levels: list[float], replicates: int,
            tau_l: float, tau_k: float, seed: int) -> list[dict[str, object]]:
    valid = [row for row in rows if row.get("exact_projection_residual") and row.get("exact_rank_residual")]
    rho_l = np.asarray([float(row["exact_projection_residual"]) for row in valid], dtype=np.float64)
    rho_k = np.asarray([float(row["exact_rank_residual"]) for row in valid], dtype=np.float64)
    finite = np.isfinite(rho_l) & np.isfinite(rho_k)
    rho_l, rho_k = rho_l[finite], rho_k[finite]
    base = classify(rho_l, rho_k, tau_l, tau_k)
    rng = np.random.default_rng(seed)
    output: list[dict[str, object]] = []
    classes = ("projection", "target-rank", "stop")
    for level in levels:
        flip_rates: list[float] = []
        retention = {name: [] for name in classes}
        for _ in range(replicates):
            # Relative perturbations preserve the scale of very small residuals
            # while the floor prevents a zero residual from staying exactly zero.
            scale_l = np.maximum(np.abs(rho_l), 1.0e-8)
            scale_k = np.maximum(np.abs(rho_k), 1.0e-8)
            pert_l = np.clip(rho_l + rng.normal(0.0, level, rho_l.size) * scale_l, 0.0, 1.0)
            pert_k = np.clip(rho_k + rng.normal(0.0, level, rho_k.size) * scale_k, 0.0, 1.0)
            observed = classify(pert_l, pert_k, tau_l, tau_k)
            flip_rates.append(float(np.mean(observed != base)))
            for name in classes:
                mask = base == name
                retention[name].append(float(np.mean(observed[mask] == name)) if np.any(mask) else math.nan)
        output.append({
            "relative_noise_sigma": level,
            "replicates": replicates,
            "states": int(rho_l.size),
            "mean_flip_rate": float(np.mean(flip_rates)),
            "p95_flip_rate": float(np.percentile(flip_rates, 95)),
            "projection_retention": float(np.mean(retention["projection"])),
            "target_rank_retention": float(np.mean(retention["target-rank"])),
            "stop_retention": float(np.mean(retention["stop"])),
            "tau_l": tau_l,
            "tau_k": tau_k,
            "seed": seed,
            "scope": "residual-pair-decision-sensitivity",
        })
    return output


def draw(summary: list[dict[str, object]], output: Path) -> None:
    image = Image.new("RGB", (1800, 920), "white")
    canvas = ImageDraw.Draw(image)
    canvas.text((900, 36), "Residual-pair decision stability", anchor="mm", fill="black", font=font(30))
    panels = [(135, 125, 820, 690), (980, 125, 1665, 690)]
    labels = ["mean label flips", "class retention"]
    for box, heading in zip(panels, labels):
        left, top, right, bottom = box
        canvas.text(((left + right) / 2, top - 34), heading, anchor="mm", fill="black", font=font(21))
        canvas.line((left, bottom, right, bottom), fill=(25, 25, 25), width=2)
        canvas.line((left, top, left, bottom), fill=(25, 25, 25), width=2)
        ymax = max(max(float(row["mean_flip_rate"]) for row in summary) * 1.35, 1.0e-4) if heading == labels[0] else 1.0
        for i in range(6):
            value = i * ymax / 5
            y = bottom - value / ymax * (bottom - top)
            canvas.line((left - 7, y, left, y), fill=(25, 25, 25), width=1)
            canvas.text((left - 14, y), f"{value:.2g}", anchor="rm", fill="black", font=font(15))
            if i:
                canvas.line((left, y, right, y), fill=(225, 230, 235), width=1)
        group = (right - left) / max(len(summary), 1)
        if heading == labels[0]:
            points = [(float(row["relative_noise_sigma"]), float(row["mean_flip_rate"]), (194, 59, 34)) for row in summary]
            values = [points]
            names = ["mean flip rate"]
        else:
            values = [[(float(row["relative_noise_sigma"]), float(row[key]), color) for row in summary]
                      for key, color in (("projection_retention", (43, 108, 176)),
                                         ("target_rank_retention", (194, 59, 34)),
                                         ("stop_retention", (82, 141, 78)))]
            names = ["projection", "target-rank", "stop"]
        for series_index, series in enumerate(values):
            coords = []
            for index, (_, value, color) in enumerate(series):
                center = left + (index + 0.5) * group
                if heading == labels[0]:
                    x = center
                else:
                    x = center + (series_index - 1) * 20
                y = bottom - value / ymax * (bottom - top)
                coords.append((x, y))
                if heading == labels[0]:
                    canvas.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)
                else:
                    canvas.rectangle((x - 12, y, x + 12, bottom), fill=color)
                if index:
                    canvas.line((coords[-2][0], coords[-2][1], x, y), fill=color, width=3)
            legend_x = left + 20 + series_index * (right - left) / max(len(names), 1)
            canvas.text((legend_x, bottom + 62), names[series_index], fill=series[0][2], font=font(16))
        for index, row in enumerate(summary):
            center = left + (index + 0.5) * group
            canvas.text((center, bottom + 19), f"{float(row['relative_noise_sigma']):.0e}", anchor="ma", fill="black", font=font(14))
        canvas.text(((left + right) / 2, bottom + 42), "relative residual noise sigma", anchor="ma", fill=(70, 70, 70), font=font(14))
    canvas.text((900, 850), "Sensitivity analysis from the full-energy state table", anchor="mm", fill=(70, 70, 70), font=font(16))
    image.save(output, dpi=(300, 300))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 1e-4, 1e-3, 1e-2])
    parser.add_argument("--tau-l", type=float, default=0.20)
    parser.add_argument("--tau-k", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.replicates < 1 or any(level < 0 for level in args.noise_levels):
        raise ValueError("replicates must be positive and noise levels non-negative")
    summary = analyze(read_csv(args.input.resolve()), args.noise_levels, args.replicates,
                      args.tau_l, args.tau_k, args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "decision_stability_summary.csv", summary)
    draw(summary, output / "fig19_decision_stability.png")
    print(f"wrote {len(summary)} perturbation levels and figure to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
