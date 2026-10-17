#!/usr/bin/env python3
"""Generate manuscript figures and controlled-spectrum evidence from CSV assets.

The script deliberately uses only NumPy and Pillow so that it runs on a clean
AutoDL image without SciPy or Matplotlib.  It never invents measurements: the
empirical panels read the state-level CSV files supplied to --work-root.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = [(42, 86, 160), (214, 91, 73), (49, 126, 83), (154, 94, 166), (210, 143, 44), (72, 150, 174)]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def num(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    windir = os.environ.get("WINDIR")
    candidates = []
    if windir:
        candidates.extend([str(Path(windir) / "Fonts" / "times.ttf"), str(Path(windir) / "Fonts" / "arial.ttf")])
    candidates.append("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def canvas(w: int = 1200, h: int = 760) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (w, h), "white")
    return im, ImageDraw.Draw(im)


def axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], xlabel: str, ylabel: str, title: str,
         xlim: tuple[float, float], ylim: tuple[float, float], log_y: bool = False,
         x_ticks: Iterable[float] | None = None):
    x0, y0, x1, y1 = box
    draw.line((x0, y1, x1, y1), fill=(30, 30, 30), width=2)
    draw.line((x0, y0, x0, y1), fill=(30, 30, 30), width=2)
    draw.text(((x0 + x1) / 2, y1 + 34), xlabel, fill="black", anchor="mm", font=font(23))
    # Keep the rotated-free y label clear of the tick labels at print scale.
    draw.text((37, (y0 + y1) / 2), ylabel, fill="black", anchor="mm", font=font(19))
    draw.text(((x0 + x1) / 2, 28), title, fill="black", anchor="mm", font=font(27))
    for value in x_ticks if x_ticks is not None else (xlim[0] + frac * (xlim[1] - xlim[0]) for frac in (0, .25, .5, .75, 1)):
        x = tx(value, box, xlim)
        draw.line((x, y1, x, y1 + 8), fill=(30, 30, 30), width=1)
        draw.text((x, y1 + 18), f"{value:g}", fill="black", anchor="mm", font=font(16))
    for frac in (0, .25, .5, .75, 1):
        y = y1 - frac * (y1 - y0)
        draw.line((x0 - 8, y, x0, y), fill=(30, 30, 30), width=1)
        if log_y:
            val = math.exp(math.log(max(ylim[0], 1e-12)) + frac * (math.log(ylim[1]) - math.log(max(ylim[0], 1e-12))))
            label = f"{val:.1e}"
        else:
            val = ylim[0] + frac * (ylim[1] - ylim[0])
            label = f"{val:g}"
        draw.text((x0 - 15, y), label, fill="black", anchor="rm", font=font(16))


def tx(v: float, box: tuple[int, int, int, int], lim: tuple[float, float]) -> float:
    return box[0] + (v - lim[0]) / (lim[1] - lim[0]) * (box[2] - box[0])


def ty(v: float, box: tuple[int, int, int, int], lim: tuple[float, float], log_y: bool = False) -> float:
    if log_y:
        v = max(v, 1e-8)
        a, b = math.log(max(lim[0], 1e-8)), math.log(lim[1])
        return box[3] - (math.log(v) - a) / (b - a) * (box[3] - box[1])
    return box[3] - (v - lim[0]) / (lim[1] - lim[0]) * (box[3] - box[1])


def legend(draw, items: Iterable[tuple[str, tuple[int, int, int]]], x: int, y: int, cols: int = 1):
    items = list(items)
    for i, (label, c) in enumerate(items):
        xx = x + (i % cols) * 180
        yy = y + (i // cols) * 26
        draw.ellipse((xx, yy - 6, xx + 12, yy + 6), fill=c)
        draw.text((xx + 20, yy), label, fill="black", anchor="lm", font=font(15))


def save_fig1(rows: list[dict[str, str]], out: Path):
    im, d = canvas()
    box = (125, 95, 1080, 635)
    axes(d, box, "rho_l", "rho_k", "Residual-pair diagnosis across 925 states", (0, 1.02), (0, 1.02))
    mats = list(dict.fromkeys(r["matrix"] for r in rows))
    for i, m in enumerate(mats):
        c = COLORS[i % len(COLORS)]
        for r in rows:
            if r["matrix"] == m:
                d.ellipse((tx(num(r, "exact_projection_residual"), box, (0, 1.02)) - 3,
                           ty(num(r, "exact_rank_residual"), box, (0, 1.02)) - 3,
                           tx(num(r, "exact_projection_residual"), box, (0, 1.02)) + 3,
                           ty(num(r, "exact_rank_residual"), box, (0, 1.02)) + 3), fill=c)
    d.line((tx(0, box, (0, 1.02)), ty(0, box, (0, 1.02)), tx(1.02, box, (0, 1.02)), ty(1.02, box, (0, 1.02))), fill=(80, 80, 80), width=2)
    d.line((tx(.2, box, (0, 1.02)), box[1], tx(.2, box, (0, 1.02)), box[3]), fill=(80, 80, 80), width=1)
    d.line((box[0], ty(.2, box, (0, 1.02)), box[2], ty(.2, box, (0, 1.02))), fill=(80, 80, 80), width=1)
    legend(d, [(m, COLORS[i % len(COLORS)]) for i, m in enumerate(mats)], 145, 680, cols=3)
    im.save(out, dpi=(300, 300))


def save_fig2(summary: list[dict[str, str]], out: Path):
    im, d = canvas(1250, 800)
    box = (125, 95, 1040, 635)
    axes(d, box, "Requested rank k", "rho_k", "Rank adequacy across matrix families", (6, 66), (0, 1.02), x_ticks=(8, 16, 32, 64))
    mats = list(dict.fromkeys(r["matrix"] for r in summary))
    for i, m in enumerate(mats):
        c = COLORS[i % len(COLORS)]; pts = []
        for r in sorted((x for x in summary if x["matrix"] == m), key=lambda z: num(z, "rank")):
            x = tx(num(r, "rank"), box, (6, 66)); y = ty(num(r, "best_rho_k_mean"), box, (0, 1.02)); pts.append((x, y))
            d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=c)
            ci = num(r, "best_rho_k_ci95_halfwidth")
            d.line((x, ty(num(r, "best_rho_k_mean") - ci, box, (0, 1.02)), x,
                    ty(num(r, "best_rho_k_mean") + ci, box, (0, 1.02))), fill=c, width=2)
        if len(pts) > 1: d.line(pts, fill=c, width=3)
    d.line((box[0], ty(.2, box, (0, 1.02)), box[2], ty(.2, box, (0, 1.02))), fill=(50, 50, 50), width=2)
    legend(d, [(m, COLORS[i % len(COLORS)]) for i, m in enumerate(mats)], 145, 705, cols=3)
    im.save(out, dpi=(300, 300))


def save_fig3(rows: list[dict[str, str]], out: Path):
    im, d = canvas(1250, 760); box = (125, 95, 1040, 635)
    axes(d, box, "GPU pipeline time (ms)", "rho_k", "Fixed-grid quality/cost frontier (450 states)", (0, max(num(r, "gpu_pipeline_ms") for r in rows) * 1.03), (1e-4, 1), True)
    mats = list(dict.fromkeys(r["matrix"] for r in rows))
    for i, m in enumerate(mats):
        c = COLORS[i % len(COLORS)]
        mr = [r for r in rows if r["matrix"] == m]
        for r in mr:
            x = tx(num(r, "gpu_pipeline_ms"), box, (0, max(num(z, "gpu_pipeline_ms") for z in rows) * 1.03))
            y = ty(num(r, "exact_rank_residual"), box, (1e-4, 1), True)
            d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=c)
        best = min(mr, key=lambda z: num(z, "exact_rank_residual")); x = tx(num(best, "gpu_pipeline_ms"), box, (0, max(num(z, "gpu_pipeline_ms") for z in rows) * 1.03)); y = ty(num(best, "exact_rank_residual"), box, (1e-4, 1), True)
        d.regular_polygon((x, y, 9), 5, rotation=0, fill=(245, 196, 57), outline="black")
    legend(d, [(m, COLORS[i % len(COLORS)]) for i, m in enumerate(mats)], 145, 680, cols=5)
    im.save(out, dpi=(300, 300))


def save_fig4(a800: list[dict[str, str]], gpu: list[dict[str, str]], out: Path):
    def matrix_name(row: dict[str, str]) -> str:
        path = Path(row.get("input", ""))
        return path.parts[-2] if len(path.parts) > 1 else path.stem

    a800_by_name = {matrix_name(row): row for row in a800}
    gpu_by_name = {matrix_name(row): row for row in gpu}
    common = [name for name in a800_by_name if name in gpu_by_name][:10]
    names = common
    im, d = canvas(1400, 900)
    stages = [("h2d_ms", "H2D"), ("spmm_forward_ms", "A*Omega"), ("spmm_transpose_ms", "A^TQ"), ("qr_ms", "QR"), ("svd_ms", "SVD")]
    maxv = max(sum(num(data[name], key) for key, _ in stages)
               for data in (gpu_by_name, a800_by_name) for name in common)
    for pi, (data, title, top) in enumerate(((gpu, "RTX 4090D", 85), (a800, "A800", 470))):
        d.text((700, top - 35), title, fill="black", anchor="mm", font=font(28))
        x0, y0, x1, y1 = 175, top, 1240, top + 275
        d.text((x0, y0 - 12), "Stage time (ms)", fill="black", anchor="ls", font=font(15))
        for fraction in (0, .5, 1):
            y = y1 - fraction * (y1 - y0)
            d.line((x0, y, x1, y), fill=(218, 224, 230), width=1)
            d.text((x0 - 12, y), f"{fraction * maxv:.0f}", fill="black", anchor="rm", font=font(14))
        d.line((x0, y1, x1, y1), fill="black", width=2); d.line((x0, y0, x0, y1), fill="black", width=2)
        by_name = {matrix_name(row): row for row in data}
        for j, name in enumerate(common):
            x = x0 + (j + .5) * (x1 - x0) / len(common); bottom = y1
            row = by_name[name]
            for si, (k, label) in enumerate(stages):
                v = num(row, k); h = v / maxv * (y1 - y0)
                d.rectangle((x - 24, bottom - h, x + 24, bottom), fill=COLORS[si]); bottom -= h
            d.text((x, y1 + 17), names[j], fill="black", anchor="ma", font=font(14))
    legend(d, [(label, COLORS[i]) for i, (_, label) in enumerate(stages)], 140, 835, cols=5)
    im.save(out, dpi=(300, 300))


def save_fig5(rows: list[dict[str, str]], out: Path):
    im, d = canvas(1250, 760); box = (145, 95, 1090, 610)
    d.line((box[0], box[3], box[2], box[3]), fill=(30, 30, 30), width=2)
    d.line((box[0], box[1], box[0], box[3]), fill=(30, 30, 30), width=2)
    d.text(((box[0] + box[2]) / 2, 28), "Fixed-k=16 q/ell decisions across precision policies", fill="black", anchor="mm", font=font(27))
    y_label = Image.new("RGBA", (560, 30), (255, 255, 255, 0))
    ImageDraw.Draw(y_label).text((280, 15), "Maximum absolute difference / observed Q error", fill="black", anchor="mm", font=font(18))
    y_label = y_label.rotate(90, expand=True)
    im.paste(y_label, (28, int((box[1] + box[3] - y_label.height) / 2)), y_label)
    for frac in (0, .25, .5, .75, 1):
        y = box[3] - frac * (box[3] - box[1])
        d.line((box[0] - 8, y, box[0], y), fill=(30, 30, 30), width=1)
        value = math.exp(math.log(1e-7) + frac * (math.log(1e-2) - math.log(1e-7)))
        d.text((box[0] - 15, y), f"{value:.1e}", fill="black", anchor="rm", font=font(16))
    mats = list(dict.fromkeys(r["matrix"] for r in rows)); modes = ["rho_l", "rho_k", "q"]
    for i, m in enumerate(mats):
        rr = [r for r in rows if r["matrix"] == m]
        vals = [max(num(r, "max_abs_rho_l_diff") for r in rr), max(num(r, "max_abs_rho_k_diff") for r in rr), max(num(r, "max_q_orth") for r in rr)]
        for j, v in enumerate(vals):
            x = box[0] + (i + (j - 1) * .22 + .5) / len(mats) * (box[2] - box[0]); y = ty(v, box, (1e-7, 1e-2), True)
            y = min(max(y, box[1]), box[3])
            d.rectangle((x - 12, y, x + 12, box[3]), fill=COLORS[j])
        d.text((box[0] + (i + .5) / len(mats) * (box[2] - box[0]), box[3] + 20), m.replace(".mtx", ""), fill="black", anchor="ma", font=font(15))
    legend(d, [("rho_l", COLORS[0]), ("rho_k", COLORS[1]), ("Q orthogonality", COLORS[2])], 160, 665, cols=3)
    im.save(out, dpi=(300, 300))


def controlled_spectra() -> list[dict[str, str]]:
    rng = np.random.default_rng(20261010); m, n = 160, 120
    out: list[dict[str, str]] = []
    for family in ("rank-limited spectrum", "projection-limited spectrum"):
        singular = np.r_[np.linspace(1, .8, 24), np.zeros(n - 24)] if family.startswith("rank") else 1 / np.sqrt(np.arange(1, n + 1))
        u, _ = np.linalg.qr(rng.standard_normal((m, m))); v, _ = np.linalg.qr(rng.standard_normal((n, n)))
        a = u[:, :n] @ np.diag(singular) @ v.T; energy = float(np.sum(a * a))
        for seed in range(5):
            for l in ([40, 64] if family.startswith("rank") else [8, 16, 32]):
                for q in range(3):
                    rng2 = np.random.default_rng(20261010 + seed)
                    qmat = np.linalg.qr(a @ rng2.standard_normal((n, l)))[0]
                    for _ in range(q): qmat = np.linalg.qr(a @ np.linalg.qr(a.T @ qmat)[0])[0]
                    b = qmat.T @ a; sb = np.linalg.svd(b, compute_uv=False)
                    rho_l = math.sqrt(max(energy - float(np.sum(b * b)), 0) / energy)
                    for k in (8, 16, 32):
                        if k > l:
                            continue
                        rho_k = math.sqrt(max(energy - float(np.sum(sb[:k] ** 2)), 0) / energy)
                        out.append({"family": family, "seed": str(seed), "sketch_width": str(l), "power_steps": str(q), "target_rank": str(k), "rho_l": f"{rho_l:.9g}", "rho_k": f"{rho_k:.9g}"})
    return out


def _resolve_csv(explicit: Path | None, work_root: Path, filename: str, hints: tuple[str, ...]) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(f"CSV not found: {explicit}")
        return explicit
    candidates = []
    for path in work_root.rglob(filename):
        text = str(path).lower()
        if any(hint in text for hint in hints):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(f"Could not locate {filename}; pass the explicit CSV option")
    return sorted(candidates, key=lambda path: str(path), reverse=True)[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--a800-csv", type=Path, help="A800 same-CUDA reference CSV; defaults to the newest path containing 'a800'")
    ap.add_argument("--gpu-csv", type=Path, help="RTX 4090D same-CUDA reference CSV; defaults to the newest non-A800 AutoDL result")
    ap.add_argument("--precision-csv", type=Path, help="FP32/FP64/Kahan comparison CSV")
    args = ap.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=True); rank = args.work_root / "review_experiments/round20_rank_adequacy_multiseed"; fixed = args.work_root / "review_experiments/round16_exact_fixed_grid"
    rg = read_csv(rank / "rank_adequacy_grid.csv"); rs = read_csv(rank / "rank_adequacy_summary.csv"); fg = read_csv(fixed / "fixed_grid.csv")
    save_fig1(rg, out / "fig1_residual_diagnostic.png"); save_fig2(rs, out / "fig2_rank_curves.png"); save_fig3(fg, out / "fig3_fixed_grid_frontier.png")
    a800_path = _resolve_csv(args.a800_csv, args.work_root, "same_cuda_sparse_reference.csv", ("a800",))
    gpu_path = _resolve_csv(args.gpu_csv, args.work_root, "same_cuda_sparse_reference.csv", ("autodl_results",))
    a800 = read_csv(a800_path); gpu = read_csv(gpu_path); save_fig4(a800, gpu, out / "fig4_cross_platform_stages.png")
    precision_path = args.precision_csv or (args.work_root / "autodl_cloud/local_precision_replay_20261003_fixed/precision_comparison.csv")
    precision = read_csv(precision_path); save_fig5(precision, out / "fig5_precision_replay.png")
    controlled = controlled_spectra();
    with (out / "controlled_spectrum_diagnosis.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(controlled[0])); w.writeheader(); w.writerows(controlled)
    save_controlled(controlled, out / "fig6_controlled_spectra.png")
    summary = [{"metric": "rank_sweep_states", "value": str(len(rg))}, {"metric": "fixed_grid_states", "value": str(len(fg))}, {"metric": "controlled_spectrum_rows", "value": str(len(controlled))}]
    with (out / "paper_asset_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value"]); w.writeheader(); w.writerows(summary)
    print(f"generated {out}"); return 0


def save_controlled(rows: list[dict[str, str]], out: Path):
    im, d = canvas(); box = (125, 95, 1080, 635)
    axes(d, box, "rho_l", "rho_k", "Controlled spectra: separating rank and projection effects", (0, 1.02), (0, 1.02))
    families = list(dict.fromkeys(r["family"] for r in rows))
    for i, family in enumerate(families):
        c = COLORS[i]
        for r in rows:
            if r["family"] != family:
                continue
            x = tx(float(r["rho_l"]), box, (0, 1.02)); y = ty(float(r["rho_k"]), box, (0, 1.02))
            d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=c)
    d.line((tx(0, box, (0, 1.02)), ty(0, box, (0, 1.02)), tx(1.02, box, (0, 1.02)), ty(1.02, box, (0, 1.02))), fill=(80, 80, 80), width=2)
    d.line((tx(.2, box, (0, 1.02)), box[1], tx(.2, box, (0, 1.02)), box[3]), fill=(80, 80, 80), width=1)
    d.line((box[0], ty(.2, box, (0, 1.02)), box[2], ty(.2, box, (0, 1.02))), fill=(80, 80, 80), width=1)
    legend(d, [(family, COLORS[i]) for i, family in enumerate(families)], 145, 680, cols=2)
    im.save(out, dpi=(300, 300))


if __name__ == "__main__":
    raise SystemExit(main())
