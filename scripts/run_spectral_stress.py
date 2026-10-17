#!/usr/bin/env python3
"""Run a deterministic CPU stress test for residual-pair interpretation.

The experiment uses dense double-precision matrices with controlled singular
value profiles.  It tests quality and mechanism only; it is deliberately not
reported as a GPU timing result.  The construction makes it possible to ask
whether the same (ell, q, k) rule behaves differently when spectral decay
changes while dimensions and randomization are held fixed.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


FAMILIES = {
    "steep-decay": lambda n: (np.arange(n, dtype=np.float64) + 1.0) ** -2.0,
    "moderate-decay": lambda n: (np.arange(n, dtype=np.float64) + 1.0) ** -0.75,
    "slow-decay": lambda n: (np.arange(n, dtype=np.float64) + 1.0) ** -0.35,
    "flat-spectrum": lambda n: np.ones(n, dtype=np.float64),
    "two-level": lambda n: np.where(np.arange(n) < 16, 1.0, 0.18),
}


def orthogonal(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((rows, cols)), mode="reduced")
    return q


def state(matrix: np.ndarray, omega: np.ndarray, rank: int, power: int) -> tuple[float, float]:
    norm2 = float(np.sum(matrix * matrix))
    q, _ = np.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(power):
        z, _ = np.linalg.qr(matrix.T @ q, mode="reduced")
        q, _ = np.linalg.qr(matrix @ z, mode="reduced")
    projected = q.T @ matrix
    singular_values = np.linalg.svd(projected, compute_uv=False)
    projected_energy = float(np.sum(projected * projected))
    retained = float(np.sum(singular_values[:rank] ** 2))
    rho_l = float(np.sqrt(max(norm2 - projected_energy, 0.0) / norm2))
    rho_k = float(np.sqrt(max(norm2 - retained, 0.0) / norm2))
    return rho_l, rho_k


def run(rows: int, cols: int, seeds: list[int], widths: list[int], powers: list[int], ranks: list[int]):
    output: list[dict[str, object]] = []
    for family, spectrum_fn in FAMILIES.items():
        for seed in seeds:
            rng = np.random.default_rng(seed)
            left = orthogonal(rng, rows, cols)
            right = orthogonal(rng, cols, cols)
            singular_values = spectrum_fn(cols)
            matrix = (left * singular_values) @ right.T
            for width in widths:
                for power in powers:
                    omega_rng = np.random.default_rng(seed + 100000 * (width + 1))
                    omega = omega_rng.standard_normal((cols, width))
                    for rank in ranks:
                        if rank > width:
                            continue
                        rho_l, rho_k = state(matrix, omega, rank, power)
                        output.append({
                            "family": family,
                            "seed": seed,
                            "rows": rows,
                            "cols": cols,
                            "sketch_width": width,
                            "power_steps": power,
                            "target_rank": rank,
                            "rho_l": rho_l,
                            "rho_k": rho_k,
                            "precision": "float64",
                            "scope": "cpu-mechanism-stress",
                        })
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["family"], row["sketch_width"], row["power_steps"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (family, width, power), group in sorted(grouped.items()):
        rank16 = [float(row["rho_k"]) for row in group if row["target_rank"] == 16]
        rho_l = [float(row["rho_l"]) for row in group if row["target_rank"] == 16]
        summary.append({
            "family": family,
            "sketch_width": width,
            "power_steps": power,
            "target_rank": 16,
            "mean_rho_l": float(np.mean(rho_l)),
            "sd_rho_l": float(np.std(rho_l, ddof=1)) if len(rho_l) > 1 else 0.0,
            "mean_rho_k": float(np.mean(rank16)),
            "sd_rho_k": float(np.std(rank16, ddof=1)) if len(rank16) > 1 else 0.0,
            "seeds": len(rank16),
            "scope": "cpu-mechanism-stress",
        })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=192)
    parser.add_argument("--cols", type=int, default=160)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261010, 20261011, 20261012, 20261013, 20261014])
    args = parser.parse_args()
    if args.rows < args.cols or max([64, *[24, 40, 64]]) > args.cols:
        raise ValueError("rows must be >= cols and cols must accommodate width 64")
    rows = run(args.rows, args.cols, args.seeds, [24, 40, 64], [0, 1, 2], [8, 16, 32])
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "spectrum_stress_rows.csv", rows)
    write_csv(args.output / "spectrum_stress_summary.csv", summarize(rows))
    print(f"wrote {len(rows)} CPU mechanism-stress states")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
