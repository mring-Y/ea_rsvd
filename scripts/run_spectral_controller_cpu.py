#!/usr/bin/env python3
"""Run a CPU controlled-spectrum replay of the finite-grid controller.

This experiment uses the same residual definitions as the GPU path but keeps
the matrix construction and timing on the CPU.  It is intended to test the
decision mechanism under controlled singular-value profiles, not to claim a
GPU performance result.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
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


def make_matrix(rows: int, cols: int, family: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = orthogonal(rng, rows, cols)
    right = orthogonal(rng, cols, cols)
    return (left * FAMILIES[family](cols)) @ right.T


def state(matrix: np.ndarray, omega: np.ndarray, rank: int, power: int, repeat: int = 1) -> tuple[float, float, np.ndarray, float]:
    energy = float(np.sum(matrix * matrix, dtype=np.float64))
    elapsed = 0.0
    rho_l = rho_k = math.nan
    for _ in range(repeat):
        start = time.perf_counter()
        q, _ = np.linalg.qr(matrix @ omega, mode="reduced")
        for _ in range(power):
            z, _ = np.linalg.qr(matrix.T @ q, mode="reduced")
            q, _ = np.linalg.qr(matrix @ z, mode="reduced")
        projected = q.T @ matrix
        singular_values = np.linalg.svd(projected, compute_uv=False)
        projected_energy = float(np.sum(projected * projected, dtype=np.float64))
        retained = float(np.sum(singular_values[:rank] ** 2, dtype=np.float64))
        rho_l = math.sqrt(max(energy - projected_energy, 0.0) / energy)
        rho_k = math.sqrt(max(energy - retained, 0.0) / energy)
        elapsed += (time.perf_counter() - start) * 1000.0
    return rho_l, rho_k, singular_values, elapsed / repeat


def run(rows: int, cols: int, seeds: list[int], widths: list[int], powers: list[int], ranks: list[int]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for family in FAMILIES:
        for seed in seeds:
            matrix = make_matrix(rows, cols, family, seed)
            for width in widths:
                for power in powers:
                    # Adjacent power states share their starting test matrix.
                    omega_rng = np.random.default_rng(seed + 100000 * (width + 1))
                    omega = omega_rng.standard_normal((cols, width))
                    for rank in ranks:
                        if rank > width:
                            continue
                        rho_l, rho_k, singular_values, elapsed = state(matrix, omega, rank, power)
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
                            "sigma_k": float(singular_values[rank - 1]),
                            "singular_spectrum": ";".join(format(float(x), ".17g") for x in singular_values),
                            "cpu_state_ms": elapsed,
                            "precision": "float64",
                            "scope": "cpu-controlled-spectrum-controller",
                        })
    return output


def controller_path(rows: list[dict[str, object]], tau_l: float, tau_k: float, tau_sigma: float) -> tuple[list[dict[str, object]], str]:
    states = {(int(r["sketch_width"]), int(r["power_steps"]), int(r["target_rank"])): r for r in rows}
    width_order = sorted({int(r["sketch_width"]) for r in rows})
    rank = min(int(r["target_rank"]) for r in rows)
    width = width_order[0]
    power = 0
    path: list[dict[str, object]] = []
    terminal = "budget-limited"
    previous_sigma = None
    while True:
        current = states[(width, power, rank)]
        path.append(current)
        rho_l, rho_k = float(current["rho_l"]), float(current["rho_k"])
        if rho_k <= tau_k:
            terminal = "stop"
            break
        sigma = np.asarray([float(x) for x in str(current["singular_spectrum"]).split(";")])[:rank]
        sigma_change = (math.inf if previous_sigma is None else
                        float(np.max(np.abs(sigma - previous_sigma))) / max(float(sigma[-1]), 1e-7))
        if power < max(int(r["power_steps"]) for r in rows) and (power == 0 or sigma_change > tau_sigma) and rho_l > tau_l:
            previous_sigma = sigma
            power += 1
            continue
        if rho_l > tau_l and width != width_order[-1]:
            width = width_order[width_order.index(width) + 1]
            power = 0
            previous_sigma = None
            continue
        terminal = "budget-limited"
        break
    return path, terminal


def summarize(rows: list[dict[str, object]], tau_l: float, tau_k: float, tau_sigma: float) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    families = sorted({str(r["family"]) for r in rows})
    for family in families:
        family_rows = [r for r in rows if r["family"] == family]
        by_seed = []
        for seed in sorted({int(r["seed"]) for r in family_rows}):
            seed_rows = [r for r in family_rows if int(r["seed"]) == seed]
            path, terminal = controller_path(seed_rows, tau_l, tau_k, tau_sigma)
            # The controller keeps k fixed.  Compare it with the finite-grid
            # oracle at that same target rank, rather than with a larger-k
            # state whose quality includes a different approximation target.
            controller_rank = min(int(r["target_rank"]) for r in seed_rows)
            rank_rows = [r for r in seed_rows if int(r["target_rank"]) == controller_rank]
            oracle = min(rank_rows, key=lambda r: (float(r["rho_k"]), float(r["cpu_state_ms"])))
            path_rho = float(path[-1]["rho_k"])
            path_cost = sum(float(r["cpu_state_ms"]) for r in path)
            oracle_cost = float(oracle["cpu_state_ms"])
            best_cost = min(float(r["cpu_state_ms"]) for r in rank_rows)
            by_seed.append({
                "path_rho": path_rho,
                "path_cost": path_cost,
                "oracle_rho": float(oracle["rho_k"]),
                "oracle_cost": oracle_cost,
                "best_cost": best_cost,
                "states": len(path),
                "terminal": terminal,
                "agreement": int(abs(path_rho - float(oracle["rho_k"])) <= 1e-12),
            })
        def avg(key: str) -> float:
            return float(np.mean([float(r[key]) for r in by_seed]))
        output.append({
            "family": family,
            "seeds": len(by_seed),
            "mean_controller_rho_k": avg("path_rho"),
            "mean_oracle_rho_k": avg("oracle_rho"),
            "mean_controller_cpu_ms": avg("path_cost"),
            "mean_oracle_cpu_ms": avg("oracle_cost"),
            "mean_fastest_state_cpu_ms": avg("best_cost"),
            "mean_states": avg("states"),
            "terminal_stop_fraction": float(np.mean([r["terminal"] == "stop" for r in by_seed])),
            "terminal_budget_fraction": float(np.mean([r["terminal"] == "budget-limited" for r in by_seed])),
            "oracle_endpoint_agreement": avg("agreement"),
            "tau_l": tau_l,
            "tau_k": tau_k,
            "tau_sigma": tau_sigma,
            "scope": "cpu-controlled-spectrum-controller",
        })
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=192)
    parser.add_argument("--cols", type=int, default=160)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261010, 20261011, 20261012, 20261013, 20261014])
    parser.add_argument("--tau-l", type=float, default=0.20)
    parser.add_argument("--tau-k", type=float, default=0.20)
    parser.add_argument("--tau-sigma", type=float, default=0.01)
    args = parser.parse_args()
    if args.rows < args.cols or args.cols < 64:
        raise ValueError("rows must be >= cols and cols must accommodate width 64")
    rows = run(args.rows, args.cols, args.seeds, [24, 40, 64], [0, 1, 2], [16, 32])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "spectral_controller_rows.csv", rows)
    write_csv(output / "spectral_controller_summary.csv", summarize(rows, args.tau_l, args.tau_k, args.tau_sigma))
    print(f"wrote {len(rows)} CPU controller states and {len(FAMILIES)} family summaries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
