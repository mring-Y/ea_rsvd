#!/usr/bin/env python3
"""Check energy-derived residuals against explicit reconstruction in NumPy.

This CPU mechanism experiment uses the same Q and B for every rank at a
given width/power/seed. It reports finite-precision error in the residual
identity and never presents CPU results as CUDA performance measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def run() -> list[dict]:
    rows = []
    for family in ["low-rank-small-tail", "slow-decay"]:
        for seed in range(20261010, 20261015):
            rng = np.random.default_rng(seed)
            u = np.linalg.qr(rng.standard_normal((128, 96)), mode="reduced")[0]
            v = np.linalg.qr(rng.standard_normal((96, 96)), mode="reduced")[0]
            spectrum = (np.r_[np.geomspace(1.0, .15, 24), np.full(72, 1e-5)]
                        if family == "low-rank-small-tail"
                        else np.arange(1, 97, dtype=np.float64) ** -.4)
            source = (u * spectrum) @ v.T
            for policy, dtype in [("FP32-factors", np.float32), ("FP64-factors", np.float64)]:
                a = source.astype(dtype)
                a64 = a.astype(np.float64)
                matrix_energy = float(np.sum(a64 * a64))
                for width in [32, 48, 64]:
                    # Shared test matrix for both arithmetic policies.
                    omega = np.random.default_rng(seed + width).standard_normal((96, width)).astype(dtype)
                    q = np.linalg.qr(a @ omega, mode="reduced")[0]
                    for power in [0, 1, 2]:
                        if power:
                            z = np.linalg.qr(a.T @ q, mode="reduced")[0]
                            q = np.linalg.qr(a @ z, mode="reduced")[0]
                        b = q.T @ a
                        ub, sigma, vt = np.linalg.svd(b, full_matrices=False)
                        q64, b64 = q.astype(np.float64), b.astype(np.float64)
                        spectrum64 = sigma.astype(np.float64)
                        projection_energy = float(np.sum(spectrum64 ** 2))
                        raw_projection_fraction = (matrix_energy - projection_energy) / matrix_energy
                        rho_l = math.sqrt(max(raw_projection_fraction, 0.0))
                        direct_l = float(np.linalg.norm(a64 - q64 @ b64) / math.sqrt(matrix_energy))
                        orth = float(np.max(np.abs(q64.T @ q64 - np.eye(width))))
                        for rank in [8, 16, 32]:
                            bk64 = ((ub[:, :rank].astype(np.float64) * spectrum64[:rank])
                                    @ vt[:rank, :].astype(np.float64))
                            rho_k = math.sqrt(max((matrix_energy - float(np.sum(spectrum64[:rank] ** 2)))
                                                  / matrix_energy, 0.0))
                            direct_k = float(np.linalg.norm(a64 - q64 @ bk64) / math.sqrt(matrix_energy))
                            truncation = float(np.linalg.norm(b64 - bk64) ** 2 / matrix_energy)
                            energy_gap = rho_k ** 2 - rho_l ** 2
                            rows.append({
                                "family": family, "seed": seed, "arithmetic": policy,
                                "rank": rank, "width": width, "power": power,
                                "energy_rho_l": rho_l, "direct_rho_l": direct_l,
                                "energy_rho_k": rho_k, "direct_rho_k": direct_k,
                                "abs_projection_residual_difference": abs(rho_l - direct_l),
                                "abs_rank_residual_difference": abs(rho_k - direct_k),
                                "abs_projection_squared_difference": abs(rho_l ** 2 - direct_l ** 2),
                                "abs_rank_squared_difference": abs(rho_k ** 2 - direct_k ** 2),
                                "abs_truncation_energy_difference": abs(energy_gap - truncation),
                                "q_orthogonality_host_fp64": orth,
                                "raw_projection_fraction": raw_projection_fraction,
                                "energy_label": "adequate" if rho_k <= .2 else (
                                    "rank-limited" if rho_l <= .2 else "capture-unresolved"),
                                "direct_label": "adequate" if direct_k <= .2 else (
                                    "rank-limited" if direct_l <= .2 else "capture-unresolved"),
                            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = run()
    with (args.output / "residual_identity_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for policy in ["FP32-factors", "FP64-factors"]:
        subset = [r for r in rows if r["arithmetic"] == policy]
        summaries.append({
            "arithmetic": policy, "rows": len(subset),
            "label_agreement": sum(r["energy_label"] == r["direct_label"] for r in subset) / len(subset),
            "projection_clipped": sum(r["raw_projection_fraction"] < 0 for r in subset),
            **{"max_" + field: max(r[field] for r in subset) for field in [
                "abs_projection_residual_difference", "abs_rank_residual_difference",
                "abs_projection_squared_difference", "abs_rank_squared_difference",
                "abs_truncation_energy_difference", "q_orthogonality_host_fp64"]}})
    (args.output / "residual_identity_summary.json").write_text(
        json.dumps({"rows": len(rows), "matrix_size": [128, 96],
                    "scope": "CPU direct-reconstruction numerical validation; no GPU timings",
                    "threshold": .2, "summaries": summaries}, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
