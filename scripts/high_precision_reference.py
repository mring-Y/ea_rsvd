#!/usr/bin/env python3
"""Compute repeated float64 ARPACK references for rank-k Frobenius error."""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import ArpackNoConvergence, svds


def one_reference(path: Path, rank: int, seed: int, tol: float, maxiter: int, ncv: int) -> tuple[float, float, int, int, int]:
    a = mmread(path)
    if not issparse(a):
        a = csr_matrix(np.asarray(a, dtype=np.float64))
    else:
        a = a.tocsr().astype(np.float64)
    a.sum_duplicates()
    fro2 = float(a.multiply(a).sum(dtype=np.float64))
    values = svds(a, k=rank, which="LM", return_singular_vectors=False,
                  tol=tol, maxiter=maxiter, ncv=min(ncv, min(a.shape) - 1),
                  solver="arpack", random_state=seed)
    values = np.sort(np.asarray(values, dtype=np.float64))[::-1]
    rho = math.sqrt(max(fro2 - float(np.dot(values, values)), 0.0) / max(fro2, 1e-300))
    return rho, math.sqrt(fro2), int(a.shape[0]), int(a.shape[1]), int(a.nnz)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrices", nargs="+", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--maxiter", type=int, default=20000)
    parser.add_argument("--ncv", type=int, default=128)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for name in args.matrices:
        path = args.data_root / name / name / f"{name}.mtx"
        values: list[float] = []
        elapsed_ms = 0.0
        error = ""
        norm = float("nan")
        shape = ("", "", "")
        for seed in range(args.repeats):
            start = time.perf_counter()
            try:
                rho, norm, rows_n, cols_n, nnz = one_reference(path, args.rank, seed, args.tol, args.maxiter, args.ncv)
                shape = (rows_n, cols_n, nnz)
                values.append(rho)
            except ArpackNoConvergence as exc:
                error = f"ArpackNoConvergence: {exc}"
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                break
            finally:
                elapsed_ms += (time.perf_counter() - start) * 1000.0
        rows.append({"matrix": name, "status": "converged" if len(values) == args.repeats else "incomplete",
                     "rows": shape[0], "cols": shape[1], "nnz": shape[2], "rank": args.rank, "repeats_requested": args.repeats,
                     "repeats_converged": len(values), "rho_min": min(values) if values else "",
                     "rho_max": max(values) if values else "", "rho_mean": float(np.mean(values)) if values else "",
                     "rho_range": max(values) - min(values) if values else "", "frobenius_norm": norm,
                     "tol": args.tol, "maxiter": args.maxiter, "ncv": args.ncv,
                     "solver": "scipy.sparse.linalg.svds(arpack), float64", "elapsed_ms": elapsed_ms,
                     "error": error})
        print(f"{name}: {rows[-1]['status']}, repeats={len(values)}/{args.repeats}, error={error}", flush=True)
    fields = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if any(row["status"] != "converged" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
