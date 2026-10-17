#!/usr/bin/env python3
"""Probe RAPIDS TruncatedSVD and label the actual algorithm that ran.

RAPIDS releases differ in the accepted ``algorithm`` values. The script tries
``randomized`` first and then supported cuML modes such as ``jacobi`` or
``full``. A jacobi/full result is an official cuML TruncatedSVD reference, not
a randomized-SVD result.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrices", nargs="+", required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--power", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cps
        from scipy.io import mmread
        from cuml.decomposition import TruncatedSVD
    except Exception as exc:  # import status is part of the result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows([["status", "error"], ["unavailable", repr(exc)]])
        print(f"cuML unavailable: {exc}")
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    try:
        params = inspect.signature(TruncatedSVD).parameters
        accepts_extra = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        params, accepts_extra = {}, True

    def supports(name: str) -> bool:
        return name in params or accepts_extra

    def fit_with_supported_mode(x, rank: int, power: int, seed: int):
        attempts: list[str] = []
        candidates = ["randomized", "jacobi", "full"] if supports("algorithm") else [None]
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                kwargs = {"n_components": rank}
                if supports("n_iter"):
                    kwargs["n_iter"] = power + 1
                if supports("random_state"):
                    kwargs["random_state"] = seed
                if candidate is not None:
                    kwargs["algorithm"] = candidate
                model = TruncatedSVD(**kwargs)
                attempts.append(candidate or "default")
                model.fit(x)
                return model, candidate or "default", ";".join(attempts)
            except Exception as exc:
                last_error = exc
                attempts.append(f"{candidate or 'default'}:{type(exc).__name__}")
        raise RuntimeError(f"no supported cuML TruncatedSVD mode; attempts={attempts}; last={last_error}")

    for name in args.matrices:
        path = args.data_root / name / name / f"{name}.mtx"
        t0 = time.perf_counter()
        try:
            host = mmread(path).tocsr().astype("float32")
            x = cps.csr_matrix(host)
            fro2 = float(cp.asarray(x.multiply(x).sum()).get())
            model, selected_mode, attempts = fit_with_supported_mode(x, args.rank, args.power, args.seed)
            singular_values = cp.asarray(model.singular_values_)
            captured = float((singular_values * singular_values).sum().get())
            rho = (max(fro2 - captured, 0.0) / max(fro2, 1e-30)) ** 0.5
            comp = cp.asarray(model.components_)
            gram = comp @ comp.T
            orth = float(cp.max(cp.abs(gram - cp.eye(args.rank, dtype=gram.dtype))).get())
            rows.append({"status": "ok", "matrix": name, "rows": host.shape[0],
                         "cols": host.shape[1], "nnz": host.nnz, "rank": args.rank,
                         "power_steps": args.power, "seed": args.seed,
                         "projection_residual": rho, "component_orthogonality": orth,
                         "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                         "algorithm_selected": selected_mode, "algorithm_attempts": attempts,
                         "backend": f"cuml.decomposition.TruncatedSVD(algorithm={selected_mode})"})
            print(name, f"rho={rho:.8g}", f"ms={rows[-1]['elapsed_ms']:.1f}")
        except Exception as exc:
            text = repr(exc)
            status = "unsupported_or_failed_input"
            rows.append({"status": status, "matrix": name, "error": text,
                         "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                         "algorithm_selected": "", "algorithm_attempts": "",
                         "backend": "cuml.decomposition.TruncatedSVD"})
            print(f"{name}: {exc}")
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if any(row.get("status") != "ok" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
