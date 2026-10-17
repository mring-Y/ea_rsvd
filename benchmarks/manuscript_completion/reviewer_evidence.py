"""Audit rank-tail interpretation and cached-output alternatives from saved data.

No solver run is performed. Energy diagnostics are not certified interval bounds
when QR/SVD factors were computed in finite precision. Prefix checks use exported
factors, with FP64 Gram/cross residuals on the effective FP32 input.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct

import numpy as np


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def tail_lower_bound(rho_k, rho_l):
    return math.sqrt(max(0.0, rho_k * rho_k - rho_l * rho_l))


def inspect_energy(grid, references):
    reference = {r["matrix"]: float(r["cpu_reference_rankk_residual"]) for r in references}
    rows = []
    for matrix in sorted(reference):
        states = [r for r in grid if r["matrix"] == matrix and int(r["rank"]) == 16]
        tails = [tail_lower_bound(float(r["exact_rank_residual"]), float(r["exact_projection_residual"])) for r in states]
        rho = [float(r["exact_rank_residual"]) for r in states]
        opt = reference[matrix]
        rows.append(dict(matrix=matrix, states=len(states), cpu_rank16_reference=opt,
            minimum_energy_tail=min(tails), maximum_energy_tail=max(tails),
            tails_above_reference_by_1e_6=sum(x > opt + 1e-6 for x in tails),
            energy_residuals_below_reference_by_1e_6=sum(x < opt - 1e-6 for x in rho),
            reference_above_target_020=opt > .20))
    return rows


def read_csr(path):
    with Path(path).open("rb") as f:
        magic, m, n, nz = struct.unpack("<8sQQQ", f.read(32))
        if magic != b"PGCSR001":
            raise ValueError("invalid CSR header")
        offsets = np.fromfile(f, dtype="<i4", count=m+1)
        columns = np.fromfile(f, dtype="<i4", count=nz)
        values = np.fromfile(f, dtype="<f8", count=nz).astype(np.float32).astype(np.float64)
    return int(m), int(n), offsets, columns, values


def check_prefixes(input_root, factor_root, audit_module):
    spec = importlib.util.spec_from_file_location("factor_audit", audit_module)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    rows = []
    hashes = {}
    for matrix in ("west2021", "lp_osa_30"):
        csr = input_root / (matrix + ".csr")
        args = read_csr(csr)
        source = factor_root / matrix / "32_cached.bin"
        hashes[matrix + "/input"] = hashlib.sha256(csr.read_bytes()).hexdigest()
        hashes[matrix + "/rank32"] = hashlib.sha256(source.read_bytes()).hexdigest()
        um, sm, vm = audit.read_factors(source)
        for k in (8, 16, 32):
            original = factor_root / matrix / (str(k) + "_cached.bin")
            hashes[matrix + "/rank" + str(k)] = hashlib.sha256(original.read_bytes()).hexdigest()
            u, s, v = audit.read_factors(original)
            prefix = audit.gram_residual(*args, um[:, :k], sm[:k], vm[:, :k])
            separate = audit.gram_residual(*args, u, s, v)
            rows.append(dict(matrix=matrix, rank=k,
                prefix_rank32_residual=prefix, separately_formed_residual=separate,
                absolute_residual_difference=abs(prefix-separate),
                maximum_singular_value_difference=float(np.max(np.abs(sm[:k]-s))),
                prefix_u_orth_frobenius=float(np.linalg.norm(um[:, :k].T@um[:, :k]-np.eye(k))),
                prefix_v_orth_frobenius=float(np.linalg.norm(vm[:, :k].T@vm[:, :k]-np.eye(k)))))
    return rows, hashes


def cache_models(rows):
    output = []
    for name in sorted({r["matrix"] for r in rows}):
        group = sorted([r for r in rows if r["matrix"] == name], key=lambda r: int(r["rank"]))
        m, n, ell = map(int, (group[0]["rows"], group[0]["cols"], group[0]["ell"]))
        build = float(group[0]["cache_build_ms"])
        form = sum(float(r["cached_factor_ms"]) for r in group)
        fresh = sum(float(r["independent_recompute_ms"]) for r in group)
        cal = float(next(r["rank16_grid_calibration_ms"] for r in group if r["rank"] == "16"))
        output.append(dict(matrix=name, sequence="8,16,32", requests=3,
            projected_state_one_sequence_ms=build+form,
            independent_solves_one_sequence_ms=fresh,
            projected_state_with_calibration_one_sequence_ms=cal+build+form,
            projected_state_device_payload_bytes=4*((m+n)*ell+ell*ell+ell),
            materialized_rank32_payload_bytes=4*((m+n)*32+32),
            materialized_rank32_fresh_solve_median_ms=float(group[-1]["independent_recompute_ms"]),
            cost_interpretation="Sum of measured per-operation medians; not a new sequence timing. Rank32 prefix access performs no factor arithmetic. Its selection/copy cost is not timed here."))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--cache-costs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--input-root", type=Path)
    p.add_argument("--factor-root", type=Path)
    p.add_argument("--audit-module", type=Path)
    a = p.parse_args()
    result = dict(energy_interpretation="Exact-arithmetic tail lower bound; saved FP32 energy residuals are diagnostics, not certified bounds.",
        rank_tail_checks=inspect_energy(read_csv(a.grid), read_csv(a.references)),
        cache_cost_models=cache_models(read_csv(a.cache_costs)))
    if any(x is not None for x in (a.input_root, a.factor_root, a.audit_module)):
        if not all(x is not None for x in (a.input_root, a.factor_root, a.audit_module)):
            p.error("prefix audit requires input-root, factor-root and audit-module together")
        result["prefix_checks"], result["prefix_source_sha256"] = check_prefixes(a.input_root, a.factor_root, a.audit_module)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
