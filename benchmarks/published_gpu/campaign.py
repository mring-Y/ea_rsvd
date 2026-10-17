"""Bounded, noninteractive matched sparse GPU solver campaign."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import tarfile
import time

import numpy as np
import scipy

from benchmark_support import prepare_matrix, read_csr, read_factors, sha256, validate_factors

ROOT = Path(__file__).resolve().parent
SOURCES = {"west2021": "HB", "lp_osa_30": "LPnetlib", "lp_pds_02": "LPnetlib",
           "ca-HepPh": "SNAP", "email-Enron": "SNAP"}
TARGETS = [.20, .40, .60, .80, .95]


def valid(row):
    return (row.get("status") == "ok" and
            all(math.isfinite(float(row.get(k, math.nan))) for k in
                ["relative_residual", "compute_ms", "u_orthogonality", "v_orthogonality"]) and
            max(row["u_orthogonality"], row["v_orthogonality"]) <= 1e-3 and
            row["compute_ms"] > 0)


def make_quality_comparisons(rows, targets=TARGETS):
    groups = defaultdict(list)
    keys = ["matrix", "csr_sha256", "precision", "rank", "seed", "gpu"]
    for row in rows:
        if row.get("solver") in {"rsvd_diagnostic_full", "primme_svds_cuda"}:
            groups[tuple(row.get(k, "") for k in keys)].append(row)
    comparisons = []
    for key, entries in sorted(groups.items()):
        for target in targets:
            result = dict(zip(keys, key))
            result["target_residual"] = target
            for short, solver in [("rsvd", "rsvd_diagnostic_full"), ("primme", "primme_svds_cuda")]:
                attempts = [r for r in entries if r["solver"] == solver]
                usable = [r for r in attempts if valid(r)]
                eligible = [r for r in usable if r["relative_residual"] <= target]
                best = min(eligible, key=lambda r: r["compute_ms"]) if eligible else None
                result[short+"_attempts"] = len(attempts)
                result[short+"_failures"] = len(attempts)-len(usable)
                result[short+"_status"] = "met_target" if best else ("target_not_reached" if usable else "no_valid_result")
                result[short+"_ms"] = best["compute_ms"] if best else None
                result[short+"_residual"] = best["relative_residual"] if best else None
                result[short+"_configuration"] = best.get("configuration", "") if best else None
                result[short+"_best_residual"] = min((r["relative_residual"] for r in usable), default=None)
            result["speedup_rsvd_over_primme"] = (result["primme_ms"]/result["rsvd_ms"]
                if result["rsvd_ms"] is not None and result["primme_ms"] is not None else None)
            comparisons.append(result)
    return comparisons


def write_csv(path, rows):
    columns = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def make_near_best_comparisons(rows, factors=(1.01, 1.05)):
    groups = defaultdict(list)
    keys = ["matrix", "csr_sha256", "precision", "rank", "seed", "gpu"]
    for row in rows:
        groups[tuple(row.get(k, "") for k in keys)].append(row)
    result = []
    for entries in groups.values():
        reference = [r for r in entries if valid(r) and r["solver"] == "primme_svds_cuda" and r.get("eps", 1) <= 1e-5]
        if not reference:
            continue
        best = min(reference, key=lambda r: r["relative_residual"])
        for factor in factors:
            target = min(1., best["relative_residual"]*factor+1e-8)
            comparisons = make_quality_comparisons(entries, [target])
            for row in comparisons:
                row.update(reference_residual=best["relative_residual"], relative_quality_allowance=factor,
                           reference_kind="validated_primme_eps_le_1e-5")
                result.append(row)
    return result


def summarize(out, rows, manifest):
    comparisons = make_quality_comparisons(rows)
    write_csv(out / "records.csv", rows)
    write_csv(out / "quality_comparisons.csv", comparisons)
    near_best = make_near_best_comparisons(rows)
    write_csv(out / "near_reference_comparisons.csv", near_best)
    status_counts = defaultdict(int)
    for row in rows:
        status_counts[row["status"]] += 1
    lines = ["# Matched sparse GPU solver results", "",
             "Both solvers return complete U, singular values, and V at the same output rank. "
             "The primary time is synchronized wall time after resident CSR setup, including "
             "per-solve allocations, RNG, host control, final vectors, and randomized-SVD diagnostics. "
             "Factor export and independent FP64 validation are reported separately.", "",
             "The rows below are observed finite-grid minima under a common actual residual target. "
             "They are descriptive offline frontiers, not held-out policy performance or a global optimum. "
             "Candidate search costs and all failures are retained in the raw records.", "",
             "PRIMME eps controls singular-triplet convergence; it is not the reconstruction target.", "",
             "Status counts: `"+json.dumps(dict(status_counts), sort_keys=True)+"`", "",
             "| Actual residual target | Paired successful cases | Geometric mean PRIMME / rSVD time |",
             "|---|---:|---:|"]
    for target in TARGETS:
        pairs = [r["speedup_rsvd_over_primme"] for r in comparisons
                 if r["target_residual"] == target and r["speedup_rsvd_over_primme"] is not None]
        ratio = math.exp(sum(math.log(x) for x in pairs)/len(pairs)) if pairs else None
        lines.append(f"| {target:.2f} | {len(pairs)} / {len(comparisons)//len(TARGETS)} | "
                     + (f"{ratio:.4f}" if ratio is not None else "not available")+" |")
    lines.extend(["", "A ratio above one means rSVD has the lower observed time at that target. "
                  "Cases with unavailable/invalid results or unattained targets are not assigned a speedup.", "",
                  "`near_reference_comparisons.csv` also restricts the target to 1% or 5% above "
                  "the smallest validated PRIMME residual from eps <= 1e-5 for that input/rank/seed. "
                  "This is a numerical reference, not a certified global optimum.", "",
                  "All requested matrix preparation statuses are in `matrix_manifest.json`. "
                  "Unattained targets are not called hardware failures, and timeouts are not called OOM.", ""])
    (out / "summary.md").write_text("\n".join(lines))
    (out / "matrix_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


def get_source(name, root, cache, logs):
    roots = [root, root / "data/suitesparse", root / "project/data/suitesparse",
             root / "downloaded_data/public_nonlp", root / "downloaded_data/public_nonlp_general"]
    for supplied in roots:
        for path in [supplied / (name+".mtx"), supplied / (name+".mtx.gz"),
                     supplied / name / name / (name+".mtx"), supplied / name / (name+".mtx")]:
            if path.is_file():
                return path
    if name not in SOURCES:
        raise FileNotFoundError(f"No supplied input or official download mapping for {name}")
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / (name+".mtx")
    if target.exists():
        return target
    archive = cache / (name+".tar.gz.part")
    urls = [f"https://sparse.tamu.edu/MM/{SOURCES[name]}/{name}.tar.gz",
            f"https://suitesparse-collection-website.herokuapp.com/MM/{SOURCES[name]}/{name}.tar.gz"]
    with (logs / ("download_"+name+".log")).open("w") as log:
        for url in urls:
            command = ["curl", "-L", "--fail", "--show-error", "-C", "-", "--retry", "5",
                       "--retry-delay", "5", "--connect-timeout", "20", "--max-time", "300",
                       "-o", str(archive), url]
            log.write(url+"\n"); log.flush()
            try:
                cp = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, timeout=900)
                if cp.returncode:
                    continue
                with tarfile.open(archive) as tf:
                    files = [m for m in tf.getmembers() if m.isfile() and Path(m.name).name == name+".mtx"]
                    if len(files) != 1:
                        raise ValueError("Dataset archive must contain exactly the requested matrix")
                    with tf.extractfile(files[0]) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                return target
            except (OSError, ValueError, tarfile.TarError, subprocess.TimeoutExpired) as exc:
                log.write(repr(exc)+"\n"); log.flush()
    raise RuntimeError(f"Dataset download failed for {name}; other inputs will continue")


def validate_record(stem, a, base, result):
    factors = Path(str(stem)+".bin")
    row = {**base, **result}
    u, sigma, v = read_factors(factors)
    if sigma.size != base["rank"]:
        raise ValueError("Solver returned the wrong output rank")
    validation_start = time.perf_counter()
    row.update(validate_factors(a, u, sigma, v))
    # The energy diagnostic describes the final solve, whereas eligibility
    # uses the worst quality across all timed solves. Keep those scopes apart.
    row["final_repeat_relative_residual"] = row["relative_residual"]
    repeat_files = [Path(str(factors)+f".repeat_{i}.bin") for i in range(result["repeats"])]
    if all(p.exists() for p in repeat_files):
        repeat_checks = [validate_factors(a, *read_factors(p)) for p in repeat_files]
        row["repeat_validations"] = [{k: r[k] for k in ["relative_residual", "u_orthogonality", "v_orthogonality",
                                                          "triplet_residual_max", "validation_method"]} for r in repeat_checks]
        for metric in ["relative_residual", "u_orthogonality", "v_orthogonality", "triplet_residual_max"]:
            row[metric] = max(r[metric] for r in repeat_checks)
        row["triplet_residual_max_relative_fro"] = row["triplet_residual_max"]/row["matrix_frobenius"]
        row["validation_scope"] = "all_measured_repeats_worst_quality"
    else:
        raise ValueError("Missing or incomplete per-repeat factors; cannot validate every measured repeat")
    row["validation_ms"] = (time.perf_counter()-validation_start)*1000
    row["energy_residual_difference"] = (abs(row["rho_k_energy"]-row["final_repeat_relative_residual"])
        if row.get("rho_k_energy") is not None else None)
    if max(row["u_orthogonality"], row["v_orthogonality"]) > 1e-3:
        row["status"] = "invalid_orthogonality"
    if row.get("primme_locking_issue", 0):
        row["status"] = "solver_warning"
    if row.get("primme_a_norm", 0) > 0:
        row["triplet_residual_max_relative_a_norm"] = row["triplet_residual_max"]/row["primme_a_norm"]
    return row


def run_one(binary, csr, a, out, base, options, timeout, retain):
    stem = base["case_id"]
    factors, report, logpath = (out / "raw" / (stem+suffix) for suffix in [".bin", ".json", ".log"])
    command = [str(binary), "--matrix", str(csr), "--output", str(factors), "--json", str(report)]
    for key, value in options.items():
        command.extend(["--"+key, str(value)])
    start = time.perf_counter()
    row = dict(base)
    with logpath.open("w") as log:
        try:
            cp = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, timeout=timeout)
            row["exit_code"] = cp.returncode
            if cp.returncode:
                row["status"] = "solver_failure"
                row["failure_tail"] = logpath.read_text(errors="replace")[-2000:]
                if not retain:
                    for p in out.joinpath("raw").glob(stem+".bin*"):
                        p.unlink()
                return row
        except subprocess.TimeoutExpired:
            row.update(status="timeout", exit_code=124, timeout_seconds=timeout)
            if not retain:
                for p in out.joinpath("raw").glob(stem+".bin*"):
                    p.unlink()
            return row
        finally:
            row["process_wall_ms"] = (time.perf_counter()-start)*1000
    try:
        result = json.loads(report.read_text())
        row.update(result)
        row.update(validate_record(out / "raw" / stem, a, base, result))
    except (ValueError, OSError, KeyError) as exc:
        row.update(status="validation_failure", validation_error=repr(exc))
    finally:
        if not retain and factors.exists():
            factors.unlink()
        if not retain:
            for p in out.joinpath("raw").glob(stem+".bin.repeat_*.bin"):
                p.unlink()
    return row


def campaign(args):
    out = args.out.resolve()
    build_manifest = args.binary.parent / "build_manifest.json"
    if build_manifest.exists():
        provenance = json.loads(build_manifest.read_text())
        expected_source = provenance.get("benchmark_source_sha256")
        expected_binary = provenance.get("benchmark_binary_sha256")
        if expected_source and expected_source != sha256(ROOT / "gpu_compare.cu"):
            raise ValueError("The GPU source differs from the build manifest; rebuild before measuring")
        if expected_binary and expected_binary != sha256(args.binary):
            raise ValueError("The GPU binary differs from the build manifest; rebuild before measuring")
    out.mkdir(parents=True, exist_ok=False)
    for part in ["logs", "raw", "inputs", "sources"]:
        (out / part).mkdir()
    source_names = ["gpu_compare.cu", "campaign.py", "benchmark_support.py", "analyse_results.py",
                    "build_gpu.py", "run_autodl_all.sh", "run_autodl_review.sh", "cpu_crosscheck.py",
                    "rebuild_summary.py"]
    for name in source_names:
        shutil.copy2(ROOT / name, out / "sources" / name)
    rows, manifest = [], []
    if build_manifest.exists():
        shutil.copy2(build_manifest, out / "build_manifest.json")
    environment = {"utc_started": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                   "numpy": np.__version__, "scipy": scipy.__version__, "arguments": dict(vars(args)),
                   "benchmark_source_sha256": sha256(ROOT / "gpu_compare.cu"),
                   "benchmark_binary_sha256": sha256(args.binary),
                   "campaign_source_sha256": sha256(Path(__file__)),
                   "source_snapshot_sha256": {name: sha256(ROOT / name) for name in source_names},
                   "protocol": "complete factors; resident CSR; synchronized wall clock",
                   "cpu_threads": {k: os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]}}
    for key, value in environment["arguments"].items():
        if isinstance(value, Path):
            environment["arguments"][key] = str(value)
    cp = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"],
                        capture_output=True, text=True, stdin=subprocess.DEVNULL)
    environment["nvidia_smi"] = cp.stdout
    gpu_name = cp.stdout.split(",", 1)[0].strip()
    (out / "environment.json").write_text(json.dumps(environment, indent=2)+"\n")
    total_configurations = 0
    for name in args.matrices:
        try:
            start = time.perf_counter()
            print(f"[prepare] {name}", flush=True)
            source = get_source(name, args.matrix_root, ROOT / "data/cache", out / "logs")
            csr = out / "inputs" / (name+".csr")
            matrix_meta = {"name": name, "status": "ok", **prepare_matrix(source, csr),
                           "prepare_ms": (time.perf_counter()-start)*1000}
            manifest.append(matrix_meta)
        except Exception as exc:
            manifest.append({"name": name, "status": "matrix_unavailable", "error": repr(exc)})
            print(f"[skip] {name}: {exc}", flush=True)
            continue
        for precision in args.precisions:
            a = read_csr(csr, precision)
            effective_hash = __import__("hashlib").sha256(a.data.astype("<f8").tobytes()).hexdigest()
            for rank in args.ranks:
                if rank > min(a.shape):
                    continue
                for seed in args.seeds:
                    jobs = []
                    if "rsvd" in args.solvers:
                        for oversampling in args.oversampling:
                            ell = min(rank+oversampling, min(a.shape))
                            for power in args.powers:
                                jobs.append(("rsvd", f"ell{ell}_q{power}", {"ell": ell, "power": power, "diagnostics": 1}))
                    if "primme" in args.solvers:
                        for eps in args.eps:
                            for block in args.block_sizes:
                                jobs.append(("primme", f"eps{eps:g}_b{block}", {"eps": eps, "block-size": block}))
                    random.Random(seed+rank).shuffle(jobs)
                    for solver, configuration, extra in jobs:
                        case_id = f"{name}_{precision}_k{rank}_s{seed}_{solver}_{configuration}"
                        base = {"matrix": name, "precision": precision, "rank": rank, "seed": seed,
                                "csr_sha256": matrix_meta["csr_sha256"], "effective_values_sha256": effective_hash,
                                "configuration": configuration, "case_id": case_id, "gpu": gpu_name,
                                "solver": "primme_svds_cuda" if solver == "primme" else "rsvd_diagnostic_full"}
                        options = {"solver": solver, "rank": rank, "precision": precision, "seed": seed,
                                   "warmup": args.warmup, "repeats": args.repeats,
                                   "max-matvecs": args.max_matvecs, **extra}
                        total_configurations += 1
                        print(f"[run {total_configurations}] {case_id} (solver timeout {args.timeout}s; then independent validation)", flush=True)
                        row = run_one(args.binary, csr, a, out, base, options, args.timeout, args.retain_factors)
                        rows.append(row)
                        with (out / "records.jsonl").open("a") as f:
                            f.write(json.dumps(row, allow_nan=False)+"\n")
                        cost = row.get("compute_ms")
                        residual = row.get("relative_residual")
                        print(f"[done] {row['status']} ms={cost} residual={residual}", flush=True)
                        summarize(out, rows, manifest)
    summarize(out, rows, manifest)
    failures = sum(row["status"] != "ok" for row in rows)
    unavailable = sum(m["status"] != "ok" for m in manifest)
    status = {"completed": True, "configurations": len(rows), "failed_configurations": failures,
              "unavailable_matrices": unavailable, "requested_matrices": len(args.matrices),
              "utc_finished": datetime.now(timezone.utc).isoformat()}
    (out / "completion.json").write_text(json.dumps(status, indent=2)+"\n")
    print(json.dumps(status), flush=True)
    return 2 if failures or unavailable or not rows else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "build/gpu_compare")
    parser.add_argument("--matrix-root", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--matrices", nargs="+", default=list(SOURCES))
    parser.add_argument("--ranks", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261013, 20261014, 20261015, 20261016, 20261017])
    parser.add_argument("--precisions", nargs="+", choices=["fp32", "fp64"], default=["fp32"])
    parser.add_argument("--solvers", nargs="+", choices=["rsvd", "primme"], default=["rsvd", "primme"])
    parser.add_argument("--oversampling", nargs="+", type=int, default=[8, 24])
    parser.add_argument("--powers", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eps", nargs="+", type=float, default=[1e-4, 1e-5, 1e-6])
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-matvecs", type=int, default=20000)
    parser.add_argument("--retain-factors", action="store_true")
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error("GPU executable is missing; run build_gpu.py first")
    args.binary = args.binary.resolve()
    if min(args.ranks) < 1 or min(args.seeds) < 0 or min(args.oversampling) < 0 or min(args.powers) < 0:
        parser.error("Ranks/seeds/width/power must be nonnegative (rank positive)")
    sys.exit(campaign(args))
