"""Independent SciPy/PROPACK spectral check, separate from GPU performance.

Run after GPU timings: this CPU solve is a numerical reference only. Each
matrix/precision job is bounded and all unavailable references are retained.
"""
import argparse
from collections import defaultdict
import json
import inspect
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.sparse.linalg import svds

from benchmark_support import read_csr, sha256, validate_factors
from campaign import valid, write_csv


def worker(csr, precision, ranks, output):
    a = read_csr(csr, precision)
    largest_rank = max(ranks)
    started = time.perf_counter()
    # SciPy 1.15 introduced rng; older supported installations use random_state.
    rng_keyword = "rng" if "rng" in inspect.signature(svds).parameters else "random_state"
    u, s, vt = svds(a, k=largest_rank, which="LM", solver="propack",
                    tol=1e-10, maxiter=min(min(a.shape), max(160, 8*largest_rank)), **{rng_keyword: 901})
    order = np.argsort(s)[::-1]
    u, s, v = u[:, order], s[order], vt[order].T
    rows = []
    for rank in ranks:
        check = validate_factors(a, u[:, :rank], s[:rank], v[:, :rank])
        numerical_ok = (max(check["u_orthogonality"], check["v_orthogonality"]) <= 1e-7
                        and check["triplet_residual_max_relative_fro"] <= 1e-7)
        rows.append({"precision": precision, "rank": rank,
                     "status": "ok" if numerical_ok else "reference_numerical_failure", **check})
    payload = {"solver": "scipy.sparse.linalg.svds_propack", "scipy": scipy.__version__,
               "tol": 1e-10, "random_seed": 901, "rng_keyword": rng_keyword,
               "csr_sha256": sha256(csr), "solve_seconds": time.perf_counter()-started,
               "rank_checks": rows}
    output.write_text(json.dumps(payload, indent=2)+"\n")


def crosscheck(campaigns, output, timeout):
    output.mkdir(parents=True, exist_ok=False)
    jobs = defaultdict(lambda: {"ranks": set(), "records": []})
    for directory in campaigns:
        for line in (directory / "records.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["matrix"], row["csr_sha256"], row["precision"])
            job = jobs[key]
            job["ranks"].add(row["rank"])
            job["records"].append(row)
            csr = directory / "inputs" / (row["matrix"]+".csr")
            if csr.is_file():
                job["csr"] = csr
    comparisons = []
    for index, (key, job) in enumerate(sorted(jobs.items()), 1):
        matrix, checksum, precision = key
        stem = output / f"{matrix}_{precision}_{checksum[:12]}"
        ranks = sorted(job["ranks"])
        print(f"[cpu {index}/{len(jobs)}] {matrix} {precision}, ranks={ranks}", flush=True)
        status, failure = "ok", ""
        try:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker-csr", str(job["csr"]),
                       "--precision", precision, "--ranks", *map(str, ranks), "--out", str(stem)+".json"]
            with Path(str(stem)+".log").open("w") as log:
                cp = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, timeout=timeout)
            if cp.returncode:
                status = "reference_solver_failure"
                failure = Path(str(stem)+".log").read_text(errors="replace")[-2000:]
        except subprocess.TimeoutExpired:
            status, failure = "reference_timeout", f"CPU reference exceeded {timeout} seconds"
        except (OSError, KeyError) as exc:
            status, failure = "reference_unavailable", repr(exc)
        checks = ({r["rank"]: r for r in json.loads(Path(str(stem)+".json").read_text())["rank_checks"]}
                  if status == "ok" else {})
        for rank in ranks:
            check = checks.get(rank, {})
            item = {"matrix": matrix, "precision": precision, "rank": rank, "csr_sha256": checksum,
                    "reference_status": check.get("status", status), "failure": failure,
                    "cpu_reference_residual": check.get("relative_residual"),
                    "cpu_reference_triplet_relative_fro": check.get("triplet_residual_max_relative_fro")}
            for short, solver in (("rsvd", "rsvd_diagnostic_full"), ("primme", "primme_svds_cuda")):
                usable = [r for r in job["records"] if r["rank"] == rank and r["solver"] == solver and valid(r)]
                best = min(usable, key=lambda r: r["relative_residual"]) if usable else None
                item[short+"_best_residual"] = best["relative_residual"] if best else None
                item[short+"_difference_from_cpu_reference"] = (best["relative_residual"]-check["relative_residual"]
                    if best and check.get("status") == "ok" else None)
                item[short+"_valid_candidates"] = len(usable)
            comparisons.append(item)
        write_csv(output / "cpu_crosscheck.csv", comparisons)
        (output / "cpu_crosscheck.json").write_text(json.dumps(comparisons, indent=2)+"\n")
    note = {"completed": True, "jobs": len(jobs), "rank_checks": len(comparisons),
            "valid_references": sum(r["reference_status"] == "ok" for r in comparisons),
            "interpretation": "independent FP64 CPU numerical reference; not a certified optimum or GPU timing baseline"}
    (output / "completion.json").write_text(json.dumps(note, indent=2)+"\n")
    print(json.dumps(note), flush=True)
    return 0 if note["valid_references"] == len(comparisons) else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaigns", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--worker-csr", type=Path)
    parser.add_argument("--precision", choices=["fp32", "fp64"])
    parser.add_argument("--ranks", nargs="+", type=int)
    args = parser.parse_args()
    if args.worker_csr:
        worker(args.worker_csr, args.precision, args.ranks, args.out)
    else:
        if not args.campaigns or args.timeout <= 0:
            parser.error("Provide --campaigns and a positive timeout")
        sys.exit(crosscheck(args.campaigns, args.out, args.timeout))
