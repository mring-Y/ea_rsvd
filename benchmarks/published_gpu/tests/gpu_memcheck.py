"""Optional Compute Sanitizer check of both complete-factor GPU backends."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy import sparse
from scipy.io import mmwrite

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_support import prepare_matrix, read_csr, read_factors, validate_factors


def memcheck(binary: Path, tool: str, directory: Path, timeout: int):
    directory.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(733)
    fixtures = {"tall": rng.normal(size=(64, 40)), "wide": rng.normal(size=(40, 64))}
    rows = []
    for name, dense in fixtures.items():
        source, csr = directory / (name+".mtx"), directory / (name+".csr")
        mmwrite(source, sparse.csr_matrix(dense))
        prepare_matrix(source, csr)
        for precision in ("fp32", "fp64"):
            a = read_csr(csr, precision)
            spectrum = np.linalg.svd(a.toarray(), compute_uv=False)
            optimal = float(np.linalg.norm(spectrum[4:])/np.linalg.norm(spectrum))
            for solver in ("rsvd", "primme"):
                case = f"{name}_{precision}_{solver}"
                stem = directory / case
                command = [tool, "--tool", "memcheck", "--error-exitcode", "99",
                           str(binary), "--matrix", str(csr), "--solver", solver,
                           "--precision", precision, "--rank", "4", "--ell", str(min(a.shape)),
                           "--power", "2", "--seed", "733", "--block-size", "4",
                           "--eps", "1e-5" if precision == "fp32" else "1e-10",
                           "--warmup", "0", "--repeats", "1",
                           "--output", str(stem)+".bin", "--json", str(stem)+".json"]
                row = {"case": case, "tool": tool, "command": command}
                try:
                    cp = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, stdin=subprocess.DEVNULL, timeout=timeout)
                    (directory / (case+".log")).write_text(cp.stdout)
                    row["exit_code"] = cp.returncode
                    row["zero_error_summary"] = "ERROR SUMMARY: 0 errors" in cp.stdout
                    if cp.returncode or not row["zero_error_summary"]:
                        row.update(status="memcheck_failure", failure_tail=cp.stdout[-3000:])
                    else:
                        checks = validate_factors(a, *read_factors(Path(str(stem)+".bin")))
                        tolerance = 2e-4 if precision == "fp32" else 1e-8
                        row.update(checks)
                        numerical_ok = (abs(checks["relative_residual"]-optimal) <= tolerance and
                                        max(checks["u_orthogonality"], checks["v_orthogonality"]) <= tolerance)
                        row.update(status="ok" if numerical_ok else "numerical_failure",
                                   optimal_residual=optimal)
                except subprocess.TimeoutExpired as exc:
                    tail = exc.stdout or b""
                    (directory / (case+".log")).write_bytes(tail if isinstance(tail, bytes) else tail.encode())
                    row.update(status="timeout", timeout_seconds=timeout)
                except (ValueError, OSError) as exc:
                    row.update(status="validation_failure", error=repr(exc))
                rows.append(row)
                (directory / "memcheck_records.json").write_text(json.dumps(rows, indent=2)+"\n")
                print(f"[memcheck] {case}: {row['status']}", flush=True)
    failures = sum(r["status"] != "ok" for r in rows)
    print(f"Compute Sanitizer: {len(rows)-failures}/{len(rows)} checks passed", flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tool", default="compute-sanitizer")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    tool_path = shutil.which(args.tool)
    if not tool_path:
        raise SystemExit("Compute Sanitizer is unavailable; install the official CUDA tool or provide --tool /path/compute-sanitizer")
    if not args.binary.is_file():
        raise SystemExit("GPU binary is missing; build it first")
    sys.exit(memcheck(args.binary.resolve(), tool_path, args.out, args.timeout))
