"""A full GPU-output correctness gate; runnable before a public campaign."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy import sparse
from scipy.io import mmwrite

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_support import prepare_matrix, read_csr, read_factors, validate_factors


def run_gate(binary: Path, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(901)
    fixtures = {"tall": rng.normal(size=(64, 40)),
                "wide": rng.normal(size=(40, 64)),
                "rank_deficient": np.diag([5., 4., 3., 2., 1., 0., 0., 0.])}
    records = []
    for name, dense in fixtures.items():
        source, csr = directory / (name+".mtx"), directory / (name+".csr")
        mmwrite(source, sparse.csr_matrix(dense))
        prepare_matrix(source, csr)
        for precision in ("fp32", "fp64"):
            a = read_csr(csr, precision)
            true_s = np.linalg.svd(a.toarray(), compute_uv=False)
            k = 4
            optimal = float(np.linalg.norm(true_s[k:])/np.linalg.norm(true_s))
            for solver, power, block in (("rsvd", 0, 0), ("rsvd", 2, 0), ("primme", 0, 1), ("primme", 0, 4)):
                stem = directory / f"{name}_{precision}_{solver}_q{power}_b{block}"
                command = [str(binary), "--matrix", str(csr), "--solver", solver,
                           "--rank", str(k), "--precision", precision,
                           "--ell", str(min(a.shape)), "--power", str(power), "--seed", "901",
                           "--eps", "1e-5" if precision == "fp32" else "1e-10",
                           "--warmup", "1", "--repeats", "2",
                           "--output", str(stem)+".bin", "--json", str(stem)+".json"]
                if block:
                    command.extend(["--block-size", str(block)])
                cp = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, timeout=120, stdin=subprocess.DEVNULL)
                (directory / (stem.name+".log")).write_text(cp.stdout)
                if cp.returncode:
                    raise RuntimeError(f"GPU correctness failed: {stem.name}, rc={cp.returncode}: {cp.stdout[-2000:]}")
                u, sigma, v = read_factors(Path(str(stem)+".bin"))
                validation = validate_factors(a, u, sigma, v)
                direct = float(np.linalg.norm(a.toarray()-(u*sigma)@v.T)/np.linalg.norm(a.toarray()))
                tolerance = 2e-4 if precision == "fp32" else 1e-8
                if abs(validation["relative_residual"]-direct) > 1e-10:
                    raise AssertionError(f"Independent residual disagrees with direct reconstruction: {stem.name}")
                if abs(direct-optimal) > tolerance:
                    raise AssertionError(f"Expected best rank-{k} approximation: {stem.name}, got {direct}, optimum {optimal}")
                if max(validation["u_orthogonality"], validation["v_orthogonality"]) > tolerance:
                    raise AssertionError(f"Incorrect singular vector output: {stem.name}")
                np.testing.assert_allclose(sigma, true_s[:k], rtol=tolerance, atol=tolerance)
                records.append({"case": stem.name, "optimal_residual": optimal, **validation})
    payload = {"complete_factor_records": records}
    (directory / "gpu_correctness.json").write_text(json.dumps(payload, indent=2)+"\n")
    print(f"GPU correctness passed: {len(records)} full-factor comparisons")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.binary.is_file():
        raise SystemExit("FAIL: full-output GPU comparison executable is not built")
    run_gate(args.binary.resolve(), args.out)
