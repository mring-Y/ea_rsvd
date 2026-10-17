#!/usr/bin/env python3
"""Run a small, non-interactive structure-sensitivity sweep.

The runner uses the exact no-probe executable and writes one CSV per case so a
failed case cannot corrupt the summary. Matrix files are supplied by the user
or by the AutoDL download profile; they are never bundled in the repository.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


DEFAULT_MATRICES = ("ca-AstroPh", "ca-HepPh", "email-EuAll", "p2p-Gnutella31")
DEFAULT_SEEDS = (20261011, 20261012)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True,
                        help="directory containing <name>/<name>.mtx or <name>.mtx")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--matrices", nargs="+", default=list(DEFAULT_MATRICES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--oversampling", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    executable = args.executable.resolve()
    matrix_root = args.matrix_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not executable.is_file():
        raise FileNotFoundError(executable)

    failures: list[dict[str, object]] = []
    for matrix_name in args.matrices:
        nested = matrix_root / matrix_name / f"{matrix_name}.mtx"
        flat = matrix_root / f"{matrix_name}.mtx"
        matrix = nested if nested.is_file() else flat
        if not matrix.is_file():
            failures.append({"matrix": matrix_name, "seed": "", "power_steps": "", "reason": "missing_matrix"})
            continue
        for seed in args.seeds:
            for power_steps in (0, 1, 2):
                stem = f"{matrix_name}_seed{seed}_q{power_steps}"
                output = output_dir / f"{stem}.csv"
                log = output_dir / f"{stem}.log"
                command = [str(executable), str(matrix), str(output), str(args.rank),
                           str(args.oversampling), str(power_steps), str(seed)]
                try:
                    completed = subprocess.run(command, capture_output=True, text=True,
                                               timeout=args.timeout, check=False)
                except subprocess.TimeoutExpired as exc:
                    log.write_text((exc.stdout or "") + (exc.stderr or "") +
                                   "\nTIMEOUT\n", encoding="utf-8", errors="replace")
                    failures.append({"matrix": matrix_name, "seed": seed,
                                     "power_steps": power_steps, "reason": "timeout"})
                    continue
                log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
                if completed.returncode != 0 or not output.is_file():
                    failures.append({"matrix": matrix_name, "seed": seed,
                                     "power_steps": power_steps,
                                     "reason": f"exit_{completed.returncode}"})
    with (output_dir / "failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["matrix", "seed", "power_steps", "reason"])
        writer.writeheader()
        writer.writerows(failures)
    print(f"completed structure sweep; failures={len(failures)}; output={output_dir}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
