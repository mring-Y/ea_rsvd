#!/usr/bin/env python3
"""Run a bounded rank/power sweep on public non-LP Matrix Market inputs.

The executable is supplied by the caller so the same script works with the
Windows development build and with the Linux AutoDL bundle.  A failed case is
recorded and the remaining cases continue; the output is therefore useful
even when a large input reaches the device-memory limit.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_last_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"no CSV row written to {path}")
    return rows[-1]


def matrix_path(root: Path, name: str) -> Path:
    candidates = [
        root / name / f"{name}_general_real.mtx",
        root / name / name / f"{name}.mtx",
        root / name / f"{name}.mtx",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no normalized Matrix Market input for {name} below {root}")


def run_case(exe: Path, matrix: Path, out_csv: Path, rank: int, oversampling: int,
             power: int, seed: int, timeout: float) -> dict[str, object]:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        out_csv.unlink()
    command = [str(exe), str(matrix), str(out_csv), str(rank), str(oversampling), str(power), str(seed)]
    started = datetime.now(timezone.utc).isoformat()
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   check=False, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {"status": "timeout", "returncode": 124, "stdout": exc.stdout or "",
                "stderr": exc.stderr or "", "started_utc": started}
    if completed.returncode != 0:
        return {"status": "failed", "returncode": completed.returncode,
                "stdout": completed.stdout, "stderr": completed.stderr,
                "started_utc": started}
    try:
        row = parse_last_row(out_csv)
    except (OSError, RuntimeError, ValueError) as exc:
        return {"status": "failed", "returncode": 65,
                "stdout": completed.stdout, "stderr": f"invalid output: {exc}",
                "started_utc": started}
    result: dict[str, object] = {
        "status": "ok", "returncode": 0, "stdout": completed.stdout,
        "stderr": completed.stderr, "started_utc": started,
    }
    result.update(row)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matrices", nargs="+", default=["ca-GrQc", "ca-HepPh", "email-Enron"])
    parser.add_argument("--ranks", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--powers", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261015, 20261016])
    parser.add_argument("--oversampling", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args()
    exe = args.executable.resolve()
    if not exe.is_file():
        raise SystemExit(f"executable not found: {exe}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for name in args.matrices:
        try:
            matrix = matrix_path(args.matrix_root.resolve(), name)
        except FileNotFoundError as exc:
            print(f"{name} missing input; skipping: {exc}", flush=True)
            rows.append({"matrix": name, "status": "missing_input", "returncode": 66,
                         "input": "", "raw_csv": ""})
            continue
        for rank in args.ranks:
            for power in args.powers:
                for seed in args.seeds:
                    raw = args.output / "raw" / f"{name}_k{rank}_q{power}_seed{seed}.csv"
                    result = run_case(exe, matrix, raw, rank, args.oversampling,
                                      power, seed, args.timeout_seconds)
                    record: dict[str, object] = {
                        "matrix": name, "rank": rank, "oversampling": args.oversampling,
                        "power_steps": power, "seed": seed,
                        "input": str(matrix), "raw_csv": str(raw),
                    }
                    record.update(result)
                    rows.append(record)
                    print(name, rank, power, seed, result.get("status"), flush=True)
    output_csv = args.output / "nonlp_rank_sweep.csv"
    fields = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    ok = sum(row.get("status") == "ok" for row in rows)
    print(f"wrote {output_csv} ({ok}/{len(rows)} cases completed)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
