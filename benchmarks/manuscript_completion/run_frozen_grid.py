"""Run the frozen held-out matrix validation without interactive input."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


MATRICES = ["ca-AstroPh", "email-EuAll", "p2p-Gnutella31", "roadNet-CA"]
WIDTHS = [24, 32, 40]
POWERS = [0, 1, 2]
SEEDS = [20261015, 20261016]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def matrix_path(root: Path, name: str) -> Path:
    candidates = [root / name / f"{name}.mtx", root / name / name / f"{name}.mtx"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(name)


def parse_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("empty solver CSV")
    return rows[-1]


def run_case(exe: Path, matrix: Path, raw: Path, rank: int, width: int,
             power: int, seed: int, timeout: float):
    raw.parent.mkdir(parents=True, exist_ok=True)
    if raw.exists():
        raw.unlink()
    started = datetime.now(timezone.utc).isoformat()
    command = [str(exe), str(matrix), str(raw), str(rank), str(width - rank), str(power), str(seed)]
    start = time.perf_counter()
    try:
        completed = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True,
                                   text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {"status": "timeout", "returncode": 124, "started_utc": started,
                "wall_ms": (time.perf_counter() - start) * 1000,
                "stdout": exc.stdout or "", "stderr": exc.stderr or ""}
    result = {"started_utc": started, "returncode": completed.returncode,
              "wall_ms": (time.perf_counter() - start) * 1000,
              "stdout": completed.stdout, "stderr": completed.stderr}
    if completed.returncode != 0:
        result["status"] = "failed"
        return result
    try:
        result.update(parse_csv(raw))
        result["status"] = "ok"
    except (OSError, ValueError) as exc:
        result.update(status="invalid_output", parse_error=repr(exc))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--executable", type=Path, required=True)
    ap.add_argument("--matrix-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args()
    exe = args.executable.resolve()
    root = args.matrix_root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "raw").mkdir()
    (out / "logs").mkdir()
    manifest = json.loads((Path(__file__).parent / "protocol.json").read_text(encoding="utf-8"))
    manifest.update({"started_utc": datetime.now(timezone.utc).isoformat(),
                     "executable_sha256": sha256(exe),
                     "protocol_sha256": sha256(Path(__file__).parent / "protocol.json"),
                     "inputs": {}})
    rows = []
    for name in MATRICES:
        try:
            matrix = matrix_path(root, name)
            manifest["inputs"][name] = {"path": str(matrix), "sha256": sha256(matrix),
                                        "bytes": matrix.stat().st_size}
        except FileNotFoundError:
            rows.append({"matrix": name, "status": "missing_input"})
            continue
        for seed in SEEDS:
            for width in WIDTHS:
                for power in POWERS:
                    case_id = f"{name}_seed{seed}_l{width}_q{power}"
                    raw = out / "raw" / f"{case_id}.csv"
                    result = run_case(exe, matrix, raw, 16, width, power, seed, args.timeout)
                    row = {"matrix": name, "seed": seed, "rank": 16, "width": width,
                           "power": power, "input": str(matrix), "raw_csv": str(raw), **result}
                    rows.append(row)
                    (out / "logs" / f"{case_id}.log").write_text(
                        "COMMAND\n" + " ".join(str(x) for x in [str(exe), str(matrix), str(raw), 16, width - 16, power, seed])
                        + "\nSTDOUT\n" + str(result.get("stdout", ""))
                        + "\nSTDERR\n" + str(result.get("stderr", "")) + "\n", encoding="utf-8")
                    print(name, seed, width, power, result.get("status"), flush=True)
    fields = sorted({key for row in rows for key in row})
    with (out / "heldout_rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["state_count"] = len(rows)
    manifest["status_counts"] = {status: sum(r.get("status") == status for r in rows)
                                  for status in sorted({r.get("status") for r in rows})}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["status_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
