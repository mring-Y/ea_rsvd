"""Summarize a fixed cohort of measured paths at successive budget caps.

An ended path retains its last measured result and cost. No new state or time
is imputed. The original available-case summary can be recovered separately.
"""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import statistics


def read(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def summarize(rows):
    paths = defaultdict(list)
    for row in rows:
        paths[(row["matrix"], str(row["seed"]))].append(row)
    if not paths:
        raise ValueError("No observed paths")
    for key, path in paths.items():
        path.sort(key=lambda row: int(row["budget_step"]))
        if [int(row["budget_step"]) for row in path] != list(range(1, len(path) + 1)):
            raise ValueError(f"Non-contiguous or duplicate path steps: {key}")
    output = []
    for cap in range(1, max(map(len, paths.values())) + 1):
        chosen = [path[min(cap, len(path)) - 1] for path in paths.values()]
        output.append(dict(
            budget_step=cap, paths=len(chosen),
            advancing_paths=sum(len(path) >= cap for path in paths.values()),
            mean_endpoint_rho_k=statistics.mean(float(r["endpoint_rho_k"]) for r in chosen),
            mean_best_seen_rho_k=statistics.mean(float(r["best_seen_rho_k"]) for r in chosen),
            mean_cumulative_gpu_ms=statistics.mean(float(r["cumulative_gpu_ms"]) for r in chosen)))
    return output


def write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read(args.input)
    result = summarize(rows)
    write(args.output, result)
    for row in result:
        print(row)


if __name__ == "__main__":
    main()
