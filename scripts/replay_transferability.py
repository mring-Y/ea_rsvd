#!/usr/bin/env python3
"""Leave-one-matrix-out replay for fixed finite-grid settings.

For each held-out matrix, choose the (ell, q) pair with the lowest mean exact
rank residual on the other matrices, then evaluate that setting on the held
out matrix. This is deterministic post-processing of stored GPU states; it
does not launch CUDA and it does not claim a new benchmark run.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else math.nan


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def replay(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = [row for row in rows if row["rank"] == "16"]
    if not rows:
        raise RuntimeError("rank-16 rows are required for transfer replay")
    matrices = sorted({row["matrix"] for row in rows})
    configs = sorted({(row["width"], row["power"]) for row in rows}, key=lambda item: (int(item[0]), int(item[1])))
    by_matrix_config: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        by_matrix_config.setdefault((row["matrix"], row["width"], row["power"]), []).append(row)

    output: list[dict[str, object]] = []
    for target in matrices:
        train_scores = {
            config: mean([f(row, "exact_rank_residual") for matrix in matrices if matrix != target for row in by_matrix_config[(matrix, *config)]])
            for config in configs
        }
        transfer_config = min(configs, key=lambda config: train_scores[config])
        target_scores = {
            config: mean([f(row, "exact_rank_residual") for row in by_matrix_config[(target, *config)]])
            for config in configs
        }
        target_times = {
            config: mean([f(row, "gpu_pipeline_ms") for row in by_matrix_config[(target, *config)]])
            for config in configs
        }
        oracle_config = min(configs, key=lambda config: target_scores[config])
        transfer_rho = target_scores[transfer_config]
        oracle_rho = target_scores[oracle_config]
        oracle_time = target_times[oracle_config]
        output.append({
            "matrix": target,
            "training_matrices": len(matrices) - 1,
            "transfer_width": transfer_config[0],
            "transfer_power": transfer_config[1],
            "transfer_rho_k": transfer_rho,
            "transfer_gpu_ms": target_times[transfer_config],
            "oracle_width": oracle_config[0],
            "oracle_power": oracle_config[1],
            "oracle_rho_k": oracle_rho,
            "oracle_gpu_ms": oracle_time,
            "transfer_excess_rho_k": transfer_rho - oracle_rho,
            "transfer_quality_ratio": transfer_rho / oracle_rho if oracle_rho > 0 else math.nan,
            "transfer_cost_ratio": target_times[transfer_config] / oracle_time if oracle_time > 0 else math.nan,
            "training_mean_rho_k": train_scores[transfer_config],
        })
    return output


def write(rows: list[dict[str, object]], output: Path) -> None:
    fields = list(rows[0])
    with (output / "transferability_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "matrices": len(rows),
        "mean_transfer_rho_k": mean([float(row["transfer_rho_k"]) for row in rows]),
        "mean_oracle_rho_k": mean([float(row["oracle_rho_k"]) for row in rows]),
        "mean_excess_rho_k": mean([float(row["transfer_excess_rho_k"]) for row in rows]),
        "median_quality_ratio": statistics.median([float(row["transfer_quality_ratio"]) for row in rows]),
        "max_quality_ratio": max(float(row["transfer_quality_ratio"]) for row in rows),
        "same_config_fraction": mean([1.0 if (row["transfer_width"], row["transfer_power"]) == (row["oracle_width"], row["oracle_power"]) else 0.0 for row in rows]),
    }
    with (output / "transferability_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    lines = [
        "# Leave-one-matrix-out finite-grid replay",
        "",
        "For each held-out matrix, the setting with the lowest mean exact rho_k on the other matrices is applied to the held-out matrix.",
        "This is deterministic post-processing of stored GPU states and does not launch CUDA.",
        "",
        f"Mean transferred rho_k: {summary['mean_transfer_rho_k']:.6f}",
        f"Mean held-out oracle rho_k: {summary['mean_oracle_rho_k']:.6f}",
        f"Median transferred/oracle quality ratio: {summary['median_quality_ratio']:.3f}x",
        f"Maximum transferred/oracle quality ratio: {summary['max_quality_ratio']:.3f}x",
        f"Transfer and held-out oracle use the same setting in {summary['same_config_fraction']:.0%} of cases.",
    ]
    (output / "transferability_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = replay(read_csv(args.input.resolve()))
    if not rows:
        raise RuntimeError("no fixed-grid rows found")
    write(rows, args.output.resolve())
    print(f"wrote {len(rows)} held-out transfer rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
