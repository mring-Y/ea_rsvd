#!/usr/bin/env python3
"""Replay a residual-pair controller path from exact stored states.

The path increments q at the current width, widens at q=2, and tests k=32
when rho_l is adequate but rho_k is not. This is post-processing only.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else math.nan


def classify(row: dict[str, str], tau_l: float, tau_k: float) -> str:
    rho_l = value(row, "exact_projection_residual")
    rho_k = value(row, "exact_rank_residual")
    if rho_k <= tau_k:
        return "stop"
    if rho_l > tau_l:
        return "sketch"
    return "target-rank"


def replay(rows: list[dict[str, str]], tau_l: float, tau_k: float) -> list[dict[str, object]]:
    states = {
        (row["matrix"], row["seed"], row["rank"], row["width"], row["power"]): row
        for row in rows
    }
    output: list[dict[str, object]] = []
    for matrix in sorted({row["matrix"] for row in rows}):
        for seed in sorted({row["seed"] for row in rows if row["matrix"] == matrix}):
            baseline = states.get((matrix, seed, "16", "24", "0"))
            if not baseline:
                continue
            selected_rows = [baseline]
            actions = ["k16,l24,q0"]
            current = baseline
            terminal = "budget-limited"
            for width, q_values in ((24, (1, 2)), (32, (2,)), (40, (2,))):
                for q in q_values:
                    candidate = states.get((matrix, seed, "16", str(width), str(q)))
                    if candidate is None:
                        continue
                    selected_rows.append(candidate)
                    actions.append(f"k16,l{width},q{q}")
                    current = candidate
                decision = classify(current, tau_l, tau_k)
                if decision == "stop":
                    terminal = "quality-target"
                    break
                if decision == "target-rank":
                    rank_candidate = states.get((matrix, seed, "32", str(width), str(current["power"])))
                    if rank_candidate is not None:
                        selected_rows.append(rank_candidate)
                        actions.append(f"k32,l{width},q{current['power']}")
                        current = rank_candidate
                        terminal = "rank-tested"
                        break
                    else:
                        terminal = "rank-limited; continue width search"
            oracle = min(selected_rows, key=lambda row: value(row, "exact_rank_residual"))
            final_rho = value(current, "exact_rank_residual")
            oracle_rho = value(oracle, "exact_rank_residual")
            cumulative_ms = sum(value(row, "gpu_pipeline_ms") for row in selected_rows)
            oracle_ms = value(oracle, "gpu_pipeline_ms")
            output.append({
                "matrix": matrix,
                "seed": seed,
                "path": ">".join(actions),
                "states_evaluated": len(actions),
                "cumulative_gpu_ms": cumulative_ms,
                "final_rho_l": value(current, "exact_projection_residual"),
                "final_rho_k": final_rho,
                "best_path_state_rho_k": oracle_rho,
                "best_path_state_gpu_ms": oracle_ms,
                "path_endpoint_gap": final_rho - oracle_rho,
                "cost_ratio_to_best_path_state": cumulative_ms / oracle_ms if oracle_ms > 0 else math.nan,
                "baseline_rho_k": value(baseline, "exact_rank_residual"),
                "baseline_gpu_ms": value(baseline, "gpu_pipeline_ms"),
                "best_path_setting": f"k={oracle['rank']},ell={oracle['width']},q={oracle['power']}",
                "terminal": terminal,
            })
    return output


def write_outputs(rows: list[dict[str, object]], output: Path) -> None:
    fields = list(rows[0])
    with (output / "controller_path_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summaries: list[dict[str, object]] = []
    for matrix in sorted({str(row["matrix"]) for row in rows}):
        subset = [row for row in rows if row["matrix"] == matrix]
        paths = Counter(str(row["path"]) for row in subset)
        terminals = Counter(str(row["terminal"]) for row in subset)
        summaries.append({
            "matrix": matrix,
            "seeds": len(subset),
            "mean_states": mean([float(row["states_evaluated"]) for row in subset]),
            "mean_cumulative_gpu_ms": mean([float(row["cumulative_gpu_ms"]) for row in subset]),
            "mean_final_rho_k": mean([float(row["final_rho_k"]) for row in subset]),
            "mean_best_path_state_rho_k": mean([float(row["best_path_state_rho_k"]) for row in subset]),
            "mean_path_endpoint_gap": mean([float(row["path_endpoint_gap"]) for row in subset]),
            "mean_cost_ratio_to_best_path_state": mean([float(row["cost_ratio_to_best_path_state"]) for row in subset]),
            "mean_oracle_rho_k": mean([float(row["best_path_state_rho_k"]) for row in subset]),
            "mean_oracle_gap": mean([float(row["path_endpoint_gap"]) for row in subset]),
            "mean_cost_ratio_to_oracle": mean([float(row["cost_ratio_to_best_path_state"]) for row in subset]),
            "path_counts": ";".join(f"{key}:{paths[key]}" for key in sorted(paths)),
            "terminal_counts": ";".join(f"{key}:{terminals[key]}" for key in sorted(terminals)),
        })
    with (output / "controller_path_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    (output / "controller_path_summary.md").write_text(
        "# Residual-pair controller path replay\n\n"
        "This deterministic replay increments q, widens ell at q=2, and tests a larger target rank after rho_l meets threshold. It reports cumulative measured state cost and the best state encountered along the selected path. It does not rerun CUDA.\n\n" +
        "\n".join(
            f"- {row['matrix']}: mean final rho_k={float(row['mean_final_rho_k']):.4f}, "
            f"endpoint gap={float(row['mean_path_endpoint_gap']):.4g}, "
            f"cost ratio={float(row['mean_cost_ratio_to_best_path_state']):.2f}x, "
            f"terminal={row['terminal_counts']}"
            for row in summaries
        ) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tau-l", type=float, default=0.20)
    parser.add_argument("--tau-k", type=float, default=0.20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = replay(read_csv(args.input.resolve()), args.tau_l, args.tau_k)
    if not rows:
        raise RuntimeError("no complete controller triplets found")
    write_outputs(rows, args.output.resolve())
    print(f"wrote {len(rows)} controller path rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
