#!/usr/bin/env python3
"""Summarize stored controller-ablation and large-scale rank scans.

The script is post-processing only. It never fabricates missing measurements:
rows are retained with their source scope, seed count, and status fields.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def controller_summary(root: Path):
    source = root / "review_experiments/stage56_controller_ablation_5seed/summary.csv"
    rows = read_csv(source)
    fields = [
        "policy", "matrix", "seeds", "mean_states", "mean_cumulative_gpu_ms",
        "mean_final_probe_residual", "mean_final_pve", "terminal_rules",
    ]
    out = [{key: row.get(key, "") for key in fields} for row in rows]
    write_csv(root / "paper_assets/controller_ablation_summary.csv", out, fields)

    policy_order = ["pve_only", "residual_only", "dual_monitor"]
    matrix_order = list(dict.fromkeys(row["matrix"] for row in rows))
    plot_rows = []
    for matrix in matrix_order:
        for policy in policy_order:
            row = next((r for r in rows if r["matrix"] == matrix and r["policy"] == policy), None)
            if row:
                plot_rows.append({
                    "matrix": matrix, "policy": policy,
                    "mean_states": row["mean_states"],
                    "mean_cumulative_gpu_ms": row["mean_cumulative_gpu_ms"],
                    "mean_final_probe_residual": row["mean_final_probe_residual"],
                    "mean_final_pve": row["mean_final_pve"], "seeds": row["seeds"],
                })
    write_csv(root / "paper_assets/controller_ablation_plot.csv", plot_rows,
              ["matrix", "policy", "mean_states", "mean_cumulative_gpu_ms",
               "mean_final_probe_residual", "mean_final_pve", "seeds"])
    return rows


def scale_summary(root: Path):
    sources = [
        (root / "review_experiments/stage55_amazon_rank/stage55_amazon_rank_sweep.csv", "amazon0312", "single-seed"),
        (root / "review_experiments/stage57_webbase_parameter_sweep/stage57_webbase_parameter_sweep.csv", "webbase-1M", "two-seed subset"),
    ]
    out = []
    for path, family, scope in sources:
        for row in read_csv(path):
            status = row.get("status", "")
            out.append({
                "matrix": row.get("matrix", family), "family": family,
                "rows": row.get("rows", ""), "cols": row.get("cols", ""),
                "nnz": row.get("nnz", ""), "target_rank": row.get("target_rank", row.get("rank", "")),
                "sketch_width": row.get("sketch_width", ""),
                "power_steps": row.get("power_steps", ""), "seed": row.get("seed", ""),
                "gpu_pipeline_ms": row.get("gpu_pipeline_ms", ""),
                "projection_residual": row.get("projection_residual_estimate", row.get("projection_residual", "")),
                "rank_k_residual": row.get("rank_k_reconstruction_residual", row.get("rank_k_residual", "")),
                "q_orthogonality": row.get("q_orthogonality_max_error", ""),
                "status": status, "scope": scope,
            })
    fields = ["matrix", "family", "rows", "cols", "nnz", "target_rank", "sketch_width",
              "power_steps", "seed", "gpu_pipeline_ms", "projection_residual",
              "rank_k_residual", "q_orthogonality", "status", "scope"]
    write_csv(root / "paper_assets/extended_scale_rows.csv", out, fields)
    grouped = []
    for (family, rank, q), group in _groups(out, "family", "target_rank", "power_steps"):
        valid = [r for r in group
                 if (not r["status"] or r["status"].startswith("PASS"))
                 and math.isfinite(f(r, "q_orthogonality"))
                 and f(r, "q_orthogonality") <= 1e-4
                 and math.isfinite(f(r, "rank_k_residual"))
                 and math.isfinite(f(r, "gpu_pipeline_ms"))]
        vals = [f(r, "rank_k_residual") for r in valid]
        times = [f(r, "gpu_pipeline_ms") for r in valid]
        grouped.append({
            "family": family, "target_rank": rank, "power_steps": q,
            "n_rows": str(len(group)), "n_valid": str(len(valid)),
            "mean_rank_k_residual": f"{sum(vals) / len(vals):.9g}" if vals else "",
            "mean_gpu_pipeline_ms": f"{sum(times) / len(times):.9g}" if times else "",
            "scope": group[0]["scope"] if group else "",
        })
    write_csv(root / "paper_assets/extended_scale_summary.csv", grouped,
              ["family", "target_rank", "power_steps", "n_rows", "n_valid",
               "mean_rank_k_residual", "mean_gpu_pipeline_ms", "scope"])
    return out


def _groups(rows, *keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    for key, rows_for_key in grouped.items():
        yield key, rows_for_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.work_root.resolve()
    controller = controller_summary(root)
    scale = scale_summary(root)
    print(f"wrote controller rows: {len(controller)}")
    print(f"wrote scale rows: {len(scale)}")


if __name__ == "__main__":
    main()
