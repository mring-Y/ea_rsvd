#!/usr/bin/env python3
"""Summarize non-LP structure-sensitivity CSV files without rerunning CUDA."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    # Accept both the flat local sweep layout and a nested AutoDL result
    # directory.  Result files are still selected by their explicit stem so
    # unrelated CSVs cannot enter the aggregation.
    for path in sorted(root.rglob("*_seed*_q*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                row["source_csv"] = path.name
                rows.append(row)
    return rows


def value(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def mean_sd(values: list[float]) -> tuple[float, float]:
    clean = [x for x in values if math.isfinite(x)]
    if not clean:
        return math.nan, math.nan
    return statistics.mean(clean), statistics.stdev(clean) if len(clean) > 1 else 0.0


def residual_fields(row: dict[str, str]) -> tuple[str, str, str]:
    """Return projection field, rank field, and the declared residual scope.

    The resident benchmark writes probe-inclusive estimates, while the
    exact-no-probe executable writes exact residuals and probe_count=0.  The
    scope is derived from the row rather than the executable name so archived
    CSV files remain self-describing.
    """
    try:
        probe_count = int(float(row.get("probe_count", "0")))
    except (TypeError, ValueError):
        probe_count = 0
    if probe_count == 0 and "exact_projection_residual" in row and "exact_rank_residual" in row:
        return "exact_projection_residual", "exact_rank_residual", "exact-no-probe"
    if "projection_residual_estimate" in row and "rank_k_reconstruction_residual" in row:
        return "projection_residual_estimate", "rank_k_reconstruction_residual", "probe-inclusive"
    raise KeyError("CSV has neither exact-no-probe nor probe-inclusive residual fields")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.input_dir.resolve())
    if not rows:
        raise RuntimeError("no structure-sweep CSV files found")
    grouped: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        _, _, scope = residual_fields(row)
        grouped[(Path(row["input"]).stem, row["power_steps"], scope)].append(row)

    summary: list[dict[str, object]] = []
    for (matrix, power_steps, scope), group in sorted(grouped.items()):
        projection_key, rank_key, _ = residual_fields(group[0])
        time_mean, time_sd = mean_sd([value(r, "gpu_pipeline_ms") for r in group])
        rho_l_mean, rho_l_sd = mean_sd([value(r, projection_key) for r in group])
        rho_k_mean, rho_k_sd = mean_sd([value(r, rank_key) for r in group])
        q_values = [value(r, "q_orthogonality_max_error") for r in group]
        q_max = max((x for x in q_values if math.isfinite(x)), default=math.nan)
        summary.append({
            "matrix": matrix,
            "power_steps": int(power_steps),
            "residual_scope": scope,
            "rows": int(group[0]["rows"]),
            "cols": int(group[0]["cols"]),
            "nnz": int(group[0]["nnz"]),
            "seeds": len(group),
            "mean_gpu_pipeline_ms": f"{time_mean:.9g}",
            "sd_gpu_pipeline_ms": f"{time_sd:.9g}",
            "mean_projection_residual": f"{rho_l_mean:.9g}",
            "sd_projection_residual": f"{rho_l_sd:.9g}",
            "mean_rank_k_residual": f"{rho_k_mean:.9g}",
            "sd_rank_k_residual": f"{rho_k_sd:.9g}",
            "max_q_orthogonality": f"{q_max:.9g}",
            "status": "PASS" if math.isfinite(q_max) and q_max <= 1e-4 else "WARN",
        })

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(summary[0])
    with (output_dir / "nonlp_structure_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# Non-LP structure sensitivity",
        "",
        "The summary aggregates the stored CUDA path at k=16 and ell=24 over the available seeds.",
        "Each row declares whether its residuals are probe-inclusive estimates or exact no-probe values.",
        "Projection and rank-k residuals are reported separately; q is the only changed algorithmic setting.",
        "Rows with Q orthogonality above 1e-4 are retained with WARN status and should not be pooled into a PASS mean.",
        "",
    ]
    for row in summary:
        lines.append(
            f"- {row['matrix']}, q={row['power_steps']}: "
            f"scope={row['residual_scope']}, "
            f"rho_l={row['mean_projection_residual']} +/- {row['sd_projection_residual']}, "
            f"rho_k={row['mean_rank_k_residual']} +/- {row['sd_rank_k_residual']}, "
            f"GPU={row['mean_gpu_pipeline_ms']} ms, seeds={row['seeds']}, status={row['status']}"
        )
    (output_dir / "nonlp_structure_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {output_dir / 'nonlp_structure_summary.csv'} ({len(summary)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
