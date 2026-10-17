#!/usr/bin/env python3
"""Compare independent full-run candidates from one measured starting state.

The default is (k=16, ell=40, q=0), with candidates (16,40,2) and
(32,40,0). Both residual gains subtract the same baseline. Candidate time
is the complete stored GPU interval, not the marginal cost of continuing a
cached factorization. This is post-processing of actual CUDA records.

Grouping a fixed diagnostic by matrix or seed is descriptive agreement;
it is not leave-one-out training or independent predictive validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_rows(grid: list[dict[str, str]], all_shared: bool = False) -> list[dict]:
    keys = [(r["matrix"], r["seed"], int(r["rank"]), int(r["width"]),
             int(r["power"])) for r in grid]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate state keys would make the comparison ambiguous")
    states = dict(zip(keys, grid))
    # Each width admits both output ranks without changing the sketch width.
    configurations = [(16, 32, 40, 0, 2)]
    if all_shared:
        configurations = [(k, k2, width, q, 2)
                          for k, k2, width in [(8, 16, 24), (8, 16, 32), (16, 32, 40)]
                          for q in [0, 1]]
    output = []
    for matrix, seed in sorted({(r["matrix"], r["seed"]) for r in grid}):
        for k, k2, width, q, q2 in configurations:
            base = states.get((matrix, seed, k, width, q))
            power = states.get((matrix, seed, k, width, q2))
            rank = states.get((matrix, seed, k2, width, q))
            if any(r is None for r in [base, power, rank]):
                continue
            rho = float(base["exact_rank_residual"])
            proj = float(base["exact_projection_residual"])
            p_rho = float(power["exact_rank_residual"])
            r_rho = float(rank["exact_rank_residual"])
            p_ms = float(power["gpu_pipeline_ms"])
            r_ms = float(rank["gpu_pipeline_ms"])
            if not all(math.isfinite(x) for x in [rho, proj, p_rho, r_rho, p_ms, r_ms]):
                raise ValueError("nonfinite residual or timing value")
            if min(p_ms, r_ms) <= 0:
                raise ValueError("candidate times must be positive")
            capture_energy = proj ** 2
            truncation_energy = max(0.0, rho ** 2 - capture_energy)
            p_gain, r_gain = rho - p_rho, rho - r_rho
            p_score, r_score = p_gain / p_ms, r_gain / r_ms
            region = "adequate" if rho <= 0.20 else (
                "rank-limited" if proj <= 0.20 else "capture-unresolved")
            threshold_direction = {"adequate": "stop", "rank-limited": "target-rank",
                                   "capture-unresolved": "power"}[region]
            # A transparent descriptive screen, evaluated without fitting.
            dominant_component = "truncation" if truncation_energy > capture_energy else "capture"
            dominant_direction = "target-rank" if dominant_component == "truncation" else "power"
            winner = "target-rank" if r_score > p_score else "power"
            output.append({
                "matrix": matrix, "seed": seed, "base_rank": k,
                "rank_candidate_rank": k2, "width": width, "base_power": q,
                "power_candidate_power": q2,
                "baseline_rho_l": proj, "baseline_rho_k": rho,
                "capture_energy_fraction": capture_energy,
                "truncation_energy_fraction": truncation_energy,
                "truncation_share_of_residual_energy": truncation_energy / rho ** 2 if rho else 0.0,
                "power_candidate_rho_k": p_rho, "rank_candidate_rho_k": r_rho,
                "power_rho_k_gain": p_gain, "rank_rho_k_gain": r_gain,
                "power_candidate_ms": p_ms, "rank_candidate_ms": r_ms,
                "power_gain_per_candidate_ms": p_score, "rank_gain_per_candidate_ms": r_score,
                "max_q_error_triplet": max(float(r["q_orthogonality_max_error"]) for r in [base, power, rank]),
                "rank_abs_projection_change": abs(float(rank["exact_projection_residual"]) - proj),
                "threshold_region": region, "threshold_direction": threshold_direction,
                "dominant_component": dominant_component,
                "candidate_score_winner": winner,
                "threshold_direction_agrees": threshold_direction == winner,
                "dominant_component_agrees": dominant_direction == winner,
            })
    return output


def summarize(rows: list[dict]) -> list[dict]:
    result = []
    groups = sorted({(r["matrix"], r["base_rank"], r["width"], r["base_power"]) for r in rows})
    for matrix, k, width, q in groups:
        subset = [r for r in rows if (r["matrix"], r["base_rank"], r["width"], r["base_power"]) == (matrix, k, width, q)]
        row = {"matrix": matrix, "base_rank": k, "width": width, "base_power": q,
               "rank_candidate_rank": subset[0]["rank_candidate_rank"], "seeds": len(subset)}
        for field in ["baseline_rho_l", "baseline_rho_k", "power_rho_k_gain", "rank_rho_k_gain",
                      "power_candidate_ms", "rank_candidate_ms", "power_gain_per_candidate_ms",
                      "rank_gain_per_candidate_ms", "truncation_share_of_residual_energy",
                      "threshold_direction_agrees", "dominant_component_agrees"]:
            row["mean_" + field] = statistics.mean(float(r[field]) for r in subset)
        for field in ["power_rho_k_gain", "rank_rho_k_gain"]:
            row["ci95_" + field] = (2.776445105 * statistics.stdev(float(r[field]) for r in subset)
                                    / math.sqrt(len(subset))) if len(subset) == 5 else math.nan
        row["max_rank_abs_projection_change"] = max(r["rank_abs_projection_change"] for r in subset)
        row["dominant_score_winner"] = "target-rank" if sum(r["candidate_score_winner"] == "target-rank" for r in subset) > len(subset)/2 else "power"
        result.append(row)
    return result


def grouped_agreement(rows: list[dict]) -> list[dict]:
    output = []
    for group in ["matrix", "seed"]:
        for label in sorted({r[group] for r in rows}):
            subset = [r for r in rows if r[group] == label]
            output.append({"group": group, "label": label, "comparisons": len(subset),
                           "threshold_agreement": statistics.mean(float(r["threshold_direction_agrees"]) for r in subset),
                           "component_agreement": statistics.mean(float(r["dominant_component_agrees"]) for r in subset)})
    return output


def draw(summary: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 12.5, "axes.labelsize": 13})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)
    x = np.arange(len(summary))
    colors = ["#2B6CB0", "#C23B22"]
    for i, (label, key) in enumerate([("Power 0 to 2", "power"), ("Rank 16 to 32", "rank")]):
        offset = (i - 0.5) * .35
        gains = [r[f"mean_{key}_rho_k_gain"] for r in summary]
        errs = [r[f"ci95_{key}_rho_k_gain"] for r in summary]
        axes[0].bar(x + offset, gains, width=.35, yerr=errs, capsize=3, color=colors[i], label=label)
        axes[1].bar(x + offset, [r[f"mean_{key}_candidate_ms"] for r in summary],
                    width=.35, color=colors[i], label=label)
    axes[0].set_title("(a) Residual gain from the same starting state")
    axes[0].set_ylabel(r"Reduction in $\rho_k$")
    axes[1].set_title("(b) Diagnostic GPU interval")
    axes[1].set_ylabel("GPU pipeline time (ms)")
    for ax in axes:
        ax.set_xticks(x, [r["matrix"] for r in summary], rotation=25, ha="right")
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", alpha=.20)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=11)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-shared", action="store_true",
                        help="include ranks 8/16 and shared widths 24/32/40 at q=0 and q=1")
    parser.add_argument("--no-figure", action="store_true", help="write numerical outputs without importing Matplotlib")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = build_rows(read_csv(args.input), all_shared=args.all_shared)
    if not rows:
        raise SystemExit("no complete common-start triplets found")
    summary = summarize(rows)
    write_csv(args.output / "action_efficiency.csv", rows)
    write_csv(args.output / "action_efficiency_summary.csv", summary)
    write_csv(args.output / "action_agreement_summary.csv", grouped_agreement(rows))
    manifest = {"analysis": "common-start independent full-run candidate comparison",
                "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
                "comparisons": len(rows), "all_shared": args.all_shared,
                "threshold_agreement": statistics.mean(float(r["threshold_direction_agrees"]) for r in rows),
                "component_agreement": statistics.mean(float(r["dominant_component_agrees"]) for r in rows),
                "cost_scope": "stored complete candidate GPU interval; excludes input transfer",
                "validation_scope": "descriptive stratification; no fitted or held-out classifier"}
    (args.output / "action_efficiency_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output / "action_efficiency_summary.md").write_text(
        "# Common-start candidate comparison\n\n"
        "Both gains subtract the same baseline. Rank candidates preserve width and power; "
        "power candidates preserve rank and width. Costs are independent complete runs, "
        "not cached continuation times. Agreement is descriptive; no training or leave-one-out fitting occurs.\n\n"
        + json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if not args.all_shared and not args.no_figure:
        draw(summary, args.output / "fig_common_start_actions.png")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
