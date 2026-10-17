"""Describe matched results, feasibility, and the measured calibration cost.

This report includes solver failures and unattained quality targets.  Its
finite-grid minima do not estimate a free online solver-selection policy.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from campaign import make_near_best_comparisons, valid, write_csv


def geometric_mean(values):
    return math.exp(sum(math.log(v) for v in values)/len(values)) if values else None


def joint_quality_comparisons(rows, triplet_targets=(1e-2, 1e-3, 1e-4)):
    """Require both actual reconstruction and common triplet residual quality."""
    keys = ["matrix", "csr_sha256", "precision", "rank", "seed", "gpu"]
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in keys)].append(row)
    result = []
    for reference in make_near_best_comparisons(rows, (1.01,)):
        entries = groups[tuple(reference.get(k, "") for k in keys)]
        for threshold in triplet_targets:
            item = {k: reference.get(k, "") for k in keys}
            item.update(target_residual=reference["target_residual"],
                        reference_residual=reference["reference_residual"],
                        target_triplet_residual_relative_fro=threshold)
            for short, solver in (("rsvd", "rsvd_diagnostic_full"), ("primme", "primme_svds_cuda")):
                usable = [r for r in entries if r["solver"] == solver and valid(r)]
                eligible = [r for r in usable if r["relative_residual"] <= item["target_residual"] and
                            r.get("matrix_frobenius", 0.) > 0 and
                            r.get("triplet_residual_max", math.inf)/r["matrix_frobenius"] <= threshold]
                best = min(eligible, key=lambda r: r["compute_ms"]) if eligible else None
                item[short+"_status"] = ("met_joint_target" if best else
                                          "unattained_joint_target" if usable else "no_valid_result")
                item[short+"_ms"] = best["compute_ms"] if best else None
                item[short+"_residual"] = best["relative_residual"] if best else None
                item[short+"_triplet_relative_fro"] = (best["triplet_residual_max"]/best["matrix_frobenius"]
                                                      if best else None)
                item[short+"_configuration"] = best.get("configuration", "") if best else None
            item["speedup_rsvd_over_primme"] = (item["primme_ms"]/item["rsvd_ms"]
                if item["primme_ms"] is not None and item["rsvd_ms"] is not None else None)
            result.append(item)
    return result


def analyse(directory: Path):
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines() if line.strip()]
    environment = json.loads((directory / "environment.json").read_text())
    groups = defaultdict(list)
    keys = ["matrix", "csr_sha256", "precision", "rank", "seed", "gpu", "solver"]
    for row in rows:
        groups[tuple(row.get(k, "") for k in keys)].append(row)
    costs = []
    for key, entries in sorted(groups.items()):
        process_ms = sum(r.get("process_wall_ms", 0.) for r in entries)
        validation_ms = sum(r.get("validation_ms", 0.) for r in entries)
        costs.append({**dict(zip(keys, key)), "configurations_attempted": len(entries),
                      "valid_configurations": sum(valid(r) for r in entries),
                      "failed_configurations": sum(not valid(r) for r in entries),
                      "candidate_process_wall_ms": process_ms,
                      "independent_validation_ms": validation_ms,
                      "total_calibration_wall_ms": process_ms+validation_ms})
    write_csv(directory / "calibration_costs.csv", costs)
    near = make_near_best_comparisons(rows)
    joint = joint_quality_comparisons(rows)
    write_csv(directory / "joint_quality_comparisons.csv", joint)
    quality = {}
    total_state_groups = len({key[:-1] for key in groups})
    for allowance in (1.01, 1.05):
        entries = [r for r in near if r["relative_quality_allowance"] == allowance]
        ratios = [r["speedup_rsvd_over_primme"] for r in entries if r["speedup_rsvd_over_primme"] is not None]
        quality[str(allowance)] = {"reference_groups": len(entries), "paired_groups": len(ratios),
                                  "total_input_rank_seed_groups": total_state_groups,
                                  "rsvd_faster_groups": sum(v > 1 for v in ratios),
                                  "primme_faster_groups": sum(v < 1 for v in ratios),
                                  "geometric_mean_primme_over_rsvd": geometric_mean(ratios),
                                  "minimum_ratio": min(ratios, default=None),
                                  "maximum_ratio": max(ratios, default=None)}
    status = Counter(r["status"] for r in rows)
    joint_summary = {}
    for threshold in (1e-2, 1e-3, 1e-4):
        entries = [r for r in joint if r["target_triplet_residual_relative_fro"] == threshold]
        ratios = [r["speedup_rsvd_over_primme"] for r in entries if r["speedup_rsvd_over_primme"] is not None]
        joint_summary[str(threshold)] = {"paired_groups": len(ratios), "reference_groups": len(entries),
                                        "total_input_rank_seed_groups": total_state_groups,
                                        "geometric_mean_primme_over_rsvd": geometric_mean(ratios),
                                        "rsvd_faster_groups": sum(r > 1 for r in ratios),
                                        "primme_faster_groups": sum(r < 1 for r in ratios)}
    report = {"status_counts": dict(status), "configuration_count": len(rows),
              "near_reference_quality": quality,
              "joint_reconstruction_triplet_quality": joint_summary,
              "run_completion": json.loads((directory / "completion.json").read_text())
                  if (directory / "completion.json").exists() else {"completed": False},
              "gpu_environment": environment.get("nvidia_smi", ""),
              "timing_scope": "resident CSR synchronized wall time; per-solve allocation, RNG, full factors, diagnostics",
              "interpretation": "descriptive finite-grid minima on this device; no held-out online-policy or global-optimality claim"}
    (directory / "analysis.json").write_text(json.dumps(report, indent=2)+"\n")
    matrix_groups = defaultdict(list)
    for row in rows:
        matrix_groups[(row["matrix"], row["precision"], row["rank"])].append(row)
    lines = ["# Published GPU solver comparison: observed results", "",
             "GPU environment: `"+environment.get("nvidia_smi", "").strip().replace("\n", "; ")+"`", "",
             "Status counts: `"+json.dumps(dict(status), sort_keys=True)+"`", "",
             "Each speed ratio compares the same CSR input, rank, precision, seed and GPU under an actual "
             "FP64-validated reconstruction threshold. Results below are descriptive finite-grid minima. "
             "All timed repeats must satisfy the quality and orthogonality checks.", "",
             "| Matrix | Precision | Rank | Valid rSVD / attempted | Valid PRIMME / attempted | 1% paired / reference / total groups | PRIMME / rSVD time, geometric mean |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for key, entries in sorted(matrix_groups.items()):
        matching = [r for r in near if (r["matrix"], r["precision"], r["rank"]) == key and r["relative_quality_allowance"] == 1.01]
        ratios = [r["speedup_rsvd_over_primme"] for r in matching if r["speedup_rsvd_over_primme"] is not None]
        value = geometric_mean(ratios)
        rs = [r for r in entries if r["solver"] == "rsvd_diagnostic_full"]
        ps = [r for r in entries if r["solver"] == "primme_svds_cuda"]
        state_count = len({(r.get("csr_sha256", ""), r["seed"], r.get("gpu", "")) for r in entries})
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {sum(valid(r) for r in rs)} / {len(rs)} | "
                     f"{sum(valid(r) for r in ps)} / {len(ps)} | {len(ratios)} / {len(matching)} / {state_count} | "
                     +(f"{value:.4f}" if value is not None else "unavailable")+" |")
    lines.extend(["", "A ratio above one means the rSVD path has the smaller measured median time. "
                  "The reference is the smallest validated PRIMME reconstruction error among eps <= 1e-5 "
                  "for each case; it is a numerical reference and is not a certified optimal SVD error.", "",
                  "`joint_quality_comparisons.csv` adds common bounds on the maximum singular-triplet "
                  "residual divided by the input Frobenius norm (1e-2, 1e-3, 1e-4), while retaining the 1% "
                  "reconstruction allowance. It uses the worst triplet residual across measured repeats. "
                  "This separates approximate low-rank reconstruction from more accurate singular-triplet computation; "
                  "a reconstruction-only speed ratio does not assert equal triplet convergence.", "",
                  "`calibration_costs.csv` sums the measured process wall time of every candidate, "
                  "including failed and timed-out candidates, and adds independent validation time. "
                  "It includes process initialization, resident setup, warm-up, repeated solves and factor export; "
                  "this cost is distinct from the chosen configuration's median solve time.", "",
                  "The solver eps is not a Frobenius threshold. Unattained thresholds and missing/invalid "
                  "results have no assigned speedup. A timeout is not evidence of OOM.", "",
                  "The rSVD side is a complete-factor reference path for the manuscript's residual diagnostics "
                  "on a shared vendor CSR backend. It does not reuse historical CUDA-event times, and these "
                  "measurements do not benchmark the earlier reduction-tree/overlap implementations.", ""])
    (directory / "analysis.md").write_text("\n".join(lines))
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_directory", type=Path)
    analyse(parser.parse_args().campaign_directory)
