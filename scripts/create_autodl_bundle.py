#!/usr/bin/env python3
"""Create a source-only archive for transfer to a Linux CUDA host."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


INCLUDED = [
    "README.md",
    "LICENSE",
    ".gitattributes",
    ".gitignore",
    "CMakeLists.txt",
    "CITATION.cff",
    "requirements.txt",
    "data/README.md",
    "docs/reproducibility.md",
    "docs/publishing.md",
    "docs/upload_guide_zh.md",
    "docs/hardware_migration.md",
    "docs/manuscript_revision_20261013.md",
    "matlab/README.md",
    "matlab/run_cpu_public_matched.m",
    "matlab/run_dashsvd_public_matched.m",
    "scripts/collect_manifest.py",
    "scripts/create_autodl_bundle.py",
    "scripts/prepare_release.py",
    "scripts/analyze_rank_actions.py",
    "scripts/cuml_randomized_baseline.py",
    "scripts/high_precision_reference.py",
    "scripts/normalize_matrix_market.py",
    "scripts/run_autodl_all.sh",
    "scripts/run_autodl_a800_minimal.sh",
    "scripts/run_experiment.sh",
    "scripts/generate_paper_assets.py",
    "scripts/summarize_extended_experiments.py",
    "scripts/make_extended_figures.py",
    "scripts/run_nonlp_structure_sweep.py",
    "scripts/summarize_nonlp_structure.py",
    "scripts/make_nonlp_structure_figure.py",
    "scripts/analyze_action_efficiency.py",
    "scripts/analyze_common_start_actions.py",
    "scripts/validate_residual_identity.py",
    "scripts/make_manuscript_print_figures.py",
    "scripts/run_nonlp_rank_sweep.py",
    "scripts/summarize_nonlp_rank_sweep.py",
    "scripts/analyze_loso_action_selection.py",
    "scripts/run_spectral_stress.py",
    "scripts/make_spectral_stress_figure.py",
    "scripts/analyze_frontier_stability.py",
    "scripts/run_spectral_controller_cpu.py",
    "scripts/make_spectral_controller_figure.py",
    "scripts/replay_controller_path.py",
    "scripts/make_controller_path_figure.py",
    "scripts/replay_transferability.py",
    "scripts/make_transferability_figure.py",
    "scripts/analyze_decision_stability.py",
    "scripts/analyze_budget_curve.py",
    "scripts/budget_cohort.py",
    "scripts/validate_release.py",
    "tests/test_normalize_matrix_market.py",
    "tests/test_scale_summary.py",
    "tests/test_release_package.py",
    "src/cuda/rsvd_mm_benchmark.cu",
    "src/cuda/gpu_sparse_reference.cu",
    "src/cuda/controller_precision_replay.cu",
    "src/cuda/atq_device_tree_partitioned.cu",
    "src/cuda/end_to_end_overlap.cu",
    "src/cuda/accumulation_stress.cu",
    "src/cuda/dual_lane_dynamic_overlap_residual.cu",
    "src/cuda/rsvd_mm_exact_noprobe.cu",
    "src/cuda/rsvd_mm_full_energy_probe.cu",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root.parent / f"{root.name}_source_bundle.zip").resolve()
    missing = [relative for relative in INCLUDED if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError("bundle source missing: " + ", ".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    included = list(INCLUDED)
    # The new complete-factor solver comparison has its own standalone
    # data bundle. Include only its portable source and provenance here.
    supplement = root / "benchmarks/published_gpu"
    if supplement.exists():
        included.extend(p.relative_to(root).as_posix() for p in supplement.iterdir()
                        if p.is_file() and p.suffix in {".py", ".cu", ".sh", ".md"})
        included.extend(p.relative_to(root).as_posix() for p in (supplement / "tests").glob("*.py"))
        included.append("benchmarks/published_gpu/data/matrix_sources.json")
        included.extend(p.relative_to(root).as_posix() for p in (supplement / "third_party").iterdir()
                        if p.is_file() and (p.suffix in {".json", ".txt"}
                                            or p.name.endswith("_LICENSE")))
    completion = root / "benchmarks/manuscript_completion"
    if completion.exists():
        included.extend(p.relative_to(root).as_posix() for p in completion.rglob("*")
                        if p.is_file() and not any(part in {"__pycache__", "results", "build", ".git"} for part in p.relative_to(completion).parts)
                        and p.suffix in {".py", ".cu", ".md", ".json", ".csv"})
    included.append("docs/manuscript_revision_20261015.md")
    included = sorted(set(included))
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in included:
            archive.write(root / relative, arcname=f"{root.name}/{relative}")
    print(f"created {output} ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
