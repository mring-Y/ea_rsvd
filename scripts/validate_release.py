#!/usr/bin/env python3
"""Static checks for the source-only GitHub release directory."""
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
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
    "benchmarks/published_gpu/gpu_compare.cu",
    "benchmarks/published_gpu/benchmark_support.py",
    "benchmarks/published_gpu/campaign.py",
    "benchmarks/published_gpu/analyse_results.py",
    "benchmarks/published_gpu/build_gpu.py",
    "benchmarks/published_gpu/fetch_assets.py",
    "benchmarks/published_gpu/verify_assets.py",
    "benchmarks/published_gpu/rebuild_summary.py",
    "benchmarks/published_gpu/cpu_crosscheck.py",
    "benchmarks/published_gpu/watch_progress.py",
    "benchmarks/published_gpu/run_autodl_all.sh",
    "benchmarks/published_gpu/run_autodl_review.sh",
    "benchmarks/published_gpu/README_zh.md",
    "benchmarks/published_gpu/THIRD_PARTY.md",
    "benchmarks/published_gpu/data/matrix_sources.json",
    "benchmarks/published_gpu/third_party/primme_source.json",
    "benchmarks/published_gpu/third_party/PRIMME_COPYING.txt",
    "benchmarks/published_gpu/tests/gpu_correctness.py",
    "benchmarks/published_gpu/tests/gpu_memcheck.py",
    "benchmarks/published_gpu/tests/test_campaign_contract.py",
    "benchmarks/published_gpu/tests/test_scientific_contract.py",
    "benchmarks/published_gpu/tests/test_recovery_contract.py",
    "benchmarks/published_gpu/tests/test_shell_runner_contract.py",
    "benchmarks/published_gpu/tests/test_progress_contract.py",
    "benchmarks/manuscript_completion/README.md",
    "benchmarks/manuscript_completion/cached_reuse.cu",
    "benchmarks/manuscript_completion/matched_replay.py",
    "benchmarks/manuscript_completion/audit_factors.py",
    "benchmarks/manuscript_completion/evidence/table5_audit.csv",
    "benchmarks/manuscript_completion/evidence/provenance.json",
    "docs/manuscript_revision_20261015.md",
}
CODE_SUFFIXES = {".py", ".sh", ".cu", ".cpp", ".h", ".hpp", ".cmake", ".m"}
FORBIDDEN_SOURCE_PATTERNS = {
    "absolute Windows path": re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"),
    "Unix home path": re.compile(r"(?:^|[\s\"'])/(?:root|home)/"),
}


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def main() -> int:
    failures: list[str] = []
    for relative in sorted(EXPECTED):
        if not (ROOT / relative).is_file():
            fail(f"missing expected file: {relative}", failures)

    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8") if (ROOT / ".gitignore").is_file() else ""
    for required_rule in ("build/", "build_a800_minimal/", "results/*", "data/suitesparse/**/*.mtx", "*.csv", "*.tar.gz", "*.tar.gz.sha256", "__pycache__/"):
        if required_rule not in ignored:
            fail(f".gitignore is missing: {required_rule}", failures)

    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(part in {"build", "build_a800_minimal", "results", "data", ".git", "__pycache__"} for part in path.relative_to(ROOT).parts):
            continue
        if path.suffix.lower() not in CODE_SUFFIXES and path.name not in {"CMakeLists.txt", "requirements.txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in FORBIDDEN_SOURCE_PATTERNS.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                fail(f"{label}: {path.relative_to(ROOT).as_posix()}:{line}", failures)
        relative_name = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() == ".sh" and re.search(r"(?m)^\s*(?:read|readarray|pause)\b", text):
            fail(f"interactive shell input: {relative_name}", failures)
        if path.suffix.lower() == ".py" and re.search(r"\binput\s*\(", text) and path.name != "validate_release.py":
            fail(f"interactive Python input: {relative_name}", failures)

    runner = (ROOT / "scripts/run_autodl_all.sh").read_text(encoding="utf-8")
    required_runner_fragments = (
        "exec </dev/null",
        "-C - --retry 5 --retry-delay 5",
        "scripts/normalize_matrix_market.py",
        "src/cuda/rsvd_mm_benchmark.cu",
        "scripts/collect_manifest.py",
    )
    for fragment in required_runner_fragments:
        if fragment not in runner:
            fail(f"runner missing required behavior: {fragment}", failures)

    a800_runner = (ROOT / "scripts/run_autodl_a800_minimal.sh").read_text(encoding="utf-8")
    for fragment in ("exec </dev/null", "-C - --retry 5 --retry-delay 5", "scripts/normalize_matrix_market.py", "scripts/collect_manifest.py"):
        if fragment not in a800_runner:
            fail(f"A800 runner missing required behavior: {fragment}", failures)

    if failures:
        print("release validation failed:", file=sys.stderr)
        for item in failures:
            print(f"- {item}", file=sys.stderr)
        return 1
    print(f"release validation passed: {len(EXPECTED)} expected files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
