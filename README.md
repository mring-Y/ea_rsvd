# EA-Het-rSVD experiment support

This repository supports *Residual Pair Diagnosis and Finite Grid Calibration for Sparse GPU Randomized SVD*. The main analyses measure projection and rank-truncation residuals, compare candidates from a common starting state, and validate numerical and device consistency. Earlier reduction-tree and overlap sources remain available as additional engineering experiments.

This local supplement extends the archived campaign without updating its
Zenodo record. `CITATION.cff` identifies this unpublished snapshot and cites
the original archive as a separate software reference. See
[the revision note](docs/manuscript_revision_20261013.md) for analysis changes
and the precise relationship between source variants and manuscript timings.

The original experiment-support code is released under the MIT License;
see [LICENSE](LICENSE). Bundled third-party notices remain in force. The
GitHub and Zenodo upload steps are documented in [docs/publishing.md](docs/publishing.md)
and the [Chinese upload guide](docs/upload_guide_zh.md).

The release contains source and a small, explicitly selected set of sanitized
numeric evidence. Matrix inputs, full factor exports, logs, CUDA build products,
Python caches, and machine-local environments are excluded from the upload
archive. The runner downloads public inputs, records their hashes, and writes
each run to a new timestamped result directory.

The [2026-10-15 manuscript supplement](benchmarks/manuscript_completion/README.md)
adds input-matched direct residual audits, rank-residual acceptance for both
search routes, an 18-candidate GPU cache cost grid with saved timing samples,
four additional graph grids, and matrix-level PRIMME sensitivity analysis.
Its [revision note](docs/manuscript_revision_20261015.md) distinguishes this local
supplement from the original DOI archive and records the public release state.
The 2026-10-16 submission update adds compact source records for all main
analyses (including the 925 rank states and seven complete-output grids),
input dimensions, source/export hashes, reference metadata, and the fixed-cohort
budget summary used by the current Table 11. These additions are local and
are not claimed to be present in the original Zenodo archive.

An additional [published sparse GPU solver comparison](benchmarks/published_gpu/README_zh.md)
uses the official PRIMME_SVDS CUDA backend, complete singular-vector output,
common resident-CSR wall-clock timing, and independent FP64 validation.
It is a new measurement protocol, distinct from the historical
diagnostic-only CUDA-event timings. Public inputs and numerical dependency
binaries are supplied only in its separate AutoDL bundle.

## Repository layout

```text
src/cuda/       CUDA implementations used by the final experiment campaign
scripts/        non-interactive runner and Python data/reference utilities
tests/          unit tests for Matrix Market normalization
data/           download/provenance instructions; downloaded inputs are ignored
docs/           reproducibility and timing-scope notes
matlab/         optional CPU rSVD and official DashSVD comparison runners
results/        ignored run outputs; only .gitkeep is tracked
```

The CUDA sources cover the resident CSR rSVD path, an independent cuSPARSE/cuSOLVER reference, exact no-probe residuals, FP32/FP64/Kahan controller replays, accumulation stress, partitioned reduction tree/pool execution, end-to-end overlap, and dynamic two-lane QR/tree/SVD scheduling.

## Quick start on Linux CUDA

Install CUDA 12.x or a compatible toolkit with cuBLAS, cuSPARSE, cuSOLVER, and cuRAND, then run a small smoke campaign:

```bash
python -m pip install -r requirements.txt
CUDA_ARCH=89 RUN_CUML=0 \
SEEDS="20261003" \
MATRIX_NAMES="west2021 ca-AstroPh" \
bash scripts/run_autodl_all.sh
```

For the historical engineering campaign on an RTX 4090D (the current
complete-output solver comparison has its own runner below):

```bash
CUDA_ARCH=89 RUN_CUML=0 bash scripts/run_autodl_all.sh
```

The same profiles are available through one parameterized entry point. For
example, a rank-diagnosis run on an RTX 4090D and the bounded Ampere profile
are started with:

```bash
EXPERIMENT_PLATFORM=4090d EXPERIMENT_MODE=rank bash scripts/run_experiment.sh
EXPERIMENT_PLATFORM=a800 EXPERIMENT_MODE=reference bash scripts/run_experiment.sh
```

For the short non-LP structure profile, which varies only `q` on the four
SNAP matrices, run:

```bash
EXPERIMENT_PLATFORM=4090d EXPERIMENT_MODE=structure \
  SEEDS="20261011 20261012" bash scripts/run_experiment.sh
```

This profile downloads and normalizes the selected matrices, compiles only
the exact-no-probe executable, runs the requested states, and archives the
result. It does not start the full campaign. On an Ampere host, set
`EXPERIMENT_PLATFORM=a800`; the result remains labeled as A800 data.

The paper figures and the controlled-spectrum check are regenerated from the
archived CSV assets with:

```bash
python3 scripts/generate_paper_assets.py --work-root /path/to/work-root --output results/paper_assets
```

When several platform archives are present, the reference inputs can be selected explicitly:

```bash
python3 scripts/generate_paper_assets.py \
  --work-root /path/to/work-root \
  --a800-csv /path/to/a800/same_cuda_sparse_reference.csv \
  --gpu-csv /path/to/rtx4090d/same_cuda_sparse_reference.csv \
  --output results/paper_assets
```

The default search selects the newest path containing `a800` for the A800 panel and the newest `autodl_results` path for the RTX 4090D panel. Matrix names are matched from the normalized input path, so absolute paths may differ between machines.

The rank-diagnosis action comparison and threshold-stability summaries are
derived from the saved rank-sweep records with:

```bash
python3 scripts/analyze_rank_actions.py --work-root /path/to/work-root --output results/paper_assets
```

This produces `action_tradeoff.csv`, `threshold_stability.csv`,
`fig7_action_tradeoff.png`, and `fig8_threshold_stability.png`. The script
only summarizes stored measurements; it does not create additional benchmark
results.

The extended controller-ablation and large-scale rank summaries are generated
from the archived campaign records with:

```bash
python3 scripts/summarize_extended_experiments.py --work-root /path/to/work-root
python3 scripts/make_extended_figures.py --assets /path/to/work-root/paper_assets
```

The first command writes the ten-matrix three-policy ablation tables and the
Amazon0312/Webbase-1M scale tables. The second writes `fig9_controller_ablation.png`
and `fig10_extended_scale.png`. Amazon0312 is a one-seed scan; Webbase-1M has
two seeds for the available states, and rows carrying a QR warning remain in
the CSV but are excluded from plotted means.

The non-LP structure sweep can be summarized and plotted without rerunning
CUDA:

```bash
python3 scripts/summarize_nonlp_structure.py \
  --input-dir /path/to/work-root/nonlp_structure \
  --output-dir /path/to/work-root/paper_assets
python3 scripts/make_nonlp_structure_figure.py \
  --summary /path/to/work-root/paper_assets/nonlp_structure_summary.csv \
  --output /path/to/work-root/paper_assets/fig11_nonlp_structure.png
```

The summary accepts both probe-inclusive and exact-no-probe CSV fields and
records the residual scope in every output row.

The independent non-LP rank-by-power validation uses the exact-no-probe CUDA
executable on normalized Matrix Market inputs. It evaluates three matrices,
three target ranks, three power depths, and two seeds (54 states):

```bash
python3 scripts/run_nonlp_rank_sweep.py \
  --executable /path/to/rsvd_mm_exact_noprobe \
  --matrix-root /path/to/work-root/data/suitesparse \
  --output results/nonlp_rank_sweep
python3 scripts/summarize_nonlp_rank_sweep.py \
  --input results/nonlp_rank_sweep/nonlp_rank_sweep.csv \
  --output results/paper_assets/nonlp_rank_sweep
```

The runner records one row per state and continues after a timeout or CUDA
failure. The summary keeps matrix, rank, power depth, seed count, residual
scope, GPU time, and Q-orthogonality visible. It consumes inputs prepared from
the verified SuiteSparse manifest and does not invent an `lp_plans` archive.

To evaluate the fixed residual-pair action rule against the measured winner
of the stored finite-grid candidates, run:

```bash
python3 scripts/analyze_common_start_actions.py \
  --input /path/to/work-root/review_experiments/round20_rank_adequacy_multiseed/rank_adequacy_grid.csv \
  --output results/paper_assets/common_start_actions
```

Both candidates subtract the residual of `(k=16,ell=40,q=0)`: power uses
`(16,40,2)` and rank uses `(32,40,0)`. The score divides each gain by a complete
independent GPU interval. The 25 comparisons and matrix/seed groups are
descriptive statistics. `--all-shared` adds the 150-comparison sensitivity
analysis. This script fits no classifier and launches no CUDA measurement.

The CPU mechanism-stress experiment varies singular-value decay while holding
the finite-grid choices and five seeds fixed. It checks the residual
interpretation independently of sparse GPU timing:

```bash
python3 scripts/run_spectral_stress.py --output results/spectral_stress
python3 scripts/make_spectral_stress_figure.py \
  --input results/spectral_stress/spectrum_stress_rows.csv \
  --output results/paper_assets/fig14_spectral_stress.png
```

The frontier and seed-stability summary is deterministic post-processing of
the exact rank-sweep states:

```bash
python3 scripts/analyze_frontier_stability.py \
  --input /path/to/work-root/review_experiments/round20_rank_adequacy_multiseed/rank_adequacy_grid.csv \
  --output /path/to/work-root/paper_assets
```

It writes `frontier_stability_summary.csv`, `frontier_seed_stability.csv`,
and `fig15_frontier_stability.png`.

The controlled-spectrum controller replay tests the decision rule on five
designed singular-value profiles. It uses FP64 CPU states and compares the
controller path with the best stored rank-16 state for each seed. These times
are mechanism-study measurements, not GPU performance results:

```bash
python3 scripts/run_spectral_controller_cpu.py \
  --output results/spectral_controller
python3 scripts/make_spectral_controller_figure.py \
  --summary results/spectral_controller/spectral_controller_summary.csv \
  --output results/paper_assets/fig16_spectral_controller.png
```

The replay writes 375 admissible state rows and five family summaries. The flat-spectrum
case is retained to exercise the budget-limited branch when energy is spread
across the unresolved tail.

To replay the action choice on the archived exact rank-sweep states, run:

```bash
python3 scripts/analyze_action_efficiency.py \
  --input /path/to/work-root/review_experiments/round20_rank_adequacy_multiseed/rank_adequacy_grid.csv \
  --output /path/to/work-root/paper_assets
```

This writes `action_efficiency.csv`, `action_efficiency_summary.csv`,
`action_efficiency_summary.md`, and `fig_common_start_actions.png`. It is a
deterministic state replay and does not launch CUDA.

The accompanying analyses are a source supplement to the archived campaign
at DOI `10.5281/zenodo.23262787`; modifying the local source directory does
not update that published archive. The supplement corrects common-start
comparisons, admissible `k <= ell` spectrum states, shared random matrices
across power depths, and numerical verification of energy residuals.

To check those identities by direct CPU reconstruction, run:

```bash
python3 scripts/validate_residual_identity.py --output results/residual_identity_check
```

This writes 540 rows under FP32 and FP64 factors, including clipping counts,
direct residual differences, orthogonality, and threshold agreement. It
reports no GPU performance measurement.

The print-sized manuscript plots can be regenerated from the campaign CSVs:

```bash
python3 scripts/make_manuscript_print_figures.py \
  --work-root /path/to/work-root --assets /path/to/paper_assets \
  --rtx4090-csv /path/to/4090-results/same_cuda_sparse_reference.csv \
  --a800-csv /path/to/a800-results/same_cuda_sparse_reference.csv
```

To replay the short residual-pair path on the exact rank-sweep states, run:

```bash
python3 scripts/replay_controller_path.py \
  --input /path/to/work-root/review_experiments/round20_rank_adequacy_multiseed/rank_adequacy_grid.csv \
  --output /path/to/work-root/paper_assets/controller_path
python3 scripts/make_controller_path_figure.py \
  --summary /path/to/work-root/paper_assets/controller_path/controller_path_summary.csv \
  --output /path/to/work-root/paper_assets/fig17_controller_path.png
```

The replay starts at `(k=16, ell=24, q=0)`, follows the residual-pair rule,
and may evaluate `(k=16, ell=40, q=2)` and `(k=32, ell=40, q=2)`. It reports
the selected endpoint and cumulative stored-state cost. It is post-processing
and does not rerun CUDA.

To test fixed-setting transferability with a held-out matrix, run:

```bash
python3 scripts/replay_transferability.py \
  --input /path/to/work-root/review_experiments/round20_rank_adequacy_multiseed/rank_adequacy_grid.csv \
  --output /path/to/work-root/paper_assets/transferability
python3 scripts/make_transferability_figure.py \
  --input /path/to/work-root/paper_assets/transferability/transferability_rows.csv \
  --output /path/to/work-root/paper_assets/fig18_transferability.png
```

For each held-out matrix, the setting with the lowest mean exact residual on
the other matrices is selected and then evaluated using its stored rows.
This is a transfer diagnostic, not a new hardware benchmark.

To quantify sensitivity of the three-way residual-pair label to measurement
noise, run:

```bash
python3 scripts/analyze_decision_stability.py \
  --input /path/to/work-root/review_experiments/round20_rank_adequacy_multiseed/rank_adequacy_grid.csv \
  --output results/decision_stability
```

This deterministic Monte Carlo replay perturbs the stored residuals at several
relative noise levels and writes `decision_stability_summary.csv` and
`fig19_decision_stability.png`. It is a diagnostic sensitivity analysis and
does not launch CUDA.

To expose the quality-cost trajectory of the stored controller paths, run:

```bash
python3 scripts/analyze_budget_curve.py \
  --input /path/to/work-root/review_experiments/round20_rank_adequacy_multiseed/rank_adequacy_grid.csv \
  --output results/budget_curve
```

This replay writes the state-level cumulative budget, a fixed-cohort summary,
and `fig20_budget_curve.png`. All budget caps include the same 25 paths;
terminated paths retain their last measured residual and cumulative cost.
Only ten paths make a sixth evaluation, as recorded in `advancing_paths`.

The runner is fully non-interactive and safe to launch with `nohup`. It uses `curl -C - --retry 5 --retry-delay 5`, keeps partial archives, skips a matrix after a failed download or normalization, and continues with the remaining cases. See [docs/reproducibility.md](docs/reproducibility.md) for environment variables, CMake, result interpretation, and optional cuML/MATLAB dependencies.

The CUDA sources are architecture-parameterized rather than hardware-specific.
For a new NVIDIA device, set `CUDA_ARCH` to its compute capability and rerun
the same profile. A migration requires recompilation on the target host; it
does not require rewriting the kernels. A new device does require fresh
measurements for timing, overlap, memory limits, and numerical status. The
`a800`/`a100` minimal profile uses `sm_80` and records the platform label in
the result archive; run it separately for each physical device.

See [docs/hardware_migration.md](docs/hardware_migration.md) for the complete
architecture table and exact commands for RTX 4090D, A800, A100, and H100.

## Data provenance

Inputs come from the official SuiteSparse Matrix Collection. The default matrix list and verified URL groups are documented in [data/README.md](data/README.md). `lp_plans` is not an authoritative SuiteSparse or Netlib matrix name and is deliberately absent.

The optional MATLAB runners in `matlab/` reproduce the matched-width CPU/DashSVD quality check. The official DashSVD source is supplied separately by the user under its upstream license.

## License and upstream components

This repository contains original experiment support code and links to public input sources. The official MATLAB DashSVD implementation is not redistributed; users who reproduce that comparison must obtain it under its upstream license and provide its source path separately.
