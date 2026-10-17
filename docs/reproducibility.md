# Reproducing the experiments

The complete campaign is designed for a Linux CUDA host such as an AutoDL RTX 4090D instance. It requires:

- CUDA toolkit with `nvcc`, cuBLAS, cuSPARSE, cuSOLVER, and cuRAND;
- Python 3 with NumPy and SciPy;
- `curl` with resume support and `tar`;
- CMake 3.25 or newer only when using the optional CMake build.

From the repository root, a short smoke run is:

```bash
CUDA_ARCH=89 RUN_CUML=0 \
SEEDS="20261003" \
MATRIX_NAMES="west2021 ca-AstroPh" \
bash scripts/run_autodl_all.sh
```

The full campaign is:

```bash
CUDA_ARCH=89 RUN_CUML=1 bash scripts/run_autodl_all.sh
```

The parameterized entry point selects the same source set for all supported
devices:

```bash
EXPERIMENT_PLATFORM=4090d EXPERIMENT_MODE=rank bash scripts/run_experiment.sh
EXPERIMENT_PLATFORM=a800 EXPERIMENT_MODE=reference bash scripts/run_experiment.sh
```

`EXPERIMENT_MODE` accepts `all`, `rank`, `precision`, `reference`, `overlap`,
and `structure`. It changes the matrix profile and optional baselines; it does
not change the residual definitions or the CUDA timing boundary.

The `structure` mode is an independent short profile. It uses
`ca-AstroPh`, `ca-HepPh`, `email-EuAll`, and `p2p-Gnutella31`, evaluates
`k=16`, `ell=24`, and `q` in `{0,1,2}` for each requested seed, and compiles
only `rsvd_mm_exact_noprobe.cu`. It exits after writing the result archive, so
it does not run the full baseline, precision, overlap, or optional cuML
campaign:

```bash
EXPERIMENT_PLATFORM=4090d EXPERIMENT_MODE=structure \
  SEEDS="20261011 20261012" bash scripts/run_experiment.sh
```

The same command with `EXPERIMENT_PLATFORM=a800` targets `sm_80`. The archive
must be described as A800 data when that platform is used; it is not an A100
measurement.

For a bounded Ampere platform comparison on an A800, run:

```bash
CUDA_ARCH=80 bash scripts/run_autodl_a800_minimal.sh
```

This profile runs the 14 fixed-reference matrices with `k=16`, `ell=24`,
`q=1`, and one seed, followed by 27 FP32/FP64/Kahan decision-replay cases on
three representative matrices. It omits the long overlap and full baseline
sweeps. The output describes an A800 (`sm_80`) and does not substitute for a
measurement on a physical A100.

The same source path can be run on an A100 by setting the platform label
explicitly:

```bash
EXPERIMENT_PLATFORM=a100 CUDA_ARCH=80 \
  bash scripts/run_autodl_a800_minimal.sh
```

The executable is rebuilt on the target host for `sm_80`; timing, overlap,
memory, energy, and failure status remain device-specific and require fresh
measurements. A migration to another supported architecture likewise needs a
new compile and a new run, but does not require a separate kernel rewrite.

The runner is non-interactive and binds standard input to `/dev/null`, so it can be detached safely:

```bash
nohup env CUDA_ARCH=89 RUN_CUML=0 bash scripts/run_autodl_all.sh \
  > run_autodl_all.log 2>&1 </dev/null &
```

Set `INCLUDE_ROADNET=1` only when the host has enough disk and host memory. Set `MATRIX_NAMES` to a space-separated verified list to limit a run. `CUDA_ARCH` is `89` for RTX 4090/4090D, `80` for A100/A800, and `90` for H100/H20.

The runner creates `results/<UTC-run-id>/`. It records `status.tsv`, `failures.tsv`, `failure_details.tsv`, per-case logs, CSV outputs, the environment, and a source/data manifest. A failed download, compilation, or case is recorded and does not stop unrelated cases. Exit-code classes are retained in `failure_details.tsv`:

- `1`: CUDA or library runtime failure;
- `2`: numerical or solver validation failure;
- `65`: invalid input or rank;
- `127`: missing executable or dependency.

The optional `RUN_CUML=1` path probes the installed RAPIDS `TruncatedSVD` API. It labels the selected algorithm; only rows with `algorithm_selected=randomized` are randomized-SVD results. Unsupported sparse input is reported as a separate result and does not relabel the native CUDA implementation.

The float64 ARPACK reference uses SciPy's `scipy.sparse.linalg.svds`. MATLAB DashSVD comparisons require the official DashSVD source to be supplied separately; that upstream source and its license are not redistributed in this repository.

The asset generator uses NumPy and Pillow with the state-level CSV files.
The print-size manuscript plots additionally use Matplotlib through
`scripts/make_manuscript_print_figures.py`. The controlled-spectrum experiment
is a CPU numerical check:

```bash
python3 scripts/generate_paper_assets.py --work-root /path/to/work-root --output results/paper_assets
```

To regenerate the action-level evidence used in the manuscript, run:

```bash
python3 scripts/analyze_rank_actions.py --work-root /path/to/work-root --output results/paper_assets
```

The analysis reads the rank-sweep CSV files and writes the matched residual
reductions, paired timing deltas, and threshold-grid stability summaries. It
is a deterministic post-processing step and does not rerun CUDA kernels.

The structure summary reads either of the two supported state formats. The
resident benchmark reports `projection_residual_estimate` and
`rank_k_reconstruction_residual` with `probe_count > 0`, while the
exact-no-probe executable reports `exact_projection_residual` and
`exact_rank_residual` with `probe_count=0`. The scope is retained in
`nonlp_structure_summary.csv` and the Markdown summary; the two scopes should
not be pooled into one mean.

For a quality-cost view of the two candidate actions, replay the exact rank
sweep:

```bash
python3 scripts/analyze_action_efficiency.py \
  --input /path/to/work-root/review_experiments/round20_rank_adequacy_multiseed/rank_adequacy_grid.csv \
  --output /path/to/work-root/paper_assets
```

Both candidates share the starting state `(k=16, ell=40, q=0)`. The power
candidate is `(16,40,2)` and the rank candidate is `(32,40,0)`. Both gains
subtract the same starting residual. Their descriptive scores divide that
gain by each complete stored GPU interval, rather than by cached continuation
latency. Rank candidates are admissible only when the application permits
rank changes. The output is post-processing of measured CUDA states.

For the extended evidence used in the manuscript, first summarize the stored
controller ablation and large-scale rank scans, then render their figures:

```bash
python3 scripts/summarize_extended_experiments.py --work-root /path/to/work-root
python3 scripts/make_extended_figures.py --assets /path/to/work-root/paper_assets
```

The ablation compares PVE-only, residual-only, and dual-monitor policies on
ten matrices with five seeds per matrix. The scale inputs are Amazon0312
(one seed) and Webbase-1M (two seeds where available). The scripts retain
status and scope columns, exclude QR-warning rows from plotted means, and do
not rerun or synthesize benchmark measurements.

The CPU mechanism-stress profile is independent of CUDA timing and can be run
on any host with NumPy:

```bash
python3 scripts/run_spectral_stress.py --output results/spectral_stress
python3 scripts/make_spectral_stress_figure.py \
  --input results/spectral_stress/spectrum_stress_rows.csv \
  --output results/paper_assets/fig14_spectral_stress.png
```

The finite-grid frontier and seed-stability analysis is deterministic:

```bash
python3 scripts/analyze_frontier_stability.py \
  --input /path/to/work-root/review_experiments/round20_rank_adequacy_multiseed/rank_adequacy_grid.csv \
  --output /path/to/work-root/paper_assets
```

The controlled-spectrum controller replay is an independent CPU mechanism
experiment. It constructs five singular-value families, evaluates the same
residual pair over `(ell,q,k)` states in FP64, and compares the controller
path with the best stored rank-16 state for each seed:

```bash
python3 scripts/run_spectral_controller_cpu.py \
  --output results/spectral_controller
python3 scripts/make_spectral_controller_figure.py \
  --summary results/spectral_controller/spectral_controller_summary.csv \
  --output results/paper_assets/fig16_spectral_controller.png
```

The output contains 375 admissible state rows and five summaries. CPU state time remains
inside this mechanism experiment and is not compared with CUDA-event timings.
The flat-spectrum family exercises the budget-limited branch and is reported
with that status.

The controller-path and held-out-transfer diagnostics are deterministic
post-processing of the exact rank-sweep CSV:

```bash
python3 scripts/replay_controller_path.py \
  --input /path/to/work-root/review_experiments/round20_rank_adequacy_multiseed/rank_adequacy_grid.csv \
  --output /path/to/work-root/paper_assets/controller_path
python3 scripts/replay_transferability.py \
  --input /path/to/work-root/review_experiments/round20_rank_adequacy_multiseed/rank_adequacy_grid.csv \
  --output /path/to/work-root/paper_assets/transferability
```

The first replay follows the residual-pair path from a small baseline to
sketch expansion and, when indicated, a larger target rank. The second selects
a fixed `(ell,q)` setting from the other matrices and reports held-out exact
residuals. Neither command launches CUDA.

The independent non-LP rank-by-power experiment is a separate CUDA campaign:

```bash
python3 scripts/run_nonlp_rank_sweep.py \
  --executable /path/to/rsvd_mm_exact_noprobe \
  --matrix-root /path/to/work-root/data/suitesparse \
  --output results/nonlp_rank_sweep
python3 scripts/summarize_nonlp_rank_sweep.py \
  --input results/nonlp_rank_sweep/nonlp_rank_sweep.csv \
  --output results/paper_assets/nonlp_rank_sweep
```

The default profile is 3 matrices x 3 ranks x 3 power depths x 2 seeds = 54
states. A failed or timed-out state is retained in the CSV and the remaining
states continue. Exact-no-probe rows remain separate from the probe-inclusive
structure summary because the two scopes use different residual estimators.

The descriptive action-agreement check is deterministic:

```bash
python3 scripts/analyze_common_start_actions.py \
  --input /path/to/work-root/review_experiments/round20_rank_adequacy_multiseed/rank_adequacy_grid.csv \
  --output results/paper_assets/common_start_actions
```

It compares threshold regions and the larger energy component with the
measured score winner on 25 common-start triplets. Matrix and seed grouping
are descriptive statistics, without leave-one-out fitting. `--all-shared`
expands to 150 comparisons at shared widths and starting powers 0 and 1.
The legacy `analyze_loso_action_selection.py` entry point calls the corrected
analysis for compatibility; its filename does not imply predictive validation.

The direct-reconstruction check produces 540 CPU rows in two arithmetic
policies and compares energy residuals with explicit FP64 reconstruction:

```bash
python3 scripts/validate_residual_identity.py --output results/residual_identity_check
```

The designed-spectrum generators exclude configurations where `k > ell` and
share their starting random test matrix across power states. The mechanism
scatter has 180 admissible rows; the stress sweep has 600. These are CPU
numerical experiments, without GPU performance claims.

The decision-sensitivity replay applies controlled relative perturbations to
the stored `rho_l` and `rho_k` values and reports label-flip and class-retention
rates:

```bash
python3 scripts/analyze_decision_stability.py \
  --input /path/to/work-root/review_experiments/round20_rank_adequacy_multiseed/rank_adequacy_grid.csv \
  --output results/decision_stability
```

The budget replay records quality after each visited state and cumulative GPU
state cost along the same stored paths:

```bash
python3 scripts/analyze_budget_curve.py \
  --input /path/to/work-root/review_experiments/round20_rank_adequacy_multiseed/rank_adequacy_grid.csv \
  --output results/budget_curve
```

Both analyses are post-processing and do not claim additional CUDA runs.

For the matched-width quality check, run `matlab/run_cpu_public_matched.m`. After obtaining the official DashSVD source and setting `DASHSVD_ROOT`, run `matlab/run_dashsvd_public_matched.m`. Both runners accept an explicit Matrix Market root and write CSV output below the ignored `results/matlab/` directory.

## CMake build

The main runner compiles directly with `nvcc` so that each build command is recorded in the run logs. For an explicit build, use:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release -j
```

## Tests and release checks

On any host with Python:

```bash
python -m unittest discover -s tests -v
python scripts/validate_release.py
bash -n scripts/run_autodl_all.sh
```

The validation script checks that no release source contains local absolute paths, interactive commands, generated binaries, or unverified `lp_plans` references.
