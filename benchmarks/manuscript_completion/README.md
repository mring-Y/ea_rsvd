# Manuscript evidence supplement dated 2026 10 15

This supplement contains the numerical audit, common-target policy reanalysis,
resident GPU cache experiment, additional graph grid, and matrix-level PRIMME
sensitivity analysis for the manuscript supplied with this release. It extends
the original campaign archived as DOI 10.5281/zenodo.23262787. That DOI has not
been updated by this local preparation. No author implementation of Tomás et al.
is distributed or claimed as a measured baseline.

The core 450 states and additional 72 states are measured diagnostic grids.
`matched_replay.py` performs post hoc routing on these rows. Both routes test
the same final rank residual; the historical projection-only stopping rule is
not relabeled as output-rank acceptance. Three additional matrices appeared in
earlier descriptive studies; roadNet-CA is newly added. Repeats and seeds are
nested within matrices. Failed and unattained cases must remain reported.

## Rebuild the reported analyses without a GPU

Run from this directory using Python 3.10+ with NumPy installed:

```bash
python matched_replay.py --grid evidence/core450.csv --out results/core --dataset core450
python matched_replay.py --grid evidence/additional72.csv --out results/additional --dataset additional72
python analyse_cached.py --results evidence/cache --validation evidence/cached_validation.json --out results/cache
python summarize_primme.py --input evidence/primme_expanded_groups.csv --out results/primme
python reviewer_evidence.py --grid evidence/core450.csv --references evidence/table5_audit.csv --cache-costs evidence/cached_costs.csv --out results/reviewer.json
python budget_cohort.py --input evidence/submission/budget_curve/budget_curve_rows.csv --output results/budget_fixed_cohort.csv
python -m unittest discover -s tests
```

The compact CSVs retain every measured state and the numeric fields needed by
these analyses; absolute paths and captured machine-local console output are
removed. Original source-file hashes and effective-input hashes are recorded in
`evidence/provenance.json`. Factors and matrices are not in this source archive.

The submission evidence in `evidence/submission/` additionally contains the
925-state rank grid, the distinct probe-inclusive 450-state grid, graph and
designed-spectrum records, paired devices, and all seven complete-output
comparison grids including failed records and candidate-selection summaries.
`source_manifest.json` gives source and export hashes and discloses omitted
path/console fields and admissibility filters. `environment_manifest.json`
records hashes for the environment, build and input manifests of all seven
solver grids. The sanitized manifests retain solver commits, compiler and
runtime versions, launch parameters and input hashes; local command paths
and device UUIDs are omitted. `matrix_dimensions.csv` lists
the measured input dimensions and nonzeros. `reference_metadata.json` records
the online reference checks on 2026-10-16. `budget_fixed_cohort.csv` supplies
Table 11: each budget cap retains all 25 paths, carrying a terminated path's
last measured result and cost. The original available-case summary remains
under `budget_curve/` for provenance and is not the current Table 11.

## Run the cache experiment on a CUDA host

From the repository root, set architecture 80 for A800 or 89 for RTX 4090D:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target cached_reuse -j
python benchmarks/manuscript_completion/run_reuse.py --binary build/cached_reuse --matrix-root data/suitesparse --out results/reuse_new
```

Supply normalized real Matrix Market inputs west2021 and lp_osa_30 in a flat,
one-level or SuiteSparse two-level directory. URLs and normalized input hashes
are in `evidence/provenance.json` and the parent download manifests. The runner
prints each case and the CUDA executable prints each rank. There are no `read`
or Python `input` prompts. Each failed or timed-out case is retained in the run
status JSON. Seven warm samples and cold initialization are saved separately.
The `.samples.csv` files permit exact median recomputation. Cached-state cost
excludes resident-input setup and factor export and includes per-use allocations.
It assumes an unchanged input and retained Q/projected SVD. The nine-state
calibration surcharge is timed with this same backend, not another timer.

The reviewer audit separates a limitation within the current subspace from
global rank infeasibility: eight of ten input-matched CPU rank-16 references
exceed 0.20. FP32 energy residuals are numerical diagnostics, not certified
interval bounds. The cache cost models compare with independent solves;
precomputing a known maximum output rank also permits prefix reuse. All six
saved rank outputs have the same FP64 Gram/cross residuals as prefixes of the
rank-32 factors. This is not a new GPU timing measurement. The manuscript
therefore reports factor-formation times and a three-rank cost model instead
of a repeated-rank break-even claim against an optimal output cache.

Audit a factor file with a CPU reference (requires SciPy):

```bash
python benchmarks/manuscript_completion/audit_factors.py --matrix data/suitesparse/west2021/west2021/west2021.mtx --factors results/reuse_new/factors/west2021/16_cached.bin --cpu-reference --out results/west_audit.json
```

For a new additional-graph run, compile `exact_noprobe` and run
`run_frozen_grid.py --executable <binary> --matrix-root <normalized-matrices>
--output <new-directory>`. Its local `protocol.json` fixes the four inputs,
two seeds and nine settings. The complete-output PRIMME campaign is separate:
use the neighboring `published_gpu` benchmark and its documented runner.

The numerical checks support residual interpretation and conditional reuse.
The fixed largest configuration is cheaper than either search in the core
cohort. No additional graph case meets the rank-16 target 0.20. The cache
measurements do not benchmark integration with dynamic tree/pool overlap.
