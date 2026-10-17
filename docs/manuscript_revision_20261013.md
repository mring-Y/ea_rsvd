# Manuscript analysis supplement

This source supplement accompanies *Residual Pair Diagnosis and Finite Grid
Calibration for Sparse GPU Randomized SVD*. The original campaign is archived
at DOI `10.5281/zenodo.23262787`; its release metadata is retained in
`CITATION.cff`. This local supplement has not been deposited as a new Zenodo
release and does not change the contents behind that DOI.

## Analysis corrections

- `analyze_common_start_actions.py` compares power and rank candidates against
  one shared starting residual. Its cost denominator is the complete stored
  candidate interval. It reports 25 primary comparisons and, with
  `--all-shared`, 150 sensitivity comparisons. Grouping by matrix and seed
  provides descriptive summaries, with no fitted or held-out classifier.
- The designed-spectrum generators exclude `k > ell` states and use the same
  initial random matrix across power depths. The mechanism, stress, and path
  experiments contain 180, 600, and 375 admissible rows, respectively.
- `validate_residual_identity.py` produces 540 actual CPU configurations with
  FP32 or FP64 factors. It compares projected-energy residuals with explicit
  FP64 Frobenius reconstruction and records clipping and threshold agreement.
- `make_manuscript_print_figures.py` redraws seven main figures from stored
  measurements. The common-start analyzer produces the eighth main figure.
  Additional plot scripts remain available for supplementary exploration;
  their historical output numbers are not manuscript figure numbers.
- The scale summary requires the declared orthogonality check for every
  valid row. A cohort containing only QR warnings has `n_valid=0` and empty
  means, while its raw records are retained. The two Webbase-1M rank-128,
  power-2 records are excluded from the main plot. Rank 64 uses two valid
  seeds and rank 256 uses one. Regression tests cover the all-warning cohort
  and a misleading `PASS` label with a failed numerical check.

## Factorization sources and timing scopes

`rsvd_mm_exact_noprobe.cu` evaluates the full projected-energy residuals without
random probes and supports the rank-grid comparisons. The legacy CSV fields
beginning with `exact_` mean deterministic computed-energy evaluation, subject
to finite-precision factors and scalar rounding.

`rsvd_mm_full_energy_probe.cu` is a copy of the source used for the 450-state
fixed grid. It computes both full-energy residuals and the 16-probe projection
estimate. Its CUDA interval includes probe work and intermediate projected
singular-value computations. Factors and the GPU Gram check use FP32; scalar
energy sums use FP64 after the event interval. The original source SHA256 is
`dd4016f9ef5c4873543b7845f37aa85e222ef88606e43f9df8b2a26ec6d72cff`.
This source is supplied
separately from the older `rsvd_mm_benchmark.cu`, whose historical campaign
interface and outputs are retained.

Build this source with the regular CMake configuration. For example:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --target full_energy_probe --config Release
./build/full_energy_probe data/suitesparse/west2021/west2021/west2021.mtx \
  results/fixed_state.csv 16 8 0 20260927
```

The six executable arguments are input, output CSV, rank, oversampling,
power depth, and seed. Width equals rank plus oversampling. The fixed grid
uses rank 16, widths 24/32/40, power depths 0/1/2, and five seeds. The source
performs no separate warm-up loop and does not materialize final singular
vectors. Its event interval excludes allocation, input transfer, random
matrix setup, and host residual summaries. Times are device-specific and
must be measured again on a different device.

## Interpretation

The contribution is a residual-diagnosis and finite-grid calibration protocol
on an existing sparse GPU randomized-SVD backend. Earlier research already
covers adaptive rank, adaptive GPU range finding, and precision-guided
low-rank approximation. A component label describes the current error; it
does not guarantee the most cost-effective next action. Rank changes are
admissible only when the application permits them. Independent-run scores
do not measure cached rank-update or continuation latency.

## Published GPU solver supplement

`benchmarks/published_gpu` compares a complete-factor randomized-SVD diagnostic
reference path against the official PRIMME_SVDS CUDA/cuBLAS backend. The pinned
PRIMME algorithm source is unmodified; a shared cuSPARSE CSR adapter supplies
the matrix and its transpose. This supplement returns U, singular values and
V and uses synchronized wall-clock timing. It therefore requires new
measurements on every GPU and cannot be combined with the earlier
diagnostic-only CUDA-event intervals.

The primary comparison fixes the input, rank, precision, seed and device and
matches independently validated Frobenius reconstruction quality. A second
comparison also imposes a common singular-triplet residual bound. PRIMME's
convergence tolerance is recorded separately from these actual quality
targets. Every timed repeat is independently validated; failures, poor
orthogonality and unattained quality targets remain in the records. The
analysis includes the measured cost of searching all finite-grid candidates.

Complete-output evidence now includes independently measured RTX 5070 Laptop,
A800 80GB PCIe, and RTX 4090D 24GB results. The A800 primary grid has 600 configurations and
50/50 reconstruction pairs with a geometric mean PRIMME/rSVD time ratio of
7.16. Its expanded grid has 200 configurations, 20/20 reconstruction pairs,
and a ratio of 3.35; PRIMME is faster in two groups. The same expanded grid,
effective input hashes, and core sources yield a local ratio of 3.33. Adding
triplet limits of 1e-3 or 1e-4 retains 16/20 or 12/20 expanded-grid pairs on
all three devices. The RTX 4090D primary and expanded grids match the A800's
600 and 200 configurations, with reconstruction time ratios of 7.66 and 3.51.
Of these configurations, 730 are valid, six return PRIMME solver failures,
and 64 exceed the factor-orthogonality limit; none time out. All 380 RTX 4090D
rSVD configurations pass. Its expanded-grid joint triplet targets yield time
ratios of 2.37 and 2.75 at limits 1e-3 and 1e-4. The local primary grid includes
rank 64 and is not identical to the two cloud primary grids. Actual toolchains
are CUDA 12.8 locally and 11.8 on both cloud devices. Numerical and analysis
sources match; the RTX 4090D campaign only adds progress-message text, with
the different source hash retained. Driver and validation-library versions
remain device specific. Historical diagnostic-event measurements keep their
original scope; the new timers include complete singular-vector formation.

PRIMME is an independent iterative truncated-SVD reference. The author code
for the Tomás et al. GPU randomized-SVD/blocked-Lanczos method remains to be
obtained and verified. This external reference adds execution and numerical
context to the manuscript's residual-diagnosis and calibration contribution.

The standalone entry point `run_autodl_review.sh` runs the 600-configuration
paper profile followed by a 200-configuration quality profile. The latter
tests deeper randomized-SVD powers and looser PRIMME tolerances while retaining
joint reconstruction/triplet validation. New result archives save core source
snapshots and check source/binary hashes against the build manifest. An optional
SciPy/PROPACK cross-check supplies an independent CPU numerical reference after
GPU timing; it does not contribute GPU performance evidence.
