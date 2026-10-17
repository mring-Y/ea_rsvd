# Source supplement dated 2026 10 15

This local version supports *Residual Pair Diagnosis and Finite Grid Calibration
for Sparse GPU Randomized SVD*. The original campaign's DOI is
10.5281/zenodo.23262787. The published record identifies version v1.0 dated
2026-10-09. `CITATION.cff` identifies the present unpublished snapshot as
1.6.0-dev and cites that original archive separately. This preparation does
not alter the public DOI or record. A new GitHub/Zenodo release should identify
this as a manuscript supplement and link both versions. No new DOI is invented.

The new directory `benchmarks/manuscript_completion` contains portable sources,
protocols and compact numeric evidence. Table 5 uses exported complete factors
and FP64 CPU references on identical FP32-rounded input values. The policy
reanalysis uses final rank residual acceptance for both search routes, keeping
projection residual as a routing signal. Resident cache medians are computed
from sorted warm samples; cold initialization is separate. The calibration
surcharge comes from all nine candidates measured with that same cache backend.
The core 450-row replay and four-matrix 72-row grid remain separate from the
complete-output PRIMME campaign and from cache wall-clock timings.

The CPU reference runner also selects SciPy's `rng` keyword on 1.15+ and the
older `random_state` keyword where needed, recording that choice and the seed.
This preserves the 1.15 campaign path while supporting older installations;
the measured GPU algorithm sources and original reference results are unchanged.

The results retain limits: the fixed large configuration costs less than either
search in the core cohort; the residual pair saves about 1.3% versus scalar
search; none of the additional graph states reaches rank-16 residual 0.20.
Cache reuse applies to unchanged inputs and retained states, with the measured
factor-formation costs reported separately from calibration. The 2026-10-16
reviewer revision removes the repeated-rank break-even claim: identical outputs
can be retained, and known maximum-rank factors can supply smaller-rank prefixes.
It reports a three-rank cost model, the nine-candidate calibration surcharge,
and six saved-factor prefix audits instead. Eight of ten input-matched CPU
rank-16 references exceed 0.20, distinguishing rank infeasibility from a
limitation of the current sampled subspace. FP32 energy differences remain
numerical diagnostics, not certified bounds. Eleven portable scientific checks
cover the residual identities, routing, cache medians, fixed budget cohort,
and this distinction. The submission update exports all main measurement
records and recomputes Table 11 over the same 25 paths at every budget cap.
This is a diagnostic/calibration study rather than a new decomposition
algorithm or a universal acceleration claim. No comparison with the unavailable
Tomás author implementation is added.

Before publishing, run the analysis commands and tests in the supplement README,
review the file manifest, and attach the source/evidence archive to the new
release. Git internals, local account paths, binary builds, datasets and full
factors are omitted from that archive. The author citation is intentionally
retained. At preparation time this checkout has no configured Git remote, so
this local package is not an uploaded repository update or Zenodo new version.
