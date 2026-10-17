# Resident cache and calibration cost

The break-even counts below compare against independent solves for every request. They do not compare against an optimal final-output cache: repeated identical outputs can be retained, and a known maximum-rank factorization supports smaller-rank prefixes. The reviewer revision uses the explicit three-rank cost model in reviewer_evidence.json instead of these repeated-rank counts as a manuscript claim.

All medians use seven saved warm samples. The first cache build is recorded separately as cold and excluded from these medians. Effective input, seed, width40, power2, precision and output are matched. Timings use synchronized wall time and include per-operation allocations; export and independent FP64 checks are outside. Spectrum lookup is a single-value copy, not an adaptive rank-selection algorithm.

| matrix | k | build ms | cached factors ms | fresh solve ms | cache break-even | with rank16 nine-state grid |
|---|---:|---:|---:|---:|---:|---:|
| lp_osa_30 | 8 | 33.816 | 0.603 | 34.349 | 2 | - |
| lp_osa_30 | 16 | 33.816 | 0.942 | 34.894 | 1 | 8 |
| lp_osa_30 | 32 | 33.816 | 1.247 | 31.710 | 2 | - |
| west2021 | 8 | 14.355 | 0.069 | 13.997 | 2 | - |
| west2021 | 16 | 14.355 | 0.157 | 13.086 | 2 | 9 |
| west2021 | 32 | 14.355 | 0.708 | 13.142 | 2 | - |

For N repeated requests on an unchanged matrix and retained state, Ccache(N)=Cbuild+N*Cfactor and Cfresh(N)=N*Tfresh. The cache-only break-even is ceil(Cbuild/(Tfresh-Cfactor)). A separate conservative scenario pays Ccal for all nine complete-factor candidates at rank16 and rebuilds the chosen cache, giving ceil((Ccal+Cbuild)/(Tfresh-Cfactor)); Ccal={'lp_osa_30': 221.6805, 'west2021': 99.1317}. The predeclared (40,2) state isolates reuse, not target attainment or the fastest feasible setting. CUDA-event replay costs are not mixed with these wall-time costs.

Six cached/fresh pairs: max direct residual difference 3.5049e-11; max cached U/V orthogonality 6.0347e-06.
