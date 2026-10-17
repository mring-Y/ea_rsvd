# Published solver and numerical dependencies

The independent solver is the official PRIMME_SVDS CUDA/cuBLAS backend.
Its algorithm source is unmodified. `gpu_compare.cu` supplies a CSR
cuSPARSE callback, common timing, and complete factor export.

Wu, Lingfei; Romero, Eloy; Stathopoulos, Andreas. PRIMME_SVDS A
High Performance Preconditioned SVD Solver for Accurate Large Scale
Computations. SIAM Journal on Scientific Computing 39(5), S248–S271,
2017. DOI: https://doi.org/10.1137/16M1082214

Upstream repository: https://github.com/primme/primme

The exact commit and archive SHA256 are recorded in
`third_party/primme_source.json`. The original BSD 3 clause license is
included as `third_party/PRIMME_COPYING.txt`. Do not describe the 2017
paper as benchmarking this exact 2026 backend version; cite the paper
and record the software version separately.

The standalone AutoDL bundle optionally includes x86_64 Linux shared
objects for CPU BLAS/LAPACK and their runtime dependencies. The source
repository can omit these binaries and use installed BLAS/LAPACK.
Versions, package URLs, licenses, and file hashes are in
`third_party/cpu_libraries.json`. OpenBLAS and LAPACK licenses are
included. GCC runtime libraries use GPLv3 with the GCC Runtime Library
Exception; their license texts are included. Corresponding GCC source:
https://ftp.gnu.org/gnu/gcc/gcc-15.2.0/gcc-15.2.0.tar.xz

OpenBLAS source: https://github.com/OpenMathLib/OpenBLAS/tree/v0.3.34

CUDA libraries are supplied by the server toolkit and are not bundled.
The input matrices are public SuiteSparse Matrix Collection datasets;
their official pages, archive URLs, and local source hashes are in
`data/matrix_sources.json`. These matrix assets belong to the standalone
experiment bundle rather than the source-only GitHub release.

PRIMME is an iterative truncated SVD solver. This benchmark is not an
official-source reproduction of Tomás et al.'s GPU randomized SVD or
blocked Lanczos implementation; that author implementation has not
been located and validated here.
