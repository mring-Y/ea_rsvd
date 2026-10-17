# Hardware migration

The CUDA source tree is shared across the tested NVIDIA platforms. A hardware
migration requires a target-architecture build and a fresh measurement archive;
it does not require a separate implementation of the residual definitions,
CSR layout, controller state schema, or post-processing scripts.

| Device family | CUDA architecture | Entry point | Measurements that must be repeated |
|---|---:|---|---|
| RTX 4090D | `sm_89` | `EXPERIMENT_PLATFORM=4090d EXPERIMENT_MODE=rank bash scripts/run_experiment.sh` | GPU time, memory limit, numerical status, overlap, and failure records |
| A800 | `sm_80` | `EXPERIMENT_PLATFORM=a800 CUDA_ARCH=80 bash scripts/run_experiment.sh` | same metrics, recorded as A800 |
| A100 | `sm_80` | `EXPERIMENT_PLATFORM=a100 CUDA_ARCH=80 bash scripts/run_experiment.sh` | same metrics on the physical A100 |
| H100/H20 | `sm_90` | `EXPERIMENT_PLATFORM=h100 CUDA_ARCH=90 bash scripts/run_experiment.sh` | same metrics plus any architecture-specific overlap behavior |

The `sm_80` binary is compatible with the Ampere family, but A800 and A100
results remain separate because clocks, memory configuration, host links, and
runtime conditions differ. The runner records the platform label in the
environment and result archive. Likewise, an `sm_90` build is a compilation
target, not evidence that an H100 measurement has been performed.

For a full target-device campaign, use the non-interactive runner after
setting the architecture explicitly:

```bash
EXPERIMENT_PLATFORM=4090d CUDA_ARCH=89 RUN_CUML=0 \
  bash scripts/run_autodl_all.sh
EXPERIMENT_PLATFORM=a800 CUDA_ARCH=80 RUN_CUML=0 \
  bash scripts/run_autodl_all.sh
EXPERIMENT_PLATFORM=a100 CUDA_ARCH=80 RUN_CUML=0 \
  bash scripts/run_autodl_all.sh
EXPERIMENT_PLATFORM=h100 CUDA_ARCH=90 RUN_CUML=0 \
  bash scripts/run_autodl_all.sh
```

The short Ampere profile is useful for a bounded cross-platform check, but it
does not replace the full-device measurements. A result archive should retain
the physical device name, CUDA version, driver, architecture, matrix hash,
seed, state parameters, event boundary, and failure status. Timing, memory
capacity, overlap, energy, sanitizer output, and third-party baseline
availability are platform-specific evidence and cannot be inferred from a
successful cross-compilation.

The current release deliberately keeps dynamic tree/pool execution,
dual-lane QR/SVD overlap, and controller-path replay as separately labelled
paths. Fusing them into one online scheduler is an engineering extension that
can reuse the existing kernels and state schema; it is not implied by the
stored replay results.

The rank-by-power runner is architecture-independent Python orchestration, but
it launches the supplied CUDA executable and therefore must be rerun after
compiling that executable for the target architecture. The rank summary and
common-start agreement scripts consume CSV state records and do not require a GPU.
A migration preserves the source code and state schema while requiring fresh
target-device measurements and a new archive label.
