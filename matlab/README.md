# MATLAB support

`run_cpu_public_matched.m` reproduces the CPU randomized range-finder quality check used alongside DashSVD at `k=16`, `ell=24`, and one power step. `run_dashsvd_public_matched.m` runs the official DashSVD implementation under the same effective width and five fixed seeds.

The repository does not redistribute the upstream DashSVD source. Obtain it under its upstream license, then either set `DASHSVD_ROOT` or pass the source directory as the third argument to `run_dashsvd_public_matched`.

Example from the repository root:

```matlab
addpath('matlab');
run_cpu_public_matched();
run_dashsvd_public_matched([], [], getenv('DASHSVD_ROOT'));
```

Both runners accept an explicit Matrix Market root. They resolve files in either `data/<name>/<name>/<name>.mtx` or `data/<name>/<name>.mtx` layout and write generated CSV files under the ignored `results/matlab/` directory by default.

For unattended runs, start MATLAB from the repository root with `-batch`, for example:

```bash
matlab -batch "addpath('matlab'); run_cpu_public_matched();"
```

If MATLAB is not on `PATH`, replace `matlab` with the full path to the installed executable. The portable source archive does not include a machine-specific MATLAB launcher.
