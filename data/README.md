# Input data

The repository intentionally does not redistribute public Matrix Market archives. The runner downloads verified inputs from the SuiteSparse Matrix Collection and records source and normalized SHA-256 hashes in each run manifest.

The default campaign uses these groups:

- `west2021`: `HB/west2021`
- `lp_80bau3b`, `lp_cre_d`, `lp_fit2p`, `lp_maros`, `lp_osa_14`, `lp_osa_30`, `lp_pds_02`, `lp_pilot87`, `lp_stocfor3`: `LPnetlib/<name>`
- `ca-AstroPh`, `ca-HepPh`, `email-EuAll`, `p2p-Gnutella31`: `SNAP/<name>`

The official archive pattern is:

```text
https://sparse.tamu.edu/MM/<group>/<name>.tar.gz
```

The mirror used by the runner is:

```text
https://suitesparse-collection-website.herokuapp.com/MM/<group>/<name>.tar.gz
```

`lp_plans` is not an entry in the official SuiteSparse `LPnetlib` index or the Netlib LP data directory and is not part of this campaign. Do not add an unverified file under that name.

Downloaded files are placed at:

```text
data/suitesparse/<name>/<name>/<name>.mtx
```

The source archive is kept in `data/suitesparse/_archives/` for resume and provenance checks. Both locations are ignored by Git.
