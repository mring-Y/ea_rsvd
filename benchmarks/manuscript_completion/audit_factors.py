"""Reproduce the Table 5 comparison on the effective FP32 input.

The script deliberately avoids SciPy so that the numerical audit can also run
inside the minimal WSL/AutoDL environment.  It parses Matrix Market, writes the
canonical PGCSR input used by the CUDA executable, and validates returned
complete factors by bounded FP64 block reconstruction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

CSR_MAGIC = b"PGCSR001"
FACTOR_MAGIC = b"PGFACT01"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_mm(path: Path):
    with path.open("r", encoding="ascii", errors="replace") as f:
        header = f.readline().strip().lower().split()
        if len(header) < 5 or header[:3] != ["%%matrixmarket", "matrix", "coordinate"]:
            raise ValueError("only Matrix Market coordinate input is supported")
        if header[3] not in {"real", "integer", "pattern"} or header[4] not in {"general", "symmetric", "hermitian"}:
            raise ValueError("only real, integer or pattern general/symmetric input is supported")
        symmetric = header[4] in {"symmetric", "hermitian"}
        line = f.readline()
        while line and line.lstrip().startswith("%"):
            line = f.readline()
        m, n, entries = map(int, line.split()[:3])
        rows = [[] for _ in range(m)]
        for _ in range(entries):
            parts = f.readline().split()
            if len(parts) < 2:
                raise ValueError("truncated Matrix Market entry")
            i, j = int(parts[0]) - 1, int(parts[1]) - 1
            value = float(parts[2]) if len(parts) >= 3 else 1.0
            rows[i].append((j, value))
            if symmetric and i != j:
                rows[j].append((i, value))
    offsets = [0]
    columns, values = [], []
    for row in rows:
        row.sort(key=lambda x: x[0])
        merged = []
        for j, value in row:
            if merged and merged[-1][0] == j:
                merged[-1] = (j, merged[-1][1] + value)
            else:
                merged.append((j, value))
        for j, value in merged:
            if value != 0.0:
                columns.append(j)
                values.append(value)
        offsets.append(len(columns))
    return m, n, np.asarray(offsets, dtype=np.int32), np.asarray(columns, dtype=np.int32), np.asarray(values, dtype=np.float64)


def write_csr(path: Path, m, n, offsets, columns, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<8sQQQ", CSR_MAGIC, m, n, len(values)))
        offsets.astype("<i4").tofile(f)
        columns.astype("<i4").tofile(f)
        values.astype("<f8").tofile(f)


def read_factors(path: Path):
    with path.open("rb") as f:
        magic, m, n, k, itemsize, reserved = struct.unpack("<8sQQQII", f.read(40))
        if magic != FACTOR_MAGIC or reserved or itemsize not in (4, 8):
            raise ValueError("invalid factor header")
        dtype = "<f4" if itemsize == 4 else "<f8"
        sigma = np.fromfile(f, dtype=dtype, count=k).astype(np.float64)
        u = np.fromfile(f, dtype=dtype, count=m * k).reshape((m, k), order="F").astype(np.float64)
        v = np.fromfile(f, dtype=dtype, count=n * k).reshape((n, k), order="F").astype(np.float64)
    if sigma.size != k or u.shape != (m, k) or v.shape != (n, k):
        raise ValueError("truncated factor payload")
    return u, sigma, v


def direct_residual(m, n, offsets, columns, values, u, sigma, v, block_rows=128):
    """Compute ||A-U diag(sigma) V^T||_F / ||A||_F in FP64 blocks."""
    norm2 = float(np.dot(values, values))
    residual2 = 0.0
    for first in range(0, m, block_rows):
        last = min(first + block_rows, m)
        dense = np.zeros((last - first, n), dtype=np.float64)
        for local, row in enumerate(range(first, last)):
            lo, hi = int(offsets[row]), int(offsets[row + 1])
            dense[local, columns[lo:hi]] = values[lo:hi]
        approx = (u[first:last] * sigma) @ v.T
        difference = dense - approx
        residual2 += float(np.sum(difference * difference, dtype=np.float64))
    return math.sqrt(max(residual2, 0.0) / norm2)


def gram_residual(m, n, offsets, columns, values, u, sigma, v):
    norm2 = float(np.dot(values, values))
    av = np.zeros((m, sigma.size), dtype=np.float64)
    for row in range(m):
        lo, hi = int(offsets[row]), int(offsets[row + 1])
        av[row] = np.sum(values[lo:hi, None] * v[columns[lo:hi]], axis=0)
    gu = u.T @ u
    gv = v.T @ v
    cross = float(np.sum(u * av * sigma))
    approx2 = float(np.sum(gu * gv * np.outer(sigma, sigma)))
    return math.sqrt(max(norm2 + approx2 - 2.0 * cross, 0.0) / norm2)



def main():
    ap=argparse.ArgumentParser(description="Direct FP64 audit of exported complete GPU factors")
    ap.add_argument('--matrix',type=Path,required=True)
    ap.add_argument('--factors',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--cpu-reference',action='store_true')
    ap.add_argument('--block-rows',type=int,default=32)
    args=ap.parse_args()
    m,n,offsets,columns,raw=read_mm(args.matrix)
    values=raw.astype(np.float32).astype(np.float64)
    u,s,v=read_factors(args.factors)
    if u.shape[0]!=m or v.shape[0]!=n:raise ValueError('matrix/factor shape mismatch')
    direct=direct_residual(m,n,offsets,columns,values,u,s,v,args.block_rows)
    result=dict(matrix=args.matrix.stem,source_sha256=sha256(args.matrix),factor_sha256=sha256(args.factors),rank=len(s),
        effective_input='FP32-rounded values promoted to FP64',direct_block_residual=direct,
        gram_cross_residual=gram_residual(m,n,offsets,columns,values,u,s,v),
        u_orth_frobenius=float(np.linalg.norm(u.T@u-np.eye(len(s)))),v_orth_frobenius=float(np.linalg.norm(v.T@v-np.eye(len(s)))))
    if args.cpu_reference:
        import scipy
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import svds
        a=csr_matrix((values,columns,offsets),shape=(m,n))
        if len(s)>=min(m,n):raise ValueError('ARPACK requires rank<min(m,n)')
        cu,cs,cvt=svds(a,k=len(s),tol=1e-12,maxiter=20000,solver='arpack',random_state=4242)
        ref=direct_residual(m,n,offsets,columns,values,cu,cs,cvt.T,args.block_rows)
        result.update(cpu_reference_rankk_residual=ref,direct_over_cpu_reference=direct/ref,scipy=scipy.__version__)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
