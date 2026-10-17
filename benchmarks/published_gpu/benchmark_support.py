"""Canonical sparse input and an independent FP64 low-rank validator.

The residual formula includes Gram and cross terms, so it does not assume
that returned factors are exact singular vectors or exactly orthonormal.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import numpy as np
from scipy import sparse
from scipy.io import mmread

CSR_MAGIC = b"PGCSR001"
FACTOR_MAGIC = b"PGFACT01"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_matrix(source: Path, target: Path) -> dict:
    a = sparse.csr_matrix(mmread(source), dtype=np.float64)
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    if max(*a.shape, a.nnz) >= 2**31:
        raise ValueError("This bounded benchmark uses 32-bit CSR indices")
    if not np.isfinite(a.data).all() or not a.nnz:
        raise ValueError("Matrix must be finite and have nonzero entries")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        f.write(struct.pack("<8sQQQ", CSR_MAGIC, *a.shape, a.nnz))
        np.asarray(a.indptr, dtype="<i4").tofile(f)
        np.asarray(a.indices, dtype="<i4").tofile(f)
        np.asarray(a.data, dtype="<f8").tofile(f)
    return {"rows": a.shape[0], "cols": a.shape[1], "nnz": a.nnz,
            "source_sha256": sha256(source), "csr_sha256": sha256(target)}


def read_csr(path: Path, precision: str = "fp64") -> sparse.csr_matrix:
    with Path(path).open("rb") as f:
        magic, m, n, nz = struct.unpack("<8sQQQ", f.read(32))
        if magic != CSR_MAGIC or min(m, n) == 0 or max(m, n, nz) >= 2**31:
            raise ValueError("Invalid canonical CSR header")
        offsets = np.fromfile(f, dtype="<i4", count=m + 1)
        columns = np.fromfile(f, dtype="<i4", count=nz)
        values = np.fromfile(f, dtype="<f8", count=nz)
        if len(values) != nz or f.read(1) or offsets[0] != 0 or offsets[-1] != nz:
            raise ValueError("Invalid canonical CSR payload")
    if precision not in {"fp32", "fp64"}:
        raise ValueError("precision must be fp32 or fp64")
    if precision == "fp32":
        values = values.astype(np.float32).astype(np.float64)
    return sparse.csr_matrix((values, columns, offsets), shape=(m, n))


def write_factors(path: Path, u, sigma, v, precision: str) -> None:
    dtype = np.dtype("<f4" if precision == "fp32" else "<f8")
    m, k = u.shape
    n = v.shape[0]
    with Path(path).open("wb") as f:
        f.write(struct.pack("<8sQQQII", FACTOR_MAGIC, m, n, k, dtype.itemsize, 0))
        np.asarray(sigma, dtype=dtype).tofile(f)
        np.asarray(u, dtype=dtype).ravel(order="F").tofile(f)
        np.asarray(v, dtype=dtype).ravel(order="F").tofile(f)


def read_factors(path: Path):
    with Path(path).open("rb") as f:
        magic, m, n, k, size, reserved = struct.unpack("<8sQQQII", f.read(40))
        if magic != FACTOR_MAGIC or size not in {4, 8} or reserved or k < 1:
            raise ValueError("Invalid factor header")
        if Path(path).stat().st_size != 40 + size * k * (1 + m + n):
            raise ValueError("Incomplete factor payload")
        dtype = "<f4" if size == 4 else "<f8"
        sigma = np.fromfile(f, dtype=dtype, count=k).astype(np.float64)
        u = np.fromfile(f, dtype=dtype, count=m*k).reshape((m, k), order="F").astype(np.float64)
        v = np.fromfile(f, dtype=dtype, count=n*k).reshape((n, k), order="F").astype(np.float64)
    return u, sigma, v


def validate_factors(a, u, sigma, v) -> dict:
    a = sparse.csr_matrix(a, dtype=np.float64)
    u, sigma, v = (np.asarray(x, dtype=np.float64) for x in (u, sigma, v))
    if sigma.ndim != 1 or u.shape != (a.shape[0], sigma.size) or v.shape != (a.shape[1], sigma.size):
        raise ValueError("Factor dimensions do not match matrix and rank")
    if not all(np.isfinite(x).all() for x in (a.data, u, sigma, v)) or (sigma < 0).any():
        raise ValueError("Factors and matrix must be finite; singular values must be nonnegative")
    norm2 = float(np.dot(a.data, a.data))
    if norm2 == 0:
        raise ValueError("A relative residual is undefined for a zero input")
    gu, gv = u.T @ u, v.T @ v
    av = a @ v
    cross = float(np.sum(u * av * sigma))
    approx2 = float(np.sum(gu * gv * np.outer(sigma, sigma)))
    residual2 = norm2 + approx2 - 2*cross
    roundoff = 64*np.finfo(np.float64).eps*(norm2 + abs(approx2) + 2*abs(cross))
    method = "fp64_gram_cross"
    if residual2 < -roundoff:
        raise ValueError("Negative residual beyond FP64 roundoff bound")
    if residual2 <= roundoff:
        # A bounded dense row block avoids catastrophic energy subtraction.
        # This path is only needed near exact reconstruction.
        rows_per_block = max(1, (32*1024*1024)//(16*a.shape[1]))
        residual2 = 0.
        for first in range(0, a.shape[0], rows_per_block):
            last = min(first + rows_per_block, a.shape[0])
            difference = a[first:last].toarray() - (u[first:last]*sigma) @ v.T
            residual2 += float(np.sum(difference*difference))
        method = "fp64_block_reconstruction"
    left = av - u*sigma
    right = a.T @ u - v*sigma
    triplets = np.sqrt(np.sum(left*left, axis=0) + np.sum(right*right, axis=0))
    identity = np.eye(sigma.size)
    return {"relative_residual": float(np.sqrt(residual2/norm2)),
            "validation_method": method,
            "u_orthogonality": float(np.linalg.norm(gu-identity, ord="fro")),
            "v_orthogonality": float(np.linalg.norm(gv-identity, ord="fro")),
            "triplet_residual_max": float(np.max(triplets)),
            "triplet_residual_max_relative_fro": float(np.max(triplets)/np.sqrt(norm2)),
            "singular_values": sigma.tolist(),
            "matrix_frobenius": float(np.sqrt(norm2))}
