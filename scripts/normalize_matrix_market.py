#!/usr/bin/env python3
"""Expand symmetric pattern Matrix Market files for the native CSR reader."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def header(path: Path) -> tuple[str, str, int, int, int]:
    with path.open(encoding="utf-8", errors="replace") as f:
        banner = f.readline().strip()
        if not banner.lower().startswith("%%matrixmarket matrix coordinate"):
            raise ValueError(f"invalid Matrix Market header: {path}")
        fields = banner.lower().split()
        if len(fields) != 5 or fields[1:3] != ["matrix", "coordinate"]:
            raise ValueError(f"only coordinate Matrix Market input is supported: {path}")
        field_type, symmetry = fields[3], fields[4]
        for line in f:
            if line.strip() and not line.startswith("%"):
                rows, cols, nnz = map(int, line.split()[:3])
                return field_type, symmetry, rows, cols, nnz
    raise ValueError(f"missing Matrix Market dimensions: {path}")


def normalize(root: Path, name: str) -> dict[str, object]:
    path = root / name / name / f"{name}.mtx"
    source_copy = path.with_suffix(".raw.mtx")
    source = source_copy if source_copy.exists() else path
    field_type, symmetry, rows, cols, nnz = header(source)
    if field_type == "real" and symmetry == "general":
        return {"name": name, "source": str(source), "source_sha256": sha256(source),
                "normalized": str(path), "normalized_sha256": sha256(path),
                "rows": rows, "cols": cols, "nnz": nnz, "expanded": source != path}
    if field_type not in {"pattern", "integer", "real"}:
        raise ValueError(f"unsupported Matrix Market field type {field_type}: {source}")
    if symmetry not in {"general", "symmetric"}:
        raise ValueError(f"unsupported Matrix Market symmetry {symmetry}: {source}")
    if source == path:
        import shutil
        shutil.copy2(path, source_copy)
        source = source_copy

    # Count output entries in a streaming pass; the CUDA reader sorts each row.
    expanded_nnz = 0
    with source.open(encoding="utf-8", errors="replace") as f:
        f.readline()
        dimensions_skipped = False
        for line in f:
            if line.strip() and not line.startswith("%"):
                if not dimensions_skipped:
                    dimensions_skipped = True
                    continue
                fields = line.split()
                if len(fields) >= 2:
                    expanded_nnz += 1 + (symmetry == "symmetric" and int(fields[0]) != int(fields[1]))
    tmp = path.with_suffix(".mtx.tmp")
    with source.open(encoding="utf-8", errors="replace") as src, tmp.open("w", encoding="ascii", newline="\n") as dst:
        src.readline()
        dst.write("%%MatrixMarket matrix coordinate real general\n")
        dst.write(f"% normalized from {field_type} {symmetry}; original={source_copy.name}\n")
        dst.write(f"{rows} {cols} {expanded_nnz}\n")
        dimensions_skipped = False
        for line in src:
            if not line.strip() or line.startswith("%"):
                continue
            if not dimensions_skipped:
                dimensions_skipped = True
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            r, c = int(fields[0]), int(fields[1])
            value = fields[2] if len(fields) > 2 else "1.0"
            dst.write(f"{r} {c} {value}\n")
            if symmetry == "symmetric" and r != c:
                dst.write(f"{c} {r} {value}\n")
    tmp.replace(path)
    return {"name": name, "source": str(source), "source_sha256": sha256(source),
            "normalized": str(path), "normalized_sha256": sha256(path),
            "rows": rows, "cols": cols, "source_nnz": nnz,
            "nnz": expanded_nnz, "expanded": symmetry == "symmetric",
            "source_field_type": field_type, "source_symmetry": symmetry}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matrices", nargs="+", required=True)
    args = parser.parse_args()
    rows = [normalize(args.root, name) for name in args.matrices]
    manifest = args.root / "autodl_matrix_manifest.json"
    import json
    manifest.write_text(json.dumps({"matrices": rows}, indent=2), encoding="utf-8")
    print(manifest)


if __name__ == "__main__":
    main()
