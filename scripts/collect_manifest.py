#!/usr/bin/env python3
"""Record portable source and result hashes for one experiment run."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SOURCE_SUFFIXES = {".cu", ".h", ".hpp", ".cpp", ".py", ".sh", ".cmake", ".cff", ".md", ".txt"}
EXCLUDED_PARTS = {".git", "build", "results", "data", "__pycache__", ".pytest_cache"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_files(root: Path, output: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    output = output.resolve()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() == output:
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        files.append({
            "path": relative.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return files


def source_files(root: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() not in SOURCE_SUFFIXES and path.name not in {"CMakeLists.txt", "requirements.txt"}:
            continue
        entries.append({"path": relative.as_posix(), "sha256": sha256(path)})
    return entries


def command(*cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.out.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root_name": root.name,
        "nvcc": command("nvcc", "--version"),
        "nvidia_smi": command("nvidia-smi"),
        "source_files": source_files(root),
        "files": portable_files(root, output),
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
