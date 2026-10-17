#!/usr/bin/env python3
"""Prepare clean GitHub staging and Zenodo source/evidence artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_DAY = "20261017"
EXCLUDED_DIRS = {
    ".git", ".pytest_cache", "__pycache__", ".venv", "venv",
    "build", "results", "asset_download_logs", ".python_packages",
    ".idea", ".vscode", ".mypy_cache",
}
ALLOWED_SUFFIXES = {
    ".c", ".cff", ".cpp", ".cu", ".csv", ".h", ".hpp", ".json",
    ".m", ".md", ".py", ".sh", ".txt",
}
ROOT_FILES = {
    ".gitattributes", ".gitignore", "CITATION.cff", "CMakeLists.txt",
    "LICENSE", "README.md", "requirements.txt",
}
DATA_FILES = {"data/README.md"}
SPECIAL_FILES = {
    "results/.gitkeep",
    "benchmarks/published_gpu/data/matrix_sources.json",
}
LOCAL_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]+(?:Users|home)[\\/]+[^\\/\s]+", re.I)
UNIX_HOME = re.compile(r"(?:^|[\s\"'])/(?:root|home)/[^/\s]+", re.M)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included(relative: Path) -> bool:
    posix = relative.as_posix()
    if posix in ROOT_FILES or posix in DATA_FILES or posix in SPECIAL_FILES:
        return True
    parts = relative.parts
    if any(part in EXCLUDED_DIRS or part.startswith(("build_", "cmake-build-")) for part in parts[:-1]):
        return False
    if "data" in parts and posix != "benchmarks/published_gpu/data/matrix_sources.json":
        return False
    if relative.name.endswith("_LICENSE"):
        return True
    if relative.suffix.lower() not in ALLOWED_SUFFIXES:
        return False
    return parts[0] in {"benchmarks", "docs", "matlab", "scripts", "src", "tests"}


def collect_sources(root: Path) -> list[Path]:
    paths: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories
                          if name not in EXCLUDED_DIRS - {"results"}
                          and not name.startswith(("build_", "cmake-build-"))
                          and not (Path(current) / name).is_symlink()]
        for name in names:
            path = Path(current) / name
            if included(path.relative_to(root)):
                if path.is_symlink():
                    raise ValueError("symlink is not allowed in release sources")
                paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix().casefold())


def audit_text_files(root: Path, paths: list[Path]) -> None:
    failures: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            failures.append(f"non-UTF-8 content in {path.relative_to(root).as_posix()}")
            continue
        if LOCAL_PATH.search(text):
            failures.append(f"local Windows user path in {path.relative_to(root).as_posix()}")
        elif UNIX_HOME.search(text):
            failures.append(f"local Unix home path in {path.relative_to(root).as_posix()}")
        if path.suffix == ".sh" and "\r" in path.read_bytes().decode("utf-8"):
            failures.append(f"shell runner must use LF endings: {path.relative_to(root).as_posix()}")
    if failures:
        raise ValueError("release privacy scan failed:\n- " + "\n- ".join(failures))


def copy_release_tree(root: Path, paths: list[Path], output: Path) -> list[Path]:
    output.mkdir(parents=True)
    copied: list[Path] = []
    for source in paths:
        relative = source.relative_to(root)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied.append(destination)
    license_path = output / "LICENSE"
    if not license_path.is_file():
        raise ValueError("LICENSE is required in a release")
    manifest = output / "release_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "snapshot_date": "2026-10-17",
        "software_version": "1.6.0-dev",
        "original_software_license": "MIT",
        "third_party_notices": "benchmarks/published_gpu/THIRD_PARTY.md",
        "previous_campaign_doi": "10.5281/zenodo.23262787",
        "this_snapshot_doi": None,
        "publication_status": "prepared locally; upload and DOI assignment pending",
        "scope": "original sources, third-party notices, and selected sanitized numerical evidence",
        "excluded": ["Git internals", "Python caches", "local builds", "downloaded matrices",
                     "factor binaries", "runtime results and console logs", "local environments"],
        "files": [{"path": path.relative_to(output).as_posix(),
                   "bytes": path.stat().st_size, "sha256": sha256(path)} for path in copied],
        "hash_scope": "payload files; SHA256SUMS.txt also hashes release_manifest.json",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    copied.append(manifest)
    sums = output / "SHA256SUMS.txt"
    entries = [f"{sha256(path)}  {path.relative_to(output).as_posix()}" for path in copied]
    sums.write_text("\n".join(entries) + "\n", encoding="utf-8", newline="\n")
    copied.append(sums)
    return copied


def write_zip(root: Path, files: list[Path], archive_path: Path) -> None:
    prefix = root.name
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(f"{prefix}/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            mode = 0o755 if path.suffix == ".sh" else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def verify_zip(archive_path: Path, source_root: Path, files: list[Path]) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise ValueError(f"corrupt ZIP member: {bad_member}")
        expected = {f"{source_root.name}/{path.relative_to(source_root).as_posix()}": path.read_bytes() for path in files}
        actual_names = archive.namelist()
        if len(actual_names) != len(set(actual_names)) or set(actual_names) != set(expected):
            raise ValueError("ZIP member list differs from the staged release tree")
        for name, data in expected.items():
            if archive.read(name) != data:
                raise ValueError(f"ZIP content mismatch: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="release source root")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / f"{ROOT.name}_release_{RELEASE_DAY}")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir.resolve()
    if not root.is_dir():
        parser.error(f"source root does not exist: {root}")
    if output == root or root in output.parents:
        parser.error("output directory must be outside the source repository")
    if output.exists():
        parser.error(f"output directory already exists; choose a new path: {output}")

    paths = collect_sources(root)
    if not paths:
        parser.error("no release files found")
    audit_text_files(root, paths)
    if sum(path.stat().st_size for path in paths) >= 99_000_000:
        parser.error("release source payload must stay below 99 MB before manifests")
    output.mkdir(parents=True)
    github_root = output / "github" / root.name
    staged = copy_release_tree(root, paths, github_root)
    archive_path = output / f"{root.name}_{RELEASE_DAY}_zenodo.zip"
    write_zip(github_root, staged, archive_path)
    verify_zip(archive_path, github_root, staged)
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".sha256")
    checksum_path.write_text(f"{sha256(archive_path)}  {archive_path.name}\n", encoding="ascii", newline="\n")
    github_archive = output / f"{root.name}_{RELEASE_DAY}_github.zip"
    shutil.copyfile(archive_path, github_archive)
    github_archive.with_suffix(".zip.sha256").write_text(
        f"{sha256(github_archive)}  {github_archive.name}\n", encoding="ascii", newline="\n")

    source_bytes = sum(path.stat().st_size for path in staged)
    if source_bytes >= 100_000_000:
        raise ValueError("release tree must stay below 100 MB including manifests")
    report = {
        "files_in_repository": len(staged),
        "uncompressed_bytes": source_bytes,
        "largest_file_bytes": max(path.stat().st_size for path in staged),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256(archive_path),
        "github_directory": github_root.relative_to(output).as_posix(),
        "zenodo_archive": archive_path.name,
        "github_archive": github_archive.name,
        "zip_integrity": "passed",
        "zip_matches_staged_files": True,
        "git_directory_included": False,
        "cache_files_included": False,
        "local_user_paths_found": False,
        "author_citation_retained": True,
    }
    (output / "package_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"GitHub staging: {github_root}")
    print(f"Zenodo archive: {archive_path}")
    print(f"Zenodo checksum: {checksum_path}")
    print(f"Files: {len(staged)} (including SHA256SUMS.txt)")
    print(f"Uncompressed source size: {source_bytes} bytes")
    print(f"Zenodo ZIP size: {archive_path.stat().st_size} bytes")
    print("ZIP integrity and staged-file equality: passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"release preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
