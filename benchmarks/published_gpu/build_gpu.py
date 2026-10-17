"""Build pinned, unmodified PRIMME C sources and the GPU benchmark.

This follows the compilation units/flags in upstream src/Makefile. Calling
the compiler directly keeps make and system package installation optional.
All generated files stay below the selected build directory.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tool(variable, default):
    value = os.environ.get(variable) or shutil.which(default)
    if not value:
        raise RuntimeError(f"Missing {variable}/{default}; see README_zh.md for dependencies")
    return value


def run(command, log, cwd=None):
    with Path(log).open("w") as f:
        f.write("Command: " + " ".join(command) + "\n")
        f.flush()
        cp = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, cwd=cwd,
                            stdin=subprocess.DEVNULL)
    if cp.returncode:
        tail = Path(log).read_text(errors="replace")[-6000:]
        raise RuntimeError(f"Build failed ({cp.returncode}); log={log}\n{tail}")


def extract_source(archive, build, checksum):
    destination = build / "upstream"
    marker = destination / "archive.sha256"
    if marker.is_file() and marker.read_text().strip() == checksum:
        roots = list(destination.glob("primme-*"))
        if len(roots) == 1:
            return roots[0]
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tf:
        for item in tf.getmembers():
            resolved = (destination / item.name).resolve()
            if not resolved.is_relative_to(destination.resolve()) or item.issym() or item.islnk():
                raise ValueError("Unsafe upstream archive path or link")
        tf.extractall(destination)
    marker.write_text(checksum + "\n")
    roots = list(destination.glob("primme-*"))
    if len(roots) != 1:
        raise ValueError("Expected one pinned upstream source directory")
    return roots[0]


def infer_arch():
    cp = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                        capture_output=True, text=True, stdin=subprocess.DEVNULL)
    arch = cp.stdout.splitlines()[0].strip().replace(".", "") if cp.stdout else ""
    if not re.fullmatch(r"\d{2,3}", arch):
        raise RuntimeError("Cannot infer GPU architecture; set CUDA_ARCH=80 (A800) or 89 (4090D)")
    return arch


def build(args):
    archive = args.upstream.resolve()
    pin = json.loads(args.pin.read_text())
    checksum = digest(archive)
    if checksum != pin["archive_sha256"]:
        raise ValueError("Pinned official PRIMME archive SHA256 mismatch")
    target = args.build_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    logs = target / "logs"
    logs.mkdir(exist_ok=True)
    upstream = extract_source(archive, target, checksum)
    cc, cxx, nvcc = tool("CC", "gcc"), tool("CXX", "g++"), tool("NVCC", "nvcc")
    cuda = Path(os.environ.get("CUDA_ROOT", Path(nvcc).resolve().parent.parent))
    candidates = [cuda / "include", cuda / "targets/x86_64-linux/include"]
    include = next((p for p in candidates if (p / "cublas_v2.h").exists()), None)
    if include is None:
        raise RuntimeError("CUDA_ROOT must contain CUDA development headers, including cublas_v2.h")
    bundled_blas = ROOT / "third_party/cpu_lib"
    blas_dir = Path(os.environ.get("BLAS_LIB_DIR", bundled_blas if bundled_blas.exists() else "/usr/lib/x86_64-linux-gnu"))
    blas_libraries = os.environ.get("BLAS_LIBRARIES", ":libopenblas.so.0" if (blas_dir / "libopenblas.so.0").exists() else "lapack,blas")
    libdirs = []
    for p in [cuda / "lib64", cuda / "lib", cuda / "targets/x86_64-linux/lib",
              blas_dir]:
        if p.is_dir() and p not in libdirs:
            libdirs.append(p)
    arch = args.arch or os.environ.get("CUDA_ARCH") or infer_arch()
    if not re.fullmatch(r"\d{2,3}", arch):
        raise ValueError("CUDA architecture must be numeric, e.g. 80 or 89")
    # The exact upstream compilation-unit list is used, rather than globs that
    # could accidentally include examples or alternative interface copies.
    makefile = (upstream / "src/Makefile").read_text()
    match = re.search(r"^SOURCES\s*:=\s*(.*?)^OBJS", makefile, re.M | re.S)
    if not match:
        raise RuntimeError("Pinned upstream compilation-unit list is not recognized")
    sources = re.findall(r"[\w/]+\.c", match.group(1))
    flags = ["-O2", "-std=c99", "-fPIC", "-DNDEBUG", "-DF77UNDERSCORE", "-DPRIMME_WITH_CUBLAS",
             "-I"+str(upstream / "include"), "-I"+str(upstream / "src/include"), "-I"+str(include)]
    fingerprint = {"archive_sha256": checksum, "cc": cc, "cxx": cxx, "nvcc": nvcc,
                   "flags": flags, "arch": arch, "libdirs": [str(p) for p in libdirs],
                   "blas_libraries": blas_libraries}
    marker = target / "compile_fingerprint.json"
    old = json.loads(marker.read_text()) if marker.exists() else None
    objects = []
    jobs = []
    for source in sources:
        obj = target / "objects" / (source.replace("/", "_")+".o")
        obj.parent.mkdir(exist_ok=True)
        objects.append(obj)
        same_objects = old is not None and all(old.get(k) == fingerprint.get(k) for k in
                                             ["archive_sha256", "cc", "flags"])
        if not same_objects or not obj.exists():
            jobs.append((source, obj))

    def compile_unit(job):
        source, obj = job
        run([cc, *flags, "-c", str(upstream / "src" / source), "-o", str(obj)],
            logs / (obj.stem+".log"))
        print("[build] " + source, flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(compile_unit, jobs))
    binary = target / "gpu_compare"
    link_flags = [flag for p in libdirs for flag in ["-L"+str(p), "-Xlinker", "-rpath", "-Xlinker", str(p)]]
    command = [nvcc, "-O2", "-std=c++17", "-lineinfo", "-arch=sm_"+arch,
               "-ccbin", cxx, "-I"+str(upstream / "include"), "-I"+str(include),
               str(ROOT / "gpu_compare.cu"), *[str(p) for p in objects], *link_flags,
               "-lcusparse", "-lcusolver", "-lcublas", "-lcurand", "-lcudart",
               *["-l"+name for name in blas_libraries.split(",") if name], "-lpthread", "-o", str(binary)]
    run(command, logs / "gpu_compare_link.log")
    marker.write_text(json.dumps(fingerprint, indent=2)+"\n")
    manifest = {**pin, "benchmark_source_sha256": digest(ROOT / "gpu_compare.cu"),
                "benchmark_binary_sha256": digest(binary),
                "cuda_arch": arch, "backend": "official PRIMME CUDA/cuBLAS with CSR cuSPARSE callback",
                "upstream_sources_modified": False, "source_units": sources,
                "blas_libraries": blas_libraries,
                "compiler_versions": {}}
    for name, compiler in [("cc", cc), ("nvcc", nvcc)]:
        cp = subprocess.run([compiler, "--version"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
        manifest["compiler_versions"][name] = cp.stdout[:1600]
    (target / "build_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print("[build] ready: " + str(binary), flush=True)
    return binary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=ROOT / "third_party/primme.tar.gz")
    parser.add_argument("--pin", type=Path, default=ROOT / "third_party/primme_source.json")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build")
    parser.add_argument("--arch", default=None)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 8:
        parser.error("Use 1..8 build jobs")
    build(args)
