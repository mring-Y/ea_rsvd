"""Verify supplied inputs and pinned source before a background campaign."""
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    pin = json.loads((ROOT / "third_party/primme_source.json").read_text())
    if sha256(ROOT / "third_party/primme.tar.gz") != pin["archive_sha256"]:
        raise SystemExit("FAIL: official PRIMME source archive hash mismatch")
    print("Official PRIMME source hash verified, commit="+pin["commit"])
    for row in json.loads((ROOT / "data/matrix_sources.json").read_text()):
        matrix = ROOT / "data" / row["file"]
        if not matrix.exists():
            print("Public matrix not supplied; campaign will download or record unavailable: "+row["name"])
            continue
        if sha256(matrix) != row["sha256"]:
            raise SystemExit("FAIL: supplied matrix hash mismatch: "+row["name"])
        print("Public matrix verified: "+row["name"])
    cpu_manifest = ROOT / "third_party/cpu_libraries.json"
    if cpu_manifest.exists() and (ROOT / "third_party/cpu_lib").exists():
        for row in json.loads(cpu_manifest.read_text())["libraries"]:
            if sha256(ROOT / "third_party/cpu_lib" / row["name"]) != row["sha256"]:
                raise SystemExit("FAIL: user-level numerical library hash mismatch")
        print("CPU numerical dependencies verified")
    else:
        print("Bundled CPU numerical libraries absent; provide installed BLAS/LAPACK for compilation")
