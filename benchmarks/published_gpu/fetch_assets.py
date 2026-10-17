"""Fetch pinned upstream source and official matrix inputs for a source clone.

The standalone AutoDL bundle already contains these assets.  A failed
matrix download is logged and leaves that matrix unavailable; it does not
prevent the later campaign from trying the other matrices.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from verify_assets import sha256

ROOT = Path(__file__).resolve().parent


def download(url, destination, log):
    partial = Path(str(destination)+".part")
    command = ["curl", "-L", "--fail", "--show-error", "-C", "-", "--retry", "5",
               "--retry-delay", "5", "--connect-timeout", "20", "--max-time", "300",
               "-o", str(partial), url]
    with log.open("a") as stream:
        stream.write(url+"\n")
        stream.flush()
        try:
            cp = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, timeout=900)
        except (OSError, subprocess.TimeoutExpired) as exc:
            stream.write(repr(exc)+"\n")
            return False
    if cp.returncode:
        return False
    partial.replace(destination)
    return True


def fetch(source_only=False):
    logs = ROOT / "asset_download_logs"
    logs.mkdir(exist_ok=True)
    pin = json.loads((ROOT / "third_party/primme_source.json").read_text())
    archive = ROOT / "third_party/primme.tar.gz"
    if not archive.exists():
        url = "https://codeload.github.com/primme/primme/tar.gz/"+pin["commit"]
        if not download(url, archive, logs / "primme.log"):
            raise SystemExit("Pinned PRIMME download failed; logs/partial file retained. Retry this command or copy the standalone bundle asset.")
    if sha256(archive) != pin["archive_sha256"]:
        raise SystemExit("Pinned PRIMME source hash mismatch; archive was not accepted")
    print("Official PRIMME archive verified", flush=True)
    if source_only:
        return
    matrices = json.loads((ROOT / "data/matrix_sources.json").read_text())
    for row in matrices:
        target = ROOT / "data" / row["file"]
        if not target.exists():
            compressed = ROOT / "data" / (row["name"]+".tar.gz")
            if not compressed.exists() and not download(row["archive"], compressed,
                                                         logs / (row["name"]+".log")):
                print("[unavailable] "+row["name"]+": download failed; other matrices continue", flush=True)
                continue
            try:
                with tarfile.open(compressed) as tf:
                    matches = [m for m in tf.getmembers() if m.isfile() and Path(m.name).name == row["file"]]
                    if len(matches) != 1:
                        raise ValueError("Expected exactly one requested Matrix Market input")
                    with tf.extractfile(matches[0]) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
            except (OSError, ValueError, tarfile.TarError) as exc:
                print("[unavailable] "+row["name"]+": "+repr(exc), flush=True)
                target.unlink(missing_ok=True)
                continue
        if sha256(target) != row["sha256"]:
            raise SystemExit("Supplied matrix hash mismatch: "+row["name"])
        print("[verified] "+row["name"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-only", action="store_true")
    fetch(parser.parse_args().source_only)
