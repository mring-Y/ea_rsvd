"""Release archives must keep build helpers and the evidence they reference."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_release", REPOSITORY / "scripts/prepare_release.py")
pack = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pack)


class ReleasePackagingTests(unittest.TestCase):
    def test_build_named_sources_and_evidence_are_kept_while_runtime_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = {
                "LICENSE",
                "benchmarks/published_gpu/build_gpu.py",
                "benchmarks/manuscript_completion/evidence/grid/build_manifest.json",
            }
            excluded = {
                ".git/config",
                "build_local/compile.json",
                "benchmarks/published_gpu/build_old/compile.json",
                "scripts/__pycache__/cache.pyc",
                "benchmarks/published_gpu/results/measurement.csv",
                "data/suitesparse/example.mtx",
            }
            for name in expected | excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            got = {p.relative_to(root).as_posix() for p in pack.collect_sources(root)}
            self.assertEqual(got, expected)

    def test_supplied_root_and_manifests_preserve_source_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            root.mkdir()
            (root / "LICENSE").write_text("MIT License\n", encoding="utf-8")
            evidence = root / "benchmarks/manuscript_completion/evidence/data.csv"
            evidence.parent.mkdir(parents=True)
            evidence.write_bytes(b"rank,residual\r\n16,0.25\r\n")
            output = Path(temporary) / "staged"
            files = pack.copy_release_tree(root, pack.collect_sources(root), output)
            self.assertEqual(evidence.read_bytes(), (output / evidence.relative_to(root)).read_bytes())
            manifest = json.loads((output / "release_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["files"]), 2)
            for line in (output / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
                digest, name = line.split("  ", 1)
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)
            archive = Path(temporary) / "release.zip"
            duplicate = Path(temporary) / "duplicate.zip"
            pack.write_zip(output, files, archive)
            pack.write_zip(output, files, duplicate)
            pack.verify_zip(archive, output, files)
            self.assertEqual(archive.read_bytes(), duplicate.read_bytes())

    def test_privacy_scan_rejects_plain_and_json_escaped_user_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "README.md"
            local_path = "Q:" + "\\" + "Users" + "\\" + "example" + "\\" + "work"
            for text in (local_path, json.dumps({"source": local_path})):
                source.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "local Windows user path"):
                    pack.audit_text_files(root, [source])

    def test_all_existing_evidence_manifest_members_are_in_the_release(self):
        supplement = REPOSITORY / "benchmarks/manuscript_completion"
        manifest = json.loads((supplement / "file_manifest.json").read_text(encoding="utf-8"))
        selected = {p.relative_to(REPOSITORY).as_posix() for p in pack.collect_sources(REPOSITORY)}
        for name in manifest:
            relative = (supplement / name).relative_to(REPOSITORY).as_posix()
            self.assertIn(relative, selected)


if __name__ == "__main__":
    unittest.main()
