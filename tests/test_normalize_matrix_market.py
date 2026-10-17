from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from normalize_matrix_market import normalize


class NormalizeMatrixMarketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def put_matrix(self, name: str, text: str) -> Path:
        path = self.root / name / name / f"{name}.mtx"
        path.parent.mkdir(parents=True)
        path.write_text(text, encoding="ascii")
        return path

    def test_symmetric_pattern_expands_and_preserves_raw_source(self) -> None:
        path = self.put_matrix(
            "toy",
            "%%MatrixMarket matrix coordinate pattern symmetric\n"
            "% toy\n3 3 2\n1 1\n1 3\n",
        )
        row = normalize(self.root, "toy")
        self.assertTrue((path.with_suffix(".raw.mtx")).exists())
        self.assertEqual(row["nnz"], 3)
        self.assertEqual(path.read_text(encoding="ascii").splitlines()[0],
                         "%%MatrixMarket matrix coordinate real general")
        second = normalize(self.root, "toy")
        self.assertEqual(second["nnz"], 3)

    def test_pattern_general_gets_explicit_unit_values(self) -> None:
        path = self.put_matrix(
            "toy",
            "%%MatrixMarket matrix coordinate pattern general\n"
            "% toy\n3 2 2\n1 2\n3 1\n",
        )
        row = normalize(self.root, "toy")
        self.assertEqual(row["nnz"], 2)
        lines = path.read_text(encoding="ascii").splitlines()
        self.assertEqual(lines[0], "%%MatrixMarket matrix coordinate real general")
        self.assertEqual(lines[-2:], ["1 2 1.0", "3 1 1.0"])
        self.assertTrue(path.with_suffix(".raw.mtx").exists())


if __name__ == "__main__":
    unittest.main()
