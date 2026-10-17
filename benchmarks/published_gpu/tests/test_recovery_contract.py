"""A stopped campaign's journal must never be relabelled as complete."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rebuild_summary


class RecoveryContract(unittest.TestCase):
    def test_only_a_partial_trailing_record_is_ignored(self):
        row = {"matrix": "toy", "csr_sha256": "a", "precision": "fp32", "rank": 1,
               "seed": 2, "gpu": "test_gpu", "solver": "rsvd_diagnostic_full",
               "status": "ok", "relative_residual": .4, "compute_ms": 1.,
               "u_orthogonality": 0., "v_orthogonality": 0.}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "records.jsonl").write_text(json.dumps(row)+"\n{\"matrix\":")
            (out / "matrix_manifest.json").write_text("[]")
            rebuild_summary.rebuild(out)
            result = json.loads((out / "summary_recovery.json").read_text())
            self.assertEqual(result["records_recovered"], 1)
            self.assertTrue(result["ignored_partial_journal_tail"])
            self.assertFalse(result["marks_run_complete"])
            self.assertFalse((out / "completion.json").exists())

    def test_interior_corruption_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "records.jsonl").write_text('not-json\n{"status":"ok"}\n')
            (out / "matrix_manifest.json").write_text("[]")
            with self.assertRaises(json.JSONDecodeError):
                rebuild_summary.rebuild(out)


if __name__ == "__main__":
    unittest.main()
