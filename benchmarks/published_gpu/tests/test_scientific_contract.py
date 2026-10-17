"""Independent checks of the quantities used to match two GPU solvers."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy import sparse
from scipy.io import mmwrite

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import benchmark_support as support
except ImportError:
    support = None


class ScientificContract(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(support, "The common input and independent validator are missing")

    def test_residual_matches_direct_reconstruction_for_nonorthogonal_factors(self):
        # Energy-only singular-value formulas would give the wrong answer here.
        a = sparse.csr_matrix([[2., -1., 0.], [0., 3., 1.]])
        u = np.array([[1., 0.], [0., 2.]])
        sigma = np.array([1.5, .5])
        v = np.array([[1., .2], [0., 1.], [.1, .3]])
        expected = np.linalg.norm(a.toarray() - (u * sigma) @ v.T) / np.linalg.norm(a.toarray())
        got = support.validate_factors(a, u, sigma, v)
        self.assertAlmostEqual(got["relative_residual"], expected, places=13)
        self.assertGreater(got["u_orthogonality"], 1.)

    def test_nearly_exact_factors_use_direct_fallback_instead_of_cancellation(self):
        a = sparse.diags([3., 2., 1.]).tocsr()
        u = np.eye(3)
        sigma = np.array([3., 2., 1. + 1e-10])
        got = support.validate_factors(a, u, sigma, np.eye(3))
        self.assertEqual(got["validation_method"], "fp64_block_reconstruction")
        self.assertAlmostEqual(got["relative_residual"], 1e-10 / np.sqrt(14.), delta=1e-17)

    def test_malformed_or_nonfinite_factors_are_rejected(self):
        a = sparse.eye(4, format="csr")
        for u, s, v in [(np.eye(3), np.ones(3), np.eye(4, 3)),
                        (np.eye(4, 2), np.array([1., np.nan]), np.eye(4, 2))]:
            with self.assertRaises(ValueError):
                support.validate_factors(a, u, s, v)

    def test_csr_roundtrip_sums_duplicates_and_preserves_rectangular_input(self):
        a = sparse.coo_matrix(([1., 2., -3., 4.], ([0, 0, 1, 2], [3, 3, 0, 1])), shape=(3, 5))
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "rect.mtx"
            dst = Path(d) / "rect.csr"
            mmwrite(src, a)
            meta = support.prepare_matrix(src, dst)
            got = support.read_csr(dst, "fp32")
            np.testing.assert_array_equal(got.toarray(), a.toarray())
            self.assertEqual(meta["nnz"], 3)
            self.assertEqual(got.shape, (3, 5))

    def test_fp32_validator_uses_the_values_actually_given_to_solver(self):
        a = sparse.csr_matrix([[1.00000006, 0.], [0., 2.]])
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "a.mtx"
            dst = Path(d) / "a.csr"
            mmwrite(src, a)
            support.prepare_matrix(src, dst)
            got = support.read_csr(dst, "fp32")
            np.testing.assert_array_equal(got.data, a.data.astype(np.float32).astype(np.float64))

    def test_factor_roundtrip_keeps_column_orientation(self):
        u = np.arange(8., dtype=np.float32).reshape(4, 2)
        v = np.arange(12., dtype=np.float32).reshape(6, 2)
        s = np.array([3., 1.], dtype=np.float32)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "factors.bin"
            support.write_factors(p, u, s, v, "fp32")
            u2, s2, v2 = support.read_factors(p)
            for left, right in [(u,u2),(s,s2),(v,v2)]:
                np.testing.assert_array_equal(left, right)

    def test_independent_cpu_reference_matches_a_known_diagonal_spectrum(self):
        import cpu_crosscheck
        import json
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            source, canonical = directory / "diagonal.mtx", directory / "diagonal.csr"
            mmwrite(source, sparse.diags([6., 5., 4., 3., 2., 1.]).tocsr())
            support.prepare_matrix(source, canonical)
            result = directory / "cpu.json"
            cpu_crosscheck.worker(canonical, "fp32", [2, 4], result)
            payload = json.loads(result.read_text())
            self.assertEqual(payload["solver"], "scipy.sparse.linalg.svds_propack")
            for check in payload["rank_checks"]:
                rank = check["rank"]
                self.assertEqual(check["status"], "ok")
                np.testing.assert_allclose(check["singular_values"], np.arange(6., 6.-rank, -1.), atol=1e-9)
                expected = np.linalg.norm(np.arange(1., 7.-rank))/np.linalg.norm(np.arange(1., 7.))
                self.assertAlmostEqual(check["relative_residual"], expected)


if __name__ == "__main__":
    unittest.main()
