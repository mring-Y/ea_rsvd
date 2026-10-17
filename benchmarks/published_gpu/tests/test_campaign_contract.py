"""Reporting must preserve failures and only compare the same input/rank."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import campaign
except ImportError:
    campaign = None


def row(solver, residual, cost, **overrides):
    return {"matrix": "toy", "csr_sha256": "aaa", "precision": "fp32", "rank": 4,
            "seed": 7, "solver": solver, "status": "ok", "relative_residual": residual,
            "compute_ms": cost, "u_orthogonality": 1e-6, "v_orthogonality": 1e-6,
            **overrides}


class CampaignContract(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(campaign, "The matched campaign and summary module is missing")

    def test_failed_solver_is_not_silently_dropped_or_counted_as_success(self):
        rows = [row("rsvd_diagnostic_full", .4, 3), row("primme_svds_cuda", .2, 1, status="timeout")]
        result = campaign.make_quality_comparisons(rows, [.5])
        self.assertEqual(result[0]["primme_status"], "no_valid_result")
        self.assertIsNone(result[0]["speedup_rsvd_over_primme"])
        self.assertEqual(result[0]["primme_attempts"], 1)
        self.assertEqual(result[0]["primme_failures"], 1)

    def test_different_ranks_and_input_hashes_are_never_paired(self):
        rows = [row("rsvd_diagnostic_full", .4, 3),
                row("primme_svds_cuda", .4, 1, rank=8),
                row("primme_svds_cuda", .4, 1, csr_sha256="bbb")]
        result = campaign.make_quality_comparisons(rows, [.5])
        self.assertTrue(all(x["speedup_rsvd_over_primme"] is None for x in result))

    def test_configuration_must_meet_the_actual_residual_target(self):
        rows = [row("rsvd_diagnostic_full", .6, 1), row("rsvd_diagnostic_full", .4, 4),
                row("primme_svds_cuda", .45, 2)]
        result = campaign.make_quality_comparisons(rows, [.5])[0]
        self.assertEqual(result["rsvd_ms"], 4)
        self.assertEqual(result["speedup_rsvd_over_primme"], .5)

    def test_bad_orthogonality_invalidates_a_fast_candidate(self):
        rows = [row("rsvd_diagnostic_full", .4, 1, u_orthogonality=.1),
                row("primme_svds_cuda", .4, 2)]
        result = campaign.make_quality_comparisons(rows, [.5])[0]
        self.assertEqual(result["rsvd_status"], "no_valid_result")

    def test_near_best_quality_uses_tight_reference_and_keeps_rank_fixed(self):
        rows = [row("rsvd_diagnostic_full", .44, 1), row("rsvd_diagnostic_full", .401, 4),
                row("primme_svds_cuda", .4, 2, eps=1e-6)]
        result = campaign.make_near_best_comparisons(rows, [1.01])[0]
        self.assertAlmostEqual(result["target_residual"], .404)
        self.assertEqual(result["rsvd_ms"], 4)
        self.assertEqual(result["reference_kind"], "validated_primme_eps_le_1e-5")

    def test_a_nonfinal_repeat_with_bad_vectors_invalidates_the_configuration(self):
        import tempfile
        import json
        import numpy as np
        from scipy import sparse
        from benchmark_support import write_factors
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "raw").mkdir()
            stem = out / "raw/toy"
            report = {"repeats": 2, "status": "ok", "compute_ms": 1., "rho_k_energy": 1/np.sqrt(3)}
            Path(str(stem)+".json").write_text(json.dumps(report))
            u, v, s = np.eye(3, 2), np.eye(3, 2), np.ones(2)
            write_factors(Path(str(stem)+".bin"), u, s, v, "fp32")
            write_factors(Path(str(stem)+".bin.repeat_0.bin"), 2*u, s, v, "fp32")
            write_factors(Path(str(stem)+".bin.repeat_1.bin"), u, s, v, "fp32")
            cp = campaign.validate_record(stem, sparse.eye(3,format="csr"), {"rank": 2}, report)
            self.assertEqual(cp["status"], "invalid_orthogonality")
            self.assertEqual(cp["validation_scope"], "all_measured_repeats_worst_quality")
            self.assertEqual(len(cp["repeat_validations"]), 2)
            self.assertAlmostEqual(cp["energy_residual_difference"], 0.)
            self.assertAlmostEqual(cp["triplet_residual_max_relative_fro"], np.sqrt(2/3))

    def test_missing_per_repeat_factors_cannot_claim_all_repeats_are_valid(self):
        import tempfile
        import numpy as np
        from scipy import sparse
        from benchmark_support import write_factors
        with tempfile.TemporaryDirectory() as d:
            stem = Path(d) / "toy"
            u, v, s = np.eye(3, 2), np.eye(3, 2), np.ones(2)
            write_factors(Path(str(stem)+".bin"), u, s, v, "fp32")
            with self.assertRaisesRegex(ValueError, "per-repeat"):
                campaign.validate_record(stem, sparse.eye(3, format="csr"), {"rank": 2},
                                         {"repeats": 2, "status": "ok", "compute_ms": 1.})

    def test_calibration_cost_includes_failed_candidates_and_validation(self):
        import tempfile
        import json
        import analyse_results
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            rows = [row("rsvd_diagnostic_full", .4, 1, process_wall_ms=10, validation_ms=2),
                    row("rsvd_diagnostic_full", .4, 1, status="timeout", process_wall_ms=30)]
            (out / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows)+"\n")
            (out / "environment.json").write_text("{}")
            analyse_results.analyse(out)
            import csv
            with (out / "calibration_costs.csv").open() as stream:
                cost = list(csv.DictReader(stream))[0]
            self.assertEqual(float(cost["total_calibration_wall_ms"]), 42.)
            self.assertEqual(int(cost["failed_configurations"]), 1)

    def test_joint_quality_rejects_fast_but_inaccurate_singular_triplets(self):
        import analyse_results
        rows = [row("rsvd_diagnostic_full", .401, 1, triplet_residual_max=.1, matrix_frobenius=1),
                row("rsvd_diagnostic_full", .401, 4, triplet_residual_max=1e-5, matrix_frobenius=1),
                row("primme_svds_cuda", .4, 2, eps=1e-6, triplet_residual_max=1e-6, matrix_frobenius=1)]
        result = analyse_results.joint_quality_comparisons(rows, (1e-3,))[0]
        self.assertEqual(result["rsvd_ms"], 4)
        self.assertEqual(result["speedup_rsvd_over_primme"], .5)

    def test_stale_gpu_source_is_rejected_before_creating_measurement_output(self):
        import json
        import tempfile
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            binary = root / "gpu_compare"
            binary.write_bytes(b"unrelated executable")
            (root / "build_manifest.json").write_text(json.dumps({"benchmark_source_sha256": "stale"}))
            output = root / "new_measurement"
            with self.assertRaisesRegex(ValueError, "source differs"):
                campaign.campaign(SimpleNamespace(binary=binary, out=output))
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
