"""Guard against treating an entirely invalid scale cohort as valid."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_extended_experiments.py"
SPEC = importlib.util.spec_from_file_location("scale_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def save(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def record(rank, error, status, cost=100, residual=.9):
    return {"target_rank": rank, "power_steps": 2, "seed": 1,
            "q_orthogonality_max_error": error, "status": status,
            "gpu_pipeline_ms": cost, "rank_k_reconstruction_residual": residual}


class ScaleSummaryValidity(unittest.TestCase):
    def test_warning_only_cohort_keeps_raw_rows_and_has_no_valid_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save(root / "review_experiments/stage55_amazon_rank/stage55_amazon_rank_sweep.csv",
                 [record(64, 1e-5, "")])
            save(root / "review_experiments/stage57_webbase_parameter_sweep/stage57_webbase_parameter_sweep.csv",
                 [record(128, 1.49e-4, "WARN(qr>1e-4)"),
                  record(128, 1.26e-4, "WARN(qr>1e-4)")])
            MODULE.scale_summary(root)
            with (root / "paper_assets/extended_scale_summary.csv").open() as f:
                summaries = list(csv.DictReader(f))
            group = next(r for r in summaries if r["family"] == "webbase-1M")
            self.assertEqual(group["n_rows"], "2")
            self.assertEqual(group["n_valid"], "0")
            self.assertEqual(group["mean_rank_k_residual"], "")
            self.assertEqual(group["mean_gpu_pipeline_ms"], "")
            with (root / "paper_assets/extended_scale_rows.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 3)

    def test_numerical_check_overrides_pass_and_missing_legacy_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save(root / "review_experiments/stage55_amazon_rank/stage55_amazon_rank_sweep.csv",
                 [record(64, 1e-5, "", cost=40), record(64, 2e-4, "", cost=800)])
            save(root / "review_experiments/stage57_webbase_parameter_sweep/stage57_webbase_parameter_sweep.csv",
                 [record(256, 8e-5, "PASS", cost=60),
                  record(256, 2e-4, "PASS", cost=900)])
            MODULE.scale_summary(root)
            with (root / "paper_assets/extended_scale_summary.csv").open() as f:
                summaries = list(csv.DictReader(f))
            self.assertEqual([r["n_valid"] for r in summaries], ["1", "1"])
            self.assertEqual([float(r["mean_gpu_pipeline_ms"]) for r in summaries], [40., 60.])


if __name__ == "__main__":
    unittest.main()
