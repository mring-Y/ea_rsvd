"""Check that a larger budget never silently changes the comparison cohort."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('budget_cohort', ROOT / 'budget_cohort.py')
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


class BudgetTests(unittest.TestCase):
    def test_ended_path_carries_actual_result_and_cost(self):
        rows = [dict(matrix='ended', seed='1', budget_step=1, endpoint_rho_k=.8,
                     best_seen_rho_k=.8, cumulative_gpu_ms=2),
                dict(matrix='continuing', seed='1', budget_step=1, endpoint_rho_k=.6,
                     best_seen_rho_k=.6, cumulative_gpu_ms=3),
                dict(matrix='continuing', seed='1', budget_step=2, endpoint_rho_k=.2,
                     best_seen_rho_k=.2, cumulative_gpu_ms=7)]
        result = budget.summarize(rows)
        self.assertEqual([r['paths'] for r in result], [2, 2])
        self.assertEqual(result[1]['advancing_paths'], 1)
        self.assertAlmostEqual(result[1]['mean_endpoint_rho_k'], .5)
        self.assertAlmostEqual(result[1]['mean_cumulative_gpu_ms'], 4.5)

    def test_missing_and_duplicate_steps_are_rejected(self):
        row = dict(matrix='a', seed='1', budget_step=1, endpoint_rho_k=.5,
                   best_seen_rho_k=.5, cumulative_gpu_ms=1)
        for second in [dict(row), dict(row, budget_step=3)]:
            with self.assertRaises(ValueError):
                budget.summarize([row, second])

    def test_measured_cohort_reproduces_six_state_budget_cap(self):
        rows = budget.read(ROOT / 'evidence/submission/budget_curve/budget_curve_rows.csv')
        summary = budget.summarize(rows)
        self.assertEqual(len(summary), 6)
        self.assertEqual([r['paths'] for r in summary], [25] * 6)
        self.assertEqual([r['advancing_paths'] for r in summary], [25] * 5 + [10])
        self.assertAlmostEqual(summary[-1]['mean_endpoint_rho_k'], .57187514, places=8)
        self.assertAlmostEqual(summary[-1]['mean_cumulative_gpu_ms'], 807.28228, places=5)


if __name__ == '__main__':
    unittest.main()
