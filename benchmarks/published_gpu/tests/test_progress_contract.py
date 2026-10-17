"""Read-only status checks for ongoing or interrupted background jobs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from watch_progress import describe


class ProgressContract(unittest.TestCase):
    def test_existing_run_without_state_keeps_partial_records_visible(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "logs").mkdir()
            (root / "campaign").mkdir()
            (root / "logs/campaign.log").write_text("[run 2] west2021_fp32\n")
            (root / "campaign/records.jsonl").write_text(json.dumps({"status": "ok"})+'\n{"status":')
            message, finished = describe(root)
            self.assertIn("configurations_saved=1", message)
            self.assertIn("[run 2]", message)
            self.assertFalse(finished)

    def test_stage_and_exit_two_do_not_erase_failure_information(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runner_state.tsv").write_text("100\tbuild\n")
            message, finished = describe(root, now=125)
            self.assertIn("stage=build stage_elapsed=25s", message)
            self.assertFalse(finished)
            (root / "runner_completion.json").write_text('{"runner_exit_code":2}')
            message, finished = describe(root, now=125)
            self.assertIn("finished exit=2", message)
            self.assertTrue(finished)


if __name__ == "__main__":
    unittest.main()
