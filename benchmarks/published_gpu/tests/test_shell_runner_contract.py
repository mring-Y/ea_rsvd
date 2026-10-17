"""Check that configuration failures are distinguished from an aborted run."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(os.name != "nt" and shutil.which("bash"), "requires Linux bash")
class ReviewRunnerContract(unittest.TestCase):
    def run_wrapper(self, completion, first_exit):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = Path(__file__).resolve().parents[1] / "run_autodl_review.sh"
            shutil.copyfile(source, root / source.name)
            stub = root / "run_autodl_all.sh"
            stub.write_text('''#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "${PROFILE}" >>"${ROOT}/calls.txt"
if [[ "${PROFILE}" == paper ]]; then
  if [[ -n "${TEST_COMPLETION}" ]]; then
    mkdir -p "${ROOT}/results/${RUN_ID}/campaign"
    printf '%s\\n' "${TEST_COMPLETION}" >"${ROOT}/results/${RUN_ID}/campaign/completion.json"
  fi
  exit "${TEST_FIRST_EXIT}"
fi
exit 0
''')
            env = dict(os.environ, PYTHON=sys.executable, RUN_ID="runner_contract",
                       TEST_COMPLETION="" if completion is None else json.dumps(completion),
                       TEST_FIRST_EXIT=str(first_exit))
            process = subprocess.run(["bash", str(root / source.name)], env=env,
                                     stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                     timeout=10)
            calls = (root / "calls.txt").read_text().splitlines()
            return process.returncode, calls

    def test_completed_configuration_failures_continue(self):
        self.assertEqual(self.run_wrapper({"completed": True}, 2), (2, ["paper", "quality"]))

    def test_exit_two_without_completion_stops(self):
        self.assertEqual(self.run_wrapper(None, 2), (2, ["paper"]))

    def test_incomplete_campaign_stops(self):
        self.assertEqual(self.run_wrapper({"completed": False}, 2), (2, ["paper"]))

    def test_other_error_stops(self):
        self.assertEqual(self.run_wrapper(None, 1), (1, ["paper"]))


if __name__ == "__main__":
    unittest.main()
