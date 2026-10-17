"""Recover summary tables from the append-only journal of a stopped run."""
import argparse
import json
from pathlib import Path

from campaign import summarize


def rebuild(directory: Path):
    journal = directory / "records.jsonl"
    rows = []
    lines = journal.read_text().splitlines()
    ignored_partial_tail = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A kill may interrupt the last journal append.  Interior
            # corruption is never treated as an incomplete trailing record.
            if index != len(lines)-1:
                raise
            ignored_partial_tail = True
    manifest_file = directory / "matrix_manifest.json"
    if not manifest_file.exists():
        raise SystemExit("matrix_manifest.json is missing; preserve the original run for inspection")
    manifest = json.loads(manifest_file.read_text())
    summarize(directory, rows, manifest)
    note = {"records_recovered": len(rows),
            "ignored_partial_journal_tail": ignored_partial_tail,
            "original_completion_present": (directory / "completion.json").exists(),
            "marks_run_complete": False}
    (directory / "summary_recovery.json").write_text(json.dumps(note, indent=2)+"\n")
    print(json.dumps(note))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_directory", type=Path)
    rebuild(parser.parse_args().campaign_directory)
