"""Show progress for a new or already-running comparison without modifying it."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def tail(path, limit=12000):
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell()-limit))
        return stream.read().decode("utf-8", errors="replace").splitlines()


def describe(run, now=None):
    now = time.time() if now is None else now
    if run is None or not run.is_dir():
        return "no results directory yet; check the launch log", False
    logs = [p for p in (run / "logs").glob("*.log") if p.is_file()]
    newest = max(logs, key=lambda p: p.stat().st_mtime) if logs else None
    state = run / "runner_state.tsv"
    stage = newest.stem if newest else "starting"
    started = run.stat().st_mtime
    if state.is_file():
        pieces = state.read_text().strip().split("\t", 1)
        if len(pieces) == 2:
            try:
                started, stage = float(pieces[0]), pieces[1]
            except ValueError:
                pass
    journal = run / "campaign/records.jsonl"
    count = 0
    statuses = {}
    if journal.is_file():
        for line in journal.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1
            key = row.get("status", "unknown")
            statuses[key] = statuses.get(key, 0)+1
    campaign = run / "logs/campaign.log"
    active = ""
    if campaign.is_file():
        starts = [line for line in tail(campaign) if line.startswith("[run ")]
        if starts:
            active = " last="+starts[-1]
    done = run / "runner_completion.json"
    completed = False
    if done.is_file():
        try:
            payload = json.loads(done.read_text())
            stage = "finished exit="+str(payload["runner_exit_code"])
            completed = True
        except (ValueError, KeyError):
            pass
    age = f" last_log_change={max(0, now-newest.stat().st_mtime):.0f}s" if newest else ""
    detail = f" detail={newest.relative_to(run)}" if newest else ""
    message = (f"run={run.name} stage={stage} stage_elapsed={max(0, now-started):.0f}s"
               f" configurations_saved={count} statuses={json.dumps(statuses, sort_keys=True)}"
               +age+active+detail)
    return message, completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, help="default: most recently modified result directory")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=20)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")
    while True:
        if args.parent_pid is not None:
            try:
                os.kill(args.parent_pid, 0)
            except ProcessLookupError:
                return
        directories = [p for p in (ROOT / "results").glob("*") if p.is_dir()]
        run = args.run or max(directories, key=lambda p: p.stat().st_mtime, default=None)
        message, completed = describe(run)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"[heartbeat {stamp}] {message}", flush=True)
        if not args.watch or (args.run is not None and completed):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
