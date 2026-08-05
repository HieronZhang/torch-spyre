# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Re-measure every configuration in the cost-model database, on this machine.

The database (``tools/cost_model/sweep_records.json``) holds one record per measured
kernel: its shape, core count, tiling, measured time and the features the model was
scored against. Hardware, the compiler or the model can all move underneath it, so
this script re-runs the measurements and folds the fresh times back in.

It takes the configuration list FROM the database rather than hard-coding a sweep, so
it stays correct as the database grows: whatever is in there gets re-measured.

    python3 docs/source/user_guide/examples/run_cost_model_sweep.py            # everything
    python3 ... run_cost_model_sweep.py --op softmax_row_tiling               # one op
    python3 ... run_cost_model_sweep.py --dry-run                             # just list

Each run appends to one timestamped log, and the log is parsed back into the database
at the end (``--no-parse`` to skip). Re-score afterwards with::

    python3 tools/cost_model/eval_model.py

A configuration that this build cannot compile prints a FAILED summary and is skipped
by the parser, so one bad shape never costs the rest of the sweep.
"""

import argparse
import collections
import json
import os
import subprocess
import sys

import regex as re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
_TOOLS = os.path.join(_ROOT, "tools", "cost_model")
sys.path.insert(0, _TOOLS)
from records import records_path  # noqa: E402

_RECORDS = records_path()
_HARNESS = os.path.join(_HERE, "profile_ops.py")

# Needs an opt-in layout preference that is not part of this feature -- those rows are
# evidence for the report's layout figure and are excluded from scoring anyway.
_SKIP_OPS = {"bmm_layout"}


def _shape_from_label(label):
    """Pull M/K/N and batch out of a record label.

    Two formats are in use: ``M=2048 K=2048 N=2048`` and a bare ``2048x2048x2048``.
    Only matmul-family ops carry these; everything else is fully described by the
    rows/cols/cores/tiles fields.
    """
    out = {}
    m = re.search(r"\bB=(\d+)", label or "")
    if m:
        out["BENCH_B"] = m[1]
    m = re.search(r"\bM=(\d+)\s+K=(\d+)\s+N=(\d+)", label or "")
    if not m:
        m = re.search(r"\b(\d+)x(\d+)x(\d+)\b", label or "")
    if m:
        out["BENCH_ROWS"], out["BENCH_COLS"], out["BENCH_N"] = m[1], m[2], m[3]
    return out


def _configs(records, only_op=None):
    """Distinct run configurations, in a stable order."""
    seen, out = set(), []
    for r in records:
        op = r.get("op")
        if not op or op in _SKIP_OPS or (only_op and op != only_op):
            continue
        env = {"BENCH_OP": op}
        for field, var in (
            ("rows", "BENCH_ROWS"),
            ("cols", "BENCH_COLS"),
            ("tiles", "BENCH_TILES"),
            ("cores", "SENCORES"),
        ):
            if isinstance(r.get(field), int):
                env[var] = str(r[field])
        env.update(_shape_from_label(r.get("label", "")))
        key = tuple(sorted(env.items()))
        if key not in seen:
            seen.add(key)
            out.append(env)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", default="", help="re-measure only this op")
    ap.add_argument("--limit", type=int, default=0, help="stop after N configurations")
    ap.add_argument("--reps", default="7", help="BENCH_REPS per configuration")
    ap.add_argument("--dry-run", action="store_true", help="list, do not run")
    ap.add_argument("--no-parse", action="store_true", help="skip the database update")
    ap.add_argument("--out", default="", help="log file (default: timestamped)")
    args = ap.parse_args()

    with open(_RECORDS, encoding="utf-8") as fh:
        records = json.load(fh)["records"]
    cfgs = _configs(records, args.op or None)
    if args.limit:
        cfgs = cfgs[: args.limit]

    by_op = collections.Counter(c["BENCH_OP"] for c in cfgs)
    print(f"{len(cfgs)} configurations over {len(by_op)} ops")
    for op, n in by_op.most_common():
        print(f"    {op:<26} {n}")
    if _SKIP_OPS:
        print(f"  skipped (needs another feature): {', '.join(sorted(_SKIP_OPS))}")
    if args.dry_run:
        return 0

    log = args.out or os.path.join(_HERE, "cost_model_sweep.log")
    print(f"\nlogging to {log}\n")
    failed = 0
    with open(log, "w", encoding="utf-8") as fh:
        for i, env in enumerate(cfgs, 1):
            tag = " ".join(f"{k}={v}" for k, v in sorted(env.items()))
            print(f"[{i}/{len(cfgs)}] {tag}", flush=True)
            fh.write(f"\n=== {tag} ===\n")
            fh.flush()
            run_env = dict(os.environ, BENCH_REPS=args.reps, SPYRE_DUMP_COST="1", **env)
            p = subprocess.run(
                [sys.executable, _HARNESS],
                env=run_env,
                capture_output=True,
                text=True,
                check=False,
            )
            fh.write(p.stdout + p.stderr)
            fh.flush()
            if "SUMMARY" not in p.stdout:
                failed += 1
                print("      no SUMMARY -- see the log", flush=True)

    print(f"\n{len(cfgs) - failed}/{len(cfgs)} produced a measurement -> {log}")
    if args.no_parse:
        print("database not updated (--no-parse)")
        return 0
    print("folding into the database...")
    subprocess.run(
        [sys.executable, os.path.join(_TOOLS, "parse_sweep_logs.py"), log],
        check=False,
    )
    print("\nre-score with:  python3 tools/cost_model/eval_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
