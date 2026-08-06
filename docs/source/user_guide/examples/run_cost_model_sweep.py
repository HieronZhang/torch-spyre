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
    python3 ... run_cost_model_sweep.py --limit 20                            # a timed pilot
    python3 ... run_cost_model_sweep.py --resume <log>                        # continue one

A CONFIGURATION IS MORE THAN A SHAPE. A forced work division, an operand layout or a
flash-attention tiling is part of what was measured, and re-running the shape without it
measures a different kernel under the old kernel's label. ``_env_from_record`` therefore
rebuilds the *whole* environment from each record -- structured fields first, the label
as a fallback -- and a record it cannot rebuild is reported and skipped rather than
approximated. See ``_RECONSTRUCTED`` for the list.

Each run appends to one timestamped log, and the log is parsed back into the database
at the end (``--no-parse`` to skip). Re-score afterwards with::

    python3 tools/cost_model/eval_model.py --build <sha>

The log opens with a provenance header (git sha, torch version, host, date). The parser
turns that sha into ``model_sha`` on every record, which is what lets a later re-score
separate this sweep's measurements from an earlier build's. Measurements from different
builds must not be pooled: kernel performance moves as the compiler develops, and during
this project a single shape moved by more than 2x between builds.

A configuration that this build cannot compile prints a FAILED summary and is skipped
by the parser, so one bad shape never costs the rest of the sweep.
"""

import argparse
import collections
import json
import os
import socket
import subprocess
import sys
import time

import regex as re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
_TOOLS = os.path.join(_ROOT, "tools", "cost_model")
sys.path.insert(0, _TOOLS)
from records import records_path  # noqa: E402

_HARNESS = os.path.join(_HERE, "profile_ops.py")

#: What ``_env_from_record`` rebuilds beyond the shape, and where each piece comes from.
#: Anything not on this list is either a plain shape/core/tile config or unreproducible.
_RECONSTRUCTED = """\
  forced work division   WD_M/WD_N/WD_K (+WD_B)  <- split_forced, or the label
  operand device layout  WD_LAYOUT_A/WD_LAYOUT_B <- layout_a/layout_b
  flash attention        FA_H/FA_LQ/FA_LK/FA_D/FA_*_TILES/FA_WD  <- the label
  transpose_outer middle TO_MID                  <- the label
  scratchpad planning    LX_PLANNING             <- the lx field
  matmul layout flag     SPYRE_MATMUL_PREFERRED_LAYOUT <- a `flag[...]` label prefix
"""

_LBL_MKN = re.compile(r"\bM=(\d+)\s+K=(\d+)\s+N=(\d+)")
_LBL_MEQN = re.compile(r"\bM=N=(\d+)\s+K=(\d+)")  # square mm: N is not spelled out
_LBL_XSHAPE = re.compile(r"\b(\d+)x(\d+)x(\d+)\b")
_LBL_B = re.compile(r"\bB=(\d+)")
# `flag[on] mmwd ...` marks a run taken with the matmul layout preference set; without it
# the re-run measures the default layout and files the result under the flagged label.
_LBL_FLAG = re.compile(r"^flag\[(on|off|output)\]")
# Three spellings of a forced matmul split are in the database: `split m=4 n=8 k=1`,
# a bare `m=4 n=8 k=1`, and the oldest `split 4x8` (m x n, k implicitly 1).
_LBL_SPLIT_KV = re.compile(r"\b(?:split\s+)?b=(\d+)\s+m=(\d+)\s+n=(\d+)\s+k=(\d+)")
_LBL_SPLIT_MNK = re.compile(r"\b(?:split\s+)?m=(\d+)\s+n=(\d+)\s+k=(\d+)")
_LBL_SPLIT_MXN = re.compile(r"\bsplit\s+(\d+)x(\d+)\b")
_LBL_TO_MID = re.compile(r"\b[mM]=(\d+)")
_LBL_FA = re.compile(
    r"\bH=(\d+)\s+Lq=(\d+)\s+Lk=(\d+)(?:\s+D=(\d+))?"
    r"(?:\s+htiles=(\d+))?(?:\s+qtiles=(\d+))?(?:\s+ktiles=(\d+))?"
)
_LBL_FA_WD = re.compile(r"\bwd=(\S+)")
_RUN_HDR = re.compile(r"^===\s+(.*?)\s+===\s*$")


def _shape_env(r):
    """M/K/N and the batch size, from the record's fields or its label.

    Matmul-family records carry these as parsed fields (``M``/``K``/``N``/``B``); older
    rows only have them in the label, in one of two spellings.
    """
    out, label = {}, r.get("label") or ""
    for field, var in (("M", "BENCH_ROWS"), ("K", "BENCH_COLS"), ("N", "BENCH_N")):
        if isinstance(r.get(field), int):
            out[var] = str(r[field])
    if isinstance(r.get("B"), int):
        out["BENCH_B"] = str(r["B"])

    if not out.get("BENCH_N"):
        if m := (_LBL_MKN.search(label) or _LBL_XSHAPE.search(label)):
            out["BENCH_ROWS"], out["BENCH_COLS"], out["BENCH_N"] = m[1], m[2], m[3]
        elif m := _LBL_MEQN.search(label):
            out["BENCH_ROWS"], out["BENCH_COLS"], out["BENCH_N"] = m[1], m[2], m[1]
    if "BENCH_B" not in out and (m := _LBL_B.search(label)):
        # Leftmost match on purpose: `lay B=2 ... A=0,1,2 B=0,1,2` spells the batch size
        # and the second operand's layout with the same letter, and the batch comes first.
        out["BENCH_B"] = m[1]
    return out


def _split_env(r):
    """A forced work division, as WD_B/WD_M/WD_N/WD_K.

    This is the piece most easily lost. 515 mmwd rows, 110 bmm_wd rows and 40 bmm_wd_3d2d
    rows in the database differ from one another ONLY by their split; re-running the shape
    without it measures the compiler's default division and files the result under the
    forced division's label.
    """
    out = {}
    forced = r.get("split_forced")
    if isinstance(forced, dict):
        for key, var in (("b", "WD_B"), ("m", "WD_M"), ("n", "WD_N"), ("k", "WD_K")):
            if isinstance(forced.get(key), int):
                out[var] = str(forced[key])
        return out

    label = r.get("label") or ""
    if m := _LBL_SPLIT_KV.search(label):
        return {"WD_B": m[1], "WD_M": m[2], "WD_N": m[3], "WD_K": m[4]}
    if m := _LBL_SPLIT_MNK.search(label):
        return {"WD_M": m[1], "WD_N": m[2], "WD_K": m[3]}
    if m := _LBL_SPLIT_MXN.search(label):
        return {"WD_M": m[1], "WD_N": m[2], "WD_K": "1"}
    return out


def _flash_env(r):
    """Flash-attention geometry and work division, from the label.

    Returns None when the label does not carry enough to rebuild the run -- flash rows
    were logged in several formats and some predate the tiling fields entirely.
    """
    label = r.get("label") or ""
    m = _LBL_FA.search(label)
    if not m:
        return None
    out = {"FA_H": m[1], "FA_LQ": m[2], "FA_LK": m[3]}
    for group, var in (
        (4, "FA_D"),
        (5, "FA_H_TILES"),
        (6, "FA_LQ_TILES"),
        (7, "FA_LK_TILES"),
    ):
        if m[group]:
            out[var] = m[group]
    if w := _LBL_FA_WD.search(label):
        # `wd=H4-Lq8-Lk8` in the label is `H:4,Lq:8,Lk:8` in the environment.
        parts = re.findall(r"([A-Za-z]+)(\d+)", w[1])
        if parts:
            out["FA_WD"] = ",".join(f"{name}:{n}" for name, n in parts)
    return out


def _env_from_record(r):
    """The full environment that reproduces one record, or None if it cannot be rebuilt.

    Returning None is deliberate: a configuration this cannot rebuild is dropped from the
    sweep and counted, because measuring an approximation of it and storing the result
    under the original label is worse than a gap -- the gap is visible.
    """
    op = r.get("op")
    if not op:
        return None  # the parser never identified the op; nothing to run
    env = {"BENCH_OP": op}
    for field, var in (
        ("rows", "BENCH_ROWS"),
        ("cols", "BENCH_COLS"),
        ("tiles", "BENCH_TILES"),
        ("cores", "SENCORES"),
    ):
        if isinstance(r.get(field), int):
            env[var] = str(r[field])
    if isinstance(r.get("lx"), int):
        env["LX_PLANNING"] = str(r["lx"])

    if op == "flash_attn":
        fa = _flash_env(r)
        if fa is None:
            return None
        env.update(fa)
        return env

    env.update(_shape_env(r))
    env.update(_split_env(r))
    if r.get("layout_a") and r.get("layout_b"):
        env["WD_LAYOUT_A"] = r["layout_a"]
        env["WD_LAYOUT_B"] = r["layout_b"]
    if op == "transpose_outer" and (m := _LBL_TO_MID.search(r.get("label") or "")):
        env["TO_MID"] = m[1]
    if m := _LBL_FLAG.match(r.get("label") or ""):
        env["SPYRE_MATMUL_PREFERRED_LAYOUT"] = "" if m[1] == "off" else m[1]
    return env


def _configs(records, only_op=None):
    """Distinct run configurations, in a stable order, plus what could not be rebuilt."""
    seen, out, lost = set(), [], collections.Counter()
    for r in records:
        op = r.get("op")
        if only_op and op != only_op:
            continue
        env = _env_from_record(r)
        if env is None:
            lost[op or "(unidentified op)"] += 1
            continue
        key = tuple(sorted(env.items()))
        if key not in seen:
            seen.add(key)
            out.append(env)
    return out, lost


def _provenance(cores):
    """Header lines identifying the build these measurements belong to.

    ``parse_sweep_logs.py`` reads the ``git:`` line into ``model_sha`` on every record,
    which is what lets a re-score separate this sweep from an earlier one. The rest is
    for a human reading the log a month later.
    """

    def _sh(cmd, default="?"):
        try:
            return subprocess.run(
                cmd, capture_output=True, text=True, check=True, cwd=_ROOT
            ).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            return default

    sha = _sh(["git", "rev-parse", "HEAD"])
    dirty = "yes" if _sh(["git", "status", "--porcelain"], "") else "no"
    try:
        import torch

        torch_v = torch.__version__
    except ImportError:  # the sweep will fail anyway, but the header should still write
        torch_v = "?"
    return [
        f"git: {sha}",
        f"branch: {_sh(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])}",
        f"dirty: {dirty}",
        f"torch: {torch_v}",
        f"host: {socket.gethostname()}",
        f"date: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"cores: {cores or '(default)'}",
    ]


def _completed(log):
    """Run tags in ``log`` that already produced a measurement.

    A tag whose block has no SUMMARY was interrupted or failed to compile; it is re-run,
    since a half-written block is not a result.
    """
    done, tag = set(), None
    if not os.path.exists(log):
        return done
    with open(log, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if m := _RUN_HDR.match(line):
                tag = m[1]
            elif tag and line.startswith("SUMMARY"):
                done.add(tag)
                tag = None
    return done


def _hms(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op", default="", help="re-measure only this op")
    ap.add_argument("--limit", type=int, default=0, help="stop after N configurations")
    ap.add_argument("--reps", default="7", help="BENCH_REPS per configuration")
    ap.add_argument("--dry-run", action="store_true", help="list, do not run")
    ap.add_argument("--no-parse", action="store_true", help="skip the database update")
    ap.add_argument("--out", default="", help="log file (default: timestamped)")
    ap.add_argument(
        "--resume",
        default="",
        help="continue an interrupted sweep: append to this log and skip the "
        "configurations in it that already produced a measurement",
    )
    args = ap.parse_args()

    # Resolved here, not at import: it can exit with setup instructions, and
    # importing this module (for --help, or from a test) must not do that.
    records = records_path()
    with open(records, encoding="utf-8") as fh:
        records = json.load(fh)["records"]
    cfgs, lost = _configs(records, args.op or None)
    if args.limit:
        cfgs = cfgs[: args.limit]

    by_op = collections.Counter(c["BENCH_OP"] for c in cfgs)
    print(f"{len(cfgs)} configurations over {len(by_op)} ops")
    for op, n in by_op.most_common():
        print(f"    {op:<26} {n}")
    if lost:
        print(
            f"\n  {sum(lost.values())} records could not be rebuilt and are NOT swept:"
        )
        for op, n in lost.most_common():
            print(f"    {op:<26} {n}")
    if args.dry_run:
        print(f"\nrebuilt beyond the shape:\n{_RECONSTRUCTED}")
        for env in cfgs[:5]:
            print("  " + " ".join(f"{k}={v}" for k, v in sorted(env.items())))
        if len(cfgs) > 5:
            print(f"  ... and {len(cfgs) - 5} more")
        return 0

    log = (
        args.resume
        or args.out
        or os.path.join(_HERE, f"cost_model_sweep_{time.strftime('%Y%m%d_%H%M%S')}.log")
    )
    done = _completed(log) if args.resume else set()
    if args.resume:
        print(f"resuming {log}: {len(done)} of {len(cfgs)} already measured")
    print(f"\nlogging to {log}\n")

    failed, ran, t0 = 0, 0, time.time()
    with open(log, "a" if args.resume else "w", encoding="utf-8") as fh:
        if not args.resume:
            for line in _provenance(os.environ.get("SENCORES", "")):
                fh.write(line + "\n")
            fh.flush()
        todo = [(i, e) for i, e in enumerate(cfgs, 1)]
        for i, env in todo:
            tag = " ".join(f"{k}={v}" for k, v in sorted(env.items()))
            if tag in done:
                continue
            ran += 1
            eta = (time.time() - t0) / ran * (len(cfgs) - len(done) - ran)
            print(
                f"[{i}/{len(cfgs)}] {_hms(time.time() - t0)} elapsed, "
                f"~{_hms(eta)} left | {tag}",
                flush=True,
            )
            fh.write(f"\n=== {tag} ===\n")
            fh.flush()
            run_env = dict(
                os.environ,
                BENCH_REPS=args.reps,
                SPYRE_DUMP_COST="1",
                # the IO/MODEL/FEATS record lines parse_sweep_logs.py consumes
                BENCH_EMIT_RECORDS="1",
                **env,
            )
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

    print(f"\n{ran - failed}/{ran} produced a measurement in {_hms(time.time() - t0)}")
    print(f"log: {log}")
    if args.no_parse:
        print("database not updated (--no-parse)")
        return 0
    print("folding into the database...")
    subprocess.run(
        [sys.executable, os.path.join(_TOOLS, "parse_sweep_logs.py"), log],
        check=False,
    )
    print(
        "\nre-score THIS build only (pooling builds is how a compiler change gets read\n"
        "as a model error):  python3 tools/cost_model/eval_model.py --build latest"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
