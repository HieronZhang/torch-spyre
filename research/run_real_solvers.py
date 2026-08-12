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

"""Run the REAL layout solvers on every contested shape and record what they choose.

Everything reported so far about `greedy`, `bestfit` and `cpsat` came from allocations
RECONSTRUCTED offline from each solver's published objective, then pinned with
`LX_FORCE_ONLY`. The measurements are real; the attribution to a named policy is not. This
replaces the reconstruction with the thing itself: one compile per (shape, solver), driven by
`LAYOUT_SOLVER`, keeping the residency the compiler actually produced, its features, and its
time -- all from the same compile, so they cannot disagree.

WHAT IT ANSWERS.

* Which policies genuinely collide, per shape -- read off the run, not inferred. The
  reconstruction says all four collapse on one shape and that greedy/bestfit collapse on two
  more; if the real solvers differ there, those shapes have arms that were never measured.
* Whether each reconstruction was faithful, by comparing the recorded LX set against the set
  the offline model derived for that policy. `bestfit` is already known not to be: it models
  the ordering only and ignores `_pick_gap`, which is the sole difference between
  `BestFitLayoutSolver` and `FirstFitLayoutSolver`.

`firstfit` is included because the claim that it duplicates `bestfit` is exactly the
assumption the source contradicts.

Output goes to a JSONL file, one line per run, flushed as it goes; stdout is a short table.
Nothing is printed from the child processes, so this does not flood a terminal.

    python3 research/run_real_solvers.py --dry-run
    python3 research/run_real_solvers.py
    python3 research/run_real_solvers.py --compare     # reconstruction vs reality, no device
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import time

import regex as re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools", "cost_model"))
_PROFILE = os.path.join(
    _ROOT, "docs", "source", "user_guide", "examples", "profile_ops.py"
)

_SUMMARY = re.compile(r"^SUMMARY .*?kernel_us=([0-9.]+)", re.M)
_FEATS = re.compile(r"^MODEL FEATS (.+)$", re.M)

SOLVERS = ["greedy", "firstfit", "bestfit", "cpsat"]


def shapes():
    """Every contested shape, INCLUDING the one the reconstruction collapsed.

    That shape was dropped from the earlier session because the reconstruction gave all four
    policies the same set -- which is precisely the verdict under test, so excluding it would
    bake the reconstruction's answer into the design.
    """
    path = os.path.join(_HERE, "forced_allocations.json")
    if not os.path.exists(path):
        return []
    out = []
    for c in json.load(open(path, encoding="utf-8"))["cases"]:
        env = dict(c.get("env") or {})
        env.setdefault("FA_B", "1")
        env.setdefault("FA_D", "128")
        for k in ("FA_H_TILES", "FA_LQ_TILES", "FA_LK_TILES"):
            env.setdefault(k, "1")
        n_distinct = len({tuple(a["lx"]) for a in c["arms"].values()})
        out.append(
            {
                "label": c["label"],
                "env": env,
                "arms": c["arms"],
                "reconstructed_distinct": n_distinct,
            }
        )
    return out


def canon(name):
    m = re.search(r"^(?:op|buf|b)(\d+)$|_buf(\d+)$", name or "")
    return f"b{m[1] or m[2]}" if m else (name or "")


def lx_set(feats):
    """The buffers the compiler actually placed in LX, canonicalised."""
    out = set()
    for op in feats or []:
        for a in op.get("args", []):
            if str(a.get("mem", "")).lower() == "lx":
                out.add(canon(a.get("name")))
    return out


def run(env_extra, reps, timeout_s):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_extra.items()})
    env.update(
        {
            "BENCH_OP": "flash_attn",
            "BENCH_EMIT_RECORDS": "1",
            "BENCH_REPS": str(reps),
            "LX_PLANNING": "1",
        }
    )
    t0 = time.time()
    try:
        p = subprocess.run(
            [sys.executable, _PROFILE],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        raw = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        raw = "TIMEOUT"
    m, f = _SUMMARY.search(raw), _FEATS.search(raw)
    feats = None
    if f:
        try:
            feats = json.loads(f.group(1))
        except Exception:  # noqa: BLE001
            feats = None
    return (float(m[1]) if m else None), feats, raw, time.time() - t0


def compare(rows):
    """Reconstruction against reality: did each policy choose what we said it would?"""
    by = collections.defaultdict(dict)
    for r in rows:
        if r.get("lx") is not None:
            by[r["label"]][r["solver"]] = set(r["lx"])
    print("\n=== reconstruction vs the real solver ===")
    for sh in shapes():
        real = by.get(sh["label"])
        if not real:
            continue
        lbl = sh["label"].replace("flash_attn ", "")[:44]
        print(f"\n  {lbl}")
        for s in SOLVERS:
            got = real.get(s)
            if got is None:
                print(f"    {s:<10} (no measurement)")
                continue
            arm = sh["arms"].get(s)
            if arm is None:
                print(f"    {s:<10} kept {len(got):>2}  (not reconstructed)")
                continue
            want = {canon(x) for x in arm["lx"]}
            same = got == want
            print(
                f"    {s:<10} kept {len(got):>2}  reconstruction said {len(want):>2}  "
                f"{'MATCH' if same else 'DIFFERS'}"
                + (
                    ""
                    if same
                    else f"  real-only {sorted(got - want)[:4]}"
                    f" recon-only {sorted(want - got)[:4]}"
                )
            )
        n_real = len({frozenset(v) for v in real.values()})
        print(
            f"    -> {n_real} distinct allocation(s) in reality, "
            f"{sh['reconstructed_distinct']} in the reconstruction"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--out", default=os.path.join(_HERE, "real_solver_results.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--compare",
        action="store_true",
        help="re-read an existing results file and compare; no device",
    )
    args = ap.parse_args()

    sh = shapes()
    if args.compare:
        rows = [json.loads(x) for x in open(args.out, encoding="utf-8") if x.strip()]
        compare(rows)
        return 0
    print(
        f"{len(sh)} contested shape(s) x {len(SOLVERS)} solvers "
        f"= {len(sh) * len(SOLVERS)} compiles\n"
    )
    if args.dry_run:
        for s in sh:
            e = s["env"]
            print(
                f"  H={e.get('FA_H')} Lq={e.get('FA_LQ')} "
                f"cores={e.get('SENCORES')}   "
                f"reconstruction: {s['reconstructed_distinct']} distinct arm(s)"
            )
        return 0

    fh = open(args.out, "a", encoding="utf-8")  # noqa: SIM115 -- open for the session
    rows = []
    print(f"  {'shape':<26}{'solver':<10}{'kept':>5}{'measured':>11}{'':>4}")
    for s in sh:
        for solver in SOLVERS:
            env = dict(s["env"])
            env["LAYOUT_SOLVER"] = solver
            us, feats, raw, dt = run(env, args.reps, args.timeout)
            got = sorted(lx_set(feats)) if feats else None
            row = {
                "label": s["label"],
                "solver": solver,
                "env": env,
                "kernel_us": us,
                "lx": got,
                "n_lx": len(got) if got else None,
                "feats": feats,
                "seconds": round(dt, 1),
                "tail": None if us else (raw or "")[-400:],
            }
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            e = s["env"]
            tag = f"H={e.get('FA_H')} L={e.get('FA_LQ')} {e.get('SENCORES')}c"
            shown = f"{us:.1f} us" if us else "FAILED"
            print(
                f"  {tag:<26}{solver:<10}{(row['n_lx'] or 0):>5}{shown:>11}"
                f"  ({dt:.0f}s)"
            )
    fh.close()

    print("\n=== measured, per shape ===")
    for s in sh:
        sel = [r for r in rows if r["label"] == s["label"] and r["kernel_us"]]
        if len(sel) < 2:
            continue
        ref = min(r["kernel_us"] for r in sel)
        e = s["env"]
        print(f"\n  H={e.get('FA_H')} Lq={e.get('FA_LQ')} cores={e.get('SENCORES')}")
        for r in sorted(sel, key=lambda r: r["kernel_us"]):
            print(
                f"    {r['solver']:<10}{r['kernel_us']:>10.1f} us  "
                f"{(r['kernel_us'] - ref) / ref * 100:>+7.1f}%  keeps {r['n_lx']}"
            )
        n = len({frozenset(r["lx"]) for r in sel if r["lx"]})
        print(
            f"    {n} distinct allocation(s); reconstruction said "
            f"{s['reconstructed_distinct']}"
        )
    compare(rows)
    print(f"\nraw: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
