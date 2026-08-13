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

"""Find an LX allocation case where the default policy is beaten by a genuine SWAP.

Every case measured so far is a *subset* case: the slow policy keeps strictly fewer buffers
than the fast one, so "more resident is faster" explains the result without any policy being
clever. One shape already breaks that pattern -- at `H=8 Lq=Lk=512` on 8 cores `greedy` keeps
`b12` (64 KB/core, read twice) and spills `b13` (128 KB/core, read once), while `cpsat` does
the reverse, same count both sides -- but the two allocations move 3x64 and 2x128 KB per
core, so the gap is 0.3 % and unmeasurable.

WHAT WOULD BE A STRONGER RESULT, and what this searches for:

    greedy != cpsat, NEITHER a subset of the other, cpsat measurably faster,
    and the cost model still orders them correctly.

That case cannot be explained by counting resident buffers. It needs the traffic each
allocation actually moves -- which is what the model computes and what the byte objective
approximates.

WHERE TO LOOK. A swap only pays when the contending buffers differ in value per byte, so the
program needs heterogeneous intermediates. Two families are swept:

* **flash attention**, untiled. The swap above is structural (`b13` is twice `b12` and read
  half as often), so the *relative* gap scales as 1/(H*D) -- small head counts amplify it.
  Core count moves the capacity boundary that creates the contention in the first place.
* **the case-study programs** in `research/workloads.py`, run untiled (`WL_*T=1`, which makes
  `_tiled` an empty ExitStack and never enters the hint that 2.13 broke). `block_norm_mlp` is
  the designed case: a cross-row reduction, a broadcast and full-width matmul operands in one
  bundle, i.e. many small high-reuse buffers competing with few large ones.

NOTHING HERE IS SIMULATED. Every allocation reported comes from a compile with `LAYOUT_SOLVER`
set to that policy; residency, features and time are read from that same run.

    python3 research/find_swap_cases.py --dry-run
    python3 research/find_swap_cases.py --smoke        # which programs compile at all
    python3 research/find_swap_cases.py --budget-min 75
    python3 research/find_swap_cases.py --analyze      # rank the results, no device
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
_PROFILE = os.path.join(
    _ROOT, "docs", "source", "user_guide", "examples", "profile_ops.py"
)

_SUMMARY = re.compile(r"^SUMMARY .*?kernel_us=([0-9.]+)", re.M)
_FEATS = re.compile(r"^MODEL FEATS (.+)$", re.M)

SOLVERS = ["greedy", "cpsat", "firstfit", "bestfit"]


def _flash(h, lq, lk, cores):
    return {
        "program": "flash_attn",
        "label": f"flash H={h} Lq={lq} Lk={lk} {cores}c",
        "env": {
            "BENCH_OP": "flash_attn",
            "FA_B": "1",
            "FA_H": str(h),
            "FA_LQ": str(lq),
            "FA_LK": str(lk),
            "FA_D": "128",
            "FA_H_TILES": "1",
            "FA_LQ_TILES": "1",
            "FA_LK_TILES": "1",
            "SENCORES": str(cores),
        },
    }


def _wl(name, cores, **kw):
    env = {"BENCH_OP": f"research:{name}", "SENCORES": str(cores)}
    env.update({f"WL_{k.upper()}": str(v) for k, v in kw.items()})
    # Untiled: every tile count 1, so `_tiled` yields an empty ExitStack.
    for t in ("bt", "st", "ft", "ht", "qt"):
        env.setdefault(f"WL_{t.upper()}", "1")
    tag = " ".join(f"{k}={v}" for k, v in sorted(kw.items()))
    return {
        "program": name,
        "label": f"{name} {tag} {cores}c",
        "env": env,
    }


def grid():
    """Configs in priority order, tuned by the smoke run.

    The smoke run settled three things. `block_norm_mlp` (6 buffers in LX) and `attn_scores`
    (5) both land in the contested band and are swept widely. `mlp_up` at Granite width put
    only ONE buffer in LX -- its intermediates are ~1 MB per core each, so capacity does not
    bind, it is simply exceeded -- so it is swept small instead, where more than one can fit.
    And flash at `H=2` does not compile at all: the work-division pass splits the head
    dimension and 2 does not divide 4 (`buf5 dim d0 size=2 is not evenly divisible by
    split=4`), so the head count starts at 4.
    """
    out = []
    # 1. The designed heterogeneous case: a reduction, a broadcast and matmul operands in
    #    one bundle. Spread across the capacity boundary in both width and length.
    for cores in (8, 32):
        for seq in (256, 512, 1024):
            for ff in (2048, 4096, 12800):
                out.append(_wl("block_norm_mlp", cores, batch=2, seq=seq, d_ff=ff))
    # 2. Attention through the softmax, unequal Lq/Lk -- the buffer ratio changes with Lk,
    #    which is the ratio that makes a swap pay.
    for cores in (8, 32):
        for heads in (4, 8):
            for lk in (512, 1024, 2048):
                out.append(
                    _wl("attn_scores", cores, batch=1, heads=heads, seq_q=512, seq_k=lk)
                )
    # 3. mlp_up, small enough that more than one intermediate fits.
    for cores in (8, 32):
        for seq in (128, 256):
            for ff in (1024, 2048):
                out.append(_wl("mlp_up", cores, batch=2, seq=seq, d_ff=ff))
    # 4. Flash, where the known swap is relatively largest (it scales as 1/(H*D)) and the
    #    core count moves the capacity boundary that creates the contention.
    for cores in (4, 8, 16):
        for h in (4, 8):
            for lseq in (512, 1024):
                out.append(_flash(h, lseq, lseq, cores))
    # 5. Flash with Lq != Lk, which changes the size ratio of the contending pair.
    for cores in (8, 16):
        for lq, lk in ((512, 2048), (2048, 512)):
            out.append(_flash(8, lq, lk, cores))
    return out


def canon(name):
    m = re.search(r"^(?:op|buf|b)(\d+)$|_buf(\d+)$", name or "")
    return f"b{m[1] or m[2]}" if m else (name or "")


def lx_set(feats):
    out = set()
    for op in feats or []:
        for a in op.get("args", []):
            if str(a.get("mem", "")).lower() == "lx":
                out.add(canon(a.get("name")))
    return out


def inventory(feats):
    """Per-buffer size, read count and live range, for the lifetime figure."""
    d = collections.defaultdict(
        lambda: {"elems": 0, "reads": 0, "writes": 0, "first": 10**6, "last": -1}
    )
    for i, op in enumerate(feats or []):
        for a in op.get("args", []):
            e = d[canon(a.get("name"))]
            e["elems"] = max(e["elems"], a.get("elems") or 0)
            e["first"], e["last"] = min(e["first"], i), max(e["last"], i)
            if a.get("role") == "output":
                e["writes"] += 1
            else:
                e["reads"] += 1
    return {k: v for k, v in d.items() if v["writes"] and v["reads"]}


def run(env_extra, reps, timeout_s):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_extra.items()})
    env.update({"BENCH_EMIT_RECORDS": "1", "BENCH_REPS": str(reps), "LX_PLANNING": "1"})
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


def classify(sets):
    """How greedy and cpsat differ: identical, one a subset, or a genuine swap."""
    g, c = sets.get("greedy"), sets.get("cpsat")
    if g is None or c is None:
        return "incomplete"
    if g == c:
        return "identical"
    if g < c:
        return "greedy-subset"
    if c < g:
        return "cpsat-subset"
    return "SWAP"


def analyze(path):
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    by = collections.defaultdict(dict)
    for r in rows:
        if r.get("kernel_us"):
            by[r["label"]][r["solver"]] = r
    scored = []
    for lbl, d in by.items():
        sets = {s: set(r["lx"] or []) for s, r in d.items()}
        kind = classify(sets)
        if "greedy" not in d or "cpsat" not in d:
            continue
        g, c = d["greedy"]["kernel_us"], d["cpsat"]["kernel_us"]
        gain = (g - c) / c * 100.0
        scored.append(
            {
                "label": lbl,
                "kind": kind,
                "greedy_us": g,
                "cpsat_us": c,
                "gain_pct": gain,
                "n_greedy": len(sets["greedy"]),
                "n_cpsat": len(sets["cpsat"]),
                "g_only": sorted(sets["greedy"] - sets["cpsat"]),
                "c_only": sorted(sets["cpsat"] - sets["greedy"]),
                "all": {s: sorted(v) for s, v in sets.items()},
            }
        )
    swaps = [s for s in scored if s["kind"] == "SWAP"]
    swaps.sort(key=lambda s: -s["gain_pct"])
    print(
        f"\n{len(scored)} configs with both greedy and cpsat measured; "
        f"{len(swaps)} are swaps\n"
    )
    print(f"  {'config':<40}{'kind':<15}{'greedy':>9}{'cpsat':>9}{'cpsat gain':>12}")
    for s in sorted(scored, key=lambda s: (s["kind"] != "SWAP", -s["gain_pct"])):
        print(
            f"  {s['label'][:38]:<40}{s['kind']:<15}"
            f"{s['greedy_us']:>9.1f}{s['cpsat_us']:>9.1f}{s['gain_pct']:>11.1f}%"
        )
    if swaps:
        b = swaps[0]
        print(f"\n=== best swap: {b['label']} ===")
        print(f"  greedy keeps {b['n_greedy']}, only: {b['g_only']}")
        print(f"  cpsat  keeps {b['n_cpsat']}, only: {b['c_only']}")
        print(f"  cpsat is {b['gain_pct']:.1f}% faster")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=200)
    ap.add_argument("--budget-min", type=float, default=90.0)
    ap.add_argument("--out", default=os.path.join(_HERE, "swap_case_results.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="one compile per program")
    ap.add_argument("--analyze", action="store_true", help="rank results, no device")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="measure only the (config, solver) pairs --out has no measurement for. "
        "A pair that FAILED is retried, since a timeout or a crash is not a result.",
    )
    args = ap.parse_args()

    if args.analyze:
        return analyze(args.out)

    cfgs = grid()
    if args.smoke:
        seen, sel = set(), []
        for c in cfgs:
            if c["program"] not in seen:
                seen.add(c["program"])
                sel.append(c)
        cfgs = sel
    # Resume by RESULT, not by position: the run order is deterministic but a config can
    # be skipped mid-sweep (the greedy short-circuit below), so "how far did it get" is
    # not the same question as "what is missing".
    have: set = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("kernel_us") is not None:
                    have.add((r["label"], r["solver"]))
        todo = sum(
            1 for c in cfgs for s in SOLVERS if (c["label"], s) not in have
        )
        print(f"resuming: {len(have)} pairs already measured, {todo} to run")

    if args.dry_run:
        pairs = [(c["label"], s) for c in cfgs for s in SOLVERS if (c["label"], s) not in have]
        print(f"{len(cfgs)} configs x {len(SOLVERS)} solvers -> {len(pairs)} runs")
        for lbl, s in pairs:
            print(f"  {lbl:50s} {s}")
        return 0

    deadline = time.time() + args.budget_min * 60
    fh = open(args.out, "a", encoding="utf-8")  # noqa: SIM115 -- open for the session
    print(f"  {'config':<40}{'solver':<10}{'kept':>5}{'measured':>11}")
    stopped = None
    for c in cfgs:
        if time.time() > deadline:
            stopped = c["label"]
            break
        solvers = ["greedy"] if args.smoke else SOLVERS
        solvers = [s for s in solvers if (c["label"], s) not in have]
        if not solvers:
            continue
        for solver in solvers:
            env = dict(c["env"])
            env["LAYOUT_SOLVER"] = solver
            us, feats, raw, dt = run(env, args.reps, args.timeout)
            got = sorted(lx_set(feats)) if feats else None
            fh.write(
                json.dumps(
                    {
                        "label": c["label"],
                        "program": c["program"],
                        "solver": solver,
                        "env": env,
                        "kernel_us": us,
                        "lx": got,
                        "n_lx": len(got) if got else None,
                        "inventory": inventory(feats) if feats else None,
                        "feats": feats,
                        "seconds": round(dt, 1),
                        "tail": None if us else (raw or "")[-400:],
                    }
                )
                + "\n"
            )
            fh.flush()
            os.fsync(fh.fileno())
            shown = f"{us:.1f} us" if us else "FAILED"
            print(
                f"  {c['label'][:38]:<40}{solver:<10}{(len(got or [])):>5}{shown:>11}"
            )
            # Short-circuit on a greedy failure: if the DEFAULT policy cannot compile a
            # config, the other three are usually a waste of device time. Suppressed
            # under --resume, where the whole point is to fill in the pairs an earlier
            # short-circuit skipped.
            if us is None and solver == "greedy" and not args.smoke and not args.resume:
                print("    (greedy failed -- skipping the other solvers)")
                break
    fh.close()
    if stopped:
        print(f"\nBUDGET REACHED -- stopped before {stopped}; the rest were not run.")
    if not args.smoke:
        analyze(args.out)
    print(f"\nraw: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
