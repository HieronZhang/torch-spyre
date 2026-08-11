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

"""Find LX-contested cases, predict how the real solvers rank on them, and emit the
hardware commands that check the prediction.

THE POINT. Everything so far has been prediction. `LAYOUT_SOLVER` is a real config knob that
makes the compiler produce genuinely different LX allocations for the same program, so the
model's ranking of those allocations is directly testable: predict the order, run the four
solvers, compare. That is the smallest experiment that can falsify "the cost model can
early-reflect expected relative performance".

WHY FLASH ATTENTION IS THE STARTING POINT. It is the only real workload in the corpus with a
large allocation space -- 20 movable buffers -- and three of its recorded configurations
exceed the 1587 KB/core budget, by up to 2.6x. An earlier version of this study reported
"37 contested bundles, all softmax" and missed every one of them, because the exhaustive
enumerator refuses bundles above 16 buffers and the skip was silent.

WHAT IS PREDICTED VERSUS WHAT IS MEASURED. The script predicts using the model over the
recorded features, with each solver's allocation reconstructed from its published objective.
Those reconstructions are approximations of the real solvers -- which is precisely why the
hardware run uses the REAL ones via `LAYOUT_SOLVER`, rather than trusting the reconstruction.

A NOTE ON FLASH'S ABSOLUTE ERROR. The recorded flash features carry the pre-fix
`tile_rows_per_core`, so absolute predictions are 15-45x high. That does not affect the
RANKING here: the underfill and spill derates depend on tile geometry, not on residency, so
they scale every allocation of a bundle by the same factor and cancel in a comparison.

    python3 research/lx_experiment.py                # find cases, predict, emit commands
    python3 research/lx_experiment.py --op flash_attn
"""

import argparse
import json
import os
import random
import sys

import regex as re

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "tools", "cost_model"))

import lx_choice as L  # noqa: E402

CAP = L.LX_CAPACITY_BYTES


def _feasible(feats, fp, life, nt, subset):
    return L.peak_footprint(feats, set(subset), fp, life, nt) <= CAP


def climb_objective(feats, value, restarts=12, seed=1):
    """Maximise ``value(subset)`` subject to capacity — how a solver meets its own goal."""
    fp, life, nt = L.buffer_footprints(feats), L.buffer_lifetimes(feats), len(feats) + 1
    names, rng = sorted(fp), random.Random(seed)

    def climb(start):
        cur = set(start)
        while cur and not _feasible(feats, fp, life, nt, cur):
            cur.discard(max(sorted(cur), key=lambda n: (fp[n], n)))
        best, bv, improved = set(cur), value(cur), True
        while improved:
            improved = False
            for n in names:
                cand = best ^ {n}
                if _feasible(feats, fp, life, nt, cand) and value(cand) > bv + 1e-9:
                    best, bv, improved = set(cand), value(cand), True
        return frozenset(best)

    starts = [set(), set(names), set(sorted(names, key=lambda n: (-fp[n], n)))]
    starts += [{n for n in names if rng.random() < 0.5} for _ in range(restarts - 3)]
    return max((climb(s) for s in starts), key=value)


def sequential(feats, order):
    """Place buffers in ``order``, keeping each that still fits — the greedy family."""
    fp, life, nt = L.buffer_footprints(feats), L.buffer_lifetimes(feats), len(feats) + 1
    chosen = []
    for n in order:
        if _feasible(feats, fp, life, nt, chosen + [n]):
            chosen.append(n)
    return frozenset(chosen)


def solver_allocations(feats):
    """What each shipped solver would keep resident, from its published objective."""
    fp, rc, life = (
        L.buffer_footprints(feats),
        L.read_counts(feats),
        L.buffer_lifetimes(feats),
    )
    return {
        # ilp_solver_ortools.py:208 -- minimise spilled (reads+1)*size, i.e. keep the most.
        "cpsat": climb_objective(
            feats, lambda s: sum((rc.get(n, 0) + 1) * fp[n] for n in s)
        ),
        # firstfit_bestfit_solver.py:204 -- (lifetime - discount)/uses ascending. Both
        # solvers share the ordering and differ only in gap choice, which does not change
        # the residency set.
        "bestfit": sequential(
            feats,
            sorted(
                fp, key=lambda n: ((life[n][1] - life[n][0]) / (rc.get(n, 0) + 1.5), n)
            ),
        ),
        # greedy_solver.py:189 -- chronological bump allocation.
        "greedy": sequential(feats, sorted(fp, key=lambda n: (life[n][0], n))),
    }


def analyse(record, exhaustive=False):
    """Predicted times for each solver's allocation, against the best available.

    With ``exhaustive`` the reference is the PROVEN optimum over every feasible allocation,
    which is what makes a regret figure trustworthy. Hill climbing is not adequate here: on
    flash attention it returns exactly CP-SAT's answer and misses an allocation 13% faster,
    so a regret measured against it reads 0% when the truth is 13%.
    """
    feats = record["feats"]
    movable = L.bundle_intermediates(feats)
    peak = L.peak_footprint(feats, set(movable))
    if peak <= CAP:
        return None  # capacity does not bind: every solver keeps everything
    if exhaustive:
        # Solve every objective exactly in one enumeration. Approximating a solver's own
        # objective is what made an earlier version of this study claim CP-SAT was 13-20%
        # off optimal when it is exactly optimal -- see `exhaustive_policies`.
        exact, evaluate, n_evals = L.exhaustive_policies(feats, CAP)
        seq = solver_allocations(feats)  # bestfit/greedy are deterministic sequences
        allocs = {
            "cpsat": exact["cpsat"],
            "largest": exact["largest"],
            "bestfit": seq["bestfit"],
            "greedy": seq["greedy"],
        }
        ref, ref_us, complete = exact["time"], evaluate(exact["time"]), True
    else:
        allocs = solver_allocations(feats)
        evaluate = L.fast_evaluator(feats)
        ref, ref_us = L.search_best(feats, CAP, seeds=list(allocs.values()))
        n_evals, complete = None, False
    return {
        "record": record,
        "movable": movable,
        "peak": peak,
        "allocs": allocs,
        "times": {k: evaluate(v) for k, v in allocs.items()},
        "ref": ref,
        "ref_us": ref_us,
        "n_evals": n_evals,
        "proven": complete,
    }


#: label token -> harness env var, for rebuilding a flash run from its recorded label.
_FA_ENV = (
    ("H", "FA_H"),
    ("Lq", "FA_LQ"),
    ("Lk", "FA_LK"),
    ("D", "FA_D"),
    ("htiles", "FA_H_TILES"),
    ("qtiles", "FA_LQ_TILES"),
    ("ktiles", "FA_LK_TILES"),
)


def hardware_commands(record):
    """The exact runs that would check the predicted ranking on a device."""
    lbl = record.get("label") or ""
    env = []
    for key, var in _FA_ENV:
        if m := re.search(rf"\b{key}=(\d+)", lbl):
            env.append(f"{var}={m[1]}")
    return " ".join(env)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="")
    ap.add_argument("--op", default="flash_attn")
    ap.add_argument(
        "--exhaustive",
        action="store_true",
        help="prove the optimum by enumerating every feasible "
        "allocation, instead of hill climbing",
    )
    args = ap.parse_args()

    import eval_model as E

    path = args.records or E.records_path()
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)["records"]

    cands = [
        r
        for r in records
        if r.get("op") == args.op and r.get("feats") and r.get("kernel_us")
    ]
    cases = [c for c in (analyse(r, args.exhaustive) for r in cands) if c]
    print(
        f"{len(cands)} {args.op} bundles with features; "
        f"{len(cases)} where LX capacity binds (budget {CAP / 1024:.0f} KB/core)\n"
    )

    for i, c in enumerate(sorted(cases, key=lambda c: -c["peak"]), 1):
        r = c["record"]
        lbl = (r.get("label") or "").replace("flash_attn ", "").replace("flash ", "")
        lbl = lbl.split(" (IR")[0].split(" [was")[0]
        print(f"--- case {i}: {lbl}")
        print(
            f"    measured {r['kernel_us']:.0f} us | {len(c['movable'])} movable buffers | "
            f"peak-if-all {c['peak'] / 1024:.0f}K vs {CAP / 1024:.0f}K budget "
            f"({c['peak'] / CAP:.1f}x over)"
        )
        ref_us = c["ref_us"]
        how = (
            f"PROVEN OPTIMAL over {c['n_evals']:,} feasible allocations"
            if c["proven"]
            else "best found by hill climbing -- NOT proven optimal"
        )
        print(
            f"      {'optimum':<8} keeps {len(c['ref']):>2}/{len(c['movable'])}  "
            f"predicted {ref_us:10.1f} us   {how}"
        )
        for k in c["allocs"]:
            keep = len(c["allocs"][k])
            reg = (c["times"][k] - ref_us) / ref_us * 100
            print(
                f"      {k:<8} keeps {keep:>2}/{len(c['movable'])}  "
                f"predicted {c['times'][k]:10.1f} us   regret {reg:+6.1f}%"
            )
        spread = max(c["times"].values()) / min(c["times"].values())
        print(f"    predicted spread across the three shipped solvers: {spread:.2f}x")
        print("\n    TO CHECK ON HARDWARE — same program, four real allocations:")
        envs = hardware_commands(r)
        for solver in ("greedy", "firstfit", "bestfit", "cpsat"):
            print(
                f"      BENCH_OP={args.op} {envs} SENCORES=32 BENCH_REPS=7 \\\n"
                f"        LAYOUT_SOLVER={solver} \\\n"
                f"        python3 docs/source/user_guide/examples/profile_ops.py"
            )
        print()

    if cases:
        print("The model predicts a ranking of those four runs. Measuring them is the")
        print(
            "smallest experiment that can falsify it — and the first check of this whole"
        )
        print("line of work against a device rather than against itself.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
