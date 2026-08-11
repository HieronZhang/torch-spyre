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

"""Where is CP-SAT NOT the time-optimal LX allocation?

CP-SAT maximises retained ``(read_count + is_intermediate) * size`` -- a BYTE objective. The
cost model ranks the same allocations by predicted TIME. Dividing every buffer's traffic by
the same bandwidth cannot reorder anything, so the two agree exactly whenever all contending
buffers move at one rate. They can only disagree when rates differ, and the model's rates
span 3.75x (`findings_lx.md` section 6).

This enumerates every feasible allocation of every bundle where capacity binds and reports
the bundles where CP-SAT's argmax is NOT time-optimal. Two ways that happens:

* **Strict loss** -- the byte-optimal set is uniquely worse in time than the time-optimal set.
* **Tie ambiguity** -- several sets tie on the byte objective but differ in predicted time.
  CP-SAT is free to return any of them, so its time is only as good as its tie-break. A byte
  objective cannot see this difference at all; a cost model can. Reported separately because
  the two have different fixes.

    python3 research/find_cpsat_gaps.py --records <db.json>
    python3 research/find_cpsat_gaps.py --records <db.json> --max-buffers 20 --verbose
"""

import argparse
import collections
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "tools", "cost_model"))

import lx_choice as L  # noqa: E402

CAP = L.LX_CAPACITY_BYTES


def enumerate_feasible(feats, cap=CAP):
    """Every feasible allocation, via the downward-closure DFS.

    Peak footprint is monotone in the allocation, so if a set does not fit no superset fits
    and the walk can prune on first infeasibility.
    """
    fp = L.buffer_footprints(feats)
    life = L.buffer_lifetimes(feats)
    nt = len(feats) + 1
    names = sorted(fp, key=lambda n: (-fp[n], n))
    out = []

    def peak(cur):
        return max(
            (
                sum(fp[b] for b in cur if life[b][0] <= t < life[b][1])
                for t in range(nt)
            ),
            default=0,
        )

    def rec(i, cur):
        if peak(cur) > cap:
            return  # prune: no superset can fit either
        if i == len(names):
            out.append(frozenset(cur))
            return
        rec(i + 1, cur)
        cur.add(names[i])
        rec(i + 1, cur)
        cur.discard(names[i])

    rec(0, set())
    return out, fp


def analyse(record, max_buffers, cap=CAP):
    """Time-optimal vs byte-optimal for one bundle, both solved exactly."""
    feats = record["feats"]
    movable = L.bundle_intermediates(feats)
    if not movable:
        return None
    if L.peak_footprint(feats, set(movable)) <= cap:
        return None  # capacity does not bind; nothing to choose
    if len(movable) > max_buffers:
        return {"skipped": True, "n": len(movable), "record": record}

    feasible, fp = enumerate_feasible(feats, cap)
    if len(feasible) < 2:
        return None
    rc = L.read_counts(feats)
    evaluate = L.fast_evaluator(feats)

    def byte_obj(s):
        return sum((rc.get(x, 0) + 1) * fp[x] for x in s)

    times = {s: evaluate(s) for s in feasible}
    best_t = min(times.values())
    top = max(byte_obj(s) for s in feasible)
    tied = [s for s in feasible if byte_obj(s) == top]
    # CP-SAT may return ANY set achieving its optimum, so bracket the outcome.
    cp_best, cp_worst = min(times[s] for s in tied), max(times[s] for s in tied)

    return {
        "skipped": False,
        "record": record,
        "n_movable": len(movable),
        "n_feasible": len(feasible),
        "opt_us": best_t,
        "cpsat_best_us": cp_best,
        "cpsat_worst_us": cp_worst,
        "n_tied": len(tied),
        "strict_regret": (cp_best - best_t) / best_t * 100.0,
        "tie_regret": (cp_worst - best_t) / best_t * 100.0,
        "patterns": collections.Counter(
            (op.get("hbm_pattern") or "default") for op in feats
        ),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="")
    ap.add_argument("--max-buffers", type=int, default=20)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import eval_model as E

    path = args.records or E.records_path()
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)["records"]

    cands = [r for r in records if r.get("feats") and r.get("kernel_us")]
    rows, skipped = [], []
    for r in cands:
        try:
            res = analyse(r, args.max_buffers)
        except Exception as exc:  # noqa: BLE001 -- one bad record must not kill the scan
            if args.verbose:
                print(f"  !! {r.get('op')}: {type(exc).__name__}: {exc}")
            continue
        if res is None:
            continue
        (skipped if res["skipped"] else rows).append(res)

    print(
        f"{len(cands)} bundles with features; {len(rows)} where LX capacity binds and"
    )
    print(f"more than one allocation is feasible (budget {CAP / 1024:.0f} KB/core).")
    if skipped:
        by = collections.Counter(f"{s['record'].get('op')}({s['n']})" for s in skipped)
        print(
            f"SKIPPED {len(skipped)} above --max-buffers={args.max_buffers}: {dict(by)}"
        )
    print()

    strict = [r for r in rows if r["strict_regret"] > 1e-9]
    ties = [r for r in rows if r["tie_regret"] > 1e-9 and r["strict_regret"] <= 1e-9]

    print("=== CP-SAT strictly worse than the time optimum ===")
    if not strict:
        print(
            "    NONE. On every binding bundle here the byte-optimal set is time-optimal."
        )
    for r in sorted(strict, key=lambda r: -r["strict_regret"]):
        lbl = (r["record"].get("label") or r["record"].get("op") or "")[:62]
        print(
            f"    {lbl:<64} +{r['strict_regret']:.1f}%  "
            f"({r['n_movable']} bufs, {r['n_feasible']} feasible)"
        )
    print()

    print("=== CP-SAT's optimum is TIED across sets that differ in time ===")
    print("    (a byte objective cannot separate these; the cost model can)")
    if not ties:
        print("    NONE.")
    for r in sorted(ties, key=lambda r: -r["tie_regret"])[:12]:
        lbl = (r["record"].get("label") or r["record"].get("op") or "")[:62]
        print(f"    {lbl:<64} up to +{r['tie_regret']:.1f}%  ({r['n_tied']} tied sets)")
    print()

    pats = collections.Counter()
    for r in rows:
        pats.update(r["patterns"])
    print("access patterns present across all binding bundles:")
    for k, v in pats.most_common():
        print(f"    {k:<24} {v}")
    print()
    print("A cost model can only beat the byte objective where these DIFFER within one")
    print(
        "bundle. If every binding bundle is single-pattern, no such case exists here and"
    )
    print("the search has to move to programs that mix transports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
