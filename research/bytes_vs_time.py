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

"""Can time-ranking and byte-ranking of an LX allocation ever disagree?

THE QUESTION. `research/findings_lx.md` shows that on every contested bundle in the recorded
corpus, the compiler's existing byte proxy -- CP-SAT's `(read_count + 1) * size` -- already
picks the time-optimal allocation (36/37, worst regret 0.6%). If that holds generally, a cost
model has nothing to contribute to LX allocation and we should say so.

It should NOT hold generally, and the reason is visible in `predict_ops`. When any op in a
bundle carries an access-pattern override, the model abandons a single bandwidth for a
per-op one (`cost_model.py:1183-1196`):

    for o in ops:
        bw = _eff_bw(o)
        if bw: mem += (ro + wo) / bw
        else:  mem += (ro + wo)/bw_peak + turnaround*min(ro, wo)

The flat overrides are mild -- `restickify` ~116 GB/s, `reduce_outer` ~113, against a ~150
default. The shape-dependent ones are not: `cat0` (`stick_scatter`) is
`clamp(144 - 9.6·log2(C/64) - 2.4·log2(R), 44, 150)` and reaches its **44 GB/s** floor at
Granite MLP dimensions, while `transpose_outer` bottoms out at **40**. So a spilled byte can
cost up to 3.75x another spilled byte, and a policy counting bytes cannot see any of it.

THE EXPERIMENT. Rather than fabricate a program -- whose synthesized features would be my
guess at what the extractor emits -- this perturbs REAL recorded features: take a contested
softmax bundle, tag one op with an access pattern, and re-rank. Everything else, including
the byte counts, is untouched.

WHAT IT DOES AND DOES NOT SHOW. A softmax `sub` is not really a transpose, so this is a
demonstration of a MECHANISM, not a measurement of a workload. It answers "can the two
rankings diverge, and how big does the bandwidth gap have to be?" -- and thereby tells the
Granite case studies (which contain a real transpose) what to look for. It does not by itself
establish that any shipping program hits the divergence.

    python3 research/bytes_vs_time.py
"""

import argparse
import copy
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "tools", "cost_model"))

import lx_choice as L  # noqa: E402

cm = L.cm

#: Patterns the model prices differently, with the CostParams field each reads.
PATTERNS = {
    "(default)": None,
    "restickify": "bw_restickify_gbps",
    "reduce_outer": "bw_reduce_outer_gbps",
    "stick_scatter": None,  # priced by transport_bw(o, p, "cat0"), not a flat constant
}


def tag_pattern(feats, op_index, pattern):
    """Copy of ``feats`` with one op given an HBM access pattern. Nothing else changes."""
    out = copy.deepcopy(feats)
    out[op_index]["hbm_pattern"] = pattern
    return out


def byte_optimal(feats, allocs):
    """CP-SAT's choice: retain the most ``(read_count + 1) * size``."""
    fp = L.buffer_footprints(feats)
    rc = L.read_counts(feats)
    return max(allocs, key=lambda s: sum((rc.get(n, 0) + 1) * fp[n] for n in s))


def time_optimal(feats, allocs):
    """The cost model's choice: the fastest feasible allocation."""
    return min(allocs, key=lambda s: L.predict(L.apply_allocation(feats, s)))


def run(record, capacity=L.LX_CAPACITY_BYTES):
    feats = record["feats"]
    allocs = L.feasible_allocations(feats, capacity)
    p = cm.CostParams()
    print(f"{record.get('label')}")
    print(
        f"  {len(allocs)} feasible allocations, LX budget {capacity / 1024:.0f} KB/core"
    )
    print(
        f"  default BW {p.bw_peak_gbps:.0f} GB/s, restickify {p.bw_restickify_gbps:.0f}, "
        f"reduce_outer {p.bw_reduce_outer_gbps:.0f}\n"
    )
    ops = [o.get("name") for o in feats]
    print(
        f"{'tagged op':<28} {'pattern':<14} {'byte-optimal':<20} {'time-optimal':<20} same?"
    )
    rows = []
    for idx, opname in enumerate(ops):
        for pat in ("(default)", "restickify", "reduce_outer", "stick_scatter"):
            f = feats if pat == "(default)" else tag_pattern(feats, idx, pat)
            b = byte_optimal(f, allocs)
            t = time_optimal(f, allocs)
            same = b == t
            rows.append((opname, pat, b, t, same))
            if pat == "(default)" and idx > 0:
                continue  # the unperturbed baseline is the same for every idx
            print(
                f"{opname:<28} {pat:<14} {str(sorted(b)):<20} {str(sorted(t)):<20} "
                f"{'yes' if same else 'NO -- diverges'}"
            )
    div = [r for r in rows if not r[4]]
    print(f"\n{len(div)} of {len(rows)} perturbations make the two rankings disagree")
    for opname, pat, b, t, _ in div:
        f = tag_pattern(feats, ops.index(opname), pat)
        cost_b = L.predict(L.apply_allocation(f, b))
        cost_t = L.predict(L.apply_allocation(f, t))
        print(
            f"   {opname} as {pat}: bytes pick {sorted(b)} ({cost_b:.1f} us), "
            f"time picks {sorted(t)} ({cost_t:.1f} us) "
            f"-> {(cost_b - cost_t) / cost_t * 100:+.1f}% regret for the byte proxy"
        )
    return rows


def flip_threshold(feats):
    """For each pair of candidate buffers, the bandwidth gap needed to reverse the ranking.

    This is the question that actually decides whether a cost model is worth wiring in. The
    byte proxy ranks two buffers by ``(read_count + 1) * size``; the cost model ranks them by
    spill TIME, which is that quantity divided by the effective bandwidth of the ops that
    touch it. So for buffers x and y with byte-values Vx > Vy, time prefers keeping y only if

        Vy / BW_y  >  Vx / BW_x      i.e.      BW_x / BW_y  >  Vx / Vy

    The model's bandwidths run from 150 (default) down to **40** (`transpose_outer` after its
    M penalty) and 44 (`cat0` floor) -- an envelope of **3.75x**. Any pair whose byte-values
    differ by more than that can never be reordered by an access-pattern difference; anything
    inside it can.

    Do not shorten this to the two flat constants. `restickify` (116) and `reduce_outer`
    (113) alone suggest a 1.33x envelope, which makes the byte proxy look unbeatable on
    almost every pair. The shape-dependent transports are where the real spread is: at
    Granite MLP dimensions (R=4096, C=12800) `cat0` sits at its 44 GB/s floor, 3.41x below
    default, and the byte proxy's regret on this bundle reaches 36.6%.
    """
    fp = L.buffer_footprints(feats)
    rc = L.read_counts(feats)
    p = cm.CostParams()
    # EVERY bandwidth the model can assign, not just the two flat constants. An earlier
    # version of this bound used only `restickify` (116) and `reduce_outer` (113) and
    # concluded the envelope was 1.33x -- which wrongly made the byte proxy look unbeatable.
    # The shape-dependent transports go far lower: `cat0` floors at 44 GB/s and
    # `transpose_outer` at 40 after its M penalty.
    floors = [
        p.bw_peak_gbps,
        p.bw_restickify_gbps,
        p.bw_reduce_outer_gbps,
        p.tx_cat0_floor_gbps,
        p.tx_touter_floor_gbps,
        getattr(p, "tx_touter_m_floor_gbps", p.tx_touter_floor_gbps),
        getattr(p, "tx_cat1_floor_gbps", p.bw_peak_gbps),
        getattr(p, "bw_broadcast_gbps", p.bw_peak_gbps),
    ]
    max_gap = p.bw_peak_gbps / min(floors)
    val = {n: (rc.get(n, 0) + 1) * fp[n] for n in fp}
    names = sorted(fp)
    print(
        f"\nachievable bandwidth spread: {max_gap:.2f}x "
        f"({p.bw_peak_gbps:.0f} / {min(floors):.0f} GB/s)"
    )
    print(
        f"\n{'pair':<16} {'byte-value ratio':>17} {'BW gap needed':>15}  reorderable?"
    )
    any_re = False
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            hi, lo = max(val[x], val[y]), min(val[x], val[y])
            if lo == 0:
                continue
            need = hi / lo
            ok = need <= max_gap
            any_re |= ok
            print(
                f"{x + ' vs ' + y:<16} {need:16.2f}x {need:14.2f}x  "
                f"{'YES' if ok else 'no -- byte order is unbeatable'}"
            )
    if not any_re:
        print(
            "\nNo pair in this bundle can be reordered by any access pattern the model "
            "knows. Here the byte proxy is not an approximation of the time ranking -- it "
            "is provably identical to it."
        )
    return any_re


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default="")
    ap.add_argument("--capacity", type=int, default=L.LX_CAPACITY_BYTES)
    args = ap.parse_args()

    import eval_model as E

    path = args.records or E.records_path()
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)["records"]

    hits = L.contested(records, args.capacity)
    if not hits:
        sys.exit("no contested bundles in this database")
    # The largest-spread bundle: the most room for the two policies to differ.
    run(hits[0][0], args.capacity)
    flip_threshold(hits[0][0]["feats"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
