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

"""Generate the report's per-section accuracy lines and data tables from the LIVE model.

WHY THIS EXISTS. Every accuracy number and table in ``cost_model_report.md`` must be
generated from the live model and the live records, never hand-typed -- hand-typed
numbers go stale silently the moment a coefficient moves, and several already had
(Part I quoted 22 points when the in-scope population is 89). Run this after ANY model
change and paste the output into the matching section.

POPULATION. Exactly the rows ``eval_model.py`` scores: not failed, has ``kernel_us``,
and passes ``in_scope`` (the standing scope decisions -- cores >= 8, fused reductions
with >= 1024 columns, no corrupt-feature SHAs). Measured time is ``kernel_us``, which is
the canonical field present on every row; ``kernel_us_min`` exists on only ~45 % of rows
(repeat-backed ones) and using it silently biases ``n``.

SECTIONS covered here are the SINGLE-OP ones (Parts I-III). Part IV (coarse tiling) is
deliberately excluded: its model is mid-revision (see cost_model_status.md).

    python3 tools/cost_model/report_tables.py             # all single-op sections
    python3 tools/cost_model/report_tables.py --section 6 # one section
    python3 tools/cost_model/report_tables.py --shapes 6  # + the per-shape table for that section
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_model as em  # noqa: E402

cm = em.cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from records import records_path  # noqa: E402

RECORDS = records_path()

# section key -> (title, ops). Ops are the harness op names, so the population is
# explicit and auditable rather than implied by a category mapping.
SECTIONS: dict[str, tuple[str, list[str]]] = {
    "1-3": ("Part I - single pointwise", ["neg", "add", "mul", "exp", "gelu"]),
    "3": ("§3 - chained adds (out of scope by design)", ["add3", "add4"]),
    "4": (
        "§4 - broadcast operands",
        ["bcast", "bcastcol", "mulbcast", "copy"],
    ),
    "5": ("§5 - reduction", ["read", "sumrow", "sumcol", "sumall", "amax", "mean"]),
    "6": ("§6 - transport", ["transpose", "transpose_outer", "cat0", "cat1"]),
    "7-11": ("§7-10 - plain matmul", ["mm"]),
    "12": ("§12 - matmul split shape", ["mmwd"]),
    "13": ("§13 - batched matmul", ["bmm_wd", "bmm_wd_3d2d", "bmm_layout"]),
}


def load():
    with open(RECORDS, encoding="utf-8") as f:
        recs = json.load(f)["records"]
    return [
        r for r in recs if not r.get("failed") and r.get("kernel_us") and em.in_scope(r)
    ]


def score(rows, ops):
    """[(rec, pred_us, meas_us, err_pct)] for every scoreable row whose op is in ``ops``."""
    dp = cm.CostParams()
    out = []
    for r in rows:
        if (r.get("op") or "") not in ops:
            continue
        feats, _src = em.features_for(r, dp)
        if feats is None:
            continue
        try:
            pred = cm.predict_ops(feats, dp) / 1000.0
        except Exception:  # noqa: BLE001
            continue
        meas = r["kernel_us"]
        out.append((r, pred, meas, (pred - meas) / meas * 100.0))
    return out


def stats(res):
    e = [x[3] for x in res]
    if not e:
        return None
    return {
        "n": len(e),
        "rms": (sum(v * v for v in e) / len(e)) ** 0.5,
        "mean": st.mean(e),
        "lo": min(e),
        "hi": max(e),
        "out": sum(1 for v in e if abs(v) > 10),
    }


def unscoreable(rows, ops):
    """Rows in the population that could NOT be scored, by reason -- reported so a
    shrunken ``n`` is never mistaken for a small experiment."""
    dp = cm.CostParams()
    reasons: dict[str, int] = {}
    for r in rows:
        if (r.get("op") or "") not in ops:
            continue
        feats, src = em.features_for(r, dp)
        if feats is None:
            reasons[src.split("(")[0]] = reasons.get(src.split("(")[0], 0) + 1
    return reasons


def emit_section(key, rows, want_shapes=False):
    title, ops = SECTIONS[key]
    res = score(rows, ops)
    s = stats(res)
    print(f"\n{'=' * 78}\n{title}   ops: {', '.join(ops)}\n{'=' * 78}")
    skip = unscoreable(rows, ops)
    if s is None:
        print(f"  NO SCOREABLE ROWS.  unscoreable: {skip}")
        return
    print(
        f"  ACCURACY LINE:  RMS **{s['rms']:.1f} %**, mean {s['mean']:+.1f} %, "
        f"range {s['lo']:+.1f}…{s['hi']:+.1f} %, over {s['n']} points "
        f"({s['out']} beyond ±10 %)"
    )
    if skip:
        print(f"  unscoreable rows excluded from that n: {skip}")

    print("\n| op | n | RMS % | mean % | err range | >10 % |")
    print("|---|---:|---:|---:|---|---:|")
    for op in ops:
        sub = stats([x for x in res if x[0].get("op") == op])
        if sub is None:
            continue
        print(
            f"| `{op}` | {sub['n']} | {sub['rms']:.1f} | {sub['mean']:+.1f} | "
            f"{sub['lo']:+.1f}…{sub['hi']:+.1f} | {sub['out']} |"
        )
    print(
        f"| **all** | **{s['n']}** | **{s['rms']:.1f}** | **{s['mean']:+.1f}** | "
        f"**{s['lo']:+.1f}…{s['hi']:+.1f}** | **{s['out']}** |"
    )

    if want_shapes:
        print("\n| op | shape | cores | measured µs | predicted µs | err % |")
        print("|---|---|---:|---:|---:|---:|")
        for r, pred, meas, err in sorted(
            res, key=lambda x: (x[0].get("op") or "", x[0].get("rows") or 0)
        ):
            shp = r.get("label") or ""
            print(
                f"| `{r.get('op')}` | {shp} | {r.get('cores')} | "
                f"{meas:.1f} | {pred:.1f} | {err:+.1f} |"
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", default="", help="one section key, e.g. 6")
    ap.add_argument("--shapes", default="", help="also emit the per-shape table")
    args = ap.parse_args()
    rows = load()
    print(f"in-scope scoreable population: {len(rows)} records")
    keys = [args.section] if args.section else list(SECTIONS)
    for k in keys:
        if k not in SECTIONS:
            raise SystemExit(f"unknown section {k!r}; have {list(SECTIONS)}")
        emit_section(k, rows, want_shapes=(args.shapes == k))


if __name__ == "__main__":
    main()
