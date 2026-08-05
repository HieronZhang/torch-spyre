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

"""Read a `run_stage_sweep.sh` log and say whether the §19 floor is mis-keyed.

The floor charges ``elements / (cores * rate)``, with no dependence on how many
stages the fused chain has. This sweep adds LX-resident stages while holding the
element count and the HBM traffic fixed, so:

  * time flat in stages   -> the element-only form is right
  * time grows with stages -> the form is wrong; it has to scale with chain length

Some HBM traffic leaks per stage in practice, so the control is quantitative rather
than pass/fail: it asks which driver the TIME tracks. If the leak were responsible,
growth would follow the HBM ratio; if the chain is, it follows the chain ratio.

    python3 notes/analyze_stage_sweep.py <log>
"""

import collections
import statistics
import sys

import regex as re

_BASE_STAGES = 5  # amax, sub, exp, sum, div -- the chain before any are added
_TENSOR_BYTES = [0]  # one tensor's bytes, filled in from the parsed rows


_ROWS_COLS = [0, 0]


def _parse(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if "SUMMARY" not in line or "FAILED" in line:
            continue
        f = dict(re.findall(r"(\w+)=([-\w.+]+)", line))
        if f.get("op") != "softmax_stages":
            continue
        try:
            _ROWS_COLS[0] = int(f.get("rows", 0))
            _ROWS_COLS[1] = int(f.get("cols", 0))
            rows.append(
                {
                    "cores": int(f["cores"]) if f.get("cores", "").isdigit() else 32,
                    "stages": int(f.get("stages", 0)),
                    "us": float(f["kernel_us"]),
                    "hbm": int(f.get("io_hbm_bytes", 0)),
                    "cv": float(f.get("kernel_us_cv", 0)),
                }
            )
        except (KeyError, ValueError):
            continue
    return rows


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rows = _parse(sys.argv[1])
    if not rows:
        sys.exit("no softmax_stages SUMMARY lines found")

    _TENSOR_BYTES[0] = _ROWS_COLS[0] * _ROWS_COLS[1] * 2

    by = collections.defaultdict(dict)
    for r in rows:
        by[r["cores"]].setdefault(r["stages"], []).append(r)

    for cores in sorted(by):
        cfg = by[cores]
        print(f"\n=== cores = {cores} ===")
        print(f"{'stages':>7} {'chain':>6} {'HBM MB':>8} {'us':>9} {'cv%':>6} {'n':>3}")
        base = None
        hbm = set()
        for st in sorted(cfg):
            v = cfg[st]
            us = statistics.median(x["us"] for x in v)
            mb = statistics.median(x["hbm"] for x in v) / 1e6
            cv = max(x["cv"] for x in v)
            hbm.add(round(mb, 1))
            if base is None:
                base = us
            print(
                f"{st:>7} {_BASE_STAGES + st:>6} {mb:>8.1f} {us:>9.1f} "
                f"{cv:>6.1f} {len(v):>3}"
            )

        # CONTROL, quantitative. A perfect run holds HBM bytes fixed. In practice a
        # little leaks per stage, so a binary pass/fail would reject a 12 % leak that
        # cannot explain a 157 % effect. Instead ask which driver the TIME tracks: if the
        # leak were responsible, growth would follow the HBM ratio, not the chain ratio.
        stages = sorted(cfg)
        if len(stages) < 2:
            continue
        lo, hi = stages[0], stages[-1]
        t_lo = statistics.median(x["us"] for x in cfg[lo])
        t_hi = statistics.median(x["us"] for x in cfg[hi])
        b_lo = statistics.median(x["hbm"] for x in cfg[lo])
        b_hi = statistics.median(x["hbm"] for x in cfg[hi])
        grew = t_hi / t_lo
        by_bytes = b_hi / b_lo
        by_chain = (_BASE_STAGES + hi) / (_BASE_STAGES + lo)
        leak = (b_hi - b_lo) / max(1, hi - lo)
        full = 2 * (_TENSOR_BYTES[0] or 1)
        print(
            f"  leak {leak / 1e6:.2f} MB per stage against a {full / 1e6:.1f} MB full "
            f"round trip ({leak / full * 100:.0f} % of one)"
        )
        print(
            f"  {lo} -> {hi} stages: time x{grew:.2f}   |   HBM x{by_bytes:.2f}   "
            f"chain length x{by_chain:.2f}   element-only x1.00"
        )
        d = {
            "element-only (the current floor)": abs(grew - 1.0),
            "HBM bytes (the leak)": abs(grew - by_bytes),
            "chain length": abs(grew - by_chain),
        }
        best = min(d, key=d.get)
        # "least bad of three" is not a match. Require the winner to be close in
        # absolute terms, or the honest answer is that none of the forms fit.
        if d[best] > 0.25:
            print(
                f"  -> NONE of these forms fit (closest is {best}, off by {d[best]:.2f})."
                " Report the numbers; do not pick a driver."
            )
            continue
        print(f"  -> time tracks {best.upper()} (off by {d[best]:.2f})")
        if best == "chain length":
            print(
                "     The element-only form is MIS-KEYED: the floor has to scale with the"
                " chain. This does not separate LX traffic from per-element work through"
                " more stages -- both are proportional to elements x stages."
            )


if __name__ == "__main__":
    main()
