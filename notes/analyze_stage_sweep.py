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

The control comes first: if HBM bytes move with the stage count, the intermediates
spilled and the timing comparison means nothing.

    python3 notes/analyze_stage_sweep.py <log>
"""

import collections
import statistics
import sys

import regex as re

_BASE_STAGES = 5  # amax, sub, exp, sum, div -- the chain before any are added


def _parse(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if "SUMMARY" not in line or "FAILED" in line:
            continue
        f = dict(re.findall(r"(\w+)=([-\w.+]+)", line))
        if f.get("op") != "softmax_stages":
            continue
        try:
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

        # CONTROL. Adding stages must not change what crosses the HBM boundary.
        if len(hbm) > 1:
            print(
                f"  CONTROL FAILED: HBM bytes move with the stage count {sorted(hbm)}"
                " -- the intermediates spilled, so the timing below is confounded."
            )
            continue

        stages = sorted(cfg)
        if len(stages) < 2:
            continue
        lo, hi = stages[0], stages[-1]
        t_lo = statistics.median(x["us"] for x in cfg[lo])
        t_hi = statistics.median(x["us"] for x in cfg[hi])
        grew = t_hi / t_lo
        # What each form predicts for this pair.
        flat = 1.0
        scaled = (_BASE_STAGES + hi) / (_BASE_STAGES + lo)
        noise = max(0.03, max(x["cv"] for v in cfg.values() for x in v) / 100 * 2)
        print(
            f"  {lo} -> {hi} stages: measured x{grew:.2f}   "
            f"element-only predicts x{flat:.2f}   chain-length predicts x{scaled:.2f}"
        )
        if abs(grew - flat) <= noise:
            print("  -> FLAT: the element-only floor is right as written.")
        elif abs(grew - scaled) <= max(noise, 0.15 * scaled):
            print(
                "  -> SCALES WITH CHAIN LENGTH: the floor is mis-keyed. Both LX traffic"
                " and per-element work through more stages fit this; the sweep does not"
                " separate them."
            )
        else:
            print(
                "  -> NEITHER form fits. Report the numbers; do not pick a mechanism."
            )


if __name__ == "__main__":
    main()
