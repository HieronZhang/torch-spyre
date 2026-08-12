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

"""Generate the single-core tables in `bmm_layout_effects.md` from the measurements.

Each table sits between a ``<!-- BEGIN:name -->`` / ``<!-- END:name -->`` pair and is
rewritten in place, so the document cannot drift from the data. Measured kernel time only --
no cost-model values appear anywhere in this study.

    python3 bmm_observation/gen_1core.py
    python3 bmm_observation/gen_1core.py --show
"""

import argparse
import collections
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DOC = os.path.join(_HERE, "bmm_layout_effects.md")
DATA = os.path.join(_ROOT, "research", "sweep_v2_results.jsonl")

#: The 32-core cube already tabulated in the report, at B = 4. Kept here so the comparison
#: is explicit; these are the report's own published numbers, not a re-measurement.
THIRTYTWO = {
    ("row", "row"): 3792.0,
    ("batch", "row"): 2246.0,
    ("row", "batch"): 2276.0,
    ("batch", "batch"): 777.0,
}
THIRTYTWO_C = {("batch", "batch"): 458.0}  # same cell with C batch-outer

NAME = {"0,1,2": "row", "1,0,2": "batch"}


def _rows():
    if not os.path.exists(DATA):
        return [], []
    rows = [json.loads(ln) for ln in open(DATA, encoding="utf-8") if ln.strip()]
    return (
        [r for r in rows if r.get("phase") == 4 and r.get("kernel_us")],
        [r for r in rows if r.get("phase") == 5 and r.get("kernel_us")],
    )


def cube():
    """The eight cells at one core, both batch sizes, beside the published 32-core column."""
    p4, _ = _rows()
    cell = {}
    for r in p4:
        cell[(r["shape"][0], NAME[r["A"]], NAME[r["B"]], r["pref"])] = r["kernel_us"]
    out = [
        "| A | B | C | B=2 (µs) | B=4 (µs) | B=4 at 32 cores (µs) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for a in ("row", "batch"):
        for b in ("row", "batch"):
            for pref, c in (("off", "row"), ("output", "batch")):
                v2 = cell.get((2, a, b, pref))
                v4 = cell.get((4, a, b, pref))
                ref = THIRTYTWO_C.get((a, b)) if c == "batch" else THIRTYTWO.get((a, b))
                out.append(
                    f"| {a} | {b} | {c} | {v2:,.0f} | {v4:,.0f} | "
                    f"{(f'{ref:,.0f}' if ref else '—')} |"
                )
    return "\n".join(out), cell


def effects(cell):
    """Each operand's effect alone and together, at one core and at 32."""

    def blk(d, tag):
        rr, br = d[("row", "row")], d[("batch", "row")]
        rb, bb = d[("row", "batch")], d[("batch", "batch")]
        a, b, both = rr / br, rr / rb, rr / bb
        return (
            f"| {tag} | {a:.2f}× | {b:.2f}× | {a * b:.2f}× | {both:.2f}× | "
            f"**{both / (a * b):.2f}×** |"
        )

    one = {
        (a, b): cell[(4, a, b, "off")]
        for a in ("row", "batch")
        for b in ("row", "batch")
    }
    out = [
        "| cores | switch A alone | switch B alone | if independent | measured together"
        " | vs independent |",
        "|---|---:|---:|---:|---:|---:|",
        blk(one, "1"),
        blk(THIRTYTWO, "32"),
    ]
    return "\n".join(out)


def output_layout(cell):
    """What asking for a batch-outer output is worth, per input combination."""
    out = [
        "| A | B | C row-outer (µs) | C batch-outer (µs) | worth |",
        "|---|---|---:|---:|---:|",
    ]
    for a in ("row", "batch"):
        for b in ("row", "batch"):
            off, on = cell.get((4, a, b, "off")), cell.get((4, a, b, "output"))
            if off and on:
                out.append(f"| {a} | {b} | {off:,.0f} | {on:,.0f} | {off / on:.2f}× |")
    return "\n".join(out)


def ladder():
    """Time against core count at a fixed layout, and what layout is worth at each."""
    _, p5 = _rows()
    by = collections.defaultdict(dict)
    for r in p5:
        by[NAME[r["A"]]][r["n_cores"]] = r["kernel_us"]
    cores = sorted({c for v in by.values() for c in v})
    out = [
        "| cores | A row-outer (µs) | A batch-outer (µs) | layout worth | "
        "speedup vs 1 core |",
        "|---:|---:|---:|---:|---:|",
    ]
    for c in cores:
        r_, b_ = by["row"].get(c), by["batch"].get(c)
        if not (r_ and b_):
            continue
        sp = by["row"][1] / r_ if by["row"].get(1) else float("nan")
        out.append(f"| {c} | {r_:,.0f} | {b_:,.0f} | {r_ / b_:.2f}× | {sp:.1f}× |")
    return "\n".join(out)


def inject(name, body):
    src = open(DOC, encoding="utf-8").read()
    a, b = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
    if a not in src or b not in src:
        return False
    i, j = src.index(a) + len(a), src.index(b)
    open(DOC, "w", encoding="utf-8").write(src[:i] + "\n" + body + "\n" + src[j:])
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    t_cube, cell = cube()
    blocks = {
        "cube_1core": t_cube,
        "effects": effects(cell),
        "output_layout": output_layout(cell),
        "ladder": ladder(),
    }
    for k, v in blocks.items():
        if args.show:
            print(f"\n===== {k} =====\n{v}")
        else:
            print(f"  {k:<16}{'injected' if inject(k, v) else 'NO MARKER'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
