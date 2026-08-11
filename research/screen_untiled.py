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

"""Contested LX allocations in the case studies WITHOUT coarse tiling.

Coarse tiling is refused for these programs, so the knob the original screen swept -- tile
counts -- is unavailable. It was never the essential one. A bundle is contested when one
buffer fits the 1587 KB budget and several do not, and that is a statement about SIZE:
shape and core count reach it just as well. Untiled flash proved the point, going from
"nothing fits" at H=32 to four contested shapes and 2.1 million feasible allocations once
H and the sequence length came down.

So this sweeps `(batch, sequence, d_ff scale) x cores` at tile counts of 1 and reports where
capacity binds with room left to choose. Buffer dimensions come from `screen_configs.PROGRAMS`,
so the two screens agree on what each program contains.

WHAT THIS IS NOT. Predicted time -- that needs extracted features. It ranks by spilled bytes
and by how many allocations remain feasible, which is enough to decide what to compile.
Untiled flash is the cautionary tale: this screen admitted 10 shapes and only 4 turned out
contested, because peak liveness is well below the naive sum. Over-admitting is the safe
direction for a screen, but the counts here are candidates, not findings.

    python3 research/screen_untiled.py
    python3 research/screen_untiled.py --top 8 --emit-commands
"""

import argparse
import itertools
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from screen_configs import LX_CAPACITY_BYTES, PROGRAMS  # noqa: E402

#: Shape scalings applied to the dims a program actually has. Chosen to bracket the budget
#: from both sides: at full Granite size a single MLP intermediate already exceeds it, and at
#: 1/8 everything fits with room to spare.
SCALES = [
    {"B": 1, "S": 1, "F": 1},
    {"B": 1, "S": 1, "F": 2},
    {"B": 1, "S": 2, "F": 1},
    {"B": 1, "S": 2, "F": 2},
    {"B": 2, "S": 1, "F": 1},
    {"B": 2, "S": 2, "F": 2},
    {"B": 1, "S": 4, "F": 4},
    {"B": 4, "S": 4, "F": 4},
]
CORES = [32, 8]

#: dim -> which scale divides it. Everything else is held fixed, so the sweep varies exactly
#: the sizes a caller can set through WL_* without touching the program.
DIVIDES = {
    "B": "B",
    "S": "S",
    "Sn": "S",
    "Sp": "S",
    "Fd": "F",
    "H": "B",
    "ScSn": "S",
    "Lq": "S",
    "Lk": "S",
    "Sc": "S",
}


def scaled(prog, scale):
    dims = dict(prog.dims)
    for d, v in dims.items():
        key = DIVIDES.get(d)
        if key and scale.get(key, 1) > 1:
            dims[d] = max(1, v // scale[key])
    return dims


def screen(prog, scale, cores, cap=LX_CAPACITY_BYTES):
    """Feasible-allocation count and spill spread for one untiled configuration."""
    dims, orig = scaled(prog, scale), prog.dims
    prog.dims = dims
    try:
        knobs = dict.fromkeys(prog.tiled.values(), 1)  # untiled
        fps = prog.footprints(knobs, cores)
        bws = prog.bandwidths(knobs)
        names = sorted(fps)
        peak_all = prog.peak(set(names), fps)
        feasible = [
            frozenset(c)
            for r in range(len(names) + 1)
            for c in itertools.combinations(names, r)
            if prog.peak(set(c), fps) <= cap
        ]
        if len(feasible) < 2:
            return None
        costs = {s: prog.spilled(s, fps, knobs, cores) for s in feasible}
        times = {s: prog.spilled_time(s, fps, knobs, cores, bws) for s in feasible}
        lo = min(costs.values())
        tied = [s for s in feasible if costs[s] == lo]
        best_t = min(times.values())
        return {
            "scale": scale,
            "cores": cores,
            "peak": peak_all,
            "binds": peak_all > cap,
            "n_feasible": len(feasible),
            "keeps": max(len(s) for s in feasible),
            "n_buffers": len(names),
            "spread": max(costs.values()) / lo if lo else 1.0,
            "tie_regret": (max(times[s] for s in tied) - best_t) / best_t * 100.0
            if best_t
            else 0.0,
            "dims": dims,
        }
    finally:
        prog.dims = orig


def wl_env(prog_name, dims):
    """The WL_* environment that reproduces a screened configuration on device."""
    out = {}
    for dim, val in dims.items():
        if dim == "B":
            out["WL_BATCH"] = val
        elif dim in ("S", "Sn"):
            out["WL_SEQ" if prog_name != "decode_block" else "WL_NEW_LEN"] = val
        elif dim == "Fd":
            out["WL_D_FF"] = val
        elif dim == "D":
            out["WL_D_MODEL"] = val
        elif dim == "H":
            out["WL_HEADS"] = val
        elif dim == "Lq":
            out["WL_SEQ_Q"] = val
        elif dim == "Lk":
            out["WL_SEQ_K"] = val
        elif dim == "Sc":
            out["WL_CACHE_LEN"] = val
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--emit-commands", action="store_true")
    args = ap.parse_args()

    print(f"LX budget {LX_CAPACITY_BYTES / 1024:.0f} KB/core, tile counts all 1.")
    print("Contested = capacity binds AND more than one allocation still fits.\n")

    best_overall = []
    for name, prog in PROGRAMS.items():
        rows = []
        for scale, cores in itertools.product(SCALES, CORES):
            r = screen(prog, scale, cores)
            if r and r["binds"] and r["keeps"] > 0:
                rows.append(r)
        rows.sort(key=lambda r: (-r["n_feasible"], -r["spread"]))
        total = len(SCALES) * len(CORES)
        print(
            f"=== {name}: {len(rows)} of {total} untiled configurations contested "
            f"({len(prog.buffers)} intermediates)"
        )
        if not rows:
            print("    none - capacity never binds, or nothing fits when it does\n")
            continue
        print(
            f"    {'B/S/F divisor':<16}{'cores':>6}{'peak':>9}{'feasible':>10}"
            f"{'keeps':>8}{'spread':>9}{'tie':>8}"
        )
        for r in rows[: args.top]:
            s = r["scale"]
            tag = f"/{s['B']} /{s['S']} /{s['F']}"
            print(
                f"    {tag:<16}{r['cores']:>6}"
                f"{r['peak'] / 1024:>8.0f}K{r['n_feasible']:>10}"
                f"{r['keeps']:>3}/{r['n_buffers']:<4}{r['spread']:>8.2f}x"
                f"{r['tie_regret']:>7.1f}%"
            )
        best_overall.append((name, rows[0]))
        if args.emit_commands:
            r = rows[0]
            env = wl_env(name, r["dims"])
            envs = " ".join(f"{k}={v}" for k, v in sorted(env.items()))
            print("\n    most contested, to compile:")
            print(f"      BENCH_OP=research:{name} {envs} \\")
            print(f"        SENCORES={r['cores']} BENCH_EMIT_RECORDS=1 BENCH_REPS=5 \\")
            print("        python3 docs/source/user_guide/examples/profile_ops.py")
        print()

    if best_overall:
        print("Most contested configuration per program:")
        for name, r in sorted(best_overall, key=lambda t: -t[1]["n_feasible"]):
            print(
                f"  {name:<16}{r['n_feasible']:>10} feasible, "
                f"{r['keeps']}/{r['n_buffers']} keepable, "
                f"spread {r['spread']:.2f}x, {r['cores']} cores"
            )
        print(
            "\nThese are CANDIDATES. The untiled-flash screen admitted 10 shapes and 4"
        )
        print(
            "were contested once real features were extracted -- peak liveness runs well"
        )
        print("below the naive sum. Compile before believing any of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
