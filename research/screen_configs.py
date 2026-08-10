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

"""Which of the 128 tile configurations are worth compiling?

Compiling and extracting features costs a device run per configuration. Most configurations
are not worth one: either everything fits in LX and there is no decision to make, or nothing
fits and there is no choice either. This screens all of them analytically first and ranks by
how contested the allocation actually is.

WHAT IS MODELLED. Each program is its op sequence plus, for every intermediate, the DIMS it
carries. A buffer's per-core tile is the product of its dimensions divided by the tile counts
of the tiled dims **it actually has** and by the core count. That distinction is the point:
in `block_norm_mlp` the norm's buffers carry `D` while the projections' carry `Fd`, and `Fd`
is tiled where `D` is not -- so raising `ft` shrinks one group and leaves the other alone,
which is what makes the residency trade-off non-trivial rather than a matter of size order.

WHAT IS NOT MODELLED. Predicted time. That needs the extractor's `OpFeatures`, so the ranking
here is by **spilled HBM traffic** -- the byte proxy CP-SAT optimises. The flash result says
byte-optimal and time-optimal can differ by 13-20 %, so this screen finds where a decision
EXISTS, not which decision is right. That is exactly what a screen should do: cheap, and
wrong only in the direction of admitting a config worth a closer look.

    python3 research/screen_configs.py
    python3 research/screen_configs.py --top 8 --verbose
"""

import argparse
import itertools
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from workloads import GRANITE, KNOB_GRID  # noqa: E402

LX_CAPACITY_BYTES = 1_625_344
DTYPE_BYTES = 2


def cat0_bw(cols, rows):
    """The model's stick-plane transport rate for a cat on a partition dim.

    ``transport_bw(o, p, "cat0")`` = ``clamp(144 - 9.6*log2(C/64) - 2.4*log2(R), 44, 150)``
    with ``C = logical[-1]`` (the stick dim) and ``R = logical[-2]``
    (``cost_model.py:1054-1076``, ``_transport_rc`` at ``:1028``). Reproduced here so the
    screen can price a spill in TIME, not only in bytes.
    """
    sp = max(1.0, cols / 64.0)
    return min(
        150.0, max(44.0, 144.0 - 9.6 * math.log2(sp) - 2.4 * math.log2(max(2, rows)))
    )


#: Effective bandwidth by access pattern. `default` is `bw_peak_gbps`; `restickify` and
#: `reduce_outer` are flat constants; `cat0` is shape-dependent and is the only one that
#: reaches far enough (44 GB/s, 3.41x down) to overturn a byte ranking.
FLAT_BW = {
    "default": 150.0,
    "broadcast": 118.0,
    "restickify": 116.0,
    "reduce_outer": 113.0,
}


class Prog:
    """A program as (dim sizes, tiled dims, ops, intermediates with their dims)."""

    def __init__(self, name, dims, tiled, ops, buffers):
        self.name, self.dims, self.tiled, self.ops = name, dims, tiled, ops
        # name -> (producer, [consumers], (dim names...)[, access pattern])
        self.buffers = buffers

    def _pattern(self, buf):
        rec = self.buffers[buf]
        return rec[3] if len(rec) > 3 else "default"

    def _tiled_dims(self, buf, knobs):
        """This buffer's dim sizes after tiling — only dims it actually has shrink."""
        out = []
        for d in self.buffers[buf][2]:
            size = self.dims[d]
            if d in self.tiled:
                size //= max(1, knobs.get(self.tiled[d], 1))
            out.append(size)
        return out

    def bandwidths(self, knobs):
        """Effective GB/s for each buffer's HBM traffic, at these tile counts."""
        out = {}
        for buf in self.buffers:
            pat = self._pattern(buf)
            if pat == "cat0":
                dims = self._tiled_dims(buf, knobs)
                cols = dims[-1] if dims else 64
                rows = dims[-2] if len(dims) > 1 else 2
                out[buf] = cat0_bw(cols, rows)
            else:
                out[buf] = FLAT_BW.get(pat, 150.0)
        return out

    def footprints(self, knobs, cores):
        """Per-core bytes of each buffer's tile, at these tile counts."""
        out = {}
        for buf in self.buffers:
            n = 1
            for size in self._tiled_dims(buf, knobs):
                n *= size
            out[buf] = max(1, n * DTYPE_BYTES // cores)
        return out

    def lifetimes(self):
        return {b: (rec[0], max(rec[1]) + 1) for b, rec in self.buffers.items()}

    def peak(self, subset, fps):
        life = self.lifetimes()
        return max(
            (
                sum(fps[b] for b in subset if life[b][0] <= t < life[b][1])
                for t in range(len(self.ops) + 1)
            ),
            default=0,
        )

    def spilled(self, subset, fps, knobs, cores):
        """HBM bytes spilling everything outside `subset` adds, over the whole loop.

        This is CP-SAT's objective: ``(read_count + is_intermediate) * size``, summed over
        the spilled buffers (``ilp_solver_ortools.py:208-224``).
        """
        trips = 1
        for n in knobs.values():
            trips *= max(1, n)
        total = 0
        for b, rec in self.buffers.items():
            if b not in subset:
                total += (1 + len(rec[1])) * fps[b] * trips * cores
        return total

    def spilled_time(self, subset, fps, knobs, cores, bws):
        """The same traffic divided by the rate each buffer actually moves at.

        This is what the cost model ranks. It differs from `spilled` only through the
        per-buffer bandwidth, so the two orderings coincide exactly whenever every
        contending buffer shares one rate — which is why CP-SAT is provably time-optimal on
        every single-pattern bundle.
        """
        trips = 1
        for n in knobs.values():
            trips *= max(1, n)
        total = 0.0
        for b, rec in self.buffers.items():
            if b not in subset:
                total += (1 + len(rec[1])) * fps[b] * trips * cores / bws[b]
        return total


G = GRANITE
PROGRAMS = {
    "mlp_up": Prog(
        "mlp_up",
        {"B": 4, "S": 512, "D": G["d_model"], "Fd": G["d_ff"]},
        {"B": "bt", "S": "st", "Fd": "ft"},
        ["mm_gate", "mm_up", "silu", "mul"],
        {
            "gate": (0, [2], ("B", "S", "Fd")),
            "up": (1, [3], ("B", "S", "Fd")),
            "act": (2, [3], ("B", "S", "Fd")),
        },
    ),
    "attn_scores": Prog(
        "attn_scores",
        {"B": 2, "H": 8, "Lq": 512, "Lk": 512, "Dh": G["head_dim"]},
        {"B": "bt", "H": "ht", "Lq": "qt"},
        [
            "transpose",
            "scale_q",
            "mm_scores",
            "amax",
            "sub",
            "exp",
            "sum",
            "div",
            "mm_out",
        ],
        {
            # kt has no Lq, so `qt` does not shrink it -- it is loop-invariant there and
            # stays large while the Lq-carrying buffers shrink. That asymmetry is what
            # makes its residency a real question.
            "kt": (0, [2], ("B", "H", "Dh", "Lk")),
            "qs": (1, [2], ("B", "H", "Lq", "Dh")),
            "scores": (2, [3, 4], ("B", "H", "Lq", "Lk")),
            "mx": (3, [4], ("B", "H", "Lq")),
            "sub": (4, [5], ("B", "H", "Lq", "Lk")),
            "ex": (5, [6, 7], ("B", "H", "Lq", "Lk")),
            "sm": (6, [7], ("B", "H", "Lq")),
            "probs": (7, [8], ("B", "H", "Lq", "Lk")),
        },
    ),
    "block_norm_mlp": Prog(
        "block_norm_mlp",
        {"B": 4, "S": 512, "D": G["d_model"], "Fd": G["d_ff"]},
        {"B": "bt", "S": "st", "Fd": "ft"},
        [
            "square",
            "mean",
            "rsqrt",
            "scale",
            "weight",
            "mm_gate",
            "mm_up",
            "silu",
            "mul",
        ],
        {
            # D-carrying buffers do NOT shrink with `ft`; Fd-carrying ones do. Two groups
            # that move independently under one knob.
            "sq": (0, [1], ("B", "S", "D")),
            "ms": (1, [2], ("B", "S")),
            "inv": (2, [3], ("B", "S")),
            "xn": (3, [4], ("B", "S", "D")),
            "xw": (4, [5, 6], ("B", "S", "D")),
            "gate": (5, [7], ("B", "S", "Fd")),
            "up": (6, [8], ("B", "S", "Fd")),
            "act": (7, [8], ("B", "S", "Fd")),
        },
    ),
    # The program built specifically to put a SLOW SMALL buffer against a FAST BIG one --
    # the only configuration in which a time ranking can differ from a byte ranking.
    # `k_all`/`v_all` are cat0 (stick_scatter); `gate`/`up` are default and much larger,
    # and `ft` shrinks THEM without touching the cache tensors, so sweeping `ft` sweeps the
    # byte ratio across the window where the two objectives disagree.
    "decode_block": Prog(
        "decode_block",
        {
            "B": 2,
            "H": 8,
            "Sn": 128,
            "Sc": 1024,
            "ScSn": 1152,
            "D": G["d_model"],
            "Dh": G["head_dim"],
            "Fd": G["d_ff"],
        },
        {"B": "bt", "H": "ht", "Fd": "ft"},
        [
            "q",
            "k_new",
            "v_new",
            "k_all",
            "v_all",
            "scores",
            "probs",
            "ctx",
            "attn",
            "ms",
            "xn",
            "gate",
            "up",
            "out",
        ],
        {
            "q": (0, [5], ("B", "H", "Sn", "Dh")),
            "k_new": (1, [3], ("B", "H", "Sn", "Dh")),
            "v_new": (2, [4], ("B", "H", "Sn", "Dh")),
            "k_all": (3, [5], ("B", "H", "ScSn", "Dh"), "cat0"),
            "v_all": (4, [7], ("B", "H", "ScSn", "Dh"), "cat0"),
            "scores": (5, [6], ("B", "H", "Sn", "ScSn")),
            "probs": (6, [7], ("B", "H", "Sn", "ScSn")),
            "ctx": (7, [8], ("B", "H", "Sn", "Dh")),
            "attn": (8, [9, 10], ("B", "H", "Sn", "D")),
            "ms": (9, [10], ("B", "H", "Sn"), "reduce_outer"),
            "xn": (10, [11, 12], ("B", "H", "Sn", "D")),
            "gate": (11, [13], ("B", "H", "Sn", "Fd")),
            "up": (12, [13], ("B", "H", "Sn", "Fd")),
        },
    ),
    # The concat here is on the d_model-WIDE hidden state, not on the KV cache, so its
    # stick dim is 4096 rather than 128 and the model prices it at ~62 GB/s instead of
    # ~110. That, plus `h` having two consumers against gate/up's one, is what lets a
    # time ranking disagree with the byte ranking.
    "prefix_block": Prog(
        "prefix_block",
        {"B": 4, "S": 1024, "D": G["d_model"], "Fd": G["d_ff"]},
        {"B": "bt", "S": "st", "Fd": "ft"},
        ["cat", "square_mean", "rsqrt", "scale", "mm_gate", "mm_up", "silu_mul"],
        {
            "h": (0, [1, 3], ("B", "S", "D"), "cat0"),
            "ms": (1, [2], ("B", "S"), "reduce_outer"),
            "inv": (2, [3], ("B", "S")),
            "xn": (3, [4, 5], ("B", "S", "D"), "broadcast"),
            "gate": (4, [6], ("B", "S", "Fd")),
            "up": (5, [6], ("B", "S", "Fd")),
        },
    ),
}


def screen(prog, knobs, cores, cap=LX_CAPACITY_BYTES):
    """Is this configuration contested, and does the BYTE optimum differ from the TIME one?

    Both objectives are solved EXACTLY by enumeration -- never approximate a solver's own
    objective, which is the error that produced a retracted claim in `flash_lx_findings.md`.
    CP-SAT may return ANY set achieving its byte optimum, so its time is bracketed over all
    byte-optimal ties rather than assumed.
    """
    fps = prog.footprints(knobs, cores)
    bws = prog.bandwidths(knobs)
    names = sorted(fps)
    all_peak = prog.peak(set(names), fps)
    feasible = [
        frozenset(c)
        for r in range(len(names) + 1)
        for c in itertools.combinations(names, r)
        if prog.peak(set(c), fps) <= cap
    ]
    if not feasible:
        return None
    costs = {s: prog.spilled(s, fps, knobs, cores) for s in feasible}
    times = {s: prog.spilled_time(s, fps, knobs, cores, bws) for s in feasible}
    best, worst = min(costs.values()), max(costs.values())
    best_t = min(times.values())
    # Every set CP-SAT could legitimately return, and the time it would then get.
    lo = min(costs.values())
    tied = [s for s in feasible if costs[s] == lo]
    cp_best, cp_worst = min(times[s] for s in tied), max(times[s] for s in tied)
    return {
        "knobs": dict(knobs),
        "cores": cores,
        "peak_all": all_peak,
        "binds": all_peak > cap,
        "n_feasible": len(feasible),
        "best": best,
        "worst": worst,
        "spread": (worst / best) if best else float("inf"),
        "largest_feasible": max(len(s) for s in feasible),
        "n_buffers": len(names),
        "strict_regret": (cp_best - best_t) / best_t * 100.0 if best_t else 0.0,
        "tie_regret": (cp_worst - best_t) / best_t * 100.0 if best_t else 0.0,
        "n_tied": len(tied),
        "bw_spread": (max(bws.values()) / min(bws.values())) if bws else 1.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cores", type=int, nargs="+", default=[32, 8])
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"LX budget {LX_CAPACITY_BYTES / 1024:.0f} KB/core. A configuration is worth")
    print("compiling when capacity BINDS and several allocations remain feasible.\n")

    for name, prog in PROGRAMS.items():
        grid = KNOB_GRID[name]
        rows = []
        for cores in args.cores:
            for combo in itertools.product(*grid.values()):
                knobs = dict(zip(grid.keys(), combo))
                r = screen(prog, knobs, cores)
                if (
                    r
                    and r["binds"]
                    and r["n_feasible"] > 1
                    and r["largest_feasible"] > 0
                ):
                    rows.append(r)
        total = len(args.cores) * len(list(itertools.product(*grid.values())))
        rows.sort(key=lambda r: (-r["n_feasible"], -r["spread"]))
        print(
            f"=== {name}: {len(rows)} of {total} configurations are contested "
            f"({prog.buffers.__len__()} intermediates)"
        )
        if not rows:
            print("    none — capacity never binds, or nothing fits when it does\n")
            continue
        print(
            f"    {'knobs':<28} {'cores':>5} {'peak':>9} {'feasible':>9} "
            f"{'keeps':>6} {'spill spread':>13}"
        )
        for r in rows[: args.top]:
            k = " ".join(f"{a}={b}" for a, b in r["knobs"].items())
            print(
                f"    {k:<28} {r['cores']:>5} {r['peak_all'] / 1024:>8.0f}K "
                f"{r['n_feasible']:>9} {r['largest_feasible']:>2}/{r['n_buffers']:<3} "
                f"{r['spread']:>12.2f}x"
            )
        gaps = [r for r in rows if r["strict_regret"] > 1e-9]
        ties = [
            r for r in rows if r["tie_regret"] > 1e-9 and r["strict_regret"] <= 1e-9
        ]
        print()
        print(
            f"    CP-SAT strictly beaten by a time-ranked search: "
            f"{len(gaps)} of {len(rows)} contested configs"
        )
        for r in sorted(gaps, key=lambda r: -r["strict_regret"])[: args.top]:
            k = " ".join(f"{a}={b}" for a, b in r["knobs"].items())
            print(
                f"       {k:<28} {r['cores']:>3} cores   +{r['strict_regret']:>6.1f}%"
                f"   bw spread {r['bw_spread']:.2f}x"
            )
        if ties:
            print(
                f"    byte-optimal TIES that differ in time: {len(ties)} configs "
                f"(up to +{max(t['tie_regret'] for t in ties):.1f}%)"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
