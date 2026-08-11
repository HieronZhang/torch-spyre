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

"""The tiling x LX-residency design space of the Granite case studies.

WHAT QUESTION THIS ANSWERS. For each real-model program in `research/workloads.py`, and at
each coarse-tile count, does LX capacity bind? If it does, how many residency choices remain,
how much do they differ, and is a *smaller tile* ever the better answer than a better
allocation at a larger tile? That last one is the interaction that makes the space large, and
it cannot be seen by varying residency at fixed tiling.

WHAT IT IS COMPUTED FROM, AND THE LIMIT OF THAT. Shapes and the op sequence, analytically --
NOT from compiled features. Tiling dimension `S` into `T` tiles across `cores` cores gives
each core `S / (T * cores)` rows of each intermediate per tile, and liveness comes from the
program's own dependence order. That is enough to answer "does it fit" and "how much HBM
traffic does spilling add", both of which follow from shapes alone.

It is NOT enough to predict time. The cost model needs the extractor's `OpFeatures`, and
this program has never been compiled -- so the traffic column here is the byte-proxy view,
the same quantity CP-SAT optimises, and not a runtime. Where traffic and time diverge is
exactly the corner case in `findings_lx.md` section 6. Treat the contested cells below as
"the allocator has a real decision here", which is what was asked for, and not as a
performance claim.

    python3 research/design_space.py
    python3 research/design_space.py --workload swiglu_mlp --verbose
"""

import argparse
import itertools

#: `_lx_planning_size()` at DXP_LX_FRAC_AVAIL=0.2 (scratchpad/allocator.py:838-855).
LX_CAPACITY_BYTES = 1_625_344
GRANITE = {"d_model": 4096, "d_ff": 12800, "head_dim": 128}


class Program:
    """A tiled program as (ops, intermediates), enough to ask capacity questions.

    `intermediates` maps a buffer name to (producer op index, consumer op indices, columns).
    Its per-core tile footprint is `rows_per_core_per_tile * columns * dtype_bytes`, and its
    live interval runs from the producer to the last consumer -- the same interval model the
    allocator uses (`scratchpad/utils.py:84`, `plan_solver.py:75-81`).
    """

    def __init__(self, name, ops, intermediates, tiled_rows, note=""):
        self.name = name
        self.ops = ops
        self.intermediates = intermediates
        self.tiled_rows = tiled_rows
        self.note = note

    def footprints(self, tiles, cores=32, dtype_bytes=2):
        rows = self.tiled_rows / (tiles * cores)
        return {
            n: rows * cols * dtype_bytes
            for n, (_, _, cols) in self.intermediates.items()
        }

    def lifetimes(self):
        return {
            n: (prod, max(cons) + 1)
            for n, (prod, cons, _) in self.intermediates.items()
        }

    def peak(self, subset, tiles, cores=32):
        fp, life = self.footprints(tiles, cores), self.lifetimes()
        return max(
            (
                sum(fp[n] for n in subset if life[n][0] <= t < life[n][1])
                for t in range(len(self.ops) + 1)
            ),
            default=0,
        )

    def spilled_traffic(self, subset, tiles, cores=32, dtype_bytes=2):
        """HBM bytes added by spilling everything NOT in `subset`, over the whole loop.

        A spilled intermediate is written once and re-read by each consumer, and that
        happens on every one of the `tiles` iterations. This is CP-SAT's `spill_cost`
        summed over the loop.
        """
        rows = self.tiled_rows / (tiles * cores)
        total = 0.0
        for n, (_, cons, cols) in self.intermediates.items():
            if n in subset:
                continue
            total += (1 + len(cons)) * rows * cols * dtype_bytes * tiles * cores
        return total


def granite_swiglu(seq=2048):
    """down(silu(x @ Wg) * (x @ Wu)) -- three full-width intermediates at `d_ff`."""
    f = GRANITE["d_ff"]
    return Program(
        "swiglu_mlp",
        ["mm_gate", "mm_up", "silu", "mul", "mm_down"],
        {
            "gate": (0, [2], f),
            "up": (1, [3], f),
            "act": (2, [3], f),
            "prod": (3, [4], f),
        },
        tiled_rows=seq,
        note=f"seq={seq} d_model={GRANITE['d_model']} d_ff={f}",
    )


def granite_rmsnorm(seq=2048):
    """RMSNorm + residual: a tiny cross-row reduction against full-width activations."""
    d = GRANITE["d_model"]
    return Program(
        "rmsnorm_residual",
        ["square", "mean", "rsqrt", "scale", "weight", "add"],
        {
            "sq": (0, [1], d),
            "ms": (1, [2], 1),
            "inv": (2, [3], 1),
            "xn": (3, [4], d),
            "xw": (4, [5], d),
        },
        tiled_rows=seq,
        note=f"seq={seq} d_model={d}",
    )


def granite_block(seq=1024):
    """norm -> MLP -> residual: reduction, elementwise and matmul operands together."""
    d, f = GRANITE["d_model"], GRANITE["d_ff"]
    return Program(
        "transformer_block",
        [
            "square",
            "mean",
            "scale",
            "mm_gate",
            "mm_up",
            "silu",
            "mul",
            "mm_down",
            "add",
        ],
        {
            "sq": (0, [1], d),
            "ms": (1, [2], 1),
            "xn": (2, [3, 4], d),
            "gate": (3, [5], f),
            "up": (4, [6], f),
            "act": (5, [6], f),
            "prod": (6, [7], f),
            "out": (7, [8], d),
        },
        tiled_rows=seq,
        note=f"seq={seq} d_model={d} d_ff={f}",
    )


PROGRAMS = {p.name: p for p in (granite_swiglu(), granite_rmsnorm(), granite_block())}


def analyse(prog, tile_counts=(1, 2, 4, 8, 16, 32), cores=32, cap=LX_CAPACITY_BYTES):
    """Per tile count: does capacity bind, and how much does the choice matter?"""
    rows = []
    for t in tile_counts:
        fp = prog.footprints(t, cores)
        names = sorted(fp)
        feasible = [
            frozenset(c)
            for r in range(len(names) + 1)
            for c in itertools.combinations(names, r)
            if prog.peak(set(c), t, cores) <= cap
        ]
        all_fits = frozenset(names) in feasible
        best = min(feasible, key=lambda s: prog.spilled_traffic(s, t, cores))
        worst = max(feasible, key=lambda s: prog.spilled_traffic(s, t, cores))
        rows.append(
            {
                "tiles": t,
                "per_tile_kb": max(fp.values()) / 1024,
                "peak_all_kb": prog.peak(set(names), t, cores) / 1024,
                "fits_all": all_fits,
                "n_feasible": len(feasible),
                "best": best,
                "best_traffic": prog.spilled_traffic(best, t, cores),
                "worst_traffic": prog.spilled_traffic(worst, t, cores),
            }
        )
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload", default="", choices=[""] + sorted(PROGRAMS))
    ap.add_argument("--cores", type=int, default=32)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"LX budget {LX_CAPACITY_BYTES / 1024:.0f} KB/core, {args.cores} cores, fp16")
    print(
        "Traffic is HBM bytes added by spilling, summed over the whole loop -- the byte"
    )
    print("proxy, NOT a predicted time. See the module docstring.\n")

    for name in [args.workload] if args.workload else sorted(PROGRAMS):
        prog = PROGRAMS[name]
        print(f"=== {name}   {prog.note}")
        print(f"    {len(prog.ops)} ops, {len(prog.intermediates)} intermediates\n")
        print(
            f"    {'tiles':>5} {'largest tile':>13} {'peak if all':>12} {'all fit?':>9} "
            f"{'choices':>8} {'best spill':>12} {'worst spill':>12} {'ratio':>7}"
        )
        for r in analyse(prog, cores=args.cores):
            ratio = (
                (r["worst_traffic"] / r["best_traffic"]) if r["best_traffic"] else 1.0
            )
            contested = (not r["fits_all"]) and r["n_feasible"] > 1 and ratio > 1.01
            print(
                f"    {r['tiles']:>5} {r['per_tile_kb']:>12.0f}K {r['peak_all_kb']:>11.0f}K "
                f"{('yes' if r['fits_all'] else 'NO'):>9} {r['n_feasible']:>8} "
                f"{r['best_traffic'] / 2**20:>11.0f}M {r['worst_traffic'] / 2**20:>11.0f}M "
                f"{ratio:>6.2f}x{'   <- CONTESTED' if contested else ''}"
            )
            if args.verbose and contested:
                print(f"          best keeps {sorted(r['best'])}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
