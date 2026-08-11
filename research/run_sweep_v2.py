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

"""Five hours of measurements: what the first session could not answer.

The 150-minute session settled two things and opened four. It confirmed that the default LX
solver is 18.6 % and 26.4 % slower than CP-SAT's allocation on two untiled flash shapes, and
that the model ranks shape/core configurations perfectly WITHIN a core count (Kendall tau
+1.00, 36/36 concordant pairs, twice). It also left:

  * a THIRD flash shape where the model's ranking is exactly inverted -- measured time falls
    monotonically with LX residency (17 buffers 622 us, 16 -> 717, 15 -> 735) while the model
    calls the 16-buffer set optimal and the 17-buffer set worst;
  * a systematic core-count scale error -- pred/meas averages 0.83x at 32 cores and 0.57x at
    8, so the two core counts cannot be compared on one axis;
  * no case-study data at all: all seven P3 runs died on harness wiring, not on physics;
  * the `bmm_observation` layout study measured only at 32 cores, where a layout effect and
    a work-division effect are not separable.

Seven phases, priority-ordered under one wall-clock budget, each capped and donating what it
does not spend to the phases after it.

  S1  case studies, untiled, feature extraction        25 min   the missing (b) evidence
  S2  wider contested-shape search for flash           50 min   more (b), and the (c) search
  S3  core-count ladder at fixed shape                 25 min   isolates the 0.57x/0.83x gap
  S4  layout cube at ONE core                          35 min   layout effect without WD
  S5  work-division ladder at fixed layout             30 min   WD effect without layout
  S6  P1 confirmation, 4 rounds, fixed override check  30 min   is the inversion real?
  S7  re-sweep remainder, softmax and bmm first        rest     ~105 min

WHY S4 AND S5 ARE SEPARATE. The layout cube runs at 32 cores, so each cell mixes the layout
under test with whatever work division the compiler chose for it. At ONE core there is no
division to choose, so S4 measures layout alone; S5 then varies cores at a FIXED layout, so
it measures division alone. Neither is interpretable without the other, and the existing
report has only the product.

    python3 research/run_sweep_v2.py --dry-run
    python3 research/run_sweep_v2.py                 # the five-hour session
    python3 research/run_sweep_v2.py --resume        # skip what already measured
    python3 research/run_sweep_v2.py --phases 4,5    # just the bmm add-on
"""

import argparse
import itertools
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools", "cost_model"))

from run_lx_experiments import (  # noqa: E402
    Budget,
    Recorder,
    _save_log,
    classify_failure,
    run_one,
)

_SWEEP = os.path.join(
    _ROOT, "docs", "source", "user_guide", "examples", "run_cost_model_sweep.py"
)
_TOOLS = os.path.join(_ROOT, "tools", "cost_model")

#: S2: a wider net than the ten shapes probed before. Only four of those were contested, and
#: three of the four collapsed to two distinct allocations -- too few arms to separate a
#: ranking claim from luck. Contention needs one buffer to fit the 1587 KB budget while
#: several do not, so this walks H and the sequence length across the band at both core
#: counts, and adds B=2 to reach footprints an H change alone cannot.
S2_GRID = [
    {"FA_B": b, "FA_H": h, "FA_LQ": ell, "FA_LK": ell, "SENCORES": c}
    for b, h, ell, c in [
        (1, 4, 512, 8),
        (1, 4, 768, 8),
        (1, 6, 512, 8),
        (1, 8, 512, 8),
        (1, 4, 1024, 32),
        (1, 6, 1024, 32),
        (1, 8, 1024, 32),
        (1, 12, 1024, 32),
        (1, 16, 1024, 32),
        (1, 4, 1536, 32),
        (1, 6, 768, 8),
        (1, 8, 768, 8),
        (1, 16, 512, 32),
        (1, 24, 512, 32),
        (1, 32, 512, 32),
        (1, 4, 2048, 32),
        (2, 4, 512, 8),
        (2, 8, 512, 32),
        (2, 4, 1024, 32),
        (2, 6, 512, 8),
        (1, 12, 512, 32),
        (1, 6, 1536, 32),
        (1, 10, 1024, 32),
        (1, 5, 1024, 32),
    ]
]

#: S3: one shape, every core count. Cores change both the per-core footprint and the work
#: division, so this is the only way to see whether the 0.57x/0.83x gap is a smooth scaling
#: error or a step. Three shapes so the answer is not one shape's accident.
S3_SHAPES = [(8, 512), (4, 1024), (16, 512)]
S3_CORES = [1, 2, 4, 8, 16, 32]

#: S4/S5: the layout study's shape, so the new cells tie back to the tabulated numbers.
BMM_SHAPES = [(4, 1024, 2048, 1024), (2, 1024, 2048, 1024)]
LAYOUTS = ["0,1,2", "1,0,2"]
PREFS = ["", "output"]
S5_CORES = [1, 2, 4, 8, 16, 32]

#: S7: what the plan still lacks, biggest gaps first. `softmax_row_tiling` matters most --
#: it is the coarse-tiled op the LX study leaned on and it has no records on this toolchain.
S7_PRIORITY = [
    "softmax_row_tiling",
    "bmm_wd",
    "bmm_layout",
    "transpose_outer",
    "softmax_unrolled",
    "bmm_k_tiling",
    "bmm_wd_3d2d",
    "ctsum",
    "cat0",
    "cat1",
    "copy",
    "transpose",
    "sumrow",
    "mean",
    "neg",
    "mulbcast",
    "read",
    "write",
    "gelu",
]


def _case_configs():
    """Untiled case-study configurations, tile counts pinned to 1.

    Every P3 run failed on wiring: the factories default to bt=2/st=4/ft=4, so a config that
    sets only shapes still asked for coarse tiling, and `prefix_block` was handed a `seq`
    keyword it does not take. Both are fixed upstream; this re-derives from the screen.
    """
    try:
        import screen_untiled as SU
        from screen_configs import PROGRAMS
    except Exception:  # noqa: BLE001
        return []
    out = []
    for name in [
        "decode_block",
        "block_norm_mlp",
        "attn_scores",
        "prefix_block",
        "mlp_up",
    ]:
        prog = PROGRAMS.get(name)
        if prog is None:
            continue
        rows = []
        for scale, cores in itertools.product(SU.SCALES, SU.CORES):
            r = SU.screen(prog, scale, cores)
            if r and r["binds"] and r["keeps"] > 0:
                rows.append(r)
        rows.sort(key=lambda r: (-r["n_feasible"], -r["spread"]))
        seen = set()
        for r in rows:
            env = dict(SU.wl_env(name, r["dims"]))
            env.update({"WL_BT": 1, "WL_ST": 1, "WL_FT": 1, "WL_HT": 1, "WL_QT": 1})
            sig = (tuple(sorted(env.items())), r["cores"])
            if sig in seen:
                continue
            seen.add(sig)
            out.append((name, env, r["cores"], r["n_feasible"]))
            if len(seen) >= 2:
                break
    return out


def _record(rec, phase, key, env, s, lx, raw, dt, **extra):
    row = {
        "phase": phase,
        "key": key,
        "env": env,
        "seconds": round(dt, 1),
        "kernel_us": (s or {}).get("kernel_us"),
        "pred_us": (s or {}).get("pred_us"),
        "cv": (s or {}).get("kernel_us_cv"),
        "layout_c": (s or {}).get("layout_c"),
        "cores": (s or {}).get("cores"),
        "lx": lx,
        "n_lx": len(lx) if lx else None,
    }
    row.update(extra)
    if not row["kernel_us"]:
        row["tail"] = (raw or "")[-400:]
        row["log"] = _save_log(key, raw)
        row["failure"] = classify_failure(raw)
    rec.add(row)
    return row


def _loop(rec, args, phase, cap_s, plan, label):
    """Run a phase's plan under its cap, aborting only on repeated DEVICE failures."""
    t_end = time.time() + cap_s
    fails = 0
    for key, env, extra in plan:
        if time.time() > t_end:
            print("    cap reached, moving on")
            break
        if args.resume and key in rec.done:
            continue
        s, lx, raw, dt = run_one(env, args.run_timeout, reps=args.reps)
        row = _record(rec, phase, key, env, s, lx, raw, dt, **extra)
        if row["kernel_us"]:
            fails = 0
        elif row.get("failure") in ("device", "timeout"):
            fails += 1
        shown = (
            f"{row['kernel_us']:.1f} us"
            if row["kernel_us"]
            else (row.get("failure") or "FAILED")
        )
        print(f"    {label(key, extra):<52}{shown:>13}  ({dt:.0f}s)")
        if args.abort_after and fails >= args.abort_after:
            print(f"    ABORT: {fails} consecutive DEVICE failures")
            return False
    return True


def s1_cases(rec, args, cap_s):
    """Case studies, untiled, with features extracted so allocations can be solved."""
    print(f"\n=== S1  case studies, untiled  (cap {cap_s / 60:.0f} min) ===")
    cfgs = _case_configs()
    if not cfgs:
        print("    screen unavailable")
        return True
    plan = []
    for name, wl, cores, nfeas in cfgs:
        env = {
            "BENCH_OP": f"research:{name}",
            "SENCORES": str(cores),
            "BENCH_EMIT_RECORDS": "1",
        }
        env.update({k: str(v) for k, v in wl.items()})
        key = f"s1:{name}:c{cores}:" + "_".join(
            f"{k}{v}" for k, v in sorted(wl.items()) if not k.startswith("WL_BT")
        )
        plan.append(
            (
                key,
                env,
                {"case": name, "cores": cores, "screened_feasible": nfeas, "wl": wl},
            )
        )
    print(f"    {len(plan)} configuration(s)")
    return _loop(
        rec,
        args,
        1,
        cap_s,
        plan,
        lambda k,
        e: f"{e['case']:<16}{e['cores']:>3}c {e['screened_feasible']:>6} feas",
    )


def s2_shapes(rec, args, cap_s):
    """A wider contested-shape search, with features, for both (b) and (c)."""
    print(f"\n=== S2  wider flash shape search  (cap {cap_s / 60:.0f} min) ===")
    plan = []
    for g in S2_GRID:
        env = {
            "BENCH_OP": "flash_attn",
            "BENCH_EMIT_RECORDS": "1",
            "FA_D": "128",
            "FA_H_TILES": "1",
            "FA_LQ_TILES": "1",
            "FA_LK_TILES": "1",
            "LX_PLANNING": "1",
        }
        env.update({k: str(v) for k, v in g.items()})
        key = f"s2:B{g['FA_B']}_H{g['FA_H']}_L{g['FA_LQ']}_c{g['SENCORES']}"
        plan.append((key, env, {"shape": g}))
    print(f"    {len(plan)} shape(s)")
    return _loop(
        rec,
        args,
        2,
        cap_s,
        plan,
        lambda k, e: f"B{e['shape']['FA_B']} H{e['shape']['FA_H']} "
        f"L{e['shape']['FA_LQ']} {e['shape']['SENCORES']}c",
    )


def s3_cores(rec, args, cap_s):
    """One shape, every core count: is the scale error smooth or a step?"""
    print(f"\n=== S3  core-count ladder  (cap {cap_s / 60:.0f} min) ===")
    plan = []
    for (h, ell), cores in itertools.product(S3_SHAPES, S3_CORES):
        env = {
            "BENCH_OP": "flash_attn",
            "BENCH_EMIT_RECORDS": "1",
            "FA_B": "1",
            "FA_H": str(h),
            "FA_LQ": str(ell),
            "FA_LK": str(ell),
            "FA_D": "128",
            "FA_H_TILES": "1",
            "FA_LQ_TILES": "1",
            "FA_LK_TILES": "1",
            "SENCORES": str(cores),
        }
        plan.append(
            (f"s3:H{h}_L{ell}_c{cores}", env, {"H": h, "L": ell, "n_cores": cores})
        )
    print(
        f"    {len(plan)} point(s): {len(S3_SHAPES)} shapes x {len(S3_CORES)} core counts"
    )
    return _loop(
        rec,
        args,
        3,
        cap_s,
        plan,
        lambda k, e: f"H{e['H']} L{e['L']} {e['n_cores']:>2} cores",
    )


def s4_layout_1core(rec, args, cap_s):
    """The A x B x C layout cube at ONE core -- layout with no work division in it."""
    print(f"\n=== S4  layout cube at 1 core  (cap {cap_s / 60:.0f} min) ===")
    plan = []
    for (b, m, k, n), la, lb, pref in itertools.product(
        BMM_SHAPES, LAYOUTS, LAYOUTS, PREFS
    ):
        env = {
            "BENCH_OP": "bmm_layout",
            "SENCORES": "1",
            "BENCH_B": str(b),
            "BENCH_ROWS": str(m),
            "BENCH_COLS": str(k),
            "BENCH_N": str(n),
            "WD_LAYOUT_A": la,
            "WD_LAYOUT_B": lb,
            "SPYRE_MATMUL_PREFERRED_LAYOUT": pref,
            "BENCH_EMIT_RECORDS": "1",
            "SPYRE_DUMP_COST": "1",
        }
        key = f"s4:B{b}_{m}x{k}x{n}_A{la.replace(',', '')}_B{lb.replace(',', '')}_p{pref or 'off'}"
        plan.append(
            (
                key,
                env,
                {
                    "shape": [b, m, k, n],
                    "A": la,
                    "B": lb,
                    "pref": pref or "off",
                    "n_cores": 1,
                },
            )
        )
    print(f"    {len(plan)} cell(s): {len(BMM_SHAPES)} shapes x 2 A x 2 B x 2 pref")
    return _loop(
        rec,
        args,
        4,
        cap_s,
        plan,
        lambda k, e: f"B{e['shape'][0]} A={e['A']} B={e['B']} pref={e['pref']}",
    )


def s5_wd_ladder(rec, args, cap_s):
    """Work division at FIXED layout: the other half of the separation S4 starts."""
    print(
        f"\n=== S5  work-division ladder, fixed layout  (cap {cap_s / 60:.0f} min) ==="
    )
    b, m, k, n = BMM_SHAPES[0]
    plan = []
    for la, cores in itertools.product(LAYOUTS, S5_CORES):
        env = {
            "BENCH_OP": "bmm_layout",
            "SENCORES": str(cores),
            "BENCH_B": str(b),
            "BENCH_ROWS": str(m),
            "BENCH_COLS": str(k),
            "BENCH_N": str(n),
            "WD_LAYOUT_A": la,
            "WD_LAYOUT_B": "0,1,2",
            "BENCH_EMIT_RECORDS": "1",
            "SPYRE_DUMP_COST": "1",
        }
        plan.append(
            (
                f"s5:A{la.replace(',', '')}_c{cores}",
                env,
                {"A": la, "n_cores": cores, "shape": [b, m, k, n]},
            )
        )
    print(f"    {len(plan)} point(s): 2 A layouts x {len(S5_CORES)} core counts")
    return _loop(
        rec, args, 5, cap_s, plan, lambda k, e: f"A={e['A']} {e['n_cores']:>2} cores"
    )


def s6_confirm(rec, args, cap_s):
    """Re-measure the P1 allocations with four rounds and the corrected override check.

    The inverted case is the one that matters. Its arms measured 622/646/717/735 us with a
    round-to-round spread under 1 %, so it is almost certainly real -- but it contradicts the
    model's ranking, and a claim that specific deserves more than two rounds.
    """
    print(f"\n=== S6  P1 confirmation, 4 rounds  (cap {cap_s / 60:.0f} min) ===")
    path = os.path.join(_HERE, "forced_allocations.json")
    if not os.path.exists(path):
        print("    no forced_allocations.json; run emit_forced_allocations.py first")
        return True
    with open(path, encoding="utf-8") as fh:
        cases = json.load(fh)["cases"]
    plan = []
    for rnd in range(4):
        for c in cases:
            by_set: dict = {}
            for name, a in c["arms"].items():
                by_set.setdefault(tuple(a["lx"]), []).append(name)
            arms = list(by_set.items())
            if len(arms) < 2:
                continue
            if rnd % 2:
                arms = list(reversed(arms))
            for lx, names in arms:
                label = "+".join(names)
                env = {
                    "BENCH_OP": "flash_attn",
                    "BENCH_EMIT_RECORDS": "1",
                    "LX_FORCE_ONLY": ",".join(lx),
                }
                env.update({k: str(v) for k, v in c["env"].items()})
                key = f"s6:{c['label'][:28]}:{label}:r{rnd}"
                plan.append(
                    (
                        key,
                        env,
                        {
                            "case": c["label"],
                            "arm": label,
                            "round": rnd,
                            "requested_lx": list(lx),
                        },
                    )
                )
    print(f"    {len(plan)} run(s)")
    return _loop(
        rec,
        args,
        6,
        cap_s,
        plan,
        lambda k, e: f"{e['case'][11:38]:<28}{e['arm'][:14]:<16}r{e['round']}",
    )


def s7_resweep(rec, args, cap_s):
    """The plan's remaining ops, biggest gaps first."""
    print(f"\n=== S7  re-sweep remainder  (cap {cap_s / 60:.0f} min) ===")
    t_end = time.time() + cap_s
    logdir = os.path.join(_ROOT, "sweep_logs")
    os.makedirs(logdir, exist_ok=True)
    for op in S7_PRIORITY:
        left = t_end - time.time()
        if left < 120:
            print("    cap reached")
            break
        log = os.path.join(logdir, f"v2_{op}.log")
        print(f"    {op:<22} up to {left / 60:.0f} min")
        try:
            subprocess.run(
                [
                    sys.executable,
                    _SWEEP,
                    "--op",
                    op,
                    "--skip-measured",
                    "--reps",
                    str(args.reps),
                    "--timeout",
                    "240",
                    "--abort-after",
                    "6",
                    "--out",
                    log,
                ],
                timeout=left,
                check=False,
            )
        except subprocess.TimeoutExpired:
            rec.add({"phase": 7, "key": f"s7:{op}", "op": op, "status": "timeout"})
            break
        rec.add({"phase": 7, "key": f"s7:{op}", "op": op, "status": "done"})
    logs = [
        os.path.join(logdir, f)
        for f in sorted(os.listdir(logdir))
        if f.startswith("v2_")
    ]
    if logs:
        print(f"    folding {len(logs)} log(s) into the database")
        subprocess.run(
            [sys.executable, os.path.join(_TOOLS, "parse_sweep_logs.py"), *logs],
            check=False,
        )
    return True


PHASES = {
    1: s1_cases,
    2: s2_shapes,
    3: s3_cores,
    4: s4_layout_1core,
    5: s5_wd_ladder,
    6: s6_confirm,
}
CAPS = {1: 25 * 60, 2: 50 * 60, 3: 25 * 60, 4: 35 * 60, 5: 30 * 60, 6: 30 * 60}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget-min", type=float, default=300.0)
    ap.add_argument("--phases", default="1,2,3,4,5,6,7")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--run-timeout", type=int, default=420)
    ap.add_argument("--abort-after", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(_HERE, "sweep_v2_results.jsonl"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    want = {int(x) for x in args.phases.split(",") if x.strip()}
    rec = Recorder(args.out)
    n = (
        len(_case_configs())
        + len(S2_GRID)
        + len(S3_SHAPES) * len(S3_CORES)
        + len(BMM_SHAPES) * 8
        + len(LAYOUTS) * len(S5_CORES)
    )
    print(
        f"budget {args.budget_min:.0f} min | phases {sorted(want)} | "
        f"~{n} device runs before S7"
    )
    if args.dry_run:
        for ph in sorted(want):
            if ph in CAPS:
                print(f"  S{ph}  cap {CAPS[ph] / 60:.0f} min")
        print(f"  S7  remainder, ops: {', '.join(S7_PRIORITY[:6])}, ...")
        return 0

    budget = Budget(args.budget_min * 60)
    for ph in sorted(p for p in want if p in PHASES):
        if budget.remaining() < 60:
            break
        if not PHASES[ph](rec, args, budget.phase_cap(CAPS[ph])):
            print(f"S{ph} aborted on device failures; stopping")
            want.discard(7)
            break
    if 7 in want and budget.remaining() > 120:
        s7_resweep(rec, args, budget.phase_cap(0, is_last=True))
    print(f"\nsession used {budget.elapsed() / 60:.1f} of {args.budget_min:.0f} min")
    print(f"raw results: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
