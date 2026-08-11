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

"""Find contested LX allocations in UNTILED flash attention.

WHY THIS CAN WORK. Coarse tiling is what broke, not flash. Untiled flash compiles today
(measured 8369 us at H=32 Lq=Lk=1024, 192 us at H=8 Lq=Lk=512) and still emits the same
~20-buffer online-softmax chain -- tiling changed the loop nest, not the program. And tiling
was never what made the allocation contested: SIZE was. A bundle is contested when one
buffer fits the 1587 KB budget and several do not, and for untiled flash the per-core
footprint is just `B*H*Lq*Lk*2/cores` -- reachable by shrinking H, the sequence length, or
the core count, none of which needs a tile hint.

Analytically, 10 of 24 (H, Lq=Lk, cores) combinations land in that band, most of them at
shapes small enough to run in well under a second.

WHAT THIS DOES. Runs each candidate, captures the measured time AND the extracted features,
writes them in database format, then solves every allocation exactly to report which shapes
are genuinely contested and by how much. The output feeds `emit_forced_allocations.py`
directly, so a contested shape becomes a measurable P1 case with no further work.

    python3 research/probe_untiled_flash.py                 # ~10 min
    python3 research/probe_untiled_flash.py --analyse-only   # re-solve without hardware
"""

import argparse
import json
import os
import subprocess
import sys
import time

import regex as re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools", "cost_model"))
_PROFILE = os.path.join(
    _ROOT, "docs", "source", "user_guide", "examples", "profile_ops.py"
)

_SUMMARY = re.compile(r"^SUMMARY .*?kernel_us=([0-9.]+)", re.M)
_FEATS = re.compile(r"^MODEL FEATS (.+)$", re.M)

CAP = 1_625_344


def candidates():
    """(H, L, cores) where one score buffer fits the budget but several do not.

    The score-chain buffers are `[B, H, Lq, Lk]` and dominate everything else, so this is a
    close enough filter to choose what to compile. It is a screen, not a claim -- the real
    footprints come from the extracted features below.
    """
    out = []
    for H in (4, 8, 16, 32):
        for L in (512, 1024, 2048):
            for cores in (32, 8):
                s = H * L * L * 2 // cores
                k = CAP // s if s else 0
                if 1 <= k <= 8:
                    out.append(
                        {"H": H, "L": L, "cores": cores, "per_core": s, "fit": k}
                    )
    # Roomiest first: more buffers fitting means a larger feasible set to choose from.
    out.sort(key=lambda c: (-c["fit"], c["H"] * c["L"] * c["L"]))
    return out


def run_one(c, reps, timeout_s):
    env = dict(os.environ)
    env.update(
        {
            "BENCH_OP": "flash_attn",
            "BENCH_EMIT_RECORDS": "1",
            "BENCH_REPS": str(reps),
            "BENCH_WARMUP": "1",
            "FA_B": "1",
            "FA_H": str(c["H"]),
            "FA_LQ": str(c["L"]),
            "FA_LK": str(c["L"]),
            "FA_D": "128",
            "FA_H_TILES": "1",
            "FA_LQ_TILES": "1",
            "FA_LK_TILES": "1",
            "SENCORES": str(c["cores"]),
            "LX_PLANNING": "1",
        }
    )
    t0 = time.time()
    try:
        p = subprocess.run(
            [sys.executable, _PROFILE],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        raw = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        raw = "TIMEOUT"
    m, f = _SUMMARY.search(raw), _FEATS.search(raw)
    feats = None
    if f:
        try:
            feats = json.loads(f.group(1))
        except Exception:  # noqa: BLE001
            feats = None
    return (float(m[1]) if m else None), feats, raw, time.time() - t0


def as_record(c, us, feats):
    """Database-shaped record, so the existing tooling can consume it unchanged."""
    return {
        "op": "flash_attn",
        "label": (
            f"flash_attn H={c['H']} Lq={c['L']} Lk={c['L']} D=128 "
            f"htiles=1 qtiles=1 ktiles=1 sencores={c['cores']} (untiled)"
        ),
        "kernel_us": us,
        "feats": feats,
        "cores": c["cores"],
        "rows": c["L"],
        "cols": 128,
        "tiles": 1,
        "lx": 1,
        "failed": False,
    }


def analyse(records):
    """Solve every allocation exactly and report what is actually contested."""
    import lx_choice as L
    import lx_experiment as X

    rows = []
    for r in records:
        feats = r.get("feats")
        if not feats or not r.get("kernel_us"):
            continue
        movable = L.bundle_intermediates(feats)
        if not movable:
            continue
        peak = L.peak_footprint(feats, set(movable))
        if peak <= CAP:
            rows.append(
                {
                    "label": r["label"],
                    "movable": len(movable),
                    "peak": peak,
                    "status": "fits entirely - no choice",
                }
            )
            continue
        exact, ev, n = L.exhaustive_policies(feats, CAP)
        if n < 2:
            rows.append(
                {
                    "label": r["label"],
                    "movable": len(movable),
                    "peak": peak,
                    "status": "nothing fits - no choice",
                }
            )
            continue
        seq = X.solver_allocations(feats)
        ref = ev(exact["time"])
        regr = {
            "cpsat": (ev(exact["cpsat"]) - ref) / ref * 100.0,
            "greedy": (ev(seq["greedy"]) - ref) / ref * 100.0,
            "bestfit": (ev(seq["bestfit"]) - ref) / ref * 100.0,
        }
        distinct = len(
            {
                tuple(sorted(s))
                for s in (exact["time"], exact["cpsat"], seq["greedy"], seq["bestfit"])
            }
        )
        rows.append(
            {
                "label": r["label"],
                "movable": len(movable),
                "peak": peak,
                "feasible": n,
                "regret": regr,
                "distinct": distinct,
                "measured_us": r["kernel_us"],
                "status": "CONTESTED",
            }
        )
    return rows


def report(rows):
    good = [r for r in rows if r["status"] == "CONTESTED"]
    print(f"\n{len(good)} of {len(rows)} shape(s) are contested\n")
    if good:
        print(
            f"  {'shape':<52}{'meas us':>9}{'feas':>8}{'arms':>5}"
            f"{'greedy':>9}{'bestfit':>9}{'cpsat':>8}"
        )
        for r in sorted(good, key=lambda r: -r["regret"]["greedy"]):
            g = r["regret"]
            lbl = r["label"].replace("flash_attn ", "").replace(" (untiled)", "")
            print(
                f"  {lbl[:51]:<52}{r['measured_us']:>9.0f}{r['feasible']:>8,}"
                f"{r['distinct']:>5}{g['greedy']:>+8.1f}%{g['bestfit']:>+8.1f}%"
                f"{g['cpsat']:>+7.1f}%"
            )
    for r in rows:
        if r["status"] != "CONTESTED":
            lbl = r["label"].replace("flash_attn ", "").replace(" (untiled)", "")
            print(
                f"  {lbl[:51]:<52} {r['status']} "
                f"({r['movable']} movable, peak {r['peak'] / 1024:.0f}K)"
            )
    if good:
        best = max(good, key=lambda r: r["regret"]["greedy"])
        print("\nBest case for P1:")
        print(f"  {best['label']}")
        print(
            f"  {best['feasible']:,} feasible allocations, "
            f"default solver {best['regret']['greedy']:+.1f}% off optimal"
        )
        print("\nNext:")
        print("  python3 research/emit_forced_allocations.py \\")
        print("      --records research/untiled_flash_records.json --op flash_attn")
        print("  python3 research/run_lx_experiments.py \\")
        print("      --records research/untiled_flash_records.json --phases 1")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--budget-min", type=float, default=15.0)
    ap.add_argument(
        "--analyse-only",
        action="store_true",
        help="re-solve an existing records file; no hardware",
    )
    ap.add_argument("--out", default=os.path.join(_HERE, "untiled_flash_records.json"))
    args = ap.parse_args()

    if args.analyse_only:
        with open(args.out, encoding="utf-8") as fh:
            recs = json.load(fh)["records"]
        report(analyse(recs))
        return 0

    cands = candidates()
    print(f"{len(cands)} candidate shape(s) in the contested band, roomiest first\n")
    print(
        f"  {'H':>4}{'Lq=Lk':>7}{'cores':>7}{'per-core':>10}{'fits':>6}"
        f"{'measured':>11}{'movable':>9}"
    )
    t_end = time.time() + args.budget_min * 60
    recs = []
    for c in cands:
        if time.time() > t_end:
            print("  budget reached")
            break
        us, feats, raw, dt = run_one(c, args.reps, args.timeout)
        rec = as_record(c, us, feats)
        recs.append(rec)
        nmov = "-"
        if feats:
            try:
                import lx_choice as L

                nmov = str(len(L.bundle_intermediates(feats)))
            except Exception:  # noqa: BLE001
                pass
        shown = f"{us:.0f}" if us else (raw or "")[-60:].strip().split("\n")[-1][:24]
        print(
            f"  {c['H']:>4}{c['L']:>7}{c['cores']:>7}{c['per_core'] / 1024:>9.0f}K"
            f"{c['fit']:>6}{shown:>11}{nmov:>9}  ({dt:.0f}s)"
        )
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"records": recs}, fh)

    report(analyse(recs))
    print(f"\nrecords: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
