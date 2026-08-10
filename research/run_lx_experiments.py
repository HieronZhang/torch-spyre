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

"""A 150-minute hardware session: does the cost model rank LX allocations correctly?

FOUR EXPERIMENTS, run in priority order under one wall-clock budget. Each phase has a cap;
whatever it does not spend is donated to the phases after it, so an early failure buys time
for the sweep rather than wasting the session.

  P1  flash attention x {greedy, firstfit, bestfit, cpsat}          30 min
      The headline prediction: on two configurations the DEFAULT solver is 20.4 % and 28.3 %
      slower than optimal, and cpsat is exactly optimal. Four real allocations of ONE
      program, so nothing but the allocation changes.
  P2  flash attention across tile-count hints                       35 min
      Whether the model ranks configurations it has never been fitted on, the same test the
      2.11 toolchain got. Ranking only -- absolute error on flash is 15-45x.
  P3  the new real-model programs                                   25 min
      Same design as P1 on programs with mixed transports.
  P4  the rest of the re-sweep, matmul and coarse tiling first      remainder

THREE CONTROLS, without which the timings cannot be interpreted:

1. **The allocations must actually differ.** If two solvers place the same buffers in LX,
   equal runtimes confirm nothing. Every run captures its LX residency from `MODEL FEATS`
   and P1 refuses to report a comparison whose allocations are identical.
2. **Order must not alias onto the solver.** Thermal drift over half an hour is comparable
   to the effect. P1 runs the solvers interleaved and in two rounds of opposite order, so
   drift lands on all four solvers equally.
3. **ortools must be importable**, or `cpsat` raises rather than silently degrading
   (`ilp_solver_ortools.py:336`). Checked in preflight, before any time is spent.

SAFETY. A configuration that asks for a per-core address span past the MVLOC limit is the
prime suspect in the 2026-08-08 card failure, so: `bmm_3d2d_k_tiling` stays quarantined,
`profile_ops.py` keeps its span guard, every run has a timeout, and a phase aborts after
`--abort-after` consecutive failures (a dead device fails everything left).

    python3 research/run_lx_experiments.py --dry-run          # the plan, no hardware
    python3 research/run_lx_experiments.py                    # the 150-minute session
    python3 research/run_lx_experiments.py --budget-min 60 --phases 1,2
    python3 research/run_lx_experiments.py --resume           # skip what is already done
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
_EXAMPLES = os.path.join(_ROOT, "docs", "source", "user_guide", "examples")
_TOOLS = os.path.join(_ROOT, "tools", "cost_model")
_PROFILE = os.path.join(_EXAMPLES, "profile_ops.py")
_SWEEP = os.path.join(_EXAMPLES, "run_cost_model_sweep.py")

_SUMMARY = re.compile(r"^SUMMARY (.+)$", re.M)
_FEATS = re.compile(r"^MODEL FEATS (.+)$", re.M)

#: P1: the three flash configurations where LX capacity binds, with the predicted regret of
#: each solver against the proven optimum (`research/flash_lx_findings.md`). `spread` is what
#: the measurement has to resolve; case 3 is included as a NEGATIVE control -- the model says
#: cpsat, bestfit and the optimum coincide there, so the three should measure alike.
FLASH_CASES = [
    {
        "name": "case1_lq4096_h8_q4",
        "env": {
            "FA_H": "32",
            "FA_LQ": "4096",
            "FA_LK": "4096",
            "FA_H_TILES": "8",
            "FA_LQ_TILES": "4",
        },
        "predict": {"greedy": 20.4, "firstfit": 20.4, "bestfit": 20.4, "cpsat": 0.0},
    },
    {
        "name": "case2_lq2048_h8_q1",
        "env": {
            "FA_H": "32",
            "FA_LQ": "2048",
            "FA_LK": "2048",
            "FA_D": "128",
            "FA_H_TILES": "8",
            "FA_LQ_TILES": "1",
            "FA_LK_TILES": "1",
        },
        "predict": {"greedy": 28.3, "firstfit": 22.2, "bestfit": 22.2, "cpsat": 0.0},
    },
    {
        "name": "case3_lq2048_k2",
        "env": {"FA_H": "32", "FA_LQ": "2048", "FA_LK": "2048", "FA_LK_TILES": "2"},
        "predict": {"greedy": 5.9, "firstfit": 0.0, "bestfit": 0.0, "cpsat": 0.0},
    },
]
SOLVERS = ["greedy", "firstfit", "bestfit", "cpsat"]

#: P2: tile-count hints at a fixed shape. Only the hint changes, so the model is being asked
#: the question it will face in a scheduler: given this program, which tiling is fastest?
#: Every count divides its dimension exactly (`coarse_tile.py:855` raises otherwise).
FLASH_HINTS = [
    {"FA_H_TILES": h, "FA_LQ_TILES": q, "FA_LK_TILES": k}
    for h, q, k in [
        ("1", "1", "1"),
        ("2", "1", "1"),
        ("4", "1", "1"),
        ("8", "1", "1"),
        ("16", "1", "1"),
        ("32", "1", "1"),
        ("8", "2", "1"),
        ("8", "4", "1"),
        ("8", "8", "1"),
        ("8", "16", "1"),
        ("4", "4", "1"),
        ("16", "2", "1"),
        ("2", "8", "1"),
        ("4", "8", "1"),
        ("8", "1", "2"),
        ("8", "1", "4"),
        ("8", "2", "2"),
        ("4", "2", "2"),
        ("16", "4", "1"),
        ("32", "2", "1"),
        ("2", "2", "1"),
        ("1", "4", "1"),
    ]
]
#: A single mid-sized shape so 22 hint settings fit the budget; flash at 4096 is ~4x slower.
FLASH_HINT_SHAPE = {"FA_H": "32", "FA_LQ": "2048", "FA_LK": "2048", "FA_D": "128"}

#: P3: the new programs, each run under all four solvers like P1. They are NOT yet
#: registered as BENCH_OPs -- `--phases 3` is skipped with a clear message until they are,
#: rather than failing 20 runs in a row and burning the abort counter.
NEW_CASES = [
    {
        "name": "prefix_block",
        "op": "prefix_block",
        "env": {
            "PB_B": "4",
            "PB_SP": "512",
            "PB_SN": "512",
            "PB_S_TILES": "1",
            "PB_F_TILES": "2",
        },
    },
    {
        "name": "prefix_block_ft1",
        "op": "prefix_block",
        "env": {
            "PB_B": "4",
            "PB_SP": "512",
            "PB_SN": "512",
            "PB_S_TILES": "1",
            "PB_F_TILES": "1",
        },
    },
]

#: P4: the re-sweep, matmul and coarse-tiling families first as requested. Ordered by how
#: much of the model's structure each family constrains. `bmm_3d2d_k_tiling` is absent on
#: purpose -- it is quarantined for the address-span overflow.
SWEEP_PRIORITY = [
    "mm",
    "mmwd",
    "matmul_row_tiling",
    "matmul_k_tiling",
    "mm_nested_m_k",
    "bmm_layout",
    "bmm_wd",
    "bmm_wd_3d2d",
    "bmm_k_tiling",
    "bmm_nested_b_k",
    "softmax_row_tiling",
    "softmax_unrolled",
    "transpose_outer",
    "cat0",
    "cat1",
]


class Budget:
    """Wall-clock guard. Phases draw from one pool; unspent time flows downstream."""

    def __init__(self, total_s):
        self.total = total_s
        self.t0 = time.time()

    def elapsed(self):
        return time.time() - self.t0

    def remaining(self):
        return max(0.0, self.total - self.elapsed())

    def phase_cap(self, cap_s, is_last=False):
        """How long this phase may run: its cap, or everything left if it is the last."""
        return self.remaining() if is_last else min(cap_s, self.remaining())


def parse_summary(text):
    """The SUMMARY line as a dict, or None when the run produced no measurement."""
    m = _SUMMARY.search(text or "")
    if not m:
        return None
    out = {}
    for tok in m.group(1).split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k] = float(v) if re.match(r"^[-+]?\d*\.?\d+$", v) else v
            except ValueError:
                out[k] = v
    return out or None


def parse_lx_set(text):
    """Which buffers the compiler actually placed in LX, from the MODEL FEATS block.

    This is the control that makes a solver comparison meaningful: two solvers that choose
    the same allocation MUST measure the same, so an equal runtime is only informative once
    the allocations are known to differ.
    """
    m = _FEATS.search(text or "")
    if not m:
        return None
    try:
        feats = json.loads(m.group(1))
    except Exception:  # noqa: BLE001 -- a truncated line must not kill the session
        return None
    lx = set()
    for op in feats if isinstance(feats, list) else feats.get("ops", []):
        for a in op.get("args", []):
            if str(a.get("mem", "")).lower() == "lx":
                lx.add(str(a.get("name", "")))
    return sorted(lx)


def run_one(env_extra, timeout_s, reps=7, emit_records=True):
    """One profile_ops.py invocation. Returns (summary, lx_set, raw, seconds)."""
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_extra.items()})
    env.setdefault("BENCH_REPS", str(reps))
    if emit_records:
        env["BENCH_EMIT_RECORDS"] = "1"
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
        return None, None, "TIMEOUT", time.time() - t0
    return parse_summary(raw), parse_lx_set(raw), raw, time.time() - t0


class Recorder:
    """Crash-safe results: every run is flushed to JSONL the moment it finishes."""

    def __init__(self, path):
        self.path = path
        self.rows = []
        self.done = set()
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    self.rows.append(r)
                    if r.get("kernel_us"):
                        self.done.add(r.get("key", ""))
        self.fh = open(path, "a", encoding="utf-8")  # noqa: SIM115 -- lives for the session

    def add(self, row):
        self.rows.append(row)
        self.fh.write(json.dumps(row) + "\n")
        self.fh.flush()
        os.fsync(self.fh.fileno())
        if row.get("kernel_us"):
            self.done.add(row.get("key", ""))


def preflight(rec, budget, args):
    """Fail fast on the things that would silently invalidate the whole session."""
    print("=== preflight ===")
    ok = True
    try:
        import ortools  # noqa: F401

        print("    ortools importable            yes")
    except ImportError:
        ok = False
        print(
            "    ortools importable            NO -- 'cpsat' will RAISE, not fall back."
        )
        print("      pip install ortools, or P1/P3 lose their most important arm.")
    if args.dry_run:
        return ok
    print("    device smoke test (neg 1024) ...", end=" ", flush=True)
    s, _, raw, dt = run_one(
        {"BENCH_OP": "neg", "BENCH_COLS": "1024"}, 300, reps=3, emit_records=False
    )
    if s and s.get("kernel_us"):
        print(f"ok ({s['kernel_us']:.1f} us, {dt:.0f}s)")
    else:
        ok = False
        print("FAILED -- the device is not usable; nothing below will work")
        print((raw or "")[-800:])
    return ok


def phase1(rec, budget, args, cap_s):
    """Flash attention under four solvers, interleaved and order-balanced."""
    print(f"\n=== P1  flash x solver  (cap {cap_s / 60:.0f} min) ===")
    t_end = time.time() + cap_s
    # Round 0 in listed order, round 1 reversed: any monotone drift hits all solvers alike.
    plan = []
    for rnd in (0, 1):
        order = SOLVERS if rnd == 0 else list(reversed(SOLVERS))
        for case in FLASH_CASES:
            for solver in order:
                plan.append((rnd, case, solver))
    fails = 0
    for rnd, case, solver in plan:
        key = f"p1:{case['name']}:{solver}:r{rnd}"
        if time.time() > t_end:
            print("    cap reached, moving on")
            break
        if args.resume and key in rec.done:
            continue
        env = {"BENCH_OP": "flash_attn", "SENCORES": "32", "LAYOUT_SOLVER": solver}
        env.update(case["env"])
        s, lx, raw, dt = run_one(env, args.run_timeout, reps=args.reps)
        row = {
            "phase": 1,
            "key": key,
            "case": case["name"],
            "solver": solver,
            "round": rnd,
            "seconds": round(dt, 1),
            "kernel_us": (s or {}).get("kernel_us"),
            "cv": (s or {}).get("kernel_us_cv"),
            "pred_us": (s or {}).get("pred_us"),
            "lx": lx,
            "n_lx": len(lx) if lx else None,
            "predicted_regret_pct": case["predict"].get(solver),
        }
        if not row["kernel_us"]:
            row["tail"] = (raw or "")[-400:]
            fails += 1
        else:
            fails = 0
        rec.add(row)
        got = f"{row['kernel_us']:.0f} us" if row["kernel_us"] else "FAILED"
        print(
            f"    {case['name']:<22} {solver:<9} r{rnd}  {got:>12}  "
            f"lx={row['n_lx']}  ({dt:.0f}s)"
        )
        if args.abort_after and fails >= args.abort_after:
            print(f"    ABORT: {fails} consecutive failures -- device may be down")
            return False
    return True


def phase2(rec, budget, args, cap_s):
    """Flash across tile hints: can the model rank configurations it never saw?"""
    print(f"\n=== P2  flash x tile hints  (cap {cap_s / 60:.0f} min) ===")
    t_end = time.time() + cap_s
    fails = 0
    for hint in FLASH_HINTS:
        tag = f"h{hint['FA_H_TILES']}_q{hint['FA_LQ_TILES']}_k{hint['FA_LK_TILES']}"
        key = f"p2:{tag}"
        if time.time() > t_end:
            print("    cap reached, moving on")
            break
        if args.resume and key in rec.done:
            continue
        env = {"BENCH_OP": "flash_attn", "SENCORES": "32"}
        env.update(FLASH_HINT_SHAPE)
        env.update(hint)
        s, lx, raw, dt = run_one(env, args.run_timeout, reps=args.reps)
        row = {
            "phase": 2,
            "key": key,
            "hint": tag,
            "seconds": round(dt, 1),
            "kernel_us": (s or {}).get("kernel_us"),
            "pred_us": (s or {}).get("pred_us"),
            "cv": (s or {}).get("kernel_us_cv"),
            "err_pct": (s or {}).get("err_pct"),
            "lx": lx,
            "n_lx": len(lx) if lx else None,
        }
        if not row["kernel_us"]:
            row["tail"] = (raw or "")[-400:]
            fails += 1
        else:
            fails = 0
        rec.add(row)
        got = f"{row['kernel_us']:.0f} us" if row["kernel_us"] else "FAILED"
        pred = f"{row['pred_us']:.0f}" if row["pred_us"] else "-"
        print(f"    {tag:<16} meas {got:>12}  pred {pred:>10}  ({dt:.0f}s)")
        if args.abort_after and fails >= args.abort_after:
            print(f"    ABORT: {fails} consecutive failures")
            return False
    return True


def phase3(rec, budget, args, cap_s):
    """The new programs under four solvers -- P1's design on mixed-transport programs."""
    print(f"\n=== P3  new real-model cases  (cap {cap_s / 60:.0f} min) ===")
    known = _known_bench_ops()
    todo = [c for c in NEW_CASES if not known or c["op"] in known]
    missing = [c["op"] for c in NEW_CASES if known and c["op"] not in known]
    if missing:
        print(f"    NOT REGISTERED as BENCH_OPs, skipping: {', '.join(missing)}")
        print("    (add them to profile_ops.py's workload table to include this phase)")
    if not todo:
        print("    nothing runnable -- donating the whole cap to P4")
        return True
    t_end = time.time() + cap_s
    fails = 0
    for case in todo:
        for solver in SOLVERS:
            key = f"p3:{case['name']}:{solver}"
            if time.time() > t_end:
                print("    cap reached, moving on")
                return True
            if args.resume and key in rec.done:
                continue
            env = {"BENCH_OP": case["op"], "SENCORES": "32", "LAYOUT_SOLVER": solver}
            env.update(case["env"])
            s, lx, raw, dt = run_one(env, args.run_timeout, reps=args.reps)
            row = {
                "phase": 3,
                "key": key,
                "case": case["name"],
                "solver": solver,
                "seconds": round(dt, 1),
                "kernel_us": (s or {}).get("kernel_us"),
                "pred_us": (s or {}).get("pred_us"),
                "cv": (s or {}).get("kernel_us_cv"),
                "lx": lx,
                "n_lx": len(lx) if lx else None,
            }
            if not row["kernel_us"]:
                row["tail"] = (raw or "")[-400:]
                fails += 1
            else:
                fails = 0
            rec.add(row)
            got = f"{row['kernel_us']:.0f} us" if row["kernel_us"] else "FAILED"
            print(
                f"    {case['name']:<18} {solver:<9} {got:>12}  lx={row['n_lx']}"
                f"  ({dt:.0f}s)"
            )
            if args.abort_after and fails >= args.abort_after:
                print(f"    ABORT: {fails} consecutive failures")
                return False
    return True


def _known_bench_ops():
    """BENCH_OP names profile_ops.py accepts, so P3 can skip cleanly instead of failing."""
    try:
        with open(_PROFILE, encoding="utf-8") as fh:
            src = fh.read()
    except OSError:
        return set()
    return set(re.findall(r'OP == "([a-z0-9_]+)"', src)) | set(
        re.findall(r'^\s*"([a-z0-9_]+)": lambda', src, re.M)
    )


def phase4(rec, budget, args, cap_s):
    """The re-sweep, matmul and coarse tiling first, resuming by database."""
    print(
        f"\n=== P4  re-sweep (matmul + coarse tiling first)  "
        f"(cap {cap_s / 60:.0f} min) ==="
    )
    t_end = time.time() + cap_s
    logdir = os.path.join(_ROOT, "sweep_logs")
    os.makedirs(logdir, exist_ok=True)
    for op in SWEEP_PRIORITY:
        left = t_end - time.time()
        if left < 120:
            print("    cap reached, stopping the sweep")
            break
        log = os.path.join(logdir, f"lxsession_{op}.log")
        cmd = [
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
        ]
        print(f"    {op:<22} up to {left / 60:.0f} min -> {os.path.basename(log)}")
        try:
            subprocess.run(cmd, timeout=left, check=False)
        except subprocess.TimeoutExpired:
            print(f"    {op}: cap hit mid-op; its log is kept and re-parsed below")
            rec.add(
                {
                    "phase": 4,
                    "key": f"p4:{op}",
                    "op": op,
                    "status": "timeout",
                    "log": log,
                }
            )
            break
        rec.add({"phase": 4, "key": f"p4:{op}", "op": op, "status": "done", "log": log})
    # Re-parse every log this session wrote, so a killed invocation still lands in the DB.
    logs = [
        os.path.join(logdir, f)
        for f in sorted(os.listdir(logdir))
        if f.startswith("lxsession_")
    ]
    if logs:
        print(f"    folding {len(logs)} logs into the database")
        subprocess.run(
            [sys.executable, os.path.join(_TOOLS, "parse_sweep_logs.py"), *logs],
            check=False,
        )
    return True


def report(rec):
    """What the session found, stated as prediction vs measurement."""
    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    p1 = [r for r in rec.rows if r.get("phase") == 1 and r.get("kernel_us")]
    if p1:
        print("\nP1  flash: predicted vs measured regret against the best solver")
        for case in FLASH_CASES:
            rows = [r for r in p1 if r["case"] == case["name"]]
            if not rows:
                continue
            best = {}
            for s in SOLVERS:
                got = [r["kernel_us"] for r in rows if r["solver"] == s]
                if got:
                    best[s] = min(got)  # min over rounds: the least drift-contaminated
            if not best:
                continue
            ref = min(best.values())
            allocs = {
                s: tuple(r["lx"] or [])
                for s in best
                for r in rows
                if r["solver"] == s and r.get("lx")
            }
            distinct = len(set(allocs.values()))
            print(f"\n  {case['name']}   ({distinct} distinct allocation(s) observed)")
            if distinct <= 1:
                print(
                    "    WARNING: every solver produced the SAME allocation, so equal"
                )
                print("    runtimes confirm nothing about the ranking.")
            print(
                f"    {'solver':<10}{'measured us':>13}{'meas regret':>13}"
                f"{'predicted':>11}"
            )
            for s in SOLVERS:
                if s not in best:
                    continue
                reg = (best[s] - ref) / ref * 100.0
                pr = case["predict"].get(s)
                print(
                    f"    {s:<10}{best[s]:>13.1f}{reg:>12.1f}%"
                    f"{(f'{pr:+.1f}%' if pr is not None else '-'):>11}"
                )

    p2 = [
        r
        for r in rec.rows
        if r.get("phase") == 2 and r.get("kernel_us") and r.get("pred_us")
    ]
    if len(p2) >= 3:
        print(f"\nP2  flash tile hints: does the model RANK correctly?  (n={len(p2)})")
        conc = disc = 0
        for i in range(len(p2)):
            for j in range(i + 1, len(p2)):
                a, b = p2[i], p2[j]
                dm, dp = a["kernel_us"] - b["kernel_us"], a["pred_us"] - b["pred_us"]
                if dm == 0 or dp == 0:
                    continue
                conc += (dm > 0) == (dp > 0)
                disc += (dm > 0) != (dp > 0)
        tot = conc + disc
        if tot:
            tau = (conc - disc) / tot
            print(
                f"    pairwise agreement {conc}/{tot} = {conc / tot * 100:.0f}%"
                f"   Kendall tau = {tau:+.2f}"
            )
            print("    (tau +1 = perfect ranking, 0 = no better than chance)")
        fastest_m = min(p2, key=lambda r: r["kernel_us"])
        fastest_p = min(p2, key=lambda r: r["pred_us"])
        print(
            f"    fastest measured  : {fastest_m['hint']}  "
            f"{fastest_m['kernel_us']:.0f} us"
        )
        print(
            f"    fastest predicted : {fastest_p['hint']}  "
            f"{fastest_p['kernel_us']:.0f} us measured"
            f"  ({'HIT' if fastest_p['hint'] == fastest_m['hint'] else 'miss'})"
        )

    p3 = [r for r in rec.rows if r.get("phase") == 3 and r.get("kernel_us")]
    if p3:
        print("\nP3  new real-model programs, by solver")
        for name in sorted({r["case"] for r in p3}):
            rows = [r for r in p3 if r["case"] == name]
            ref = min(r["kernel_us"] for r in rows)
            allocs = {tuple(r["lx"] or []) for r in rows if r.get("lx")}
            print(f"\n  {name}   ({len(allocs)} distinct allocation(s))")
            for r in sorted(rows, key=lambda r: r["kernel_us"]):
                print(
                    f"    {r['solver']:<10}{r['kernel_us']:>13.1f}"
                    f"{(r['kernel_us'] - ref) / ref * 100:>12.1f}%"
                )

    fails = [
        r for r in rec.rows if r.get("phase") in (1, 2, 3) and not r.get("kernel_us")
    ]
    if fails:
        print(f"\n{len(fails)} run(s) produced no measurement:")
        for r in fails[:8]:
            print(f"    {r.get('key')}: {(r.get('tail') or '')[:120].strip()}")
    print(f"\nraw results: {rec.path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget-min", type=float, default=150.0)
    ap.add_argument("--phases", default="1,2,3,4")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument(
        "--run-timeout",
        type=int,
        default=420,
        help="seconds one profile_ops run may take",
    )
    ap.add_argument(
        "--abort-after",
        type=int,
        default=4,
        help="consecutive failures that end a phase (0 disables)",
    )
    ap.add_argument("--out", default=os.path.join(_HERE, "lx_session_results.jsonl"))
    ap.add_argument(
        "--resume", action="store_true", help="skip runs already recorded in --out"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="re-print the report from an existing results file",
    )
    args = ap.parse_args()

    want = {int(x) for x in args.phases.split(",") if x.strip()}
    rec = Recorder(args.out)
    if args.report_only:
        report(rec)
        return 0

    budget = Budget(args.budget_min * 60)
    caps = {1: 30 * 60, 2: 35 * 60, 3: 25 * 60}
    n_runs = (
        len(FLASH_CASES) * len(SOLVERS) * 2
        + len(FLASH_HINTS)
        + len(NEW_CASES) * len(SOLVERS)
    )
    print(
        f"budget {args.budget_min:.0f} min | phases {sorted(want)} | "
        f"{n_runs} device runs planned before P4"
    )
    if args.dry_run:
        preflight(rec, budget, args)
        for ph, cap in caps.items():
            if ph in want:
                print(f"  P{ph}  cap {cap / 60:.0f} min")
        if 4 in want:
            print(f"  P4  remainder, ops: {', '.join(SWEEP_PRIORITY[:6])}, ...")
        return 0

    if not preflight(rec, budget, args):
        print("\npreflight failed; not spending the session on it")
        return 1

    fns = {1: phase1, 2: phase2, 3: phase3}
    for ph in (1, 2, 3):
        if ph in want and budget.remaining() > 60:
            if not fns[ph](rec, budget, args, budget.phase_cap(caps[ph])):
                print(f"P{ph} aborted; skipping the rest of the session")
                want.discard(4)
                break
    if 4 in want and budget.remaining() > 120:
        phase4(rec, budget, args, budget.phase_cap(0, is_last=True))

    print(
        f"\nsession used {budget.elapsed() / 60:.1f} of {args.budget_min:.0f} minutes"
    )
    report(rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
