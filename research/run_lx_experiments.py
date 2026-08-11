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

  P1  flash attention x ALLOCATION                                  30 min
      The headline prediction: on two configurations the DEFAULT solver's allocation is
      20.4 % and 28.3 % slower than the proven optimum. Each arm is pinned directly with
      `LX_FORCE_ONLY`, so nothing but the allocation changes and no arm depends on a solver
      choosing what we hoped it would.
  P2  flash attention across tile-count hints                       35 min
      Whether the model ranks configurations it has never been fitted on, the same test the
      2.11 toolchain got. Ranking only -- absolute error on flash is 15-45x.
  P3  the new real-model programs                                   25 min
      Same design as P1 on programs with mixed transports.
  P4  the rest of the re-sweep, matmul and coarse tiling first      remainder

THREE CONTROLS, without which the timings cannot be interpreted:

1. **The allocations must actually differ.** Comparing solvers only tests a ranking when
   the solvers disagree, and on two of these three configurations several of them pick the
   SAME set -- an equal runtime would then confirm nothing. So P1 pins each allocation with
   `LX_FORCE_ONLY` (`_lx_force_override`, `scratchpad/allocator.py`), collapses arms that
   name one set, and VERIFIES from `MODEL FEATS` that the compiler honoured the override.
   An ignored override is the one failure that would quietly invalidate everything.
2. **Order must not alias onto the allocation.** Thermal drift over half an hour is
   comparable to the effect, so arms run interleaved in two rounds of opposite order and
   the report takes the per-arm minimum.
3. **ortools must be importable** for the `cpsat` SOLVER arm; it raises rather than
   silently degrading (`ilp_solver_ortools.py:336`). With forced allocations the important
   arm no longer needs it -- the proven optimum is pinned by name.

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


_COMPILE_ERR = (
    "InductorError",
    "finalize_layouts",
    "NotImplementedError",
    "Traceback (most recent call last)",
)


def classify_failure(raw):
    """Why did this run produce no measurement -- the CONFIG, or the DEVICE?

    The distinction decides whether to skip one configuration or end the session. A compile
    error means this program cannot be built on this toolchain: every later arm of the SAME
    configuration will fail identically, but other configurations are unaffected, so it must
    NOT trip the device-death counter. Only a timeout or an unexplained silent failure
    suggests the card itself, which is what `--abort-after` exists for.

    Learned the hard way: flash at Lq=Lk=4096 stopped compiling ("restickify needed but
    infeasible"), and under the original rule its four arms would have aborted the run and
    taken P2, P3 and the re-sweep with them.
    """
    if raw == "TIMEOUT":
        return "timeout"
    if any(marker in (raw or "") for marker in _COMPILE_ERR):
        return "compile"
    if "span_overflow" in (raw or ""):
        return "span_overflow"
    return "device"


_LOGDIR = os.path.join(_HERE, "lx_session_logs")


def _save_log(key, raw):
    """Keep the FULL output of a failed run, not just the tail in the JSONL.

    400 characters is enough to notice a failure and rarely enough to diagnose one -- a
    compile traceback and the span-guard report are both longer than that. Written per run
    so a failing arm can be handed over or re-read without re-running the device.
    """
    try:
        os.makedirs(_LOGDIR, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key or "run")
        path = os.path.join(_LOGDIR, f"{safe}.log")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(raw or "(no output)")
        return path
    except OSError:
        return None


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


def solve_allocations(args):
    """Derive P1's allocations from the measurement database, here and now.

    Done inside the session on purpose. The sets are only valid for the FEATURES they were
    solved from, so a stale `forced_allocations.json` left over from before a re-sweep would
    pin the wrong buffers and the whole comparison would be against the wrong reference.
    Solving takes a couple of CPU minutes against ~10^6 feasible allocations and needs no
    device, so there is no reason to trust a file instead.
    """
    out_path = os.path.join(_HERE, "forced_allocations.json")
    if args.skip_solve:
        print("    allocations                   reusing forced_allocations.json")
        return os.path.exists(out_path)
    try:
        sys.path.insert(0, os.path.join(_ROOT, "tools", "cost_model"))
        import emit_forced_allocations as EFA
        import eval_model as E

        path = args.records or E.records_path()
        if not os.path.exists(path):
            print(f"    allocations                   NO DATABASE at {path}")
            return False
        t0 = time.time()
        print(
            f"    solving allocations from      {os.path.basename(path)} ...",
            end=" ",
            flush=True,
        )
        cases = EFA.emit(path, out_path, quiet=True)
        print(f"{len(cases)} contested case(s), {time.time() - t0:.0f}s")
        if not cases:
            print("      no contested flash configuration in this database -- P1 will")
            print("      fall back to comparing LAYOUT_SOLVER settings instead.")
            return True
        for c in cases:
            n = c["distinct_allocations"]
            print(f"      {c['label'][:52]:<52} {n} distinct arm(s)")
            for grp in c.get("duplicate_arms", []):
                print(f"        collapsed: {' == '.join(grp)}")
        return True
    except Exception as exc:  # noqa: BLE001 -- fall back rather than lose the session
        print(f"FAILED ({type(exc).__name__}: {exc})")
        print("      P1 will fall back to comparing LAYOUT_SOLVER settings.")
        return True


def preflight(rec, budget, args):
    """Fail fast on the things that would silently invalidate the whole session."""
    print("=== preflight ===")
    ok = True
    solve_allocations(args)
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


_BUF_ID = re.compile(r"^(?:op|buf|b)(\d+)$|_buf(\d+)$")


def _canon_buf(name):
    """Match the allocator's canonicalisation, so op3 / buf3 / b3 compare equal."""
    m = _BUF_ID.search(name or "")
    return f"b{m[1] or m[2]}" if m else (name or "")


def _short(label):
    """A compact tag for a flash configuration."""
    t = label.replace("flash_attn ", "").replace("flash ", "")
    return re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_")[:38] or "case"


def _p1_arms():
    """The allocations P1 measures, and how each is produced.

    Preferred: ``forced_allocations.json`` (from ``emit_forced_allocations.py``) names exact
    buffer sets, which ``LX_FORCE_ONLY`` pins directly. That is strictly better than picking
    a solver and hoping -- distinct sets are guaranteed to be distinct allocations, arms
    naming the SAME set are collapsed instead of measured twice, and the arm that matters
    (the proven optimum) needs neither ortools installed nor CP-SAT agreeing.

    Fallback when the file is absent: drive ``LAYOUT_SOLVER``, and let the report say so if
    the solvers turn out to coincide.
    """
    path = os.path.join(_HERE, "forced_allocations.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:  # noqa: BLE001 -- a bad file must not end the session
        return None
    cases = []
    for c in data.get("cases", []):
        by_set: dict = {}
        for name, a in c.get("arms", {}).items():
            by_set.setdefault(tuple(a["lx"]), []).append((name, a))
        arms = [
            {
                "label": "+".join(n for n, _ in members),
                "lx": list(lx),
                "pred_us": members[0][1]["pred_us"],
                "predicted_regret_pct": members[0][1]["predicted_regret_pct"],
            }
            for lx, members in by_set.items()
        ]
        arms.sort(key=lambda a: a["predicted_regret_pct"])
        cases.append(
            {
                "name": _short(c.get("label", "")),
                "env": c.get("env", {}),
                "arms": arms,
                "measured_us": c.get("measured_us"),
            }
        )
    return cases or None


def phase1(rec, budget, args, cap_s):
    """Flash attention across ALLOCATIONS, interleaved and order-balanced.

    The whole point of P1 is that the allocation is the only thing that varies. Forcing the
    set makes that literally true; selecting a solver only approximates it, and on two of
    these three configurations several solvers pick the same set.
    """
    forced = _p1_arms()
    mode = (
        "forced allocations" if forced else "LAYOUT_SOLVER (no forced_allocations.json)"
    )
    print(f"\n=== P1  flash x allocation  (cap {cap_s / 60:.0f} min, via {mode}) ===")
    t_end = time.time() + cap_s

    plan = []
    if forced:
        for rnd in (0, 1):
            for case in forced:
                arms = case["arms"] if rnd == 0 else list(reversed(case["arms"]))
                for arm in arms:
                    plan.append((rnd, case, arm))
        n_alloc = sum(len(c["arms"]) for c in forced)
        print(
            f"    {len(forced)} case(s), {n_alloc} distinct allocation(s), "
            f"2 rounds = {len(plan)} runs"
        )
    else:
        for rnd in (0, 1):
            order = SOLVERS if rnd == 0 else list(reversed(SOLVERS))
            for case in FLASH_CASES:
                for solver in order:
                    plan.append(
                        (
                            rnd,
                            case,
                            {
                                "label": solver,
                                "solver": solver,
                                "predicted_regret_pct": case["predict"].get(solver),
                            },
                        )
                    )

    fails = 0
    for rnd, case, arm in plan:
        key = f"p1:{case['name']}:{arm['label']}:r{rnd}"
        if time.time() > t_end:
            print("    cap reached, moving on")
            break
        if args.resume and key in rec.done:
            continue
        env = {"BENCH_OP": "flash_attn", "SENCORES": "32"}
        env.update(case["env"])
        if arm.get("lx") is not None:
            env["LX_FORCE_ONLY"] = ",".join(arm["lx"])
        else:
            env["LAYOUT_SOLVER"] = arm["solver"]
        s, lx, raw, dt = run_one(env, args.run_timeout, reps=args.reps)
        row = {
            "phase": 1,
            "key": key,
            "case": case["name"],
            "arm": arm["label"],
            "round": rnd,
            "seconds": round(dt, 1),
            "kernel_us": (s or {}).get("kernel_us"),
            "cv": (s or {}).get("kernel_us_cv"),
            "pred_us": arm.get("pred_us") or (s or {}).get("pred_us"),
            "requested_lx": arm.get("lx"),
            "lx": lx,
            "n_lx": len(lx) if lx else None,
            "predicted_regret_pct": arm.get("predicted_regret_pct"),
        }
        # Did the override actually take? A silently-ignored LX_FORCE_ONLY would make every
        # arm identical and the comparison worthless, so verify instead of assuming.
        if arm.get("lx") is not None and lx is not None:
            got = {_canon_buf(x) for x in lx}
            want = {_canon_buf(x) for x in arm["lx"]}
            row["override_honoured"] = got == want
            row["lx_unexpected"] = sorted(got - want)
            row["lx_missing"] = sorted(want - got)
        if not row["kernel_us"]:
            row["tail"] = (raw or "")[-400:]
            row["log"] = _save_log(key, raw)
            row["failure"] = classify_failure(raw)
            # A config that cannot be compiled is not a dead card; skip it, do not abort.
            fails = fails + 1 if row["failure"] in ("device", "timeout") else fails
        else:
            fails = 0
        rec.add(row)
        shown = f"{row['kernel_us']:.0f} us" if row["kernel_us"] else "FAILED"
        flag = ""
        if row.get("override_honoured") is False:
            flag = f"  !! kept {len(lx or [])}, asked {len(arm['lx'])}"
        print(
            f"    {case['name'][:24]:<24} {arm['label'][:18]:<18} r{rnd}  "
            f"{shown:>12}  lx={row['n_lx']}{flag}  ({dt:.0f}s)"
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
            row["log"] = _save_log(key, raw)
            row["failure"] = classify_failure(raw)
            # A config that cannot be compiled is not a dead card; skip it, do not abort.
            fails = fails + 1 if row["failure"] in ("device", "timeout") else fails
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
                row["log"] = _save_log(key, raw)
                row["failure"] = classify_failure(raw)
                fails = fails + 1 if row["failure"] in ("device", "timeout") else fails
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
        print("\nP1  flash: predicted vs measured regret, per allocation")
        for cname in sorted({r["case"] for r in p1}):
            rows = [r for r in p1 if r["case"] == cname]
            arms = sorted({r["arm"] for r in rows})
            best = {}
            for a in arms:
                got = [r["kernel_us"] for r in rows if r["arm"] == a]
                if got:
                    best[a] = min(got)  # min over rounds: least drift-contaminated
            if not best:
                continue
            ref = min(best.values())
            observed = {
                tuple(sorted(_canon_buf(x) for x in (r["lx"] or [])))
                for r in rows
                if r.get("lx")
            }
            ignored = [r for r in rows if r.get("override_honoured") is False]
            print(f"\n  {cname}   ({len(observed)} distinct allocation(s) observed)")
            if len(observed) <= 1 and len(best) > 1:
                print(
                    "    WARNING: every arm produced the SAME allocation on device, so"
                )
                print(
                    "    equal runtimes confirm nothing. Check LX_FORCE_ONLY took"
                    " effect."
                )
            if ignored:
                print(
                    f"    WARNING: {len(ignored)} run(s) did not honour LX_FORCE_ONLY"
                )
                for r in ignored[:3]:
                    print(
                        f"      {r['arm']}: extra={r.get('lx_unexpected')} "
                        f"missing={r.get('lx_missing')}"
                    )
            print(
                f"    {'allocation':<20}{'measured us':>13}{'measured':>11}"
                f"{'predicted':>11}"
            )
            for a in sorted(best, key=lambda a: best[a]):
                reg = (best[a] - ref) / ref * 100.0
                pr = next(
                    (r.get("predicted_regret_pct") for r in rows if r["arm"] == a), None
                )
                print(
                    f"    {a[:20]:<20}{best[a]:>13.1f}{reg:>10.1f}%"
                    f"{(f'{pr:+.1f}%' if pr is not None else '-'):>11}"
                )
            # The claim under test is the ORDER, so state whether it survived.
            meas_order = [a for a in sorted(best, key=lambda a: best[a])]
            pred_order = sorted(
                best,
                key=lambda a: next(
                    (r.get("predicted_regret_pct") or 0.0)
                    for r in rows
                    if r["arm"] == a
                ),
            )
            print(f"    order  measured {' < '.join(x[:12] for x in meas_order)}")
            print(
                f"           predicted {' < '.join(x[:12] for x in pred_order)}"
                f"   -> {'AGREES' if meas_order == pred_order else 'DISAGREES'}"
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
    ap.add_argument(
        "--records",
        default="",
        help="measurement database P1's allocations are solved from "
        "(default: the checked-in sweep_records.json)",
    )
    ap.add_argument(
        "--skip-solve",
        action="store_true",
        help="reuse forced_allocations.json instead of re-solving; only safe "
        "when the database has not changed since it was written",
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
                # Only a device-level abort reaches here now; a config that fails to
                # compile is skipped inside the phase. So this really is fatal.
                print(
                    f"P{ph} aborted on repeated DEVICE failures; stopping the session"
                )
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
