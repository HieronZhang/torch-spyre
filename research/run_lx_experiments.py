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

  P1  UNTILED flash attention x ALLOCATION                          30 min
      The headline prediction: on two configurations the DEFAULT solver's allocation is
      20.4 % and 28.3 % slower than the proven optimum. Each arm is pinned directly with
      `LX_FORCE_ONLY`, so nothing but the allocation changes and no arm depends on a solver
      choosing what we hoped it would.
  P2  untiled flash across SHAPE and CORES                          35 min
      Whether the model ranks configurations it was never fitted on. Coarse tiling no
      longer compiles, so shape and core count replace the tile-hint ladder -- they vary
      the same underlying quantities and need no hint. Nine points already give tau +0.78
      at 0.3-1.0x absolute error.
  P3  the case studies, UNTILED, at their most contested shapes     45 min
      Feature EXTRACTION, not yet a comparison: these programs have no measured features,
      and their allocations cannot be solved until they do.
  P4  the rest of the re-sweep, matmul first, tiled ops skipped     remainder

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

COARSE TILING IS BROKEN on this toolchain -- every program tested compiles untiled and
fails tiled. That is why P1-P3 are untiled throughout and P4 consults `tiling_probe.json`
to skip ops that cannot build. Contention never needed tiling: a bundle is contested when
one buffer fits the budget and several do not, which shape and core count reach directly.

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
#: P1 falls back to comparing solvers if no forced-allocation file exists. Kept only as a
#: degraded path -- forced allocations are strictly better, see `_p1_arms`.
SOLVERS = ["greedy", "firstfit", "bestfit", "cpsat"]

#: P2: UNTILED flash across shape and core count. Coarse tiling no longer compiles, so the
#: hint ladder the original P2 swept is unavailable -- but varying the hint was never the
#: point. The question is whether the model RANKS configurations it was not fitted on, and
#: shape and cores vary the same underlying quantities (bytes moved, per-core work) without
#: needing a tile hint. Nine such points already give Kendall tau +0.78 at 0.3-1.0x absolute
#: error; this widens the grid so the rank statistic rests on more than nine.
UNTILED_GRID = [
    (h, ell, cores)
    for h, ell in [
        (4, 512),
        (8, 512),
        (16, 512),
        (32, 512),
        (4, 1024),
        (8, 1024),
        (16, 1024),
        (4, 2048),
        (8, 2048),
    ]
    for cores in (32, 8)
]

#: P3: the case studies, UNTILED, at the shapes `screen_untiled.py` finds most contested.
#: Read from that screen at run time rather than copied here, so the two never disagree.
CASE_PROGRAMS = ["decode_block", "block_norm_mlp", "attn_scores", "prefix_block"]

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


def _default_records():
    """Prefer the untiled-flash records: they are the only contested set that compiles.

    The checked-in database was measured with coarse tiling, which no longer builds, so its
    flash features describe programs this toolchain cannot produce. `probe_untiled_flash.py`
    writes replacements extracted by today's compiler.
    """
    untiled = os.path.join(_HERE, "untiled_flash_records.json")
    if os.path.exists(untiled):
        return untiled
    import eval_model as E

    return E.records_path()


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

        path = args.records or _default_records()
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
        # No forced allocations means no contested set was solved -- which now means the
        # records file has no compilable contested configuration, not that solvers should
        # be compared instead. Say so rather than measuring something uninformative.
        print("    NO forced allocations. P1 needs a records file with a contested")
        print("    configuration; run research/probe_untiled_flash.py first.")
        return True

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
            # Compare only the buffers the override CONTROLS. `lx` lists every arg the
            # compiler placed in LX, including graph inputs and outputs, which are not the
            # allocator's to choose -- so a raw set comparison always fails. It did, on all
            # 18 runs of the first session, none of which had actually gone wrong.
            got = {_canon_buf(x) for x in lx}
            want = {_canon_buf(x) for x in arm["lx"]}
            row["override_honoured"] = want <= got and not (want - got)
            row["lx_missing"] = sorted(want - got)
            row["lx_extra_nonmovable"] = len(got - want)
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
    """Untiled flash across shape and cores: does the model RANK what it never saw?

    Reports Kendall tau over every measured pair, and whether the configuration the model
    calls fastest actually is. Both are ranking statistics on purpose -- ranking is what a
    scheduler needs, and it survives a constant scale error.
    """
    print(f"\n=== P2  untiled flash x (shape, cores)  (cap {cap_s / 60:.0f} min) ===")
    t_end = time.time() + cap_s
    fails = 0
    for h, ell, cores in UNTILED_GRID:
        tag = f"H{h}_L{ell}_c{cores}"
        key = f"p2:{tag}"
        if time.time() > t_end:
            print("    cap reached, moving on")
            break
        if args.resume and key in rec.done:
            continue
        env = {
            "BENCH_OP": "flash_attn",
            "SENCORES": str(cores),
            "FA_B": "1",
            "FA_H": str(h),
            "FA_LQ": str(ell),
            "FA_LK": str(ell),
            "FA_D": "128",
            "FA_H_TILES": "1",
            "FA_LQ_TILES": "1",
            "FA_LK_TILES": "1",
        }
        s_, lx, raw, dt = run_one(env, args.run_timeout, reps=args.reps)
        row = {
            "phase": 2,
            "key": key,
            "hint": tag,
            "H": h,
            "L": ell,
            "cores": cores,
            "seconds": round(dt, 1),
            "kernel_us": (s_ or {}).get("kernel_us"),
            "pred_us": (s_ or {}).get("pred_us"),
            "cv": (s_ or {}).get("kernel_us_cv"),
            "err_pct": (s_ or {}).get("err_pct"),
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
        shown = f"{row['kernel_us']:.0f} us" if row["kernel_us"] else "FAILED"
        pred = f"{row['pred_us']:.0f}" if row["pred_us"] else "-"
        print(f"    {tag:<16} meas {shown:>12}  pred {pred:>10}  ({dt:.0f}s)")
        if args.abort_after and fails >= args.abort_after:
            print(f"    ABORT: {fails} consecutive DEVICE failures")
            return False
    return True


def _case_configs(top=2):
    """The most contested untiled configuration(s) per case study, from the screen.

    Taken from `screen_untiled.py` at run time so the session and the screen cannot drift
    apart. Returns (program, WL_* env, cores, feasible-allocation count).
    """
    try:
        import itertools as _it

        import screen_untiled as SU
        from screen_configs import PROGRAMS
    except Exception:  # noqa: BLE001 -- the screen is optional; P3 just gets nothing
        return []
    out = []
    for name in CASE_PROGRAMS:
        prog = PROGRAMS.get(name)
        if prog is None:
            continue
        rows = []
        for scale, cores in _it.product(SU.SCALES, SU.CORES):
            r = SU.screen(prog, scale, cores)
            if r and r["binds"] and r["keeps"] > 0:
                rows.append(r)
        rows.sort(key=lambda r: (-r["n_feasible"], -r["spread"]))
        seen = set()
        for r in rows:
            env = SU.wl_env(name, r["dims"])
            # Different scalings can land on the same shape when a program lacks the dim
            # being scaled -- attn_scores has no d_ff, so every /F variant is one config.
            sig = (tuple(sorted(env.items())), r["cores"])
            if sig in seen:
                continue
            seen.add(sig)
            # Pin every tile count to 1. The factories default to bt=2/st=4/ft=4, so a
            # shape-only config still requests coarse tiling -- which does not compile,
            # and cost P3 all seven of its runs.
            env = dict(env)
            env.update({"WL_BT": 1, "WL_ST": 1, "WL_FT": 1, "WL_HT": 1, "WL_QT": 1})
            out.append((name, env, r["cores"], r["n_feasible"]))
            if len(seen) >= top:
                break
    return out


def phase3(rec, budget, args, cap_s):
    """The case studies, untiled, at their most contested shapes.

    Same design as P1 -- vary only the allocation -- but these programs have no measured
    features yet, so this pass EXTRACTS them (`BENCH_EMIT_RECORDS=1`). What it produces is
    the input to a forced-allocation comparison, not the comparison itself: the allocations
    cannot be solved until the features exist.
    """
    print(f"\n=== P3  untiled case studies  (cap {cap_s / 60:.0f} min) ===")
    cfgs = _case_configs()
    if not cfgs:
        print("    screen unavailable; nothing to run")
        return True
    print(f"    {len(cfgs)} configuration(s) across {len(CASE_PROGRAMS)} program(s)")
    t_end = time.time() + cap_s
    fails = 0
    for name, wl, cores, nfeas in cfgs:
        key = f"p3:{name}:{'_'.join(f'{k}{v}' for k, v in sorted(wl.items()))}:c{cores}"
        if time.time() > t_end:
            print("    cap reached, moving on")
            break
        if args.resume and key in rec.done:
            continue
        env = {"BENCH_OP": f"research:{name}", "SENCORES": str(cores)}
        env.update({k: str(v) for k, v in wl.items()})
        s_, lx, raw, dt = run_one(env, args.run_timeout, reps=args.reps)
        row = {
            "phase": 3,
            "key": key,
            "case": name,
            "wl": wl,
            "cores": cores,
            "screened_feasible": nfeas,
            "seconds": round(dt, 1),
            "kernel_us": (s_ or {}).get("kernel_us"),
            "pred_us": (s_ or {}).get("pred_us"),
            "cv": (s_ or {}).get("kernel_us_cv"),
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
        shown = f"{row['kernel_us']:.0f} us" if row["kernel_us"] else "FAILED"
        wls = " ".join(f"{k}={v}" for k, v in sorted(wl.items()))
        print(f"    {name:<16}{cores:>3}c {wls[:34]:<36}{shown:>12}  ({dt:.0f}s)")
        if args.abort_after and fails >= args.abort_after:
            print(f"    ABORT: {fails} consecutive DEVICE failures")
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


def _tiling_broken():
    """Ops `probe_tiling.py` found compile untiled but fail tiled. Empty if it never ran."""
    path = os.path.join(_HERE, "tiling_probe.json")
    if not os.path.exists(path):
        return set()
    try:
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
    except Exception:  # noqa: BLE001
        return set()
    return {
        r["op"] for r in rows if str(r.get("verdict", "")).startswith("TILING BROKEN")
    }


def phase4(rec, budget, args, cap_s):
    """The re-sweep, matmul and coarse tiling first, resuming by database."""
    print(
        f"\n=== P4  re-sweep (matmul + coarse tiling first)  "
        f"(cap {cap_s / 60:.0f} min) ==="
    )
    # Most of the plan tiles, and tiling currently fails on every program tested. Running
    # it anyway would spend the tail of the session collecting compile errors, so consult
    # the probe if it has run and drop the ops it found broken.
    broken = _tiling_broken()
    if broken:
        print(
            f"    tiling probe says these do NOT tile, skipping: "
            f"{', '.join(sorted(broken)[:8])}"
        )
    t_end = time.time() + cap_s
    logdir = os.path.join(_ROOT, "sweep_logs")
    os.makedirs(logdir, exist_ok=True)
    for op in [o for o in SWEEP_PRIORITY if o not in broken]:
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
        print(
            f"\nP2  untiled flash across shape/cores: does the model RANK correctly?"
            f"  (n={len(p2)})"
        )
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
        # Absolute scale, now that features come from this compiler rather than from the
        # pre-fix tile_rows_per_core era that made flash read 15-45x high.
        rat = [r["pred_us"] / r["kernel_us"] for r in p2 if r["kernel_us"]]
        if rat:
            print(f"    absolute pred/meas: {min(rat):.2f}x to {max(rat):.2f}x")
        by_cores: dict = {}
        for r in p2:
            by_cores.setdefault(r.get("cores"), []).append(
                r["pred_us"] / r["kernel_us"]
            )
        for c, v in sorted(by_cores.items(), key=lambda kv: -(kv[0] or 0)):
            print(f"      {c} cores: {sum(v) / len(v):.2f}x mean over {len(v)}")
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
    # 150 minutes: P1 is the headline and cheap (untiled flash runs in under a
    # millisecond; compile dominates), P2 needs breadth for its rank statistic, P3 has the
    # most configurations and the least certainty, P4 takes whatever is left.
    caps = {1: 30 * 60, 2: 35 * 60, 3: 45 * 60}
    n_runs = len(UNTILED_GRID) + len(_case_configs()) + 16
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
