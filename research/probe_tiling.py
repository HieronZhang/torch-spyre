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

"""Which operations still COARSE TILE on this toolchain?

THE QUESTION THIS SETTLES. Every tiled flash configuration now fails in
`insert_restickify.finalize_layouts`, while untiled flash compiles fine. That is either
(a) specific to flash's transpose, in which case the LX-allocation experiment simply moves
to another coarse-tiled op, or (b) general to coarse tiling, in which case there is no
contested allocation to measure anywhere and the regression itself is the result. Those two
worlds need completely different sessions, and nothing measured so far distinguishes them.

METHOD. For each op the sweep plan exercises with `BENCH_TILES > 1`, run the SAME shape
twice -- once untiled, once tiled. The pair is the control: an op that fails both ways is
broken for an unrelated reason and says nothing about tiling, while OK-then-FAIL isolates
tiling as the cause. Shapes come from the plan, so they are known-valid rather than invented.

Reading the result:

* tiling works broadly, flash alone fails  -> flash's transpose; move the experiment to
  `softmax_row_tiling` (33 contested bundles, default solver 18.2 % off)
* every tiled op fails                     -> a general coarse-tiling regression; that is
  the finding, and the session becomes a bisect rather than a measurement
* a mixed picture                          -> the OK column IS the candidate list, and
  whichever contested op survives becomes P1's target

    python3 research/probe_tiling.py                 # ~10-15 min
    python3 research/probe_tiling.py --ops softmax_row_tiling,matmul_row_tiling
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
_PROFILE = os.path.join(
    _ROOT, "docs", "source", "user_guide", "examples", "profile_ops.py"
)
_PLAN = os.path.join(_ROOT, "tools", "cost_model", "sweep_plan.json")

_SUMMARY = re.compile(r"^SUMMARY .*?kernel_us=([0-9.]+)", re.M)

#: Quarantined: its per-core span exceeds the MVLOC limit and it is the prime suspect in the
#: 2026-08-08 card failure. Never probe it.
SKIP = {"bmm_3d2d_k_tiling"}

#: op -> does the program materialise a transpose. Filled for the case studies so the
#: summary can report whether failures track the transpose, which is the standing hypothesis
#: for the flash breakage.
_TRANSPOSE: dict = {}

#: The case studies, probed through the `research:` dispatch. Each is run untiled and then
#: at its screened tile counts. `transpose` records whether the program materialises one --
#: flash dies in `insert_restickify`, which a transpose drives, so if that hypothesis holds
#: the two programs carrying one should fail while the three without should compile.
CASE_STUDIES = [
    ("mlp_up", False, {"WL_BT": "1", "WL_ST": "2", "WL_FT": "4"}),
    ("block_norm_mlp", False, {"WL_BT": "1", "WL_ST": "2", "WL_FT": "8"}),
    ("prefix_block", False, {"WL_BT": "1", "WL_ST": "1", "WL_FT": "2"}),
    ("attn_scores", True, {"WL_BT": "1", "WL_HT": "2", "WL_QT": "2"}),
    ("decode_block", True, {"WL_BT": "1", "WL_HT": "2", "WL_FT": "2"}),
]


#: Ops whose tiling matters most to the experiment, tried first so a truncated run still
#: answers the question. `softmax_row_tiling` is the fallback P1 target; the matmul family is
#: what the re-sweep prioritises.
PRIORITY = [
    "softmax_row_tiling",
    "softmax_unrolled",
    "matmul_row_tiling",
    "matmul_k_tiling",
    "bmm_k_tiling",
    "bmm_layout",
    "mmwd",
    "amax",
    "ctsum",
    "transpose_outer",
]


def classify(raw):
    if raw == "TIMEOUT":
        return "timeout"
    for pat, tag in (
        ("finalize_layouts", "restickify"),
        ("carry propagation", "reduction-tiling"),
        ("span_overflow", "span-overflow"),
        ("does not support", "unsupported"),
        ("InductorError", "inductor"),
        ("Traceback", "exception"),
    ):
        if pat in (raw or ""):
            return tag
    return "no-summary"


def run(env_extra, timeout_s):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_extra.items()})
    env.update({"BENCH_REPS": "1", "BENCH_WARMUP": "1"})
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
    m = _SUMMARY.search(raw)
    return (float(m[1]) if m else None), raw, time.time() - t0


def pairs_from_plan(only):
    """(op, untiled env, tiled env) for every op the plan tiles, shapes taken from it."""
    with open(_PLAN, encoding="utf-8") as fh:
        plan = json.load(fh)
    cfgs = plan["configs"] if isinstance(plan, dict) else plan
    best: dict = {}
    for c in cfgs:
        op = c.get("BENCH_OP")
        if not op or op in SKIP or (only and op not in only):
            continue
        try:
            t = int(c.get("BENCH_TILES", "0") or 0)
        except ValueError:
            continue
        if t > 1 and (op not in best or t < best[op][0]):
            best[op] = (t, c)  # smallest tile count that still exercises tiling
    out = []
    for op, (t, c) in best.items():
        tiled = {k: v for k, v in c.items() if k != "BENCH_OP"}
        untiled = dict(tiled, BENCH_TILES="1")
        out.append((op, untiled, tiled, t))
    order = {op: i for i, op in enumerate(PRIORITY)}
    out.sort(key=lambda r: (order.get(r[0], 99), r[0]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ops", default="", help="comma list; default = every tiled op")
    ap.add_argument(
        "--cases",
        action="store_true",
        help="probe the research/ case studies instead of the plan ops",
    )
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--budget-min", type=float, default=20.0)
    ap.add_argument("--out", default=os.path.join(_HERE, "tiling_probe.json"))
    args = ap.parse_args()

    only = {o.strip() for o in args.ops.split(",") if o.strip()}
    if args.cases:
        cases = []
        for name, has_tp, knobs in CASE_STUDIES:
            if only and name not in only:
                continue
            untiled = {k: "1" for k in knobs}
            cases.append(
                (
                    f"research:{name}",
                    untiled,
                    dict(knobs),
                    max(int(v) for v in knobs.values()),
                )
            )
        _TRANSPOSE.update({f"research:{n}": t for n, t, _ in CASE_STUDIES})
    else:
        cases = pairs_from_plan(only)
    if not cases:
        print("no tiled configurations in the plan for those ops")
        return 1

    print(f"probing {len(cases)} op(s): untiled vs tiled, same shape, 1 rep each")
    print(f"budget {args.budget_min:.0f} min\n")
    print(f"  {'op':<24}{'tiles':>6}{'untiled':>12}{'tiled':>12}   verdict")
    t_end = time.time() + args.budget_min * 60
    rows = []
    for op, untiled, tiled, t in cases:
        if time.time() > t_end:
            print("  budget reached; stopping")
            break
        u_us, u_raw, _ = run(dict(untiled, BENCH_OP=op), args.timeout)
        t_us, t_raw, _ = run(dict(tiled, BENCH_OP=op), args.timeout)
        u_tag = "OK" if u_us else classify(u_raw)
        t_tag = "OK" if t_us else classify(t_raw)
        if u_us and t_us:
            verdict = "tiling works"
        elif u_us and not t_us:
            verdict = f"TILING BROKEN ({t_tag})"
        elif not u_us and not t_us:
            verdict = f"op broken either way ({u_tag})"
        else:
            verdict = "untiled broken, tiled ok (odd)"
        rows.append(
            {
                "op": op,
                "tiles": t,
                "untiled_us": u_us,
                "tiled_us": t_us,
                "untiled": u_tag,
                "tiled_tag": t_tag,
                "verdict": verdict,
                "tiled_env": tiled,
                "tail": None if t_us else (t_raw or "")[-500:],
            }
        )
        print(
            f"  {op:<24}{t:>6}{(f'{u_us:.0f}' if u_us else u_tag):>12}"
            f"{(f'{t_us:.0f}' if t_us else t_tag):>12}   {verdict}"
        )
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)

    if _TRANSPOSE:
        tp_fail = [
            r
            for r in rows
            if _TRANSPOSE.get(r["op"]) and r["verdict"].startswith("TILING BROKEN")
        ]
        tp_ok = [
            r
            for r in rows
            if _TRANSPOSE.get(r["op"]) and r["verdict"] == "tiling works"
        ]
        no_tp_fail = [
            r
            for r in rows
            if not _TRANSPOSE.get(r["op"]) and r["verdict"].startswith("TILING BROKEN")
        ]
        no_tp_ok = [
            r
            for r in rows
            if not _TRANSPOSE.get(r["op"]) and r["verdict"] == "tiling works"
        ]
        print(f"\n  with a transpose:    {len(tp_ok)} tile, {len(tp_fail)} broken")
        print(f"  without a transpose: {len(no_tp_ok)} tile, {len(no_tp_fail)} broken")
        if tp_fail and no_tp_ok and not no_tp_fail:
            print(
                "  -> failures track the TRANSPOSE exactly, as the flash error suggested"
            )
        elif no_tp_fail:
            print(
                "  -> transpose-free programs fail too, so the transpose is NOT the whole"
                " story"
            )

    works = [r for r in rows if r["verdict"] == "tiling works"]
    broken = [r for r in rows if r["verdict"].startswith("TILING BROKEN")]
    dead = [r for r in rows if r["verdict"].startswith("op broken")]
    print(
        f"\n  tiling works on {len(works)}, broken on {len(broken)}, "
        f"op dead either way on {len(dead)}"
    )
    if works:
        print("\nCoarse tiling is NOT generally broken. Usable for the LX experiment:")
        for r in works:
            print(f"    {r['op']} (tiles={r['tiles']})")
        if any(r["op"] == "softmax_row_tiling" for r in works):
            print("\n  softmax_row_tiling tiles -> P1 can use it directly:")
            print("    python3 research/emit_forced_allocations.py \\")
            print("        --op softmax_row_tiling --top 4")
    elif broken:
        print(
            "\nEvery tiled op fails while its untiled twin compiles. That is a GENERAL"
        )
        print("coarse-tiling regression, not a flash problem -- the finding, not a")
        print("blocker. Failure kinds:")
        kinds: dict = {}
        for r in broken:
            kinds.setdefault(r["tiled_tag"], []).append(r["op"])
        for k, v in sorted(kinds.items(), key=lambda kv: -len(kv[1])):
            print(f"    {k:<18} {len(v):>2}  ({', '.join(v[:5])})")
    print(f"\nfull results: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
