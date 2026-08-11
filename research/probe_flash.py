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

"""Which flash-attention configurations still compile on this toolchain?

Three hand-picked configurations failed, and I wrongly generalised that to "flash does not
compile". Two facts say that was premature: the plan holds TEN flash configurations spanning
`FA_H_TILES` 4/8/16, sequence lengths 1024-4096 and two work divisions, and the re-sweep never
reached them (they sit at plan positions 685-694, and it stopped at 249). So the empty flash
set in the fresh database is evidence of nothing.

This runs every flash configuration IN THE PLAN, exactly as the sweep would, with one warmup
and one rep -- enough to reach a compile failure or a first measurement, not enough to waste
the session. It classifies each outcome so the answer is a table rather than an impression.

    python3 research/probe_flash.py                 # all ten, from the plan
    python3 research/probe_flash.py --extra         # plus untiled / small-shape fallbacks
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
_PROFILE = os.path.join(_EXAMPLES, "profile_ops.py")
_PLAN = os.path.join(_ROOT, "tools", "cost_model", "sweep_plan.json")

_SUMMARY = re.compile(r"^SUMMARY .*?kernel_us=([0-9.]+)", re.M)

#: Configurations the plan does not contain, to separate "flash is broken" from "this
#: particular tiling is broken". If the untiled one compiles, the failure is about coarse
#: tiling, not about flash.
EXTRA = [
    {
        "name": "untiled_1024",
        "env": {
            "FA_H": "32",
            "FA_LQ": "1024",
            "FA_LK": "1024",
            "FA_H_TILES": "1",
            "FA_LQ_TILES": "1",
            "FA_LK_TILES": "1",
        },
    },
    {
        "name": "h_only_1024",
        "env": {
            "FA_H": "32",
            "FA_LQ": "1024",
            "FA_LK": "1024",
            "FA_H_TILES": "8",
            "FA_LQ_TILES": "1",
            "FA_LK_TILES": "1",
        },
    },
    {
        "name": "q_only_1024",
        "env": {
            "FA_H": "32",
            "FA_LQ": "1024",
            "FA_LK": "1024",
            "FA_H_TILES": "1",
            "FA_LQ_TILES": "4",
            "FA_LK_TILES": "1",
        },
    },
    {
        "name": "untiled_512",
        "env": {
            "FA_H": "8",
            "FA_LQ": "512",
            "FA_LK": "512",
            "FA_H_TILES": "1",
            "FA_LQ_TILES": "1",
            "FA_LK_TILES": "1",
        },
    },
]


def classify(raw):
    """The failure in one word, taken from the message the compiler actually printed."""
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


def run(env_extra, timeout_s=420):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in env_extra.items()})
    env.update({"BENCH_OP": "flash_attn", "BENCH_REPS": "1", "BENCH_WARMUP": "1"})
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--extra",
        action="store_true",
        help="also probe untiled / small-shape configurations",
    )
    ap.add_argument("--timeout", type=int, default=420)
    ap.add_argument("--out", default=os.path.join(_HERE, "flash_probe.json"))
    args = ap.parse_args()

    with open(_PLAN, encoding="utf-8") as fh:
        plan = json.load(fh)
    cfgs = plan["configs"] if isinstance(plan, dict) else plan
    cases = [
        {"name": f"plan#{i}", "env": {k: v for k, v in c.items() if k != "BENCH_OP"}}
        for i, c in enumerate(cfgs)
        if c.get("BENCH_OP") == "flash_attn"
    ]
    if args.extra:
        cases += EXTRA

    print(f"probing {len(cases)} flash configuration(s), 1 warmup + 1 rep each\n")
    print(
        f"  {'config':<14}{'H':>4}{'Lq':>6}{'Lk':>6}{'hT':>4}{'qT':>4}{'kT':>4}"
        f"{'result':>16}{'us':>10}"
    )
    rows = []
    for c in cases:
        e = c["env"]
        us, raw, dt = run(e, args.timeout)
        tag = "OK" if us else classify(raw)
        rows.append(
            {
                "name": c["name"],
                "env": e,
                "kernel_us": us,
                "result": tag,
                "seconds": round(dt, 1),
                "tail": None if us else (raw or "")[-600:],
            }
        )
        print(
            f"  {c['name']:<14}{e.get('FA_H', '-'):>4}{e.get('FA_LQ', '-'):>6}"
            f"{e.get('FA_LK', '-'):>6}{e.get('FA_H_TILES', '1'):>4}"
            f"{e.get('FA_LQ_TILES', '1'):>4}{e.get('FA_LK_TILES', '1'):>4}"
            f"{tag:>16}{(f'{us:.0f}' if us else '-'):>10}"
        )
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)

    ok = [r for r in rows if r["kernel_us"]]
    print(f"\n{len(ok)} of {len(rows)} compiled and measured.")
    if ok:
        print("Flash IS usable. Use these for the allocation experiment:")
        for r in ok:
            print(f"    {r['name']}: {r['kernel_us']:.0f} us  {r['env']}")
    else:
        by = {}
        for r in rows:
            by.setdefault(r["result"], []).append(r["name"])
        print("None compiled. Failures by kind:")
        for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
            print(
                f"    {k:<18} {len(v):>2}  ({', '.join(v[:4])}"
                f"{'...' if len(v) > 4 else ''})"
            )
        print("\nWith --extra, an untiled configuration that compiles would show the")
        print("break is in coarse tiling rather than in flash attention itself.")
    print(f"\nfull results: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
