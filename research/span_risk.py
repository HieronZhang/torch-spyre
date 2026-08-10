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

"""Which sweep configurations could trip the per-core address-span limit?

WHY. On 2026-08-07 exactly one configuration in 1640 emitted
``[CRITICAL] per-core tensor span 256.000 MB (shape=[4,1024,1024,1024]) exceeds hardware
limit``, and the run immediately after it began 1389 consecutive DDR-init failures that
killed the card. One co-occurrence is not proof of cause, but before spending another night
on hardware it is worth knowing whether anything else in the plan can reach the same state.

THE MECHANISM. Coarse tiling inserts a read copy for every graph input
(``coarse_tile.py::_insert_read_copy_ops``). When the tiled dimension is a matmul's
REDUCTION dim, that copy is materialised over the full iteration space rather than over the
input, so a `[B,M,K] @ [K,N]` K-tiled by T becomes a `[B, M, N, K/T]` staging buffer. The
inputs were 20 MB; the buffer is 8.6 GB.

    per_core_span  =  B * M * N * (K / tiles) * dtype_bytes / cores

That formula reproduces the logged case exactly -- shape, total bytes, per-core span
268,435,456 B, and its 4,096-byte overshoot of ``MAX_SPAN_BYTES = 65535 * 4096``. It is
therefore used here as a predictor, not a guess.

SCOPE, AND WHAT THIS DOES NOT COVER. It applies only to REDUCTION-dim (K) tiling of the
matmul family. Output-dim tiling (``matmul_row_tiling``, the softmax ops) does not build a
read copy over the reduction space and is not screened. The screen is also cross-checked
against measurement, but read that cross-check carefully: it reports the BUILD on which a
configuration last succeeded, not a boolean. Every flagged configuration last measured fine
on a July or early-August build -- and so did the quarantined one, at ~1171 us on three of
them, right before it regressed. On the 2026-08-07 build these ops never compiled at all:
the card died at run 252 and they sort after it, so every later attempt failed at device
init. There is therefore NO evidence either way about them on the current build.

    python3 research/span_risk.py
"""

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

#: work_division.py:73 -- MVLOC addresses 65535 entries of 4096 bytes.
MAX_SPAN_BYTES = 65535 * 4096

#: Ops whose coarse tiling targets a matmul REDUCTION dimension, and are therefore subject
#: to the read-copy amplification above. Output-dim tiling is a different shape and safe.
K_TILED_OPS = {
    "matmul_k_tiling",
    "bmm_k_tiling",
    "bmm_3d2d_k_tiling",
    "bmm_nested_b_k",
    "mm_nested_m_k",
}


def predicted_span(cfg, dtype_bytes=2):
    """Per-core bytes of the read-copy staging buffer, or None if not applicable."""
    op = cfg.get("BENCH_OP")
    if op not in K_TILED_OPS:
        return None
    try:
        tiles = int(cfg.get("BENCH_TILES", 0) or 0)
        if tiles < 2:
            return None  # untiled: no read copy is inserted
        m = int(cfg["BENCH_ROWS"])
        k = int(cfg["BENCH_COLS"])
        n = int(cfg.get("BENCH_N", k))
        b = int(cfg.get("BENCH_B", 1) or 1)
        cores = int(cfg.get("SENCORES", 32) or 32)
    except (KeyError, TypeError, ValueError):
        return None
    if op in ("matmul_k_tiling", "mm_nested_m_k"):
        b = 1  # 2-D mm has no batch dimension
    return b * m * n * (k // tiles) * dtype_bytes // max(1, cores)


def measured_ok(cfg, records):
    """When did this configuration last produce a real measurement, and on which build?

    "It measured fine before" is only reassuring if "before" means the CURRENT build. The
    quarantined `bmm_3d2d_k_tiling` measured fine at ~1171 us on three July builds and then
    emitted a CRITICAL span on the August build -- the regression is precisely what changed
    between them. So this returns the build, not a boolean.
    """
    sys.path.insert(0, os.path.join(_ROOT, "docs", "source", "user_guide", "examples"))
    import run_cost_model_sweep as S

    key = tuple(sorted(cfg.items()))
    best = None
    for r in records:
        if r.get("failed") or not r.get("kernel_us"):
            continue
        env = S._env_from_record(r)
        if env and tuple(sorted(env.items())) == key:
            stamp = (r.get("log_date") or "", r.get("model_sha") or "?")
            if best is None or stamp > best:
                best = stamp
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--plan", default=os.path.join(_ROOT, "tools/cost_model/sweep_plan.json")
    )
    ap.add_argument("--records", default="", help="database to cross-check against")
    args = ap.parse_args()

    with open(args.plan, encoding="utf-8") as fh:
        plan = json.load(fh)["configs"]
    records = []
    if args.records and os.path.exists(args.records):
        with open(args.records, encoding="utf-8") as fh:
            records = json.load(fh)["records"]

    rows = []
    for c in plan:
        span = predicted_span(c)
        if span is not None:
            rows.append((span, c))
    rows.sort(key=lambda t: -t[0])

    print(
        f"MAX_SPAN_BYTES = {MAX_SPAN_BYTES:,} ({MAX_SPAN_BYTES / 2**20:.2f} MiB/core)"
    )
    print(
        f"{len(rows)} reduction-dim-tiled configurations screened "
        f"out of {len(plan)} in the plan\n"
    )

    over = [(s, c) for s, c in rows if s > MAX_SPAN_BYTES]
    near = [(s, c) for s, c in rows if MAX_SPAN_BYTES / 2 < s <= MAX_SPAN_BYTES]

    def show(title, items):
        if not items:
            print(f"{title}: none\n")
            return
        print(f"{title}: {len(items)}")
        for s, c in items:
            tag = ""
            if records:
                got = measured_ok(c, records)
                tag = (
                    f"  [last OK {got[0][:8] or '?'} sha {got[1]}]"
                    if got
                    else "  [NEVER measured]"
                )
            print(
                f"   {s / 2**20:8.1f} MiB/core  {c.get('BENCH_OP'):<20} "
                f"B={c.get('BENCH_B', 1)} M={c.get('BENCH_ROWS')} K={c.get('BENCH_COLS')} "
                f"N={c.get('BENCH_N')} tiles={c.get('BENCH_TILES')} "
                f"cores={c.get('SENCORES')}{tag}"
            )
        print()

    show("OVER the limit -- would emit CRITICAL", over)
    show("within 2x of the limit -- worth watching", near)
    if not over:
        print("No other planned configuration is predicted to exceed the limit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
