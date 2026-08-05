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

"""Generate the coarse-tiling section of the report from the LIVE model.

Emits (a) the coefficient table and (b) EVERY coarse-tiling data point, so no number
in the write-up is ever hand-typed. Run after any model change:

    python3 tools/cost_model/coarse_tables.py            # markdown to stdout
    python3 tools/cost_model/coarse_tables.py --all      # include out-of-target rows too

POPULATION. "Target" = the realistic band the accuracy claim is about: coarse-tiling ops
excluding flash_attn, ROWS >= 2048, COLS >= 2048, cores = 32, and passing
``eval_model.in_scope`` (which drops rows whose recorded FEATURES predate the corrected
per-arg ``loop_factor`` -- scoring a model against a wrong byte count is meaningless).
Measured time is ``kernel_us``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_model as em  # noqa: E402
import regex as re  # noqa: E402

cm = em.cm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from records import records_path  # noqa: E402

RECORDS = records_path()

COARSE_OPS = [
    "matmul_row_tiling",
    "matmul_k_tiling",
    "mm_nested_m_k",
    "softmax_row_tiling",
    "softmax_noexp_row_tiling",
]


def realistic(r):
    return (
        r.get("cores") == 32
        and (r.get("rows") or 0) >= 2048
        and (r.get("cols") or 0) >= 2048
    )


def shape_of(r):
    """(M, K, N) from the label; N is None for the softmax ops (no third dim)."""
    lab = str(r.get("label") or "")
    m = re.search(r"(\d+)x(\d+)x(\d+)", lab) or re.search(
        r"M=(\d+) K=(\d+) N=(\d+)", lab
    )
    if m:
        return int(m[1]), int(m[2]), int(m[3])
    return r.get("rows"), r.get("cols"), None


def collect(target_only=True):
    with open(RECORDS, encoding="utf-8") as f:
        recs = json.load(f)["records"]
    dp = cm.CostParams()
    out = []
    for r in recs:
        if r.get("op") not in COARSE_OPS or r.get("failed") or not r.get("kernel_us"):
            continue
        if not em.in_scope(r):
            continue
        if target_only and not realistic(r):
            continue
        feats, _ = em.features_for(r, dp)
        if feats is None:
            continue
        try:
            pred = cm.predict_ops(feats, dp) / 1000.0
        except Exception:  # noqa: BLE001
            continue
        mm = next((o for o in feats if getattr(o, "is_matmul", False)), None)
        trpc = 0
        if mm is not None:
            trpc = int(
                getattr(mm, "tile_rows_per_core", 0)
                or getattr(mm, "matmul_rows_per_core", 0)
            )
        M, K, N = shape_of(r)
        out.append(
            dict(
                op=r["op"],
                M=M,
                K=K,
                N=N,
                t=int(r.get("tiles") or 1),
                cores=r.get("cores"),
                trpc=trpc,
                meas=r["kernel_us"],
                pred=pred,
                err=(pred - r["kernel_us"]) / r["kernel_us"] * 100.0,
                cv=r.get("kernel_us_cv"),
                log=str(r.get("log_file") or "")[:24],
            )
        )
    return out


def emit_params():
    p = cm.CostParams()
    print("### Coefficients that act on a coarse-tiled kernel\n")
    print("| coefficient | value | role |")
    print("|---|---:|---|")
    rows = [
        ("BW (read/write)", f"{p.mm_bw_read_gbps:.0f} GB/s", "HBM streaming rate"),
        (
            "alpha turnaround",
            f"{p.rw_turnaround_ns_per_byte} ns/B",
            "read/write bus turnaround",
        ),
        (
            "MAC peak",
            f"{p.mac_peak_per_core_ns:.0f} MAC/ns/core",
            "systolic-array rate",
        ),
        (
            "re-read scale",
            f"{p.loop_reread_scale}",
            "fraction of a full HBM pass per repeat",
        ),
        (
            "matmul spill cap",
            f"{p.mm_spill_ws_cap_bytes / 1048576:.0f} MB",
            "per-core LX before the tile spills",
        ),
        ("matmul spill exp", f"{p.mm_spill_ws_exp}", "spilled-traffic derate"),
        (
            "coarse underfill",
            f"r_full={p.coarse_underfill_rfull:.0f}, exp={p.coarse_underfill_exp}, cap={p.coarse_underfill_cap}",
            "short-tile pipeline fill (softmax-calibrated)",
        ),
    ]
    for n, v, d in rows:
        print(f"| `{n}` | {v} | {d} |")


def emit_table(rows, cap=28):
    print("\n### Every coarse-tiling data point in the target band\n")
    print("‼ marks a run beyond ±20 %.  `t` is the number of tiles the loop runs.\n")
    print(
        "| op | M | K | N | tiles | rows per core | measured µs | predicted µs | err % |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows = sorted(
        rows, key=lambda x: (x["op"], x["M"] or 0, x["K"] or 0, x["N"] or 0, x["t"])
    )
    total = len(rows)
    if total > cap:  # thin evenly rather than truncating, so the range still shows
        step = total / cap
        rows = [rows[int(i * step)] for i in range(cap)]
        print(
            f"\nA representative {cap} of the {total} scored runs; "
            "all of them: `python3 tools/cost_model/coarse_tables.py`.\n"
        )
    for d in rows:
        n = "-" if d["N"] is None else d["N"]
        flag = " ‼" if abs(d["err"]) > 20 else ""
        print(
            f"| `{d['op']}` | {d['M']} | {d['K']} | {n} | {d['t']} | "
            f"{d['trpc'] or '-'} | {d['meas']:.1f} | {d['pred']:.1f} | "
            f"{d['err']:+.1f}{flag} |"
        )


def emit_summary(rows):
    print("\n### Accuracy\n")
    print("| op | n | RMS % | mean % | worst % | >20 % |")
    print("|---|---:|---:|---:|---:|---:|")
    by = collections.defaultdict(list)
    for d in rows:
        by[d["op"]].append(d["err"])
    allе = []
    for op in sorted(by, key=lambda o: -len(by[o])):
        e = by[op]
        allе += e
        rms = (sum(v * v for v in e) / len(e)) ** 0.5
        print(
            f"| `{op}` | {len(e)} | {rms:.1f} | {st.mean(e):+.1f} | "
            f"{max(e, key=abs):+.1f} | {sum(1 for v in e if abs(v) > 20)} |"
        )
    rms = (sum(v * v for v in allе) / len(allе)) ** 0.5
    print(
        f"| **all** | **{len(allе)}** | **{rms:.1f}** | **{st.mean(allе):+.1f}** | "
        f"**{max(allе, key=abs):+.1f}** | **{sum(1 for v in allе if abs(v) > 20)}** |"
    )
    print("\n| \\|err\\| band | count |")
    print("|---|---:|")
    for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 1000)):
        print(
            f"| {lo}-{hi if hi < 1000 else '&infin;'} % | "
            f"{sum(1 for v in allе if lo <= abs(v) < hi)} |"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--all", action="store_true", help="include rows outside the target band"
    )
    args = ap.parse_args()
    rows = collect(target_only=not args.all)
    emit_params()
    emit_summary(rows)
    emit_table(rows)


if __name__ == "__main__":
    main()
