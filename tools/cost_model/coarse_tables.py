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


# Resolved lazily: importing this module must not exit with setup instructions.
def RECORDS():
    return records_path()


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
    with open(RECORDS(), encoding="utf-8") as f:
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
            f"r_full={p.coarse_underfill_rfull:g} at COLS {p.coarse_underfill_col_ref:.0f}, "
            f"exp={p.coarse_underfill_exp}, col_exp={p.coarse_underfill_col_exp}, "
            f"cap={p.coarse_underfill_cap}",
            "small-tile pipeline fill (softmax-calibrated)",
        ),
        (
            "coarse underfill (matmul)",
            f"r_full={p.coarse_underfill_rfull_matmul:g}, "
            f"exp={p.coarse_underfill_exp_matmul}, "
            f"cap={p.coarse_underfill_cap_matmul}",
            "same term, pre-re-fit rows-only curve, frozen",
        ),
        (
            "softmax spill exp",
            f"{p.lx_spill_exp}",
            "large-tile decline, re-fit jointly with the underfill surface",
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
    alle = []
    for op in sorted(by, key=lambda o: -len(by[o])):
        e = by[op]
        alle += e
        rms = (sum(v * v for v in e) / len(e)) ** 0.5
        print(
            f"| `{op}` | {len(e)} | {rms:.1f} | {st.mean(e):+.1f} | "
            f"{max(e, key=abs):+.1f} | {sum(1 for v in e if abs(v) > 20)} |"
        )
    rms = (sum(v * v for v in alle) / len(alle)) ** 0.5
    print(
        f"| **all** | **{len(alle)}** | **{rms:.1f}** | **{st.mean(alle):+.1f}** | "
        f"**{max(alle, key=abs):+.1f}** | **{sum(1 for v in alle if abs(v) > 20)}** |"
    )
    print("\n| \\|err\\| band | count |")
    print("|---|---:|")
    for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 20), (20, 1000)):
        print(
            f"| {lo}-{hi if hi < 1000 else '&infin;'} % | "
            f"{sum(1 for v in alle if lo <= abs(v) < hi)} |"
        )


def underfill_surface():
    """§16's surface table: the efficiency the MEASUREMENT requires at each (h, COLS),
    beside the one the model supplies.

    "Requires" is backed out per run rather than fitted to: with BOTH the underfill and
    the LX-spill derates switched off, ``eff_needed = predicted / measured`` -- valid
    because these kernels are memory-bound, and it is solved through the real
    ``predict_ops`` so the fill constant and the byte accounting are handled exactly. Cell
    = the mean over every run at that (h, COLS); n is how many. 32 cores only (below that
    the memory term has no core-count scaling, an unmodelled gap this table must not
    absorb) and tiled runs only.
    """
    with open(RECORDS(), encoding="utf-8") as f:
        recs = json.load(f)["records"]
    dp = cm.CostParams()
    _uf, _sp = cm.coarse_underfill_eff, cm._lx_spill_bw_derate
    cells = collections.defaultdict(list)
    for r in recs:
        if r.get("op") != "softmax_row_tiling" or r.get("failed"):
            continue
        if not r.get("kernel_us") or r.get("cores") != 32 or not em.in_scope(r):
            continue
        feats, _ = em.features_for(r, dp)
        if feats is None:
            continue
        h = min(
            (
                o.tile_rows_per_core
                for o in feats
                if o.loop_trip > 1 and o.tiles_output_dim and o.tile_rows_per_core > 0
            ),
            default=0.0,
        )
        if not h:
            continue
        cm.coarse_underfill_eff = lambda *a, **k: 1.0
        cm._lx_spill_bw_derate = lambda *a, **k: 1.0
        try:
            bare = cm.predict_ops(feats, dp) / 1000.0
        finally:
            cm.coarse_underfill_eff, cm._lx_spill_bw_derate = _uf, _sp
        cells[(h, r["cols"])].append(bare / r["kernel_us"])

    widths = sorted({c for _, c in cells})
    print(
        "\n### §16 -- the efficiency each tile requires, and the one the model gives\n"
    )
    print("| `h` | " + " | ".join(f"COLS {c}" for c in widths) + " |")
    print("|---:|" + "---:|" * len(widths))
    # A LADDER of heights, not every one: the climb is the informative part and the plateau
    # repeats itself. Same convention as emit_table -- thin, and say so below.
    heights = sorted({hh for hh, _ in cells})
    ladder = [h for h in heights if h in (2, 4, 8, 16, 64, 256)] or heights
    for h in ladder:
        out = []
        for c in widths:
            v = cells.get((h, c))
            if not v:
                out.append("—")
                continue
            need = st.mean(v)
            ws = 2.0 * h * c * 2  # ~2 live intermediate tiles, fp16
            spill = (
                min(1.0, (dp.lx_spill_cap_bytes / ws) ** dp.lx_spill_exp)
                if ws > dp.lx_spill_cap_bytes
                else 1.0
            )
            got = cm.coarse_underfill_eff(h, c, dp) * spill
            out.append(
                f"{need:.2f} / {got:.2f}" + (f" ({len(v)})" if len(v) > 1 else "")
            )
        print(f"| {h:g} | " + " | ".join(out) + " |")
    print(
        f"\nEach cell is `required / modelled`, averaged over the runs at that shape "
        f"(count in brackets). {len(ladder)} of the {len(heights)} tile heights measured; "
        f"the widths below 1024 that identify the width term are in the figure."
    )


def spill_bands():
    """§17's table: achieved bandwidth against the working set one core holds.

    Pure measurement -- no model term enters -- but it belongs here rather than typed into
    the section, because both the levels and the run counts move with the compiler build.
    Same population as §16's surface table: tiled softmax, 32 cores, in scope.
    """
    with open(RECORDS(), encoding="utf-8") as f:
        recs = json.load(f)["records"]
    bands = [(0.0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float("inf"))]
    got = collections.defaultdict(list)
    for r in recs:
        if r.get("op") != "softmax_row_tiling" or r.get("failed"):
            continue
        if not r.get("kernel_us") or r.get("cores") != 32 or not em.in_scope(r):
            continue
        t = int(r.get("tiles") or 1)
        if t < 2 or not r.get("io_hbm_bytes"):
            continue
        ws = 2 * (r["rows"] / (32 * t)) * r["cols"] * 2 / 1e6
        for lo, hi in bands:
            if lo <= ws < hi:
                got[(lo, hi)].append(int(r["io_hbm_bytes"]) / 1e3 / r["kernel_us"])
                break
    print("\n### §17 -- achieved bandwidth by per-core working set\n")
    print("| per-core working set (MB) | runs | achieved bandwidth (GB/s) |")
    print("|---|---:|---:|")
    for lo, hi in bands:
        v = got.get((lo, hi))
        if not v:
            continue
        lab = f"above {lo:g}" if hi == float("inf") else f"{lo:g} – {hi:g}"
        print(f"| {lab} | {len(v)} | {st.mean(v):.1f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--all", action="store_true", help="include rows outside the target band"
    )
    args = ap.parse_args()
    rows = collect(target_only=not args.all)
    emit_params()
    emit_summary(rows)
    underfill_surface()
    spill_bands()
    emit_table(rows)


if __name__ == "__main__":
    main()
