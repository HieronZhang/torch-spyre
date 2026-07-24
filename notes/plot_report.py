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

"""Regenerate the figures for notes/cost_model_report.md from sweep_records.json.

Each figure is a function ``fig_<name>`` that reads the stored measured times and writes
a PNG to notes/figures/. Reproducible -- the data is the committed sweep records, not
hand-drawn. Run one figure or all:

    python notes/plot_report.py                 # all figures
    python notes/plot_report.py pointwise_baseline
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIGDIR = os.path.join(_HERE, "figures")
_RECORDS = os.path.join(_HERE, "sweep_records.json")


def _load(current_only=True):
    recs = json.load(open(_RECORDS, encoding="utf-8"))["records"]
    recs = [r for r in recs if not r.get("failed") and r.get("kernel_us")]
    if current_only:
        recs = [r for r in recs if r.get("is_current")]
    return recs


def _save(fig, name, dpi=130):
    os.makedirs(_FIGDIR, exist_ok=True)
    path = os.path.join(_FIGDIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, _HERE)}")


# ============================================================================
# §1 -- pointwise is memory-I/O bound: neg time is linear in bytes through the
# origin (no fixed per-kernel cost).
# ============================================================================
def fig_pointwise_baseline(recs):
    pts = [
        (int(r["io_hbm_bytes"]) / 1e6, r["kernel_us"])
        for r in recs
        if r["op"] == "neg" and r.get("io_hbm_bytes")
    ]
    allx = np.array([x for x, _ in pts])
    ally = np.array([y for _, y in pts])
    # linear fit T[us] = a*MB + b (b -> fixed cost; a -> 1/BW)
    a, b = np.polyfit(allx, ally, 1)
    r2 = 1 - np.sum((ally - (a * allx + b)) ** 2) / np.sum((ally - ally.mean()) ** 2)
    bw = 1e3 / a  # MB/us = GB/s

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    xline = np.array([0, allx.max() * 1.03])
    ax.plot(
        xline,
        a * xline + b,
        "-",
        color="0.5",
        lw=1.2,
        zorder=1,
        label=f"linear fit: {bw:.0f} GB/s, $R^2$={r2:.4f}",
    )
    ax.scatter(
        allx,
        ally,
        s=42,
        color="#1f77b4",
        label="neg",
        zorder=3,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("HBM traffic  (MB, device / stick-padded)")
    ax.set_ylabel("kernel time  (µs)")
    ax.set_title("§1  Pointwise is memory-I/O bound (neg, 1R:1W)")
    ax.annotate(
        f"intercept b = {b:+.1f} µs  ≈ 0\n(no fixed per-kernel cost)",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=9,
        color="0.25",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.8"),
    )
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.margins(x=0.02)
    _save(fig, "fig1_pointwise_baseline")


# ============================================================================
# §2 -- the read/write mix sets the effective BW: a symmetric turnaround valley.
# effBW vs write-fraction w=W/(R+W): high at w=0 (read-only) and w=1 (write-only),
# lowest at w=0.5 (balanced). Model: effBW = 1/(1/BW_peak + alpha*min(w,1-w)).
# ============================================================================
def _pw_ratio_write_points():
    """(w, effBW) for the write-only anchor from the pw_ratio decoupler log (not yet in
    sweep_records). ROWS=2048 only (small operand resident, pre-spill)."""
    import glob

    import regex as re

    logs = sorted(
        glob.glob(os.path.join(_HERE, "..", "haoyang_logs", "pw_ratio_*.log"))
    )
    if not logs:
        return []
    R = W = None
    pts = []
    for ln in open(logs[-1], encoding="utf-8"):
        m = re.search(r"^MODEL\s+R=(\d+) B .*?W=(\d+) B", ln)
        if m:
            R, W = int(m[1]), int(m[2])
        s = re.search(r"SUMMARY op=write rows=2048 .*kernel_us=([0-9.]+)", ln)
        if s and R is not None:
            tot = R + W
            pts.append((W / tot, tot / 1e3 / float(s[1])))
            R = W = None
    return pts


def fig_pointwise_vcurve(_recs):
    recs = _load(
        current_only=False
    )  # measured times are version-independent; recover the
    # read-only / neg groups that the newest is_current sweep does not carry.
    bw_peak, alpha = 150.0, 0.00574  # current model
    reds = ("sumrow", "read", "amax", "mean")  # read-only anchor (ROWS=2048, clean)
    # label -> (points, color, marker). Each op is swept at several sizes -> several
    # points per class (small vertical spread = mild size drift, ~2-3%).
    groups = {
        "read-only": ([], "#2ca02c", "o"),
        "2R:1W (add)": ([], "#9467bd", "o"),
        "1R:1W (neg)": ([], "#1f77b4", "o"),
        "write-only": ([], "#d62728", "o"),
    }
    for r in recs:
        m = r.get("model") or {}
        R, W = m.get("R"), m.get("W")
        # ROWS=2048 (pipeline-filled) and lx=0 (canonical): single ops have no intermediate,
        # so scratchpad is irrelevant -- but one old overnight lx=1 sweep measured `add` slow.
        if not R or not W or r.get("rows") != 2048 or r.get("lx") not in (0, None):
            continue
        eff = (R + W) / 1e3 / r["kernel_us"]
        w = W / (R + W)
        if r["op"] in reds:
            groups["read-only"][0].append((w, eff))
        elif r["op"] == "add":
            groups["2R:1W (add)"][0].append((w, eff))
        elif r["op"] == "neg":
            groups["1R:1W (neg)"][0].append((w, eff))
    groups["write-only"][0].extend(_pw_ratio_write_points())

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ww = np.linspace(0, 1, 200)
    model = 1.0 / (1.0 / bw_peak + alpha * np.minimum(ww, 1 - ww))
    ax.plot(
        ww,
        model,
        "-",
        color="0.55",
        lw=1.3,
        zorder=1,
        label=f"model (BW_peak={bw_peak:.0f}, α={alpha})",
    )
    for lab, (pts, c, mk) in groups.items():
        if pts:
            xs, ys = zip(*pts)
            kw = dict(s=34, color=c, label=lab, marker=mk, zorder=3)
            if mk != "x":  # filled markers get a white edge; 'x' is a stroke marker
                kw.update(edgecolors="white", linewidths=0.4)
            ax.scatter(xs, ys, **kw)
    ax.set_xlabel("write fraction  w = W / (R + W)")
    ax.set_ylabel("effective BW  (R+W)/time  (GB/s)")
    ax.set_title("§2  R/W mix sets effective BW (turnaround valley)")
    ax.set_ylim(90, 158)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=8,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    _save(fig, "fig2_pointwise_vcurve")


# ============================================================================
# §3 -- n-ary adds: the byte model predicts the SAME effBW across arity (all sit
# at w=1/3), but measured effBW DECLINES with each chained op -> a per-op derate.
# ============================================================================
def _arity_avg(cols=4096):
    """Mean kernel_us per op at ROWS=2048, given COLS, over the control runs (the
    new_experiments sweeps that carry add3_sep/add4_sep/add_indep2). ROWS=2048 ONLY --
    the ARITYX shapes reuse the same COLS at other ROWS and would corrupt the mean."""
    from collections import defaultdict

    acc = defaultdict(list)
    for r in _load(current_only=False):
        if (
            str(r.get("log_file", "")).startswith("new_experiments")
            and r.get("rows") == 2048
            and r.get("cols") == cols
            and r.get("kernel_us")
        ):
            acc[(r["op"], r.get("lx"))].append(r["kernel_us"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def fig_pointwise_arity(recs):
    # §3 fig A: where the add3 margin comes from, in ABSOLUTE µs (ROWS=2048, COLS=4096, buf
    # fits LX). One clean panel: the no-dependency control (add_indep2) sits on the byte-count
    # baseline; the same chain as separate kernels (add3_sep) and fused (add3) both sit +7%
    # above -- so the margin is the read-after-write DEPENDENCY and fusion is free; with
    # scratchpad on the intermediate stays on-chip and the margin is gone.
    a = _arity_avg(4096)
    add = a[("add", 0)]
    base = 2 * add
    bars = [
        ("add_indep2\n(no dependency)", a[("add_indep2", 0)], "#2ca02c", ""),
        ("add3_sep\n(dep, separate)", a[("add3_sep", 0)], "#ff7f0e", ""),
        ("add3\n(dep, fused)", a[("add3", 0)], "#1f77b4", ""),
        ("add3, LX on\n(buf on-chip)", a[("add3", 1)], "#9ecae1", "//"),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    xs = range(len(bars))
    ax.bar(
        xs,
        [b[1] for b in bars],
        color=[b[2] for b in bars],
        hatch=[b[3] for b in bars],
        edgecolor="white",
        zorder=3,
        width=0.66,
    )
    ax.axhline(base, ls="--", color="0.35", lw=1.6, zorder=2)
    ax.text(
        3.42,
        base,
        "byte-count baseline\n(2 × add)",
        va="center",
        ha="left",
        fontsize=11,
        color="0.35",
    )
    for x, b in zip(xs, bars):
        ax.annotate(
            f"{b[1]:.0f} µs\n({100 * (b[1] / base - 1):+.0f}%)",
            (x, b[1]),
            ha="center",
            va="bottom",
            fontsize=12.5,
            color="0.1",
            weight="bold",
        )
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars], fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_ylabel("kernel time  (µs)", fontsize=13)
    ax.set_ylim(0, base * 1.55)
    ax.set_xlim(-0.6, 4.3)
    ax.set_title(
        "§3  Where the add-chain margin comes from  (add3; ROWS=2048, COLS=4096)\n"
        "no-dependency = baseline · dependency = +7% (fused = separate) · on-chip = gone",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "fig3_pointwise_arity")


def fig_pointwise_arity_reads(recs):
    # §3 fig B: total EXCESS cost vs chain length, fused vs separate. excess(add_n) =
    # t(add_n)/t(add) - (n-1) = "extra single-adds' worth of time beyond the byte count."
    # Plotted against the number of dependent reads (n-2). The dependency/fusion cost is the
    # HEIGHT of each curve; the per-read cost is its SLOPE; the fusion cost is the GAP between
    # fused and separate. No reindexing -- reads directly. FUSED covers add3..add6 (1-4 reads);
    # SEPARATE covers add3_sep, add4_sep (1-2) until add5_sep/add6_sep land.
    cols = [1024, 4096, 16384]  # the replicated reference shapes (ROWS=2048 only)
    rows = [
        r
        for r in _load(current_only=False)
        if str(r.get("log_file", "")).startswith("new_experiments")
        and r.get("rows") == 2048
        and r.get("lx") == 0
        and r.get("kernel_us")
        and r.get("cols") in cols
    ]

    def times(
        op, C
    ):  # ALL measurements (every run × replicate) of op at ROWS=2048, COLS=C
        return [r["kernel_us"] for r in rows if r["op"] == op and r["cols"] == C]

    def excess(
        seq,
    ):  # {n_dep_reads: [excess per measurement, over shapes × replicates]}
        out = {}
        for n, op in seq:
            vals = []
            for C in cols:
                add_t = times("add", C)
                if not add_t:
                    continue
                a = sum(add_t) / len(add_t)  # this shape's single-add baseline
                vals += [v / a - (n - 1) for v in times(op, C)]
            if vals:
                out[n - 2] = vals  # x = number of dependent reads
        return out

    fused = excess([(3, "add3"), (4, "add4"), (5, "add5"), (6, "add6")])
    sep = excess([(3, "add3_sep"), (4, "add4_sep"), (5, "add5_sep"), (6, "add6_sep")])

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.axhline(0, color="0.6", lw=1.0, zorder=1, label="byte count (no excess)")
    means = {}
    for data, col, mk, lab in [
        (fused, "#1f77b4", "o", "fused chain (add3…add6)"),
        (sep, "#ff7f0e", "s", "separate kernels (add3_sep…add6_sep)"),
    ]:
        xs = sorted(data)
        for k in xs:
            ax.scatter(
                [k] * len(data[k]),
                data[k],
                s=26,
                color=col,
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
            )
        ax.plot(
            xs,
            [sum(data[k]) / len(data[k]) for k in xs],
            "-",
            color=col,
            marker=mk,
            ms=7,
            lw=1.8,
            zorder=4,
            label=lab,
        )
        means[lab[:5]] = {k: sum(data[k]) / len(data[k]) for k in xs}
    # reference line fit through the FUSED means -> the read-after-write dependency
    # accumulates LINEARLY in chain length. Fused add4 lies ON this line; it is add4_sep
    # that falls BELOW it. (Fit recomputed from the live data -- never hand-typed.)
    xr = sorted(fused)
    slope, icpt = np.polyfit(xr, [means["fused"][k] for k in xr], 1)
    ax.plot(
        [xr[0] - 0.15, xr[-1] + 0.15],
        [slope * (xr[0] - 0.15) + icpt, slope * (xr[-1] + 0.15) + icpt],
        "--",
        color="0.5",
        lw=1.1,
        zorder=2,
        label="linear read-after-write trend (fit to fused)",
    )
    # the lone divergence at add4 (depth 2): it is add4_SEP that dips BELOW the trend,
    # NOT the fused op rising above it. The fused chain is on-trend and the cost model
    # predicts fused add4 to ~2% -- so this is a separate-kernel (multi-launch) control
    # artifact, not a fused-kernel pathology.
    if 2 in fused and 2 in sep:
        fu, se = means["fused"][2], means["separ"][2]
        ax.annotate(
            "",
            xy=(2, fu),
            xytext=(2, se),
            arrowprops=dict(arrowstyle="<->", color="#a03000", lw=1.4),
        )
        ax.annotate(
            "add4_sep stays FLAT (its dependency\ncost engages only at the 4th launch,\n"
            "add5_sep); the fused chain already\nstepped — the gap is the control,\nnot a fused pathology",
            xy=(2, se),
            xytext=(2.28, se - 0.05),
            fontsize=7.2,
            color="#a03000",
            va="center",
        )
    ax.annotate(
        "fused = separate at 1, 3, 4 reads\n→ read-after-write dependency\n"
        "   (both chains grow together)",
        xy=(3.5, (means.get("fused", {}).get(4, 0.78))),
        xytext=(0.7, 0.60),
        fontsize=7.6,
        color="#1f4e8c",
        arrowprops=dict(arrowstyle="->", color="0.5", lw=0.8),
    )
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1\n(add3)", "2\n(add4)", "3\n(add5)", "4\n(add6)"])
    ax.set_xlabel("number of dependent reads in the chain")
    ax.set_ylabel(
        "excess cost   t(add$_n$)/t(add) − (n−1)   [extra adds' worth of time]"
    )
    ax.set_title(
        "§3  Extra cost vs chain length: read-after-write accumulates (fused = separate)\n"
        "— except at add4, where the separate control stays flat (not a fused pathology)"
    )
    ax.legend(loc="upper left", fontsize=7.6, framealpha=0.9)
    _save(fig, "fig3b_pointwise_arity_reads")


# ============================================================================
# §4 -- a broadcast operand raises the effective BW: broadcast-operand ops
# (copy/bcast/bcastcol/mulbcast) run ~118 GB/s, above plain 1R:1W neg (~105).
# Small-ROWS points are underfilled (2-8 rows/core) and unreliable -> flagged.
# ============================================================================
def fig_broadcast_effbw(_recs):
    # measured times are version-independent -> use all records (not just is_current,
    # which the newest-log SHA can shrink). neg (1R:1W) baseline + the broadcast ops.
    recs = _load(current_only=False)
    neg = [
        int(r["io_hbm_bytes"]) / 1e3 / r["kernel_us"]
        for r in recs
        if r["op"] == "neg" and r.get("io_hbm_bytes")
    ]
    # all broadcast points at ROWS=2048 from the records (includes the small-COLS sweep)
    from collections import defaultdict

    brec = defaultdict(lambda: defaultdict(list))
    for r in _load(current_only=False):
        if (
            r.get("op") in ("copy", "bcast", "bcastcol", "mulbcast")
            and r.get("rows") == 2048
            and r.get("io_hbm_bytes")
        ):
            brec[r["op"]][r["cols"]].append(
                int(r["io_hbm_bytes"]) / 1e3 / r["kernel_us"]
            )
    colors = {
        "copy": "#d62728",
        "bcast": "#2ca02c",
        "bcastcol": "#9467bd",
        "mulbcast": "#ff7f0e",
    }

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.axhspan(
        min(neg),
        max(neg),
        color="#1f77b4",
        alpha=0.12,
        zorder=0,
        label=f"neg (1R:1W) range, all sizes: {min(neg):.0f}–{max(neg):.0f}",
    )
    ax.axhline(
        np.mean(neg),
        ls="--",
        color="#1f77b4",
        lw=1.1,
        label=f"neg mean ≈ {np.mean(neg):.0f}",
    )
    ax.axhline(
        118, ls="--", color="0.5", lw=1.0, label="broadcast rate = 118 (large-size)"
    )
    for op, c in colors.items():
        pts = sorted((C, sum(v) / len(v)) for C, v in brec[op].items())
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=c, ms=4.5, label=op)
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384])
    ax.set_xticklabels(["256", "512", "1k", "2k", "4k", "8k", "16k"])
    ax.set_xlabel("COLS  (ROWS = 2048)")
    ax.set_ylabel("effective BW  (R+W)/time  (GB/s)")
    ax.set_title("§4  Broadcast ops: ~118 at large size, rising to ~130 when small")
    ax.set_ylim(100, 140)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9, ncol=2)
    _save(fig, "fig4_broadcast_effbw")


# ============================================================================
# §5 -- reductions are read-dominated -> read-only rate (~150), EXCEPT the output
# [R] is stick-inflated to R×64 and written scattered; at large R it drags effBW down.
# ============================================================================
def fig_reduction(_recs):
    # The read rate FALLS with ROWS, op-independently. Use all records (version-
    # independent measured times) for the full ROWS sweep; overlay the model.
    recs = _load(current_only=False)
    ops = ("read", "sumrow", "amax", "mean", "sumall")
    colors = {
        "read": "#1f77b4",
        "sumrow": "#2ca02c",
        "amax": "#9467bd",
        "mean": "#ff7f0e",
        "sumall": "#8c564b",
    }
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    # model: effBW = min(150, 114 + 61*exp(-ROWS/3700))
    xs = np.linspace(2048, 16384, 100)
    ax.plot(
        xs,
        np.minimum(150, 114 + 61 * np.exp(-xs / 3700)),
        "-",
        color="0.4",
        lw=1.6,
        zorder=2,
        label="model 114+61·e^(−ROWS/3700)",
    )
    for op in ops:
        pts = {}
        for r in recs:
            if r["op"] == op and r.get("io_hbm_bytes") and (r.get("cols") == 2048):
                pts.setdefault(r.get("rows") or 0, []).append(
                    int(r["io_hbm_bytes"]) / 1e3 / r["kernel_us"]
                )
        pts = sorted((x, sum(v) / len(v)) for x, v in pts.items())
        if pts:
            xx, yy = zip(*pts)
            ax.scatter(
                xx,
                yy,
                s=36,
                color=colors[op],
                label=op,
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks([2048, 4096, 8192, 16384])
    ax.set_xticklabels(["2048", "4096", "8192", "16384"])
    ax.set_xlabel("ROWS  (input height;  COLS = 2048)")
    ax.set_ylabel("read rate  (R+W)/time  (GB/s)")
    ax.set_title("§5  Reduction read rate falls with ROWS (op-independent)")
    ax.set_ylim(105, 158)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    _save(fig, "fig5_reduction")


def _broadcast_log_rows():
    """Parse the standalone broadcast sweep log (not folded into sweep_records)."""
    import glob

    import regex as re

    logs = sorted(
        glob.glob(os.path.join(_HERE, "..", "haoyang_logs", "broadcast_*.log"))
    )
    if not logs:
        return []
    rows = []
    for ln in open(logs[-1], encoding="utf-8"):
        s = re.search(
            r"SUMMARY op=(\w+) rows=(\d+) cols=(\d+).*io_hbm_bytes=(\d+) kernel_us=([0-9.]+)",
            ln,
        )
        if s:
            rows.append((s[1], int(s[2]), int(s[3]), int(s[4]), float(s[5])))
    return rows


# ============================================================================
# §4b -- write spill: per-output-byte cost vs COLS, one line per ROWS. Rises with
# C (the b[1,C] row operand) AND with R -> C-dominant but not a clean single spill.
# ============================================================================
def fig_write_spill(_recs):
    # effective BW = HBM bytes moved / time. The outer-product write does far more work than
    # its counted bytes suggest, so effBW collapses as ROWS *and* COLS grow. Plot EVERY ROWS
    # present (measured, solid) with the refit model overlaid (dashed) so fit + coverage are
    # both visible. Uses ALL write records (measured times are version-independent).
    from collections import defaultdict

    cm = _cost_model()
    recs = _load(current_only=False)
    meas = defaultdict(dict)  # meas[R][C] = mean measured effBW
    model = defaultdict(
        dict
    )  # model[R][C] = model effBW (from predict_ops on the feats)
    tmp = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r.get("op") != "write" or not r.get("io_hbm_bytes") or not r.get("cols"):
            continue
        if r.get("cores") not in (32, None) or r.get("failed"):
            continue
        R, C, io = r.get("rows"), r["cols"], int(r["io_hbm_bytes"])
        tmp[R][C].append(io / 1e3 / r["kernel_us"])
        if C not in model[R] and r.get("feats"):  # one model point per (R,C)
            try:
                feats = r["feats"]
                feats = feats if isinstance(feats, list) else json.loads(feats)
                pred = cm.predict_ops(cm.ops_from_json(json.dumps(feats))) / 1e3
                model[R][C] = io / 1e3 / pred
            except Exception:  # noqa: BLE001 - a bad feats row just omits its model point
                pass
    for R in tmp:
        for C, v in tmp[R].items():
            meas[R][C] = sum(v) / len(v)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    # only ROWS with >=2 COLS points -- a lone point (R=64/256, measured only at C=16384)
    # shows no trend, so it is omitted rather than drawn as an isolated dot.
    rows_present = sorted(R for R in meas if len(meas[R]) >= 2)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(rows_present)))
    for color, R in zip(cmap, rows_present):
        mpts = sorted(meas[R].items())
        xs, ys = zip(*mpts)
        ax.plot(xs, ys, "-o", color=color, ms=5, label=f"ROWS={R}")
        gpts = sorted(model.get(R, {}).items())
        if gpts:
            gx, gy = zip(*gpts)
            ax.plot(gx, gy, "--", color=color, lw=1.1, alpha=0.7)
    ax.plot([], [], "k--", lw=1.1, label="model (refit)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 2048, 4096, 8192, 16384])
    ax.set_xticklabels(["1k", "2k", "4k", "8k", "16k"])
    ax.set_xlabel("COLS")
    ax.set_ylabel("effective BW  (R+W)/time  (GB/s)")
    ax.set_title(
        "§4  write: effective BW vs COLS, every ROWS (solid=measured, dashed=model)"
    )
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9, ncol=2)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    _save(fig, "fig4b_write_spill")


# ============================================================================
# §4c -- small-ROWS broadcast valley: bcast effBW vs ROWS, one line per COLS. The
# minimum sits at ROWS = COLS/64 (output stick-planes) -- a V that is only a rising
# side for small COLS (dip below range) and a full V for large COLS. Model overlaid.
# ============================================================================
def fig_broadcast_smallr(_recs):
    from collections import defaultdict

    cm = _cost_model()
    recs = _load(current_only=False)
    meas = defaultdict(dict)  # meas[C][R]
    model = defaultdict(dict)
    tmp = defaultdict(lambda: defaultdict(list))
    for r in recs:  # the dense small-ROWS sweep (bcast), cores=32
        if r.get("op") != "bcast" or r.get("cores") != 32 or r.get("failed"):
            continue
        if "bcast_smallr" not in (
            r.get("log_file") or ""
        ):  # dense same-build sweep only
            continue
        R, C, io = r.get("rows"), r.get("cols"), r.get("io_hbm_bytes")
        if not io or not R or not C or R > 1024:
            continue
        tmp[C][R].append(io / 1e3 / r["kernel_us"])
        if R not in model[C] and r.get("feats"):
            try:
                feats = r["feats"]
                feats = feats if isinstance(feats, list) else json.loads(feats)
                pred = cm.predict_ops(cm.ops_from_json(json.dumps(feats))) / 1e3
                model[C][R] = io / 1e3 / pred
            except Exception:  # noqa: BLE001
                pass
    for C in tmp:
        for R, v in tmp[C].items():
            meas[C][R] = sum(v) / len(v)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    cols = sorted(meas)
    cmap = plt.cm.plasma(np.linspace(0, 0.85, len(cols)))
    for color, C in zip(cmap, cols):
        mp = sorted(meas[C].items())
        xs, ys = zip(*mp)
        ax.plot(xs, ys, "-o", color=color, ms=6, label=f"COLS={C // 1024}k")
        gp = sorted(model.get(C, {}).items())
        if gp:
            gx, gy = zip(*gp)
            ax.plot(gx, gy, "--", color=color, lw=1.1, alpha=0.75)
        dipR = C / 64.0  # the model's valley floor: ROWS = COLS/64 (stick-planes)
        if xs[0] <= dipR <= xs[-1]:
            ax.axvline(dipR, color=color, ls=":", lw=0.8, alpha=0.5)
    ax.plot([], [], "k--", lw=1.1, label="model (quad COLS≤4k / V COLS≥8k)")
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256, 512, 1024])
    ax.set_xticklabels([64, 128, 256, 512, 1024])
    ax.set_xlabel("ROWS")
    ax.set_ylabel("effective BW  (R+W)/time  (GB/s)")
    ax.set_title("§4  small-ROWS bcast: a V-valley with minimum at ROWS = COLS/64")
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9, ncol=2)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    _save(fig, "fig4c_broadcast_smallr")


# ============================================================================
# §6 -- transport ops (transpose / cat) lower to a byte-copy (`clone`); the only
# difference from a plain copy is the access pattern, which sets an effective BW.
# transpose is flat-fast (116); cat0 and transpose_outer fall with size.
# ============================================================================
def fig_transport(_recs):
    # effBW = HBM bytes / time. At a FIXED R=2048 (so the C-dependence is isolated), plot every
    # op's effBW vs the row width C, measured (solid) with the LIVE model overlaid (dashed).
    # transpose (block-swapped inside the stick) and cat1 (copies stored outermost) stay flat;
    # cat0 (a strided per-row stick gather) and transpose_outer (a tiled block-transpose, shown
    # at its middle dim M=8) fall as C grows -> more 64-element stick blocks per row to gather.
    # cores=32 only (a memory op's effBW scales with active cores; mixing cores adds scatter).
    from collections import defaultdict

    cm = _cost_model()
    recs = _load(current_only=False)
    RFIX = 2048
    meas = defaultdict(dict)  # meas[op][C]
    model = defaultdict(dict)  # model[op][C] = io/pred (live model)
    tmp = defaultdict(lambda: defaultdict(list))
    for r in recs:
        op = r.get("op")
        if op not in ("transpose", "transpose_outer", "cat0", "cat1"):
            continue
        if r.get("rows") != RFIX or r.get("cores") not in (32, None) or r.get("failed"):
            continue
        if not r.get("io_hbm_bytes") or not r.get("cols") or not r.get("feats"):
            continue
        C, io = r["cols"], int(r["io_hbm_bytes"])
        feats = r["feats"]
        feats = feats if isinstance(feats, list) else json.loads(feats)
        if op == "transpose_outer":  # M=8 only (the modeled case)
            oe = feats[0].get("out_elems", 0)
            if round(oe / (RFIX * C)) != 8:
                continue
        tmp[op][C].append(io / 1e3 / r["kernel_us"])
        if C not in model[op]:
            try:
                pred = cm.predict_ops(cm.ops_from_json(json.dumps(feats))) / 1e3
                model[op][C] = io / 1e3 / pred
            except Exception:  # noqa: BLE001
                pass
    for op in tmp:
        for C, v in tmp[op].items():
            meas[op][C] = sum(v) / len(v)

    styles = {
        "transpose": ("#2ca02c", "o", "transpose (block-swap, flat)"),
        "cat1": ("#1f77b4", "s", "cat1 (copies outermost, flat)"),
        "transpose_outer": ("#9467bd", "D", "transpose_outer M=8 (tiled)"),
        "cat0": ("#d62728", "^", "cat0 (strided gather)"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for op, (c, m, lab) in styles.items():
        mpts = sorted(meas.get(op, {}).items())
        if not mpts:
            continue
        xs, ys = zip(*mpts)
        ax.plot(xs, ys, "-", color=c, marker=m, ms=5, label=lab, zorder=3)
        gpts = sorted(model.get(op, {}).items())
        if gpts:
            gx, gy = zip(*gpts)
            ax.plot(gx, gy, "--", color=c, lw=1.1, alpha=0.75)
    ax.plot([], [], "k--", lw=1.1, label="model")
    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048, 4096, 8192, 16384, 32768])
    ax.set_xticklabels(["512", "1k", "2k", "4k", "8k", "16k", "32k"])
    ax.set_xlabel("row width C  (stick blocks per row = C/64)")
    ax.set_ylabel("effective BW  (R+W)/time  (GB/s)")
    ax.set_title(
        f"§6  Transport effective BW vs C at R={RFIX} (solid=measured, dashed=model)"
    )
    ax.set_ylim(40, 130)
    ax.legend(loc="lower left", fontsize=7.0, framealpha=0.9, ncol=1)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    _save(fig, "fig6_transport")


# ============================================================================
# §8-§11 -- matmul. These need model predictions, so load cost_model the same
# hardware-free way eval_model does (importlib; it imports only math/dataclasses).
# ============================================================================
def _cost_model():
    import importlib.util

    path = os.path.join(
        os.path.dirname(_HERE), "torch_spyre", "_inductor", "cost_model.py"
    )
    spec = importlib.util.spec_from_file_location("cost_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _mm_rows(section_prefix):
    recs = _load(current_only=False)
    out = []
    for r in recs:
        if r.get("op") not in ("mm", "mmwd") or not r.get("feats"):
            continue
        if not (r.get("section") or "").startswith(section_prefix):
            continue
        out.append(r)
    return out


def fig_matmul_hbm(_recs):
    # Accuracy of the single-rate baseline memory model (R+W)/150 + a*min(R,W) on the
    # COMPUTE-FREE matmuls (thin-K write-heavy, thin-M read-heavy). Measured vs predicted,
    # labeled by shape -> shows coverage AND where the baseline strays (read-heavy corner).
    cm = _cost_model()
    p = cm.CostParams()
    rows = _mm_rows("M1")
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    lim = 300
    ax.plot(
        [0, lim], [0, lim], "-", color="0.6", lw=1.0, zorder=1, label="perfect (y = x)"
    )
    ax.plot([0, lim], [0, lim * 1.1], ":", color="0.75", lw=0.8, zorder=1)
    ax.plot(
        [0, lim], [0, lim * 0.9], ":", color="0.75", lw=0.8, zorder=1, label="±10 %"
    )
    for r in rows:
        # Drop the degenerate K=64 write-heavy corner: a single-stick contraction dim
        # (K=64 = one 64-elem stick) runs OFF-trend -- 2048x64x2048 measures 56.6 us,
        # LOWER than K=16 (64.7) and K=32 (69.6) despite more bytes -- so it is not a
        # clean memory-term point. K=16/32 remain as the write-heavy evidence.
        if (r.get("K") or 0) == 64 and (r.get("M") or 0) >= 2048:
            continue
        feats = r["feats"]
        feats = feats if isinstance(feats, list) else json.loads(feats)
        ops = cm.ops_from_json(json.dumps(feats))
        R, W = cm._fused_hbm_bytes(ops)
        base = (
            (R + W) / p.bw_peak_gbps + p.rw_turnaround_ns_per_byte * min(R, W)
        ) / 1e3
        meas, M, N, K = r["kernel_us"], r.get("M"), r.get("N"), r.get("K")
        wheavy = (K or 0) <= 64
        col = "#d62728" if wheavy else "#1f77b4"
        ax.scatter(
            base, meas, color=col, s=44, zorder=3, edgecolors="white", linewidths=0.5
        )
        ax.annotate(
            f"{M}×{K}×{N}",
            (base, meas),
            textcoords="offset points",
            xytext=(5, -1),
            fontsize=6.3,
            color="0.25",
        )
    ax.scatter(
        [], [], color="#d62728", s=44, label="write-heavy  (thin K ∈ {16,32})"
    )
    ax.scatter([], [], color="#1f77b4", s=44, label="read-heavy  (thin M ∈ {32,64})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("predicted µs   —   baseline  (R+W)/150 + α·min(R,W)")
    ax.set_ylabel("measured µs")
    ax.set_title("§8  Baseline memory model vs measured (compute-free matmuls)")
    ax.annotate(
        "within ~4 % on write-heavy;\nread-heavy (large N, thin M)\nunder-predicted ~8–18 %",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=7.5,
        color="0.3",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.8"),
    )
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    _save(fig, "fig8_matmul_hbm")


def fig_matmul_spill(_recs):
    # THE OBSERVATION: with spill OFF, the base model leaves a residual (measured - base)
    # that grows with the per-core OUTPUT-tile AREA (M/m)*(N/n). Two balanced (4x8) decouple
    # sweeps grow the tile: DC2 varies M/m (N/n=256 fixed), DC1 varies N/n (M/m=512 fixed) --
    # both trace the SAME area axis. Residual is ~0/negative below the on-chip capacity knee
    # (~64K elems) and climbs positive above it (the re-read). The spill term (dashed) is one
    # area-driven curve. Every point labeled with its full 2-D per-core tile M/m x N/n.
    cm = _cost_model()
    p0 = cm.CostParams(mm_spill_slope=0.0)  # base model, spill OFF

    def collect(section):
        out = []
        for r in _mm_rows(section):
            feats = r["feats"]
            feats = feats if isinstance(feats, list) else json.loads(feats)
            ops = cm.ops_from_json(json.dumps(feats))
            mm = next(o for o in ops if getattr(o, "is_matmul", False))
            rpc, cpc = mm.matmul_rows_per_core, mm.matmul_cols_per_core
            area = rpc * cpc * 2  # per-core output-tile size in BYTES (fp16) -> x-axis
            base = cm.predict_ops(ops, p0) / 1e3
            resid = r["kernel_us"] - base
            spill = cm.predict_ops(ops) / 1e3 - base  # modeled spill effect
            out.append((area, resid, spill, rpc, cpc))
        return sorted(out)

    dc2 = collect("DC2")  # vary M/m (N/n held at 256)
    dc1 = collect("DC1")  # vary N/n (M/m held at 512)
    knee = cm.CostParams().mm_spill_area0 * 2  # elems -> bytes (fp16)
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.axhline(0, color="0.6", lw=1.0, zorder=1)
    ax.axvline(
        knee,
        ls="--",
        color="0.5",
        lw=1.1,
        zorder=1,
        label="on-chip capacity ≈ 128 KB",
    )
    for data, col, lab in [
        (dc2, "#8c564b", "vary M/m  (N/n = 256 fixed, split 4×8)"),
        (dc1, "#1f77b4", "vary N/n  (M/m = 512 fixed, split 4×8)"),
    ]:
        xs = [d[0] for d in data]
        ax.scatter(
            xs,
            [d[1] for d in data],
            color=col,
            s=48,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
            label=f"residual: {lab}",
        )
        ax.plot(xs, [d[2] for d in data], "--", color=col, lw=1.1, alpha=0.7, zorder=2)
        for area, resid, spill, rpc, cpc in data:
            # label EVERY point with the full 2-D per-core tile  M/m × N/n
            ax.annotate(
                f"{rpc:.0f}×{cpc:.0f}",
                (area, resid),
                textcoords="offset points",
                xytext=(5, 3),
                fontsize=6.2,
                color=col,
            )
    ax.plot([], [], "--", color="0.5", lw=1.1, label="modeled spill (dashed)")
    ax.set_xscale("log", base=2)
    # Label ticks in plain element counts (32K, 64K, ...) not 2^n -- more legible, and the
    # 64K tick lines up with the "on-chip capacity" knee.
    import matplotlib.ticker as _mt

    ticks = [65536, 131072, 262144, 524288, 1048576]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(
        _mt.FuncFormatter(
            lambda v, _: f"{v / 1048576:.0f} MB"
            if v >= 1048576
            else f"{v / 1024:.0f} KB"
        )
    )
    ax.xaxis.set_minor_formatter(_mt.NullFormatter())
    ax.set_xlabel("per-core output-tile size   2·(M/m)·(N/n)   [bytes, fp16]")
    ax.set_ylabel("residual:  measured − base model (no spill)   (µs)")
    ax.set_title(
        "§11  Base-model residual grows once the output tile overflows on-chip"
    )
    ax.annotate(
        "tile fits → residual ≈ 0\ntile overflows → under-predict (re-read)",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=7.2,
        color="0.3",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.8"),
    )
    ax.legend(loc="lower right", fontsize=7.2, framealpha=0.9)
    _save(fig, "fig11_matmul_spill")


def fig_matmul_split(_recs):
    # THE OBSERVATION (§12): with the split term OFF, the base model leaves a residual on
    # LOPSIDED matmul splits (from the forced-split sweep) that grows with the per-core tile
    # SIZE -- but ONLY when the long output dimension is fanned across more than 8 cores.
    # Points are colored by fan_long (m if M>=N else n): fanout<=8 stays flat near 0 at ANY
    # tile size; fanout=16 and 32 climb once the tile passes the ~256 KB size gate. Dashed =
    # the modeled split term, which tracks each climb. k=1, non-tiny rows only.
    cm = _cost_model()
    p0 = cm.CostParams(mm_split_reread_us_per_elem=0.0)  # base model, split term OFF
    a0_bytes = cm.CostParams().mm_split_area0 * 2  # elems -> bytes (fp16)
    groups = {"bal": [], "long16": [], "long32": [], "short": []}
    for r in _load(current_only=False):
        if r.get("op") != "mmwd" or not r.get("feats"):
            continue
        s = r.get("split_forced") or {}
        if s.get("k", 1) != 1:
            continue
        M, N, K = r.get("M"), r.get("N"), r.get("K")
        if None in (M, N, K) or min(M, N) < 512 or K < 256:
            continue
        m, n = s.get("m", 1), s.get("n", 1)
        if (
            m == 1 and n == 1
        ):  # unsplit (cores=1): a cores->BW effect, NOT a split -- exclude
            continue
        fan_long, mx = (m if M >= N else n), max(m, n)
        if mx <= 8:
            g = "bal"
        elif fan_long >= 32:
            g = "long32"
        elif fan_long >= 16:
            g = "long16"
        else:
            g = "short"  # a fanout > 8 but on the SHORT dim -> deliberately unmodeled
        feats = r["feats"]
        feats = feats if isinstance(feats, list) else json.loads(feats)
        ops = cm.ops_from_json(json.dumps(feats))
        mm = next(o for o in ops if getattr(o, "is_matmul", False))
        area_bytes = mm.matmul_rows_per_core * mm.matmul_cols_per_core * 2
        base = cm.predict_ops(ops, p0) / 1e3
        groups[g].append(
            (
                area_bytes,
                r["kernel_us"] - base,
                cm.predict_ops(ops) / 1e3 - base,
                f"{m}×{n}",
                f"{M}×{K}×{N}",
            )
        )

    import matplotlib.ticker as _mt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.axhline(0, color="0.6", lw=1.0, zorder=1)
    ax.axvline(
        a0_bytes, ls="--", color="0.5", lw=1.1, zorder=1, label="size gate  a₀ ≈ 256 KB"
    )
    # balanced: flat at ~0
    bal = sorted(groups["bal"])
    ax.scatter(
        [d[0] for d in bal],
        [d[1] for d in bal],
        color="#7f7f7f",
        s=30,
        zorder=3,
        edgecolors="white",
        linewidths=0.4,
        label="residual: fanout ≤ 8 (balanced) ≈ 0",
    )

    def _annotate(data, col):
        # tag each fanned point with its split m×n (the config that drives the climb);
        # alternate the vertical offset so neighbours at the same tile size don't collide.
        for i, d in enumerate(data):
            ax.annotate(
                d[3],
                (d[0], d[1]),
                textcoords="offset points",
                xytext=(0, 6 if i % 2 == 0 else -11),
                ha="center",
                fontsize=5.6,
                color=col,
                zorder=4,
            )

    # long-dim fanned: climb, tracked by the term (dashed)
    for g, col, lab in [
        ("long16", "#ff7f0e", "long-dim fanout = 16"),
        ("long32", "#d62728", "long-dim fanout = 32"),
    ]:
        data = sorted(groups[g])
        if not data:
            continue
        xs = [d[0] for d in data]
        ax.scatter(
            xs,
            [d[1] for d in data],
            color=col,
            s=36,
            zorder=3,
            edgecolors="white",
            linewidths=0.4,
            label=f"residual: {lab}",
        )
        ax.plot(xs, [d[2] for d in data], "--", color=col, lw=1.2, alpha=0.8, zorder=2)
        _annotate(data, col)
    # short-dim split: the second term (lighter, higher knee) tracks these too
    sh = sorted(groups["short"])
    if sh:
        ax.scatter(
            [d[0] for d in sh],
            [d[1] for d in sh],
            color="#8c564b",
            s=36,
            marker="^",
            zorder=3,
            edgecolors="white",
            linewidths=0.4,
            label="residual: short-dim split",
        )
        ax.plot(
            [d[0] for d in sh],
            [d[2] for d in sh],
            "--",
            color="#8c564b",
            lw=1.2,
            alpha=0.8,
            zorder=2,
        )
        _annotate(sh, "#8c564b")
    ax.plot([], [], "--", color="0.5", lw=1.2, label="modeled split term (both sides)")
    ax.set_xscale("log", base=2)
    ticks = [65536, 131072, 262144, 524288, 1048576]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(
        _mt.FuncFormatter(
            lambda v, _: f"{v / 1048576:.0f} MB"
            if v >= 1048576
            else f"{v / 1024:.0f} KB"
        )
    )
    ax.xaxis.set_minor_formatter(_mt.NullFormatter())
    ax.set_xlabel("per-core output-tile size   2·(M/m)·(N/n)   [bytes, fp16]")
    ax.set_ylabel("residual:  measured − base model (no split term)   (µs)")
    ax.set_title("§12  Lopsided-split residual: a large tile fanned across many cores")
    ax.annotate(
        "balanced (fanout ≤ 8): flat at ~0 for any tile\n"
        "fanout > 8: climbs once the tile passes the gate",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=7.2,
        color="0.3",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.8"),
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        fontsize=7.0,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.3,
    )
    _save(fig, "fig12_matmul_split")


def fig_matmul_bmm(_recs):
    # §13: measured/predicted vs B for a balanced (4x8) batched matmul. A plain 2-D matmul
    # (a single batch) is accurate (~1.0, the anchor at B=1). The FULL bmm (a distinct weight
    # per batch, solid) plateaus at 2-4.6x; the SHARED-weight projection [B,M,K]@[K,N] (dashed)
    # plateaus far lower. Both flat in B. Only shapes with a full B-sweep are shown.
    import collections

    lines = collections.defaultdict(dict)  # (op, shape) -> {B: ratio}
    mm_anchor = {}  # shape -> ratio of the plain 2-D matmul (the "1 batch" reference)
    for r in _load(current_only=False):
        s = r.get("split_forced") or {}
        if s.get("m") != 4 or s.get("n") != 8 or not r.get("kernel_us"):
            continue
        sh = f"{r['M']}×{r['K']}×{r['N']}"
        if r["op"] == "mmwd":
            mm_anchor[sh] = r["kernel_us"] / r["pred_us"]
        elif (
            str(r.get("log_file", "")).startswith("overnight_")
            and r["op"] in ("bmm_wd", "bmm_wd_3d2d")
            and s.get("b") == 1
            and r.get("B", 1) >= 2
        ):
            lines[(r["op"], sh)][r["B"]] = r["kernel_us"] / r["pred_us"]
    # keep only shapes with a real sweep (>=3 B points) for BOTH full and 3d2d
    shapes = sorted(
        {
            sh
            for (op, sh), d in lines.items()
            if len(d) >= 3
            and ("bmm_wd", sh) in lines
            and len(lines.get(("bmm_wd_3d2d", sh), {})) >= 3
        }
    )
    colors = {sh: c for sh, c in zip(shapes, ["#1f77b4", "#d62728", "#2ca02c"])}

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.axhline(1.0, color="0.6", lw=1.0, zorder=1)
    for sh in shapes:
        col = colors[sh]
        for op, style, lab in [
            ("bmm_wd", "-o", "full bmm"),
            ("bmm_wd_3d2d", "--s", "shared-weight (3d2d)"),
        ]:
            d = lines.get((op, sh), {})
            xs = sorted(d)
            ax.plot(
                xs,
                [d[b] for b in xs],
                style,
                color=col,
                lw=1.8 if op == "bmm_wd" else 1.2,
                ms=5,
                markerfacecolor=col if op == "bmm_wd" else "white",
                label=f"{lab}  {sh}",
            )
            # tag each point with its multiplier; full above the marker, 3d2d below
            up = op == "bmm_wd"
            for b in xs:
                ax.annotate(
                    f"{d[b]:.1f}×",
                    (b, d[b]),
                    textcoords="offset points",
                    xytext=(0, 6 if up else -12),
                    ha="center",
                    fontsize=5.8,
                    color=col,
                    zorder=5,
                )
        if sh in mm_anchor:  # the plain 2-D matmul, plotted at B=1
            ax.plot(
                [1],
                [mm_anchor[sh]],
                "*",
                color=col,
                ms=13,
                zorder=4,
                markeredgecolor="0.3",
                markeredgewidth=0.5,
            )
    ax.plot(
        [],
        [],
        "*",
        color="0.4",
        ms=12,
        markeredgecolor="0.3",
        label="plain 2-D matmul (1 batch) ≈ 1",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("batch  B   (batches run serially on the same cores)")
    ax.set_ylabel("measured / predicted   (base model = B × one matmul)")
    ax.set_title("§13  Batched matmul runs 2–4.6× the predicted cost, flat in B")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        fontsize=7.2,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    _save(fig, "fig13_matmul_bmm")


def fig_matmul_bmm_control(_recs):
    # §13: the shared-weight control. Full bmm and the shared-weight projection have IDENTICAL
    # MACs (same x), but the full bmm's per-batch excess (kernel-pred)/B is ~10x the shared
    # one -- so the miss is NOT compute-rate (equal MACs would give equal excess); it is the
    # per-batch weight re-read (memory). Matched (shape, B), balanced 4x8, b=1.
    full, d2d = {}, {}
    for r in _load(current_only=False):
        if not str(r.get("log_file", "")).startswith("overnight_") or not r.get(
            "kernel_us"
        ):
            continue
        s = r.get("split_forced") or {}
        if s.get("b") != 1 or s.get("m") != 4 or s.get("n") != 8 or r.get("B", 1) < 2:
            continue
        key = (r["M"], r["K"], r["N"], r["B"])
        macs = r["M"] * r["K"] * r["N"] * r["B"]
        exc = (r["kernel_us"] - r["pred_us"]) / r["B"]  # per-batch excess
        (full if r["op"] == "bmm_wd" else d2d)[key] = (macs, exc)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for d, col, mk, lab in [
        (full, "#d62728", "o", "full bmm (distinct weight per batch)"),
        (d2d, "#1f77b4", "s", "shared-weight (weight read once)"),
    ]:
        pts = sorted(d.values())
        ax.scatter(
            [p[0] / 1e9 for p in pts],
            [p[1] for p in pts],
            color=col,
            marker=mk,
            s=42,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
            label=lab,
        )
    # connect matched pairs to show equal-MACs, unequal-excess
    for k in sorted(set(full) & set(d2d)):
        ax.plot(
            [full[k][0] / 1e9] * 2,
            [full[k][1], d2d[k][1]],
            color="0.7",
            lw=0.8,
            zorder=1,
        )
    ax.axhline(0, color="0.6", lw=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(
        "MACs per batched matmul  (G)   — identical for the two ops at each pair"
    )
    ax.set_ylabel("per-batch excess  (measured − predicted)/B   (µs)")
    ax.set_title(
        "§13  Equal MACs, very different excess → not compute, it's the weight re-read"
    )
    ax.annotate(
        "same MACs (vertical line), but the full bmm's\n"
        "per-batch excess is ~10× the shared-weight one",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=7.4,
        color="0.3",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.8"),
    )
    ax.legend(
        loc="upper left", bbox_to_anchor=(0.02, 0.80), fontsize=7.6, framealpha=0.9
    )
    _save(fig, "fig13b_matmul_bmm_control")


def fig_matmul_bmm_layout(_recs):
    # §13: the DEVICE TILE-ORDER (layout) is the bmm penalty. The SAME
    # [4,1024,2048] @ [4,2048,1024] bmm under all 4 operand dim_order combos, at IDENTICAL
    # bytes (41.9 MB, IR-confirmed no inserted copy) -- time spans 3.3x purely from dataflow.
    import regex as re

    ir = os.path.join(os.path.dirname(_HERE), "haoyang_logs", "ir")
    combos = [("0,1,2", "0,1,2"), ("0,1,2", "1,0,2"), ("1,0,2", "0,1,2"), ("1,0,2", "1,0,2")]
    data, io = [], 0
    for la, lb in combos:
        fn = os.path.join(
            ir,
            f"bmm_layout_B4_1024x2048x1024_A{la.replace(',', '')}_B{lb.replace(',', '')}.txt",
        )
        txt = open(fn, encoding="utf-8").read()
        us = float(re.search(r"kernel_us=([\d.]+)", txt)[1])
        io = int(re.search(r"io_hbm_bytes=(\d+)", txt)[1])
        data.append((la, lb, us))
    best = min(d[2] for d in data)
    worst = max(d[2] for d in data)
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    xs = list(range(len(data)))
    cols = [
        "#d62728" if us == worst else "#2ca02c" if us == best else "#8aa9cf"
        for _la, _lb, us in data
    ]
    ax.bar(xs, [d[2] for d in data], color=cols, edgecolor="white", zorder=3, width=0.62)
    for i, (_la, _lb, us) in enumerate(data):
        ax.annotate(
            f"{us:.0f} µs\n{io / us / 1e3:.0f} GB/s\n{us / best:.2f}×",
            (i, us),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    tags = {0: "\n(compiler default)", 3: "\n(best)"}
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"A [{la}]\nB [{lb}]{tags.get(i, '')}" for i, (la, lb, _u) in enumerate(data)],
        fontsize=7.4,
    )
    ax.set_ylabel("kernel time  (µs)")
    ax.set_ylim(0, worst * 1.20)
    ax.set_title(
        "§13  bmm: same bytes (41.9 MB, IR-confirmed no inserted copy),\n"
        "3.3× from the device tile-order alone  ([4,1024,2048] @ [4,2048,1024])"
    )
    ax.grid(axis="y", alpha=0.3, zorder=0)
    _save(fig, "fig13c_matmul_bmm_layout")


def _mm_balanced_points(K_set=(2048, 4096), M=2048, N=2048):
    """All (cores, m, n, k, t) for M×N×K matmuls, deduped, from every sweep."""
    recs = _load(current_only=False)
    out = {K: [] for K in K_set}
    for r in recs:
        if r.get("op") not in ("mm", "mmwd") or r.get("M") != M or r.get("N") != N:
            continue
        K, sa, c = r.get("K"), r.get("split_actual") or {}, r.get("cores")
        if K not in out or not c:
            continue
        out[K].append((c, sa.get("m"), sa.get("n"), sa.get("k", 1), r["kernel_us"]))
    return out


def fig_matmul_peak(_recs):
    # Compute-dominant: kernel time is linear in 1/cores at a fixed matmul; slope = 1/peak.
    # Two problem sizes (K=2048, K=4096 at M=N=2048). BALANCED (k=1) splits at equal cores
    # COLLAPSE (time tracks cores, not the m*n factoring). Every point labeled with m*n.
    pts = _mm_balanced_points()
    fig, ax = plt.subplots(figsize=(6.4, 4.7))
    colors = {2048: "#1f77b4", 4096: "#2ca02c"}

    def _combos32(cls):  # list the 32-core m×n splits that belong to this marker class
        return ", ".join(
            f"{m}×{n}" for m, n in sorted({(m, n) for c, m, n, t in cls if c == 32})
        )

    for K in (4096, 2048):
        # Average only EXACT replicates of the same (cores, m, n) split -- never merge
        # different splits, so each split shows at its own (1/cores, t). "Balanced" = the
        # most-square split achievable at each core count (max min(m,n)); everything
        # thinner is drawn with a distinct marker so 2x16 / 1x32 / 32x1 are visibly OFF
        # the line rather than crushed into one blob.
        agg: dict = {}
        for c, m, n, k, t in pts[K]:
            if k != 1 or not (m and n):
                continue
            agg.setdefault((c, m, n), []).append(t)
        k1 = sorted((c, m, n, sum(v) / len(v)) for (c, m, n), v in agg.items())
        best_min = {}
        for c, m, n, t in k1:
            best_min[c] = max(best_min.get(c, 0), min(m, n))
        bal = [(c, m, n, t) for c, m, n, t in k1 if min(m, n) == best_min[c]]
        thin = [(c, m, n, t) for c, m, n, t in k1 if min(m, n) < best_min[c]]
        # peak line: fit on BALANCED splits only (lopsided 1x32/32x1 would inflate it).
        xb = [1.0 / c for c, *_ in bal]
        yb = [t for *_, t in bal]
        a, b = np.polyfit(xb, yb, 1)
        ax.plot(
            np.linspace(0, 0.27, 20),
            a * np.linspace(0, 0.27, 20) + b,
            "-",
            color=colors[K],
            lw=1.0,
            alpha=0.6,
            zorder=1,
        )
        # Drop fully-lopsided min(m,n)=1 splits at cores<32 (1×4, 1×8, … — degenerate, ~2×
        # off, they only blow the y-axis); keep 1×32 / 32×1 AT 32 cores (documented ~2×).
        thin_plot = [p for p in thin if not (min(p[1], p[2]) == 1 and p[0] != 32)]
        ax.scatter(
            xb,
            yb,
            color=colors[K],
            s=42,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
            label=f"K={K}:  {_combos32(bal)}",
        )
        if thin_plot:
            ax.scatter(
                [1.0 / c for c, *_ in thin_plot],
                [t for *_, t in thin_plot],
                color=colors[K],
                s=48,
                zorder=4,
                marker="x",
                linewidths=1.3,
                label=f"K={K}:  {_combos32(thin)}",
            )
        # label only the lopsided 1xN / Nx1 splits in the main plot (they sit ~2x off
        # the line); the min(m,n) in {2,4} splits are labelled in the zoomed inset below.
        for c, m, n, t in [p for p in k1 if p[0] == 32 and min(p[1], p[2]) == 1]:
            ax.annotate(
                f"{m}×{n}",
                (1.0 / c, t),
                textcoords="offset points",
                xytext=(6, -2),
                fontsize=6.0,
                color=colors[K],
            )
    # Zoomed inset on the 32-core cluster: the full 0-4000 us range crushes the split
    # spread, so enlarge min(m,n) in {2,4} (the ~2x lopsided 1x32/32x1 sit off the top).
    # This is where the balanced (4x8/8x4) collapse vs the thinner 2x16 (~+5%) is visible.
    axin = ax.inset_axes([0.07, 0.55, 0.40, 0.40])
    for K in (4096, 2048):
        pk: dict = {}
        for c, m, n, k, t in pts[K]:
            if k == 1 and m and n and c == 32:
                pk.setdefault((m, n), []).append(t)
        for (m, n), v in pk.items():
            t = sum(v) / len(v)
            axin.scatter(
                [min(m, n)],
                [t],
                color=colors[K],
                s=30,
                zorder=3,
                marker=("o" if min(m, n) >= 4 else "x"),
                linewidths=1.1,
            )
            axin.annotate(
                f"{m}×{n}",
                (min(m, n), t),
                textcoords="offset points",
                xytext=(5, -1),
                fontsize=5.4,
                color=colors[K],
            )
    axin.set_xscale("log", base=2)
    axin.set_xlim(1.6, 6.5)
    axin.set_ylim(360, 760)
    axin.set_xticks([2, 4])
    axin.set_xticklabels(["2", "4"], fontsize=6)
    axin.set_xlabel("min(m,n) of the 32-core split", fontsize=6)
    axin.tick_params(labelsize=5.6)
    axin.set_title("32-core split spread (zoom)", fontsize=6.5)
    ax.set_xlim(0, 0.27)
    ax.set_ylim(0, 4600)  # focus on cores 4-32; the off-axis 2-core points (x=1/2) would
    # otherwise stretch y to ~8000 and flatten the split spread.
    ax.set_xticks([1 / 4, 1 / 8, 1 / 16, 1 / 32])
    ax.set_xticklabels(["4", "8", "16", "32"])
    ax.set_xlabel(
        "cores used   (axis positioned at 1/cores → straight line = time ∝ 1/cores)"
    )
    ax.set_ylabel("kernel time  (µs)")
    ax.set_title(
        "§9  Time ∝ 1/cores (M=N=2048); at equal cores the split collapses\n"
        "only when neither factor is too thin (legend lists the 32-core splits)"
    )
    ax.legend(loc="lower right", fontsize=6.8, framealpha=0.9)
    _save(fig, "fig9_matmul_peak")


def fig_matmul_overlap(_recs):
    # §10: does the overlap term generalize? For EVERY 32-core matmul run (all regimes),
    # plot prediction error vs the MEMORY FRACTION memory/(compute+memory). Each run gets
    # TWO markers joined by a line: the ADDITIVE model (gamma=0, open) and the OVERLAP model
    # (gamma=0.46, filled). The additive error grows with memory fraction (up to +40%);
    # the overlap term pulls every point back toward 0. Color = regime.
    import regex as re

    cm = _cost_model()
    p = cm.CostParams()
    p_add = cm.CostParams(overlap_gamma=0.0)

    def sp(lab):
        m = re.search(r"\bm=(\d+)\s+n=(\d+)\s+k=(\d+)", lab)
        return (int(m[1]), int(m[2]), int(m[3])) if m else None

    def opsp(feats, s):
        ops = cm.ops_from_json(json.dumps(feats))
        for o in ops:
            if getattr(o, "is_matmul", False):
                if not o.matmul_m_split or o.matmul_m_split == 1:
                    o.matmul_m_split = s[0]
                if not o.matmul_n_split or o.matmul_n_split == 1:
                    o.matmul_n_split = s[1]
        return ops

    def regime(M, K, N, m, n):
        # four regimes: a lopsided WORK DIVISION (fanout > 8); a TENSOR-SIZE extreme (thin
        # reduction, K ≤ 128 -- memory-heavy, compute-starved); a TINY kernel (small output,
        # min(M,N) ≤ 512); else realistic. Checked in this order, so a config matching more
        # than one is assigned to the first (work-division dominates, then tiny output).
        if max(m, n) > 8:
            return "workdiv"
        if min(M, N) <= 512:
            return "tiny"
        if K <= 128:
            return "tensor"
        return "realistic"

    seen, rows = set(), []
    for r in _load(current_only=False):
        if r.get("op") != "mmwd" or not r.get("feats"):
            continue
        s = sp(r.get("label", ""))
        M, N, K = r.get("M"), r.get("N"), r.get("K")
        # keep only the cases that use all 32 cores with no K-split (m·n = cores = 32, k = 1)
        if (
            not s
            or None in (M, N, K)
            or s[2] != 1
            or s[0] * s[1] != 32
            or r.get("cores") != 32
        ):
            continue
        # keep only power-of-2 tensor dimensions
        if any(d & (d - 1) for d in (M, N, K)):
            continue
        key = (M, N, K, s[0], s[1])
        if key in seen:
            continue
        seen.add(key)
        feats = r["feats"]
        feats = feats if isinstance(feats, list) else json.loads(feats)
        mm = next(o for o in opsp(feats, s) if getattr(o, "is_matmul", False))
        t_add = cm.predict_ops(opsp(feats, s), p_add) / 1e3
        t_ov = cm.predict_ops(opsp(feats, s), p) / 1e3
        meas = r["kernel_us"]
        compute = mm.matmul_macs / mm.cores / 1140 / 1e3  # µs
        frac = max(0.0, min(1.0, (t_add - compute) / t_add)) if t_add > 0 else 0.0
        rows.append(
            (
                frac,
                100 * (t_add - meas) / meas,
                100 * (t_ov - meas) / meas,
                regime(M, K, N, s[0], s[1]),
                M,
                K,
                N,
                s,
            )
        )

    colors = {
        "realistic": "#2ca02c",
        "workdiv": "#ff7f0e",
        "tensor": "#1f77b4",
        "tiny": "#7f7f7f",
    }
    labels = {
        "realistic": "realistic (balanced split, normal shape)",
        "workdiv": "work-division extreme (fanout > 8)",
        "tensor": "tensor-size extreme (thin reduction, K ≤ 128)",
        "tiny": "tiny (small output, min(M,N) ≤ 512)",
    }
    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    ax.axhspan(-10, 10, color="0.92", zorder=0, label="±10 %")
    ax.axhline(0, color="0.6", lw=1.0, zorder=1)
    for frac, ea, eo, reg, M, K, N, s in rows:
        col = colors[reg]
        ax.plot([frac, frac], [ea, eo], "-", color="0.82", lw=0.7, zorder=1)
        ax.scatter(frac, ea, facecolors="none", edgecolors=col, s=60, lw=1.2, zorder=2)
        ax.scatter(
            frac, eo, color=col, s=60, zorder=3, edgecolors="white", linewidths=0.4
        )
    # label the residual outliers (|overlap err| > 15%) with BOTH the tensor size M×K×N and
    # the work-division split m×n×k
    neg = 0
    for frac, ea, eo, reg, M, K, N, s in rows:
        if abs(eo) <= 15:
            continue
        up = eo > 0
        lab = f"{M}×{K}×{N}\n{s[0]}×{s[1]}×{s[2]}"
        ax.annotate(
            lab,
            xy=(frac, eo),
            textcoords="offset points",
            xytext=(7, 4) if up else (7, -11 - 22 * (neg % 2)),
            fontsize=7.0,
            color=colors[reg],
            zorder=5,
            linespacing=0.9,
        )
        if not up:
            neg += 1
    # legends: model (open/filled) + regime (color)
    ax.scatter(
        [], [], facecolors="none", edgecolors="0.35", s=60, label="additive  γ=0 (open)"
    )
    ax.scatter([], [], color="0.35", s=60, label="overlap  γ=0.46 (filled)")
    for reg, col in colors.items():
        n = sum(1 for r in rows if r[3] == reg)
        ax.scatter([], [], color=col, s=60, label=f"{labels[reg]}  (n={n})")
    ax.set_xlabel("memory fraction   memory / (compute + memory)", fontsize=11)
    ax.set_ylabel("prediction error  (%)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.set_title(
        "§10  Overlap (γ=0.46, filled) lowers every prediction, closing the additive\n"
        "model's over-prediction — which grows with the memory fraction",
        fontsize=12,
    )
    ax.set_ylim(-35, 60)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=3,
        fontsize=9.5,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.4,
    )
    _save(fig, "fig10_matmul_overlap", dpi=200)


# ============================================================================
# §14 -- coarse tiling LX-spill: the spilled bytes ARE counted (HBM), but the
# EFFECTIVE BANDWIDTH falls once the per-core working set overflows LX (~512 KB).
# Measured effBW = (R+W)/time per config; the model derates BW past the knee.
# ============================================================================
def fig_coarse_spill(_recs):
    cm = _cost_model()
    recs = _load(current_only=False)
    palette = {
        (16384, 4096): "#d62728",
        (16384, 2048): "#1f77b4",
        (8192, 2048): "#2ca02c",
        (4096, 4096): "#ff7f0e",
        (4096, 2048): "#8c564b",
        (2048, 2048): "#9467bd",
    }
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.axvspan(0.5, 100, color="#f2dede", alpha=0.5, zorder=0)
    ax.axvline(
        0.5, ls="--", color="0.4", lw=1.2, zorder=1, label="usable LX ≈ 512 KB/core"
    )
    seen = set()
    for r in recs:
        if r.get("op") != "softmax_row_tiling" or not r.get("feats"):
            continue
        R, C, t = r.get("rows"), r.get("cols"), r.get("tiles")
        if not (R and C and t) or t < 2:  # tiles=1 untiled = HBM by design, not a spill
            continue
        f = r["feats"]
        f = f if isinstance(f, list) else json.loads(f)
        ops = cm.ops_from_json(json.dumps(f))
        Rb, Wb = cm._fused_hbm_bytes(ops)
        effbw = (Rb + Wb) / 1e3 / r["kernel_us"]  # GB/s
        ws = 2 * (R / t / 32) * C * 2 / 1e6  # MB/core
        col = palette.get((R, C), "0.4")
        ax.scatter(
            ws, effbw, s=42, color=col, zorder=3, edgecolors="white", linewidths=0.4
        )
        if (R, C) not in seen:
            ax.scatter([], [], color=col, s=42, label=f"softmax {R}×{C}")
            seen.add((R, C))
    # model: effBW = balanced-softmax peak (~100) * spill derate past the knee
    wsx = np.logspace(np.log2(0.06), np.log2(9), 60, base=2)
    peak = 100.0
    model = [peak * min(1.0, (0.5 / w) ** 0.15) if w > 0.5 else peak for w in wsx]
    ax.plot(
        wsx,
        model,
        "-",
        color="0.35",
        lw=1.5,
        label="model: 100·min(1,(0.5/ws)$^{0.15}$)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("per-core working set  (MB;  ≈ 2 · rows/core · COLS · 2 B)")
    ax.set_ylabel("effective BW  (R+W)/time   (GB/s)")
    ax.set_title(
        "§14  Spilled bytes are counted (HBM); the effective BW just falls past LX"
    )
    ax.set_ylim(40, 115)
    ax.legend(loc="lower left", fontsize=7.2, framealpha=0.9)
    _save(fig, "fig12_coarse_spill")


# ============================================================================
# §15 -- coarse underfill `eff`: a short per-core tile (rpc rows) never fills the
# streaming pipeline. Softmax effective BW climbs with rpc to a plateau; the model
# eff = min(0.95, (rpc/13)^0.68) (calibrated) captures the rise. Above rpc~32 a
# mild decline is unmodeled.
# ============================================================================
def fig_coarse_eff(_recs):
    # One marker per CONFIG (not averaged): color = ROWS×COLS shape, label = tile count.
    # x = per-core tile height h = ROWS/(cores·tiles); LX-fitting points only.
    from collections import defaultdict

    recs = _load(current_only=False)
    pts = defaultdict(list)  # (R,C) -> [(h, effBW, tiles)]
    for r in recs:
        if (
            r.get("op") != "softmax_row_tiling"
            or not r.get("io_hbm_bytes")
            or not r.get("tiles")
        ):
            continue
        R, C, t = r.get("rows"), r.get("cols"), r.get("tiles")
        if not (R and C and t) or t < 2:
            continue
        h = R / t / 32
        ws = 2 * h * C * 2 / 1e6
        if ws > 1.2:  # LX-fitting only (isolate underfill from §14 spill)
            continue
        pts[(R, C)].append((h, int(r["io_hbm_bytes"]) / 1e3 / r["kernel_us"], t))
    plateau = max(e for v in pts.values() for _, e, _ in v)  # filled-pipeline effBW
    palette = {
        (16384, 4096): "#d62728",
        (16384, 2048): "#1f77b4",
        (8192, 2048): "#2ca02c",
        (8192, 4096): "#9467bd",
        (4096, 4096): "#ff7f0e",
        (4096, 2048): "#8c564b",
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for (R, C), v in sorted(pts.items()):
        col = palette.get((R, C), "0.4")
        for h, eff, t in sorted(v):
            ax.scatter(
                h, eff, s=46, color=col, zorder=3, edgecolors="white", linewidths=0.5
            )
            ax.annotate(
                f"{t}t",  # tile count identifies the config within a shape
                (h, eff),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6.2,
                color=col,
            )
        ax.scatter([], [], color=col, s=46, label=f"{R}×{C}  (ROWS×COLS)")
    rr = np.logspace(np.log2(1.5), np.log2(160), 60, base=2)
    model = plateau * np.minimum(0.95, (rr / 13) ** 0.68)
    ax.plot(
        rr,
        model,
        "-",
        color="0.4",
        lw=1.5,
        label="model: BW·min(0.95, (height/13)$^{0.68}$)",
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel(
        "per-core tile height  =  ROWS / (cores × tiles)   [rows]   (label = tile count)"
    )
    ax.set_ylabel("effective BW  (R+W)/time  (GB/s)")
    ax.set_title(
        "§15  Underfill `eff`: BW climbs with the per-core tile, then plateaus"
    )
    ax.legend(loc="lower right", fontsize=7.0, framealpha=0.9)
    _save(fig, "fig13_coarse_eff")


def fig_coarse_designspace(_recs):
    # §17: tile count is a design-space knob. For a FIXED problem the model tracks the
    # measured cost vs tile count (relative), so it picks a near-optimal tile size. Left:
    # measured (solid) vs predicted (dashed) normalized to each problem's best, for a few
    # softmax shapes. Right: the model-chosen tile's measured cost / the true best (regret).
    import collections

    import regex as re

    cm = _cost_model()
    p = cm.CostParams()

    def shp(r):
        m = re.search(r"\[(\d+),(\d+)\]", r.get("label", ""))
        if m:
            return f"{m[1]}×{m[2]}"
        m = re.search(r"M=(\d+)\s+K=(\d+)\s+N=(\d+)", r.get("label", ""))
        return f"{m[1]}×{m[2]}×{m[3]}" if m else None

    grp = collections.defaultdict(dict)  # (op, shape) -> {tiles: (meas, pred)}
    for r in _load(current_only=False):
        op = r.get("op")
        if op not in ("softmax_row_tiling", "matmul_k_tiling") or not r.get("feats"):
            continue
        s, t = shp(r), r.get("tiles")
        if s is None or t is None:
            continue
        pred = cm.predict_ops(cm.ops_from_json(json.dumps(r["feats"])), p) / 1e3
        cur = grp[(op, s)].get(t)
        grp[(op, s)][t] = (
            (r["kernel_us"], pred)
            if cur is None
            else ((cur[0] + r["kernel_us"]) / 2, (cur[1] + pred) / 2)
        )

    # left panel: a few softmax shapes with a real optimum in the sweep
    show = [
        ("softmax_row_tiling", "8192×2048"),
        ("softmax_row_tiling", "16384×2048"),
        ("softmax_row_tiling", "16384×4096"),
        ("softmax_row_tiling", "2048×2048"),
    ]
    cols = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(9.4, 3.9), gridspec_kw=dict(width_ratios=[1.5, 1])
    )
    for key, col in zip(show, cols):
        d = grp.get(key)
        if not d:
            continue
        tiles = sorted(d)
        mn = min(d[t][0] for t in tiles)
        pmn = min(d[t][1] for t in tiles)
        axL.plot(
            tiles,
            [d[t][0] / mn for t in tiles],
            "-o",
            color=col,
            ms=4,
            label=f"{key[1]}  measured",
        )
        axL.plot(
            tiles,
            [d[t][1] / pmn for t in tiles],
            "--s",
            color=col,
            ms=4,
            markerfacecolor="white",
            lw=1.1,
        )
    axL.axhline(1.0, color="0.7", lw=0.8, zorder=0)
    axL.set_xscale("log", base=2)
    axL.set_xticks([1, 2, 4, 8, 16, 32])
    axL.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    axL.set_xlabel("tile count  (design-space knob)")
    axL.set_ylabel("cost / that problem's best  (relative)")
    axL.set_title(
        "Model tracks the cost-vs-tiles curve\n(solid = measured, dashed = predicted)"
    )
    axL.legend(fontsize=6.6, framealpha=0.9)

    # right panel: selection regret = measured(model's pick) / measured(true best)
    regrets = []
    for (op, s), d in grp.items():
        if len(d) < 3:
            continue
        tiles = sorted(d)
        bt = min(tiles, key=lambda t: d[t][0])
        pt = min(tiles, key=lambda t: d[t][1])
        regrets.append((f"{op.split('_')[0]} {s}", d[pt][0] / d[bt][0]))
    regrets.sort(key=lambda x: x[1])
    ys = range(len(regrets))
    axR.barh(list(ys), [(r - 1) * 100 for _, r in regrets], color="#4c72b0")
    axR.set_yticks(list(ys))
    axR.set_yticklabels([n for n, _ in regrets], fontsize=6.0)
    axR.set_xlabel("regret of model-picked tile  (% over optimal)")
    axR.set_title(
        "Picking the model's best tile size\ncosts ≤ ~9 % over the true optimum"
    )
    axR.axvline(0, color="0.6", lw=0.8)
    fig.tight_layout()
    _save(fig, "fig17_coarse_designspace")


_FIGS = {
    "matmul_hbm": fig_matmul_hbm,
    "matmul_spill": fig_matmul_spill,
    "matmul_split": fig_matmul_split,
    "matmul_bmm": fig_matmul_bmm,
    "matmul_bmm_control": fig_matmul_bmm_control,
    "matmul_bmm_layout": fig_matmul_bmm_layout,
    "coarse_spill": fig_coarse_spill,
    "coarse_eff": fig_coarse_eff,
    "coarse_designspace": fig_coarse_designspace,
    "matmul_peak": fig_matmul_peak,
    "matmul_overlap": fig_matmul_overlap,
    "pointwise_baseline": fig_pointwise_baseline,
    "pointwise_vcurve": fig_pointwise_vcurve,
    "pointwise_arity": fig_pointwise_arity,
    "pointwise_arity_reads": fig_pointwise_arity_reads,
    "broadcast_effbw": fig_broadcast_effbw,
    "write_spill": fig_write_spill,
    "broadcast_smallr": fig_broadcast_smallr,
    "reduction": fig_reduction,
    "transport": fig_transport,
}


def main():
    import sys

    recs = _load()
    want = sys.argv[1:] or list(_FIGS)
    for name in want:
        if name not in _FIGS:
            raise SystemExit(f"unknown figure {name!r} (have: {', '.join(_FIGS)})")
        print(f"figure {name}:")
        _FIGS[name](recs)


if __name__ == "__main__":
    main()
