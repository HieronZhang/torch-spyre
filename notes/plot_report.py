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


def _save(fig, name):
    os.makedirs(_FIGDIR, exist_ok=True)
    path = os.path.join(_FIGDIR, f"{name}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
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


def fig_pointwise_vcurve(recs):
    bw_peak, alpha = 150.0, 0.00574  # current model
    reds = ("sumrow", "read", "amax", "mean")  # read-only anchor (ROWS=2048, clean)
    # label -> (points, color, marker). Each op is swept at several sizes -> several
    # points per class (small vertical spread = mild size drift, ~2-3%).
    groups = {
        "read-only": ([], "#2ca02c", "o"),
        "2R:1W (add)": ([], "#9467bd", "o"),
        "1R:1W (neg)": ([], "#1f77b4", "o"),
        "write-only": ([], "#d62728", "o"),
        "n-ary (→§3)": ([], "#ff7f0e", "x"),
    }
    for r in recs:
        m = r.get("model") or {}
        R, W = m.get("R"), m.get("W")
        if not R or not W:
            continue
        eff = (R + W) / 1e3 / r["kernel_us"]
        w = W / (R + W)
        if r["op"] in reds and r.get("rows") == 2048:
            groups["read-only"][0].append((w, eff))
        elif r["op"] == "add":
            groups["2R:1W (add)"][0].append((w, eff))
        elif r["op"] == "neg":
            groups["1R:1W (neg)"][0].append((w, eff))
        elif r["op"] in ("add3", "add4"):
            groups["n-ary (→§3)"][0].append((w, eff))
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
def fig_pointwise_arity(recs):
    bw_peak, alpha, derate = 150.0, 0.00574, 0.075
    ops = [("add", 1), ("add3", 2), ("add4", 3)]  # n = number of chained binary adds
    meas = {n: [] for _, n in ops}
    for r in recs:
        for op, n in ops:
            if r["op"] == op:
                m = r.get("model") or {}
                R, W = m.get("R"), m.get("W")
                if R and W:
                    meas[n].append((R + W) / 1e3 / r["kernel_us"])
    # byte model: same f=1/3 for all arities -> flat effBW
    eff_byte = 1.0 / (1.0 / bw_peak + alpha * (1.0 / 3.0))
    ns = [n for _, n in ops]

    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.axhline(
        eff_byte,
        ls="--",
        color="0.6",
        lw=1.2,
        label=f"byte model, no derate (flat = {eff_byte:.0f})",
    )
    ax.plot(
        ns,
        [eff_byte / (1 + derate * (n - 1)) for n in ns],
        "-",
        color="#d62728",
        lw=1.3,
        marker="s",
        ms=5,
        zorder=2,
        label=f"× (1 + {derate}·(n−1)) derate",
    )
    for n in ns:
        ax.scatter(
            [n] * len(meas[n]),
            meas[n],
            s=40,
            color="#1f77b4",
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
            label="measured" if n == 1 else None,
        )
    ax.set_xticks(ns)
    ax.set_xticklabels(["add\n(n=1)", "add3\n(n=2)", "add4\n(n=3)"])
    ax.set_xlabel("chained binary adds")
    ax.set_ylabel("effective BW  (R+W)/time  (GB/s)")
    ax.set_title("§3  n-ary derate: effBW falls per chained op")
    ax.set_ylim(94, 120)
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.9)
    _save(fig, "fig3_pointwise_arity")


# ============================================================================
# §4 -- a broadcast operand raises the effective BW: broadcast-operand ops
# (copy/bcast/bcastcol/mulbcast) run ~118 GB/s, above plain 1R:1W neg (~105).
# Small-ROWS points are underfilled (2-8 rows/core) and unreliable -> flagged.
# ============================================================================
def fig_broadcast_effbw(recs):
    # neg (1R:1W) baseline from the records; broadcast ops from the dedicated COLS sweep.
    neg = [
        int(r["io_hbm_bytes"]) / 1e3 / r["kernel_us"]
        for r in recs
        if r["op"] == "neg" and r.get("io_hbm_bytes")
    ]
    blog = _broadcast_log_rows()
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
    ax.axhline(118, ls="--", color="0.5", lw=1.0, label="broadcast rate = 118")
    for op, c in colors.items():
        pts = sorted(
            (C, io / 1e3 / k) for o, R, C, io, k in blog if o == op and R == 2048
        )
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=c, ms=4.5, label=op)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 2048, 4096, 8192, 16384])
    ax.set_xticklabels(["1k", "2k", "4k", "8k", "16k"])
    ax.set_xlabel("COLS  (ROWS = 2048, well-filled)")
    ax.set_ylabel("effective BW  (R+W)/time  (GB/s)")
    ax.set_title("§4  Broadcast-operand ops run ~118 GB/s, flat across size")
    ax.set_ylim(100, 138)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9, ncol=2)
    _save(fig, "fig4_broadcast_effbw")


# ============================================================================
# §5 -- reductions are read-dominated -> read-only rate (~150), EXCEPT the output
# [R] is stick-inflated to R×64 and written scattered; at large R it drags effBW down.
# ============================================================================
def fig_reduction(recs):
    ops = ("read", "sumrow", "amax", "mean")
    colors = {
        "read": "#1f77b4",
        "sumrow": "#2ca02c",
        "amax": "#9467bd",
        "mean": "#ff7f0e",
    }
    fig, ax = plt.subplots(figsize=(5.0, 3.7))
    ax.axhline(150, ls="--", color="0.55", lw=1.2, label="read-only rate ≈ 150 (model)")
    for op in ops:
        pts = []
        for r in recs:
            if r["op"] == op and r.get("io_hbm_bytes"):
                io = int(r["io_hbm_bytes"])
                pts.append((r.get("rows") or 0, io / 1e3 / r["kernel_us"]))
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(
                xs,
                ys,
                s=40,
                color=colors[op],
                label=op,
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks([2048, 8192])
    ax.set_xticklabels(["2048\n(64 rows/core)", "8192\n(256 rows/core)"])
    ax.set_xlabel("ROWS  (reduced tensor height)")
    ax.set_ylabel("effective BW  (R+W)/time  (GB/s)")
    ax.set_title("§5  Reductions: on the read rate, but slower at large ROWS")
    ax.set_ylim(110, 160)
    ax.annotate(
        "the read runs slower at\n256 rows/core than at 64\n(mechanism unresolved)",
        xy=(0.5, 0.06),
        xycoords="axes fraction",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="0.3",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.8"),
    )
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
    rows = [r for r in _broadcast_log_rows() if r[0] == "write"]
    fig, ax = plt.subplots(figsize=(5.0, 3.7))
    colors = {512: "#1f77b4", 2048: "#ff7f0e", 8192: "#d62728"}
    for R in (512, 2048, 8192):
        pts = sorted((C, k / (R * C * 2 / 1e6)) for op, Rr, C, io, k in rows if Rr == R)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=colors[R], ms=5, label=f"ROWS={R}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1024, 4096, 16384])
    ax.set_xticklabels(["1024", "4096", "16384"])
    ax.set_xlabel("COLS")
    ax.set_ylabel("time per MB of output  (µs/MB)")
    ax.set_title("§4  write: per-output cost rises with COLS (and weakly ROWS)")
    ax.annotate(
        "fast at small size;\nper-output cost rises\nwith COLS (and, weakly, ROWS)",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=8,
        color="0.3",
        bbox=dict(boxstyle="round", fc="#f5f5f5", ec="0.8"),
    )
    ax.legend(loc="upper center", fontsize=8.5, framealpha=0.9)
    _save(fig, "fig4b_write_spill")


# ============================================================================
# §6 -- transport ops (transpose / cat) lower to a byte-copy (`clone`); the only
# difference from a plain copy is the access pattern, which sets an effective BW.
# transpose is flat-fast (116); cat0 and transpose_outer fall with size.
# ============================================================================
def fig_transport(_recs):
    # transport times are version-independent, so use ALL records (not just
    # is_current) for full size coverage; average repeats at each (op, bytes).
    recs = _load(current_only=False)

    def effbw(r):
        return int(r["io_hbm_bytes"]) / 1e3 / r["kernel_us"]

    def series(op):
        agg = {}
        for r in recs:
            if r["op"] == op and r.get("io_hbm_bytes"):
                agg.setdefault(int(r["io_hbm_bytes"]), []).append(effbw(r))
        return sorted((b, sum(v) / len(v)) for b, v in agg.items())

    # baseline: plain copy at large operands (>=16 MB); the sub-16 MB tail runs
    # artificially fast and would not be a fair copy reference for these sizes.
    neg = [
        effbw(r)
        for r in recs
        if r["op"] == "neg"
        and r.get("io_hbm_bytes")
        and int(r["io_hbm_bytes"]) >= 16 << 20
    ]
    styles = {
        "transpose": ("#2ca02c", "o", "transpose (stick-swap copy)"),
        "cat1": ("#1f77b4", "s", "cat1 (outer-dim concat)"),
        "cat0": ("#d62728", "^", "cat0 (stick-dim concat)"),
        "transpose_outer": ("#9467bd", "D", "transpose_outer (3-D outer swap)"),
    }
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    ax.axhspan(
        min(neg),
        max(neg),
        color="0.6",
        alpha=0.18,
        zorder=0,
        label=f"neg copy baseline (large operands): {min(neg):.0f}–{max(neg):.0f}",
    )
    for op, (c, m, lab) in styles.items():
        pts = series(op)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-", color=c, lw=1.0, alpha=0.6)
            ax.scatter(xs, ys, color=c, marker=m, s=34, label=lab, zorder=3)
    ax.axhline(116, ls="--", color="#2ca02c", lw=0.9)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("bytes moved  (R + W, log scale)")
    ax.set_ylabel("effective BW  (R+W)/time  (GB/s)")
    ax.set_title("§6  Transport ops are copies; access pattern sets the rate")
    ax.set_ylim(50, 130)
    ax.legend(loc="lower left", fontsize=7.2, framealpha=0.9)
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
    # Compute-free matmuls: thin-K (write-heavy) and thin-M (read-heavy). Plot the
    # DOMINANT-operand rate (dominant bytes / time) -- the honest "is there a distinct
    # read vs write rate?" view. The two clusters OVERLAP and both sit below the 150
    # copy peak: one effective rate, not two, and nothing near the artifact 156.
    rows = _mm_rows("M1")
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.axhline(150, ls="--", color="0.4", lw=1.3, label="copy peak = 150 (§1)")
    ax.axhline(
        156,
        ls=":",
        color="#d62728",
        lw=1.1,
        label="old two-rate BW_w = 156 (unphysical)",
    )
    for r in rows:
        M, N, K, t = r.get("M"), r.get("N"), r.get("K"), r["kernel_us"]
        w = M * N * 2 / 1e3  # output bytes (KB) -> GB/s when divided by µs
        rd = (M * K + K * N) * 2 / 1e3
        if w >= rd:  # write-heavy (thin K)
            ax.scatter(
                int(r["io_hbm_bytes"]) / 1e6, w / t, color="#d62728", s=40, zorder=3
            )
        else:  # read-heavy (thin M)
            ax.scatter(
                int(r["io_hbm_bytes"]) / 1e6, rd / t, color="#1f77b4", s=40, zorder=3
            )
    ax.scatter(
        [], [], color="#d62728", s=40, label="write rate (thin K, output-dominated)"
    )
    ax.scatter(
        [], [], color="#1f77b4", s=40, label="read rate (thin M, operand-dominated)"
    )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("bytes moved  (MB, log scale)")
    ax.set_ylabel("dominant-operand rate  (GB/s)")
    ax.set_title("§8  Compute-free matmuls: read & write rates overlap, both < peak")
    ax.set_ylim(90, 165)
    ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    _save(fig, "fig8_matmul_hbm")


def fig_matmul_spill(_recs):
    # Fixed 4x8 split, vary M -> per-core tile M/m = M/4 grows. Cost per output row
    # (time/M) rises then saturates: the operand re-read (spill) signature.
    rows = sorted(_mm_rows("DC2"), key=lambda r: r.get("M") or 0)
    xs = [(r.get("M") or 0) / 4 for r in rows]  # per-core tile rows M/m, m=4
    ys = [r["kernel_us"] * 1e3 / (r.get("M") or 1) for r in rows]  # ns per output row
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(xs, ys, "-o", color="#8c564b", ms=6)
    ax.axvline(448, ls="--", color="0.5", lw=1.1, label="on-chip knee ≈ 448 rows/core")
    for x, y in zip(xs, ys):
        ax.annotate(
            f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 7), fontsize=7.5
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("per-core tile  (M/m rows, m = 4)")
    ax.set_ylabel("cost per output row  (ns)")
    ax.set_title("§9  Per-row cost rises with the per-core tile, then saturates")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "fig9_matmul_spill")


def fig_matmul_peak(_recs):
    # Compute-dominant cores scan (K=4096): kernel time is linear in 1/cores; the
    # slope is the per-core MAC rate (peak). A slope is immune to any constant offset.
    rows = sorted(_mm_rows("GD"), key=lambda r: r.get("cores") or 0)
    cores = [r.get("cores") for r in rows]
    inv = [1.0 / c for c in cores]
    t = [r["kernel_us"] for r in rows]
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(inv, t, "o", color="#2ca02c", ms=7, zorder=3, label="measured (K=4096)")
    # least-squares line through the points (visual: slope -> peak)
    a, b = np.polyfit(inv, t, 1)
    xs = np.linspace(0, max(inv) * 1.05, 20)
    ax.plot(
        xs, a * xs + b, "-", color="0.5", lw=1.1, label="linear fit (slope ∝ 1/peak)"
    )
    for x, c in zip(inv, cores):
        ax.annotate(
            f"{c} cores",
            (x, a * x + b),
            textcoords="offset points",
            xytext=(6, -4),
            fontsize=7.5,
        )
    ax.set_xlabel("1 / cores")
    ax.set_ylabel("kernel time  (µs)")
    ax.set_title("§10  Compute-dominant time is linear in 1/cores")
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, "fig10_matmul_peak")


def fig_matmul_overlap(_recs):
    # Balanced cores scan: measured vs additive (gamma=0) vs overlap model. The
    # additive model over-predicts; the overlap term closes the gap.
    cm = _cost_model()
    rows = sorted(_mm_rows("GD"), key=lambda r: r.get("cores") or 0)
    cores, meas, add, ov = [], [], [], []
    p_add = cm.CostParams(overlap_gamma=0.0)
    for r in rows:
        feats = r["feats"]
        feats = feats if isinstance(feats, list) else json.loads(feats)
        ops = cm.ops_from_json(json.dumps(feats))
        cores.append(r.get("cores"))
        meas.append(r["kernel_us"])
        add.append(cm.predict_ops(ops, p_add) / 1e3)
        ov.append(cm.predict_ops(ops) / 1e3)
    x = np.arange(len(cores))
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(x, meas, "o-", color="#1f77b4", ms=6, label="measured")
    ax.plot(x, add, "s--", color="#d62728", ms=5, label="compute + memory added (γ=0)")
    ax.plot(x, ov, "^-", color="#2ca02c", ms=5, label="with overlap (γ≈0.46)")
    ax.set_yscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}" for c in cores])
    ax.set_xlabel("cores (balanced split, K=4096)")
    ax.set_ylabel("kernel time  (µs, log)")
    ax.set_title("§11  Adding compute+memory over-predicts; overlap closes it")
    ax.legend(loc="upper right", fontsize=8)
    _save(fig, "fig11_matmul_overlap")


_FIGS = {
    "matmul_hbm": fig_matmul_hbm,
    "matmul_spill": fig_matmul_spill,
    "matmul_peak": fig_matmul_peak,
    "matmul_overlap": fig_matmul_overlap,
    "pointwise_baseline": fig_pointwise_baseline,
    "pointwise_vcurve": fig_pointwise_vcurve,
    "pointwise_arity": fig_pointwise_arity,
    "broadcast_effbw": fig_broadcast_effbw,
    "write_spill": fig_write_spill,
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
