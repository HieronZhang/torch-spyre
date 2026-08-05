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

"""Regenerate the figures for docs/source/compiler/cost_model_report.md from sweep_records.json.

Each figure is a function ``fig_<name>`` that reads the stored measured times and writes
a PNG to docs/source/compiler/cost_model_figures/. Reproducible -- the data is the committed sweep records, not
hand-drawn. Run one figure or all:

    python tools/cost_model/plot_report.py                 # all figures
    python tools/cost_model/plot_report.py pointwise_baseline
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # tools/cost_model -> repo root
_FIGDIR = os.path.join(_ROOT, "docs/source/compiler/cost_model_figures")
sys.path.insert(0, _HERE)
from records import records_path  # noqa: E402

_RECORDS = records_path()


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


def _load(current_only=True):
    recs = json.load(open(_RECORDS, encoding="utf-8"))["records"]
    recs = [r for r in recs if not r.get("failed") and r.get("kernel_us")]
    if current_only:
        recs = [r for r in recs if r.get("is_current")]
    return recs


def _pred_live(r, params=None):
    """Predict a record with the LIVE model, in microseconds, or None.

    Figures must never plot the stored ``pred_us``: that field was written by whatever
    model generation happened to produce the record, so a figure built on it silently
    mixes model versions and goes stale the moment a coefficient changes.
    """
    cm = _cost_model()
    f = r.get("feats")
    if not f:
        return None
    try:
        ops = cm.ops_from_json(json.dumps(f) if isinstance(f, list) else f)
        return cm.predict_ops(ops, params or cm.CostParams()) / 1000.0
    except Exception:  # noqa: BLE001
        return None


def _base_matmul_params():
    """Model with the batched-matmul rate terms switched OFF (§13's 'base model').

    §13 shows what a batched matmul costs BEFORE its own term exists, so the gates are
    disabled explicitly rather than relying on records that predate the term.
    """
    cm = _cost_model()
    p = cm.CostParams()
    p.bmm_default_min_batch = 10**9  # layout-pair rate never fires
    p.bmm_3d2d_mac_peak_lo_ns = p.mac_peak_per_core_ns  # 3d-2d keeps the plain peak
    p.bmm_3d2d_mac_peak_hi_ns = p.mac_peak_per_core_ns
    return p


def _save(fig, name, dpi=130):
    os.makedirs(_FIGDIR, exist_ok=True)
    path = os.path.join(_FIGDIR, f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, _HERE)}")


# ============================================================================
# SHARED FIGURE STYLE. Every figure in the report uses this so the document reads
# as one piece: large readable fonts, axis labels that always carry UNITS, and a
# coverage note so a reader can see at a glance which configurations a claim rests
# on. Annotation boxes go in a corner the data does not occupy -- never over points.
# ============================================================================
FS_TITLE, FS_LABEL, FS_TICK, FS_ANNOT, FS_LEGEND = 13, 12, 11, 10, 10


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=FS_TITLE)
    ax.set_xlabel(xlabel, fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    ax.tick_params(labelsize=FS_TICK)
    ax.grid(alpha=0.25, lw=0.6)
    return ax


def _coverage(ax, text, loc="upper left"):
    """Small box stating what the figure covers, so the reader can judge the claim.

    Lines are WRAPPED and capped: an un-wrapped coverage line once stretched a figure to
    2352 px wide and hid a data label behind the box. Keep it to a few short lines --
    summarise the span of the configurations rather than listing every one.
    """
    import textwrap

    text = "\n".join(
        "\n".join(textwrap.wrap(ln, 52)) if len(ln) > 52 else ln
        for ln in text.split("\n")
    )
    xy = {
        "upper left": (0.03, 0.97),
        "upper right": (0.97, 0.97),
        "lower left": (0.03, 0.03),
        "lower right": (0.97, 0.03),
    }[loc]
    ax.annotate(
        text,
        xy=xy,
        xycoords="axes fraction",
        va="top" if "upper" in loc else "bottom",
        ha="left" if "left" in loc else "right",
        fontsize=FS_ANNOT,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.92),
    )


# ============================================================================
# §1 -- pointwise is memory-I/O bound: neg time is linear in bytes through the
# origin (no fixed per-kernel cost).
# ============================================================================
def fig_pointwise_baseline(recs):
    """§1: kernel time is linear in bytes moved, with no fixed per-kernel cost.

    Held fixed: 32 cores (the full machine). Mixing core counts here would be wrong --
    fewer cores move the same bytes more slowly, which shows up as vertical scatter and
    is a DIFFERENT effect, derived later. Three ops are drawn so the second claim of the
    section -- that the arithmetic itself is free -- is visible in the same picture.
    """
    OPS = {"neg": ("#1f77b4", "o"), "gelu": ("#d62728", "s"), "exp": ("#2ca02c", "^")}
    rows = [
        r
        for r in _load(current_only=False)
        if r.get("op") in OPS
        and r.get("io_hbm_bytes")
        and r.get("kernel_us")
        # Full machine only. Fewer cores move the same bytes more slowly (the intercept
        # reaches +245 us at 1 core), which is a DIFFERENT effect derived later; mixing
        # core counts here drops the fit to R^2 = 0.957. Rows whose `cores` field was
        # never recorded are full-machine runs from the earliest sweeps -- on their own
        # they fit 102.4 and 102.8 GB/s -- so they are kept rather than silently dropped.
        and (r.get("cores") in (32, None, "-"))
    ]
    x = np.array([r["io_hbm_bytes"] / 1e6 for r in rows])
    y = np.array([r["kernel_us"] for r in rows])
    a, b = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (a * x + b)) ** 2) / np.sum((y - y.mean()) ** 2)
    bw = 1e3 / a

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    xline = np.array([0, x.max() * 1.04])
    ax.plot(
        xline,
        a * xline + b,
        "-",
        color="0.45",
        lw=1.6,
        zorder=1,
        label=f"fit: {bw:.0f} GB/s,  $R^2$={r2:.5f}",
    )
    for op, (c, m) in OPS.items():
        pts = [(r["io_hbm_bytes"] / 1e6, r["kernel_us"]) for r in rows if r["op"] == op]
        if not pts:
            continue
        ax.scatter(
            [p[0] for p in pts],
            [p[1] for p in pts],
            s=52,
            c=c,
            marker=m,
            zorder=3,
            edgecolors="white",
            linewidths=0.6,
            label=f"{op}  (n={len(pts)})",
        )
    shapes = sorted(
        {(r.get("rows"), r.get("cols")) for r in rows},
        key=lambda t: (t[0] or 0) * (t[1] or 0),
    )
    _style(
        ax,
        "§1  Pointwise kernel time is set by bytes moved  (32 cores)",
        "HBM traffic moved by the kernel  (MB)",
        "measured kernel time  (µs)",
    )
    _coverage(
        ax,
        f"{len(rows)} runs, {len(shapes)} tensor shapes\n"
        f"{shapes[0][0]}×{shapes[0][1]} … {shapes[-1][0]}×{shapes[-1][1]}\n"
        f"traffic {x.min():.0f}–{x.max():.0f} MB\n"
        f"intercept = {b:+.1f} µs  (no fixed start-up cost)",
    )
    ax.legend(loc="lower right", fontsize=FS_LEGEND, framealpha=0.92)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    _save(fig, "s01_pointwise_baseline")


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
    _p0 = _cost_model().CostParams()
    bw_peak, alpha = _p0.mm_bw_read_gbps, _p0.rw_turnaround_ns_per_byte  # current model
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
        # Drop KNOWN partial-machine runs (leaving unrecorded core counts alone): they
        # land at 16-77 GB/s and the y-limit below silently clips them, so they would be
        # in the population without being visible. This curve is the full-machine one.
        if isinstance(r.get("cores"), int) and r["cores"] != 32:
            continue
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
    _save(fig, "s02_read_write_ratio")


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
        ("two independent\nadditions", a[("add_indep2", 0)], "#2ca02c", ""),
        ("dependent chain,\nseparate kernels", a[("add3_sep", 0)], "#ff7f0e", ""),
        ("dependent chain,\none fused kernel", a[("add3", 0)], "#1f77b4", ""),
        ("same chain, result\nkept on chip", a[("add3", 1)], "#9ecae1", "//"),
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
        "predicted from bytes\n(two additions)",
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
    _style(
        ax,
        "§3  A dependent sum costs more than its bytes — and fusing it changes nothing\n"
        "(three inputs, 2048 × 4096)",
        "",
        "kernel time  (µs)",
    )
    fig.tight_layout()
    _save(fig, "s03_read_after_write")


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

    # Byte count comes from the stored total when present and otherwise from the
    # recorded features. Gating on `io_hbm_bytes` alone silently dropped every record
    # that carries its bytes only in `feats`, which blanked the middle of this sweep.
    cm = _cost_model()
    brec = defaultdict(lambda: defaultdict(list))
    for r in _load(current_only=False):
        if r.get("op") not in ("copy", "bcast", "bcastcol", "mulbcast"):
            continue
        if r.get("rows") != 2048 or not r.get("kernel_us"):
            continue
        b = r.get("io_hbm_bytes")
        if not b and r.get("feats"):
            f = r["feats"]
            ops = cm.ops_from_json(json.dumps(f) if isinstance(f, list) else f)
            rb, wb = cm._fused_hbm_bytes(ops)
            b = rb + wb
        if b:
            brec[r["op"]][r["cols"]].append(int(b) / 1e3 / r["kernel_us"])
    colors = {
        "copy": "#d62728",
        "bcast": "#2ca02c",
        "bcastcol": "#9467bd",
        "mulbcast": "#ff7f0e",
    }

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhspan(
        min(neg),
        max(neg),
        color="#1f77b4",
        alpha=0.12,
        zorder=0,
        label=f"plain copy, every size measured: {min(neg):.0f}–{max(neg):.0f}",
    )
    ax.axhline(
        np.mean(neg),
        ls="--",
        color="#1f77b4",
        lw=1.1,
        label=f"plain copy, mean {np.mean(neg):.0f}",
    )
    ax.axhline(118, ls="--", color="0.5", lw=1.0, label="rate used by the model: 118")
    plain = {
        "copy": "add a constant",
        "bcast": "broadcast a row",
        "bcastcol": "broadcast a column",
        "mulbcast": "multiply by a row",
    }
    for op, c in colors.items():
        pts = sorted((C, sum(v) / len(v)) for C, v in brec[op].items())
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=c, ms=4.5, label=plain[op])
    ax.set_xscale("log", base=2)
    ax.set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384])
    ax.set_xticklabels(["256", "512", "1k", "2k", "4k", "8k", "16k"])
    _style(
        ax,
        "§4  A small resident operand lifts the rate above a plain copy",
        "columns in the tensor  (2048 rows throughout)",
        "effective bandwidth  (GB/s)",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=FS_LEGEND - 1,
        frameon=False,
    )
    _save(fig, "s04_broadcast_bandwidth")


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
            # cores MUST be filtered: this curve is the FULL-machine rate (the
            # core-count effect is a separate paragraph in the section). Partial-machine
            # runs sit at 14-80 GB/s, and averaging them in dragged the 4096- and
            # 8192-row means to ~71-79 GB/s -- which set_ylim then clipped off the
            # bottom, so four of the five ops silently vanished at those two row counts
            # while the figure still claimed all five trace one curve.
            if (
                r["op"] == op
                and r.get("io_hbm_bytes")
                and r.get("cols") == 2048
                and r.get("cores") == 32
            ):
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
    _save(fig, "s06_reduction_read_rate")


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
    _save(fig, "s04b_broadcast_small_rows")


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
    _save(fig, "s07_transport_access_pattern")


# ============================================================================
# §8-§11 -- matmul. These need model predictions, so load cost_model the same
# hardware-free way eval_model does (importlib; it imports only math/dataclasses).
# ============================================================================
def _cost_model():
    import importlib.util

    path = os.path.join(_ROOT, "torch_spyre", "_inductor", "cost_model.py")
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
    ax.scatter([], [], color="#d62728", s=44, label="write-heavy  (thin K ∈ {16,32})")
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
    _save(fig, "s09_matmul_memory_term")


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
            lambda v, _: (
                f"{v / 1048576:.0f} MB" if v >= 1048576 else f"{v / 1024:.0f} KB"
            )
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
    _save(fig, "s12_accumulator_spill")


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
            lambda v, _: (
                f"{v / 1048576:.0f} MB" if v >= 1048576 else f"{v / 1024:.0f} KB"
            )
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
    _save(fig, "s13_split_shape")


def fig_matmul_bmm_layout(_recs):
    """§13: the batched-matmul penalty is set by how each operand is laid out in memory.

    Every point is a batched matmul run at 32 cores. The four groups are the four
    combinations of the two operands' memory layouts. "Batch-outer" means the batch index
    is the slowest-varying dimension in memory; "row-outer" means the row index is. The
    byte counts are IDENTICAL across the four -- only the traversal order differs -- so
    any spread is dataflow, not traffic.
    """
    import regex as re

    LBL = {"0,1,2": "batch-outer", "1,0,2": "row-outer"}
    rows = [
        r
        for r in _load(current_only=False)
        if r.get("op") == "bmm_layout"
        and r.get("kernel_us")
        and not r.get("failed")
        and r.get("layout_a") in LBL
        and r.get("layout_b") in LBL
    ]
    combos = [
        ("0,1,2", "0,1,2"),
        ("0,1,2", "1,0,2"),
        ("1,0,2", "0,1,2"),
        ("1,0,2", "1,0,2"),
    ]

    # normalise each run against the FASTEST layout of its own shape, so shapes of very
    # different absolute cost can be compared on one axis.
    def shape_of(r):
        """(batch, M, K, N). Labels appear in TWO formats in the record set --
        `B=2 M=1024 K=2048 N=1024` and `B=4 1024x2048x1024` -- so both are parsed.
        Batch is part of the key because a batched matmul's cost scales with it, so
        runs may only be normalised against the same batch."""
        lab = str(r.get("label") or "")
        b = re.search(r"\bB=(\d+)", lab)
        m = re.search(r"M=(\d+)\s+K=(\d+)\s+N=(\d+)", lab) or re.search(
            r"(\d+)x(\d+)x(\d+)", lab
        )
        if not (b and m):
            return None
        return (int(b[1]), int(m[1]), int(m[2]), int(m[3]))

    best = {}
    for r in rows:
        sh = shape_of(r)
        if sh is None:
            continue
        k = (sh, r.get("cores"))
        if r["layout_a"] == "1,0,2" and r["layout_b"] == "1,0,2":
            best[k] = min(best.get(k, 1e18), r["kernel_us"])

    # Collect EVERY group first, so the axis limits can leave room for the group-mean
    # labels. Placing a label above a limit computed from one group clipped the lead
    # group's mean off the canvas -- the figure then showed 3 of 4 labels, and the
    # missing one carried the headline number.
    shapes_seen = set()
    series, drawn = [], 0
    for la, lb in combos:
        ys = []
        for r in rows:
            if r.get("layout_a") != la or r.get("layout_b") != lb:
                continue
            sh = shape_of(r)
            b = best.get((sh, r.get("cores")))
            if not sh or not b:
                continue
            ys.append(r["kernel_us"] / b)
            shapes_seen.add(sh)
        series.append(ys)
        drawn += len(ys)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = ["#d62728", "#ff7f0e", "#9467bd", "#2ca02c"]
    top = max(max(ys) for ys in series if ys)
    for xi, ys in enumerate(series):
        if not ys:
            continue
        ax.scatter(
            [xi + 0.06 * (k % 5 - 2) for k in range(len(ys))],
            ys,
            s=58,
            c=colors[xi],
            zorder=3,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.annotate(
            f"{np.mean(ys):.2f}×",
            xy=(xi, max(ys) + 0.06 * top),
            ha="center",
            fontsize=FS_ANNOT + 1,
            fontweight="bold",
            color=colors[xi],
        )
    ax.axhline(1.0, color="0.5", ls="--", lw=1.2, zorder=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"A {LBL[a]}\nB {LBL[b]}" for a, b in combos], fontsize=FS_TICK)
    _style(
        ax,
        "§14  Batched multiply: the operands' memory layout sets the cost  (32 cores)",
        "memory layout of the two operands",
        "time ÷ time of the fastest layout\n(same shape, same bytes)",
    )
    bs = sorted({t[0] for t in shapes_seen})
    dims = sorted({d for t in shapes_seen for d in t[1:]})
    _coverage(
        ax,
        f"{drawn} runs · {len(shapes_seen)} shapes\n"
        f"batch {bs[0]}–{bs[-1]} · M,K,N {dims[0]}–{dims[-1]}\n"
        f"identical bytes in all four groups",
        loc="upper right",
    )
    ax.set_ylim(bottom=0.9, top=top * 1.22)
    _save(fig, "s14_bmm_operand_layout")


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
    ax.set_ylim(
        0, 4600
    )  # focus on cores 4-32; the off-axis 2-core points (x=1/2) would
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
    _save(fig, "s10_matmul_compute_rate")


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
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.axvspan(0.5, 100, color="#f2dede", alpha=0.45, zorder=0)
    ax.axvline(0.5, ls="--", color="0.35", lw=1.4, zorder=1)
    seen = set()
    for r in recs:
        if r.get("op") != "softmax_row_tiling" or not r.get("feats"):
            continue
        R, C, t = r.get("rows"), r.get("cols"), r.get("tiles")
        if not (R and C and t) or t < 2:  # tiles=1 untiled = HBM by design, not a spill
            continue
        # cores MUST be filtered: the section's table is the 32-core population, and
        # partial-machine runs sit at 8-48 GB/s -- they were previously plotted and then
        # silently CLIPPED by the y-limit, so the figure disagreed with its own legend.
        if r.get("cores") != 32:
            continue
        f = r["feats"]
        f = f if isinstance(f, list) else json.loads(f)
        ops = cm.ops_from_json(json.dumps(f))
        Rb, Wb = cm._fused_hbm_bytes(ops)
        effbw = (Rb + Wb) / 1e3 / r["kernel_us"]  # GB/s
        ws = 2 * (R / t / 32) * C * 2 / 1e6  # MB/core
        if ws < 0.25:  # short-tile underfill regime -- that is §16's figure, not this
            continue
        col = palette.get((R, C), "0.4")
        ax.scatter(
            ws, effbw, s=42, color=col, zorder=3, edgecolors="white", linewidths=0.4
        )
        key = (R, C) if (R, C) in palette else "other"
        if key not in seen:
            lab = f"{R}×{C}" if key != "other" else "other shapes"
            ax.scatter([], [], color=col, s=42, label=lab)
            seen.add(key)
    # model: effBW = balanced-softmax peak (~100) * spill derate past the knee
    wsx = np.logspace(np.log2(0.25), np.log2(9), 60, base=2)
    # cap/exponent come from the LIVE params so the curve cannot drift from the model.
    _p = cm.CostParams()
    _cap = _p.lx_spill_cap_bytes / 1e6
    _exp = _p.lx_spill_exp
    peak = 100.0  # measured plateau of this population, not a model parameter
    model = [peak * min(1.0, (_cap / w) ** _exp) for w in wsx]
    ax.plot(
        wsx,
        model,
        "-",
        color="0.25",
        lw=1.8,
        zorder=2,
    )
    ax.set_xscale("log", base=2)
    ax.set_ylim(55, 118)
    ax.set_xlim(0.22, 10)
    _style(
        ax,
        "§17  The bytes are all counted — a bigger per-core tile just moves them slower",
        "per-core working set  (MB)",
        "effective bandwidth  (R+W)/time  (GB/s)",
    )
    # Label the two regions and the model curve in place, so no legend box sits
    # over the data (the shape key goes outside the axes, below).
    ax.annotate(
        "fits in LX",
        xy=(0.27, 44),
        fontsize=FS_ANNOT,
        color="0.35",
        ha="left",
        va="bottom",
    )
    ax.annotate(
        "spills → slower",
        xy=(0.55, 57),
        fontsize=FS_ANNOT,
        color="#a33",
        ha="left",
        va="bottom",
    )
    ax.annotate(
        "fitted threshold",
        xy=(0.5, 116),
        xytext=(0.53, 116),
        fontsize=FS_ANNOT,
        color="0.3",
        ha="left",
        va="top",
    )
    ax.annotate(
        "model",
        xy=(6.0, peak * (0.5 / 6.0) ** 0.15),
        xytext=(6.6, 68),
        fontsize=FS_ANNOT,
        color="0.25",
        ha="center",
        va="top",
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=4,
        fontsize=FS_LEGEND,
        frameon=False,
    )
    _coverage(
        ax,
        "softmax_row_tiling, 32 cores, tiles ≥ 2.\nShorter tiles are §16.",
        "upper right",
    )
    _save(fig, "s18_lx_spill")


# ============================================================================
# §15 -- coarse underfill `eff`: a short per-core tile (rpc rows) never fills the
# streaming pipeline. Softmax effective BW climbs with rpc to a plateau; the model
# eff = min(0.95, (rpc/13)^0.68) (calibrated) captures the rise. Above rpc~32 a
# mild decline is unmodeled.
# ============================================================================
def fig_coarse_eff(_recs):
    # One marker per CONFIG (not averaged): color = ROWS×COLS shape, label = tile count.
    # x = per-core tile height h = ROWS/(cores·tiles); LX-fitting points only.
    #
    # POPULATION must match the one §16 scores, or the curve reads as an upper envelope
    # over a cloud. Two families used to be drawn here that the report excludes from every
    # number: runs below 32 cores (their effective BW tracks the core count -- median
    # 11/21/41/63/79 GB/s at 1/2/4/8/16 cores -- which is aggregate bandwidth, not
    # underfill), and reductions narrower than 1024 columns (a different regime, see the
    # permanent exclusions). Together they were 73 of 131 points and 60 of the 60 that sat
    # far below the curve. On the scored population, no point falls 25 % below it.
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
        if r.get("cores") != 32 or C < 1024:
            continue  # the scored population -- see the note above
        h = R / t / (r.get("cores") or 32)
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
    model = plateau * np.minimum(
        0.95,
        (rr / _cost_model().CostParams().coarse_underfill_rfull)
        ** _cost_model().CostParams().coarse_underfill_exp,
    )
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
        "§16  Underfill `eff`: BW climbs with the per-core tile, then plateaus"
    )
    # Outside the axes: at 32 cores the points run from h=2 up the climb, straight
    # through where a lower-right legend used to sit.
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.0, framealpha=0.9
    )
    fig.subplots_adjust(right=0.68)
    _save(fig, "s17_underfill")


_FIGS = {
    "matmul_hbm": fig_matmul_hbm,
    "matmul_spill": fig_matmul_spill,
    "matmul_split": fig_matmul_split,
    "matmul_bmm_layout": fig_matmul_bmm_layout,
    "coarse_spill": fig_coarse_spill,
    "coarse_eff": fig_coarse_eff,
    "matmul_peak": fig_matmul_peak,
    "pointwise_baseline": fig_pointwise_baseline,
    "pointwise_vcurve": fig_pointwise_vcurve,
    "pointwise_arity": fig_pointwise_arity,
    "broadcast_effbw": fig_broadcast_effbw,
    "broadcast_smallr": fig_broadcast_smallr,
    "reduction": fig_reduction,
    "transport": fig_transport,
}


def main():
    import sys

    # current_only=False DELIBERATELY. `is_current` marks rows whose stored pred_us came
    # from the newest logged SHA; it is NOT a data-quality flag, and it collapses to a
    # handful of rows every time a new sweep lands (16 of 2653, all one op, on
    # 2026-08-04). Figures that consumed it silently lost their data -- fig1 degraded to
    # a single point with R^2 = -inf while its caption still claimed R^2 = 0.99996.
    # Measured time is version-independent, so every figure wants ALL rows.
    recs = _load(current_only=False)
    want = sys.argv[1:] or list(_FIGS)
    for name in want:
        if name not in _FIGS:
            raise SystemExit(f"unknown figure {name!r} (have: {', '.join(_FIGS)})")
        print(f"figure {name}:")
        _FIGS[name](recs)


if __name__ == "__main__":
    main()
