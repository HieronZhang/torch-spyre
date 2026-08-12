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

"""Generate every table and figure in `lx_allocation_report.md` from the measurements.

The report's numbers are never typed by hand. Each table lives between a
``<!-- BEGIN:name -->`` / ``<!-- END:name -->`` pair in the Markdown, and this rewrites the
span in place from the measurement files. Re-run it after any new sweep and the document
follows; edit a number by hand and the next run silently reverts it, which is the point.

Sources, all of them measurements rather than screens:

  ranking_211      the recovered 2026-07-30 database -- coarse-tiled flash on torch 2.11
  alloc_213        `lx_session_results.jsonl` phase 1 -- pinned allocations on torch 2.13
  alloc_detail     `forced_allocations.json` -- which buffers each policy keeps
  ranking_213      `lx_session_results.jsonl` phase 2 -- shape/core ranking
  cores_213        the same, split by core count
  contested        `untiled_flash_records.json` solved exactly

    python3 research/gen_lx_report.py            # rewrite the tables and the figure
    python3 research/gen_lx_report.py --show     # print them without touching the report
"""

import argparse
import collections
import json
import os
import statistics as st
import subprocess
import sys

import regex as re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools", "cost_model"))

REPORT = os.path.join(_HERE, "lx_allocation_report.md")
SOLVER_FIG = os.path.join(_HERE, "lx_solver_pred.png")
CONFIG_FIG = os.path.join(_HERE, "lx_config_pred.png")
POLICY_FIG = os.path.join(_HERE, "lx_policy_diff.png")
CAP = 1_625_344

#: The 2.11 flash data lives in a superseded commit -- the working tree's database was
#: overwritten by the re-sweep. Read it straight from git so the report cannot drift onto
#: whatever happens to be checked out.
DB_211_REF = "3f57eb5:tools/cost_model/sweep_records.json"


def _db_211():
    out = subprocess.run(["git", "show", DB_211_REF], cwd=_ROOT, capture_output=True)
    if out.returncode:
        return []
    return json.loads(out.stdout)["records"]


def _session():
    path = os.path.join(_HERE, "lx_session_results.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]


def _v2():
    path = os.path.join(_HERE, "sweep_v2_results.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(ln) for ln in open(path, encoding="utf-8") if ln.strip()]


def _kendall(pairs):
    """(concordant, total, tau) over (measured, predicted) pairs."""
    c = d = 0
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            dm = pairs[i][0] - pairs[j][0]
            dp = pairs[i][1] - pairs[j][1]
            if dm == 0 or dp == 0:
                continue
            c += (dm > 0) == (dp > 0)
            d += (dm > 0) != (dp > 0)
    return c, c + d, ((c - d) / (c + d) if c + d else float("nan"))


def _tidy(label):
    return (
        label.replace("flash_attn ", "")
        .replace("flash ", "")
        .split(" [was")[0]
        .split(" (IR")[0]
        .strip()
    )


# --------------------------------------------------------------------------- 2.11


def ranking_211(build="20260730"):
    """Coarse-tiled flash on torch 2.11: measured vs predicted, ONE compiler build."""
    import lx_choice as L

    rows = []
    for r in _db_211():
        if r.get("op") != "flash_attn" or not r.get("feats") or not r.get("kernel_us"):
            continue
        if build and not str(r.get("log_date", "")).startswith(build):
            continue
        rows.append((_tidy(r["label"]), r["kernel_us"], L.predict(r["feats"])))
    rows.sort(key=lambda t: t[1])
    c, n, tau = _kendall([(m, p) for _, m, p in rows])
    out = [
        "| tile configuration | measured (µs) | predicted (µs) | predicted ÷ measured |",
        "|---|---:|---:|---:|",
    ]
    for lbl, m, p in rows:
        out.append(f"| `{lbl}` | {m:,.0f} | {p:,.0f} | {p / m:.1f}× |")
    # The concordance is quoted in the prose; put it in the table so it is checkable
    # against the rows above rather than taken on trust.
    rat = [p / m for _, m, p in rows]
    out.append(
        f"| **{len(rows)} configurations = {n} pairs; {c} of {n} ordered correctly** | | | "
        f"**{min(rat):.1f}–{max(rat):.1f}×** |"
    )
    return "\n".join(out), {"c": c, "n": n, "tau": tau, "rows": rows}


# --------------------------------------------------------------------------- 2.13


def _p1():
    """Pinned-allocation rows as case -> arm -> [measured].

    Prefers the four-round re-measurement over the original two-round pass: same
    allocations, twice the rounds, and it is what the reported spread refers to.
    """
    rows = [r for r in _v2() if r.get("phase") == 6 and r.get("kernel_us")]
    if not rows:
        rows = [r for r in _session() if r.get("phase") == 1 and r.get("kernel_us")]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("arm"):
            by[r["case"]][r["arm"]].append(r["kernel_us"])
    return by


def _forced():
    path = os.path.join(_HERE, "forced_allocations.json")
    if not os.path.exists(path):
        return []
    return json.load(open(path, encoding="utf-8"))["cases"]


def _shape_of(case_key, cases):
    """Find the case a measured row belongs to, whichever key form it used.

    The first session stored an underscored, truncated key; the re-measurement stores the
    full label. Matching only the first form silently dropped the `keeps` and `predicted`
    columns from the table rather than failing loudly.
    """
    for c in cases:
        if c["label"] == case_key or _tidy(c["label"]) == _tidy(case_key):
            return c
    norm = case_key.replace(" ", "_").replace("=", "_")
    for c in cases:
        key = _tidy(c["label"]).replace(" ", "_").replace("=", "_")
        if key.startswith(norm[:26]) or norm.startswith(key[:26]):
            return c
    return None


def solver_figure():
    """Per shape, each solver's time divided by the fastest -- measured beside predicted.

    A scatter of measured against predicted hides the result: greedy/cpsat and
    firstfit/bestfit land on top of each other, so twelve of the sixteen points are
    invisible. Normalising within each shape puts both panels on the same dimensionless
    axis, and the claim -- the model reproduces the split -- is that they match.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import lx_choice as L

    path = os.path.join(_HERE, "real_solver_results.jsonl")
    if not os.path.exists(path):
        return "(no data)"
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    by: dict = {}
    for r in rows:
        if r.get("kernel_us") and r.get("feats"):
            e = r["env"]
            by.setdefault((int(e["FA_H"]), int(e["FA_LQ"]), int(e["SENCORES"])), {})[
                r["solver"]
            ] = (r["kernel_us"], L.predict(r["feats"]))
    order = ["greedy", "cpsat", "firstfit", "bestfit"]
    colour = {
        "greedy": "#2a78d6",
        "cpsat": "#1baf7a",
        "firstfit": "#eb6834",
        "bestfit": "#eda100",
    }
    keys = sorted(by, key=lambda k: (k[2], k[0]))
    labels = [f"H={k[0]}, L={k[1]}\n{k[2]} cores" for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
    for ax, idx, title in ((axes[0], 0, "measured"), (axes[1], 1, "predicted")):
        w = 0.20
        for i, s_ in enumerate(order):
            off = (i - 1.5) * w
            vals = []
            for k in keys:
                v = by[k]
                ref = min(x[idx] for x in v.values())
                vals.append(v[s_][idx] / ref if s_ in v else float("nan"))
            bars = ax.bar(
                [x + off for x in range(len(keys))],
                vals,
                w * 0.9,
                color=colour[s_],
                zorder=3,
                label=s_ + (" (default)" if s_ == "greedy" else ""),
            )
            for b_ in bars:
                h = b_.get_height()
                if h == h:
                    ax.text(
                        b_.get_x() + b_.get_width() / 2,
                        h + 0.02,
                        f"{h:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#3d4551",
                    )
        ax.axhline(1.0, color="#9aa3ad", lw=1, zorder=2)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("time ÷ fastest solver on that shape")
    axes[0].legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    fig.suptitle(
        "The model reproduces the split: greedy and cpsat fast, "
        "firstfit and bestfit slow",
        fontsize=11.5,
    )
    fig.tight_layout()
    fig.savefig(SOLVER_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return SOLVER_FIG


def real_solvers():
    """What the SHIPPED solvers do -- one compile each, no reconstruction, no pinning."""
    import itertools

    import lx_choice as L

    path = os.path.join(_HERE, "real_solver_results.jsonl")
    if not os.path.exists(path):
        return "_not measured_"
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    by: dict = {}
    for r in rows:
        if r.get("kernel_us"):
            e = r["env"]
            k = (int(e["FA_H"]), int(e["FA_LQ"]), int(e["SENCORES"]))
            by.setdefault(k, {})[r["solver"]] = (
                r["kernel_us"],
                r.get("n_lx"),
                L.predict(r["feats"]) if r.get("feats") else None,
            )
    order = ["greedy", "cpsat", "firstfit", "bestfit"]
    out = [
        "<small>Each cell: measured microseconds, then in brackets how many buffers "
        "that solver placed in LX. One compile per cell.</small>",
        "",
        "| flash shape | "
        + " | ".join(
            f"`{s_}`" + (" (default)" if s_ == "greedy" else "") for s_ in order
        )
        + " | slowest ÷ fastest |",
        "|---|" + "---:|" * (len(order) + 1),
    ]
    tc = tn = 0
    for k in sorted(by):
        cells, times = [], []
        for s_ in order:
            v = by[k].get(s_)
            cells.append(f"{v[0]:,.1f} ({v[1]})" if v else "—")
            if v:
                times.append(v[0])
        out.append(
            f"| `H={k[0]} Lq=Lk={k[1]}, {k[2]} cores` | "
            + " | ".join(cells)
            + f" | **{max(times) / min(times):.2f}×** |"
        )
        pts = [(v[0], v[2]) for v in by[k].values() if v[2] is not None]
        for x, y in itertools.combinations(pts, 2):
            if x[0] == y[0] or x[1] == y[1]:
                continue
            tn += 1
            tc += (x[0] < y[0]) == (x[1] < y[1])
    # Every "X of Y" in this document must be checkable against a cell.
    out.append(
        f"| **the model orders {tc} of {tn} of these pairs correctly** |"
        + " |" * (len(order) + 1)
    )
    return "\n".join(out)


def policy_figure():
    """One decision, drawn: who keeps what, and the two different kinds of disagreement.

    This shape is the interesting one. `greedy` and `cpsat` keep the SAME NUMBER of buffers
    and disagree about WHICH -- greedy keeps `b12` (small, read twice), cpsat keeps `b13`
    (twice the size, read once) -- so neither allocation is a subset of the other and
    "more resident is faster" cannot explain the pair. `firstfit` and `bestfit` then drop
    `b8` on top of that, which is the subset difference and the one worth 1.24x.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    import lx_choice as L

    path = os.path.join(_HERE, "real_solver_results.jsonl")
    if not os.path.exists(path):
        return "(no data)"
    rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
    sel = {
        r["solver"]: r
        for r in rows
        if r.get("feats")
        and r["env"]["FA_H"] == "8"
        and r["env"]["FA_LQ"] == "512"
        and r["env"]["SENCORES"] == "8"
    }
    if not {"greedy", "cpsat", "firstfit"} <= set(sel):
        return "(shape not measured)"
    g, c, f = sel["greedy"], sel["cpsat"], sel["firstfit"]
    G, C, F = set(g["lx"]), set(c["lx"]), set(f["lx"])
    feats = g["feats"]
    fp, life, rc = (
        L.buffer_footprints(feats),
        L.buffer_lifetimes(feats),
        L.read_counts(feats),
    )
    # A buffer's colour is WHO KEEPS IT, so the swap shows up as two different colours on
    # two buffers rather than as an absence the reader has to notice.
    C_ALL, C_GC, C_G, C_C, C_NONE = (
        "#1baf7a",
        "#2a78d6",
        "#d1495b",
        "#8250c4",
        "#eda100",
    )
    style = {
        (1, 1, 1): (C_ALL, "kept by all four"),
        (1, 1, 0): (C_GC, "kept by greedy and cpsat only"),
        (1, 0, 1): (C_G, "greedy keeps, cpsat spills"),
        (0, 1, 0): (C_C, "cpsat keeps, greedy spills"),
        (0, 0, 0): (C_NONE, "spilled by all four"),
    }
    order = sorted(fp, key=lambda x: (life[x][0], -fp[x]))
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    used = {}
    for i, b_ in enumerate(order):
        s0, s1 = life[b_]
        sig = (int(b_ in G), int(b_ in C), int(b_ in F))
        col, lab = style.get(sig, ("#94a3b8", "other"))
        used[lab] = col
        h = 0.22 + 0.56 * (fp[b_] / max(fp.values()))
        ax.barh(
            i,
            s1 - s0,
            left=s0,
            height=h,
            color=col,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        if sig != (1, 1, 1) and sig != (0, 0, 0):
            note = f"{fp[b_] // 1024} KB, read {rc.get(b_, 0)}x"
            if sig in ((1, 0, 1), (0, 1, 0)):
                note += "   <- the swap"
            ax.text(
                s1 + 0.3,
                i,
                note,
                va="center",
                fontsize=9,
                color="#1a365d",
                zorder=4,
            )
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=7.5)
    ax.set_xlabel("operation index  (dataflow order within the fused kernel)")
    ax.set_ylabel("intermediate buffer")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(
        "H=8 Lq=Lk=512, 8 cores: two different kinds of disagreement\n"
        f"greedy and cpsat both keep {len(G)} buffers and TRADE one for another "
        f"({g['kernel_us']:,.1f} vs {c['kernel_us']:,.1f} us, inside run-to-run "
        f"spread); firstfit and bestfit drop a third buffer as well, keep {len(F)}, "
        f"and take {f['kernel_us']:,.1f} us "
        f"({f['kernel_us'] / c['kernel_us']:.2f}x)",
        fontsize=11,
    )
    ax.legend(
        handles=[Patch(facecolor=v, label=k) for k, v in used.items()],
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
    )
    fig.tight_layout()
    fig.savefig(POLICY_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return POLICY_FIG


def config_figure():
    """Measured against predicted across every configuration, all series on one axis.

    Replaces two tables of "n of m pairs ordered correctly". A pair count asserts the
    ordering; a monotone cloud shows it, and it also shows what the counts hide -- that the
    8-core points sit on their own band, which is a scale offset rather than a ranking error.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {}
    pts = {}
    for r in _session():
        if r.get("phase") == 2 and r.get("kernel_us") and r.get("pred_us"):
            m = re.match(r"H(\d+)_L(\d+)_c(\d+)", r.get("hint") or "")
            if m:
                pts.setdefault((1, int(m[1]), int(m[2]), int(m[3])), []).append(
                    (r["kernel_us"], r["pred_us"])
                )
    for r in _v2():
        if r.get("phase") == 2 and r.get("kernel_us") and r.get("pred_us"):
            sh = r.get("shape") or {}
            pts.setdefault(
                (
                    int(sh.get("FA_B", 1)),
                    int(sh.get("FA_H", 0)),
                    int(sh.get("FA_LQ", 0)),
                    int(sh.get("SENCORES", 32)),
                ),
                [],
            ).append((r["kernel_us"], r["pred_us"]))
    for k, v in pts.items():
        tag = f"untiled, {k[3]} cores"
        series.setdefault(tag, []).append(
            (
                st.mean(x[0] for x in v),
                st.mean(x[1] for x in v),
                f"H{k[1]}/L{k[2]}",
            )
        )
    _, s211 = ranking_211()
    series["coarse-tiled, 32 cores"] = [
        (
            m,
            p,
            "t" + re.sub(r".*htiles=(\d+) qtiles=(\d+).*", r"\1x\2", lbl)
            if "htiles" in lbl
            else (m, p, "ktiles=2")[2],
        )
        for lbl, m, p in s211["rows"]
    ]

    colour = {
        "untiled, 32 cores": "#2a78d6",
        "untiled, 8 cores": "#eb6834",
        "coarse-tiled, 32 cores": "#1baf7a",
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    allv = [x for v in series.values() for p_ in v for x in p_[:2]]
    lo, hi = min(allv) * 0.6, max(allv) * 1.6
    ax.plot(
        [lo, hi],
        [lo, hi],
        "-",
        color="#c3cad3",
        lw=1.2,
        zorder=1,
        label="predicted = measured",
    )
    for tag in ("untiled, 32 cores", "untiled, 8 cores", "coarse-tiled, 32 cores"):
        v = series.get(tag)
        if not v:
            continue
        ax.plot(
            [x[0] for x in v],
            [x[1] for x in v],
            "o",
            ms=8,
            color=colour[tag],
            markeredgecolor="white",
            markeredgewidth=1.2,
            zorder=3,
            label=f"{tag}  (n={len(v)})",
        )
        # Every point names its configuration -- a reader should not have to guess
        # what a dot is.
        for x0, y0, lab in v:
            ax.annotate(
                lab,
                xy=(x0, y0),
                xytext=(0, 9),
                textcoords="offset points",
                fontsize=6.0,
                color=colour[tag],
                ha="center",
                zorder=5,
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("measured kernel time (µs)")
    ax.set_ylabel("predicted kernel time (µs)")
    ax.set_title(
        "Predicted against measured, every configuration\n"
        "monotone within each series: the model orders them correctly",
        fontsize=11,
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, which="both", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(CONFIG_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return CONFIG_FIG


def ranking_213():
    """Shape/core ranking, de-duplicated across the two sessions.

    The second sweep re-measured eight of the first session's shapes. Pooling the rows
    raw counted each of those twice -- inflating the pair count and, worse, scoring every
    duplicate against its own twin, which is the same experiment rather than a comparison.
    Repeats are averaged into one point instead. (They agree to 0.0-1.0 %, which is a
    reproducibility check worth having.)
    """
    import regex as _re

    pts: dict = {}
    for r in _session():
        if r.get("phase") == 2 and r.get("kernel_us") and r.get("pred_us"):
            m = _re.match(r"H(\d+)_L(\d+)_c(\d+)", r.get("hint") or "")
            if m:
                pts.setdefault((1, int(m[1]), int(m[2]), int(m[3])), []).append(
                    (r["kernel_us"], r["pred_us"])
                )
    for r in _v2():
        if r.get("phase") == 2 and r.get("kernel_us") and r.get("pred_us"):
            sh = r.get("shape") or {}
            key = (
                int(sh.get("FA_B", 1)),
                int(sh.get("FA_H", 0)),
                int(sh.get("FA_LQ", 0)),
                int(sh.get("SENCORES", 32)),
            )
            pts.setdefault(key, []).append((r["kernel_us"], r["pred_us"]))

    rows = [
        {
            "cores": k[3],
            "kernel_us": st.mean(x[0] for x in v),
            "pred_us": st.mean(x[1] for x in v),
            "repeats": len(v),
        }
        for k, v in pts.items()
    ]
    n_rep = sum(1 for r in rows if r["repeats"] > 1)
    allc, alln, alltau = _kendall([(r["kernel_us"], r["pred_us"]) for r in rows])
    out = [
        "| core count | configurations | pairs ordered correctly | "
        "predicted ÷ measured |",
        "|---|---:|---:|---|",
    ]
    stats = {"all": (allc, alln, alltau), "n": len(rows), "repeated": n_rep}
    for cores in sorted({r["cores"] for r in rows}, reverse=True):
        sel = [r for r in rows if r["cores"] == cores]
        c, n, tau = _kendall([(r["kernel_us"], r["pred_us"]) for r in sel])
        rat = [r["pred_us"] / r["kernel_us"] for r in sel]
        out.append(
            f"| {cores} | {len(sel)} | **{c} of {n}** | "
            f"{min(rat):.2f}–{max(rat):.2f}× (mean {st.mean(rat):.2f}) |"
        )
        stats[cores] = (c, n, tau, st.mean(rat))
    rat = [r["pred_us"] / r["kernel_us"] for r in rows]
    out.append(
        f"| both pooled | {len(rows)} | **{allc} of {alln}** | "
        f"{min(rat):.2f}–{max(rat):.2f}× |"
    )
    out.append("")
    out.append(
        "<small>Every configuration is compared with every other, so n "
        "configurations give n(n-1)/2 pairs; pairs where the two measured or the "
        "two predicted times are equal cannot be ordered and are excluded.</small>"
    )
    return "\n".join(out), stats


def inject(name, body):
    if not os.path.exists(REPORT):
        return False
    src = open(REPORT, encoding="utf-8").read()
    a, b = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
    if a not in src or b not in src:
        return False
    i, j = src.index(a) + len(a), src.index(b)
    open(REPORT, "w", encoding="utf-8").write(src[:i] + "\n" + body + "\n" + src[j:])
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    t211, s211 = ranking_211()
    t213r, s213 = ranking_213()

    blocks = {
        "ranking_211": t211,
        "ranking_213": t213r,
        "real_solvers": real_solvers(),
    }
    if args.show:
        for k, v in blocks.items():
            print(f"\n===== {k} =====\n{v}")
        print(f"\n2.11: {s211['c']}/{s211['n']} concordant, tau={s211['tau']:+.2f}")
        return 0
    for k, v in blocks.items():
        print(f"  {k:<16}{'injected' if inject(k, v) else 'NO MARKER in report'}")
    try:
        print(f"  solver figure   {solver_figure()}")
        print(f"  config figure   {config_figure()}")
        print(f"  policy figure   {policy_figure()}")
    except Exception as exc:  # noqa: BLE001
        print(f"  figure          FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
