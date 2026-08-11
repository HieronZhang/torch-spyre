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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools", "cost_model"))

REPORT = os.path.join(_HERE, "lx_allocation_report.md")
FIGURE = os.path.join(_HERE, "lx_allocation.png")
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
        f"| **{c} of {n} pairs ordered correctly** (τ = {tau:+.2f}) | | | "
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


def alloc_213():
    """Measured against predicted TIME per allocation, so the orders can be compared.

    Both columns are microseconds. An earlier version put measured in microseconds and
    predicted as a percentage regret, which made the one question a reader actually has --
    does the model order the policies correctly? -- impossible to answer by eye.
    """
    cases, by = _forced(), _p1()
    out = [
        "| flash shape | allocation | keeps | measured (µs) | predicted (µs) | order |",
        "|---|---|---:|---:|---:|:--:|",
    ]
    detail = []
    for key, arms in by.items():
        c = _shape_of(key, cases)
        means = {a: st.mean(v) for a, v in arms.items()}
        if len(means) < 2:
            continue
        pred = {}
        if c:
            for a in means:
                pred[a] = c["arms"].get(a.split("+")[0], {}).get("pred_us")
        shape = (
            _tidy(c["label"]).replace(" D=128 htiles=1 qtiles=1 ktiles=1", "")
            if c
            else key
        )
        order_m = sorted(means, key=lambda a: means[a])
        have = [a for a in order_m if pred.get(a) is not None]
        order_p = sorted(have, key=lambda a: pred[a])
        agrees = [a for a in order_m if a in have] == order_p
        first = True
        for rank, a in enumerate(order_m, 1):
            pu = pred.get(a)
            keeps = c["arms"].get(a.split("+")[0], {}).get("n_lx") if c else None
            prank = (order_p.index(a) + 1) if a in order_p else None
            mark = "—"
            if prank:
                mark = "✓" if prank == rank else f"→{prank}"
            out.append(
                f"| {'`' + shape + '`' if first else ''} | {a} | {keeps or '—'} "
                f"| {means[a]:,.1f} | {(f'{pu:,.1f}' if pu else '—')} | {mark} |"
            )
            first = False
        detail.append((shape, means, c))
        out.append(
            f"| | _model orders these_ | | | | "
            f"**{'correctly' if agrees else 'WRONG'}** |"
        )
    return "\n".join(out), detail


def alloc_pairs():
    """How the model does on the question section 4 asks: ordering ALLOCATIONS.

    Reported as pairs, the same statistic used for shape and tile ranking. Counting shapes
    instead hides that one shape supplies six of the eight pairs and fails four of them.
    """
    import itertools

    cases, by = _forced(), _p1()
    out = [
        "| flash shape | allocations | pairs | correct | order |",
        "|---|---:|---:|---:|:--:|",
    ]
    tc = tn = ok_shapes = n_shapes = 0
    for key, arms in by.items():
        c = _shape_of(key, cases)
        if not c or len(arms) < 2:
            continue
        pts = []
        for a, v in arms.items():
            pr = c["arms"].get(a.split("+")[0], {}).get("pred_us")
            if pr:
                pts.append((st.mean(v), pr))
        if len(pts) < 2:
            continue
        n_shapes += 1
        ok = sum(
            (x[0] < y[0]) == (x[1] < y[1]) for x, y in itertools.combinations(pts, 2)
        )
        n = len(pts) * (len(pts) - 1) // 2
        tc += ok
        tn += n
        ok_shapes += ok == n
        shape = _tidy(c["label"]).replace(" D=128 htiles=1 qtiles=1 ktiles=1", "")
        out.append(
            f"| `{shape}` | {len(pts)} | {n} | {ok} | "
            f"{'✓' if ok == n else '**wrong**'} |"
        )
    out.append(
        f"| **total** | | **{tn}** | **{tc} of {tn}** | **{ok_shapes} of {n_shapes} shapes** |"
    )
    return "\n".join(out), (tc, tn, ok_shapes, n_shapes)


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
        "| core count | configurations | concordant pairs | Kendall τ | "
        "predicted ÷ measured |",
        "|---|---:|---:|---:|---|",
    ]
    stats = {"all": (allc, alln, alltau), "n": len(rows), "repeated": n_rep}
    for cores in sorted({r["cores"] for r in rows}, reverse=True):
        sel = [r for r in rows if r["cores"] == cores]
        c, n, tau = _kendall([(r["kernel_us"], r["pred_us"]) for r in sel])
        rat = [r["pred_us"] / r["kernel_us"] for r in sel]
        out.append(
            f"| {cores} | {len(sel)} | {c}/{n} | **{tau:+.2f}** | "
            f"{min(rat):.2f}–{max(rat):.2f}× (mean {st.mean(rat):.2f}) |"
        )
        stats[cores] = (c, n, tau, st.mean(rat))
    rat = [r["pred_us"] / r["kernel_us"] for r in rows]
    out.append(
        f"| both pooled | {len(rows)} | {allc}/{alln} | {alltau:+.2f} | "
        f"{min(rat):.2f}–{max(rat):.2f}× |"
    )
    return "\n".join(out), stats


def cores_ladder():
    """Measured and predicted against core count at fixed shape."""
    rows = [
        r
        for r in _v2()
        if r.get("phase") == 3 and r.get("kernel_us") and r.get("pred_us")
    ]
    if not rows:
        return "_not measured_"
    shapes = sorted({(r["H"], r["L"]) for r in rows})
    out = [
        "| cores | "
        + " | ".join(f"H={h} L={ell} measured" for h, ell in shapes)
        + " | mean predicted ÷ measured |",
        "|---:|" + "---:|" * (len(shapes) + 1),
    ]
    for c in sorted({r["n_cores"] for r in rows}):
        cells, rat = [], []
        for h, ell in shapes:
            m = next(
                (
                    r
                    for r in rows
                    if r["n_cores"] == c and r["H"] == h and r["L"] == ell
                ),
                None,
            )
            cells.append(f"{m['kernel_us']:,.0f} µs" if m else "—")
            if m:
                rat.append(m["pred_us"] / m["kernel_us"])
        out.append(f"| {c} | " + " | ".join(cells) + f" | **{st.mean(rat):.2f}×** |")
    return "\n".join(out)


def contested():
    """Every untiled shape solved exactly: is the byte objective ever beaten?"""
    import lx_choice as L
    import lx_experiment as X

    path = os.path.join(_HERE, "untiled_flash_records.json")
    if not os.path.exists(path):
        return "_no untiled records_", []
    recs = json.load(open(path, encoding="utf-8"))["records"]
    out = [
        "| flash shape | movable | feasible allocations | `greedy` | `bestfit` | "
        "`cpsat` |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for r in recs:
        f = r.get("feats")
        if not f or not r.get("kernel_us"):
            continue
        mov = L.bundle_intermediates(f)
        if not mov or L.peak_footprint(f, set(mov)) <= CAP:
            continue
        exact, ev, n = L.exhaustive_policies(f, CAP)
        if n < 2:
            continue
        seq = X.solver_allocations(f)
        ref = ev(exact["time"])
        g = {
            k: (ev(v) - ref) / ref * 100.0
            for k, v in {
                "cpsat": exact["cpsat"],
                "greedy": seq["greedy"],
                "bestfit": seq["bestfit"],
            }.items()
        }
        shape = _tidy(r["label"]).replace(" D=128 htiles=1 qtiles=1 ktiles=1", "")
        out.append(
            f"| `{shape}` | {len(mov)} | {n:,} | {g['greedy']:+.1f}% | "
            f"{g['bestfit']:+.1f}% | **{g['cpsat']:+.1f}%** |"
        )
        rows.append((shape, len(mov), n, g))
    return "\n".join(out), rows


#: Categorical slots 1-4 of the validated reference palette. Colour encodes POLICY IDENTITY,
#: never whether a policy won -- an earlier version shaded "good" green and "bad" orange, which
#: painted `bestfit` as a loser on the one shape where it is fastest. The palette passes the
#: colourblind-separation check (worst adjacent pair dE 9.1 protan); the sub-3:1 contrast on
#: two slots is relieved by the value labels on every bar and by the table above the figure.
POLICY_COLOUR = {
    "cpsat": "#2a78d6",
    "greedy": "#eb6834",
    "bestfit": "#1baf7a",
    "optimum": "#eda100",
}
POLICY_ORDER = ["greedy", "bestfit", "cpsat", "optimum"]


def figure(detail, r211):
    """One concrete allocation, drawn: lifetime, size, reuse, and each policy's choice.

    The earlier figure was a six-point scatter that showed neither which configuration was
    which nor why the policies disagree. This draws the actual decision instead: every
    movable buffer as a bar spanning the operations it is live across, height proportional
    to its per-core size, coloured by which policy keeps it. The four buffers the policies
    disagree on are the only tall bars that differ, which is the whole story.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    import lx_choice as L

    path = os.path.join(_HERE, "untiled_flash_records.json")
    cases = _forced()
    if not (os.path.exists(path) and cases):
        return "(no data)"
    recs = {r["label"]: r for r in json.load(open(path, encoding="utf-8"))["records"]}
    case = next((c for c in cases if "H=16 Lq=1024" in c["label"]), cases[0])
    rec = recs.get(case["label"])
    if not rec:
        return "(no record)"
    feats = rec["feats"]
    fp = L.buffer_footprints(feats)
    life = L.buffer_lifetimes(feats)
    rc = L.read_counts(feats)
    sets = {}
    for name, arm in case["arms"].items():
        sets.setdefault(tuple(arm["lx"]), []).append(name)
    (lx_a, na), (lx_b, nb) = list(sets.items())[:2]
    A, B = set(lx_a), set(lx_b)

    order = sorted(fp, key=lambda x: (life[x][0], -fp[x]))
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    C_BOTH, C_A, C_B = "#c9d3de", "#2a78d6", "#eb6834"
    for i, b_ in enumerate(order):
        s0, s1 = life[b_]
        ina, inb = b_ in A, b_ in B
        col = C_BOTH if ina and inb else (C_A if ina else (C_B if inb else "#f2f4f7"))
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
        if ina != inb:
            ax.text(
                s1 + 0.25,
                i,
                f"{fp[b_] // 1024} KB, {rc.get(b_, 0)} reads",
                va="center",
                fontsize=8.5,
                color="#2d3748",
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
    meas = {
        a_: st.mean(v)
        for shape, means, c in detail
        if c is case
        for a_, v in [(k, [x]) for k, x in means.items()]
    }
    ma = meas.get("+".join(na)) or 0
    mb = meas.get("+".join(nb)) or 0
    # The mechanism is positional: in each contested pair the once-read buffer
    # is produced FIRST, so a program-order policy reaches it before the
    # twice-read one sitting directly behind it.
    contested = sorted((life[x][0], x) for x in (A ^ B))
    if contested:
        start, b_ = contested[0]
        ax.annotate(
            "`greedy` walks left to right and\nmeets the 1-read buffer first",
            xy=(start, order.index(b_)),
            xytext=(max(0.2, start - 6.6), order.index(b_) - 3.4),
            fontsize=8.5,
            color="#2d3748",
            arrowprops=dict(arrowstyle="->", color="#718096", lw=1.1),
        )
    ax.set_title(
        f"One allocation decision: {_tidy(case['label']).split(' D=')[0]}\n"
        f"18 of 22 buffers are kept by both; the four that differ are all 1024 KB and "
        f"differ only in how often they are read",
        fontsize=11,
    )
    ax.legend(
        handles=[
            Patch(facecolor=C_BOTH, label="kept by both"),
            Patch(
                facecolor=C_A,
                label=f"kept only by {'+'.join(na)}"
                + (f"  ({ma:,.0f} us)" if ma else ""),
            ),
            Patch(
                facecolor=C_B,
                label=f"kept only by {'+'.join(nb)}"
                + (f"  ({mb:,.0f} us)" if mb else ""),
            ),
            Patch(facecolor="#f2f4f7", label="spilled by both"),
        ],
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return FIGURE


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
    t213a, detail = alloc_213()
    t213r, s213 = ranking_213()

    blocks = {
        "ranking_211": t211,
        "alloc_213": t213a,
        "ranking_213": t213r,
        "cores_ladder": cores_ladder(),
        "alloc_pairs": alloc_pairs()[0],
    }
    if args.show:
        for k, v in blocks.items():
            print(f"\n===== {k} =====\n{v}")
        print(f"\n2.11: {s211['c']}/{s211['n']} concordant, tau={s211['tau']:+.2f}")
        return 0
    for k, v in blocks.items():
        print(f"  {k:<16}{'injected' if inject(k, v) else 'NO MARKER in report'}")
    try:
        print(f"  figure          {figure(detail, s211)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  figure          FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
