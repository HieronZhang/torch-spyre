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

"""Does the cost model pick the right coarse tile size, without running anything?

A cost model earns its keep by RANKING plans, not by predicting absolute time. This
scores that directly: for every measured coarse-tiling ladder -- a fixed shape and core
count swept over tile counts -- it asks whether the model would have chosen the tile
count that actually measured fastest.

Three numbers, in order of how much they matter:

* **regret** -- how much slower the model's choice is than the true optimum. This is
  what a compiler using the model would actually pay. Zero regret means the model's
  pick was tied for best, even if it was not the same tile count.
* **correct sign** -- of the adjacent pairs where the model predicts a change at all,
  how many move the way the measurement moves.
* **flat pairs** -- pairs where the model predicts *exactly* no change. These are not
  wrong, they are blind: the model cannot distinguish the two plans.

Run:
    python3 notes/verify_direction.py             # summary + per-ladder table
    python3 notes/verify_direction.py --markdown  # markdown, for the write-up
    python3 notes/verify_direction.py --pairs     # every adjacent pair
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
RECORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_records.json")

# Every op whose coarse tiling we have swept. flash_attn is excluded: it is tiled, but
# it is mispredicted by 14-45x for a separate reason (see cost_model_directions.md), so
# including it would swamp every statistic here.
COARSE_OPS = (
    "matmul_row_tiling",
    "matmul_k_tiling",
    "mm_nested_m_k",
    "softmax_row_tiling",
    "softmax_noexp_row_tiling",
    "bmm_k_tiling",
    "bmm_3d2d_k_tiling",
    "bmm_nested_b_k",
)

MIN_TILE_COUNTS = 3  # a ladder needs at least three rungs to have a shape


def _shape(rec):
    """(M, K, N) from the label when present, else (rows, cols, None)."""
    label = str(rec.get("label") or "").replace(" ", "")
    m = re.search(r"(\d+)x(\d+)x(\d+)", label)
    if m:
        return int(m[1]), int(m[2]), int(m[3])
    return rec.get("rows"), rec.get("cols"), None


def collect_ladders():
    """Group in-scope coarse runs into ladders keyed by (op, shape, cores)."""
    with open(RECORDS, encoding="utf-8") as f:
        records = json.load(f)["records"]

    params = cm.CostParams()
    ladders: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for rec in records:
        if rec.get("op") not in COARSE_OPS or rec.get("failed"):
            continue
        if not rec.get("kernel_us") or not rec.get("feats"):
            continue
        if not em.in_scope(rec):
            continue
        try:
            pred = cm.predict_ops(cm.ops_from_json(json.dumps(rec["feats"])), params)
        except Exception:  # noqa: BLE001
            continue
        key = (rec["op"], _shape(rec), rec.get("cores"))
        tiles = int(rec.get("tiles") or 1)
        ladders[key][tiles].append((rec["kernel_us"], pred / 1000.0))

    out = []
    for key, by_tiles in ladders.items():
        if len(by_tiles) < MIN_TILE_COUNTS:
            continue
        rungs = []
        for tiles in sorted(by_tiles):
            runs = by_tiles[tiles]
            rungs.append(
                {
                    "tiles": tiles,
                    "n": len(runs),
                    # median over repeats: one noisy run must not decide a ranking
                    "meas": st.median(r[0] for r in runs),
                    "pred": st.median(r[1] for r in runs),
                }
            )
        out.append({"op": key[0], "shape": key[1], "cores": key[2], "rungs": rungs})
    # A shape may carry None (a 2-D op has no N), so sort on a None-free key.
    return sorted(
        out,
        key=lambda d: (
            d["op"],
            tuple(x or 0 for x in (d["shape"] or ())),
            d["cores"] or 0,
        ),
    )


def score(ladder):
    """Per-ladder verdict: chosen vs best tile count, regret, and pair directions."""
    rungs = ladder["rungs"]
    best_meas = min(rungs, key=lambda r: r["meas"])
    best_pred = min(rungs, key=lambda r: r["pred"])
    # What the model's choice actually costs, versus the best available.
    regret = (
        best_meas
        and (best_pred["meas"] - best_meas["meas"]) / best_meas["meas"] * 100.0
    )

    pairs = []
    for lo, hi in zip(rungs, rungs[1:]):
        d_meas = hi["meas"] - lo["meas"]
        d_pred = hi["pred"] - lo["pred"]
        pairs.append(
            {
                "from": lo["tiles"],
                "to": hi["tiles"],
                "meas_pct": 100.0 * d_meas / lo["meas"],
                "pred_pct": 100.0 * d_pred / lo["pred"],
                "flat": abs(d_pred) < 1e-9,
                "correct": (d_meas > 0) == (d_pred > 0)
                if abs(d_pred) >= 1e-9
                else None,
            }
        )
    # How many rungs TIE for the model's minimum. This matters more than it looks:
    # min() returns the first, i.e. the smallest tile count, so a model that cannot
    # discriminate still scores a "hit" whenever the smallest tile count happens to be
    # best. Counting ties is what stops a blind model from looking like a good one --
    # disable the re-read charge and 18 of 26 ladders tie, with most apparent hits
    # decided here rather than by the model.
    lowest = min(r["pred"] for r in rungs)
    n_tied = sum(1 for r in rungs if abs(r["pred"] - lowest) < 1e-9)

    return {
        "best_meas": best_meas["tiles"],
        "best_pred": best_pred["tiles"],
        "hit": best_meas["tiles"] == best_pred["tiles"],
        "regret_pct": regret,
        "pairs": pairs,
        "n_tied": n_tied,
        "tie_decided": n_tied > 1 and best_meas["tiles"] == best_pred["tiles"],
    }


def _fmt_shape(shape):
    return "x".join(str(x) for x in shape if x is not None)


def report(ladders, markdown=False, show_pairs=False):
    scored = [(lad, score(lad)) for lad in ladders]

    by_op: dict = collections.defaultdict(
        lambda: {
            "lad": 0,
            "hit": 0,
            "pairs": 0,
            "flat": 0,
            "ok": 0,
            "regret": [],
            "tied": 0,
        }
    )
    for lad, sc in scored:
        acc = by_op[lad["op"]]
        acc["lad"] += 1
        acc["hit"] += int(sc["hit"])
        acc["tied"] += int(sc["n_tied"] > 1)
        acc["regret"].append(sc["regret_pct"])
        for p in sc["pairs"]:
            acc["pairs"] += 1
            if p["flat"]:
                acc["flat"] += 1
            elif p["correct"]:
                acc["ok"] += 1

    bar = "|" if markdown else " "
    if markdown:
        print(
            "| operation | ladders | pairs | correct sign | flat | picks best |"
            " median regret | worst regret | tied |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    else:
        print(
            f"{'operation':>26}{'lad':>5}{'pairs':>7}{'sign':>9}{'flat':>6}"
            f"{'best':>7}{'med reg':>9}{'max reg':>9}{'tied':>6}"
        )

    tot = {
        "lad": 0,
        "hit": 0,
        "pairs": 0,
        "flat": 0,
        "ok": 0,
        "regret": [],
        "tied": 0,
    }
    for op in sorted(by_op, key=lambda o: -by_op[o]["lad"]):
        a = by_op[op]
        for k in ("lad", "hit", "pairs", "flat", "ok", "tied"):
            tot[k] += a[k]
        tot["regret"] += a["regret"]
        sign = f"{a['ok']}/{a['pairs'] - a['flat']}"
        best = f"{a['hit']}/{a['lad']}"
        med, mx = st.median(a["regret"]), max(a["regret"])
        if markdown:
            print(
                f"| `{op}` {bar} {a['lad']} | {a['pairs']} | {sign} | {a['flat']} |"
                f" {best} | {med:.1f} % | {mx:.1f} % | {a['tied']} |"
            )
        else:
            print(
                f"{op:>26}{a['lad']:>5}{a['pairs']:>7}{sign:>9}{a['flat']:>6}"
                f"{best:>7}{med:>8.1f}%{mx:>8.1f}%{a['tied']:>6}"
            )

    sign = f"{tot['ok']}/{tot['pairs'] - tot['flat']}"
    best = f"{tot['hit']}/{tot['lad']}"
    med, mx = st.median(tot["regret"]), max(tot["regret"])
    if markdown:
        print(
            f"| **all** | **{tot['lad']}** | **{tot['pairs']}** | **{sign}** |"
            f" **{tot['flat']}** | **{best}** | **{med:.1f} %** | **{mx:.1f} %** |"
            f" **{tot['tied']}** |"
        )
    else:
        print(
            f"{'ALL':>26}{tot['lad']:>5}{tot['pairs']:>7}{sign:>9}{tot['flat']:>6}"
            f"{best:>7}{med:>8.1f}%{mx:>8.1f}%{tot['tied']:>6}"
        )
    print()
    print("  'tied' counts ladders where two or more tile counts share the model's")
    print("  minimum. There the choice is made by the tie-break -- min() returns the")
    print("  SMALLEST tile count -- not by the model, so a blind model can score well")
    print("  by luck. Disable the re-read charge and this jumps from 6 to 18 of 26.")

    print()
    if markdown:
        print(
            "| operation | shape | cores | tile counts | best measured |"
            " best predicted | regret |"
        )
        print("|---|---|---:|---|---:|---:|---:|")
    else:
        print(
            f"{'operation':>26}{'shape':>20}{'cores':>6}  {'tiles':<26}"
            f"{'meas':>5}{'pred':>6}{'regret':>9}"
        )
    for lad, sc in sorted(scored, key=lambda t: -t[1]["regret_pct"]):
        tiles = ",".join(str(r["tiles"]) for r in lad["rungs"])
        flag = "" if sc["hit"] else ("  <-- " if not markdown else " ")
        if markdown:
            print(
                f"| `{lad['op']}` | {_fmt_shape(lad['shape'])} | {lad['cores']} |"
                f" {tiles} | {sc['best_meas']} | {sc['best_pred']} |"
                f" {sc['regret_pct']:.1f} % |"
            )
        else:
            print(
                f"{lad['op']:>26}{_fmt_shape(lad['shape']):>20}{lad['cores']:>6}  "
                f"{tiles:<26}{sc['best_meas']:>5}{sc['best_pred']:>6}"
                f"{sc['regret_pct']:>8.1f}%{flag}"
            )

    if show_pairs:
        print()
        print("adjacent pairs where the model moves the WRONG way:")
        for lad, sc in scored:
            for p in sc["pairs"]:
                if p["correct"] is False:
                    print(
                        f"  {lad['op']:>22} {_fmt_shape(lad['shape']):>18} "
                        f"c{lad['cores']:<3} t{p['from']}->t{p['to']}: "
                        f"measured {p['meas_pct']:+6.1f} %  "
                        f"predicted {p['pred_pct']:+6.1f} %"
                    )
        print()
        print("adjacent pairs where the model predicts NO change:")
        for lad, sc in scored:
            for p in sc["pairs"]:
                if p["flat"]:
                    print(
                        f"  {lad['op']:>22} {_fmt_shape(lad['shape']):>18} "
                        f"c{lad['cores']:<3} t{p['from']}->t{p['to']}: "
                        f"measured {p['meas_pct']:+6.1f} %  predicted   0.0 %"
                    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markdown", action="store_true", help="emit markdown tables")
    ap.add_argument(
        "--pairs", action="store_true", help="list wrong-way and flat pairs"
    )
    args = ap.parse_args()
    report(collect_ladders(), markdown=args.markdown, show_pairs=args.pairs)


if __name__ == "__main__":
    main()
