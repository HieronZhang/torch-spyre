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

"""Generate the end-of-Part data tables for the report, from the LIVE model.

Part IV has its own generator (``coarse_tables.py``); this covers Parts II and III, so
that no accuracy number in the write-up is ever hand-typed and none can go stale after a
coefficient changes.

    python3 tools/cost_model/part_tables.py 1          # Part I   (pointwise)
    python3 tools/cost_model/part_tables.py 2          # Part II  (other memory-bound ops)
    python3 tools/cost_model/part_tables.py 3          # Part III (matmul)

Measured time is ``kernel_us``; predictions are recomputed with ``cost_model.predict_ops``
on the recorded features. Rows are pooled by configuration and reported with their run
count, so a configuration measured many times does not outvote one measured twice.
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

cm = em.cm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from records import records_path  # noqa: E402


# Resolved lazily: importing this module must not exit with setup instructions.
def RECORDS():
    return records_path()


# Which ops belong to which Part. Names are the harness's; the report's prose uses plain
# language, so the table below carries a translation column.
PART_OPS = {
    1: [
        ("add", "add two tensors"),
        ("neg", "negate"),
        ("gelu", "apply a smooth activation"),
        ("mul", "multiply two tensors"),
        ("exp", "exponentiate"),
    ],
    2: [
        ("bcast", "broadcast a row across every row"),
        ("mulbcast", "multiply by a broadcast row"),
        ("bcastcol", "broadcast a column across every column"),
        ("copy", "add a constant"),
        ("write", "build a grid from a row and a column"),
        ("sumrow", "sum along each row"),
        ("amax", "maximum along each row"),
        ("mean", "mean along each row"),
        ("sumall", "sum every element"),
        ("read", "read a tensor"),
        ("sumcol", "sum down each column"),
        ("transpose", "transpose"),
        ("transpose_outer", "transpose the outer dimensions"),
        ("cat0", "join two tensors along the rows"),
        ("cat1", "join two tensors along the columns"),
    ],
    3: [
        ("mm", "plain matrix multiply"),
        ("mmwd", "plain matrix multiply, work division forced"),
        ("bmm_wd", "batched multiply, a weight matrix per batch"),
        ("bmm_wd_3d2d", "batched multiply, one shared weight matrix"),
        ("bmm_layout", "batched multiply, operand layouts varied"),
    ],
}


def _layout_name(lay):
    """Plain-language name for a device tile-order pair. ``[1,0,2]`` is the fast order
    (batch outermost), ``[0,1,2]`` the compiler default. ``-`` when the log predates
    layout recording, which is a real and separate population."""
    a, b = lay
    if not a or not b:
        return "-"
    nm = {"1,0,2": "fast", "0,1,2": "default"}
    return f"{nm.get(str(a), a)}/{nm.get(str(b), b)}"


def collect(part):
    with open(RECORDS(), encoding="utf-8") as f:
        recs = json.load(f)["records"]
    want = {o for o, _ in PART_OPS[part]}
    dp = cm.CostParams()
    by = collections.defaultdict(list)
    for r in recs:
        if r.get("op") not in want or r.get("failed") or not r.get("kernel_us"):
            continue
        if not em.in_scope(r):
            continue
        feats, _ = em.features_for(r, dp)
        if feats is None:
            continue
        try:
            pred = cm.predict_ops(feats, dp) / 1000.0
        except Exception:  # noqa: BLE001
            continue
        # For a matmul the SHAPE is (M,K,N), and the batch size and the forced work
        # division are distinguishing variables -- pooling across them would merge
        # measurements of genuinely different kernels.
        sp = r.get("split_forced") or {}
        lay = (r.get("layout_a"), r.get("layout_b"))
        key = (
            r["op"],
            r.get("M") or r.get("rows"),
            r.get("K") or r.get("cols"),
            r.get("N"),
            r.get("B"),
            (sp.get("m"), sp.get("n"), sp.get("b")),
            lay,
            r.get("cores"),
        )
        by[key].append((r["kernel_us"], pred))
    out = []
    for (op, R, C, N, B, sp, lay, cores), v in by.items():
        meas = st.median(x[0] for x in v)
        pred = st.median(x[1] for x in v)
        out.append(
            dict(
                op=op,
                rows=R,
                cols=C,
                N=N,
                B=B,
                split="×".join(str(x) for x in sp if x) or "-",
                layout=_layout_name(lay),
                cores=cores,
                n=len(v),
                meas=meas,
                pred=pred,
                err=(pred - meas) / meas * 100.0,
            )
        )
    return out


def _pow2(x):
    return bool(x) and (x & (x - 1)) == 0


def realistic(part, rows):
    """The rows a reader actually wants: full machine, ordinary power-of-2 shapes.

    The complete listing is always reproducible with ``--full``; printing all of it in
    the report buries the shape of the data under several hundred near-duplicate lines.
    """
    out = [
        d
        for d in rows
        if d["cores"] == 32
        and _pow2(d["rows"])
        and _pow2(d["cols"])
        and (d["rows"] or 0) >= 1024
        and (d["cols"] or 0) >= 1024
    ]
    if (
        part == 3
    ):  # matmul: the balanced division only; lopsided ones are §12's evidence
        out = [d for d in out if d["split"] in ("-", "4×8", "8×4")]
    # at most a handful per operation, spread across the shape range
    byop = collections.defaultdict(list)
    for d in out:
        byop[d["op"]].append(d)
    keep = []
    for op, v in byop.items():
        v.sort(
            key=lambda x: (
                (x["rows"] or 0) * (x["cols"] or 0),
                x["N"] or 0,
                x["B"] or 0,
            )
        )
        step = max(1, len(v) // 4)
        keep += v[::step][:4]
    return keep


def emit(part, rows, full=False):  # noqa: C901
    label = {1: "Part I", 2: "Part II", 3: "Part III"}[part]
    names = dict(PART_OPS[part])
    print(f"### {label} data — every run behind the terms above\n")
    print(
        "Predictions recomputed from the live model, not read from storage. "
        "Configurations measured more than once are pooled to their median and the run "
        "count is shown.\n"
    )
    print("| operation | n | RMS % | mean % | worst % | beyond ±10 % |")
    print("|---|---:|---:|---:|---:|---:|")
    byop = collections.defaultdict(list)
    for d in rows:
        byop[d["op"]].append(d)
    alle = []
    for op in sorted(byop, key=lambda o: -len(byop[o])):
        e = [d["err"] for d in byop[op]]
        alle += e
        rms = (sum(x * x for x in e) / len(e)) ** 0.5
        print(
            f"| {names.get(op, op)} | {len(e)} | {rms:.1f} | {st.mean(e):+.1f} | "
            f"{max(e, key=abs):+.1f} | {sum(1 for x in e if abs(x) > 10)} |"
        )
    rms = (sum(x * x for x in alle) / len(alle)) ** 0.5
    print(
        f"| **all** | **{len(alle)}** | **{rms:.1f}** | **{st.mean(alle):+.1f}** | "
        f"**{max(alle, key=abs):+.1f}** | **{sum(1 for x in alle if abs(x) > 10)}** |"
    )
    shown = rows if full else realistic(part, rows)
    if not full:
        print(
            f"\nA representative subset follows — full machine, ordinary shapes. "
            f"All {len(rows)} configurations: `python3 tools/cost_model/part_tables.py {part} --full`."
        )
    rows = shown
    wide = part == 3
    hdr = (
        "| operation | M | K | N | batches | layouts | division | cores | runs | "
        "measured µs | predicted µs | err % |"
        if wide
        else "| operation | rows | columns | cores | runs | measured µs | "
        "predicted µs | err % |"
    )
    print("\n‼ marks a run beyond ±10 %.\n")
    print(hdr)
    print("|---" * (12 if wide else 8) + "|")
    for d in sorted(
        rows,
        key=lambda x: (
            x["op"],
            x["rows"] or 0,
            x["cols"] or 0,
            x["N"] or 0,
            x["B"] or 0,
        ),
    ):
        flag = " ‼" if abs(d["err"]) > 10 else ""
        if wide:
            print(
                f"| `{d['op']}` | {d['rows']} | {d['cols']} | {d['N'] or '-'} | "
                f"{d['B'] or 1} | {d['layout']} | {d['split']} | {d['cores']} | "
                f"{d['n']} | "
                f"{d['meas']:.1f} | {d['pred']:.1f} | {d['err']:+.1f}{flag} |"
            )
        else:
            print(
                f"| `{d['op']}` | {d['rows']} | {d['cols']} | {d['cores']} | {d['n']} | "
                f"{d['meas']:.1f} | {d['pred']:.1f} | {d['err']:+.1f}{flag} |"
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("part", type=int, choices=(1, 2, 3))
    ap.add_argument("--full", action="store_true", help="list every configuration")
    a = ap.parse_args()
    emit(a.part, collect(a.part), full=a.full)


if __name__ == "__main__":
    main()
