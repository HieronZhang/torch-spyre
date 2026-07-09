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

"""Offline cost-model accuracy evaluator -- score a model version WITHOUT hardware.

The measured device time (``kernel_us``) is version-INDEPENDENT and already stored in
``notes/sweep_records.json``; only the model's PREDICTION changes when we edit the model.
So new accuracy is a pure computation: take each run's serialized feature vector, call
``cost_model.predict_ops`` with the current (or overridden) params, and compare to the
stored ``kernel_us``. No Spyre run, and the arithmetic is done by this program -- not by
hand -- so it is reliable at scale.

Feature source per record (in priority order):
  1. ``feats`` -- the exact serialized OpFeatures the harness now dumps (``MODEL FEATS``).
     Populated on every sweep run going forward (free -- one extra log line).
  2. ``io`` -- for OLDER rows without ``feats``, reconstruct OpFeatures from the stored
     device-I/O block. RECONSTRUCTION IS SELF-VALIDATED: we re-predict with the CURRENT
     default params and require it to reproduce the row's stored ``pred_us`` (which the
     current model produced) within a tolerance; rows that fail are reported and EXCLUDED,
     never silently trusted. Matmul is not reconstructed (split-derived features are
     unreliable from I/O alone) -> those rows need a ``feats`` re-run.

``cost_model.py`` imports only ``dataclasses``/``math`` (no torch), so this runs anywhere.

    python notes/eval_model.py                         # is_current rows, current params
    python notes/eval_model.py --all                   # every row (not just is_current)
    python notes/eval_model.py --category matmul --op softmax_row_tiling
    python notes/eval_model.py --params overlap_gamma=0.40,mac_peak_per_core_ns=1190
    python notes/eval_model.py --verify                # feature-fidelity vs stored pred_us
    python notes/eval_model.py --update                # write recomputed pred_us/err back
"""

import argparse
import importlib.util
import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _load_cost_model():
    """Import cost_model.py standalone (it has no torch dependency)."""
    path = os.path.join(_ROOT, "torch_spyre", "_inductor", "cost_model.py")
    spec = importlib.util.spec_from_file_location("cost_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cm = _load_cost_model()

# op -> reporting category
_CATEGORY = {}
for _c, _ops in {
    "pointwise": "neg copy gelu relu sigmoid exp mul add add3 add4".split(),
    "reduction": "sumrow sumcol sumall amax mean read".split(),
    "transport": "transpose transpose_outer cat0 cat1".split(),
    "broadcast": "bcast mulbcast bcastcol write".split(),
    "matmul": "mm mmwd".split(),
    "matmul_row": ["matmul_row_tiling"],
    "softmax": ["softmax_row_tiling", "softmax_noexp_row_tiling"],
    "coarse_reduction": "ctsum ctamax ctamin chain".split(),
}.items():
    for _o in _ops:
        _CATEGORY[_o] = _c

# hbm_pattern the extractor assigns per bench op -- used ONLY for I/O reconstruction of
# old rows, and every use is self-validated against the row's stored pred_us.
_HBM_PATTERN_BY_OP = {
    "transpose": "restickify",
    "cat0": "stick_scatter",
    "sumcol": "reduce_outer",
}
_NO_RECONSTRUCT = {"mm", "mmwd", "matmul_row_tiling"}  # need `feats`


def category(op):
    return _CATEGORY.get(op, "other")


def make_params(overrides):
    """CostParams with ``k=v`` overrides (values parsed as float)."""
    p = cm.CostParams()
    for kv in overrides:
        k, _, v = kv.partition("=")
        k = k.strip()
        if not hasattr(p, k):
            raise SystemExit(f"unknown CostParams field: {k!r}")
        setattr(p, k, float(v))
    return p


def reconstruct_from_io(rec):
    """Best-effort OpFeatures for an old row from its stored device-I/O block. Returns a
    list[OpFeatures] or None if unsupported (matmul / no I/O). SELF-VALIDATED by the
    caller against the stored pred_us before it is trusted."""
    io = rec.get("io")
    op_name = rec.get("op")
    if not io or op_name in _NO_RECONSTRUCT:
        return None
    cores = int(rec.get("cores") or 1)
    tiles = int(rec.get("tiles") or 0)
    tiled = tiles >= 2
    rows = rec.get("rows")
    tile_rpc = rows / (cores * tiles) if (tiled and rows and cores) else 0.0
    pat = _HBM_PATTERN_BY_OP.get(op_name, "")
    ops = []
    for o in io:
        args, out_elems = [], 0
        for t in o.get("tensors", []):
            elems = int(t["elems"])
            args.append(
                cm.ArgTraffic(
                    name=t["name"],
                    role=t["role"],
                    mem=t["mem"].lower(),
                    elems=elems,
                    broadcast="broadcast" in (t.get("flags") or ""),
                    loop_factor=1,
                )
            )
            if t["role"] == "output":
                out_elems = elems
        is_red = o.get("kind") == "reduction"
        red_cores = (
            max(1, cores // out_elems) if (is_red and 0 < out_elems < cores) else 1
        )
        ops.append(
            cm.OpFeatures(
                name=o["op"],
                is_reduction=is_red,
                out_elems=out_elems,
                cores=cores,
                dtype_bytes=2,
                args=args,
                reduction_cores=red_cores,
                loop_trip=tiles if tiled else 1,
                tiles_output_dim=tiled,
                tile_rows_per_core=tile_rpc,
                hbm_pattern="" if is_red and pat != "reduce_outer" else pat,
            )
        )
    return ops


def features_for(rec, default_params, tol=0.02):
    """Return (ops, source) for a record, or (None, reason). ``source`` is 'feats' or
    'io(verified)'. I/O reconstruction is accepted only if predicting it with the DEFAULT
    params reproduces the stored pred_us within ``tol`` (the stored pred came from the
    current model, so a faithful reconstruction must match it)."""
    if rec.get("feats"):
        return [cm.op_from_dict(d) for d in rec["feats"]], "feats"
    ops = reconstruct_from_io(rec)
    if ops is None:
        return None, "no-feats"
    stored = rec.get("pred_us")
    if not stored:
        return None, "no-oracle"
    got = cm.predict_ops(ops, default_params) / 1000.0
    if abs(got - stored) / stored > tol:
        return None, f"io-mismatch({got:.1f}vs{stored:.1f})"
    return ops, "io(verified)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", default=os.path.join(_HERE, "sweep_records.json"))
    ap.add_argument("--all", action="store_true", help="all rows (default: is_current)")
    ap.add_argument("--category", default="", help="filter to one reporting category")
    ap.add_argument("--op", default="", help="filter to one op")
    ap.add_argument(
        "--params", default="", help="comma list of CostParams k=v overrides"
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="report feature-fidelity (pred vs stored pred_us) and exit",
    )
    ap.add_argument(
        "--update",
        action="store_true",
        help="write recomputed pred_us/err_pct back to the records",
    )
    ap.add_argument(
        "--list-worst", type=int, default=0, help="print the N worst |err| rows"
    )
    args = ap.parse_args()

    with open(args.records, encoding="utf-8") as f:
        records = json.load(f)["records"]
    default_params = cm.CostParams()
    overrides = [s for s in args.params.split(",") if s.strip()]
    params = make_params(overrides)

    rows = [r for r in records if not r.get("failed") and r.get("kernel_us")]
    if not args.all:
        rows = [r for r in rows if r.get("is_current")]
    if args.category:
        rows = [r for r in rows if category(r.get("op")) == args.category]
    if args.op:
        rows = [r for r in rows if r.get("op") == args.op]

    evaluated, skipped = [], {}
    for r in rows:
        ops, src = features_for(r, default_params)
        if ops is None:
            skipped[src] = skipped.get(src, 0) + 1
            continue
        pred = cm.predict_ops(ops, params) / 1000.0
        meas = r["kernel_us"]
        err = (pred - meas) / meas * 100.0
        evaluated.append((r, pred, meas, err, src))

    if (
        args.verify
    ):  # fidelity check: default-param pred should reproduce stored pred_us
        print(f"feature fidelity ({len(evaluated)} rows):")
        by_src = {}
        for r, _, _, _, src in evaluated:
            by_src[src] = by_src.get(src, 0) + 1
        for s, n in sorted(by_src.items()):
            print(f"  {n:4d}  {s}")
        for s, n in sorted(skipped.items()):
            print(f"  {n:4d}  SKIPPED: {s}")
        # for feats rows, also confirm pred(default) == stored pred_us
        drift = []
        for r, _, _, _, src in evaluated:
            if src != "feats" or not r.get("pred_us"):
                continue
            pd = cm.predict_ops(
                [cm.op_from_dict(d) for d in r["feats"]], default_params
            )
            pd /= 1000.0
            if abs(pd - r["pred_us"]) / r["pred_us"] > 0.02:
                drift.append((r["op"], round(pd, 1), r["pred_us"]))
        if drift:
            print(
                f"  WARNING: {len(drift)} feats rows drift from stored pred_us "
                f"(model changed since the log): e.g. {drift[:3]}"
            )
        return

    if overrides:
        print(f"params overrides: {overrides}")
    print(
        f"scored {len(evaluated)} rows"
        + (f"  (skipped: {dict(skipped)})" if skipped else "")
    )
    print(f"\n{'category':>16} {'n':>4} {'RMS%':>7} {'mean%':>7} {'range':>14}")
    cats = {}
    for r, pred, meas, err, src in evaluated:
        cats.setdefault(category(r["op"]), []).append(err)
    allerr = []
    for c in sorted(cats):
        es = cats[c]
        allerr += es
        rms = math.sqrt(sum(e * e for e in es) / len(es))
        print(
            f"{c:>16} {len(es):>4} {rms:>7.1f} {sum(es) / len(es):>+7.1f}"
            f"  [{min(es):+.0f}..{max(es):+.0f}]"
        )
    if allerr:
        rms = math.sqrt(sum(e * e for e in allerr) / len(allerr))
        print(
            f"{'OVERALL':>16} {len(allerr):>4} {rms:>7.1f} "
            f"{sum(allerr) / len(allerr):>+7.1f}  "
            f"[{min(allerr):+.0f}..{max(allerr):+.0f}]"
        )

    if args.list_worst:
        print(f"\nworst {args.list_worst} |err|:")
        for r, pred, meas, err, src in sorted(evaluated, key=lambda x: -abs(x[3]))[
            : args.list_worst
        ]:
            print(
                f"  {err:+6.1f}%  {r['op']:20} "
                f"[{r.get('rows')},{r.get('cols')}] "
                f"pred={pred:.1f} meas={meas:.1f} ({src})"
            )

    if args.update:
        idx = {r["id"]: (pred, err) for r, pred, _, err, _ in evaluated}
        for r in records:
            if r["id"] in idx:
                pred, err = idx[r["id"]]
                r["pred_us"], r["err_pct"] = round(pred, 3), round(err, 1)
        with open(args.records, "w", encoding="utf-8") as f:
            json.dump({"records": records}, f, indent=2)
        print(f"\nupdated pred_us/err_pct for {len(idx)} rows -> {args.records}")


if __name__ == "__main__":
    main()
