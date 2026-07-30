#!/usr/bin/env python3
# Copyright 2024-2025 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
"""Cat-4 (batched matmul device-layout effect) analysis, for adversarial review.

The BMMLAY sweep ran `bmm_layout` at matched (B,M,K,N,split) across all 4 combos of
each operand's device tile order in {[0,1,2],[1,0,2]}. This reproduces the evidence:
  (A) current-model error by layout combo -- the model predicts near the FAST layout.
  (B) slowdown default[0,1,2]/[0,1,2] vs fast[1,0,2]/[1,0,2], by batch B.
  (C) how the layout is detectable from the recorded device dims of input A.
  (D) do the "real" bmm ops (bmm_wd = plain forced-split bmm, bmm_k_tiling) use the
      default (slow) layout? -> is the -68% "bmm residual" just the default-layout penalty?
Not committed -- a scratch tool.
"""
import importlib.util
import json
import math
import os
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _cm():
    path = os.path.join(_ROOT, "torch_spyre", "_inductor", "cost_model.py")
    spec = importlib.util.spec_from_file_location("cost_model", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cm = _cm()


def _load():
    return json.load(open(os.path.join(_HERE, "sweep_records.json")))["records"]


def _err(r):
    pred = cm.predict_ops([cm.op_from_dict(d) for d in r["feats"]]) / 1000
    return (pred - r["kernel_us"]) / r["kernel_us"] * 100, pred


def main():
    recs = _load()

    # (A) model error by layout combo
    bylay = defaultdict(list)
    for r in recs:
        if r.get("op") != "bmm_layout" or r.get("failed") or not r.get("feats"):
            continue
        if not r.get("kernel_us"):
            continue
        bylay[(r.get("layout_a"), r.get("layout_b"))].append(_err(r)[0])
    print("=== (A) current-model error by layout combo (A/B) ===")
    for k in sorted(bylay):
        e = bylay[k]
        rms = math.sqrt(sum(x * x for x in e) / len(e))
        print(f"  A={k[0]} B={k[1]}: n={len(e)} mean={sum(e)/len(e):+.0f}% rms={rms:.0f}%")

    # (B) slowdown default vs fast by B
    g = defaultdict(dict)
    for r in recs:
        if r.get("op") != "bmm_layout" or r.get("failed") or not r.get("kernel_us"):
            continue
        g[(r.get("B"), r.get("M"), r.get("K"), r.get("N"))][
            (r.get("layout_a"), r.get("layout_b"))
        ] = r["kernel_us"]
    print("\n=== (B) slowdown default[0,1,2]^2 vs fast[1,0,2]^2, by B ===")
    byB = defaultdict(list)
    for key, c in sorted(g.items()):
        if ("0,1,2", "0,1,2") in c and ("1,0,2", "1,0,2") in c:
            ratio = c[("0,1,2", "0,1,2")] / c[("1,0,2", "1,0,2")]
            byB[key[0]].append(ratio)
            print(f"  B={key[0]} {key[1]}x{key[2]}x{key[3]}: slow/fast={ratio:.2f}x")
    print("  mean slow/fast by B:", {b: round(sum(v) / len(v), 2) for b, v in sorted(byB.items())})

    # (C) layout detectability from device dims of input A (arg0)
    print("\n=== (C) input-A device dims per layout (detection signature) ===")
    shown = set()
    for r in recs:
        if r.get("op") != "bmm_layout" or r.get("failed") or not r.get("feats"):
            continue
        la, lb = r.get("layout_a"), r.get("layout_b")
        if (la, lb) in shown:
            continue
        shown.add((la, lb))
        o = cm.op_from_dict(r["feats"][0])
        a = [x for x in o.args if x.role == "input"][0]
        print(f"  A={la} B={lb}: inputA dims={a.dims}  (B={r.get('B')})")

    # (D) do the plain bmm ops use the default (slow) layout?
    print("\n=== (D) 'real' bmm ops: input-A dims + model err (default layout => slow) ===")
    seen = set()
    for r in recs:
        op = r.get("op")
        if op not in ("bmm_wd", "bmm_k_tiling", "bmm_3d2d_k_tiling"):
            continue
        if r.get("failed") or not r.get("feats") or r.get("cores") != 32:
            continue
        key = (op, r.get("B"), r.get("M"), r.get("K"), r.get("N"))
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > 8:
            break
        o = cm.op_from_dict(r["feats"][0])
        a = [x for x in o.args if x.role == "input"][0]
        print(f"  {op} B={r.get('B')} {r.get('M')}x{r.get('K')}x{r.get('N')}: "
              f"err={_err(r)[0]:+.0f}% inputA_dims={a.dims}")


if __name__ == "__main__":
    main()
