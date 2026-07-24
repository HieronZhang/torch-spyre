#!/usr/bin/env python3
# Copyright 2024-2025 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
"""Scratch analysis for cost-model category 2 (transport).

Loads notes/sweep_records.json, scores every transport record with the CURRENT
cost model (recomputed, not the stored pred_us), and prints a per-config table:
op | R x C | cores | cv% | meas us | pred us | err% | meas effBW | pat.

effBW = io_hbm_bytes / kernel_us  (GB/s). Not committed -- a scratch tool.
"""

import importlib.util
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _load_cost_model():
    path = os.path.join(_ROOT, "torch_spyre", "_inductor", "cost_model.py")
    spec = importlib.util.spec_from_file_location("cost_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cm = _load_cost_model()
TX = {"transpose", "transpose_outer", "cat0", "cat1"}


def main():
    recs = json.load(open(os.path.join(_HERE, "sweep_records.json")))["records"]
    rows = [
        r
        for r in recs
        if r.get("op") in TX
        and not r.get("failed")
        and r.get("kernel_us")
        and r.get("feats")
    ]
    print(f"# transport records with feats: {len(rows)}\n")
    hdr = (
        f"{'op':16} {'RxC':>13} {'cor':>3} {'cv%':>5} {'meas_us':>9} "
        f"{'pred_us':>9} {'err%':>7} {'effBW':>7} {'pat':>13}"
    )
    for op in ["transpose", "transpose_outer", "cat0", "cat1"]:
        sub = [r for r in rows if r["op"] == op]
        # dedup: keep the lowest-cv row per (rows,cols,cores)
        best = {}
        for r in sub:
            k = (r.get("rows"), r.get("cols"), r.get("cores"))
            cv = r.get("kernel_us_cv")
            cv = 999 if cv is None else cv
            if k not in best or cv < best[k][0]:
                best[k] = (cv, r)
        print(f"===== {op}  ({len(best)} distinct shapes) =====")
        print(hdr)
        for k in sorted(best, key=lambda k: (k[2] or 0, k[0] or 0, k[1] or 0)):
            cv, r = best[k]
            ops = [cm.op_from_dict(d) for d in r["feats"]]
            pred = cm.predict_ops(ops) / 1000.0
            meas = r["kernel_us"]
            err = 100.0 * (pred - meas) / meas
            io = r.get("io_hbm_bytes") or 0
            eff = io / (meas * 1e-6) / 1e9 if meas else 0.0
            pat = ops[0].hbm_pattern or "-"
            cvs = f"{cv:.1f}" if cv < 900 else "-"
            print(
                f"{op:16} {str(k[0]) + 'x' + str(k[1]):>13} {k[2]!s:>3} {cvs:>5} "
                f"{meas:9.1f} {pred:9.1f} {err:+7.1f} {eff:7.1f} {pat:>13}"
            )
        print()


if __name__ == "__main__":
    main()
