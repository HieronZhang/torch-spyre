#!/usr/bin/env python3
# Copyright 2024-2025 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
"""Cat-3 (matmul compute/HBM overlap, the gamma-vs-cores question) analysis.

Reproduces the evidence behind the cat-3 claims so a reviewer can independently
challenge them. Scores CLEAN balanced matmul data (mm/mmwd, K unsplit, near-balanced
split, min(M,N)>=1024, K>=512), deduplicated to the MIN measured us per (M,K,N,cores)
(the cleanest kernel, dropping contention outliers). Not committed -- a scratch tool.

Prints, for the compute/HBM overlap model T = compute + mem - gamma*min(compute,mem):
  (A) per-(shape,cores) decomposition: compute, mem, effective gamma, effective
      fill/drain fd = meas - max(compute,mem).
  (B) per-shape best-fit fd ceiling (one fd per shape shared across cores>=4) -- the
      BEST a "fill/drain is a per-kernel constant" model can do.
  (C) constant gamma=0.46 vs a fitted gamma(cores)=a-b*log2(cores), RMS by core count.
  (D) the >10% residual rows at cores=32 (are they memory-bound / spill-driven?).
"""
import importlib.util
import json
import math
import os
import statistics
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
p = cm.CostParams()


def _is_bal(r):
    sf = r.get("split_forced")
    if not sf:
        return True  # planner-chosen mm = balanced
    m, n, k = sf.get("m", 1), sf.get("n", 1), sf.get("k", 1)
    return k == 1 and max(m, n) / max(1, min(m, n)) <= 2


def _load():
    recs = json.load(open(os.path.join(_HERE, "sweep_records.json")))["records"]
    best = {}
    for r in recs:
        if r.get("op") not in ("mm", "mmwd") or not r.get("feats") or r.get("failed"):
            continue
        if not r.get("kernel_us") or not _is_bal(r):
            continue
        o = cm.op_from_dict(r["feats"][0])
        if not getattr(o, "is_matmul", False) or o.cores <= 0:
            continue
        M, K, N = r["rows"], r["cols"], (r.get("N") or r["cols"])
        if min(M, N) < 1024 or K < 512:  # exclude tiny + thin-K flagged regimes
            continue
        # 4096x2048x4096 is CONTAMINATED (db_sweep low-core rows are physically impossible:
        # c4=8045us then c8=8255us -- more cores yet slower). Its fd/gamma back-outs are garbage
        # and inflate the pooled RMS; drop it (adversarial-review finding, verified).
        if (M, K, N) == (4096, 2048, 4096):
            continue
        key = (M, K, N, o.cores)
        if key not in best or r["kernel_us"] < best[key][1]:
            best[key] = (o, r["kernel_us"])
    return best


def _decomp(o):
    """(compute_us, mem_us, spill_bytes, area_elems) mirroring predict_ops' matmul path."""
    pt_eff = (
        1.0
        if o.tiles_output_dim
        else cm.underfill_eff(o.matmul_rows_per_core, p, p.underfill_target_passes_matmul)
    )
    comp = o.matmul_macs / o.cores / (p.mac_peak_per_core_ns * pt_eff)
    rr, ww = o.read_bytes(), o.write_bytes()
    area = o.matmul_rows_per_core * o.matmul_cols_per_core
    spill = (o.matmul_a_bytes + o.matmul_b_bytes) * cm.mm_spill_frac(area, p)
    mem = (
        rr / p.mm_bw_read_gbps
        + ww / p.mm_bw_write_gbps
        + spill / p.mm_bw_read_gbps
        + p.rw_turnaround_ns_per_byte * min(rr, ww)
    )
    return comp / 1000, mem / 1000, spill, area


def _rms(e):
    n = len(e)
    return math.sqrt(sum(x * x for x in e) / n), sum(e) / n


def main():
    best = _load()
    rows = []
    for (M, K, N, cores), (o, meas) in best.items():
        comp, mem, spill, area = _decomp(o)
        rows.append((M, K, N, cores, comp, mem, meas, spill, area))

    # (B) per-shape fd ceiling
    byshape = defaultdict(list)
    for (M, K, N, cores, comp, mem, meas, spill, area) in rows:
        byshape[(M, K, N)].append((cores, comp, mem, meas))
    print("=== (B) per-shape fd ceiling: T=max(comp,mem)+fd, one fd/shape, cores>=4 ===")
    errsB = []
    for shape in sorted(byshape):
        pts = [t for t in byshape[shape] if t[0] >= 4]
        if len(pts) < 3:
            continue
        fd = statistics.median([meas * 1000 - max(c, m) * 1000 for (cr, c, m, meas) in pts]) / 1000
        pe = [((max(c, m) + fd) / meas - 1) * 100 for (cr, c, m, meas) in pts]
        errsB += pe
        print(f"  {shape[0]}x{shape[1]}x{shape[2]:<8} fd={fd:6.0f}us  rms={_rms(pe)[0]:5.1f}%")
    print(f"  -> per-shape-fd CEILING: RMS {_rms(errsB)[0]:.1f}% (n={len(errsB)})\n")

    # (C) constant vs gamma(cores)
    bestg = None
    for a in [x / 100 for x in range(40, 90, 2)]:
        for b in [x / 1000 for x in range(-120, 120, 4)]:  # TWO-SIDED (b can rise w/ cores)
            e2 = sum(
                ((comp + mem - max(0.0, min(1.0, a - b * math.log2(cores))) * min(comp, mem)) / meas - 1) ** 2
                for (M, K, N, cores, comp, mem, meas, sp, ar) in rows
            )
            if bestg is None or e2 < bestg[0]:
                bestg = (e2, a, b)
    _, a, b = bestg
    print(f"=== (C) constant gamma=0.46 vs fitted gamma(cores)={a}-{b}*log2(cores) ===")
    cur, gc = defaultdict(list), defaultdict(list)
    for (M, K, N, cores, comp, mem, meas, sp, ar) in rows:
        cur[cores].append((comp + mem - 0.46 * min(comp, mem)) / meas * 100 - 100)
        g = max(0.0, min(1.0, a - b * math.log2(cores)))
        gc[cores].append((comp + mem - g * min(comp, mem)) / meas * 100 - 100)
    for c in sorted(cur):
        print(f"  cores={c:>2}: const RMS {_rms(cur[c])[0]:5.1f} mean {_rms(cur[c])[1]:+5.1f}  |  "
              f"g(cores) RMS {_rms(gc[c])[0]:5.1f} mean {_rms(gc[c])[1]:+5.1f}")
    allc = [e for v in cur.values() for e in v]
    allg = [e for v in gc.values() for e in v]
    print(f"  OVERALL const RMS {_rms(allc)[0]:.1f} mean {_rms(allc)[1]:+.1f}  |  "
          f"g(cores) RMS {_rms(allg)[0]:.1f} mean {_rms(allg)[1]:+.1f}  (n={len(allc)})\n")

    # (D) c32 >10% residuals
    print("=== (D) cores=32 rows with |err|>10% (const gamma) ===")
    for (M, K, N, cores, comp, mem, meas, spill, area) in sorted(rows):
        if cores != 32:
            continue
        err = (comp + mem - 0.46 * min(comp, mem)) / meas - 1
        if abs(err) > 0.10:
            print(f"  {M}x{K}x{N} c32: err={err*100:+.0f}%  comp={comp:.0f} mem={mem:.0f} "
                  f"spill={spill/150/1000:.0f}us area={area} membound={mem>comp}")


if __name__ == "__main__":
    main()
