#!/usr/bin/env python3
# Copyright 2024-2025 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
"""MMCORE forced-core isolation: characterize compute/HBM OVERLAP empirically.

For a FIXED matmul shape swept over cores {1,2,4,8,16,32} (balanced split ladder),
decompose the model into compute_us and mem_us EXACTLY as predict_ops does, then
measure the "hidden" overlap = (compute + mem) - measured. This is the amount the
naive additive (zero-overlap) model over-predicts, i.e. the real pipelining.

hidden = compute + mem - meas          (us)
gamma_eff = hidden / min(compute, mem)  (the overlap the constant gamma=0.46 tries to hit)

Answers:
 (1) At 1 core (no split) is it purely compute-bound (overlap invisible)? At what core
     count does compute ~= mem so overlap becomes visible?
 (2) Does the hidden fraction track the per-core PASS structure -- #M-passes, or the
     ratio per-pass-compute / per-pass-load -- rather than cores per se?
 (3) (companion script) batched vs non-batched at matched MACs.
"""
import importlib.util
import json
import math
import os
import statistics
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_PT_ROWS = 8  # PT block rows per corelet (hardware pass granularity)


def _cm():
    path = os.path.join(_ROOT, "torch_spyre", "_inductor", "cost_model.py")
    spec = importlib.util.spec_from_file_location("cost_model", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cm = _cm()
p = cm.CostParams()

# TRUE single-core compute rate, backed out from cores=1 (NO split) MMCORE rows where the
# kernel is purely compute-bound (HBM floor is 2-5% of measured). Median 1041, spread
# 1037-1073, stdev ~16 across 37 compute-dominant rows in 3 independent sweep sections.
# The SHIPPED model uses 1140 (datasheet 1536); 1140 is ~9% high and is the reason a
# constant gamma is needed -- it corrects the peak error, entangled with real overlap.
MAC_PEAK_TRUE = 1046.0


def _fill_cores1(o, M, K, N):
    """cores=1 records have M/m,N/n,|A|,|B| unpopulated (0). Reconstruct the physical
    values: unsplit -> M/m=M, N/n=N, |A|=M*K, |B|=K*N."""
    if o.matmul_rows_per_core == 0.0 and o.cores == 1:
        o.matmul_rows_per_core = float(M)
        o.matmul_cols_per_core = float(N)
        o.matmul_a_bytes = M * K * o.dtype_bytes
        o.matmul_b_bytes = K * N * o.dtype_bytes
        o.matmul_m_split = 1
        o.matmul_n_split = 1
    return o


def _decomp(o):
    """(compute_us, mem_us, spill_us, area) exactly mirroring predict_ops' matmul path."""
    pt_eff = (
        1.0
        if o.tiles_output_dim
        else cm.underfill_eff(
            o.matmul_rows_per_core, p, p.underfill_target_passes_matmul
        )
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
    return comp / 1000.0, mem / 1000.0, spill / p.mm_bw_read_gbps / 1000.0, area


def load_mmcore():
    recs = json.load(open(os.path.join(_HERE, "sweep_records.json")))["records"]
    best = {}
    for r in recs:
        if (r.get("section") or "")[:6] != "MMCORE":
            continue
        if r.get("op") not in ("mm", "mmwd") or not r.get("feats") or r.get("failed"):
            continue
        if not r.get("kernel_us"):
            continue
        M, K, N = r["rows"], r["cols"], (r.get("N") or r["cols"])
        # EXCLUDE the contaminated shape (physically impossible c4->c8) and tiny/thin-K
        if (M, K, N) == (4096, 2048, 4096):
            continue
        if min(M, N) < 1024 or K < 512:
            continue
        o = cm.op_from_dict(r["feats"][0])
        if not getattr(o, "is_matmul", False) or o.cores <= 0:
            continue
        o = _fill_cores1(o, M, K, N)
        key = (M, K, N, o.cores)
        # dedup to MIN kernel_us
        if key not in best or r["kernel_us"] < best[key][1]:
            best[key] = (o, r["kernel_us"], M, K, N)
    return best


def main():
    best = load_mmcore()
    rows = []
    for (M, K, N, cores), (o, meas, M2, K2, N2) in best.items():
        comp, mem, spill_us, area = _decomp(o)
        add = comp + mem  # zero-overlap additive
        hidden = add - meas
        mn = min(comp, mem)
        gamma_eff = hidden / mn if mn > 0 else float("nan")
        # PASS structure
        Mpc = o.matmul_rows_per_core  # M/m per core
        Npc = o.matmul_cols_per_core  # N/n per core
        m_passes = Mpc / _PT_ROWS  # #M-passes per core (streamed PT tiles)
        # per-pass compute vs per-pass load (proxy). Per M-pass a core does
        #   _PT_ROWS * Npc * K MACs  and loads ~ _PT_ROWS*K (A slab) + K*Npc (B slab).
        # We report the ratio compute/mem directly and #passes.
        rows.append(
            dict(M=M, K=K, N=N, cores=cores, comp=comp, mem=mem, meas=meas,
                 add=add, hidden=hidden, gamma_eff=gamma_eff, ratio=comp / mem,
                 Mpc=Mpc, Npc=Npc, m_passes=m_passes, spill_us=spill_us, area=area,
                 m_split=o.matmul_m_split, n_split=o.matmul_n_split)
        )

    # ---- (1) per-shape: at each core count, is it compute- or mem-bound? ----
    byshape = defaultdict(list)
    for r in rows:
        byshape[(r["M"], r["K"], r["N"])].append(r)
    print("=" * 118)
    print("(1) PER-SHAPE ladder: compute_us / mem_us / ratio / measured / hidden / gamma_eff / #Mpasses")
    print("=" * 118)
    for shape in sorted(byshape):
        M, K, N = shape
        print(f"\n  {M}x{K}x{N}  (MACs={M*K*N:.3e})")
        print(f"    {'c':>2} {'comp':>8} {'mem':>7} {'c/m':>6} {'meas':>8} {'add':>8} "
              f"{'hidden':>7} {'gEff':>6} {'Mpass':>7} {'Mpc':>6} {'Npc':>6} bound")
        for r in sorted(byshape[shape], key=lambda x: x["cores"]):
            bound = "COMP" if r["ratio"] > 1.15 else ("MEM" if r["ratio"] < 0.87 else "~bal")
            print(f"    {r['cores']:>2} {r['comp']:>8.1f} {r['mem']:>7.1f} {r['ratio']:>6.2f} "
                  f"{r['meas']:>8.1f} {r['add']:>8.1f} {r['hidden']:>7.1f} {r['gamma_eff']:>6.2f} "
                  f"{r['m_passes']:>7.1f} {r['Mpc']:>6.0f} {r['Npc']:>6.0f} {bound}")

    # ---- (2) hidden fraction vs the candidate access-pattern organizers ----
    # Only rows where min(comp,mem)>0 and where overlap is even meaningful.
    print("\n" + "=" * 118)
    print("(2) Does gamma_eff organize by CORES, by RATIO(comp/mem), or by #M-PASSES?")
    print("=" * 118)
    # group by cores
    bycores = defaultdict(list)
    for r in rows:
        if min(r["comp"], r["mem"]) > 0:
            bycores[r["cores"]].append(r["gamma_eff"])
    print("\n  gamma_eff grouped by CORES:")
    for c in sorted(bycores):
        v = [x for x in bycores[c] if not math.isnan(x)]
        if v:
            print(f"    cores={c:>2}: gamma_eff median={statistics.median(v):+.2f} "
                  f"mean={statistics.mean(v):+.2f} n={len(v)} range=[{min(v):+.2f},{max(v):+.2f}]")

    # bin by ratio comp/mem
    print("\n  gamma_eff grouped by comp/mem RATIO bin:")
    bins = [(0, 0.6), (0.6, 0.9), (0.9, 1.1), (1.1, 1.6), (1.6, 3), (3, 100)]
    for lo, hi in bins:
        v = [r["gamma_eff"] for r in rows
             if lo <= r["ratio"] < hi and not math.isnan(r["gamma_eff"])
             and min(r["comp"], r["mem"]) > 0]
        if v:
            print(f"    ratio in [{lo:>4},{hi:>4}): gamma_eff median={statistics.median(v):+.2f} "
                  f"mean={statistics.mean(v):+.2f} n={len(v)}")

    # bin by #M-passes
    print("\n  gamma_eff grouped by #M-PASSES (M/m / 8) bin:")
    pbins = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 100000)]
    for lo, hi in pbins:
        v = [r["gamma_eff"] for r in rows
             if lo <= r["m_passes"] < hi and not math.isnan(r["gamma_eff"])
             and min(r["comp"], r["mem"]) > 0]
        if v:
            print(f"    Mpasses in [{lo:>5},{hi:>6}): gamma_eff median={statistics.median(v):+.2f} "
                  f"mean={statistics.mean(v):+.2f} n={len(v)}")

    # ---- (3) At what core count does compute ~= mem (crossover)? ----
    print("\n" + "=" * 118)
    print("(3) CROSSOVER: per shape, the core count where comp/mem crosses 1 (overlap becomes visible)")
    print("=" * 118)
    for shape in sorted(byshape):
        M, K, N = shape
        pts = sorted(byshape[shape], key=lambda x: x["cores"])
        cs = [(r["cores"], r["ratio"]) for r in pts]
        # crossover: first core count where ratio drops below 1
        cross = next((c for c, ra in cs if ra < 1.0), None)
        print(f"  {M}x{K}x{N:<6}: " + " ".join(f"c{c}:{ra:.2f}" for c, ra in cs)
              + f"   crossover~cores={cross}")


if __name__ == "__main__":
    main()
