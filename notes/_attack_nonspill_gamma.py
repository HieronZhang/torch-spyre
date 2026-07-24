#!/usr/bin/env python3
"""Independent re-derivation to attack the 'non-spill gamma_eff is tight' claim.

Re-derives gamma_eff from raw records with the prescribed clean filters, splits by
spilling vs non-spilling (per-core tile area vs 65536), and tests whether the non-spill
set is genuinely tight or just tiny, and whether any access-pattern variable correlates
with gamma_eff WITHIN the non-spill set.
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
        return True  # planner-chosen mm = balanced (K unsplit)
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
        if min(M, N) < 1024 or K < 512:
            continue
        if (M, K, N) == (4096, 2048, 4096):  # contaminated
            continue
        if o.cores < 4:  # low-core mac_peak confound
            continue
        key = (M, K, N, o.cores)
        if key not in best or r["kernel_us"] < best[key][1]:
            best[key] = (o, r["kernel_us"], r.get("kernel_us_cv"))
    return best


def _decomp(o):
    """compute_us, mem_us(with spill), mem_nospill_us, spill_bytes, area, r, w."""
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
    turn = p.rw_turnaround_ns_per_byte * min(rr, ww)
    mem = rr / p.mm_bw_read_gbps + ww / p.mm_bw_write_gbps + spill / p.mm_bw_read_gbps + turn
    mem_ns = rr / p.mm_bw_read_gbps + ww / p.mm_bw_write_gbps + turn
    return comp / 1000, mem / 1000, mem_ns / 1000, spill, area, rr, ww


def _stats(xs):
    n = len(xs)
    if n == 0:
        return (0, float("nan"), float("nan"))
    m = statistics.mean(xs)
    s = statistics.pstdev(xs) if n > 1 else 0.0
    return (n, m, s)


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
    if sx == 0 or sy == 0:
        return float("nan")
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    return cov / (sx * sy)


def main():
    best = _load()
    rows = []
    for (M, K, N, cores), (o, meas, cv) in best.items():
        comp, mem, mem_ns, spill, area, rr, ww = _decomp(o)
        denom = min(comp, mem)
        g_eff = (comp + mem - meas) / denom if denom > 0 else float("nan")
        denom2 = min(comp, mem_ns)
        g_eff_ns = (comp + mem_ns - meas) / denom2 if denom2 > 0 else float("nan")
        Mtile, Ntile = o.matmul_rows_per_core, o.matmul_cols_per_core
        ai = o.matmul_macs / max(1, (rr + ww))
        rows.append(dict(
            M=M, K=K, N=N, cores=cores, comp=comp, mem=mem, mem_ns=mem_ns,
            meas=meas, spill=spill, area=area, g_eff=g_eff, g_eff_ns=g_eff_ns,
            m_split=getattr(o, "matmul_m_split", 1),
            n_split=getattr(o, "matmul_n_split", 1),
            Mtile=Mtile, Ntile=Ntile,
            aspect=max(Mtile, Ntile) / max(1e-9, min(Mtile, Ntile)),
            mintile=min(Mtile, Ntile), ai=ai, cv=cv, membound=(mem > comp),
        ))

    nonspill = [r for r in rows if r["spill"] <= 0]
    spilling = [r for r in rows if r["spill"] > 0]

    print(f"TOTAL clean cores>=4 points: {len(rows)}")
    print(f"  non-spill (area<=65536): {len(nonspill)}")
    print(f"  spilling  (area> 65536): {len(spilling)}\n")

    for label, sub in [("FULL clean", rows), ("NON-SPILL", nonspill), ("SPILLING", spilling)]:
        g = [r["g_eff"] for r in sub if not math.isnan(r["g_eff"])]
        n, m, s = _stats(g)
        med = statistics.median(g) if g else float("nan")
        lo, hi = (min(g), max(g)) if g else (float("nan"), float("nan"))
        # standard error of the mean and 95% half-width for the sd estimate
        sem = s / math.sqrt(n) if n > 1 else float("nan")
        print(f"{label:12s} n={n:3d}  gamma_eff mean={m:+.3f} sd={s:.3f} sem={sem:.3f} "
              f"median={med:+.3f} range=[{lo:+.3f},{hi:+.3f}]")
    print()

    g_ns = [r["g_eff_ns"] for r in nonspill]
    print(f"NON-SPILL g_eff_ns (spill-free mem, identical for non-spill): "
          f"mean={statistics.mean(g_ns):+.3f} sd={statistics.pstdev(g_ns):.3f}\n")

    print("=== NON-SPILL coverage ===")
    shapes = sorted(set((r["M"], r["K"], r["N"]) for r in nonspill))
    print(f"  distinct (M,K,N) shapes: {len(shapes)}: {shapes}")
    print(f"  cores present: {sorted(set(r['cores'] for r in nonspill))}")
    print(f"  K values: {sorted(set(r['K'] for r in nonspill))}")
    areas = sorted(set(int(r["area"]) for r in nonspill))
    print(f"  per-core areas: min={min(areas)} max={max(areas)} (cap=65536)")
    print(f"  tile aspect: {min(r['aspect'] for r in nonspill):.2f}..{max(r['aspect'] for r in nonspill):.2f}")
    print(f"  min tile dim: {min(r['mintile'] for r in nonspill):.0f}..{max(r['mintile'] for r in nonspill):.0f}")
    print(f"  arith intensity: {min(r['ai'] for r in nonspill):.1f}..{max(r['ai'] for r in nonspill):.1f}")
    print(f"  mem-bound: {sum(1 for r in nonspill if r['membound'])}/{len(nonspill)}\n")

    print("=== NON-SPILL gamma_eff per (M,K,N) ===")
    byshape = defaultdict(list)
    for r in nonspill:
        byshape[(r["M"], r["K"], r["N"])].append(r)
    for sh in sorted(byshape):
        sub = byshape[sh]
        g = [r["g_eff"] for r in sub]
        print(f"  {sh[0]}x{sh[1]}x{sh[2]:<6} n={len(sub)} cores={sorted(r['cores'] for r in sub)} "
              f"g_eff mean={statistics.mean(g):+.3f} range=[{min(g):+.3f},{max(g):+.3f}]")
    print()

    print("=== correlation of gamma_eff vs candidate vars (NON-SPILL) ===")
    for var in ["K", "cores", "aspect", "mintile", "ai", "area", "Mtile", "Ntile",
                "m_split", "n_split", "comp", "mem"]:
        xs = [r[var] for r in nonspill]
        ys = [r["g_eff"] for r in nonspill]
        rP = _pearson(xs, ys)
        try:
            rL = _pearson([math.log2(max(1e-9, x)) for x in xs], ys)
        except Exception:
            rL = float("nan")
        print(f"  {var:8s}: pearson(lin)={rP:+.2f} pearson(log2)={rL:+.2f}  vals[{min(xs):.1f}..{max(xs):.1f}]")
    print()

    print("=== correlation of gamma_eff vs candidate vars (FULL clean) ===")
    for var in ["K", "cores", "aspect", "mintile", "ai", "area", "spill"]:
        xs = [r[var] for r in rows]
        ys = [r["g_eff"] for r in rows]
        print(f"  {var:8s}: pearson(lin)={_pearson(xs, ys):+.2f}  vals[{min(xs):.1f}..{max(xs):.1f}]")
    print()

    print("=== NON-SPILL rows (sorted) ===")
    print(f"  {'M':>5} {'K':>5} {'N':>5} {'cr':>3} {'comp':>6} {'mem':>6} {'meas':>6} "
          f"{'g_eff':>6} {'area':>7} {'Mt':>6} {'Nt':>6} {'asp':>5} {'AI':>6} {'mb':>2} {'cv':>5}")
    for r in sorted(nonspill, key=lambda r: (r["M"], r["K"], r["N"], r["cores"])):
        cv = r["cv"] if r["cv"] is not None else float("nan")
        print(f"  {r['M']:5d} {r['K']:5d} {r['N']:5d} {r['cores']:3d} "
              f"{r['comp']:6.1f} {r['mem']:6.1f} {r['meas']:6.1f} {r['g_eff']:+6.3f} "
              f"{int(r['area']):7d} {r['Mtile']:6.0f} {r['Ntile']:6.0f} "
              f"{r['aspect']:5.2f} {r['ai']:6.1f} {int(r['membound']):2d} {cv:5.2f}")


if __name__ == "__main__":
    main()
