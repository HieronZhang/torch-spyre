#!/usr/bin/env python3
"""Part 2: is the non-spill 'tightness' real or a small-n artifact, and do the
in-set correlations survive? Also: does the spill TERM actually explain the extra
spread of the spilling set (the claim's mechanism)?"""
import importlib.util
import json
import math
import os
import random
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
        return True
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
        if (M, K, N) == (4096, 2048, 4096):
            continue
        if o.cores < 4:
            continue
        key = (M, K, N, o.cores)
        if key not in best or r["kernel_us"] < best[key][1]:
            best[key] = (o, r["kernel_us"])
    return best


def _decomp(o):
    pt_eff = (
        1.0 if o.tiles_output_dim
        else cm.underfill_eff(o.matmul_rows_per_core, p, p.underfill_target_passes_matmul)
    )
    comp = o.matmul_macs / o.cores / (p.mac_peak_per_core_ns * pt_eff)
    rr, ww = o.read_bytes(), o.write_bytes()
    area = o.matmul_rows_per_core * o.matmul_cols_per_core
    spill = (o.matmul_a_bytes + o.matmul_b_bytes) * cm.mm_spill_frac(area, p)
    turn = p.rw_turnaround_ns_per_byte * min(rr, ww)
    mem = rr / p.mm_bw_read_gbps + ww / p.mm_bw_write_gbps + spill / p.mm_bw_read_gbps + turn
    mem_ns = rr / p.mm_bw_read_gbps + ww / p.mm_bw_write_gbps + turn
    return comp / 1000, mem / 1000, mem_ns / 1000, spill, area, rr, ww


def _pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n / (sx * sy)


def _perm_p(xs, ys, iters=20000):
    """Two-sided permutation p-value for |pearson r|."""
    r0 = abs(_pearson(xs, ys))
    if math.isnan(r0):
        return float("nan")
    ys2 = list(ys)
    cnt = 0
    for _ in range(iters):
        random.shuffle(ys2)
        if abs(_pearson(xs, ys2)) >= r0 - 1e-12:
            cnt += 1
    return cnt / iters


def main():
    random.seed(1234)
    best = _load()
    rows = []
    for (M, K, N, cores), (o, meas) in best.items():
        comp, mem, mem_ns, spill, area, rr, ww = _decomp(o)
        denom = min(comp, mem)
        g = (comp + mem - meas) / denom if denom > 0 else float("nan")
        Mt, Nt = o.matmul_rows_per_core, o.matmul_cols_per_core
        rows.append(dict(
            M=M, K=K, N=N, cores=cores, comp=comp, mem=mem, meas=meas,
            spill=spill, area=area, g=g,
            aspect=max(Mt, Nt) / max(1e-9, min(Mt, Nt)),
            mintile=min(Mt, Nt), ai=o.matmul_macs / max(1, rr + ww),
            n_split=getattr(o, "matmul_n_split", 1),
        ))
    nonspill = [r for r in rows if r["spill"] <= 0]
    spilling = [r for r in rows if r["spill"] > 0]
    gN = [r["g"] for r in nonspill]
    gS = [r["g"] for r in spilling]
    gAll = [r["g"] for r in rows]

    # --- 1. Is the sd difference significant? Levene-ish + bootstrap CI on sd. ---
    print("=== 1. Is the non-spill sd genuinely smaller (vs small-n luck)? ===")
    sdN, sdS = statistics.pstdev(gN), statistics.pstdev(gS)
    print(f"  sd(non-spill,n={len(gN)})={sdN:.3f}  sd(spilling,n={len(gS)})={sdS:.3f}")

    # Bootstrap 95% CI for sd of each set (resample with replacement).
    def boot_sd(g, iters=20000):
        out = []
        for _ in range(iters):
            s = [random.choice(g) for _ in g]
            out.append(statistics.pstdev(s))
        out.sort()
        return out[int(0.025 * iters)], out[int(0.975 * iters)]

    loN, hiN = boot_sd(gN)
    loS, hiS = boot_sd(gS)
    print(f"  bootstrap 95% CI sd(non-spill): [{loN:.3f},{hiN:.3f}]")
    print(f"  bootstrap 95% CI sd(spilling) : [{loS:.3f},{hiS:.3f}]")
    print(f"  -> CIs {'OVERLAP' if hiN >= loS else 'disjoint'} "
          f"(overlap => can't claim the non-spill sd is really smaller)")

    # Permutation test: pool all 65 g's, draw a random 9 with the SAME shape/core coverage
    # is complex; simplest honest test: draw 9 at random from the pool and see how often
    # a random size-9 subset has sd <= sdN. This asks: is sd=0.086 unusual for n=9?
    iters = 50000
    cnt = 0
    for _ in range(iters):
        sub = random.sample(gAll, len(gN))
        if statistics.pstdev(sub) <= sdN + 1e-12:
            cnt += 1
    print(f"  P(random size-{len(gN)} subset of the 65 has sd<=0.086) = {cnt/iters:.3f}")
    print(f"     (if this is not tiny, the low sd is expected from small n alone)\n")

    # --- 2. Do the in-set correlations survive a permutation test? ---
    print("=== 2. In-set (non-spill) correlations: permutation p-values (n=9) ===")
    for var in ["aspect", "ai", "n_split", "K", "cores", "mintile"]:
        xs = [r[var] for r in nonspill]
        r = _pearson(xs, gN)
        pval = _perm_p(xs, gN)
        print(f"  {var:8s}: r={r:+.2f}  perm-p={pval:.3f}  "
              f"{'(sig<0.05)' if pval < 0.05 else ''}")
    print()

    # --- 3. Confound: aspect/AI vs cores collinearity in the non-spill set. ---
    print("=== 3. Confound check: is 'aspect' just 'cores' in disguise? (non-spill) ===")
    for c in sorted(set(r["cores"] for r in nonspill)):
        sub = [r for r in nonspill if r["cores"] == c]
        asp = sorted(set(round(r["aspect"], 2) for r in sub))
        print(f"  cores={c}: n={len(sub)} aspects={asp} "
              f"g mean={statistics.mean([r['g'] for r in sub]):+.3f}")
    # partial: within cores=32 only (holds cores fixed), does aspect still correlate?
    c32 = [r for r in nonspill if r["cores"] == 32]
    if len(c32) >= 3:
        rr = _pearson([r["aspect"] for r in c32], [r["g"] for r in c32])
        print(f"  within cores=32 (n={len(c32)}): pearson(aspect,g)={rr:+.2f}  "
              f"aspects={sorted(round(r['aspect'],2) for r in c32)}")
    print()

    # --- 4. The CLAIM's MECHANISM: does the spill TERM explain the spilling spread? ---
    # If the extra spread of the spilling set is 'inherited spill-term error', then the
    # spill magnitude should correlate with the gamma_eff DEVIATION from the ~0.61 center.
    print("=== 4. Does spill magnitude drive the gamma_eff spread? (spilling set) ===")
    center = statistics.median(gAll)
    spill_us = [r["spill"] / 150 / 1000 for r in spilling]
    dev = [abs(r["g"] - center) for r in spilling]
    print(f"  center gamma (median all) = {center:.3f}")
    print(f"  pearson(spill_us, |g-center|) = {_pearson(spill_us, dev):+.2f}")
    print(f"  pearson(spill_frac_of_mem, |g-center|) = "
          f"{_pearson([ (r['spill']/150/1000)/r['mem'] for r in spilling], dev):+.2f}")
    # If spill really drives error, LOW-spill spilling rows should be as tight as non-spill.
    lowspill = sorted(spilling, key=lambda r: r["spill"])[:len(spilling)//3]
    highspill = sorted(spilling, key=lambda r: r["spill"])[-len(spilling)//3:]
    print(f"  spilling LOW-spill tercile (n={len(lowspill)}): "
          f"g sd={statistics.pstdev([r['g'] for r in lowspill]):.3f} "
          f"mean={statistics.mean([r['g'] for r in lowspill]):+.3f}")
    print(f"  spilling HIGH-spill tercile(n={len(highspill)}): "
          f"g sd={statistics.pstdev([r['g'] for r in highspill]):.3f} "
          f"mean={statistics.mean([r['g'] for r in highspill]):+.3f}")
    print("     (claim predicts LOW-spill tercile ~ as tight as non-spill;")
    print("      if LOW-spill is ALSO wide, the spread is NOT a spill-term artifact)\n")

    # --- 5. Compare like-with-like: restrict the SPILLING set to cores in {16,32} and
    #        min(M,N)==1024 etc. to match the non-spill coverage, then compare sd. ---
    print("=== 5. Coverage-matched comparison (cores in {16,32}) ===")
    nsp_c = [r for r in nonspill if r["cores"] in (16, 32)]
    sp_c = [r for r in spilling if r["cores"] in (16, 32)]
    print(f"  non-spill cores16/32 n={len(nsp_c)} g sd={statistics.pstdev([r['g'] for r in nsp_c]):.3f}")
    print(f"  spilling  cores16/32 n={len(sp_c)} g sd={statistics.pstdev([r['g'] for r in sp_c]):.3f}")
    # And the spilling set with area just above the knee (65536 < area <= 131072): the
    # 'least spilled' spilling rows -- closest neighbors to the non-spill set.
    near = [r for r in spilling if r["area"] <= 131072]
    if near:
        print(f"  spilling area<=131072 (just past knee) n={len(near)} "
              f"g sd={statistics.pstdev([r['g'] for r in near]):.3f} "
              f"mean={statistics.mean([r['g'] for r in near]):+.3f}")


if __name__ == "__main__":
    main()
