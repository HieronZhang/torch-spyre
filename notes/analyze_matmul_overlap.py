#!/usr/bin/env python3
# Copyright 2024-2025 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
"""Cat-3 (matmul compute/HBM overlap) RE-FIT HARNESS — the correct, review-informed
version (replaces the old gamma(cores) scratch tool, which fit the REJECTED variable-gamma
form). Pure Python: reads cost_model.py + notes/sweep_records.json, no Spyre HW needed.

It operationalises the 2026-07-24 adversarial-panel findings so the Part III re-fit is one
command when the clean overnight data lands. On the CURRENT repeat-backed MMCORE cohort it
should REPRODUCE the panel's numbers (a self-check that we can verify the reviewer).

Prints:
  (A) NOISE/CV audit — the repeat structure of the cohort (only repeat-backed points are trusted).
  (B) SATURATION census — per cores, how many points sit in read < gamma*compute (where the
      min() picks `read` regardless of gamma, so gamma is UNIDENTIFIABLE there). This is WHY a
      single gamma is only a central value, not a measured constant.
  (C) ENTANGLEMENT 2x2 matrix — overlap-form {shipped gamma*min(compute,mem) | read-overlap
      min(read, gamma*compute)} x spill-form {symmetric (|A|+|B|) | operand-min 2*min(|A|,|B|)},
      each cell best-fit over gamma at PEAK (default 1040). Shows read-overlap+2min is the best
      cell and that 2min helps under BOTH overlap forms (so spill is NOT the entangled piece).
  (D) GAMMA-IDENTIFIABILITY — RMS-vs-gamma on the UNSATURATED (gamma-binding) subset only; if the
      valley is flat over ~0.4-0.7 the data cannot pin gamma (report the width honestly).

Model forms (all times in ns, /1000 -> us):
  compute = macs / cores / (peak * pt_eff)
  read    = (operand_bytes + spill) / bw_read         # loads, double-buffered
  write   = out_bytes / bw_write                       # stores, serial
  turn    = rw_turnaround * min(operand_bytes, out_bytes)
  shipped-overlap : T = compute + (read+write+turn) - gamma*min(compute, read+write+turn)
  read-overlap    : T = compute + read + write + turn - min(read, gamma*compute)
  spill = FRAC * mm_spill_frac(area);  FRAC = (|A|+|B|)  [sym]  or  2*min(|A|,|B|)  [min]
"""
import importlib.util
import json
import math
import os
import statistics
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
PEAK = float(os.environ.get("PEAK", "1040"))  # SOLID single-core value; 1140 = shipped


def _cm():
    path = os.path.join(_ROOT, "torch_spyre", "_inductor", "cost_model.py")
    spec = importlib.util.spec_from_file_location("cost_model", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cm = _cm()
p = cm.CostParams()


def _load(section_prefix=("MMISO_CORE", "MMCORE"), repeat_only=True):
    """Repeat-backed forced-core mm/mmwd, deduped to the MIN measured us per (M,K,N,cores).
    Loads the new clean MMISO_CORE sweep AND the older MMCORE cohort."""
    recs = json.load(open(os.path.join(_HERE, "sweep_records.json")))["records"]
    best = {}
    for r in recs:
        if r.get("op") not in ("mm", "mmwd") or not r.get("feats") or r.get("failed"):
            continue
        if not (r.get("section") or "").startswith(section_prefix):
            continue
        if repeat_only and not r.get("reps"):
            continue
        t = r.get("kernel_us_min") or r.get("kernel_us")
        if not t:
            continue
        o = cm.op_from_dict(r["feats"][0])
        if not getattr(o, "is_matmul", False) or o.cores <= 0:
            continue
        M, K, N = r["rows"], r["cols"], (r.get("N") or r["cols"])
        # contaminated shape (low-core rows physically impossible), verified earlier
        if (M, K, N) == (4096, 2048, 4096):
            continue
        key = (M, K, N, o.cores)
        cv = r.get("kernel_us_cv") or 0.0
        if key not in best or t < best[key][1]:
            best[key] = (o, t, cv)
    return best


def _terms(o, spill_form):
    """(compute_us, read_us, write_us, turn_us) at PEAK for the given spill form."""
    pt_eff = (
        1.0
        if o.tiles_output_dim
        else cm.underfill_eff(o.matmul_rows_per_core, p, p.underfill_target_passes_matmul)
    )
    comp = o.matmul_macs / o.cores / (PEAK * pt_eff)
    rr, ww = o.read_bytes(), o.write_bytes()
    area = o.matmul_rows_per_core * o.matmul_cols_per_core
    a, b = o.matmul_a_bytes, o.matmul_b_bytes
    frac = (a + b) if spill_form == "sym" else 2 * min(a, b)
    spill = frac * cm.mm_spill_frac(area, p)
    read = (rr + spill) / p.mm_bw_read_gbps
    write = ww / p.mm_bw_write_gbps
    turn = p.rw_turnaround_ns_per_byte * min(rr, ww)
    return comp / 1000, read / 1000, write / 1000, turn / 1000


def _pred(comp, read, write, turn, overlap, gamma):
    if overlap == "shipped":
        mem = read + write + turn
        return comp + mem - gamma * min(comp, mem)
    return comp + read + write + turn - min(read, gamma * comp)  # read-overlap


def _rms(e):
    return (math.sqrt(sum(x * x for x in e) / len(e)) if e else 0.0)


def _best_gamma(rows, overlap, spill_form, grid=None):
    grid = grid or [g / 100 for g in range(0, 101, 2)]
    best = None
    terms = [(_terms(o, spill_form), meas) for (o, meas, cv) in rows]
    for g in grid:
        errs = [(_pred(*t, overlap, g) / meas - 1) * 100 for (t, meas) in terms]
        r = _rms(errs)
        if best is None or r < best[0]:
            best = (r, g)
    return best  # (rms, gamma)


def main():
    best = _load()
    rows = list(best.values())  # [(op, meas_us, cv), ...]
    print(f"PEAK={PEAK}  cohort=MMCORE repeat-backed, deduped: n={len(rows)}\n")

    # (A) CV audit
    cvs = sorted(cv for (_o, _m, cv) in rows)
    print("=== (A) NOISE/CV audit (kernel_us_cv, %) ===")
    print(f"  median {statistics.median(cvs):.2f}  max {max(cvs):.2f}  "
          f">0.5%: {sum(1 for c in cvs if c > 0.5)}/{len(cvs)}  "
          f"<=0.15%: {sum(1 for c in cvs if c <= 0.15)}/{len(cvs)}\n")

    # (B) saturation census at gamma=0.6, read-overlap terms
    print("=== (B) SATURATION census (read < 0.6*compute => gamma unidentifiable) ===")
    bycore = defaultdict(lambda: [0, 0])
    for (o, meas, cv) in rows:
        comp, read, _w, _t = _terms(o, "min")
        sat = read < 0.6 * comp
        bycore[o.cores][0] += 1
        bycore[o.cores][1] += 1 if sat else 0
    for c in sorted(bycore):
        tot, sat = bycore[c]
        print(f"  cores={c:>2}: {sat}/{tot} saturated (gamma-blind)"
              + ("   <- gamma BINDS here" if sat < tot else ""))
    print()

    # (C) entanglement 2x2 matrix
    print(f"=== (C) ENTANGLEMENT 2x2 (best-fit gamma at PEAK={PEAK:.0f}) ===")
    print(f"  {'':16} {'spill=sym(|A|+|B|)':22} {'spill=min 2*min(A,B)':22}")
    for overlap in ("shipped", "read"):
        cells = []
        for spill_form in ("sym", "min"):
            r, g = _best_gamma(rows, overlap, spill_form)
            cells.append(f"RMS {r:4.1f}% (g*={g:.2f})")
        print(f"  overlap={overlap:8} {cells[0]:22} {cells[1]:22}")
    print("  (read-overlap + 2min should win; 2min should help under BOTH overlap rows)\n")

    # (D) gamma identifiability on the UNSATURATED subset (read-overlap + 2min)
    print("=== (D) GAMMA-IDENTIFIABILITY: RMS vs gamma, UNSATURATED subset only ===")
    unsat = []
    for (o, meas, cv) in rows:
        comp, read, w, t = _terms(o, "min")
        if read >= 0.6 * comp:  # gamma actually binds
            unsat.append(((comp, read, w, t), meas, cv))
    print(f"  unsaturated points: n={len(unsat)} (cv median "
          f"{statistics.median([c for _t,_m,c in unsat]):.2f}%)" if unsat else "  (none)")
    for g in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        errs = [(_pred(*t, "read", g) / meas - 1) * 100 for (t, meas, _cv) in unsat]
        print(f"    gamma={g:.1f}: RMS {_rms(errs):4.1f}%")
    print("  (read the valley shape: a broad flat basin = gamma is only a central value; a\n"
          "   clear min with steep sides = the binding subset DOES constrain it — report which,\n"
          "   and remember this cohort is noisy (cv~0.8%, small n) so do not over-pin.)")


if __name__ == "__main__":
    main()
