# Copyright 2024-2026 IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache-2.0
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

"""Explore the FUNCTIONAL FAMILY of the matmul overlap leftover term.

Background.  The shipped model is ``compute + mem - g*min(compute,mem)``, which is
algebraically identical to ``max(compute,mem) + (1-g)*min(compute,mem)``.  So the model
already *is* "max plus something"; the open question is what that something is.  An earlier
pass tested leftovers that were all proportional to ``min`` (or constant, or proportional to
``write``) and none beat the shipped form globally.

This script widens the search to different functional FAMILIES, each with a mechanism:

  F0  max + (1-g)*min                shipped: a fixed fraction of the shorter stream is exposed
  P   (c^s + m^s)^(1/s)              p-norm roofline softening; s=1 -> sum, s->inf -> max
  S   max(c, m, (c+m)/k)             SHARED-PORT bound: compute and DMA contend for one
                                     resource, so their SUM is also rate-limited.  k in (1,2).
                                     Key difference from F0: when the streams are unbalanced
                                     this charges NOTHING, whereas F0 always charges (1-g)*min.
  G   max + a * min^p * max^(1-p)    geometric blend (dimensionally consistent); p=1 -> F0
  R   max + min * (a + b*rho)        leftover fraction varies with the balance rho=min/max
  SR  max(c,m,(c+m)/k) + a*min       shared port PLUS a residual exposed fraction
  C   max + min*(a + b/cores)        leftover depends on per-core pipeline depth

Components (compute, mem, split) are extracted EXACTLY from the live model by solving it at
two parameter settings, never re-derived by hand.

Usage:
    python3 notes/explore_overlap_forms.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import statistics

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)


def _cm():
    path = os.path.join(_ROOT, "torch_spyre", "_inductor", "cost_model.py")
    spec = importlib.util.spec_from_file_location("_cm_explore", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CM = _cm()


def _predict(feats, **over):
    p = CM.CostParams()
    for k, v in over.items():
        setattr(p, k, v)
    return CM.predict_ops(CM.ops_from_json(json.dumps(feats)), p) / 1e3


# EVERY rate that `_matmul_mac_peak` can return.  Zeroing only `mac_peak_per_core_ns` does NOT
# kill the compute term for a gated batched matmul -- that path returns the bmm layout rates and
# ignores the plain peak entirely, which silently dropped 170 records (the whole bmm population).
_PEAK_KILL = {
    "mac_peak_per_core_ns": 1e12,
    "bmm_layout_peak_fast_ns": 1e12,
    "bmm_layout_peak_a_default_ns": 1e12,
    "bmm_layout_peak_b_default_ns": 1e12,
    "bmm_default_mac_peak_per_core_ns": 1e12,
}
# The real split coefficients.  `CostParams` is a plain (non-frozen, non-slots) dataclass, so a
# misspelled override is silently accepted as a dead attribute and changes nothing -- which is how
# an earlier version of this file measured `split` as identically zero and folded it into `mem`.
_SPLIT_KILL = {"mm_split_reread_us_per_elem": 0.0, "mm_split_short_us_per_elem": 0.0}


def decompose(rec):
    """Return (compute, mem, split) in us, extracted exactly from the live model.

    Verified by reconstruction: ``max(c,m) + (1-gamma)*min(c,m) + split`` reproduces the live
    model to 0.0000 % on all 755 matmul-family records.  Re-run ``--verify`` after any change.
    """
    f = rec["feats"]
    # gamma=0 -> t = compute + mem + split
    full = _predict(f, overlap_gamma=0.0)
    nocomp = _predict(f, overlap_gamma=0.0, **_PEAK_KILL)
    nosplit = _predict(f, overlap_gamma=0.0, **_SPLIT_KILL)
    split = max(0.0, full - nosplit)
    mem = max(0.0, nocomp - split)
    comp = max(0.0, full - split - mem)
    return comp, mem, split


def verify_decomposition(recs):
    """Assert the decomposition reconstructs the live model on every matmul-family record."""
    g = CM.CostParams().overlap_gamma
    bad = []
    for r in recs:
        if r.get("failed") or not r.get("feats") or not r.get("kernel_us"):
            continue
        if not (r.get("op") or "").startswith(("mm", "matmul", "bmm")):
            continue
        c, m, sp = decompose(r)
        recon = max(c, m) + (1.0 - g) * min(c, m) + sp
        live = _predict(r["feats"])
        if abs(recon - live) / max(live, 1e-9) > 1e-4:
            bad.append((r.get("op"), recon, live))
    return bad


# ---------------------------------------------------------------- forms


def f_shipped(c, m, sp, cores, par):
    return max(c, m) + par[0] * min(c, m) + sp


def f_pnorm(c, m, sp, cores, par):
    s = par[0]
    if c <= 0 and m <= 0:
        return sp
    return (c**s + m**s) ** (1.0 / s) + sp


def f_shared(c, m, sp, cores, par):
    k = par[0]
    return max(c, m, (c + m) / k) + sp


def f_geom(c, m, sp, cores, par):
    a, p = par
    lo, hi = min(c, m), max(c, m)
    if lo <= 0:
        return hi + sp
    return hi + a * (lo**p) * (hi ** (1.0 - p)) + sp


def f_ratio(c, m, sp, cores, par):
    a, b = par
    lo, hi = min(c, m), max(c, m)
    rho = lo / hi if hi > 0 else 0.0
    return hi + lo * (a + b * rho) + sp


def f_shared_res(c, m, sp, cores, par):
    k, a = par
    return max(c, m, (c + m) / k) + a * min(c, m) + sp


def f_cores(c, m, sp, cores, par):
    a, b = par
    n = max(1, cores or 1)
    return max(c, m) + min(c, m) * (a + b / n) + sp


FORMS = {
    "F0 shipped   max + a*min": (f_shipped, [(0.30, 0.80, 0.02)]),
    "P  p-norm    (c^s+m^s)^1/s": (f_pnorm, [(1.0, 6.0, 0.05)]),
    "S  sharedport max(c,m,(c+m)/k)": (f_shared, [(1.0, 2.0, 0.01)]),
    "G  geometric max + a*lo^p*hi^(1-p)": (f_geom, [(0.1, 1.2, 0.05), (0.0, 2.0, 0.1)]),
    "R  ratio     max + lo*(a+b*rho)": (f_ratio, [(-0.4, 1.0, 0.05), (-1.0, 1.0, 0.05)]),
    "SR sharedport+res": (f_shared_res, [(1.0, 2.0, 0.02), (0.0, 0.6, 0.02)]),
    "C  cores     max + lo*(a+b/cores)": (f_cores, [(0.0, 0.8, 0.05), (-2.0, 8.0, 0.25)]),
}


def _grid(specs):
    out = [[]]
    for lo, hi, step in specs:
        n = int(round((hi - lo) / step)) + 1
        vals = [lo + i * step for i in range(n)]
        out = [prev + [v] for prev in out for v in vals]
    return out


def score(fn, par, rows):
    errs = []
    for r in rows:
        pred = fn(r["c"], r["m"], r["sp"], r["cores"], par)
        errs.append((pred - r["t"]) / r["t"] * 100.0)
    return errs


def summarize(errs):
    n = len(errs)
    rms = math.sqrt(sum(e * e for e in errs) / n)
    return {
        "rms": rms,
        "mean_abs": statistics.mean(abs(e) for e in errs),
        "bias": statistics.mean(errs),
        "worst": max(abs(e) for e in errs),
        "over10": sum(1 for e in errs if abs(e) > 10),
        "n": n,
    }


def main():
    recs = json.load(open(os.path.join(_HERE, "sweep_records.json")))["records"]

    def cores_of(r):
        try:
            return int(r.get("cores"))
        except (TypeError, ValueError):
            return None

    cohorts = {}
    rows_all = []
    dropped = []
    for r in recs:
        if r.get("failed") or not r.get("feats") or not r.get("kernel_us"):
            continue
        op = r.get("op") or ""
        if not (op.startswith("mm") or op.startswith("matmul") or op.startswith("bmm")):
            continue
        c, m, sp = decompose(r)
        if c <= 0 or m <= 0:
            dropped.append(r.get("op"))
            continue
        rows_all.append(
            {
                "c": c,
                "m": m,
                "sp": sp,
                "cores": cores_of(r) or 32,
                "t": r.get("kernel_us_min") or r["kernel_us"],
                "op": op,
                "reps": r.get("reps"),
                "cv": r.get("kernel_us_cv"),
                "tag": r.get("tag") or r.get("section") or "",
            }
        )

    plain = [r for r in rows_all if r["op"] in ("mm", "mmwd")]
    cohorts["repeat-backed mm/mmwd"] = [r for r in plain if r["reps"]]
    cohorts["ALL mm/mmwd"] = plain
    cohorts["ALL matmul-family"] = rows_all

    bad = verify_decomposition(recs)
    print(f"decomposition self-check: {len(bad)} records fail to reconstruct the live model")
    if bad:
        raise SystemExit("decomposition is WRONG -- fix before trusting any number below")
    from collections import Counter
    print(f"decomposed {len(rows_all)} matmul-family records; dropped {len(dropped)}: {Counter(dropped)}")
    for k, v in cohorts.items():
        print(f"   {k}: {len(v)}")

    fit_on = cohorts["repeat-backed mm/mmwd"]
    print("\nfitting each form on the repeat-backed cohort, judging on all three\n")
    hdr = f"{'form':38s} {'params':22s}"
    for k in cohorts:
        hdr += f" {k[:20]:>21s}"
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for name, (fn, specs) in FORMS.items():
        best = None
        for par in _grid(specs):
            try:
                s = summarize(score(fn, par, fit_on))
            except (ValueError, ZeroDivisionError, OverflowError):
                continue
            if best is None or s["rms"] < best[0]["rms"]:
                best = (s, par)
        s, par = best
        line = f"{name:38s} {str([round(x, 3) for x in par]):22s}"
        row = {}
        for k, v in cohorts.items():
            sk = summarize(score(fn, par, v))
            row[k] = sk
            line += f"  {sk['rms']:6.2f}/{sk['bias']:+6.2f}/{sk['over10']:4d}"
        results[name] = (par, row)
        print(line)
    print("\n(each cell: RMS% / bias% / #points>10%)")

    # --- where do the families actually differ?  the unbalanced regime ---
    print("\nbalance census on ALL mm/mmwd (rho = min(c,m)/max(c,m)):")
    for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)):
        sub = [r for r in plain if lo <= min(r["c"], r["m"]) / max(r["c"], r["m"]) < hi]
        if not sub:
            continue
        line = f"   rho {lo:.2f}-{hi:.2f}  n={len(sub):4d}"
        for name, (par, _) in results.items():
            fn = FORMS[name][0]
            sk = summarize(score(fn, par, sub))
            line += f"   {name.split()[0]}={sk['rms']:6.2f}"
        print(line)


if __name__ == "__main__":
    main()
