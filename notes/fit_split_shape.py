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

"""Split-shape term BAKE-OFF.

The matmul model under-predicts lopsided splits; the effect is real, size-gated, and
U-shaped in the split, but Round 1 could not identify its FORM (everything at cores=32
where m*n=32 confounds m, n, M-tile, N-tile, weight). This harness fits several candidate
ADDITIVE terms (each is what would be added on top of the current pred_us) to the measured
residual = kernel_us - pred_us, and ranks them by LEAVE-ONE-SHAPE-OUT RMSE so a form is
chosen for generalization, not in-sample fit. It is OFFLINE (reads notes/sweep_records.json;
no hardware). Run after Round 2 (run_split_shape_r2.sh) is folded into the DB.

  python notes/fit_split_shape.py                 # mmwd, k=1, non-tiny
  python notes/fit_split_shape.py --include-bmm   # + bmm_wd with b=1 (batch serialized)

Add a candidate by appending to CANDIDATES: a list of basis functions f(row)->float
(design-matrix columns; the term is sum coef_i * f_i, fit with NO intercept so it vanishes
for balanced splits). Forms with a knee sweep it via the `knees` grid.
"""

import argparse
import json


def load_rows(path, include_bmm):
    recs = json.load(open(path))
    recs = recs if isinstance(recs, list) else recs.get("records", recs)
    rows = []
    for r in recs:
        if not str(r.get("log_file", "")).startswith("split_shape"):
            continue
        if r.get("failed"):
            continue
        op = r.get("op")
        s = r.get("split_forced") or {}
        if s.get("k", 1) != 1:  # exclude K-split (its own PSUM miss)
            continue
        M, K, N = r.get("M"), r.get("K"), r.get("N")
        ku, pu = r.get("kernel_us"), r.get("pred_us")
        if None in (M, K, N, ku, pu):
            continue
        if min(M, N) < 512 or K < 256:  # exclude the small-kernel over-prediction floor
            continue
        m, n, b = s.get("m", 1), s.get("n", 1), s.get("b", 1)
        if op == "mmwd":
            pass
        elif op == "bmm_wd" and include_bmm and b == 1:  # batch serialized only
            pass
        else:
            continue
        rows.append(
            dict(
                op=op, M=M, K=K, N=N, B=r.get("B", 1), m=m, n=n,
                Mm=M / m, Nn=N / n, cores=b * m * n, area=(M / m) * (N / n),
                weight=K * N, act=M * K, y=ku - pu, shape=(M, K, N, op),
            )
        )
    return rows


def _solve(ata, aty):
    """Gaussian elimination for the small normal-equations system A^T A x = A^T y."""
    p = len(aty)
    a = [row[:] + [aty[i]] for i, row in enumerate(ata)]
    for c in range(p):
        piv = max(range(c, p), key=lambda r: abs(a[r][c]))
        if abs(a[piv][c]) < 1e-12:
            return None
        a[c], a[piv] = a[piv], a[c]
        for r in range(p):
            if r == c:
                continue
            f = a[r][c] / a[c][c]
            for k in range(c, p + 1):
                a[r][k] -= f * a[c][k]
    return [a[i][p] / a[i][i] for i in range(p)]


def fit(rows, cols):
    """Least squares (no intercept) of y ~ sum coef_i * cols_i. Returns (coefs, design)."""
    X = [[f(r) for f in cols] for r in rows]
    y = [r["y"] for r in rows]
    p = len(cols)
    ata = [[sum(X[i][a] * X[i][b] for i in range(len(X))) for b in range(p)] for a in range(p)]
    aty = [sum(X[i][a] * y[i] for i in range(len(X))) for a in range(p)]
    return _solve(ata, aty), X


def rmse(rows, cols, coefs):
    if coefs is None:
        return float("inf")
    se = 0.0
    for r in rows:
        pred = sum(c * f(r) for c, f in zip(coefs, cols))
        se += (r["y"] - pred) ** 2
    return (se / len(rows)) ** 0.5


def loso_rmse(rows, cols):
    """Leave-one-SHAPE-out RMSE: fit on all shapes but one, predict the held-out shape."""
    shapes = sorted({r["shape"] for r in rows})
    if len(shapes) < 2:
        return float("inf")
    se, k = 0.0, 0
    for held in shapes:
        train = [r for r in rows if r["shape"] != held]
        test = [r for r in rows if r["shape"] == held]
        coefs, _ = fit(train, cols)
        if coefs is None:
            return float("inf")
        for r in test:
            pred = sum(c * f(r) for c, f in zip(coefs, cols))
            se += (r["y"] - pred) ** 2
            k += 1
    return (se / k) ** 0.5


def relu(x):
    return x if x > 0 else 0.0


# Candidate additive terms. Each: name -> (list of basis cols, optional knee grid + builder).
# `builder(knee)` returns the cols list for a given knee value.
def CANDIDATES(kw=8, ka=16):
    def maxfan(r):
        return max(r["m"], r["n"])

    return {
        # refuted baseline: weight/activation broadcast fanout (the original proposal)
        "proposed |W|(m-kw)+|A|(n-ka)": ([
            lambda r: r["weight"] * relu(r["m"] - kw),
            lambda r: r["act"] * relu(r["n"] - ka),
        ], None),
        # symmetric M x split-excess
        "M*(relu(m-8)+relu(n-8))": ([
            lambda r: r["M"] * (relu(r["m"] - 8) + relu(r["n"] - 8)),
        ], None),
        # per-core tile edge x own-axis fanout-excess
        "Mm*relu(m-8)+Nn*relu(n-8)": ([
            lambda r: r["Mm"] * relu(r["m"] - 8) + r["Nn"] * relu(r["n"] - 8),
        ], None),
        # long per-core edge x max-fanout-excess
        "longedge*relu(maxfan-8)": ([
            lambda r: max(r["Mm"], r["Nn"]) * relu(maxfan(r) - 8),
        ], None),
        # per-core AREA x max-fanout-excess
        "area*relu(maxfan-8)": ([
            lambda r: r["area"] * relu(maxfan(r) - 8),
        ], None),
        # absolute M x max-fanout-excess (symmetric gate)
        "M*relu(maxfan-8)": ([
            lambda r: r["M"] * relu(maxfan(r) - 8),
        ], None),
        # anisotropy of the per-core tile (knee on log2 aspect)
        "relu(log2(aniso)-c)": (None, [1, 2, 3]),  # knee grid handled below
        # CONTROL: pure area (what the existing spill term already keys on) -- should lose
        "area (control)": ([lambda r: r["area"]], None),
    }


def aniso_cols(c):
    import math

    def f(r):
        lo = min(r["Mm"], r["Nn"])
        hi = max(r["Mm"], r["Nn"])
        return relu(math.log2(hi / lo) - c) * r["M"]  # scaled by M (size-gating)

    return [f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="notes/sweep_records.json")
    ap.add_argument("--include-bmm", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.db, args.include_bmm)
    shapes = sorted({r["shape"] for r in rows})
    base = (sum(r["y"] ** 2 for r in rows) / len(rows)) ** 0.5 if rows else 0
    print(f"{len(rows)} rows over {len(shapes)} shapes "
          f"(op mix: {', '.join(sorted({r['op'] for r in rows}))})")
    print(f"baseline RMSE (no term, |residual|) = {base:.0f} us\n")
    if len(shapes) < 3:
        print("WARNING: <3 distinct shapes -- LOSO not meaningful yet. Run Round 2 first "
              "(run_split_shape_r2.sh) and re-fold the DB.\n")

    results = []
    for name, (cols, knees) in CANDIDATES().items():
        if knees is not None:  # knee sweep (aniso)
            for c in knees:
                cc = aniso_cols(c)
                coefs, _ = fit(rows, cc)
                results.append((f"{name} c={c}", rmse(rows, cc, coefs),
                                loso_rmse(rows, cc), coefs))
        else:
            coefs, _ = fit(rows, cols)
            results.append((name, rmse(rows, cols, coefs), loso_rmse(rows, cols), coefs))

    results.sort(key=lambda t: t[2])  # rank by LOSO-RMSE (generalization)
    print(f"{'candidate term':34}{'in-RMSE':>9}{'LOSO-RMSE':>11}  coefs")
    for name, insamp, loso, coefs in results:
        cs = ", ".join(f"{c:.3e}" for c in coefs) if coefs else "-"
        print(f"{name:34}{insamp:>9.0f}{loso:>11.0f}  [{cs}]")
    print("\nLower LOSO-RMSE = generalizes better across held-out shapes. A form is only a "
          "candidate to ship if LOSO-RMSE << baseline AND it does not regress the balanced "
          "rows (check err on m,n<=8 separately before fitting a coefficient into CostParams).")


if __name__ == "__main__":
    main()
