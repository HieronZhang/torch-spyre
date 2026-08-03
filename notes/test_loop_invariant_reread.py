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

"""OFFLINE test of the loop-invariant re-read fix -- no hardware needed.

THE DEFECT (proven from IR, not inferred from timing). In
``haoyang_logs/ir/coarsemm_matmul_row_tiling_2048x2048x2048_t4.txt`` the coarse-tiled
matmul's ``inner_fn`` reads::

    i0, i1 = index                                # ranges=[2048, 2048]
    tmp0 = ops.load(arg0_1, r0_0 + 2048 * i0)     # A[M,K] -- index CONTAINS i0
    tmp1 = ops.load(arg1_1, i1   + 2048 * r0_0)   # B[K,N] -- index has NO i0

and the op carries ``loop_tiled_dims=[[0]]`` with
``DimHint(dim_names=['M'], loop_var=d0)``, so the tiled symbol is ``i0``. A therefore
ADVANCES with the loop (each iteration touches a fresh row-tile) while B is
loop-INVARIANT: every iteration re-enters the SAME K x N operand.

The extractor charges B ONCE, because ``loop_factor`` is computed as two PER-OP
scalars (``out_factor`` / ``in_factor`` in ``dump_cost_model._matmul_features``'s
caller) rather than per arg. Consequence: the loop trip count never enters the memory
term, and the model is FLAT in L -- its prediction for the 2048^3 shape is
byte-identical (358.205 us) at L=4 and L=8 while measurement grows 1.52x.

WHY THIS IS TESTABLE OFFLINE. ``cost_model.py`` is pure Python, and the recorded
``feats`` in ``sweep_records.json`` carry per-arg ``loop_factor`` -- which
``_fused_hbm_bytes`` already applies per arg (``a.elems * a.loop_factor``). So the
whole fix can be exercised by patching the recorded features and re-scoring.

WHAT THIS SCRIPT DOES NOT DO. It does not decide the RATE the re-read is charged at.
Charging it at the full HBM rate is the strong hypothesis; the measured marginal cost
(~40 us per extra B-read vs ~128 us for a full HBM read of B) says it is cheaper, i.e.
scratchpad residency is neither free nor absent. This script reports the overshoot so
that discount can be quantified rather than guessed.

    python3 notes/test_loop_invariant_reread.py
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_model as em  # noqa: E402

cm = em.cm

RECORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_records.json")

# Ops whose coarse loop tiles an OUTPUT dim (M), making the second matmul operand
# (the K x N weight) loop-invariant. Verified against the IR dump for
# matmul_row_tiling; mm_nested_m_k / bmm_nested_b_k tile M the same way but their IR
# has NOT been dumped, so they are reported separately and flagged.
IR_VERIFIED = {"matmul_row_tiling"}
SAME_SHAPE_UNVERIFIED = {"mm_nested_m_k", "bmm_nested_b_k"}


def load_rows():
    raw = json.load(open(RECORDS))
    rows = raw if isinstance(raw, list) else raw.get("records", [])
    return [r for r in rows if em.in_scope(r)]


def invariant_input(op):
    """The loop-invariant operand of a matmul whose loop tiles the ROW (M) dim.

    For ``a @ b`` the reads are ``arg0_1`` (A, index contains the tiled symbol i0) and
    ``arg1_1`` (B, index does not). Offline we cannot re-read the index expression, so
    we identify B by graph-input NAME ORDER, which matches the IR dump. Returns None
    when the op has anything other than exactly two external ``arg*`` inputs, so a
    fused or rewritten bundle is skipped rather than guessed at.
    """
    ins = [
        a
        for a in op.args
        if a.role == "input" and a.mem == "hbm" and a.name.startswith("arg")
    ]
    if len(ins) != 2:
        return None
    ins.sort(key=lambda a: a.name)
    return ins[-1]


def score(rows, alpha):
    """Re-score with the invariant operand charged (1 + alpha*(L-1)) times.

    alpha = 0.0 -> today's model (B read once, the defect).
    alpha = 1.0 -> the pure mechanism (B genuinely re-read from HBM every iteration).
    Between: the fraction of each re-read that reaches HBM rather than being served
    from on-chip/LX residency.
    """
    params = em.make_params({})
    out = []
    for rec, base_ops in rows:
        ops = [cm.op_from_dict(d) for d in rec["feats"]]
        mm = next((o for o in ops if getattr(o, "is_matmul", False)), None)
        if mm is None:
            continue
        L = int(getattr(mm, "loop_trip", 1) or 1)
        b = invariant_input(mm)
        if b is None:
            continue
        if L > 1:
            b.loop_factor = 1.0 + alpha * (L - 1)
        try:
            pred = cm.predict_ops(ops, params) / 1000.0
        except Exception:  # noqa: BLE001
            continue
        meas = rec["kernel_us_min"]
        out.append((rec, L, meas, pred, (pred - meas) / meas * 100.0))
    return out


def report(tag, res):
    if not res:
        print(f"  {tag}: no rows")
        return
    errs = [e for *_, e in res]
    rms = (sum(e * e for e in errs) / len(errs)) ** 0.5
    print(
        f"  {tag:>26}  n={len(res):<4} RMS={rms:6.1f}%  mean={st.mean(errs):+6.1f}%  "
        f">10%={sum(1 for e in errs if abs(e) > 10):<4}"
    )
    return rms


def main():
    rows_all = load_rows()
    ladder = {}
    for group, names in (
        ("IR-VERIFIED", IR_VERIFIED),
        ("UNVERIFIED", SAME_SHAPE_UNVERIFIED),
    ):
        sel = [
            (r, None)
            for r in rows_all
            if r.get("feats")
            and r.get("kernel_us_min")
            and any(n in str(r.get("label", "")) or n == r.get("op") for n in names)
        ]
        print(f"\n=== {group}  ({', '.join(sorted(names))})  {len(sel)} records ===")
        if not sel:
            continue
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            res = score(sel, alpha)
            report(f"alpha={alpha:.2f}", res)
            ladder[(group, alpha)] = res

        # Per-L breakdown at the two ends, so a global RMS win cannot hide a
        # per-L regression (the bug's whole signature is its L dependence).
        print(f"\n  per-L signed error, {group}:")
        print(
            f"    {'L':>4} {'n':>4} {'alpha=0 (today)':>18} {'alpha=1 (pure HBM)':>20}"
        )
        for L in sorted({r[1] for r in ladder[(group, 0.0)]}):
            a0 = [e for _, ll, _, _, e in ladder[(group, 0.0)] if ll == L]
            a1 = [e for _, ll, _, _, e in ladder[(group, 1.0)] if ll == L]
            if not a0:
                continue
            print(
                f"    {L:>4} {len(a0):>4} {st.mean(a0):>+17.1f}% "
                f"{(st.mean(a1) if a1 else float('nan')):>+19.1f}%"
            )

    print(
        "\nREAD IT LIKE THIS:\n"
        "  alpha=0 reproduces today's model, so its per-L ladder must match the known\n"
        "  -0.9/-6.4/-9.5/-23.4/-58.8 %. If alpha=1 flips the high-L rows from strongly\n"
        "  NEGATIVE to strongly POSITIVE, the counting fix is real but the HBM rate\n"
        "  over-charges it -- i.e. the re-read is partly served on-chip, and the\n"
        "  crossing alpha measures how much. That crossing is a MEASUREMENT of\n"
        "  residency, not a fitted fudge, only if it is stable across L; if the best\n"
        "  alpha itself drifts with L, residency is capacity-dependent and needs a\n"
        "  working-set model rather than a constant."
    )


if __name__ == "__main__":
    main()
