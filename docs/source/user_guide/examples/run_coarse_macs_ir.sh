#!/usr/bin/env bash
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
#
# ============================================================================
# COARSE-MATMUL IR CAPTURE -- settle the `matmul_macs` semantics inconsistency.
#
# THE DEFECT, stated precisely (measured over every coarse record in
# notes/sweep_records.json, using TOTAL = B*M*K*N parsed from the run label):
#
#   op                     tiles_output_dim   TOTAL/recorded matmul_macs
#   matmul_row_tiling            True         1.0  at loop_trip 2/4/8/16  (n=60)  <-- TOTAL
#   mm_nested_m_k                True         == loop_trip                (n=20)  <-- PER-TILE
#   bmm_nested_b_k               True         == loop_trip                (n= 6)  <-- PER-TILE
#   matmul_k_tiling              False        == loop_trip                (n=30)  <-- PER-TILE
#   bmm_k_tiling                 False        == loop_trip                (n=18)  <-- PER-TILE
#   bmm_3d2d_k_tiling            False        == loop_trip                (n=14)  <-- PER-TILE
#
# So `matmul_row_tiling` is the ONE op whose macs already covers all tiles, and the
# cost model multiplies NOTHING by loop_trip (cost_model.py: `compute += matmul_macs /
# cores / (mac_peak*pt_eff)`).  Every per-tile op is therefore under-counting compute by
# loop_trip, which is the leading suspect for mm_nested_m_k's -31% signed error.
#
# WHY THIS NEEDS AN IR DUMP AND CANNOT BE FIXED OFFLINE:
#  1. `tiles_output_dim` does NOT discriminate -- matmul_row_tiling (TOTAL) and
#     mm_nested_m_k (PER-TILE) both report True.  No recorded feature separates them.
#  2. Deriving TOTAL from other features fails: M_dev*N_dev*(a_bytes/(dtype*M_dev))
#     matches TOTAL on only 298 of 738 rows, because for bmm `matmul_rows_per_core`
#     picks up the BATCH rather than M (a known hazard called out in the extractor's
#     own docstring in dump_cost_model.py:_matmul_features).
#  3. The existing dumps for these two ops are 1-2 line STUBS -- SPYRE_DUMP_IR never
#     fired for them (a real dump is ~1000 lines, cf. ir/bmm_layout_*.txt).
#
# WIDENED (session 5): this capture is now the gate on ALL of cat 6, because the same
# per-iteration-vs-per-loop ambiguity shows up THREE ways, not one:
#   1. matmul_macs   -- TOTAL for matmul_row_tiling, PER-TILE for every other coarse op.
#   2. a_bytes/b_bytes/rows_per_core -- PER-TILE while macs is TOTAL *in the same op*.
#   3. loop_factor   -- pinned at 1 while loop_trip is 2/4/8, so a per-iteration HBM round
#      trip is charged once per LOOP. Proof: mm_nested_m_k at a FIXED 2048^3 shape records
#      byte-IDENTICAL traffic (25,165,888 elems) at tiles=2 and tiles=4 while measured time
#      differs 1.56x (1096 vs 1705 us); and tiling that fixed matmul at all costs 2.9-4.5x.
# So ALSO record, for every arg of every op in the bundle: role, mem, elems, loop_factor, and
# the op's loop_trip -- not just the MAC count. The question to answer from the IR is simply:
# for each quantity, is it per ITERATION or per LOOP, and does loop_factor say so correctly?
#
# WHAT TO LOOK FOR in the resulting files.  In `_matmul_features`,
# `macs = out_elems * k_size` with `out_elems` from the committed DEVICE layout and
# `k_size = prod(reduction_ranges)`.  So compare, between the two ops at the same
# tile count:
#   * the ComputedBuffer's layout size  -> is it the FULL M x N, or M/tiles x N ?
#   * `reduction_ranges`                -> is it the FULL K, or K/tiles ?
# matmul_row_tiling coming out TOTAL means BOTH are full there; mm_nested_m_k coming out
# PER-TILE means at least one is divided.  That difference is the bug, and knowing which
# of the two factors moves tells us whether to fix the extractor or to add a feature.
#
# Cheap: 6 runs, ~2 min total (these compile in ~100-130 s only when profiling reps;
# here reps=1).  Run on the Spyre machine, then commit the ir/ files back.
#
#   bash docs/source/user_guide/examples/run_coarse_macs_ir.sh
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs haoyang_logs/ir
LOG="haoyang_logs/coarse_macs_ir_$(date +%Y%m%d_%H%M%S).log"
echo "==== COARSE-MACS IR CAPTURE $(date) ====" | tee "$LOG"

# The two ops that disagree, at matched tile counts, plus a t=1 control where both
# definitions coincide (ratio is 1.0 either way) so a broken dump is obvious.
for op in matmul_row_tiling mm_nested_m_k; do
  for t in 1 4 8; do
    irf="haoyang_logs/ir/coarsemacs_${op}_2048x2048x2048_t${t}.txt"
    echo "-- IR $op tiles=$t (-> $irf)" | tee -a "$LOG"
    SENCORES="${SENCORES:-32}" LX_PLANNING=0 SPYRE_DUMP_IR=1 \
      BENCH_OP="$op" BENCH_ROWS=2048 BENCH_COLS=2048 BENCH_N=2048 \
      BENCH_TILES="$t" BENCH_REPS=1 BENCH_WARMUP=1 \
      timeout -k 30 "${COARSE_IR_TIMEOUT:-400}" python "$PROFILE_OPS" > "$irf" 2>&1
    haveir=$(grep -qc 'LoopLevel IR - AFTER pre-scheduling' "$irf" 2>/dev/null && echo yes || echo NO)
    lines=$(wc -l < "$irf")
    echo "   loop_ir=$haveir  lines=$lines   <-- lines<10 means the dump did NOT fire" | tee -a "$LOG"
    grep -E '^SUMMARY' "$irf" 2>/dev/null | head -1 | tee -a "$LOG"
    grep -oE 'reduction_ranges=\[[^]]*\]' "$irf" 2>/dev/null | sort -u | head -3 | sed 's/^/   /' | tee -a "$LOG"
  done
done

echo "==== DONE -- compare layout size and reduction_ranges between the two ops ====" | tee -a "$LOG"
echo "Then: python3 notes/parse_sweep_logs.py is NOT needed; these are IR-only." | tee -a "$LOG"
