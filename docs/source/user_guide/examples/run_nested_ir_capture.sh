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
# NESTED-LOOP IR CAPTURE -- the input needed to redesign `loop_factor` per LEVEL.
#
# READ THIS FIRST: SPYRE_DUMP_IR WAS BROKEN. Every previous attempt to dump these
# ops produced a ~51-line stub, and the reason was not "the dump did not fire" -- it
# ABORTED THE COMPILE:
#
#   File "torch_spyre/_inductor/dump_loop_ir.py", line 48, in dump_loop_ir
#       from .passes import _format_operations
#   InductorError: ImportError: cannot import name '_format_operations'
#
# The formatter had moved to `pass_utils.format_operations`, and the import sat
# OUTSIDE the try/except that is supposed to guarantee "a debug dump must never break
# compilation", so the ImportError escaped. Both are fixed in `dump_loop_ir.py`
# (import corrected AND moved inside the guard). THIS SCRIPT REQUIRES THAT FIX --
# it verifies the dump actually produced loop metadata and fails loudly if not,
# which is exactly the check the earlier capture lacked.
#
# WHY WE NEED IT. `ArgTraffic.loop_factor` is a single scalar per arg. That is
# sufficient for a SINGLE-level loop (matmul_row_tiling tiles only M, so B is
# invariant and re-read L times -- IR-proven, and the shipped `_loop_reread_bytes`
# term is built on it). It is NOT sufficient for a NESTED loop, where
# `CoarseTileInfo` carries per-level lists:
#
#     loop_group_id = (0, 0)          two levels
#     loop_count    = [L1, L2]        one trip count per level
#     loop_tiled_dims = [[...],[...]] one tiled-dim list PER LEVEL
#
# because an operand can be INVARIANT at one level and ADVANCE at another. For
# `mm_nested_m_k` (M outer, K inner) B[K,N] is invariant w.r.t. M but SLICED by K,
# so its true multiplier is L_M, not L_M*L_K and not 1. Applying the single-level
# row-tiling rule to it makes the model WORSE (RMS 46.0 % -> 142.8 %), which is why
# it is currently excluded from the fix. The correct general form is
#
#     factor(arg) = PRODUCT over levels L of
#                     ( loop_count[L] if arg's index contains NO tiled symbol of
#                       level L, else 1 )
#
# and this capture is the evidence needed to implement and verify it.
#
# WHAT TO EXTRACT from each dump, per op:
#   1. `loop_info=CoarseTileInfo(...)`  -> loop_group_id, loop_count, and the
#      per-level loop_tiled_dims / loop_tiled_reduction_dims.
#   2. The `inner_fn` index expression of EVERY `ops.load(...)` -> which loop
#      symbols each operand's address actually depends on.
#   3. `op_it_space_splits` and `dim_hints` -> which symbol is which named dim.
#   4. `allocation={...}` -> whether the buffer is LX-resident (an "lx" key) or HBM.
#      This also independently re-checks the finding that a coarse matmul's operands
#      are never LX-pinned.
# Cross-referencing 1 and 2 gives, per arg per level, advance-or-repeat -- the same
# derivation that settled matmul_row_tiling, generalised.
#
# OPS COVERED
#   mm_nested_m_k    the nested case the redesign is for (M outer, K inner)
#   matmul_k_tiling  single-level REDUCTION tiling. Needed because `loop_factor > 1`
#                    already appears on its INPUTS today, which is suspect: under
#                    K-tiling each iteration takes a fresh K-slice, so the inputs
#                    should ADVANCE and only the accumulator repeats. Charging those
#                    existing factors moved matmul_k 7.9 % -> 32.4 %, so the recorded
#                    semantics need settling from IR, not guessed.
#   matmul_row_tiling  t=4 CONTROL. We already have a good dump of this and know the
#                    answer, so it proves the capture is working before trusting the
#                    two unknown ops.
#
# Cheap: 7 compiles, reps=1, no profiling. Minutes, not hours.
#
#   bash docs/source/user_guide/examples/run_nested_ir_capture.sh
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs haoyang_logs/ir
LOG="haoyang_logs/nested_ir_$(date +%Y%m%d_%H%M%S).log"

echo "==== NESTED IR CAPTURE  $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD)" | tee -a "$LOG"

# Guard: the capture is worthless without the dump_loop_ir fix.
if grep -q 'from .passes import _format_operations' torch_spyre/_inductor/dump_loop_ir.py; then
  echo "ABORT: dump_loop_ir.py still imports the removed 'passes._format_operations'." | tee -a "$LOG"
  echo "       SPYRE_DUMP_IR=1 will abort every compile. Apply the fix first." | tee -a "$LOG"
  exit 1
fi

M=${M:-4096}; K=${K:-2048}; N=${N:-2048}
ok=0; bad=0

capture() {
  local op="$1" t="$2"
  local irf="haoyang_logs/ir/nested_${op}_${M}x${K}x${N}_t${t}.txt"
  echo "" | tee -a "$LOG"
  echo "-- $op tiles=$t -> $irf" | tee -a "$LOG"
  SENCORES="${SENCORES:-32}" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$op" BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
    BENCH_TILES="$t" BENCH_REPS=1 BENCH_WARMUP=1 \
    timeout -k 30 "${IR_TIMEOUT:-600}" python "$PROFILE_OPS" > "$irf" 2>&1

  # Verify the dump ACTUALLY produced loop metadata. The previous capture skipped
  # this and shipped 51-line stubs that looked like data.
  local lines li fail
  lines=$(wc -l < "$irf")
  li=$(grep -c 'loop_info=' "$irf" 2>/dev/null || echo 0)
  fail=$(grep -c 'ImportError\|InductorError\|FAILED' "$irf" 2>/dev/null || echo 0)
  if [ "$li" -gt 0 ]; then
    ok=$((ok+1))
    echo "   OK  lines=$lines  loop_info stamps=$li" | tee -a "$LOG"
    grep -ohE 'loop_info=CoarseTileInfo\([^)]*\)[^)]*\)*' "$irf" | sort -u | head -4 \
      | sed 's/^/     /' | tee -a "$LOG"
    grep -ohE 'ops\.load\([^)]*\)' "$irf" | sort -u | head -6 | sed 's/^/     /' | tee -a "$LOG"
  else
    bad=$((bad+1))
    echo "   *** NO loop_info (lines=$lines, error markers=$fail) ***" | tee -a "$LOG"
    grep -E 'ImportError|InductorError|FAILED' "$irf" | head -2 | sed 's/^/     /' | tee -a "$LOG"
  fi
}

# Control first -- if this one lacks loop_info the capture itself is broken.
capture matmul_row_tiling 4
capture mm_nested_m_k 2
capture mm_nested_m_k 4
capture mm_nested_m_k 8
capture matmul_k_tiling 2
capture matmul_k_tiling 4
capture matmul_k_tiling 8

{
  echo ""
  echo "==== RESULT: $ok captured, $bad empty ===="
  echo "The CONTROL (matmul_row_tiling t=4) must show loop_info with"
  echo "  loop_group_id=(0,), loop_count=[4], loop_tiled_dims=[[0]]"
  echo "and two ops.load lines whose indices differ in whether they contain the"
  echo "tiled symbol. If the control is empty, nothing else in this run is usable."
  echo ""
  echo "For mm_nested_m_k expect loop_group_id=(0, 0) and loop_count=[L1, L2] with"
  echo "TWO per-level entries in loop_tiled_dims / loop_tiled_reduction_dims. Record,"
  echo "for each operand, which level's tiled symbols appear in its index -- that is"
  echo "the per-level advance/repeat table the new loop_factor needs."
} | tee -a "$LOG"
