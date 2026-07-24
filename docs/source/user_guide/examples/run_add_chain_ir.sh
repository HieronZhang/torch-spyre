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
# ADD-CHAIN IR CAPTURE -- the CONFIRMATORY experiment for the §3 add4 wrinkle.
#
# Finding (from timing data + the fuser code): at add4 the FUSED chain is on its
# clean linear trend (model within ~2 %); it is `add4_sep` -- the SEPARATE,
# multi-launch control -- that runs anomalously cheap. The fuser packs add4 as ONE
# ordinary left-associative bundle (4 inputs + 1 output = 5 tensors, within the
# per-bundle tensor limit; the first bundle split is at add5/add6, never add4), so
# there is NO add4-specific fusion structure. This sweep DUMPS the loop-level IR to
# prove that positively, structure-not-timing (1 rep each):
#
#   * bundle count per fused add{n}: count `async_compile.sdsc(` in the wrapper.
#     EXPECT add3 = add4 = 1 bundle; first split at add5 (tensor-pool on) or add6.
#     If add4 were specially split, THIS is where it would show -- it must not.
#   * op chain shape: grep the LoopLevel IR for the op list -- EXPECT a strictly
#     left-associative linear chain (op0 = arg0+arg1, op_k = buf_{k-1}+arg_{k+1}),
#     NO balanced tree (a+b)+(c+d), NO reassociation.
#   * separate path: add4_sep must emit 3 independent sdsc kernels chained through
#     HBM (host-orchestrated) -- confirming the cheapness is a launch-schedule
#     property of the separate control, not a device-kernel difference.
#
# All runs are LX_PLANNING=0 (intermediates materialise in HBM -> the regime the
# report's fig3b uses), ROWS=2048 COLS=4096. IR -> haoyang_logs/ir/addchain_*.txt;
# a summary (op, bundle_count) is teed to the run log. Forward haoyang_logs/ir/.
#
#   bash docs/source/user_guide/examples/run_add_chain_ir.sh
# Wall-clock: ~8 compiles, a few minutes. No HW timing needed (structure only).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs haoyang_logs/ir
LOG="haoyang_logs/addchain_ir_$(date +%Y%m%d_%H%M%S).log"
echo "==== ADD-CHAIN IR CAPTURE $(date) ====" | tee "$LOG"

# op -> expected #distinct external tensors for the FUSED chain (n inputs + 1 out).
# add3=4, add4=5, add5=6, add6=7. Bundle tensor limit is len(SEGMENT_OFFSETS)-1 (=6)
# or -2 (=5) with a pool -> first split lands at add5 (pool) or add6 (no pool).
for op in add3 add4 add5 add6 add3_sep add4_sep add5_sep add6_sep; do
  irf="haoyang_logs/ir/addchain_${op}.txt"
  echo "-- IR $op (-> $irf)" | tee -a "$LOG"
  SENCORES="${SENCORES:-32}" LX_PLANNING=0 SPYRE_DUMP_IR=1 \
    BENCH_OP="$op" BENCH_ROWS=2048 BENCH_COLS=4096 BENCH_REPS=1 BENCH_WARMUP=1 \
    timeout -k 30 "${ADDCHAIN_TIMEOUT:-400}" python "$PROFILE_OPS" > "$irf" 2>&1
  bundles=$(grep -c 'async_compile.sdsc(' "$irf" 2>/dev/null || echo 0)
  haveir=$(grep -qc 'LoopLevel IR - AFTER pre-scheduling' "$irf" 2>/dev/null && echo yes || echo NO)
  summ=$(grep -E '^SUMMARY' "$irf" 2>/dev/null | head -1)
  echo "   bundles(async_compile.sdsc)=$bundles  loop_ir=$haveir" | tee -a "$LOG"
  echo "   $summ" | tee -a "$LOG"
done

echo "==== DONE -- read haoyang_logs/ir/addchain_*.txt ====" | tee -a "$LOG"
echo "   EXPECT: add3,add4 bundles=1 ; first split (bundles>1) at add5 or add6 ;" | tee -a "$LOG"
echo "           add{n}_sep = n-1 separate sdsc kernels ; all chains left-associative." | tee -a "$LOG"
