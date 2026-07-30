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
# BASELINE RE-CHECK -- are the 2447 recorded measurements still comparable?
#
# WHY. Every record in notes/sweep_records.json was measured on the `dev1` line,
# whose upstream base is ~Jul 3. The `layout211` branch sits on upstream Jul 27
# (22e63bb), so ~3.5 weeks of compiler/runtime changes sit in between -- among
# them Olivier's tiling_expr_to_device_expr (#3344, #3350), the coprime-stick
# work-division miscompile fix (#3340), scratchpad/boundary-clone changes
# (#3212) and physical-core-mapping consolidation (#3268). Any of those can move
# kernel time WITHOUT any cost-model change.
#
# If timings moved, old and new records are NOT directly comparable, and the
# layout A/B would be confounded: a difference attributed to the layout flag
# could just be the branch change. This script re-measures FIVE configurations
# that are already in the database (repeat-backed, cv < 0.4 %) and prints the
# delta against the recorded value.
#
# READ IT LIKE THIS
#   |delta| within ~2x the recorded cv  -> unchanged; the 2447 records stand.
#   a consistent shift in ONE category  -> that category's records are stale;
#                                          re-measure it before folding new data.
#   a broad shift across categories     -> re-baseline everything; do not mix
#                                          old and new records in one fit.
#
# Run with the layout flag UNSET -- this measures the branch, not the feature.
#
#   bash docs/source/user_guide/examples/run_baseline_recheck.sh
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/baseline_recheck_$(date +%Y%m%d_%H%M%S).log"

export BENCH_REPS="${BENCH_REPS:-7}"
unset SPYRE_MATMUL_PREFERRED_LAYOUT 2>/dev/null || true

echo "==== BASELINE RE-CHECK  $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD)  reps=$BENCH_REPS  layout flag: UNSET" | tee -a "$LOG"

# label|recorded kernel_us_min|recorded cv|env...
run_one() {
  local label="$1" recorded="$2" cv="$3"; shift 3
  echo "" | tee -a "$LOG"
  echo "-- $label   (recorded ${recorded} us, cv ${cv}%)" | tee -a "$LOG"
  out=$(env "$@" timeout -k 30 "${RUN_TIMEOUT:-400}" python "$PROFILE_OPS" 2>&1)
  line=$(printf '%s\n' "$out" | grep -E '^SUMMARY' | head -1)
  if [ -z "$line" ]; then
    echo "   FAILED -- no SUMMARY" | tee -a "$LOG"
    printf '%s\n' "$out" | tail -4 | sed 's/^/   /' | tee -a "$LOG"
    return
  fi
  printf '%s\n' "$line" | tee -a "$LOG"
  now=$(printf '%s\n' "$line" | grep -oE 'kernel_us_min=[0-9.]+' | cut -d= -f2)
  [ -z "$now" ] && return
  python3 -c "
now=float('$now'); rec=float('$recorded'); cv=float('$cv')
d=(now-rec)/rec*100
tol=max(2*cv, 1.0)
verdict='UNCHANGED' if abs(d) <= tol else ('MOVED' if abs(d) > 3*tol else 'marginal')
print(f'   recorded {rec:9.1f} -> now {now:9.1f}   delta {d:+6.2f}%   (tol +/-{tol:.2f}%)   {verdict}')
" | tee -a "$LOG"
}

# Five configs already in the database, one per category, all reps=7 cv<0.4
run_one "mmwd 2048x2048x4096 split 4x8 [plain matmul CONTROL]" 767.4 0.32 \
  BENCH_OP=mmwd BENCH_ROWS=2048 BENCH_COLS=2048 BENCH_N=4096 WD_M=4 WD_N=8 WD_K=1 SENCORES=32

run_one "bmm_wd cores=8 1024x2048x1024 [cat 4 batched matmul]" 6348.9 0.34 \
  BENCH_OP=bmm_wd BENCH_B=4 BENCH_ROWS=1024 BENCH_COLS=2048 BENCH_N=1024 WD_B=1 WD_M=4 WD_N=8 WD_K=1 SENCORES=8

run_one "matmul_row_tiling M=8192 K=2048 N=2048 tiles=8 [cat 6 coarse]" 1353.0 0.37 \
  BENCH_OP=matmul_row_tiling BENCH_ROWS=8192 BENCH_COLS=2048 BENCH_N=2048 BENCH_TILES=8 SENCORES=32

run_one "softmax_row_tiling 2048x512 tiles=8 cores=1 [cat 5 fused]" 589.6 0.16 \
  BENCH_OP=softmax_row_tiling BENCH_ROWS=2048 BENCH_COLS=512 BENCH_TILES=8 SENCORES=1

run_one "transpose_outer R=16384 C=2048 M=8 [cat 2 transport]" 13636.1 0.27 \
  BENCH_OP=transpose_outer BENCH_ROWS=16384 BENCH_COLS=2048 TO_MID=8 SENCORES=32

{
  echo ""
  echo "==== WHAT TO DO WITH THIS ===="
  echo "  all UNCHANGED     -> the 2447 records stand; proceed to the layout A/B."
  echo "  coarse only MOVED -> expected if a coarse-tiling fix landed; re-measure cat 6"
  echo "                       before using those records, but the rest are fine."
  echo "  broad MOVE        -> re-baseline: do not mix pre- and post-branch records"
  echo "                       in a single fit, and note the split in cost_model_status.md."
} | tee -a "$LOG"
