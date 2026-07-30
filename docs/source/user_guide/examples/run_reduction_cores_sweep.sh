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
# REDUCTION BW(cores) SWEEP -- the deciding experiment for the §5 g(cores) derate.
#
# The shipped g(cores) reduction-bandwidth factor (cost_model.py red_bw_cores_g:
# {1:0.11,2:0.22,4:0.43,8:0.54,16:0.54,32:1}) was calibrated on SINGLE-SHOT (reps=None)
# low-core points from old logs, at essentially ONE shape per aspect ratio. Two open
# questions the adversarial review flagged:
#   (1) Is the c8/c16 PLATEAU (g8~=g16~=0.54) real, or single-shot scatter? The repeat-
#       backed mmwd (matmul) sweep shows a smooth monotone climb with NO plateau -- so the
#       plateau may be noise. This sweep runs REPS>=7 to settle it.
#   (2) Is g(cores) SHAPE-independent? We only have read@[8192,2048] and amax/sumrow/mean@
#       [2048,8192]. This sweeps BOTH aspect ratios for EACH op to confirm g collapses.
#
# It forces each row-reduction onto {1,2,4,8,16,32} cores via SENCORES and measures the
# effective BW = (R+W)/time with the noise protocol. Fold, then recompute g(cores)=
# BW(c)/BW(32) per (op,shape) and check: does the plateau persist under reps? do the two
# aspect ratios agree? Update red_bw_cores_g (esp. c8/c16) from the repeat-backed result.
#
#   bash docs/source/user_guide/examples/run_reduction_cores_sweep.sh
#   BENCH_REPS=15 bash .../run_reduction_cores_sweep.sh        # tighter
# ~48 runs (4 ops x 2 shapes x 6 core counts); the profiled region is small, reps are cheap.
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/reduction_cores_$(date +%Y%m%d_%H%M%S).log"
export BENCH_REPS="${BENCH_REPS:-7}"
echo "==== REDUCTION-CORES $(date)  git:$(git rev-parse --short HEAD 2>/dev/null)  reps:$BENCH_REPS ====" | tee "$LOG"

_emit() {
  local all out; all=$(cat)
  out=$(printf '%s\n' "$all" | grep -E '^IO |^MODEL |^SUMMARY|^TIMING ')
  if printf '%s\n' "$out" | grep -q '^SUMMARY'; then printf '%s\n' "$out" | tee -a "$LOG"
  else { echo "SUMMARY $1 FAILED"; printf '%s\n' "$all" | grep -vE '^\s*$' | tail -6 | sed 's/^/FAILDIAG /'; } | tee -a "$LOG"; fi
}
runred() {  # op ROWS COLS cores
  local op=$1 R=$2 C=$3 c=$4 _t0=$SECONDS
  echo "-- $op ROWS=$R COLS=$C sencores=$c" | tee -a "$LOG"
  SENCORES="$c" SPYRE_DUMP_COST=1 BENCH_OP="$op" BENCH_ROWS="$R" BENCH_COLS="$C" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "$op ROWS=$R COLS=$C cores=$c"
  echo "TIMING_RUN $op $R/$C c$c $((SECONDS-_t0))s" | tee -a "$LOG"
}

# preflight
PF=$(SENCORES=32 BENCH_OP=neg BENCH_ROWS=64 BENCH_COLS=64 BENCH_REPS=1 BENCH_WARMUP=1 \
  timeout -k 10 150 python "$PROFILE_OPS" 2>&1)
printf '%s\n' "$PF" | grep -q '^SUMMARY .*kernel_us=' \
  || { echo "## PREFLIGHT FAILED -- device busy/wedged; ABORTING." | tee -a "$LOG"; \
       printf '%s\n' "$PF" | tail -6 | sed 's/^/PREFLIGHT /' | tee -a "$LOG"; exit 1; }

# Each op at BOTH aspect ratios (2048x8192 and 8192x2048), forced onto 1..32 cores.
for op in read amax sumrow mean; do
  for shape in "2048 8192" "8192 2048"; do
    set -- $shape
    for c in 1 2 4 8 16 32; do runred "$op" "$1" "$2" "$c"; done
  done
done
echo "==== REDUCTION-CORES DONE in ${SECONDS}s ($(date)) -- fold + recompute g(cores)=BW(c)/BW(32) per (op,shape); check the c8/c16 plateau under reps and cross-shape agreement ====" | tee -a "$LOG"
