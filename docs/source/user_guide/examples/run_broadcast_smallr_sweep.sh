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
# BROADCAST SMALL-ROWS SWEEP -- resolve the NON-MONOTONIC small-ROWS x large-COLS
# region of the broadcast effective-BW surface. The dense overnight sweep found the
# row-broadcast ops (bcast/mulbcast, operand b[1,C]) run FAST at 64x16384 (~129 GB/s)
# but SLOW at 256x16384 (~92), confirmed real (old & new builds agree within 1-2%).
# That non-monotonicity is under-sampled, so the cost model leaves ROWS<1024 on the
# flat 118 rate (FLAGGED). This sweep densely maps R in {64..1024} x C to settle
# whether 256 is a real dip or a sampling artifact, so we can model it properly.
#
# Same build/conditions as the main sweep. cores=32 (SENCORES=32). BENCH_REPS reps
# per point (noise protocol -> kernel_us_min/median/std/cv). ~160 configs x 7 reps.
#
#   bash docs/source/user_guide/examples/run_broadcast_smallr_sweep.sh
#   BENCH_REPS=9 bash .../run_broadcast_smallr_sweep.sh          # tighter noise
# Output: <repo-root>/haoyang_logs/bcast_smallr_<timestamp>.log (forward it + the
# folded notes/sweep_records.*).
# ============================================================================

set -u
trap 'echo "## INTERRUPTED"; exit 130' INT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/bcast_smallr_$(date +%Y%m%d_%H%M%S).log"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export BENCH_REPS="${BENCH_REPS:-7}"

echo "==== broadcast small-ROWS sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  reps: $BENCH_REPS" | tee -a "$LOG"

_emit() {
  local all out; all=$(cat)
  out=$(printf '%s\n' "$all" | grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY|^TIMING ')
  if printf '%s\n' "$out" | grep -q '^SUMMARY'; then printf '%s\n' "$out" | tee -a "$LOG"
  else { echo "SUMMARY $1 FAILED"; printf '%s\n' "$all" | grep -vE '^\s*$' | tail -6 | sed 's/^/FAILDIAG /'; } | tee -a "$LOG"; fi
}
runpw() {  # runpw <op> <rows> <cols>   cores=32, lx=0
  local _t0=$SECONDS
  echo "-- $1 [$2,$3]" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING=0 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "$1 [$2,$3]"
  echo "TIMING_RUN $1 [$2,$3] $((SECONDS - _t0))s" | tee -a "$LOG"
}

# Preflight: one tiny op to confirm the accelerator is up before the sweep.
PF=$(SENCORES=32 BENCH_OP=neg BENCH_ROWS=64 BENCH_COLS=64 BENCH_REPS=1 BENCH_WARMUP=1 \
  timeout -k 10 150 python "$PROFILE_OPS" 2>&1)
if ! printf '%s\n' "$PF" | grep -q '^SUMMARY .*kernel_us='; then
  echo "## PREFLIGHT FAILED -- device not up. Aborting." | tee -a "$LOG"
  printf '%s\n' "$PF" | tail -6 | sed 's/^/PREFLIGHT /' | tee -a "$LOG"; exit 1
fi

# Dense small-ROWS grid (incl. the 64-fast / 256-slow region) x COLS, all 4 broadcast ops.
# bcast/mulbcast = row-broadcast b[1,C] (the effect); copy/bcastcol = controls.
for op in bcast mulbcast bcastcol copy; do
  echo "## $op" | tee -a "$LOG"
  for R in 64 96 128 192 256 320 384 512 640 768 1024; do
    for C in 2048 4096 8192 16384; do runpw "$op" "$R" "$C"; done
  done
done

DB_SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)"
echo "==== DONE (${SECONDS}s) -> parsing into notes/sweep_records.* ====" | tee -a "$LOG"
python "$ROOT/notes/parse_sweep_logs.py" "$LOG" --current-sha "$DB_SHA" \
  || echo "## parse failed (run: python notes/parse_sweep_logs.py $LOG)"
echo "==== broadcast small-ROWS sweep COMPLETE -- forward $LOG ====" | tee -a "$LOG"
