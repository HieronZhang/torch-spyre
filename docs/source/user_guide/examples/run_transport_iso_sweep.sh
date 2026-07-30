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
# TRANSPORT ISOLATION SWEEP (cost-model category 2).
#
# The dense R×C grid (XPORT in run_outlier_sweep.sh) already showed that the
# transport ops' effective bandwidth falls with the stick-plane count sp = C/64
# (the strided per-row stick gather) and mildly with R. This sweep isolates the
# pieces that grid CANNOT separate, so the model is MECHANISTIC, not curve-fit:
#
#   TOMID  transpose_outer's middle (outer-swap) dim M is fixed at 8 in the grid.
#          [R,M,C]->[M,R,C] is a transpose of an R×M array of C-wide stick-rows.
#          Sweep M at fixed (R,C): if effBW depends on M -> it is a real 2D
#          block-transpose (M belongs in the model); if flat in M -> only sp and
#          R matter (a pure strided gather). THIS IS THE DECISIVE EXPERIMENT.
#   SPSAT  larger stick-plane counts (C up to 32768, sp up to 512) -> is the sp
#          droop saturating (a floor) or still falling? (cat0's current floor-clamp
#          form over-predicts at large C.)
#   RSHAPE R at fixed sp, out to both extremes (256 and 32768) -> map the U-shape
#          (both very small and very large R measured slow at fixed sp in the grid).
#   XIR    a handful of runs with the loop-level IR saved, to CONFIRM the access
#          pattern (which tensor streams contiguous, the gather stride) directly.
#
# All runs: cores=32, BENCH_REPS back-to-back profiled measurements (noise
# protocol -> kernel_us_min/median/std/cv), IR teed per run to haoyang_logs/ir/.
# Output: <repo-root>/haoyang_logs/transport_iso_<timestamp>.log. Fold with
# notes/parse_sweep_logs.py; the middle dim M is recovered from the feats
# (out_elems = R*M*C), so no parser change is needed.
#
#   bash docs/source/user_guide/examples/run_transport_iso_sweep.sh            # all
#   SECTIONS="TOMID" bash .../run_transport_iso_sweep.sh                       # one section
#   BENCH_REPS=5 bash .../run_transport_iso_sweep.sh                           # fewer reps
# 46 runs total (TOMID 21, SPSAT 6, RSHAPE 14, XIR 5). ~30-45 min at <1 min/run.
# ============================================================================

set -u
trap 'echo "## INTERRUPTED -- aborting sweep (SIGINT)"; exit 130' INT
trap 'echo "## TERMINATED -- aborting sweep (SIGTERM)"; exit 143' TERM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs haoyang_logs/ir
LOG="haoyang_logs/transport_iso_$(date +%Y%m%d_%H%M%S).log"
SECTIONS="${SECTIONS:-TOMID SPSAT RSHAPE XIR}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export BENCH_REPS="${BENCH_REPS:-7}"

echo "==== transport isolation sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS  reps: $BENCH_REPS" \
  | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {  # keep the parseable lines; on a crash, surface WHY (mirror run_outlier_sweep.sh)
  local all out; all=$(cat)
  out=$(printf '%s\n' "$all" | grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY|^TIMING ')
  if printf '%s\n' "$out" | grep -q '^SUMMARY'; then
    printf '%s\n' "$out" | tee -a "$LOG"
  else
    { echo "SUMMARY $1 FAILED"
      printf '%s\n' "$all" | grep -vE '^\s*$' | tail -6 | sed 's/^/FAILDIAG /'; } | tee -a "$LOG"
  fi
}
sect() { SECT="$1"; SECT_T0=$SECONDS; echo "## $1 -- ${2:-}" | tee -a "$LOG"; }
esect() { echo "## $SECT DONE in $((SECONDS - SECT_T0))s" | tee -a "$LOG"; }

# runtx <op> <rows> <cols>  -- transport op at cores=32, middle dim = TO_MID (default 8 for
# transpose_outer; ignored by the 2D ops). IR teed per run so the access pattern is on record.
runtx() {
  local op=$1 R=$2 C=$3 M="${TO_MID:-8}" _t0=$SECONDS
  local tag="$op"; [[ "$op" == transpose_outer ]] && tag="${op}_m${M}"
  local irf="haoyang_logs/ir/${tag}_${R}x${C}.txt"
  local lbl="$op [$R,$C]"; [[ "$op" == transpose_outer ]] && lbl="$op [$R,$C] m=$M"
  echo "-- $lbl (IR -> $irf)" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING=0 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 TO_MID="$M" \
    BENCH_OP="$op" BENCH_ROWS="$R" BENCH_COLS="$C" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | tee "$irf" \
    | _emit "$lbl"
  echo "TIMING_RUN $lbl $((SECONDS - _t0))s" | tee -a "$LOG"
}
# runto <rows> <cols> <M>  -- transpose_outer with a chosen middle dim M (the crux knob).
runto() { TO_MID="$3" runtx transpose_outer "$1" "$2"; }

# ---- preflight ----
if [[ -z "${SKIP_PREFLIGHT:-}" ]]; then
  echo "-- preflight: trivial neg to check the Spyre device is up ..." | tee -a "$LOG"
  PF=$(SENCORES=32 BENCH_OP=neg BENCH_ROWS=64 BENCH_COLS=64 BENCH_REPS=1 BENCH_WARMUP=1 \
    timeout -k 10 150 python "$PROFILE_OPS" 2>&1)
  if printf '%s\n' "$PF" | grep -q '^SUMMARY .*kernel_us='; then
    echo "-- preflight OK (device reachable)" | tee -a "$LOG"
  else
    { echo "## PREFLIGHT FAILED -- a trivial op did not run; ABORTING before the sweep."
      printf '%s\n' "$PF" | grep -vE '^\s*$' | tail -8 | sed 's/^/PREFLIGHT /'
      echo "## Likely the Spyre accelerator is busy/wedged (VFIO). Recover (no root needed):"
      echo "##   ps -u \$(whoami) -o pid,stat,cmd | grep -Ei 'python|profile_ops'  # find it"
      echo "##   kill -9 <pid>            # works on YOUR own process unless it is in D state"
      echo "## (Set SKIP_PREFLIGHT=1 to bypass this check.)"; } | tee -a "$LOG"
    exit 1
  fi
fi

# ============ TOMID (crux): does the outer-swap count M drive effBW? ==========
# transpose_outer [R,M,C] -> [M,R,C]. Fixed (R,C); sweep M. Flat-in-M => strided gather
# (sp,R only). Rising cost with M => a real 2D block-transpose (M belongs in the model).
has TOMID && { sect TOMID "transpose_outer middle-dim M sweep at fixed (R,C)"
  for M in 1 2 4 8 16 32 64; do runto 2048 2048 "$M"; done   # sp=32, mid-R
  for M in 1 2 4 8 16 32;    do runto 2048 8192 "$M"; done   # sp=128 (M=64 -> 2 GB, skipped)
  for M in 1 4 16 64;        do runto  512 2048 "$M"; done   # sp=32, small R (M×R interaction)
  for M in 1 4 16 64;        do runto 8192 2048 "$M"; done   # sp=32, large R
  esect; }

# ============ SPSAT: larger stick-plane count -> does the sp droop saturate? ==
has SPSAT && { sect SPSAT "C up to 32768 (sp up to 512): is the effBW droop bottoming out?"
  for op in cat0 transpose_outer cat1 transpose; do runtx "$op" 2048 32768; done  # sp=512
  for op in cat0 transpose_outer;                do runtx "$op" 1024 32768; done
  esect; }

# ============ RSHAPE: R at fixed sp, both extremes -> map the U-shape =========
has RSHAPE && { sect RSHAPE "R to 256 and 32768 at fixed sp -> why are both ends slow?"
  for op in cat0 transpose_outer cat1; do            # sp=32, extend the R ends
    runtx "$op"   256 2048
    runtx "$op" 32768 2048
  done
  for op in cat0 transpose_outer; do                 # sp=128, fill the interior of the U
    for R in 256 1024 4096 16384; do runtx "$op" "$R" 8192; done
  done
  esect; }

# ============ XIR: confirm the access pattern from the loop-level IR ==========
# A few runs whose IR is saved and eyeballed (which tensor streams contiguously, the gather
# stride). transpose (fast, within-stick) vs the plane-scatter ops, and M=8 vs M=32.
has XIR && { sect XIR "IR capture: access pattern of each transport op (see haoyang_logs/ir/)"
  runtx transpose       2048 2048
  runtx cat0            2048 2048
  runtx cat1            2048 2048
  runto                 2048 2048 8
  runto                 2048 2048 32
  esect; }

echo "==== transport isolation sweep DONE in ${SECONDS}s -- $LOG ====" | tee -a "$LOG"
