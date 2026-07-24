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
# FLASH-ATTENTION RE-SWEEP (cost-model category 7) -- fixed after the overnight run.
#
# WHY THE OVERNIGHT FLASH RUNS FAILED (diagnosed): the swept work_div hints were
# INVALID on two independent counts, so the compiler rejected them
# (InductorError: "work_division_hint: buf5 dim d0 size=2 is not evenly divisible
# by split=4" etc.):
#   1. CORE OVER-SUBSCRIPTION. The default "H:4,Lq:8,Lk:8" multiplies to 4*8*8=256
#      intra-tile cores, but a Spyre accelerator has 32. The work_div product per op
#      must be <= 32.
#   2. NON-DIVISIBLE SPLIT. A split must divide the PER-TILE size of the dim for
#      EVERY buffer that uses it, including small softmax intermediates. With H=32,
#      htiles=8 the per-tile H is only 32/8=4, and some intermediate (buf5) is as
#      small as 2 -- so a split of 8 (or even 4) does not divide it and is rejected.
#
# FIX HERE: sweep only work_div whose product is <= 32 (guarded), with SMALL splits
# (<=4) that divide the per-tile dims, and capture the loop IR for the survivors so
# the exact intermediate-buffer constraint (buf5's dims) is finally visible. Invalid
# configs are SKIPPED with a clear reason (the guard), not run. We are NOT modeling
# flash attn yet -- this is data + IR capture (see notes/flash_attn_hints.md).
#
#   bash docs/source/user_guide/examples/run_flash_resweep.sh
# Output: <repo-root>/haoyang_logs/flash_resweep_<timestamp>.log + IR in
# haoyang_logs/ir/. Fold with notes/parse_sweep_logs.py.
# ============================================================================

set -u
trap 'echo "## INTERRUPTED (SIGINT)"; exit 130' INT
trap 'echo "## TERMINATED (SIGTERM)"; exit 143' TERM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs haoyang_logs/ir
LOG="haoyang_logs/flash_resweep_$(date +%Y%m%d_%H%M%S).log"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export BENCH_REPS="${BENCH_REPS:-7}"
echo "==== flash re-sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  reps: $BENCH_REPS" | tee -a "$LOG"

_emit() {
  local all out; all=$(cat)
  out=$(printf '%s\n' "$all" | grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY|^TIMING ')
  if printf '%s\n' "$out" | grep -q '^SUMMARY'; then printf '%s\n' "$out" | tee -a "$LOG"
  else { echo "SUMMARY $1 FAILED"; printf '%s\n' "$all" | grep -vE '^\s*$' | tail -6 | sed 's/^/FAILDIAG /'; } | tee -a "$LOG"; fi
}
# _wd_product H:2,Lq:4,Lk:2 -> 16
_wd_product() { local p=1 part; for part in ${1//,/ }; do p=$(( p * ${part#*:} )); done; echo "$p"; }

runfa() {  # runfa <H> <Lq> <Lk> <D> <htiles> <qtiles> <ktiles> <wd>  -- work_div GUARDED <=32
  local H=$1 Lq=$2 Lk=$3 D=$4 ht=$5 qt=$6 kt=$7 wd=$8 _t0=$SECONDS
  local prod; prod=$(_wd_product "$wd")
  if (( prod > 32 )); then
    echo "SKIP flash wd=$wd product=$prod > 32 cores (H=$H ht=$ht Lq=$Lq qt=$qt)" | tee -a "$LOG"; return 0
  fi
  local wdtag="${wd//:/}"; wdtag="${wdtag//,/-}"
  local irf="haoyang_logs/ir/flashRS_H${H}_Lq${Lq}_Lk${Lk}_h${ht}q${qt}k${kt}_${wdtag}.txt"
  echo "-- flash_attn H=$H Lq=$Lq Lk=$Lk D=$D htiles=$ht qtiles=$qt ktiles=$kt wd=$wdtag prod=$prod (IR -> $irf)" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING=1 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=flash_attn FA_H="$H" FA_LQ="$Lq" FA_LK="$Lk" FA_D="$D" \
    FA_H_TILES="$ht" FA_LQ_TILES="$qt" FA_LK_TILES="$kt" FA_WD="$wd" \
    BENCH_ROWS="$Lq" BENCH_COLS="$D" BENCH_TILES="$qt" \
    timeout -k 30 "${FLASH_TIMEOUT:-600}" python "$PROFILE_OPS" 2>&1 | tee "$irf" \
    | _emit "flash_attn H=$H Lq=$Lq Lk=$Lk D=$D htiles=$ht qtiles=$qt ktiles=$kt wd=$wdtag"
  echo "TIMING_RUN flash H=$H Lq=$Lq h=$ht q=$qt k=$kt wd=$wdtag $((SECONDS - _t0))s" | tee -a "$LOG"
}

# ---- preflight ----
if [[ -z "${SKIP_PREFLIGHT:-}" ]]; then
  PF=$(SENCORES=32 BENCH_OP=neg BENCH_ROWS=64 BENCH_COLS=64 BENCH_REPS=1 BENCH_WARMUP=1 \
    timeout -k 10 150 python "$PROFILE_OPS" 2>&1)
  printf '%s\n' "$PF" | grep -q '^SUMMARY .*kernel_us=' \
    && echo "-- preflight OK" | tee -a "$LOG" \
    || { echo "## PREFLIGHT FAILED -- device busy; recover then SKIP_PREFLIGHT=1"; printf '%s\n' "$PF" | tail -6 | sed 's/^/PREFLIGHT /'; } | tee -a "$LOG"
  printf '%s\n' "$PF" | grep -q '^SUMMARY .*kernel_us=' || exit 1
fi

# ============ VALID work_div sweep (product<=32, small splits) ============
# Baseline: NO work_div (tiles only) -- this always compiles, anchors the tile study.
echo "## FLASH: tiles-only baselines (no work_div) + valid small-split work_div" | tee -a "$LOG"
for shape in "32 2048 2048 128" "32 4096 4096 128" "32 1024 1024 128"; do
  set -- $shape; H=$1 Lq=$2 Lk=$3 D=$4
  # tiles-only: vary H/Lq tiling with an EMPTY work_div (safe)
  for ht in 1 2 4 8; do runfa "$H" "$Lq" "$Lk" "$D" "$ht" 4 1 "Lq:1"; done
  for qt in 1 2 4 8; do runfa "$H" "$Lq" "$Lk" "$D" 8 "$qt" 1 "Lq:1"; done
  # valid work_div: product<=32, splits<=4 (divide the per-tile dims). ht=4 -> per-tile H=8.
  for wd in "H:2" "H:4" "Lq:2" "Lq:4" "H:2,Lq:2" "H:2,Lq:4" "H:4,Lq:2" "H:4,Lq:4" \
            "H:2,Lq:2,Lk:2" "Lq:4,Lk:2" "H:2,Lk:2"; do
    runfa "$H" "$Lq" "$Lk" "$D" 4 4 1 "$wd"
  done
done
echo "==== flash re-sweep DONE in ${SECONDS}s -- $LOG ====" | tee -a "$LOG"
