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
# FLASH-ATTENTION RE-SWEEP (cost-model category 7) -- data + IR capture.
#
# WHY THE OVERNIGHT FLASH RUNS FAILED -- CORRECTED DIAGNOSIS (2026-07-24 review):
# The earlier "product 256 > 32 cores caused the rejections" story was WRONG.
#   * CORE OVER-SUBSCRIPTION IS NOT AN ERROR. The work-division pass SILENTLY SKIPS a
#     split it cannot place (source: `continue`, not `raise`); a work_div product > 32
#     is absorbed, not rejected -- the same product-256 config both timed out AND
#     succeeded elsewhere, so it is neither the failure cause nor an error condition.
#   * THE DOMINANT FAILURE WAS A COMPILE TIMEOUT (~600 s), ~15/18 of the failed configs
#     -- the flash compile is simply heavy; the timeout was mis-read as a "rejection."
#   * DIVISIBILITY is a REAL but MINORITY compile-error mode (~2/18): a split must divide
#     the op's stick-adjusted iteration-space extent of that named dim, checked per dim
#     AFTER the over-subscription skip -- and some softmax INTERMEDIATE (e.g. buf5, as
#     small as 2) is smaller than the per-tile dim, so a split can divide the per-tile
#     dim yet still fail on the intermediate. The front end cannot see buf5 up front, so
#     a divisibility guard here is a NECESSARY (not sufficient) filter; the true
#     intermediate constraint surfaces in the captured IR.
#
# FIX HERE: (a) guard work_div product <= 32 (harmless -- avoids a silently-skipped,
# not-what-you-asked split); (b) guard per-tile divisibility (the necessary condition);
# (c) RAISE the timeout (FLASH_TIMEOUT, default 900 s) and log TIMEOUT distinctly from
# FAILED so the timeout-dominance is measurable; (d) capture the loop IR / the actual
# compile error for both survivors and failures. NOT modeling flash attn yet -- this is
# data + IR capture (see notes/flash_attn_hints.md).
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
    echo "SKIP flash wd=$wd product=$prod > 32 cores (would be silently skipped, not run: H=$H ht=$ht Lq=$Lq qt=$qt)" | tee -a "$LOG"; return 0
  fi
  # per-tile divisibility guard (NECESSARY, not sufficient -- intermediate buffers may be
  # smaller than the per-tile dim and still fail; that surfaces in the captured IR).
  local part dim s pt bad=""
  for part in ${wd//,/ }; do
    dim=${part%:*}; s=${part#*:}
    case "$dim" in
      H)  pt=$(( H / ht )) ;;
      Lq) pt=$(( Lq / qt )) ;;
      Lk) pt=$(( Lk / kt )) ;;
      *)  pt=0 ;;
    esac
    if (( pt == 0 || s > pt || pt % s != 0 )); then bad="$dim:$s(per-tile=$pt)"; break; fi
  done
  if [[ -n "$bad" ]]; then
    echo "SKIP flash wd=$wd non-divisible per-tile split $bad (H=$H ht=$ht Lq=$Lq qt=$qt Lk=$Lk kt=$kt)" | tee -a "$LOG"; return 0
  fi
  local wdtag="${wd//:/}"; wdtag="${wdtag//,/-}"
  local irf="haoyang_logs/ir/flashRS_H${H}_Lq${Lq}_Lk${Lk}_h${ht}q${qt}k${kt}_${wdtag}.txt"
  echo "-- flash_attn H=$H Lq=$Lq Lk=$Lk D=$D htiles=$ht qtiles=$qt ktiles=$kt wd=$wdtag prod=$prod (IR -> $irf)" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING=1 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=flash_attn FA_H="$H" FA_LQ="$Lq" FA_LK="$Lk" FA_D="$D" \
    FA_H_TILES="$ht" FA_LQ_TILES="$qt" FA_LK_TILES="$kt" FA_WD="$wd" \
    BENCH_ROWS="$Lq" BENCH_COLS="$D" BENCH_TILES="$qt" \
    timeout -k 30 "${FLASH_TIMEOUT:-900}" python "$PROFILE_OPS" 2>&1 | tee "$irf" \
    | _emit "flash_attn H=$H Lq=$Lq Lk=$Lk D=$D htiles=$ht qtiles=$qt ktiles=$kt wd=$wdtag"
  local rc=${PIPESTATUS[0]}  # timeout(1) returns 124 (or 128+9=137 on -k kill)
  if (( rc == 124 || rc == 137 )); then
    echo "   TIMEOUT (rc=$rc) after ${FLASH_TIMEOUT:-900}s -- the DOMINANT overnight failure mode (heavy compile, not a rejected config)" | tee -a "$LOG"
  fi
  echo "TIMING_RUN flash H=$H Lq=$Lq h=$ht q=$qt k=$kt wd=$wdtag rc=$rc $((SECONDS - _t0))s" | tee -a "$LOG"
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
