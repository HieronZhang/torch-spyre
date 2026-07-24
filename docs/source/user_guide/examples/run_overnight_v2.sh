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
# OVERNIGHT SWEEP v2 -- one launch, everything the cost-model rework needs.
# Runs, in order:
#   1. run_matmul_family_sweep.sh  -- clean noise-protocol data for cats 3/4/5/6
#      (forced-core matmul overlap, split-shape, bmm, coarse-tile fill/drain,
#      BW-vs-cores). This is the crux -- the coefficients the Part III rework needs.
#   2. run_flash_resweep.sh        -- cat 7 flash (fixed valid work_div) + IR.
#   3. IRCAP                       -- LOWER-LEVEL IR dumps (SPYRE_DUMP_IR=1 -> full
#      ATen FX + lowering + LoopLevel IR + op_it_space_splits, teed per op) for the
#      ops whose MECHANISM needs the IR to confirm, not just the timing:
#        * coarse matmul (cat 6): matmul_row_tiling / mm_nested / matmul_k_tiling
#          over tiles -> the CoarseTileInfo / loop_count / per-tile loop structure
#          (why mm_nested's loop_trip is stuck at 2; the per-tile fill/drain).
#        * softmax_unrolled (cat 5): the unrolled chain has NO CoarseTileInfo and
#          runs cores=1 -> confirm the extractor under-count from the IR directly.
#        * bmm_layout (cat 4): the 4 device-layout combos -> confirm no inserted
#          restickify/clone (the layout signal is not confounded by a copy).
#        * plain matmul (cat 3): a balanced + a lopsided split -> op_it_space_splits.
#      (The compute/HBM OVERLAP itself is NOT in any dumpable IR -- it is in the
#      DeepTools backend -- so IRCAP is for the tiling/fusion/split STRUCTURE, which
#      the front-end does control. See notes/compiler_pipeline_deep_dive.md.)
#
# Sub-sweeps each write their own haoyang_logs/*.log; IRCAP writes IR to
# haoyang_logs/ir/ircap_*.txt + a summary log. Forward ALL of haoyang_logs/
# (logs + ir/) + notes/sweep_records.* after it folds. Each stage runs
# independently; a failure in one does not abort the others.
#
#   bash docs/source/user_guide/examples/run_overnight_v2.sh          # all 3 stages
#   STAGES="MMFAMILY FLASH" bash .../run_overnight_v2.sh              # skip IRCAP
#   BENCH_REPS=7 bash .../run_overnight_v2.sh
# Realistic wall-clock: ~2-4 h (flash compiles dominate). Launch it and let it run.
# ============================================================================

set -u
trap 'echo "## INTERRUPTED (SIGINT)"; exit 130' INT
trap 'echo "## TERMINATED (SIGTERM)"; exit 143' TERM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs haoyang_logs/ir
LOG="haoyang_logs/overnight_v2_$(date +%Y%m%d_%H%M%S).log"
STAGES="${STAGES:-MMFAMILY FLASH IRCAP}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export BENCH_REPS="${BENCH_REPS:-7}"
echo "==== OVERNIGHT v2 $(date) -- stages: $STAGES ====" | tee "$LOG"
has() { [[ " $STAGES " == *" $1 "* ]]; }

# ---- one preflight for the whole run; sub-sweeps then SKIP_PREFLIGHT ----
echo "-- preflight: trivial neg to check the Spyre device is up ..." | tee -a "$LOG"
PF=$(SENCORES=32 BENCH_OP=neg BENCH_ROWS=64 BENCH_COLS=64 BENCH_REPS=1 BENCH_WARMUP=1 \
  timeout -k 10 150 python "$PROFILE_OPS" 2>&1)
if printf '%s\n' "$PF" | grep -q '^SUMMARY .*kernel_us='; then
  echo "-- preflight OK (device reachable)" | tee -a "$LOG"
else
  { echo "## PREFLIGHT FAILED -- device busy/wedged; ABORTING. Recover (no root needed):"
    printf '%s\n' "$PF" | grep -vE '^\s*$' | tail -8 | sed 's/^/PREFLIGHT /'
    echo "##   ps -u \$(whoami) -o pid,stat,cmd | grep -Ei 'python|profile_ops'  ; kill -9 <pid>"; } | tee -a "$LOG"
  exit 1
fi
export SKIP_PREFLIGHT=1

# ---- IRCAP helper: run one op with FULL IR dump, tee to a per-op file ----
ircap() {  # ircap <tag> <ENV=VAL ...> -- last arg style: pass BENCH_* via env before the call
  local tag="$1"; shift
  local irf="haoyang_logs/ir/ircap_${tag}.txt" _t0=$SECONDS
  echo "-- IRCAP $tag (IR -> $irf)" | tee -a "$LOG"
  SENCORES="${SENCORES:-32}" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 "$@" \
    timeout -k 30 "${IRCAP_TIMEOUT:-400}" python "$PROFILE_OPS" > "$irf" 2>&1
  # surface the SUMMARY + whether a restickify/clone got inserted (bmm_layout check)
  grep -E '^SUMMARY|op_it_space_splits' "$irf" | head -3 | sed 's/^/   /' | tee -a "$LOG"
  grep -qiE 'restickify|insert_copy|\bclone\b' "$irf" && echo "   (note: copy/restickify present in IR)" | tee -a "$LOG"
  echo "TIMING_RUN ircap $tag $((SECONDS-_t0))s" | tee -a "$LOG"
}

# ============ STAGE 1: matmul-family (cats 3/4/5/6) ============
if has MMFAMILY; then
  echo "==== STAGE 1/3: matmul-family calibration ($(date)) ====" | tee -a "$LOG"
  bash "$SCRIPT_DIR/run_matmul_family_sweep.sh" || echo "## STAGE MMFAMILY had a nonzero exit (continuing)" | tee -a "$LOG"
fi

# ============ STAGE 2: flash re-sweep (cat 7) ============
if has FLASH; then
  echo "==== STAGE 2/3: flash re-sweep ($(date)) ====" | tee -a "$LOG"
  bash "$SCRIPT_DIR/run_flash_resweep.sh" || echo "## STAGE FLASH had a nonzero exit (continuing)" | tee -a "$LOG"
fi

# ============ STAGE 3: IRCAP -- lower-level IR for cats 4/5/6/3 structure ====
if has IRCAP; then
  echo "==== STAGE 3/3: IR captures ($(date)) ====" | tee -a "$LOG"
  # cat 6: coarse matmul over tiles -> CoarseTileInfo / loop_count / per-tile loop
  for t in 1 2 4 8 16; do
    ircap "matmul_row_t${t}" env BENCH_OP=matmul_row_tiling BENCH_ROWS=2048 BENCH_COLS=2048 BENCH_N=2048 BENCH_TILES="$t" BENCH_REPS=1
  done
  for t in 1 2 4 8; do
    ircap "mm_nested_t${t}"  env BENCH_OP=mm_nested_m_k BENCH_ROWS=2048 BENCH_COLS=2048 BENCH_N=2048 BENCH_TILES="$t" BENCH_REPS=1
  done
  for t in 2 8; do
    ircap "matmul_k_t${t}"   env BENCH_OP=matmul_k_tiling BENCH_ROWS=2048 BENCH_COLS=2048 BENCH_N=2048 BENCH_TILES="$t" BENCH_REPS=1
  done
  # cat 5: softmax_unrolled (no CoarseTileInfo, cores=1) + softmax_row_tiling control
  for t in 1 4 16; do
    ircap "smx_unrolled_${t}x_1024x512" env BENCH_OP=softmax_unrolled BENCH_ROWS=1024 BENCH_COLS=512 BENCH_TILES="$t" BENCH_REPS=1
    ircap "smx_unrolled_${t}x_2048x512" env BENCH_OP=softmax_unrolled BENCH_ROWS=2048 BENCH_COLS=512 BENCH_TILES="$t" BENCH_REPS=1
  done
  for t in 1 8; do
    ircap "smx_rowtile_${t}x" env BENCH_OP=softmax_row_tiling BENCH_ROWS=2048 BENCH_COLS=512 BENCH_TILES="$t" BENCH_REPS=1
  done
  # cat 4: bmm_layout 4 device-layout combos (confirm no inserted copy)
  for la in "0,1,2" "1,0,2"; do for lb in "0,1,2" "1,0,2"; do
    ircap "bmm_layout_A${la//,/}_B${lb//,/}" \
      env BENCH_OP=bmm_layout BENCH_B=4 BENCH_ROWS=1024 BENCH_COLS=2048 BENCH_N=1024 \
          WD_B=1 WD_M=4 WD_N=8 WD_K=1 WD_LAYOUT_A="$la" WD_LAYOUT_B="$lb" BENCH_REPS=1
  done; done
  # cat 3: plain matmul balanced vs lopsided split -> op_it_space_splits
  ircap "mmwd_2048_4x8" env BENCH_OP=mmwd BENCH_ROWS=2048 BENCH_COLS=2048 BENCH_N=2048 WD_M=4 WD_N=8 WD_K=1 BENCH_REPS=1
  ircap "mmwd_2048_1x1" env BENCH_OP=mmwd BENCH_ROWS=2048 BENCH_COLS=2048 BENCH_N=2048 WD_M=1 WD_N=1 WD_K=1 BENCH_REPS=1
fi

echo "==== OVERNIGHT v2 DONE in ${SECONDS}s ($(date)) -- forward haoyang_logs/ (logs + ir/) + notes/sweep_records.* ====" | tee -a "$LOG"
