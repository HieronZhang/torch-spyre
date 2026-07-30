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
# GRAND SWEEP -- re-baseline + new capability after the layout211 merge.
#
# WHY A GRAND SWEEP AND NOT AN INCREMENT. The 2447 records in
# notes/sweep_records.json were measured at model_sha c87ef4a. HEAD is 159
# commits ahead, and they touch every subsystem the cost model measures:
#   cat1 broadcast 15 file-touches | cat2 transport 25 | cat3 matmul 20
#   cat4 bmm       26              | cat5 reduction 16 | cat6 coarse   14
#   cat7 flash     54
# Three of those changes invalidate standing assumptions outright:
#
#   (a) FRONTEND LOOP UNROLLING WAS REMOVED (#3235, #3114): codegen/unroll.py
#       deleted, UNROLL_LOOPS gone (0 refs left), ~2066 lines removed. The
#       cat-6 blocker was a claim about the OLD representation -- that
#       matmul_macs is TOTAL for matmul_row_tiling but PER-TILE elsewhere, and
#       that loop_factor is pinned at 1 while loop_trip>1. That claim must be
#       RE-TESTED before any coarse work is built on it. Section S1.
#   (b) THE FUSION ARGUMENT LIMIT WAS REMOVED (#3069), and the coarse-tile
#       stick-dim ambiguity was fixed (#3143). Those were the two blockers that
#       made flash_attn unrunnable. Section S3 finds out what runs now.
#   (c) The matmul preferred-layout PR (#3364) is merged. Section S2.
#
# WHAT SURVIVED THE MERGE UNCHANGED: the LX budget (dxp_lx_frac_avail=0.2 ->
# 1638 KB/core), so LX-threshold reasoning still holds.
#
# ON LAYOUTS -- WE CHASE THE GOOD ONES. Measured earlier: [1,0,2] on the two
# bmm INPUTS is ~3.3x faster at byte-identical traffic, whereas the PR's
# preferred OUTPUT layout measured 39% SLOWER on B=4 1024x2048x1024. So S2
# spends its budget on input-side fast layouts (forced manually via
# WD_LAYOUT_A/B, which works today and is independent of the PR) and treats the
# output-only mode as a control, not a target.
#
# SECTIONS -- ordered so an early cut-off still lands the most valuable data.
#   S0 RECHECK   is anything stale at all?                  5 runs   ~5 min
#   S1 SEMANTICS is the cat-6 blocker still real?           9 runs  ~19 min  (+4 IR dumps)
#   S2 LAYOUT    good layouts: manual + PR flag            60 runs  ~46 min
#   S3 FLASH     what runs now that fusion is unlimited     8 runs  ~40 min  (worst case; may fail fast)
#   S4 CORE      re-baseline cats 1-4                      69 runs  ~40 min
#   S5 COARSE    re-baseline cats 5-6                      63 runs  ~41 min
#                                                    TOTAL 214 runs ~3.2 h
# Counts are exact (from DRY=1). Times are from MEASURED per-run wall-clock:
# mmwd 37 s, bmm_layout 50 s, bmm_wd 24 s, matmul_row_tiling 106 s,
# mm_nested 114 s, matmul_k_tiling 127 s, bmm_k_tiling 108 s,
# pointwise/reduction 14-18 s, softmax_row_tiling 14 s.
#
#   bash docs/source/user_guide/examples/run_grand_sweep.sh
#   SECTIONS="S0 S1 S2" bash .../run_grand_sweep.sh      # a subset
#   MAX_SECONDS=7200 bash .../run_grand_sweep.sh         # 2 h budget cap
#   DRY=1 bash .../run_grand_sweep.sh                    # print the plan
# ============================================================================

set -u
trap 'echo "## INTERRUPTED"; exit 130' INT TERM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs haoyang_logs/ir

export BENCH_REPS="${BENCH_REPS:-7}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
SECTIONS="${SECTIONS:-S0 S1 S2 S3 S4 S5}"
MAX_SECONDS="${MAX_SECONDS:-0}"
DRY="${DRY:-0}"
if [[ "$DRY" == "1" ]]; then
  LOG="haoyang_logs/dryrun_grand_$(date +%Y%m%d_%H%M%S).log"
else
  LOG="haoyang_logs/grand_$(date +%Y%m%d_%H%M%S).log"
fi
_START=$SECONDS; NRUN=0
echo "==== GRAND SWEEP $(date)  reps=$BENCH_REPS  sections: $SECTIONS ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD)" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
budget_ok() { [[ "$MAX_SECONDS" == 0 ]] && return 0; (( SECONDS-_START < MAX_SECONDS )) && return 0
  echo "## BUDGET ${MAX_SECONDS}s reached -- stopping cleanly." | tee -a "$LOG"; return 1; }
sect()  { echo "" | tee -a "$LOG"; echo "## $1 -- $2 ($(date +%H:%M:%S))" | tee -a "$LOG"; SECT_T0=$SECONDS; }
esect() { echo "## $1 DONE in $((SECONDS-SECT_T0))s" | tee -a "$LOG"; }

_emit() {
  local all out; all=$(cat)
  out=$(printf '%s\n' "$all" | grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY|^TIMING ')
  if printf '%s\n' "$out" | grep -q '^SUMMARY'; then printf '%s\n' "$out" | tee -a "$LOG"
  else { echo "SUMMARY $1 FAILED"; printf '%s\n' "$all" | grep -vE '^\s*$' | tail -5 | sed 's/^/FAILDIAG /'; } | tee -a "$LOG"; fi
}
run() {
  local label="$1"; shift
  budget_ok || return 1
  NRUN=$((NRUN+1))
  if [[ "$DRY" == "1" ]]; then echo "PLAN[$NRUN] $label :: $*" | tee -a "$LOG"; return 0; fi
  local t0=$SECONDS
  echo "-- $label" | tee -a "$LOG"
  env "$@" timeout -k 30 "${RUN_TIMEOUT:-500}" python "$PROFILE_OPS" 2>&1 | _emit "$label"
  echo "TIMING_RUN ${label%% *} $label $((SECONDS-t0))s" | tee -a "$LOG"
}

if [[ "$DRY" != "1" ]]; then
  PF=$(SENCORES=32 BENCH_OP=neg BENCH_ROWS=64 BENCH_COLS=64 BENCH_REPS=1 BENCH_WARMUP=1 \
    timeout -k 10 180 python "$PROFILE_OPS" 2>&1)
  printf '%s\n' "$PF" | grep -q '^SUMMARY .*kernel_us=' \
    || { echo "## PREFLIGHT FAILED -- device busy/wedged; ABORTING." | tee -a "$LOG"
         printf '%s\n' "$PF" | tail -6 | sed 's/^/PREFLIGHT /' | tee -a "$LOG"; exit 1; }
  echo "-- preflight OK" | tee -a "$LOG"
fi
export SKIP_PREFLIGHT=1

# ============================== S0: RECHECK =================================
# Five configs already in the DB, one per category, all reps=7 cv<0.4. If these
# come back unchanged the old records are still comparable and S4/S5 can be
# trimmed; if they moved, the re-baseline is mandatory. Run FIRST and read it
# before committing hours to the rest.
if has S0; then
  sect S0 "recheck: are the 2447 recorded measurements still comparable?"
  run "recheck mmwd 2048x2048x4096 [was 767.4us]" \
    BENCH_OP=mmwd BENCH_ROWS=2048 BENCH_COLS=2048 BENCH_N=4096 WD_M=4 WD_N=8 WD_K=1 SENCORES=32
  run "recheck bmm_wd c8 1024x2048x1024 [was 6348.9us]" \
    BENCH_OP=bmm_wd BENCH_B=4 BENCH_ROWS=1024 BENCH_COLS=2048 BENCH_N=1024 WD_B=1 WD_M=4 WD_N=8 WD_K=1 SENCORES=8
  run "recheck matmul_row_tiling 8192x2048x2048 t8 [was 1353.0us]" \
    BENCH_OP=matmul_row_tiling BENCH_ROWS=8192 BENCH_COLS=2048 BENCH_N=2048 BENCH_TILES=8 SENCORES=32
  run "recheck softmax_row_tiling 2048x512 t8 c1 [was 589.6us]" \
    BENCH_OP=softmax_row_tiling BENCH_ROWS=2048 BENCH_COLS=512 BENCH_TILES=8 SENCORES=1
  run "recheck transpose_outer 16384x2048 M=8 [was 13636.1us]" \
    BENCH_OP=transpose_outer BENCH_ROWS=16384 BENCH_COLS=2048 TO_MID=8 SENCORES=32
  esect S0
fi

# ============================= S1: SEMANTICS ================================
# THE CAT-6 GATE. Loop unrolling was removed, so re-derive the feature semantics
# from scratch rather than assuming the old finding. For each coarse op at a
# FIXED shape across tile counts we need: matmul_macs vs B*M*K*N, loop_trip,
# loop_factor on every arg, and the counted traffic. The IR dumps make the
# ground truth checkable; the timed runs make the consequence measurable.
# Decision rule afterwards: if macs*loop_trip == TOTAL uniformly now, the
# blocker is GONE and cat 6 is unblocked; if it is still split, MACSIR tells us
# whether to fix the extractor.
if has S1; then
  sect S1 "cat 6 gate: re-derive coarse feature semantics after unroll removal"
  for op in matmul_row_tiling mm_nested_m_k matmul_k_tiling; do
    for t in 1 4 8; do
      run "sem $op 2048x2048x2048 t=$t" \
        BENCH_OP="$op" BENCH_ROWS=2048 BENCH_COLS=2048 BENCH_N=2048 BENCH_TILES="$t" SENCORES=32
    done
  done
  # IR ground truth at the two ops that disagreed before
  [[ "$DRY" == "1" ]] || for op in matmul_row_tiling mm_nested_m_k; do
    for t in 4 8; do
      irf="haoyang_logs/ir/grandsem_${op}_t${t}.txt"
      echo "-- IR $op t=$t -> $irf" | tee -a "$LOG"
      SENCORES=32 LX_PLANNING=0 SPYRE_DUMP_IR=1 BENCH_OP="$op" \
        BENCH_ROWS=2048 BENCH_COLS=2048 BENCH_N=2048 BENCH_TILES="$t" \
        BENCH_REPS=1 BENCH_WARMUP=1 timeout -k 30 500 python "$PROFILE_OPS" > "$irf" 2>&1
      echo "   lines=$(wc -l < "$irf")  loop_ir=$(grep -qc 'LoopLevel IR' "$irf" && echo yes || echo NO)" | tee -a "$LOG"
    done
  done
  esect S1
fi

# =============================== S2: LAYOUT =================================
# GOOD LAYOUTS ONLY. Two independent mechanisms:
#   (a) MANUAL, via WD_LAYOUT_A/B -> _to_dev -> t.to(DEVICE, device_layout=stl).
#       Works today, no PR needed, and it is how the 3.3x was originally
#       measured. This is the ceiling we care about.
#   (b) THE PR FLAG, SPYRE_MATMUL_PREFERRED_LAYOUT in {"",output,on}. Its
#       output-only mode measured 39% SLOWER, so it is carried as a CONTROL.
# The quads (dd/df/fd/ff) re-measure the additive per-operand model on current
# code; the flag runs test whether the compiler picks the good layout by itself.
if has S2; then
  sect S2 "cat 4 layouts: manual fast [1,0,2] + the preferred-layout flag"
  # (a) manual layout quads -- the additive model, re-baselined, incl. the two
  #     shapes that carry the residual (512x2048x512 fast corner, 1024^3 ff).
  for B in 2 4 8; do
    for sh in "1024 2048 1024" "2048 2048 1024" "512 2048 512" "1024 1024 1024"; do
      for lay in "0,1,2 0,1,2" "1,0,2 1,0,2" "0,1,2 1,0,2" "1,0,2 0,1,2"; do
        set -- $sh $lay
        run "lay B=$B $1x$2x$3 A=$4 B=$5" BENCH_OP=bmm_layout BENCH_B="$B" \
          BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" WD_B=1 WD_M=4 WD_N=8 WD_K=1 \
          WD_LAYOUT_A="$4" WD_LAYOUT_B="$5" SENCORES=32 || break 3
      done
    done
  done
  # (b) the flag, on shapes where we need to learn whether INPUTS flip. Compare
  #     against the manual ff row above at the same shape: if the flag reaches
  #     the manual fast time, the compiler is picking the good layout by itself.
  for mode in "" "output" "on"; do
    for sh in "1024 2048 1024" "2048 2048 1024" "512 2048 512"; do
      set -- $sh
      run "flag[${mode:-off}] bmm_wd B=4 $1x$2x$3" SPYRE_MATMUL_PREFERRED_LAYOUT="$mode" \
        BENCH_OP=bmm_wd BENCH_B=4 BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
        WD_B=1 WD_M=4 WD_N=8 WD_K=1 SENCORES=32 || break 2
    done
    # plain 2-D control: the flag is a NO-OP at rank 2, so this must not move
    run "flag[${mode:-off}] mmwd 2048x2048x2048 [CONTROL, must not move]" \
      SPYRE_MATMUL_PREFERRED_LAYOUT="$mode" \
      BENCH_OP=mmwd BENCH_ROWS=2048 BENCH_COLS=2048 BENCH_N=2048 WD_M=4 WD_N=8 WD_K=1 SENCORES=32
  done
  esect S2
fi

# ================================ S3: FLASH =================================
# CAT 7, PREVIOUSLY UNRUNNABLE. Both blockers I diagnosed are gone upstream:
# the SDSC bundle-tensor limit (#3069 removed the fusion argument limit) and the
# coarse-tile stick-dim ambiguity (#3143). Plus SDPA layout/size fixes (#3184,
# #3205), flash LX-pinning fixes (#3084) and exp-padding masking (#2365a59).
# Start SMALL and widen: the previous failures were timeouts, so a short
# RUN_TIMEOUT here is deliberate -- we want to learn what runs, not to hang.
if has S3; then
  sect S3 "cat 7 flash: what runs now that fusion is unlimited"
  # Tile COUNTS per dim (FA_*_TILES) drive the loop nest; FA_WD is the work_div hint.
  # Start at the small shape that previously failed, and widen only if it runs.
  for ht_qt in "4 2" "8 2" "8 4" "16 4"; do
    set -- $ht_qt
    run "flash H=32 Lq=1024 Lk=1024 htiles=$1 qtiles=$2" RUN_TIMEOUT=300 \
      BENCH_OP=flash_attn FA_H=32 FA_LQ=1024 FA_LK=1024 \
      FA_H_TILES="$1" FA_LQ_TILES="$2" FA_LK_TILES=1 SENCORES=32
  done
  for lq in 2048 4096; do
    run "flash H=32 Lq=$lq Lk=$lq htiles=8 qtiles=4" RUN_TIMEOUT=300 \
      BENCH_OP=flash_attn FA_H=32 FA_LQ="$lq" FA_LK="$lq" \
      FA_H_TILES=8 FA_LQ_TILES=4 FA_LK_TILES=1 SENCORES=32
  done
  # LK tiling is the axis that previously hit the bundle-tensor limit -- now that
  # the fusion argument limit is gone (#3069), test it explicitly.
  for kt in 2 4; do
    run "flash H=32 Lq=2048 Lk=2048 ktiles=$kt [was blocked by bundle limit]" RUN_TIMEOUT=300 \
      BENCH_OP=flash_attn FA_H=32 FA_LQ=2048 FA_LK=2048 \
      FA_H_TILES=8 FA_LQ_TILES=4 FA_LK_TILES="$kt" SENCORES=32
  done
  esect S3
fi

# ================================ S4: CORE ==================================
# Re-baseline cats 1-4 on current code. These are the categories the model
# currently predicts WELL (transport 6.0%, broadcast 9.3%, matmul_split 14.0%),
# so the job is to confirm they still hold -- a regression here would mean a
# shipped term has gone stale, which matters more than a new fit.
if has S4; then
  sect S4 "re-baseline cats 1-4 (broadcast / transport / matmul / bmm)"
  for rc in "2048 4096" "8192 2048" "2048 16384" "4096 8192"; do
    set -- $rc
    for op in copy bcast bcastcol mulbcast write; do
      run "$op R=$1 C=$2" BENCH_OP="$op" BENCH_ROWS="$1" BENCH_COLS="$2" SENCORES=32 || break 2
    done
  done
  for M in 2 4 8 32; do
    for rc in "2048 8192" "8192 2048" "2048 2048"; do
      set -- $rc
      run "transpose_outer R=$1 C=$2 M=$M" BENCH_OP=transpose_outer \
        BENCH_ROWS="$1" BENCH_COLS="$2" TO_MID="$M" SENCORES=32 || break 2
    done
  done
  for c in 1 2 4 8 16 32; do
    for sh in "2048 2048 2048" "4096 2048 2048"; do
      set -- $sh
      run "mmwd cores=$c $1x$2x$3" BENCH_OP=mmwd BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
        WD_M=4 WD_N=8 WD_K=1 SENCORES="$c" || break 2
    done
  done
  for sh in "2048 2048 2048" "4096 4096 2048" "2048 4096 4096"; do
    for mn in "4 8" "8 4" "2 16"; do
      set -- $sh $mn
      run "mmwd split $4x$5 $1x$2x$3" BENCH_OP=mmwd BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
        WD_M="$4" WD_N="$5" WD_K=1 SENCORES=32 || break 2
    done
  done
  for B in 2 4 8 16; do
    for sh in "1024 2048 1024" "2048 2048 1024"; do
      set -- $sh
      run "bmm_wd B=$B $1x$2x$3" BENCH_OP=bmm_wd BENCH_B="$B" BENCH_ROWS="$1" BENCH_COLS="$2" \
        BENCH_N="$3" WD_B=1 WD_M=4 WD_N=8 WD_K=1 SENCORES=32 || break 2
      run "bmm_wd_3d2d B=$B $1x$2x$3" BENCH_OP=bmm_wd_3d2d BENCH_B="$B" BENCH_ROWS="$1" \
        BENCH_COLS="$2" BENCH_N="$3" WD_B=1 WD_M=4 WD_N=8 WD_K=1 SENCORES=32 || break 2
    done
  done
  esect S4
fi

# =============================== S5: COARSE =================================
# Re-baseline cats 5-6. These are the WORST categories (matmul_nested 39.3%,
# matmul_row 23.5%, softmax 21.8%) and the ones most disturbed by the unroll
# removal, so this is where new modelling is most likely. Two one-variable
# ladders each, so cores / tiles / shape can be separated rather than fitted
# jointly on a sparse grid -- the mistake that stalled cat 5 the first time.
if has S5; then
  sect S5 "re-baseline cats 5-6 (reductions, softmax, coarse matmul)"
  for c in 1 2 4 8 16 32; do
    for op in read amax sumrow mean; do
      run "$op cores=$c 4096x2048" BENCH_OP="$op" BENCH_ROWS=4096 BENCH_COLS=2048 SENCORES="$c" || break 2
    done
  done
  # softmax COLS x CORES at fixed tiles, then TILES x CORES at fixed COLS:
  # the two ladders the cat-5 residual needed and never had.
  for c in 1 2 4 8 16 32; do
    for C in 512 2048; do
      run "softmax_row_tiling cores=$c 4096x$C t=8" BENCH_OP=softmax_row_tiling \
        BENCH_ROWS=4096 BENCH_COLS="$C" BENCH_TILES=8 SENCORES="$c" || break 2
    done
  done
  for t in 1 4 8 16; do
    for c in 1 8 32; do
      run "softmax_row_tiling cores=$c 4096x2048 t=$t" BENCH_OP=softmax_row_tiling \
        BENCH_ROWS=4096 BENCH_COLS=2048 BENCH_TILES="$t" SENCORES="$c" || break 2
    done
  done
  for op in matmul_row_tiling matmul_k_tiling mm_nested_m_k; do
    for t in 1 2 4 8 16; do
      run "$op 4096x2048x2048 t=$t" BENCH_OP="$op" BENCH_ROWS=4096 BENCH_COLS=2048 \
        BENCH_N=2048 BENCH_TILES="$t" SENCORES=32 || break 2
    done
  done
  esect S5
fi

{
  echo ""
  echo "==== GRAND SWEEP DONE in $((SECONDS-_START))s -- $NRUN runs ===="
  echo "Fold ONLY the new logs (re-parsing everything balloons the curated file):"
  echo "  python3 notes/parse_sweep_logs.py --out notes/sweep_records.json haoyang_logs/grand_*.log"
  echo "Then, IN THIS ORDER:"
  echo "  1. read S0 -- if configs MOVED, the old records are stale; say so in"
  echo "     cost_model_status.md and do not mix pre/post records in one fit."
  echo "  2. read S1 -- does macs*loop_trip == TOTAL uniformly now? If yes the"
  echo "     cat-6 blocker is GONE and 121 outlier points become actionable."
  echo "  3. read S2 -- does the flag reach the manual fast time? That decides"
  echo "     whether the compiler picks good layouts on its own."
  echo "  4. python3 notes/eval_model.py --all   (expect movement; re-fit per category)"
} | tee -a "$LOG"
