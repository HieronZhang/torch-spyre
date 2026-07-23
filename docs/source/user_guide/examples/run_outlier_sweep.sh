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
# OUTLIER-ATTACK SWEEP -- the overnight superset that gathers the additional data
# needed to rework the six cost-model outlier categories PLUS flash-attention (a
# real multi-op coarse-tiled program). See the plan
# ~/.claude/plans/shimmering-percolating-rivest.md, notes/cost_model_report.md, and
# notes/flash_attn_hints.md. Every run takes BENCH_REPS back-to-back PROFILED
# measurements (default 7) and reports kernel_us_min/median/std/cv so a jittery
# point is visible (cv), not read as a model error. Priority-ordered so an early
# cut-off still lands the crux data. Sized for a ~9 h overnight window.
#
#   NOISE0   quantify true run-to-run cv on the control + the flagged noisy points.
#   MMCORE   (cat 3) plain matmul FORCED to cores {1,2,4,8,16,32} across K-ratios and
#            shapes -> the full γ(cores) curve (single core = work-division-free anchor).
#   BMMLAY   (cat 4) full bmm with a chosen device dim_order ([1,0,2] vs [0,1,2]) at
#            matched shape/MACs/split -> the pure layout effect. SMOKE-checks the IR for
#            an inserted restickify/clone (a copy would confound -> WARN).
#   XPORT    (cat 2) dense independent R×C grid for transpose/transpose_outer/cat0/cat1
#            -> map the BW_eff(R,C) access-pattern surface (models transpose_outer).
#   BCAST    (cat 1) dense R×C for write/bcast/bcastcol/mulbcast/copy, large-C focus.
#   SMXUNROLL(cat 5) softmax_unrolled + small-COLS softmax_row_tiling with IR captured
#            per run -> find the extractor under-count (no CoarseTileInfo, cores=1).
#   CTMM     (cat 6) coarse matmul/bmm tile ladders incl. the LX-spill regime.
#   BMM2R    (cat 4 baseline) full-vs-3d2d two-rate validation under the noise protocol.
#   FLASH    (NEW, runs LAST) flash-attention with env-configurable coarse-tiling +
#            work_div hints -> measured time + IR for a multi-op coarse program (beyond
#            softmax). Data only -- we do NOT model flash attn yet. Last because its
#            compile is heaviest / it is the most exploratory; every crux run lands first.
#
# Constraints: cores=m*n*k<=32, every split divides its dim, every MNK<=3.4e10 (hang
# bound). No `set -e`; each run has a RUN_TIMEOUT guard (FLASH_TIMEOUT for flash, whose
# compile is heavier) and self-reports FAILED.
#
#   MAX_SECONDS=28800 bash docs/source/user_guide/examples/run_outlier_sweep.sh  # 8 h cap
#   bash docs/source/user_guide/examples/run_outlier_sweep.sh                     # all, no cap
#   SECTIONS="NOISE0 MMCORE FLASH" bash .../run_outlier_sweep.sh                  # a subset
#   BENCH_REPS=5 bash .../run_outlier_sweep.sh                                    # fewer reps
# 515 runs total (verified by a stubbed dry-run: NOISE0 12, MMCORE 88, BMMLAY 44, XPORT 88,
# BCAST 84, SMXUNROLL 46, CTMM 81, BMM2R 28, FLASH 44; 0 guard SKIPs). Realistic compile
# times fill ~8 h; MAX_SECONDS caps it and priority order trims only the flash tail.
# Output: <repo-root>/haoyang_logs/outlier_<timestamp>.log (+ per-run IR under
# haoyang_logs/ir/). Forward the log + notes/sweep_records.* after it folds in.
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs haoyang_logs/ir
LOG="haoyang_logs/outlier_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-NOISE0 MMCORE BMMLAY XPORT BCAST SMXUNROLL CTMM BMM2R FLASH}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export BENCH_REPS="${BENCH_REPS:-7}"      # profiled reps per run (noise protocol)
# Wall-clock budget: stop cleanly once SECONDS >= MAX_SECONDS (0 = unlimited). The sweep
# is priority-ordered (crux/outlier sections first, flash last), so hitting the budget only
# trims the least-critical tail. Sized so realistic compile times fill it; if compiles are
# fast it just finishes early (still a full dataset). Run an 8 h night with MAX_SECONDS=28800.
MAX_SECONDS="${MAX_SECONDS:-0}"

echo "==== outlier sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS  reps: $BENCH_REPS" \
  | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_over() { [[ "$MAX_SECONDS" -gt 0 && "$SECONDS" -ge "$MAX_SECONDS" ]]; }  # wall-clock budget hit?
# `run` gate: a section is entered only if enabled AND under budget; a helper early-returns
# (silently, so the tail drains fast) once the budget is hit mid-section.
do_sect() { has "$1" && ! _over; }
_emit() {  # keep the parseable lines for parse_sweep_logs.py; on a crash, surface WHY
  local all out; all=$(cat)   # the run's full stdout+stderr
  out=$(printf '%s\n' "$all" | grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY|^TIMING ')
  if printf '%s\n' "$out" | grep -q '^SUMMARY'; then
    printf '%s\n' "$out" | tee -a "$LOG"      # a SUMMARY (ok OR profile_ops' own FAILED reason)
  else
    # no SUMMARY at all -> crashed before main() (e.g. import error). Log the tail so the
    # error is diagnosable instead of a bare FAILED.
    { echo "SUMMARY $1 FAILED"
      printf '%s\n' "$all" | grep -vE '^\s*$' | tail -6 | sed 's/^/FAILDIAG /'; } | tee -a "$LOG"
  fi
}
sect() { SECT="$1"; SECT_T0=$SECONDS; echo "## $1 -- ${2:-}" | tee -a "$LOG"; }
esect() { echo "## $SECT DONE in $((SECONDS - SECT_T0))s" | tee -a "$LOG"; }
_ok_mm() {  # _ok_mm M K N m n k : cores<=32, splits divide dims, MNK<=3.4e10
  local M=$1 K=$2 N=$3 m=$4 n=$5 k=$6
  (( m * n * k <= 32 )) || { echo "SKIP cores>32 $*" | tee -a "$LOG"; return 1; }
  (( M % m == 0 && N % n == 0 && K % k == 0 )) \
    || { echo "SKIP indivisible $*" | tee -a "$LOG"; return 1; }
  (( M * K * N <= 34000000000 )) || { echo "SKIP MNK>3.4e10 $*" | tee -a "$LOG"; return 1; }
}

# ---- run helpers (each prints TIMING_RUN with its wall-clock) ---------------
runpw() {  # runpw <op> <rows> <cols> <lx>   pointwise/broadcast/transport, cores=32
  _over && return 0
  local _t0=$SECONDS
  echo "-- $1 [$2,$3] lx=$4" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING="$4" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "$1 [$2,$3] lx=$4"
  echo "TIMING_RUN $1 [$2,$3] $((SECONDS - _t0))s" | tee -a "$LOG"
}
runmmwd() {  # runmmwd <M> <K> <N> <m> <n> <k>   FORCED plain-matmul split, cores=m*n*k
  _over && return 0
  local M=$1 K=$2 N=$3 m=$4 n=$5 k=$6 cores=$(( $4 * $5 * $6 )) _t0=$SECONDS
  _ok_mm "$M" "$K" "$N" "$m" "$n" "$k" || return 0
  echo "-- mmwd M=$M K=$K N=$N split m=$m n=$n k=$k (cores=$cores M/m=$((M/m)) N/n=$((N/n)))" \
    | tee -a "$LOG"
  SENCORES=32 WD_M="$m" WD_N="$n" WD_K="$k" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=mmwd BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "mmwd M=$M K=$K N=$N split m=$m n=$n k=$k"
  echo "TIMING_RUN mmwd M=$M K=$K N=$N m=$m n=$n k=$k $((SECONDS - _t0))s" | tee -a "$LOG"
}
runbmmwd() {  # runbmmwd <op> <B> <M> <K> <N> <b> <m> <n> <k>   FORCED bmm split
  _over && return 0
  local op=$1 B=$2 M=$3 K=$4 N=$5 b=$6 m=$7 n=$8 k=$9 cores=$(( $6 * $7 * $8 * $9 )) _t0=$SECONDS
  echo "-- $op B=$B M=$M K=$K N=$N b=$b m=$m n=$n k=$k (cores=$cores)" | tee -a "$LOG"
  SENCORES=32 WD_B="$b" WD_M="$m" WD_N="$n" WD_K="$k" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$op" BENCH_B="$B" BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "$op B=$B M=$M K=$K N=$N b=$b m=$m n=$n k=$k"
  echo "TIMING_RUN $op B=$B M=$M K=$K N=$N $((SECONDS - _t0))s" | tee -a "$LOG"
}
runbmml() {  # runbmml <B> <M> <K> <N> <b> <m> <n> <k> <layoutA> <layoutB>   layout bmm
  _over && return 0
  local B=$1 M=$2 K=$3 N=$4 b=$5 m=$6 n=$7 k=$8 la=$9 lb=${10} _t0=$SECONDS
  local irf="haoyang_logs/ir/bmm_layout_B${B}_${M}x${K}x${N}_A${la//,/}_B${lb//,/}.txt"
  echo "-- bmm_layout B=$B M=$M K=$K N=$N b=$b m=$m n=$n k=$k layoutA=$la layoutB=$lb" \
    | tee -a "$LOG"
  SENCORES=32 WD_B="$b" WD_M="$m" WD_N="$n" WD_K="$k" \
    WD_LAYOUT_A="$la" WD_LAYOUT_B="$lb" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=bmm_layout BENCH_B="$B" BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | tee "$irf" \
    | _emit "bmm_layout B=$B M=$M K=$K N=$N b=$b m=$m n=$n k=$k layoutA=$la layoutB=$lb"
  if grep -qiE 'restickify|\bclone\b|insert_copy' "$irf"; then
    echo "WARN bmm_layout $la/$lb INSERTED A COPY (see $irf) -- layout row is confounded" \
      | tee -a "$LOG"
  fi
  echo "TIMING_RUN bmm_layout B=$B ${M}x${K}x${N} $la/$lb $((SECONDS - _t0))s" | tee -a "$LOG"
}
runmk() {  # runmk <op> <M> <K> <N> <tiles>   coarse matmul (row / k / nested), cores=32
  _over && return 0
  local _t0=$SECONDS
  echo "-- $1 M=$2 K=$3 N=$4 tiles=$5" | tee -a "$LOG"
  SENCORES=32 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" BENCH_N="$4" BENCH_TILES="$5" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "$1 M=$2 K=$3 N=$4 tiles=$5"
  echo "TIMING_RUN $1 M=$2 K=$3 N=$4 tiles=$5 $((SECONDS - _t0))s" | tee -a "$LOG"
}
runmkb() {  # runmkb <op> <B> <M> <K> <N> <tiles>   coarse bmm (uses BENCH_B), cores=32
  _over && return 0
  local _t0=$SECONDS
  echo "-- $1 B=$2 M=$3 K=$4 N=$5 tiles=$6" | tee -a "$LOG"
  SENCORES=32 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_B="$2" BENCH_ROWS="$3" BENCH_COLS="$4" BENCH_N="$5" BENCH_TILES="$6" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "$1 B=$2 M=$3 K=$4 N=$5 tiles=$6"
  echo "TIMING_RUN $1 B=$2 ${3}x${4}x${5} tiles=$6 $((SECONDS - _t0))s" | tee -a "$LOG"
}
runsmx() {  # runsmx <op> <rows> <cols> <tiles>   softmax, IR saved per run (cat 5)
  _over && return 0
  local _t0=$SECONDS
  local irf="haoyang_logs/ir/${1}_${2}x${3}_t${4}.txt"
  echo "-- $1 [$2,$3] tiles=$4 (IR -> $irf)" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING=1 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" BENCH_TILES="$4" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | tee "$irf" \
    | _emit "$1 [$2,$3] tiles=$4"
  echo "TIMING_RUN $1 [$2,$3] tiles=$4 $((SECONDS - _t0))s" | tee -a "$LOG"
}
runfa() {  # runfa <H> <Lq> <Lk> <D> <htiles> <qtiles> <ktiles> <wd_env>  flash attn
  _over && return 0
  local H=$1 Lq=$2 Lk=$3 D=$4 ht=$5 qt=$6 kt=$7 wd=$8 _t0=$SECONDS
  local wdtag="${wd//:/}"; wdtag="${wdtag//,/-}"   # "H:4,Lq:8,Lk:8" -> "H4-Lq8-Lk8"
  local irf="haoyang_logs/ir/flash_H${H}_Lq${Lq}_Lk${Lk}_h${ht}q${qt}k${kt}_${wdtag}.txt"
  echo "-- flash_attn H=$H Lq=$Lq Lk=$Lk D=$D htiles=$ht qtiles=$qt ktiles=$kt wd=$wdtag (IR -> $irf)" \
    | tee -a "$LOG"
  SENCORES=32 LX_PLANNING=1 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=flash_attn BENCH_ROWS="$Lq" BENCH_COLS="$D" BENCH_TILES="$qt" \
    FA_H="$H" FA_LQ="$Lq" FA_LK="$Lk" FA_D="$D" \
    FA_H_TILES="$ht" FA_LQ_TILES="$qt" FA_LK_TILES="$kt" FA_WD="$wd" \
    timeout -k 30 "${FLASH_TIMEOUT:-600}" python "$PROFILE_OPS" 2>&1 | tee "$irf" \
    | _emit "flash_attn H=$H Lq=$Lq Lk=$Lk D=$D htiles=$ht qtiles=$qt ktiles=$kt wd=$wdtag"
  echo "TIMING_RUN flash_attn H=$H Lq=$Lq h=$ht q=$qt k=$kt $((SECONDS - _t0))s" | tee -a "$LOG"
}

# ============ NOISE0: true run-to-run cv on control + flagged points =========
do_sect NOISE0 && { sect NOISE0 "control + flagged noisy points, x3 invocations, 10 reps each"
  for rep in 1 2 3; do
    BENCH_REPS=10 runpw neg      2048 2048  0
    BENCH_REPS=10 runpw write    2048 16384 0
    BENCH_REPS=10 runpw bcast    256  16384 0
    BENCH_REPS=10 runpw mulbcast 2048 1024  0
  done
  esect; }

# ============ MMCORE (cat 3): γ(cores) -- forced balanced split 1..32 cores ==
# cores {1,2,4,8,16,32} = splits {1x1,1x2,2x2,2x4,4x4,4x8}. Many shapes so the γ(cores)
# curve is seen at many compute/memory ratios and aspect ratios (crux -- expanded most).
do_sect MMCORE && { sect MMCORE "matmul forced to cores {1,2,4,8,16,32}, many K-ratios + shapes"
  ALLC=("1 1 1" "1 2 1" "2 2 1" "2 4 1" "4 4 1" "4 8 1")  # 6 balanced core counts (1..32)
  C2UP=("1 2" "2 2" "2 4" "4 4" "4 8")                    # cores 2..32 (shapes too big at 1 core)
  for KK in 512 1024 2048 4096; do for s in "${ALLC[@]}"; do runmmwd 2048 "$KK" 2048 $s; done; done
  for KK in 1024 2048 4096;     do for s in "${ALLC[@]}"; do runmmwd 1024 "$KK" 1024 $s; done; done
  for KK in 512 2048;           do for s in "${ALLC[@]}"; do runmmwd 512  "$KK" 512  $s; done; done
  for s in "${ALLC[@]}"; do runmmwd 4096 2048 2048 $s; done       # M=4096
  for s in "${ALLC[@]}"; do runmmwd 2048 2048 4096 $s; done       # N=4096
  for s in "${ALLC[@]}"; do runmmwd 1024 2048 4096 $s; done       # M<N
  for s in "${ALLC[@]}"; do runmmwd 4096 2048 1024 $s; done       # M>N
  for s in "${C2UP[@]}"; do runmmwd 4096 1024 4096 $s 1; done     # big M&N, K=1024 (cores 2..32)
  for s in "${C2UP[@]}"; do runmmwd 8192 2048 1024 $s 1; done     # tall M, small N (cores 2..32)
  esect; }

# ============ BMMLAY (cat 4): device layout [1,0,2] vs [0,1,2], copy-checked ==
do_sect BMMLAY && { sect BMMLAY "full bmm, {[0,1,2],[1,0,2]}^2 layouts, many B + shapes"
  L4=("0,1,2 0,1,2" "1,0,2 0,1,2" "0,1,2 1,0,2" "1,0,2 1,0,2")
  for B in 2 4 8 16; do for LL in "${L4[@]}"; do runbmml "$B" 1024 2048 1024 1 4 8 1 $LL; done; done
  for B in 4 8;      do for LL in "${L4[@]}"; do runbmml "$B" 2048 2048 1024 1 4 8 1 $LL; done; done
  for B in 4 8;      do for LL in "${L4[@]}"; do runbmml "$B" 1024 2048 2048 1 4 8 1 $LL; done; done
  for B in 8 16;     do for LL in "${L4[@]}"; do runbmml "$B" 512  2048 512  1 4 8 1 $LL; done; done
  for LL in "${L4[@]}"; do runbmml 4 1024 1024 1024 1 4 8 1 $LL; done
  esect; }

# ============ XPORT (cat 2): BW_eff(R,C) surface for the 4 transport ops ======
# Dense, independent R×C so both dependences are constrained (crux for transpose_outer).
do_sect XPORT && { sect XPORT "transpose/transpose_outer/cat0/cat1: dense fix-C sweep-R + fix-R sweep-C"
  for op in transpose transpose_outer cat0 cat1; do
    for R in 512 1024 2048 4096 8192 16384; do runpw "$op" "$R" 2048 0; done  # fix C=2048
    for C in 512 1024 4096 8192 16384;       do runpw "$op" 2048 "$C" 0; done  # fix R=2048
    for R in 512 2048 8192;                  do runpw "$op" "$R" 8192 0; done  # fix C=8192
    for C in 512 2048 8192;                  do runpw "$op" 8192 "$C" 0; done  # fix R=8192
    for R in 512 2048 8192;                  do runpw "$op" "$R" 4096 0; done  # fix C=4096
    for S in "1024 1024" "4096 4096";        do runpw "$op" $S 0; done         # diagonal
  done
  esect; }

# ============ BCAST (cat 1): write + broadcast ops, dense R×C ================
do_sect BCAST && { sect BCAST "write full grid + bcast/mulbcast/bcastcol/copy dense R×C"
  for R in 512 2048 8192 16384; do
    for C in 1024 2048 4096 8192 16384; do runpw write "$R" "$C" 0; done
  done
  for op in bcast mulbcast bcastcol copy; do
    for R in 256 2048 8192 16384; do
      for C in 2048 4096 8192 16384; do runpw "$op" "$R" "$C" 0; done
    done
  done
  esect; }

# ============ SMXUNROLL (cat 5): unrolled softmax + small-COLS, IR captured ===
do_sect SMXUNROLL && { sect SMXUNROLL "softmax_unrolled shapes x tiles + softmax_row_tiling small-COLS"
  for sh in "1024 512" "2048 512" "4096 512" "1024 2048" "2048 2048" "4096 2048"; do
    for t in 1 4 8 16; do runsmx softmax_unrolled $sh "$t"; done
  done
  for C in 128 256 512; do for t in 1 4 8 16; do runsmx softmax_row_tiling 2048 "$C" "$t"; done; done
  for C in 128 256 512; do for t in 4 8;      do runsmx softmax_row_tiling 4096 "$C" "$t"; done; done
  for C in 128 256;     do for t in 4 8;      do runsmx softmax_row_tiling 8192 "$C" "$t"; done; done
  esect; }

# ============ CTMM (cat 6): coarse matmul/bmm tile ladders + LX-spill =========
do_sect CTMM && { sect CTMM "matmul_row/k/nested ladders (multi-shape) + coarse bmm + LX-spill"
  for sh in "2048 2048 2048" "4096 2048 2048" "2048 2048 4096" "2048 4096 2048" "4096 4096 2048"; do
    for t in 1 2 4 8 16; do runmk matmul_row_tiling $sh "$t"; done
  done
  for sh in "2048 2048 2048" "1024 4096 1024" "4096 2048 2048"; do
    for t in 1 2 4 8 16; do runmk matmul_k_tiling $sh "$t"; done   # K-tiling control
  done
  for sh in "2048 2048 2048" "4096 2048 2048" "2048 2048 4096"; do
    for t in 1 2 4 8; do runmk mm_nested_m_k $sh "$t"; done
  done
  for t in 1 2 4 8 16; do runmkb bmm_k_tiling      4 1024 2048 1024 "$t"; done
  for t in 1 2 4 8 16; do runmkb bmm_k_tiling      4 1024 1024 1024 "$t"; done
  for t in 1 2 4 8;    do runmkb bmm_k_tiling      8 1024 2048 1024 "$t"; done
  for t in 1 2 4 8 16; do runmkb bmm_3d2d_k_tiling 4 1024 2048 1024 "$t"; done
  for t in 1 2 4 8;    do runmkb bmm_3d2d_k_tiling 8 1024 2048 1024 "$t"; done
  for t in 1 2 4; do runmk matmul_row_tiling 8192 2048 2048 "$t"; done  # LX-spill regime
  for t in 1 2 4; do runmk matmul_row_tiling 4096 2048 4096 "$t"; done
  esect; }

# ============ BMM2R (cat 4 baseline): full-vs-3d2d two-rate under noise =======
do_sect BMM2R && { sect BMM2R "full bmm vs 3d2d (shared weight), balanced 4x8, sweep B/K/shape"
  for B in 2 4 8 16; do
    runbmmwd bmm_wd      "$B" 1024 2048 1024 1 4 8 1
    runbmmwd bmm_wd_3d2d "$B" 1024 2048 1024 1 4 8 1
  done
  for K in 256 512 1024 2048 4096; do
    runbmmwd bmm_wd      8 1024 "$K" 1024 1 4 8 1
    runbmmwd bmm_wd_3d2d 8 1024 "$K" 1024 1 4 8 1
  done
  for sh in "4 2048 2048 1024" "4 1024 2048 2048" "4 2048 2048 2048" "8 2048 2048 1024" "4 512 2048 512"; do
    runbmmwd bmm_wd      $sh 1 4 8 1
    runbmmwd bmm_wd_3d2d $sh 1 4 8 1
  done
  esect; }

# ============ FLASH (NEW, LAST): flash-attention hint sweep, IR captured ======
# Vary the coarse tile counts (htiles/qtiles/ktiles) and the intra-tile work_div. Data
# + IR only (we do NOT model flash attn yet). See notes/flash_attn_hints.md. Heavier
# compile -> FLASH_TIMEOUT. Runs LAST + budget-guarded, so it soaks up remaining time
# after every crux run has landed. Three shapes; a full-shape (htiles,qtiles) grid too.
do_sect FLASH && { sect FLASH "flash_attn: sweep Lq/H tiles + work_div, 3 shapes + a tile grid, IR saved"
  # -- full shape (Lq=Lk=4096): vary Lq tiling, H tiling, work_div, + an Lk-tiling probe
  for qt in 1 2 4 8;  do runfa 32 4096 4096 128 8 "$qt" 1 "H:4,Lq:8,Lk:8"; done
  for ht in 2 4 8 16; do runfa 32 4096 4096 128 "$ht" 4 1 "H:4,Lq:8,Lk:8"; done
  for wd in "H:4,Lq:8,Lk:8" "H:2,Lq:8" "H:4,Lq:4" "H:8,Lq:4" "H:1,Lq:8,Lk:4" "H:2,Lq:4"; do
    runfa 32 4096 4096 128 8 4 1 "$wd"
  done
  runfa 32 4096 4096 128 8 4 2 "H:4,Lq:8,Lk:8"   # ktiles=2 probes the "no Lk tiling" FIXME
  # -- smaller shape (Lq=Lk=2048): denser + a (htiles x qtiles) grid
  for qt in 1 2 4 8 16; do runfa 32 2048 2048 128 8 "$qt" 1 "H:4,Lq:8,Lk:8"; done
  for ht in 2 4 8 16 32; do runfa 32 2048 2048 128 "$ht" 4 1 "H:4,Lq:8,Lk:8"; done
  for ht in 2 4 8; do for qt in 2 4 8; do runfa 32 2048 2048 128 "$ht" "$qt" 1 "H:4,Lq:8,Lk:8"; done; done
  for wd in "H:4,Lq:8" "H:2,Lq:8" "H:4,Lq:4"; do runfa 32 2048 2048 128 8 4 1 "$wd"; done
  # -- small shape (Lq=Lk=1024): fastest-compiling, fills the tail
  for qt in 1 2 4 8; do runfa 32 1024 1024 128 8 "$qt" 1 "H:4,Lq:8,Lk:8"; done
  for ht in 2 4 8;   do runfa 32 1024 1024 128 "$ht" 4 1 "H:4,Lq:8,Lk:8"; done
  esect; }

# Standalone: fold our own log into the DB (under run_db_sweep the master parses instead).
if [[ -z "${DB_LOG:-}" ]]; then
  DB_SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)"
  echo "==== runs DONE (${SECONDS}s total) -> parsing $LOG into notes/sweep_records.* ====" \
    | tee -a "$LOG"
  python "$ROOT/notes/parse_sweep_logs.py" "$LOG" --current-sha "$DB_SHA" \
    || echo "## !!! parse_sweep_logs.py failed (run by hand: python notes/parse_sweep_logs.py $LOG)"
fi
echo "==== outlier sweep COMPLETE -- forward $LOG + notes/sweep_records.* + haoyang_logs/ir/ ====" \
  | tee -a "$LOG"
