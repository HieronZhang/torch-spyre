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
# MATMUL-FAMILY CALIBRATION SWEEP (cost-model categories 3, 4, 5, 6) -- one run.
#
# Merges the matmul-overlap isolation sweep and the coarse-tile + BW(cores)
# sweep so the whole matmul-family model can be re-fit on CLEAN, noise-protocol
# data in a single session. The overnight data was too contended to fit these
# coefficients (gamma_eff swung +/-0.43 across repeats; coarse c_fill 641/645/
# 1260us at one config) -- every run here takes BENCH_REPS back-to-back profiled
# measurements -> min/median/std/cv so the fits land on clean points.
#
# Sections (SECTIONS env selects a subset; default = all):
#   MMISO_CORE  (cat 3) force the SAME matmul onto cores {1,2,4,8,16,32} balanced,
#               across the compute/memory ratio (K) and output size (M*N). 1 core =
#               NO split -> the pure per-core compute+overlap baseline; also nails
#               the low-core compute rate (mac_peak ~1046 vs the shipped 1140).
#   MMISO_SPLIT (cat 3) fixed cores=32, split SHAPE over the balanced range
#               {2x16,4x8,8x4,16x2} (not the 1x32/32x1 extremes) -> the split-shape
#               effect at fixed per-core work.
#   MMISO_BATCH (cat 4) true bmm (DEFAULT layout) forced cores {1,2,4,8} -> does
#               batching change the overlap / does the slow rate hold across cores.
#   BMMLAY      (cat 4) bmm_layout at MATCHED shape, default[0,1,2]^2 (slow) vs
#               [1,0,2]^2 (fast) + the two mixes, across shape+B, cores=32 4x8 split.
#               The clean noise-protocol source for the layout SLOW RATE -- the crux
#               cat-4 coefficient. Layout tagged into the label so the parser records
#               layout_a/layout_b (analyze_bmm_layout.py needs them). MMISO_BATCH is
#               default-layout only; the overnight bmm_layout data is too contended
#               (cv~35%) and IRCAP runs it at reps=1 with an unparseable tag.
#   CTFILL      (cat 6) coarse matmul (matmul_row_tiling, mm_nested_m_k) over tiles
#               {1,2,4,8,16} x shapes -> the per-tile pipeline FILL/DRAIN loop
#               overhead (io_hbm is constant across tiles, time rises ~ loop_trip *
#               c_fill; the SAME physics as the cat-3 overlap, per coarse tile).
#   BWCORES     (cat 5) memory-bound ops (read/sumrow/amax/mean/neg/copy +
#               softmax_row_tiling) forced to cores {1,2,4,8,16,32} -> BW(cores):
#               reductions run ~9x slower at 1 core (why softmax_unrolled, which
#               runs sencores=1, is -93%).
#
# 1-core matmul rows capped at MNK<=1.8e10 to stay under RUN_TIMEOUT. Output:
# <repo-root>/haoyang_logs/mm_family_<timestamp>.log. Fold with
# notes/parse_sweep_logs.py; analyze with notes/analyze_matmul_overlap.py.
#
#   bash docs/source/user_guide/examples/run_matmul_family_sweep.sh          # all
#   SECTIONS="MMISO_CORE CTFILL" bash .../run_matmul_family_sweep.sh          # subset
#   BENCH_REPS=5 bash .../run_matmul_family_sweep.sh                          # fewer reps
# ~205 runs. Ordered so the cat-3 crux (MMISO_CORE) lands first.
# ============================================================================

set -u
trap 'echo "## INTERRUPTED (SIGINT)"; exit 130' INT
trap 'echo "## TERMINATED (SIGTERM)"; exit 143' TERM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs haoyang_logs/ir
LOG="haoyang_logs/mm_family_$(date +%Y%m%d_%H%M%S).log"
SECTIONS="${SECTIONS:-MMISO_CORE MMISO_SPLIT MMISO_BATCH BMMLAY CTFILL BWCORES}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
export BENCH_REPS="${BENCH_REPS:-7}"
echo "==== matmul-family calibration sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS  reps: $BENCH_REPS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local all out; all=$(cat)
  out=$(printf '%s\n' "$all" | grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY|^TIMING ')
  if printf '%s\n' "$out" | grep -q '^SUMMARY'; then printf '%s\n' "$out" | tee -a "$LOG"
  else { echo "SUMMARY $1 FAILED"; printf '%s\n' "$all" | grep -vE '^\s*$' | tail -6 | sed 's/^/FAILDIAG /'; } | tee -a "$LOG"; fi
}
sect() { SECT="$1"; SECT_T0=$SECONDS; echo "## $1 -- ${2:-}" | tee -a "$LOG"; }
esect() { echo "## $SECT DONE in $((SECONDS - SECT_T0))s" | tee -a "$LOG"; }
_ok() {  # M K N m n k : cores<=32, splits divide dims, MNK<=3.4e10, 1-core cap 1.8e10
  local M=$1 K=$2 N=$3 m=$4 n=$5 k=$6 cores=$(( $4*$5*$6 ))
  (( cores<=32 )) || { echo "SKIP cores>32 $*" | tee -a "$LOG"; return 1; }
  (( M%m==0 && N%n==0 && K%k==0 )) || { echo "SKIP indivisible $*" | tee -a "$LOG"; return 1; }
  (( M*K*N<=34000000000 )) || { echo "SKIP MNK>3.4e10 $*" | tee -a "$LOG"; return 1; }
  (( cores>1 || M*K*N<=18000000000 )) || { echo "SKIP 1-core MNK>1.8e10 $*" | tee -a "$LOG"; return 1; }
}
runmmwd() {  # M K N m n k
  local M=$1 K=$2 N=$3 m=$4 n=$5 k=$6 cores=$(( $4*$5*$6 )) _t0=$SECONDS
  _ok "$@" || return 0
  echo "-- mmwd M=$M K=$K N=$N split m=$m n=$n k=$k (cores=$cores)" | tee -a "$LOG"
  SENCORES=32 WD_M="$m" WD_N="$n" WD_K="$k" SPYRE_DUMP_COST=1 \
    BENCH_OP=mmwd BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "mmwd M=$M K=$K N=$N split m=$m n=$n k=$k"
  echo "TIMING_RUN mmwd $M/$K/$N $m/$n/$k $((SECONDS-_t0))s" | tee -a "$LOG"
}
runbmmwd() {  # B M K N b m n k
  local B=$1 M=$2 K=$3 N=$4 b=$5 m=$6 n=$7 k=$8 cores=$(( $5*$6*$7*$8 )) _t0=$SECONDS
  (( cores<=32 && M%m==0 && N%n==0 && K%k==0 && B%b==0 )) || { echo "SKIP bmm $*" | tee -a "$LOG"; return 0; }
  echo "-- bmm_wd B=$B M=$M K=$K N=$N b=$b m=$m n=$n k=$k (cores=$cores)" | tee -a "$LOG"
  SENCORES=32 WD_B="$b" WD_M="$m" WD_N="$n" WD_K="$k" SPYRE_DUMP_COST=1 \
    BENCH_OP=bmm_wd BENCH_B="$B" BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "bmm_wd B=$B M=$M K=$K N=$N b=$b m=$m n=$n k=$k"
  echo "TIMING_RUN bmm_wd B=$B $M/$K/$N $((SECONDS-_t0))s" | tee -a "$LOG"
}
runbmml() {  # B M K N b m n k layoutA layoutB  -- bmm_layout, matched shape, chosen dim_order
  local B=$1 M=$2 K=$3 N=$4 b=$5 m=$6 n=$7 k=$8 la=$9 lb=${10} cores=$(( $5*$6*$7*$8 )) _t0=$SECONDS
  (( cores<=32 && M%m==0 && N%n==0 && K%k==0 && B%b==0 )) || { echo "SKIP bmml $*" | tee -a "$LOG"; return 0; }
  local irf="haoyang_logs/ir/bmml_B${B}_${M}x${K}x${N}_A${la//,/}_B${lb//,/}.txt"
  echo "-- bmm_layout B=$B M=$M K=$K N=$N b=$b m=$m n=$n k=$k layoutA=$la layoutB=$lb (cores=$cores)" | tee -a "$LOG"
  SENCORES=32 WD_B="$b" WD_M="$m" WD_N="$n" WD_K="$k" \
    WD_LAYOUT_A="$la" WD_LAYOUT_B="$lb" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=bmm_layout BENCH_B="$B" BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | tee "$irf" \
    | _emit "bmm_layout B=$B M=$M K=$K N=$N b=$b m=$m n=$n k=$k layoutA=$la layoutB=$lb"
  grep -qiE 'restickify|\bclone\b|insert_copy' "$irf" \
    && echo "WARN bmm_layout $la/$lb INSERTED A COPY (see $irf) -- layout row confounded" | tee -a "$LOG"
  echo "TIMING_RUN bmm_layout B=$B ${M}x${K}x${N} $la/$lb $((SECONDS-_t0))s" | tee -a "$LOG"
}
runmk() {  # <op> <M> <K> <N> <tiles>  coarse matmul, cores=32
  local _t0=$SECONDS
  echo "-- $1 M=$2 K=$3 N=$4 tiles=$5" | tee -a "$LOG"
  SENCORES=32 SPYRE_DUMP_COST=1 BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" BENCH_N="$4" BENCH_TILES="$5" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "$1 M=$2 K=$3 N=$4 tiles=$5"
  echo "TIMING_RUN $1 $2/$3/$4 t=$5 $((SECONDS-_t0))s" | tee -a "$LOG"
}
runcore() {  # <op> <rows> <cols> <cores>  memory-bound op forced to N cores
  local _t0=$SECONDS
  echo "-- $1 [$2,$3] cores=$4" | tee -a "$LOG"
  SENCORES="$4" LX_PLANNING=0 SPYRE_DUMP_COST=1 BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "$1 [$2,$3] cores=$4"
  echo "TIMING_RUN $1 [$2,$3] c=$4 $((SECONDS-_t0))s" | tee -a "$LOG"
}
# balanced (m,n) for cores 1..32: 1x1 1x2 2x2 2x4 4x4 4x8
CORES=(1 2 4 8 16 32); MSPL=(1 1 2 2 4 4); NSPL=(1 2 2 4 4 8)

# ---- preflight ----
if [[ -z "${SKIP_PREFLIGHT:-}" ]]; then
  echo "-- preflight ..." | tee -a "$LOG"
  PF=$(SENCORES=32 BENCH_OP=neg BENCH_ROWS=64 BENCH_COLS=64 BENCH_REPS=1 BENCH_WARMUP=1 \
    timeout -k 10 150 python "$PROFILE_OPS" 2>&1)
  printf '%s\n' "$PF" | grep -q '^SUMMARY .*kernel_us=' \
    && echo "-- preflight OK" | tee -a "$LOG" \
    || { echo "## PREFLIGHT FAILED -- device busy/wedged; recover then SKIP_PREFLIGHT=1"; printf '%s\n' "$PF" | tail -8 | sed 's/^/PREFLIGHT /'; } | tee -a "$LOG"
  printf '%s\n' "$PF" | grep -q '^SUMMARY .*kernel_us=' || exit 1
fi

# ============ MMISO_CORE (cat 3): cores 1..32 balanced, across ratio/size ====
has MMISO_CORE && { sect MMISO_CORE "cores {1,2,4,8,16,32} balanced, across K (ratio) and M*N (size)"
  for K in 512 1024 2048 4096; do
    for i in 0 1 2 3 4 5; do runmmwd 2048 "$K" 2048 "${MSPL[$i]}" "${NSPL[$i]}" 1; done
  done
  for MN in 1024 4096; do
    for i in 0 1 2 3 4 5; do runmmwd "$MN" 2048 "$MN" "${MSPL[$i]}" "${NSPL[$i]}" 1; done
  done
  esect; }

# ============ MMISO_SPLIT (cat 3): cores=32, balanced split shapes ==========
has MMISO_SPLIT && { sect MMISO_SPLIT "cores=32, split {2x16,4x8,8x4,16x2} (balanced, no 1x32/32x1)"
  for S in "2048 2048 2048" "4096 2048 2048" "2048 2048 4096"; do
    for mn in "2 16" "4 8" "8 4" "16 2"; do runmmwd $S $mn 1; done
  done
  esect; }

# ============ MMISO_BATCH (cat 4): bmm forced cores {1,2,4,8} ================
has MMISO_BATCH && { sect MMISO_BATCH "bmm forced cores {1,2,4,8}: batched overlap / layout rate"
  for B in 2 4 8; do
    runbmmwd "$B" 1024 1024 1024 1 1 1 1
    runbmmwd "$B" 1024 1024 1024 1 1 2 1
    runbmmwd "$B" 1024 1024 1024 1 2 2 1
    runbmmwd "$B" 1024 1024 1024 1 2 4 1
  done
  esect; }

# ============ BMMLAY (cat 4): bmm layout SLOW RATE, matched [0,1,2] vs [1,0,2] ==
# The layout-keyed slow COMPUTE RATE is the crux cat-4 coefficient. It needs CLEAN,
# noise-protocol timing of the matched default[0,1,2]^2 (slow) vs [1,0,2]^2 (fast)
# combos across shapes + B, with the layout tagged into the label so the parser
# records layout_a/layout_b (analyze_bmm_layout.py groups on them). The overnight
# bmm_layout data is layout-tagged but too CONTENDED (cv median 35%) to fit the rate;
# MMISO_BATCH above is DEFAULT-layout only (no fast baseline); IRCAP runs bmm_layout
# at reps=1 with an unparseable tag. This section is the only clean matched source.
# Fixed cores=32, balanced 4x8 split (the mm split term is separable on top).
has BMMLAY && { sect BMMLAY "bmm_layout matched slow[0,1,2]^2 vs fast[1,0,2]^2 + mixes, across shape+B"
  for B in 2 4 8; do
    for S in "1024 2048 1024" "2048 2048 1024" "1024 2048 2048"; do
      for lay in "0,1,2 0,1,2" "1,0,2 1,0,2" "0,1,2 1,0,2" "1,0,2 0,1,2"; do
        runbmml "$B" $S 1 4 8 1 $lay
      done
    done
  done
  esect; }

# ============ CTFILL (cat 6): coarse matmul per-tile fill/drain =============
has CTFILL && { sect CTFILL "coarse matmul over tiles {1,2,4,8,16}: per-tile fill/drain c_fill"
  for S in "2048 2048 2048" "4096 2048 2048" "2048 2048 4096" "4096 4096 2048" "8192 2048 2048"; do
    for t in 1 2 4 8 16; do runmk matmul_row_tiling $S "$t"; done
  done
  for S in "2048 2048 2048" "4096 2048 2048"; do
    for t in 1 2 4 8; do runmk mm_nested_m_k $S "$t"; done
  done
  for t in 1 2 4 8 16; do runmk matmul_k_tiling 2048 2048 2048 "$t"; done   # K-tiling control
  esect; }

# ============ BWCORES (cat 5): effective BW vs active cores =================
has BWCORES && { sect BWCORES "memory-bound ops forced to cores {1,2,4,8,16,32}: BW(cores) curve"
  for op in read sumrow amax mean neg copy; do
    for c in 1 2 4 8 16 32; do runcore "$op" 2048 8192 "$c"; done
  done
  for c in 1 2 4 8 32; do runcore softmax_row_tiling 2048 512 "$c"; done
  # CLEAN softmax_unrolled timing (forces sencores=1 internally) at the -93% outlier
  # shapes -> a low-cv validation anchor for the cat-5 BW(cores)/extractor fix. IRCAP
  # dumps its IR (structure) but only at reps=1; the fixed prediction needs a clean
  # measured point to land <10% against.
  for shape in "1024 512" "2048 512"; do
    set -- $shape
    for t in 1 4 16; do
      echo "-- softmax_unrolled [$1,$2] tiles=$t" | tee -a "$LOG"
      SENCORES=1 LX_PLANNING=1 SPYRE_DUMP_COST=1 BENCH_OP=softmax_unrolled \
        BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_TILES="$t" \
        timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
        | _emit "softmax_unrolled [$1,$2] tiles=$t"
    done
  done
  esect; }

echo "==== matmul-family sweep DONE in ${SECONDS}s -- $LOG ====" | tee -a "$LOG"
