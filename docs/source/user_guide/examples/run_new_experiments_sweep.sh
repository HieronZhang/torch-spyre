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
# NEW COST-MODEL EXPERIMENTS -- the comprehensive overnight sweep (supersedes the
# stale run_overnight_sweep.sh). One script covering every open item in
# notes/new_experiments_plan.md, each in a regime where the target effect
# dominates. Measured via the AIU profiler; every run dumps op_it_space_splits +
# MODEL FEATS + SUMMARY so parse_sweep_logs.py folds it into
# notes/sweep_records.{json,csv} and eval_model.py scores offline (no hardware).
#
#   Tier 0 (foundational -- everything downstream inherits these):
#     GAMMA   (X1, §10) γ(cores): ONE matmul shape at cores {4,8,16,32} -> the
#             overlap fraction vs core count (fit γ(L), not a single γ).
#     BWCORES (X2, §16) BW(cores): a memory-bound op at SENCORES {1..32} -> the
#             per-core effective-BW curve (fixes softmax_unrolled; validates ≥2-core).
#   Tier 1 (open modeling questions):
#     ARITY   (P1, §3) dependent add-chain vs the SAME bytes with NO dependency:
#             add{,3,4,5,6} + add_indep2, under LX_PLANNING 0 (buf round-trips) AND 1
#             (buf on-chip). Isolates the read-after-write cost from the byte count.
#     THINK   (M1, §12) tensor-size extreme: vary ONLY K into K<=128 at fixed M=N.
#     TINY    (M2, §12) small output: shrink M=N at fixed K (fixed-µs vs per-byte?).
#     MSPLIT  (M3, §12) deep-tail lopsided splits at 3 base shapes -> refine the knees.
#     BMM     (M4, §13) full bmm vs 3d2d (shared weight) across B and K -> validate the
#             two-rate model (BW_stream / BW_reread) on a fuller set.
#   Tier 2 (coarse):
#     COARSE  (C1/C2, §16) matmul_row tile ladder + mm_nested vs matmul_k_tiling control.
#
# cores<=32, every split divides its dim, every MNK<=3.4e10 (hang bound). No set -e;
# each run has a RUN_TIMEOUT guard.
#
#   bash docs/source/user_guide/examples/run_new_experiments_sweep.sh          # all
#   SECTIONS="GAMMA BWCORES" bash .../run_new_experiments_sweep.sh             # a subset
# Output: <repo-root>/haoyang_logs/new_experiments_<timestamp>.log (forward it).
# ============================================================================

set -u
# ${BASH_SOURCE[0]} is the script's real path whether run (`bash x.sh`) or sourced;
# plain $0 would be the shell name under `source`, breaking every path below.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/new_experiments_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-GAMMA BWCORES ARITY THINK TINY MSPLIT BMM COARSE}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== new-experiments sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runpw() {  # runpw <op> <rows> <cols> <lx>   pointwise, cores=32, scratchpad lx=0/1
  echo "-- $1 [$2,$3] lx=$4" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING="$4" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "$1 [$2,$3] lx=$4"
}
runcb() {  # runcb <op> <rows> <cols> <sencores> [tiles]   cores->BW calibration
  echo "-- $1 [$2,$3] sencores=$4 tiles=${5:-1}" | tee -a "$LOG"
  SENCORES="$4" LX_PLANNING=1 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" BENCH_TILES="${5:-1}" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "$1 [$2,$3] sencores=$4 tiles=${5:-1}"
}
runmmwd() {  # runmmwd <M> <K> <N> <m> <n> <k>   FORCED plain-matmul split, cores=m*n*k
  local M=$1 K=$2 N=$3 m=$4 n=$5 k=$6 cores=$(( $4 * $5 * $6 ))
  echo "-- mmwd M=$M K=$K N=$N split m=$m n=$n k=$k (cores=$cores M/m=$((M/m)) N/n=$((N/n)))" \
    | tee -a "$LOG"
  SENCORES=32 WD_M="$m" WD_N="$n" WD_K="$k" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=mmwd BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "mmwd M=$M K=$K N=$N split m=$m n=$n k=$k"
}
runbmmwd() {  # runbmmwd <op> <B> <M> <K> <N> <b> <m> <n> <k>   FORCED bmm split
  local op=$1 B=$2 M=$3 K=$4 N=$5 b=$6 m=$7 n=$8 k=$9 cores=$(( $6 * $7 * $8 * $9 ))
  echo "-- $op B=$B M=$M K=$K N=$N b=$b m=$m n=$n k=$k (cores=$cores)" | tee -a "$LOG"
  SENCORES=32 WD_B="$b" WD_M="$m" WD_N="$n" WD_K="$k" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$op" BENCH_B="$B" BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "$op B=$B M=$M K=$K N=$N b=$b m=$m n=$n k=$k"
}
runmk() {  # runmk <op> <M> <K> <N> <tiles>   coarse matmul (row / k / nested), cores=32
  echo "-- $1 M=$2 K=$3 N=$4 tiles=$5" | tee -a "$LOG"
  SENCORES=32 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" BENCH_N="$4" BENCH_TILES="$5" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "$1 M=$2 K=$3 N=$4 tiles=$5"
}

# ============ GAMMA (X1, §10): overlap fraction vs core count ================
# ONE fixed shape at cores {4,8,16,32} (compute ≈ memory so overlap is visible). The
# hidden fraction (measured vs additive/overlap) should fall with cores -> fit γ(L).
has GAMMA && { echo "## GAMMA γ(cores): 2048³ & 4096×2048×2048 at cores {4,8,16,32}" | tee -a "$LOG"
  for s in "2 2 1" "2 4 1" "4 4 1" "4 8 1"; do runmmwd 2048 2048 2048 $s; done
  for s in "2 2 1" "2 4 1" "4 4 1" "4 8 1"; do runmmwd 4096 2048 2048 $s; done
  # compute-dominant (K=4096) and near-HBM (K=1024) anchors at the same ladder
  for s in "2 2 1" "2 4 1" "4 4 1" "4 8 1"; do runmmwd 2048 4096 2048 $s; done; }

# ============ BWCORES (X2, §16): effective-BW vs core count ==================
has BWCORES && { echo "## BWCORES memory-bound op at SENCORES {1,2,4,8,16,32}" | tee -a "$LOG"
  for op in neg read; do for c in 1 2 4 8 16 32; do runcb "$op" 8192 2048 "$c"; done; done
  for c in 1 2 4 8 16 32; do runcb softmax_row_tiling 8192 2048 "$c" 8; done
  for c in 1 2 4 8 16 32; do runcb neg 16384 4096 "$c"; done; }

# ============ ARITY (P1, §3): dependent chain vs its controls, ± scratchpad ===
# add{,3..6}   = the dependent chain FUSED into one kernel (intermediates in HBM when LX off).
# add3/4_sep   = the SAME dependent chain, SEPARATE kernels (identical RAW dependency + bytes)
#               -> add3 vs add3_sep isolates the fusion/bundling cost with the dependency fixed.
# add_indep2   = TWO INDEPENDENT adds, same 4R:2W as add3 but NO dependency (harness fix applied).
has ARITY && { echo "## ARITY add{,3,4,5,6}+add_indep2+add3/4_sep at 3 shapes, LX 0 and 1" | tee -a "$LOG"
  for lx in 0 1; do
    for sh in "2048 1024" "2048 4096" "2048 16384"; do
      for op in add add3 add4 add5 add6 add_indep2 add3_sep add4_sep; do runpw "$op" $sh "$lx"; done
    done
  done; }

# ============ THINK (M1, §12): tensor-size extreme -- thin reduction K<=128 ===
has THINK && { echo "## THINK vary K in {16..512} at M=N=2048 & 4096, balanced 4x8" | tee -a "$LOG"
  for K in 16 32 64 128 256 512; do runmmwd 2048 "$K" 2048 4 8 1; done
  for K in 16 32 64 128 256 512; do runmmwd 4096 "$K" 4096 4 8 1; done; }

# ============ TINY (M2, §12): small output -- shrink M=N at fixed K ==========
has TINY && { echo "## TINY M=N in {128..2048} at K=2048, balanced 4x8" | tee -a "$LOG"
  for MN in 128 256 512 1024 2048; do runmmwd "$MN" 2048 "$MN" 4 8 1; done
  for MN in 256 512 1024;          do runmmwd "$MN" 4096 "$MN" 4 8 1; done; }

# ============ MSPLIT (M3, §12): deep-tail lopsided splits ====================
has MSPLIT && { echo "## MSPLIT {32x1,1x32,16x2,2x16,8x4} at 3 base shapes" | tee -a "$LOG"
  for s in "32 1" "1 32" "16 2" "2 16" "8 4"; do runmmwd 2048 4096 2048 $s 1; done
  for s in "32 1" "1 32" "16 2" "2 16" "8 4"; do runmmwd 6144 2048 2048 $s 1; done
  for s in "32 1" "1 32" "16 2" "2 16" "8 4"; do runmmwd 4096 4096 2048 $s 1; done; }

# ============ BMM (M4, §13): full vs 3d2d two-rate validation ================
# full re-reads the [K,N] weight per batch; 3d2d reads it once. full − 3d2d = the reread.
has BMM && { echo "## BMM full-vs-3d2d, balanced 4x8 b=1, sweep B then K" | tee -a "$LOG"
  for B in 2 4 8 16; do
    runbmmwd bmm_wd      "$B" 1024 2048 1024 1 4 8 1
    runbmmwd bmm_wd_3d2d "$B" 1024 2048 1024 1 4 8 1
  done
  for K in 256 1024 4096; do
    runbmmwd bmm_wd      8 1024 "$K" 1024 1 4 8 1
    runbmmwd bmm_wd_3d2d 8 1024 "$K" 1024 1 4 8 1
  done
  # the forced-b=B guard case (planner never emits it -> a guard, not a term)
  for B in 2 4 8; do runbmmwd bmm_wd "$B" 1024 2048 1024 "$B" 2 2 1; done; }

# ============ COARSE (C1/C2, §16): tile ladder + nested control ==============
has COARSE && { echo "## COARSE matmul_row tiles{1,2,4,8,16}x3 + mm_nested + k-tiling control" \
    | tee -a "$LOG"
  for t in 1 2 4 8 16; do runmk matmul_row_tiling 2048 2048 2048 "$t"; done
  for t in 1 2 4 8 16; do runmk matmul_row_tiling 4096 2048 2048 "$t"; done
  for t in 1 2 4 8;    do runmk mm_nested_m_k     2048 2048 2048 "$t"; done
  for t in 1 2 4 8 16; do runmk matmul_k_tiling   2048 2048 2048 "$t"; done; }

# Standalone: fold our own log into the DB (under run_db_sweep the master parses instead).
if [[ -z "${DB_LOG:-}" ]]; then
  DB_SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)"
  echo "==== runs DONE -> parsing $LOG into notes/sweep_records.{json,csv} ====" | tee -a "$LOG"
  python "$ROOT/notes/parse_sweep_logs.py" "$LOG" --current-sha "$DB_SHA" \
    || echo "## !!! parse_sweep_logs.py failed (run by hand: python notes/parse_sweep_logs.py $LOG)"
fi
echo "==== new-experiments sweep COMPLETE -- forward $LOG + notes/sweep_records.* ====" | tee -a "$LOG"
