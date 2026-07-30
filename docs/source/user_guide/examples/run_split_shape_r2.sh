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
# SPLIT-SHAPE ROUND 2 -- break the confounds Round 1 could not. Round 1 ran everything
# at cores=32 where m*n=32, so the split-shape miss (real, up to -61%, size-gated,
# U-shaped in the split) could NOT be pinned: m, n, per-core M-tile, N-tile, and weight
# bytes all move together. The adversarial review refuted the "weight-broadcast fanout"
# reading (m=1 blows up with zero fanout; residual anti-correlates with weight bytes and
# tracks the per-core M-tile). Each section here is ONE deciding experiment:
#
#   R2A  Weight-vs-M-tile: FIX weight (K*N) + fanout (m=16,n=2), sweep M -> only M/m moves.
#        If residual tracks M/m, the multiplier is the per-core M-tile, not the weight.
#   R2B  n-side mirror at HEALTHY N/n (>=4 sticks): fix activation + n=16, vary N.
#   R2C  Transpose (m,n)<->(n,m) at an N>>M aspect: does the "asymmetry" flip with aspect
#        (=> tile geometry, not operand identity)?
#   R2D  Full m-ladder at 3 NON-anchor large shapes: does a knee reproduce off one shape?
#   R2E  m=1 / n=1 long-tile at LOW fanout: does a long unsplit tile blow up with no cohort?
#   R2F  Stick-quant vs n-fanout: large n at HEALTHY N/n (grow N) -- is the n-tail real or
#        just 1-2-stick tiny tiles?
#   R2G  bmm split-shape with batch SERIALIZED (b=1, the planner's real choice): does the
#        mm split-shape pattern carry to bmm? (Effect-2 check; NOT the forced-b=B pathology.)
#
# k=1, non-tiny (min(M,N)>=512, K>=256), cores<=32, each split divides its dim (N in 64-elem
# STICKS: n must divide N/64), every MNK <= ~3.4e10 (well under the ~6.9e10 hang bound).
#
#   bash docs/source/user_guide/examples/run_split_shape_r2.sh          # all
#   SECTIONS="R2A R2B" bash .../run_split_shape_r2.sh                   # the weight-vs-Mtile crux
# Output: <repo-root>/haoyang_logs/split_shape_r2_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/split_shape_r2_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null
SECTIONS="${SECTIONS:-R2A R2B R2C R2D R2E R2F R2G}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== split-shape round 2 $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runmmwd() {  # runmmwd <M> <K> <N> <m> <n> <k>
  local M=$1 K=$2 N=$3 m=$4 n=$5 k=$6 cores=$(( $4 * $5 * $6 ))
  echo "-- mmwd M=$M K=$K N=$N split m=$m n=$n k=$k (cores=$cores M/m=$((M/m)) N/n=$((N/n)))" \
    | tee -a "$LOG"
  SENCORES=32 WD_M="$m" WD_N="$n" WD_K="$k" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=mmwd BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "mmwd M=$M K=$K N=$N split m=$m n=$n k=$k"
}
runbmmwd() {  # runbmmwd <op> <B> <M> <K> <N> <b> <m> <n> <k>
  local op=$1 B=$2 M=$3 K=$4 N=$5 b=$6 m=$7 n=$8 k=$9 cores=$(( $6 * $7 * $8 * $9 ))
  echo "-- $op B=$B M=$M K=$K N=$N b=$b m=$m n=$n k=$k (cores=$cores M/m=$((M/m)) N/n=$((N/n)))" \
    | tee -a "$LOG"
  SENCORES=32 WD_B="$b" WD_M="$m" WD_N="$n" WD_K="$k" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$op" BENCH_B="$B" BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "$op B=$B M=$M K=$K N=$N b=$b m=$m n=$n k=$k"
}

# ===== R2A: weight FIXED (K*N=2048^2), fanout FIXED (m=16,n=2), sweep M -> M/m 64..512 =====
has R2A && { echo "## R2A weight+fanout fixed, M-tile varies (m=16 n=2): M/m in {64,128,256,512}" \
    | tee -a "$LOG"
  for M in 1024 2048 4096 8192; do runmmwd "$M" 2048 2048 16 2 1; done
  # M/m=512 held, fanout m varies 4->16 (absolute M and cores vary): fanout vs M-tile
  runmmwd 2048 2048 2048 4 2 1     # M/m=512, m=4,  cores8
  runmmwd 4096 2048 2048 8 2 1; }  # M/m=512, m=8,  cores16   (8192 16x2 above = m=16,cores32)

# ===== R2B: n-side mirror at HEALTHY N/n (fix activation M*K, n=16, vary N) =====
has R2B && { echo "## R2B n-side mirror (m=2 n=16), N/n in {256,512} (>=4 sticks)" | tee -a "$LOG"
  runmmwd 2048 2048 4096 2 16 1    # N/n=256
  runmmwd 2048 2048 8192 2 16 1; } # N/n=512  (matches R2A M/m=512 geometry, transposed)

# ===== R2C: transpose across an N>>M aspect (was only tested at M>>N) =====
has R2C && { echo "## R2C transpose at N>>M (2048x2048x8192): (16,2)/(2,16), (32,1)/(1,32)" \
    | tee -a "$LOG"
  for s in "16 2" "2 16" "32 1" "1 32"; do runmmwd 2048 2048 8192 $s 1; done; }

# ===== R2D: full m-ladder (cores=32) at 3 NON-anchor large shapes =====
has R2D && { echo "## R2D m-ladder m in {1,2,4,8,16,32} at 6144x2048x2048, 4096x4096x2048, 3072x2048x4096" \
    | tee -a "$LOG"
  for mn in "1 32" "2 16" "4 8" "8 4" "16 2" "32 1"; do runmmwd 6144 2048 2048 $mn 1; done
  for mn in "1 32" "2 16" "4 8" "8 4" "16 2" "32 1"; do runmmwd 4096 4096 2048 $mn 1; done
  for mn in "1 32" "2 16" "4 8" "8 4" "16 2" "32 1"; do runmmwd 3072 2048 4096 $mn 1; done; }

# ===== R2E: m=1 / n=1 long unsplit tile at LOW fanout (zero cohort) =====
has R2E && { echo "## R2E long unsplit tile, low fanout: m=1 sweep M (n=4); n=1 sweep N (m=4)" \
    | tee -a "$LOG"
  for M in 2048 4096 8192; do runmmwd "$M" 2048 2048 1 4 1; done   # m=1: M/m=M, cores4
  for N in 2048 4096 8192; do runmmwd 2048 2048 "$N" 4 1 1; done; } # n=1: N/n=N, cores4

# ===== R2F: stick-quant vs activation-fanout -- large n at HEALTHY N/n=512 =====
has R2F && { echo "## R2F large n at N/n=512 (8 sticks): n=16 and n=32, grow N to keep tile healthy" \
    | tee -a "$LOG"
  runmmwd 2048 1024 8192  2 16 1   # n=16, N/n=512
  runmmwd 1024 1024 8192  1 16 1   # n=16, N/n=512, cores16
  runmmwd 1024 1024 16384 1 32 1; } # n=32, N/n=512 (vs anchor n=32 at N/n=64) -- real n or quant?

# ===== R2G: bmm split-shape with batch SERIALIZED (b=1) -- Effect-2 check =====
has R2G && { echo "## R2G bmm B=4, b=1 (batch NOT split), m-ladder -- does mm split-shape carry to bmm?" \
    | tee -a "$LOG"
  for mn in "1 32" "2 16" "4 8" "8 4" "16 2" "32 1"; do runbmmwd bmm_wd 4 1024 2048 2048 1 $mn 1; done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
