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
# COARSE-TILING SWEEP -- characterize the coarse-tiling model over TILE COUNT and
# shape. The golden-profiler spot checks showed softmax_row_tiling +11.8% (fused
# turnaround a bit high) and matmul_row_tiling -9.9% (small per-tile overhead);
# this sweeps BENCH_TILES + shapes so those residuals can be fit as a function of
# tile size instead of a single point. Every op mirrors a coarse_tile/ example.
#
#   CT1  softmax_row_tiling: (ROWS,COLS) x TILES {1,4,8,16}.
#   CT2  matmul_row_tiling : (M,K,N)     x TILES {1,2,4,8}.
#   CT3  dim0 reductions sum/amax/amin (ctsum/ctamax/ctamin) x TILES {1,2,4,8}, ctsum LX{0,1}.
#   CT4  matmul_k_tiling + mm_nested_m_k (mirror run_matmul_k_tiled / mm_nested) x TILES {1..16}.
#   CT5  bmm: k-tiled / 3d2d shared-weight / nested B+K (mirror run_bmm_*) x TILES {1..16}.
#   CT6  softmax_unrolled: [B,D] tiled over B, unrolled loop, sencores=1 x TILES {1,4..32}.
#
# cores=32 (softmax_unrolled forces sencores=1). Measured via the AIU profiler; each
# run dumps ^MODEL (our prediction) + ^SUMMARY (kernel_us). TILES=1 = untiled baseline.
#
#   bash docs/source/user_guide/examples/run_coarse_tiling_sweep.sh   # all CT1..CT6
#   SECTIONS="CT4 CT5" bash .../run_coarse_tiling_sweep.sh            # just matmul-K + bmm
# Output: <repo-root>/haoyang_logs/coarse_tiling_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/coarse_tiling_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-CT1 CT2 CT3 CT4 CT5 CT6}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== coarse-tiling sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runtile() {  # runtile <op> <rows> <cols> <tiles> [lx=1]  softmax/ct-reduce, cores=32
  echo "-- $1 [$2,$3] tiles=$4 lx=${5:-1}" | tee -a "$LOG"
  SENCORES=32 LX_PLANNING="${5:-1}" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" BENCH_TILES="$4" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "$1 [$2,$3] tiles=$4 lx=${5:-1}"
}
runmt() {  # runmt <M> <K> <N> <tiles>   matmul_row_tiling (needs BENCH_N), cores=32
  echo "-- matmul_row_tiling M=$1 K=$2 N=$3 tiles=$4" | tee -a "$LOG"
  SENCORES=32 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP=matmul_row_tiling BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
    BENCH_TILES="$4" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "matmul_row_tiling M=$1 K=$2 N=$3 tiles=$4"
}
runmk() {  # runmk <op> <M> <K> <N> <tiles>   matmul_k_tiling / mm_nested_m_k, cores=32
  echo "-- $1 M=$2 K=$3 N=$4 tiles=$5" | tee -a "$LOG"
  SENCORES=32 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" BENCH_N="$4" BENCH_TILES="$5" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "$1 M=$2 K=$3 N=$4 tiles=$5"
}
runbmm() {  # runbmm <op> <B> <M> <K> <N> <tiles>   bmm_* (BENCH_B + BENCH_N), cores=32
  echo "-- $1 B=$2 M=$3 K=$4 N=$5 tiles=$6" | tee -a "$LOG"
  SENCORES=32 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_B="$2" BENCH_ROWS="$3" BENCH_COLS="$4" BENCH_N="$5" \
    BENCH_TILES="$6" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 | _emit "$1 B=$2 M=$3 K=$4 N=$5 tiles=$6"
}

# ============ CT1: softmax_row_tiling, tile-count x shape ===================
# tiles=1 is the UNTILED reference (intermediates in HBM). LX-SAFE for the tiled runs:
# keep tile_rows = ROWS/tiles <= 4096 so the per-tile intermediate (tile_rows x COLS x 2B)
# stays <= 33.5MB. tiles=2 on [16384,4096] (67MB tile) would overflow the LX planner and
# hang, so it is dropped (tiles=1 there is fine -- HBM, no LX tile).
has CT1 && { echo "## CT1 softmax_row_tiling: tiles=1 ref + tiled (tile_rows<=4096)" \
    | tee -a "$LOG"
  for t in 1 4 8 16;   do runtile softmax_row_tiling 16384 4096 "$t"; done  # tr 4096..1024
  for t in 1 4 8 16;   do runtile softmax_row_tiling 16384 2048 "$t"; done  # tr 4096..1024
  for t in 1 2 4 8 16; do runtile softmax_row_tiling 8192  2048 "$t"; done  # tr 4096..512
  for t in 1 2 4 8;    do runtile softmax_row_tiling 4096  4096 "$t"; done; }  # tr 2048..512

# ============ CT2: matmul_row_tiling, tile-count x shape ====================
has CT2 && { echo "## CT2 matmul_row_tiling: 3 shapes x TILES {1,2,4,8}" | tee -a "$LOG"
  for mkn in "2048 2048 2048" "4096 2048 2048" "2048 2048 4096"; do
    for t in 1 2 4 8; do runmt $mkn "$t"; done
  done; }

# ============ CT3: dim0 reductions (sum/amax/amin), TILE-COUNT sweep ========
# Mirror run_{sum,amax,amin}_dim0_tiled: reduce [B,D] over dim0, tiling B. tiles=1 is
# the UNTILED reference. ctsum also swept LX off to read the LX-reuse saving.
has CT3 && { echo "## CT3 ctsum/ctamax/ctamin [4096,2048] TILES{1,2,4,8}; ctsum also LX=0" \
    | tee -a "$LOG"
  for op in ctsum ctamax ctamin; do
    for t in 1 2 4 8; do runtile "$op" 4096 2048 "$t"; done
  done
  for t in 1 2 4 8; do runtile ctsum 4096 2048 "$t" 0; done; }  # LX off, compare

# ============ CT4: matmul K-tiling + nested, TILE-COUNT sweep ================
# Normal shapes (dims 1k-2k), K divisible by every tile count. tiles=1 = untiled
# reference. matmul_k_tiling tiles K (reduction); mm_nested_m_k = outer-M x2, inner-K.
has CT4 && { echo "## CT4 matmul_k_tiling (M=K=N=2048 + 1024x4096x1024) + mm_nested, TILES{1..16}" \
    | tee -a "$LOG"
  for t in 1 2 4 8 16; do runmk matmul_k_tiling 2048 2048 2048 "$t"; done  # K/tile 2048..128
  for t in 1 2 4 8 16; do runmk matmul_k_tiling 1024 4096 1024 "$t"; done  # longer K
  for t in 1 2 4 8;    do runmk mm_nested_m_k   2048 2048 2048 "$t"; done; }  # nested M+K

# ============ CT5: batched matmul, TILE-COUNT sweep =========================
# Normal shapes B=4 M=1024 K=2048 N=1024 (B*M*N*K=8.6e9). tiles=1 = untiled reference.
# K-tiled / 3d2d shared weight / nested outer-B inner-K.
has CT5 && { echo "## CT5 bmm k-tiled / 3d2d / nested B+K: B=4 M=1024 K=2048 N=1024, TILES{1..16}" \
    | tee -a "$LOG"
  for t in 1 2 4 8 16; do runbmm bmm_k_tiling      4 1024 2048 1024 "$t"; done
  for t in 1 2 4 8 16; do runbmm bmm_3d2d_k_tiling 4 1024 2048 1024 "$t"; done  # shared 2D weight
  for t in 1 2 4 8;    do runbmm bmm_nested_b_k    4 1024 2048 1024 "$t"; done; }  # outer B, inner K

# ============ CT6: softmax_unrolled (unrolled loop, no CoarseTileInfo) ======
# sencores=1 (single core). tiles=1 = untiled reference (full working set -> may spill).
# D=512 so the tiled per-tile working set 2*(B/tiles)*D*2B stays ~<=512KB LX.
has CT6 && { echo "## CT6 softmax_unrolled: [B,512] tiled over B (unroll_loops, sencores=1), TILES{1..32}" \
    | tee -a "$LOG"
  for t in 1 4 8 16;  do runtile softmax_unrolled 1024 512 "$t"; done
  for t in 1 8 16 32; do runtile softmax_unrolled 2048 512 "$t"; done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
