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
# OVERNIGHT SWEEP -- ONE self-contained job + auto-fold (~200 runs, CPU-time-bound).
#
#   (A) BMM batch floor + effects (BF*/BS/BP/BN). A forced b=1 (batch serialized) run is ~-78%
#       even at a BALANCED split -> bmm is NOT just mm x batches; isolate the batch floor:
#       scaling in B, per-batch weight (full vs 3d2d), K/M/N dependence, split-shape carry,
#       forced-b=B pathology, natural coarse bmm.
#   (B) cores -> effective-BW calibration (CB). The coarse-tiling review found the ONE genuinely
#       missing datum is effective BW vs core count: softmax_unrolled is -91% because it runs at
#       cores=1 and the memory term has NO per-core BW scaling (~6-7 GB/s at 1 core vs ~100
#       assumed). Sweep several memory-bound op families over SENCORES {1..32}.
#   (C) coarse-matmul dataset (CR/CN/CK). The other coarse residuals are CODE fixes (mm_nested
#       compute*loop_trip, bmm_nested trpc, matmul_row underfill) -- NOT swept for discovery, but
#       a broad shape x tile dataset here validates those fixes and helps design the matmul_row
#       feature the review flagged (two rows with identical features measured 29% apart).
#       See notes/coarse_tiling_sweep_plan.md.
#
# Constraints: cores<=32, splits divide their dim (N in 64-elem STICKS), MNK<=3.4e10, LX-safe.
#
#   bash docs/source/user_guide/examples/run_overnight_sweep.sh          # everything + fold
#   SECTIONS="BF1 BF2" bash .../run_overnight_sweep.sh                   # a subset
# Output: ONE haoyang_logs/overnight_<ts>.log + updated notes/sweep_records.{json,csv}. Forward both.
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/overnight_$(date +%Y%m%d_%H%M%S).log"
SECTIONS="${SECTIONS:-BF1 BF2 BF3 BF4 BS BP BN CB1 CB2 CB3 CB4 CR CN CK}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
DB_SHA="$(git rev-parse --short HEAD 2>/dev/null)"

echo "==== overnight sweep $(date) ====" | tee "$LOG"
echo "git: $DB_SHA  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runbmmwd() {  # runbmmwd <op> <B> <M> <K> <N> <b> <m> <n> <k>   FORCED bmm split
  local op=$1 B=$2 M=$3 K=$4 N=$5 b=$6 m=$7 n=$8 k=$9 cores=$(( $6 * $7 * $8 * $9 ))
  echo "-- $op B=$B M=$M K=$K N=$N b=$b m=$m n=$n k=$k (cores=$cores M/m=$((M/m)) N/n=$((N/n)))" \
    | tee -a "$LOG"
  SENCORES=32 WD_B="$b" WD_M="$m" WD_N="$n" WD_K="$k" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$op" BENCH_B="$B" BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "$op B=$B M=$M K=$K N=$N b=$b m=$m n=$n k=$k"
}
runbmm() {  # runbmm <op> <B> <M> <K> <N> <tiles>   NATURAL coarse bmm (planner split, b=1)
  echo "-- $1 B=$2 M=$3 K=$4 N=$5 tiles=$6" | tee -a "$LOG"
  SENCORES=32 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_B="$2" BENCH_ROWS="$3" BENCH_COLS="$4" BENCH_N="$5" BENCH_TILES="$6" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "$1 B=$2 M=$3 K=$4 N=$5 tiles=$6"
}
runcb() {  # runcb <op> <rows> <cols> <sencores> [tiles]   cores->BW calibration
  echo "-- $1 [$2,$3] sencores=$4 tiles=${5:-1}" | tee -a "$LOG"
  SENCORES="$4" LX_PLANNING=1 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" BENCH_TILES="${5:-1}" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "$1 [$2,$3] sencores=$4 tiles=${5:-1}"
}
runmk() {  # runmk <op> <M> <K> <N> <tiles>   coarse matmul (row / k / nested), cores=32
  echo "-- $1 M=$2 K=$3 N=$4 tiles=$5" | tee -a "$LOG"
  SENCORES=32 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_ROWS="$2" BENCH_COLS="$3" BENCH_N="$4" BENCH_TILES="$5" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "$1 M=$2 K=$3 N=$4 tiles=$5"
}

# ============================ (A) BMM batch floor ============================
# BF1: batch floor -- full bmm, b=1, BALANCED split, sweep B, across 5 shape/split configs.
has BF1 && { echo "## BF1 full bmm b=1 balanced, sweep B (B=1 = single-matmul anchor), 5 configs" | tee -a "$LOG"
  for B in 1 2 4 8 16;    do runbmmwd bmm_wd "$B" 1024 2048 1024 1 4 8 1; done  # 4x8 base
  for B in 1 2 4 8 16;    do runbmmwd bmm_wd "$B" 1024 2048 1024 1 2 2 1; done  # 2x2 (cores4)
  for B in 1 2 4 8;       do runbmmwd bmm_wd "$B" 2048 2048 1024 1 4 8 1; done  # bigger M
  for B in 1 2 4 8 16;    do runbmmwd bmm_wd "$B" 1024 1024 1024 1 4 8 1; done  # smaller K
  for B in 1 2 4 8 16 32; do runbmmwd bmm_wd "$B"  512 2048  512 1 4 8 1; done; }  # small per-batch, B..32

# BF2: per-batch WEIGHT -- 3d2d (shared) at the BF1 shapes (pair with BF1 full) -> reload delta.
has BF2 && { echo "## BF2 3d2d (shared weight) b=1 4x8, matched to BF1 -> per-batch weight-reload delta" \
    | tee -a "$LOG"
  for B in 1 2 4 8 16; do runbmmwd bmm_wd_3d2d "$B" 1024 2048 1024 1 4 8 1; done
  for B in 1 2 4 8;    do runbmmwd bmm_wd_3d2d "$B" 2048 2048 1024 1 4 8 1; done
  for B in 1 2 4 8 16; do runbmmwd bmm_wd_3d2d "$B" 1024 1024 1024 1 4 8 1; done; }

# BF3: weight/K dependence -- full, b=1, 4x8, sweep K at 2 batch sizes.
has BF3 && { echo "## BF3 full bmm b=1 4x8, sweep K -> does the floor grow with |weight|=K*N?" | tee -a "$LOG"
  for K in 256 512 1024 2048 4096;  do runbmmwd bmm_wd 8 1024 "$K" 1024 1 4 8 1; done
  for K in 512 1024 2048 4096 8192; do runbmmwd bmm_wd 4 1024 "$K" 1024 1 4 8 1; done; }

# BF4: M/N (output-size) dependence -- full, b=1, 4x8, B=8, vary M then N.
has BF4 && { echo "## BF4 full bmm b=1 4x8 B=8, vary M (fixed K,N) then N -> floor vs output size" \
    | tee -a "$LOG"
  for M in 512 1024 2048; do runbmmwd bmm_wd 8 "$M" 2048 1024 1 4 8 1; done
  for N in 512 1024 2048; do runbmmwd bmm_wd 8 1024 2048 "$N" 1 4 8 1; done; }

# BS: split-shape carry -- b=1, m-ladder at 3 aspects (tall / wide / square).
has BS && { echo "## BS split-shape carry (b=1): m-ladder at M>N, N>M, M=N -> does §12a carry to bmm?" \
    | tee -a "$LOG"
  for mn in "2 16" "4 8" "8 4" "16 2" "32 1";      do runbmmwd bmm_wd 4 2048 2048 1024 1 $mn 1; done  # M>N
  for mn in "1 32" "2 16" "4 8" "8 4" "16 2" "32 1"; do runbmmwd bmm_wd 4 1024 2048 2048 1 $mn 1; done  # N>M
  for mn in "2 16" "4 8" "8 4" "16 2";             do runbmmwd bmm_wd 2 2048 2048 2048 1 $mn 1; done; } # M=N

# BP: forced-b=B pathology (guard) -- b=B batch split, several arrangements + 3d2d control.
has BP && { echo "## BP forced-b=B pathology (guard): b=B, per-batch {2x2,4x2}, + 3d2d control" | tee -a "$LOG"
  for B in 2 4 8; do runbmmwd bmm_wd      "$B" 1024 2048 1024 "$B" 2 2 1; done
  for B in 2 4;   do runbmmwd bmm_wd      "$B" 1024 2048 1024 "$B" 4 2 1; done
  for B in 2 4 8; do runbmmwd bmm_wd_3d2d "$B" 1024 2048 1024 "$B" 2 2 1; done; }

# BN: natural coarse bmm (planner b=1) over tile counts, 3 variants + a 2nd B.
has BN && { echo "## BN natural coarse bmm: k-tiling / 3d2d / nested, B=4 1024x2048x1024 + B=8" | tee -a "$LOG"
  for t in 1 2 4 8 16; do runbmm bmm_k_tiling      4 1024 2048 1024 "$t"; done
  for t in 1 2 4 8;    do runbmm bmm_nested_b_k    4 1024 2048 1024 "$t"; done
  for t in 1 2 4 8 16; do runbmm bmm_3d2d_k_tiling 4 1024 2048 1024 "$t"; done
  for t in 1 2 4 8;    do runbmm bmm_k_tiling      8 1024 2048 1024 "$t"; done; }

# ==================== (B) cores -> effective-BW calibration ==================
# CB1: pointwise (gelu, copy, add) x 2 shapes x SENCORES {1..32}.
has CB1 && { echo "## CB1 pointwise gelu/copy/add, [2048,8192]+[4096,4096], SENCORES {1,2,4,8,16,32}" \
    | tee -a "$LOG"
  for op in gelu copy add; do
    for sh in "2048 8192" "4096 4096"; do
      for c in 1 2 4 8 16 32; do runcb "$op" $sh "$c"; done
    done
  done; }

# CB2: reduction (sumrow, amax, mean) x [2048,8192] x SENCORES -- does cores add to the ROWS-rate?
has CB2 && { echo "## CB2 reduction sumrow/amax/mean [2048,8192], SENCORES {1,2,4,8,16,32}" | tee -a "$LOG"
  for op in sumrow amax mean; do
    for c in 1 2 4 8 16 32; do runcb "$op" 2048 8192 "$c"; done
  done; }

# CB3: broadcast + transport (other memory-bound families) x SENCORES.
has CB3 && { echo "## CB3 bcast + transpose [2048,8192], SENCORES {1,2,4,8,16,32}" | tee -a "$LOG"
  for c in 1 2 4 8 16 32; do runcb bcast     2048 8192 "$c"; done
  for c in 1 2 4 8 16 32; do runcb transpose 2048 8192 "$c"; done; }

# CB4: LOOPED softmax (matched to softmax_unrolled) x 2 shapes x SENCORES -- the direct control.
has CB4 && { echo "## CB4 softmax_row_tiling [2048,512]t16 + [1024,512]t8 (LX-safe), SENCORES {1..32}" \
    | tee -a "$LOG"
  for c in 1 2 4 8 16 32; do runcb softmax_row_tiling 2048 512 "$c" 16; done
  for c in 1 2 4 8 16 32; do runcb softmax_row_tiling 1024 512 "$c" 8;  done; }

# ==================== (C) coarse-matmul dataset (validate code fixes) ========
# CR: matmul_row_tiling (tiles M) x 5 shapes x tiles {1,2,4,8,16}.
has CR && { echo "## CR matmul_row_tiling tiles{1,2,4,8,16} x 5 shapes" | tee -a "$LOG"
  for sh in "2048 2048 2048" "4096 2048 2048" "2048 2048 4096" "4096 2048 4096" "8192 2048 2048"; do
    for t in 1 2 4 8 16; do runmk matmul_row_tiling $sh "$t"; done
  done; }

# CN: mm_nested_m_k (outer M x2, inner K) x 3 shapes x tiles {1,2,4,8}.
has CN && { echo "## CN mm_nested_m_k tiles{1,2,4,8} x 3 shapes" | tee -a "$LOG"
  for sh in "2048 2048 2048" "4096 2048 2048" "2048 2048 4096"; do
    for t in 1 2 4 8; do runmk mm_nested_m_k $sh "$t"; done
  done; }

# CK: matmul_k_tiling (tiles K -- the ACCURATE control) x 2 shapes x tiles {1,2,4,8,16}.
has CK && { echo "## CK matmul_k_tiling tiles{1,2,4,8,16} x 2 shapes (accurate control)" | tee -a "$LOG"
  for sh in "2048 2048 2048" "4096 2048 2048"; do
    for t in 1 2 4 8 16; do runmk matmul_k_tiling $sh "$t"; done
  done; }

echo "==== runs DONE -> parsing $LOG into notes/sweep_records.{json,csv} ====" | tee -a "$LOG"
python "$ROOT/notes/parse_sweep_logs.py" "$LOG" --current-sha "$DB_SHA" \
  || echo "## !!! parse_sweep_logs.py failed (run it by hand: python notes/parse_sweep_logs.py $LOG)"
echo "==== overnight sweep COMPLETE -- forward $LOG + notes/sweep_records.* ===="
