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
# SPLIT-SHAPE SWEEP -- the model under-predicts lopsided matmul splits (16x2 is
# 1.65x slower than 4x8 on HW at the SAME per-core tile area, yet predicted equal).
# The area-based spill term is symmetric in M/N so it cannot see this. The error
# tracks the per-core N-tile width (N/n) -- but at fixed 32 cores m*n=32, so "wide
# N-tile (small n)" and "weight broadcast to many M-cores (large m)" are CONFOUNDED.
# This sweep FORCES splits (spyre_hint work_div) to break that confound:
#   H1 = wide per-core N-tile (N/n) is intrinsically slow.
#   H2 = re-broadcasting the [K,N] weight to a large-m fanout is slow.
# All fit rows are k=1 and non-tiny (min(M,N)>=512, K>=256): the effect only
# separates cleanly away from the small-kernel floor and the K-split PSUM error.
#
#   DA   confounded reference: plain mm, full split enumeration at 2 shapes.
#   DB   H1 isolator: N/n varies with m (fanout) PINNED -> err vs N/n at fixed m.
#   DC   H2 isolator: m varies with N/n PINNED (via core count) -> err vs m at fixed N/n.
#   EDGE the missing edge pairs (32x1 vs 1x32, 8x2 vs 2x8) at 2 large shapes.
#   AR   aspect ratios / 2nd base shape so the fit is not tied to one geometry.
#   LADDER core-count ladder: m in {1,2,4,8,16} at fixed N/n (H2 over a wide m range).
#   BM   bmm serialization: per-batch split FIXED 2x2, vary B (cores 4..32); PLUS a
#        plain-mm control at each core count (subtract the model's core-scaling error).
#   BM2  3d2d-vs-full at matched B/split: the CLEAN weight-fanout (H2) isolator.
#   BMA  bmm Column-A: fixed B, forced per-batch split arrangement {2x2,4x1,1x4}.
#   KCTL k=2 controls (few, labeled) -- confirm the K-split PSUM path (do not fit these).
#
# cores<=32, every split divides its dim, every MNK<=3.4e10 (hang bound). Measured via
# the AIU profiler; each run dumps op_it_space_splits + MODEL FEATS + SUMMARY so the
# parser folds it into notes/sweep_records.{json,csv} and eval_model.py scores offline.
#
#   bash docs/source/user_guide/examples/run_split_shape_sweep.sh          # all (exhaustive)
#   SECTIONS="DB DC" bash .../run_split_shape_sweep.sh                     # just the crux
# Output: <repo-root>/haoyang_logs/split_shape_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/split_shape_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the unified log
SECTIONS="${SECTIONS:-DA DB DC EDGE AR LADDER BM BM2 BMA REPEAT KCTL}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== split-shape sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
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
runbmmwd() {  # runbmmwd <op> <B> <M> <K> <N> <b> <m> <n> <k>   FORCED bmm split, cores=b*m*n*k
  local op=$1 B=$2 M=$3 K=$4 N=$5 b=$6 m=$7 n=$8 k=$9 cores=$(( $6 * $7 * $8 * $9 ))
  echo "-- $op B=$B M=$M K=$K N=$N b=$b m=$m n=$n k=$k (cores=$cores M/m=$((M/m)) N/n=$((N/n)))" \
    | tee -a "$LOG"
  SENCORES=32 WD_B="$b" WD_M="$m" WD_N="$n" WD_K="$k" SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$op" BENCH_B="$B" BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "$op B=$B M=$M K=$K N=$N b=$b m=$m n=$n k=$k"
}

# ============ DA: confounded full-split enumeration (2 shapes) ===============
has DA && { echo "## DA plain-mm full split enum: 1024x2048x1024 + 8192x2048x2048" | tee -a "$LOG"
  for s in "8 4" "4 8" "16 2" "2 16" "32 1" "1 32"; do runmmwd 1024 2048 1024 $s 1; done
  for s in "8 4" "4 8" "16 2" "2 16" "32 1" "1 32"; do runmmwd 8192 2048 2048 $s 1; done; }

# ============ DB: H1 isolator -- N/n varies, m (fanout) PINNED ===============
# m=8 held throughout; N/n spans 128..2048. If err tracks N/n here (m pinned) -> H1.
has DB && { echo "## DB H1: m=8 pinned, sweep N -> N/n in {128..2048}" | tee -a "$LOG"
  for N in 512 1024 2048 4096; do runmmwd 2048 2048 "$N" 8 4 1; done   # n=4: N/n 128..1024
  for N in 1024 2048 4096;     do runmmwd 2048 2048 "$N" 8 2 1; done   # n=2: N/n 512..2048
  for N in 1024 2048 4096;     do runmmwd 4096 2048 "$N" 8 4 1; done   # 2nd M, N/n 256..1024
  for N in 512 1024 2048;      do runmmwd 2048 4096 "$N" 8 4 1; done   # K=4096, N/n 128..512
  # SIZE CONTROL: N/n=512 fixed at 3 absolute sizes (the effect should grow with time).
  runmmwd 1024 2048 2048 8 4 1; }   # N/n=512 @ MNK=4.3e9 (vs 2048^3 @8.6e9, 4096x2048x2048 @1.7e10)

# ============ DC: H2 isolator -- m varies, N/n PINNED =========================
# N/n held (256 then 512); m swept via core count. If err tracks m here (N/n pinned) -> H2.
has DC && { echo "## DC H2: N/n pinned, sweep m via cores {8,16,32} at 4 N/n levels" | tee -a "$LOG"
  for m in 2 4 8; do runmmwd 2048 2048 512  "$m" 4 1; done   # N/n=128, m in {2,4,8}
  for m in 2 4 8; do runmmwd 2048 2048 1024 "$m" 4 1; done   # N/n=256
  for m in 2 4 8; do runmmwd 2048 2048 2048 "$m" 4 1; done   # N/n=512
  for m in 2 4 8; do runmmwd 2048 2048 4096 "$m" 4 1; done; }  # N/n=1024 (MNK<=1.7e10)

# ============ EDGE: the missing edge pairs at 2 large shapes ==================
has EDGE && { echo "## EDGE 32x1 vs 1x32, 8x2 vs 2x8 (+16x2/4x8 ref): 3 shapes" | tee -a "$LOG"
  for s in "32 1" "1 32" "8 2" "2 8" "16 2" "4 8"; do runmmwd 8192 2048 2048 $s 1; done
  for s in "32 1" "1 32" "8 4" "4 8";              do runmmwd 4096 2048 4096 $s 1; done
  for s in "16 2" "2 16" "8 4" "4 8";              do runmmwd 2048 2048 2048 $s 1; done; }

# ============ AR: aspect ratios / 2nd base shape =============================
has AR && { echo "## AR aspect ratios: 4 base shapes" | tee -a "$LOG"
  for s in "4 8" "16 2" "8 4" "2 16"; do runmmwd 2048 4096 2048 $s 1; done  # K=4096
  for s in "16 2" "4 8" "8 4";        do runmmwd 4096 2048 2048 $s 1; done  # tall M
  for s in "16 2" "4 8" "8 4";        do runmmwd 2048 2048 4096 $s 1; done  # wide N
  for s in "16 2" "4 8" "8 4" "2 16"; do runmmwd 4096 4096 1024 $s 1; done; }  # narrow N base

# ============ LADDER: m in {1,2,4,8,16} at FIXED N/n (wide-m H2) =============
has LADDER && { echo "## LADDER m in {1,2,4,8,16} at N/n=512 (n=2) + N/n=256 (n=4) endpoints" \
    | tee -a "$LOG"
  for m in 1 2 4 8 16; do runmmwd 2048 2048 1024 "$m" 2 1; done   # N/n=512, cores 2..32
  runmmwd 2048 2048 1024 1 4 1; }                                  # N/n=256, m=1 (extends DC)

# ============ BM: bmm serialization + plain-mm core-scaling control ==========
# per-batch split FIXED 2x2, batch fully split (b=B) -> total cores 4,8,16,32. If time/B
# rises, it is serialization OR the model's core-scaling error -- the mm control isolates.
has BM && { echo "## BM bmm b=B, m=n=2 (cores 4..32) + matched plain-mm control, 2 shapes" \
    | tee -a "$LOG"
  # shape A: 1024x2048x1024 (per-batch core tile 512x512)
  for B in 1 2 4 8; do runbmmwd bmm_wd "$B" 1024 2048 1024 "$B" 2 2 1; done
  runmmwd 1024 2048 1024 2 2 1   # cores4  per-core 512x512  (mm control, no batch)
  runmmwd 2048 2048 1024 4 2 1   # cores8  per-core 512x512
  runmmwd 2048 2048 2048 4 4 1   # cores16 per-core 512x512
  runmmwd 4096 2048 2048 8 4 1   # cores32 per-core 512x512
  # shape B: 2048x2048x1024 (per-batch core tile 1024x512) -- check the rise is shape-robust
  for B in 1 2 4 8; do runbmmwd bmm_wd "$B" 2048 2048 1024 "$B" 2 2 1; done
  runmmwd 2048 2048 1024 2 2 1   # cores4  per-core 1024x512  (mm control)
  runmmwd 4096 2048 1024 4 2 1   # cores8  per-core 1024x512
  runmmwd 4096 2048 2048 4 4 1   # cores16 per-core 1024x512
  runmmwd 8192 2048 2048 8 4 1; }  # cores32 per-core 1024x512

# ============ BM2: 3d2d-vs-full at matched B/split (clean weight-fanout) =====
has BM2 && { echo "## BM2 full-bmm vs 3d2d shared-weight, matched B and 2x2 split" | tee -a "$LOG"
  for B in 2 4 8; do
    runbmmwd bmm_wd      "$B" 1024 2048 1024 "$B" 2 2 1
    runbmmwd bmm_wd_3d2d "$B" 1024 2048 1024 "$B" 2 2 1
  done; }

# ============ BMA: bmm Column-A -- fixed B, split arrangement ================
has BMA && { echo "## BMA bmm fixed B, per-batch split {2x2,4x1,1x4} (cores fixed)" | tee -a "$LOG"
  runbmmwd bmm_wd 4 1024 2048 1024 4 2 2 1   # 2x2  cores16
  runbmmwd bmm_wd 4 1024 2048 1024 4 4 1 1   # 4x1  cores16
  runbmmwd bmm_wd 4 1024 2048 1024 4 1 4 1   # 1x4  cores16
  runbmmwd bmm_wd 8 1024 2048 1024 8 4 1 1   # 4x1  cores32
  runbmmwd bmm_wd 8 1024 2048 1024 8 1 4 1; }  # 1x4  cores32

# ============ REPEAT: run-to-run noise on the key anchors ====================
has REPEAT && { echo "## REPEAT 3x the two anchors (16x2 slow, 4x8 fast) for noise" | tee -a "$LOG"
  for _ in 1 2 3; do runmmwd 8192 2048 2048 16 2 1; done
  for _ in 1 2 3; do runmmwd 8192 2048 2048 4 8 1; done; }

# ============ KCTL: k=2 controls (do NOT fit -- PSUM path sanity) ============
has KCTL && { echo "## KCTL WD_K=2 controls (labeled; excluded from the split-shape fit)" | tee -a "$LOG"
  runmmwd 2048 2048 2048 2 2 2   # cores8
  runmmwd 2048 2048 2048 4 2 2   # cores16
  runmmwd 2048 2048 2048 2 4 2   # cores16
  runmmwd 2048 2048 2048 4 4 2; }  # cores32

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
