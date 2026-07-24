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
# BATCHED-MATMUL (bmm) SWEEP -- measure COMPUTE-bound bmm so the cost model's
# matmul compute term can be validated/fit for batching. All UNTILED (BENCH_TILES=1,
# no coarse tiling): a plain torch.bmm, work-divided across cores=32. Each run dumps
# ^MODEL (our prediction, incl. the parsed M/m + MACs) and ^SUMMARY (measured
# kernel_us), so we can (a) confirm the batch-dim mis-parse (M/m shows the batch,
# not the real rows) and (b) answer the open question:
#
#   Does the BATCH help fill the systolic array?  i.e. is pt_eff keyed on the
#   per-core rows M/m alone, or on batch*M/m (each batch adds row-passes)?
#
# BM1 answers it directly: fix M,N,K and scale B -- if time is linear in B at a
# constant per-MAC rate, the batch does NOT change fill; if the per-MAC rate
# improves as B grows, it does.
#
#   BM1  batch scaling:  M=N=1024, K=2048, B in {1,2,4,8}   (compute grows with B)
#   BM2  row scaling:    B=4, N=1024, K=2048, M in {256,512,1024,2048}  (pt_eff on M/m)
#   BM3  3d2d shared weight + a memory-bound control + K scaling (compute fraction)
#
# Shapes keep B*M*N*K <= ~1.7e10 (>=6.9e10 hangs). cores=32.
#
#   bash docs/source/user_guide/examples/run_bmm_sweep.sh          # BM1 BM2 BM3
# Output: <repo-root>/haoyang_logs/bmm_<timestamp>.log (forward it).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/bmm_$(date +%Y%m%d_%H%M%S).log"
[[ -n "${DB_LOG:-}" ]] && LOG=/dev/null   # under run_db_sweep: master writes the log
SECTIONS="${SECTIONS:-BM1 BM2 BM3}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1

echo "==== bmm sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  sections: $SECTIONS" | tee -a "$LOG"

has() { [[ " $SECTIONS " == *" $1 "* ]]; }
_emit() {
  local out; out=$(grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY|compute = MACs')
  echo "${out:-SUMMARY $1 FAILED}" | tee -a "$LOG"
}
runbmm() {  # runbmm <op> <B> <M> <K> <N>   UNTILED bmm (BENCH_TILES=1), cores=32
  echo "-- $1 B=$2 M=$3 K=$4 N=$5" | tee -a "$LOG"
  SENCORES=32 SPYRE_DUMP_IR=1 SPYRE_DUMP_COST=1 \
    BENCH_OP="$1" BENCH_B="$2" BENCH_ROWS="$3" BENCH_COLS="$4" BENCH_N="$5" \
    BENCH_TILES=1 \
    timeout -k 20 "${RUN_TIMEOUT:-240}" python "$PROFILE_OPS" 2>&1 | _emit "$1 B=$2 M=$3 K=$4 N=$5"
}

# ===== BM1: batch scaling -- does the batch fill the array? =================
# Fixed M=N=1024, K=2048; B doubles 1->8. Per-core rows M/m are CONSTANT across
# these (only the batch grows), so any change in the per-MAC rate is a batch effect.
has BM1 && { echo "## BM1 batch scaling: M=N=1024 K=2048, B in {1,2,4,8}" | tee -a "$LOG"
  for b in 1 2 4 8; do runbmm bmm_k_tiling "$b" 1024 2048 1024; done; }

# ===== BM2: row (M) scaling -- the pt_eff / per-core-rows curve =============
# Fixed B=4, N=1024, K=2048; M 256->2048 so per-core rows M/32 span 8->64.
has BM2 && { echo "## BM2 row scaling: B=4 N=1024 K=2048, M in {256,512,1024,2048}" \
    | tee -a "$LOG"
  for m in 256 512 1024 2048; do runbmm bmm_k_tiling 4 "$m" 2048 1024; done; }

# ===== BM3: 3d2d shared weight, memory-bound control, K (compute-fraction) ==
has BM3 && { echo "## BM3 3d2d + memory-bound control + K scaling" | tee -a "$LOG"
  # 3d2d shared weight (weight counted once) -- compare vs full bmm at same shape
  runbmm bmm_3d2d_k_tiling 8 1024 2048 1024
  runbmm bmm_k_tiling      8 1024 2048 1024
  # memory-bound control: tiny K -> memory dominates (compute term ~0)
  runbmm bmm_k_tiling 8 1024 64 1024
  # K scaling: raise compute fraction at fixed output
  for k in 512 1024 2048; do runbmm bmm_k_tiling 4 1024 "$k" 1024; done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
