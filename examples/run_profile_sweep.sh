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
# GOLDEN re-sweep: collect torch.profiler per-kernel DEVICE times (profile_ops
# SUMMARY lines) across the op ladder, to REBUILD the cost model from kernel time.
# We forget the old SPYRE_PROFILE_SYNC fit (its ~20us "fixed" was non-deterministic
# Memset/host overhead, not kernel cost) and refit fill + BW on `kernel_us`.
#
# Needs the kineto-spyre wheel (see docs/.../profiling/pytorch_profiler.md).
# Output: haoyang_logs/profile_sweep_<timestamp>.log (forward this file).
#
#   bash examples/run_profile_sweep.sh           # full sweep
#   SECTIONS="A D" bash examples/run_profile_sweep.sh   # just sections A and D
#
# WARNING: the profiled call has hit a ~60s sync hang ("possible lost completion").
# RUN SECTION A FIRST (12 runs) to gauge per-run wall time before the full sweep.

set -u
cd "$(dirname "$0")/.." || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/profile_sweep_$(date +%Y%m%d_%H%M%S).log"
SECTIONS="${SECTIONS:-A B C D E F}"
ROWS="${ROWS:-2048}"   # bigger than 512 so each of 32 cores gets many rows (>=64/core)
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1   # recompile each run

echo "==== golden profiler re-sweep $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)  rows: $ROWS  sections: $SECTIONS" \
  | tee -a "$LOG"

run() {  # run <op> <cols> -> append SUMMARY (io_hbm_bytes / kernel_us / bw_gbps / memset)
  local out
  out=$(BENCH_OP="$1" BENCH_ROWS="$ROWS" BENCH_COLS="$2" \
        python examples/profile_ops.py 2>/dev/null | grep '^SUMMARY')
  echo "${out:-SUMMARY op=$1 cols=$2 FAILED}" | tee -a "$LOG"
}
has() { [[ " $SECTIONS " == *" $1 "* ]]; }

# A) size-sweep fit on a clean 1R+1W op -> kernel = fill + bytes/BW
has A && { echo "## A size-sweep fit (neg, gelu)" | tee -a "$LOG"
  for n in 512 1024 2048 4096 8192 16384; do run neg "$n"; run gelu "$n"; done; }
# B) arithmetic-free: activations at one size (kernel times equal?)
has B && { echo "## B arithmetic-free" | tee -a "$LOG"
  for op in neg gelu relu sigmoid exp; do run "$op" 4096; done; }
# C) traffic / passes: neg(2) mul(3) add3(4) add4(5)
has C && { echo "## C traffic / stream count" | tee -a "$LOG"
  for op in neg mul add3 add4; do for n in 1024 4096 16384; do run "$op" "$n"; done; done; }
# D) bandwidth: read(1R) write(1W) copy/neg(1R1W) -> read/write/balanced BW
has D && { echo "## D bandwidth read/write/balanced" | tee -a "$LOG"
  for op in read write copy neg; do
    for n in 1024 4096 16384 65536; do run "$op" "$n"; done; done; }
# E) broadcast (cached -> 2-pass kernel?): row-vec bcast/mulbcast vs full add/mul, plus
#    col-vec bcastcol (b[R,1], a degenerate dim -> stick-padded device size). Check the
#    per-tensor dims/hbm-counted in the dump + bw_gbps to confirm caching both directions.
has E && { echo "## E broadcast (row + col)" | tee -a "$LOG"
  for op in bcast bcastcol add mulbcast mul; do
    for n in 1024 4096; do run "$op" "$n"; done; done; }
# F) reductions: read-dominated BW (sumrow size sweep); arith-free (sum/amax/mean); the
#    OTHER reduced axis (sumcol = dim 0); and the ring combine (sumall -> scalar output).
has F && { echo "## F reductions" | tee -a "$LOG"
  for op in sumrow sumcol amax mean sumall; do
    for n in 1024 4096 16384; do run "$op" "$n"; done; done; }

echo "==== DONE -> forward $LOG ====" | tee -a "$LOG"
