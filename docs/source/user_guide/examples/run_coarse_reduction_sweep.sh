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
# COARSE-REDUCTION SWEEP -- the deciding experiment for the softmax_unrolled miss
# (cat-5's headline op, -90%; scoped to cat-6 because it is a coarse/tiled effect).
#
# HYPOTHESIS (from the existing SINGLE-SHOT data, to confirm with reps here): a FUSED
# coarse reduction (softmax) runs at a REDUCTION-like bandwidth that (a) scales with
# active cores (the same shared-HBM-bus effect as the shipped plain-reduction g(cores)),
# and (b) carries a TILED fill/drain penalty -- the coarse loop (loop_count>1) halves the
# rate vs the untiled kernel. Evidence (n=1, needs reps):
#   softmax_unrolled @ cores=1:  UNTILED (tiles=1) bw ~= 24.8 GB/s (VERY consistent, 5
#     shapes 24.6-24.9); TILED (tiles>=4) bw ~= 11 GB/s  -> tiled/untiled ~= 0.44.
#   softmax_row_tiling (tiled) bw scales with cores: 11/20/37/60/75/124 at c1/2/4/8/16/32.
# The cost model currently charges bw_peak=150 on the fused (len>1) branch regardless of
# cores or tiling -> the -90% miss. A fix must distinguish tiled-vs-untiled AND cores, and
# -- unlike the plain-reduction g(cores) -- it TOUCHES cores=32 softmax too (which is also
# mispredicted, "softmax" category 34.6%), so it is NOT gold-safe by construction and must
# be fit on REPEAT-BACKED data, not the current singletons. Hence this sweep.
#
# It sweeps softmax_row_tiling over cores x tiles (forced via SENCORES / BENCH_TILES) and
# softmax_unrolled (cores=1 by design) over tiles, all with the noise protocol. Fold, then:
#   (1) confirm the untiled~=24.8 / tiled~=11 split at cores=1 holds under reps;
#   (2) map bw(cores, tiled?) for the coarse fused reduction;
#   (3) check whether one (cores-scale x tiled-derate) form fits both softmax ops AND leaves
#       cores=32 within tolerance -> only THEN ship a fused-coarse-reduction bw term.
#
#   bash docs/source/user_guide/examples/run_coarse_reduction_sweep.sh
# ~66 runs; reps cheap. IR is also dumped for the loop_count/CoarseTileInfo per (tiles).
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs haoyang_logs/ir
LOG="haoyang_logs/coarse_reduction_$(date +%Y%m%d_%H%M%S).log"
export BENCH_REPS="${BENCH_REPS:-7}"
echo "==== COARSE-REDUCTION $(date)  git:$(git rev-parse --short HEAD 2>/dev/null)  reps:$BENCH_REPS ====" | tee "$LOG"

_emit() {
  local all out; all=$(cat)
  out=$(printf '%s\n' "$all" | grep -E 'op_it_space_splits|loop_count|^IO |^MODEL |^SUMMARY|^TIMING ')
  if printf '%s\n' "$out" | grep -q '^SUMMARY'; then printf '%s\n' "$out" | tee -a "$LOG"
  else { echo "SUMMARY $1 FAILED"; printf '%s\n' "$all" | grep -vE '^\s*$' | tail -6 | sed 's/^/FAILDIAG /'; } | tee -a "$LOG"; fi
}
runsmx() {  # op ROWS COLS tiles cores  (cores via SENCORES; softmax_unrolled ignores it, forces 1)
  local op=$1 R=$2 C=$3 t=$4 c=$5 _t0=$SECONDS
  local ir="haoyang_logs/ir/coarsered_${op}_${R}x${C}_t${t}_c${c}.txt"
  echo "-- $op ROWS=$R COLS=$C tiles=$t sencores=$c" | tee -a "$LOG"
  SENCORES="$c" SPYRE_DUMP_IR="${DUMP_IR:-0}" SPYRE_DUMP_COST=1 \
    BENCH_OP="$op" BENCH_ROWS="$R" BENCH_COLS="$C" BENCH_TILES="$t" \
    timeout -k 20 "${RUN_TIMEOUT:-240}" python "$PROFILE_OPS" 2>&1 \
    | { [[ "${DUMP_IR:-0}" == 1 ]] && tee "$ir" || cat; } \
    | _emit "$op ROWS=$R COLS=$C tiles=$t cores=$c"
  echo "TIMING_RUN $op $R/$C t$t c$c $((SECONDS-_t0))s" | tee -a "$LOG"
}

PF=$(SENCORES=32 BENCH_OP=neg BENCH_ROWS=64 BENCH_COLS=64 BENCH_REPS=1 BENCH_WARMUP=1 \
  timeout -k 10 150 python "$PROFILE_OPS" 2>&1)
printf '%s\n' "$PF" | grep -q '^SUMMARY .*kernel_us=' \
  || { echo "## PREFLIGHT FAILED -- ABORTING." | tee -a "$LOG"; printf '%s\n' "$PF" | tail -6 | sed 's/^/PREFLIGHT /' | tee -a "$LOG"; exit 1; }

# softmax_row_tiling: cores x tiles grid (the coarse fused reduction that CAN vary cores)
echo "## softmax_row_tiling: cores {1,2,4,8,16,32} x tiles {1,4,8,16}, 2 shapes" | tee -a "$LOG"
for shape in "2048 512" "4096 2048"; do
  set -- $shape
  for t in 1 4 8 16; do
    for c in 1 2 4 8 16 32; do runsmx softmax_row_tiling "$1" "$2" "$t" "$c"; done
  done
done
# softmax_unrolled: cores=1 by design; sweep tiles to lock the untiled(24.8)/tiled(11) split under reps
echo "## softmax_unrolled: tiles {1,4,8,16} x shapes (cores forced to 1 internally)" | tee -a "$LOG"
for shape in "2048 512" "4096 2048"; do
  set -- $shape
  for t in 1 4 8 16; do runsmx softmax_unrolled "$1" "$2" "$t" 1; done
done
# a couple of IR dumps to read loop_count/CoarseTileInfo per tiles
DUMP_IR=1 runsmx softmax_row_tiling 2048 512 1 8
DUMP_IR=1 runsmx softmax_row_tiling 2048 512 8 8
echo "==== COARSE-REDUCTION DONE in ${SECONDS}s ($(date)) -- fold; confirm bw(cores,tiled) + the tiled derate under reps before any fused-reduction bw term ====" | tee -a "$LOG"
