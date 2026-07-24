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
# GAMMA-BINDING SWEEP -- the deciding experiment for the cat-3 overlap fraction.
#
# WHY: the 2026-07-24 adversarial panel found gamma (the double-buffer window
# fraction in T = compute + read + write + turn - min(read, gamma*compute)) is
# UNIDENTIFIABLE on all the existing clean data. Every clean low-core point is
# in the SATURATED regime read < gamma*compute, where min(read, gamma*compute)
# = read for ANY gamma >= ~0.2 -- so the data proves "reads hide" but cannot pin
# gamma. gamma only BINDS where read ~= gamma*compute, and there the existing
# data is exactly the noisiest cohort (16/32-core, CV median 0.55), so ~0.6 is a
# central value, not a measured constant. This sweep lands points ON BOTH SIDES
# of the transition with enough reps to beat that CV.
#
# HOW (verified against the cost model): per-core read/compute
#   = (dtype * cores * peak / bw) * (1/M + 1/N)
# is INDEPENDENT of K, so we cross the transition by sweeping small M=N at high
# cores (NOT by sweeping K, which the old MMISO_CORE sweep did -- leaving it all
# saturated). At cores=32, split 4x8, read/compute goes 512->1.73, 1024->0.87,
# 1536->0.58, 2048->0.43, 2560->0.35 -- straddling gamma~0.6. Shapes are chosen
# so BOTH factors are stick-aligned (M/m, N/n multiples of 64) and pt_eff=1
# throughout (rpc=M/m in 128..640), so overlap is isolated from array-fill.
#
# High reps (default 25) so the per-config CV (~0.55 in the binding cohort)
# averages down (SE ~ 0.55/sqrt(25) ~ 0.11). Two independent crossings (cores 16
# and 32) constrain a single gamma; a couple K=4096 points confirm K-independence.
#
# ANALYSIS (offline, no HW): fold with parse_sweep_logs.py, then for each point
# compute read/compute from the feats and fit the single gamma that best places
# the min(read, gamma*compute) switch across the M=N sweep (notes/analyze_matmul_overlap.py).
# Report the RMS-vs-gamma valley WIDTH -- if it is still flat over 0.4-0.7 here
# (unsaturated, low-CV), gamma genuinely is only a central value; if it sharpens,
# gamma is pinned. Either outcome settles the cat-3 downgrade honestly.
#
#   bash docs/source/user_guide/examples/run_gamma_bind_sweep.sh
#   BENCH_REPS=40 bash .../run_gamma_bind_sweep.sh          # even tighter
# ~20 runs x 25 reps; the profiled region is a small fraction of each run.
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/gamma_bind_$(date +%Y%m%d_%H%M%S).log"
export BENCH_REPS="${BENCH_REPS:-25}"
echo "==== GAMMA-BIND $(date)  git:$(git rev-parse --short HEAD 2>/dev/null)  reps:$BENCH_REPS ====" | tee "$LOG"

_emit() {
  local all out; all=$(cat)
  out=$(printf '%s\n' "$all" | grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY|^TIMING ')
  if printf '%s\n' "$out" | grep -q '^SUMMARY'; then printf '%s\n' "$out" | tee -a "$LOG"
  else { echo "SUMMARY $1 FAILED"; printf '%s\n' "$all" | grep -vE '^\s*$' | tail -6 | sed 's/^/FAILDIAG /'; } | tee -a "$LOG"; fi
}
runmmwd() {  # M K N m n k
  local M=$1 K=$2 N=$3 m=$4 n=$5 k=$6 cores=$(( $4*$5*$6 )) _t0=$SECONDS
  (( M%m==0 && N%n==0 && K%k==0 )) || { echo "SKIP indivisible $*" | tee -a "$LOG"; return 0; }
  (( (M/m)%64==0 && (N/n)%64==0 )) || { echo "SKIP not stick-aligned $*" | tee -a "$LOG"; return 0; }
  echo "-- mmwd M=$M K=$K N=$N split m=$m n=$n k=$k (cores=$cores)" | tee -a "$LOG"
  SENCORES=32 WD_M="$m" WD_N="$n" WD_K="$k" SPYRE_DUMP_COST=1 \
    BENCH_OP=mmwd BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
    timeout -k 20 "${RUN_TIMEOUT:-180}" python "$PROFILE_OPS" 2>&1 \
    | _emit "mmwd M=$M K=$K N=$N split m=$m n=$n k=$k"
  echo "TIMING_RUN mmwd $M/$K/$N $m/$n/$k $((SECONDS-_t0))s" | tee -a "$LOG"
}

# preflight
PF=$(SENCORES=32 BENCH_OP=neg BENCH_ROWS=64 BENCH_COLS=64 BENCH_REPS=1 BENCH_WARMUP=1 \
  timeout -k 10 150 python "$PROFILE_OPS" 2>&1)
printf '%s\n' "$PF" | grep -q '^SUMMARY .*kernel_us=' \
  || { echo "## PREFLIGHT FAILED -- device busy/wedged; ABORTING." | tee -a "$LOG"; \
       printf '%s\n' "$PF" | tail -6 | sed 's/^/PREFLIGHT /' | tee -a "$LOG"; exit 1; }
# drift control once up front
BENCH_OP=neg BENCH_ROWS=2048 BENCH_COLS=2048 SENCORES=32 \
  timeout -k 20 150 python "$PROFILE_OPS" 2>&1 | _emit "CONTROL neg 2048x2048"

echo "## cores=32 (split 4x8): M=N sweep crossing read/compute 1.73 -> 0.35" | tee -a "$LOG"
for MN in 512 1024 1536 2048 2560; do runmmwd "$MN" 2048 "$MN" 4 8 1; done
echo "## cores=16 (split 4x4): M=N sweep crossing read/compute 0.87 -> 0.43" | tee -a "$LOG"
for MN in 512 768 1024; do runmmwd "$MN" 2048 "$MN" 4 4 1; done
echo "## K-independence cross-check (K=4096 at two M=N, cores=32)" | tee -a "$LOG"
for MN in 1024 2048; do runmmwd "$MN" 4096 "$MN" 4 8 1; done

echo "==== GAMMA-BIND DONE in ${SECONDS}s ($(date)) -- fold with parse_sweep_logs.py + analyze_matmul_overlap.py ====" | tee -a "$LOG"
