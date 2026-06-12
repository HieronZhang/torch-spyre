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
# Runs the cost-model calibration plan (see notes/cost_model_design.md sec 8) and
# collects every step's output into ONE log file you can forward back.
#
#   bash examples/run_cost_model_plan.sh
#
# Output: stdout (live) AND cost_model_results_<timestamp>.log (forward this file).
# Each run prints: cost-model PREDICTION (SPYRE_DUMP_COST, stderr) + measured
# per-kernel DEVICE latency table (SPYRE_PROFILE_SYNC).

set -u
cd "$(dirname "$0")/.." || exit 1

LOG="cost_model_results_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

# Common environment for every run.
export SPYRE_PROFILE=1            # record per-kernel times
export SPYRE_PROFILE_SYNC=1       # -> per-kernel DEVICE latency (deterministic min)
export SPYRE_DUMP_COST=1          # print cost-model prediction at compile
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1   # recompile each run so dumps fire
export BENCH_RUNS="${BENCH_RUNS:-100}"
export BENCH_WARMUP="${BENCH_WARMUP:-20}"

echo "================ cost-model calibration run ================"
echo "date:    $(date)"
echo "host:    $(hostname)"
echo "git:     $(git rev-parse --short HEAD 2>/dev/null) $(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
echo "python:  $(python -c 'import sys,torch; print(sys.version.split()[0], "torch", torch.__version__)' 2>/dev/null)"
echo "runs=$BENCH_RUNS warmup=$BENCH_WARMUP  SENCORES=${SENCORES:-default(32)}"
echo "log:     $LOG"
echo "==========================================================="

step() {
  local label="$1"; shift
  echo ""
  echo "########################################################################"
  echo "## $label"
  echo "## + $*"
  echo "########################################################################"
  ( "$@" ) || echo "[STEP FAILED: $label]"
}

# ---------------------------------------------------------------------------
# RUNG 1 — fit fill + BW_HBM: single gelu, size sweep (ROWS=512, COLS varies).
#   Expect: device min linear in elements. slope -> BW_HBM, intercept -> fill.
# ---------------------------------------------------------------------------
for n in 512 1024 2048 4096 8192; do
  step "RUNG1 size sweep: gelu[512x${n}]" \
    env BENCH_OP=gelu BENCH_ROWS=512 BENCH_COLS="$n" python examples/bench_ops.py
done

# ---------------------------------------------------------------------------
# RUNG 2 — arithmetic free? relu vs gelu, same size.
#   Expect: ~equal if memory-bound; gelu slower => activations add a compute term.
# ---------------------------------------------------------------------------
for op in relu gelu sigmoid exp; do
  step "RUNG2 arithmetic: ${op}[512x1024]" \
    env BENCH_OP="$op" BENCH_ROWS=512 BENCH_COLS=1024 python examples/bench_ops.py
done

# ---------------------------------------------------------------------------
# RUNG 3 — traffic counting: 1-input (gelu) vs 2-input (mul/add).
#   Expect: mul/add ~1.5x gelu (3 memory passes vs 2).
# ---------------------------------------------------------------------------
for op in gelu mul add; do
  step "RUNG3 traffic: ${op}[512x1024]" \
    env BENCH_OP="$op" BENCH_ROWS=512 BENCH_COLS=1024 python examples/bench_ops.py
done

# ---------------------------------------------------------------------------
# RUNG 4 — BW_LX: unary gelu chain, ALL intermediates forced into LX.
#   Expect: HBM traffic fixed (read x + write y); latency slope vs depth =
#   2*|x_tile|/BW_LX. First confirm intermediates actually got 'lx'.
# ---------------------------------------------------------------------------
echo ""
echo "## RUNG4 LX-placement check (depth=4): expect 'lx' allocations on intermediates"
env SPYRE_DUMP_IR=1 SPYRE_DUMP_IR_FILE=/tmp/lxchk_ir.txt \
    BENCH_LX_ALL=1 BENCH_DEPTH=4 BENCH_ROWS=512 BENCH_COLS=1024 \
    python examples/bench_ops.py || echo "[depth-4 dump run failed]"
echo "-- intermediate allocations in the depth-4 chain IR --"
grep -cE "'lx'" /tmp/lxchk_ir.txt 2>/dev/null && echo "(count of lx allocations above)"
grep -E "allocation=" /tmp/lxchk_ir.txt 2>/dev/null | head -20

for d in 1 2 4 8 16; do
  step "RUNG4 LX chain: gelu depth=${d} (all-LX)" \
    env BENCH_OP=gelu BENCH_LX_ALL=1 BENCH_DEPTH="$d" BENCH_ROWS=512 BENCH_COLS=1024 \
    python examples/bench_ops.py
done

# ---------------------------------------------------------------------------
# RUNG 5 — shared vs per-core HBM BW: gelu[512x1024], sweep SENCORES.
#   Expect: latency ∝ 1/cores => per-core BW; flat => shared HBM BW.
# ---------------------------------------------------------------------------
for c in 1 2 4 8 16 32; do
  step "RUNG5 SENCORES=${c}: gelu[512x1024]" \
    env SENCORES="$c" BENCH_OP=gelu BENCH_ROWS=512 BENCH_COLS=1024 \
    python examples/bench_ops.py
done

echo ""
echo "================ DONE. forward this file: $LOG ================"
