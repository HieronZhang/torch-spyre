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
# WHAT SETS THE FUSED-CHAIN FLOOR: ELEMENT COUNT, OR LX TRAFFIC?
#
# §19 of notes/cost_model_report.md charges a floor of
#
#     time >= elements / (cores * 1.51 per ns per core)
#
# keyed on the ELEMENT COUNT and independent of how many stages the fused chain
# has. That form fits the calibration data. So does a completely different story:
# LX traffic is not free, and the chain is limited by per-core LX bandwidth.
#
# THE TWO ARE CONFOUNDED IN EVERY RUN WE HOLD. On the calibration sweep the
# whole-loop LX traffic is 2.10 MB at 1, 4 and 8 tiles -- exactly as invariant as
# the element count -- so both predict the same flat time; and LX is per-core, so
# both scale as 1/cores. Since LX traffic is itself proportional to the element
# count, they even share a functional form. Nothing in the database separates them:
# the one cell that varies the stage count sits at 32 cores, where the floor never
# binds, and its longest arm is a single un-repeated run.
#
# WHAT THIS SWEEP DOES. It varies the number of fused stages while holding the
# element count and the HBM traffic fixed. Each extra stage reads from LX and
# writes to LX, so LX traffic grows linearly with BENCH_STAGES and nothing else
# moves. Run at LOW cores, where the floor actually binds.
#
#     time FLAT in stages    -> the element-only floor is right as written
#     time GROWS with stages -> the floor is MIS-KEYED and must scale with the chain
#                               length; the fitted 1.51 has a 5-stage chain baked in.
#                               This does NOT by itself name the mechanism: LX traffic
#                               and per-element work through more stages are both
#                               proportional to elements*stages.
#
# READ THE CONTROL FIRST. Every row prints its HBM bytes. If they move with
# BENCH_STAGES, the intermediates have spilled to HBM and the comparison is
# confounded -- report that instead of the timing.
#
# COST: 24 configs, a few seconds each. This is deliberately small.
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")"
OUT="${OUT:-stage_sweep_$(date +%Y%m%d_%H%M%S).log}"
# SHAPE AND TILING ARE LOAD-BEARING. The intermediates only live in LX if the chain is
# coarse-tiled AND the per-tile buffer is small enough that a LONGER chain still fits.
# 2048x512 at 8 tiles is 0.26 MB per buffer against a ~1.6 MB/core budget, so even a
# 13-stage chain stays resident. The first attempt at 4096x2048 UNTILED spilled every
# stage to HBM (+33.55 MB each) and failed its own control.
ROWS="${ROWS:-2048}"
COLS="${COLS:-512}"
TILES="${TILES:-8}"

echo "writing to $OUT"
{
  echo "# stage sweep: does the fused-chain floor scale with LX traffic?"
  echo "# shape ${ROWS}x${COLS}, ${TILES} tiles, sigmoid stages between the two reductions"
  echo "# CONTROL: HBM bytes must NOT move with BENCH_STAGES"
} | tee "$OUT"

# cores 1 and 2: the floor binds hardest, and two points test the 1/cores claim.
# cores 32 included as the null -- the floor never binds there, so any stage
# dependence seen at 32 is real work, not the floor.
for CORES in 1 2 32; do
  for STAGES in 0 2 4 8; do
    echo "=== cores=$CORES stages=$STAGES ===" | tee -a "$OUT"
    SENCORES="$CORES" \
    BENCH_OP=softmax_stages \
    BENCH_ROWS="$ROWS" BENCH_COLS="$COLS" \
    BENCH_TILES="$TILES" \
    BENCH_STAGES="$STAGES" \
    BENCH_REPS="${REPS:-7}" \
    SPYRE_DUMP_COST=1 \
      python3 profile_ops.py 2>&1 | tee -a "$OUT"
  done
done

echo
echo "done -> $OUT"
echo
echo "To read it:  python3 ../../../../notes/analyze_stage_sweep.py $OUT"
