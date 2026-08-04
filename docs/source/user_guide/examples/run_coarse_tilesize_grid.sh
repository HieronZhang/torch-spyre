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
# COARSE TILE-SIZE GRID -- separate TILE COUNT (t) from TILE SIZE (rows/core).
#
# WHY. With the per-arg `loop_factor` fixed and the re-read term live,
# `matmul_row_tiling` sits at 16.8 % RMS / +11.2 % mean: the model is systematically
# TOO SLOW once tiling starts, i.e. the hardware gains something from small tiles that
# the model does not have. The residual correlates with tile size (|r| ~ 0.75 against
# log2 rows/core, log2 working set, and log2 t) -- but those three are ALIASED, because
# EVERY row measured so far has M = 4096, which pins
#
#       rows/core = M / (t * row_split)
#
# so t and rows/core move together and cannot be told apart. Worse, at nominally
# MATCHED conditions the residual does not even agree: at t=4, rows/core=128 the two
# points read +4.5 % and +17.4 % (12.9 pts apart), and at t=8, rows/core=64 they read
# +28.4 % and +17.1 %. Fitting a term to that would be fitting the aliasing -- the same
# trap the (L-1) re-read aliasing set earlier, which cost a wrong alpha=0.5.
#
# THE MISSING AXIS IS M. Varying M at FIXED t changes rows/core WITHOUT changing the
# tile count, which is the only way to separate them:
#
#       fixed t, M doubles  -> rows/core doubles, t unchanged
#       fixed M, t doubles  -> rows/core halves,  t doubled
#
# A 2-D grid over (M, t) therefore identifies which one the residual actually follows.
#
# WHAT THE ANSWER LOOKS LIKE
#   residual flat along M at fixed t      -> it is a TILE-COUNT effect (per-iteration
#                                            overhead); model it on t.
#   residual flat along t at fixed rows/core -> it is a TILE-SIZE effect (on-chip
#                                            residency); model it on rows/core or on the
#                                            per-core working set, which is what the
#                                            currently matmul-gated `_lx_spill_bw_derate`
#                                            was built for.
#   neither flat                          -> a genuine interaction; report it and do NOT
#                                            fit a 1-D term.
#
# N is held at 2048 and cores at 32 so the only things moving are M and t. The t=1 row
# of each M is the untiled ANCHOR: the residual there is the pre-existing plain-matmul
# error and must be subtracted before attributing anything to tiling.
#
# All M and t are powers of two: `coarse_tile` requires M to be evenly divisible by t
# (an earlier sweep lost 8 runs to `range 4096 is not divisible by loop_count 3`).
#
# LOG FORMAT IS LOAD-BEARING: `parse_sweep_logs.py` needs a `git: <sha>` line,
# `## <section>`, `-- <label>`, and SUMMARY at COLUMN 0. SPYRE_DUMP_COST=1 is REQUIRED
# or the rows carry no `feats` and cannot be scored against a changed model.
#
# COST: 4 M-values x 4 t-values = 16 runs, reps=5. M=16384 is ~4x the work of M=4096,
# so budget accordingly.
#
#   bash docs/source/user_guide/examples/run_coarse_tilesize_grid.sh
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/coarse_tilesize_$(date +%Y%m%d_%H%M%S).log"

K=${K:-2048}
N=${N:-2048}
REPS=${BENCH_REPS:-5}
CORES=${SENCORES:-32}
MVALS=${MVALS:-"2048 4096 8192 16384"}
TVALS=${TVALS:-"1 4 8 16"}

echo "==== COARSE TILE-SIZE GRID  $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD)" | tee -a "$LOG"
echo "# K=$K N=$N cores=$CORES reps=$REPS ; M x t grid separates tile size from tile count" \
  | tee -a "$LOG"
echo "## coarse_tilesize" | tee -a "$LOG"

for M in $MVALS; do
  echo "# --- M=$M (rows/core scales with M at fixed t) ---" | tee -a "$LOG"
  for t in $TVALS; do
    out=$(SENCORES="$CORES" SPYRE_DUMP_COST=1 BENCH_OP=matmul_row_tiling \
          BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
          BENCH_TILES="$t" BENCH_REPS="$REPS" \
          timeout -k 30 "${RUN_TIMEOUT:-900}" python "$PROFILE_OPS" 2>&1)
    line=$(printf '%s\n' "$out" | grep -E '^SUMMARY' | head -1)
    echo "-- matmul_row_tiling ${M}x${K}x${N} t=$t cores=$CORES" | tee -a "$LOG"
    if [ -z "$line" ]; then
      echo "#   FAILED" | tee -a "$LOG"
      printf '%s\n' "$out" | tail -3 | sed 's/^/#     /' | tee -a "$LOG"
    else
      printf '%s\n' "$out" | grep -E '^(IO |MODEL |op_it_space_splits)' | tee -a "$LOG"
      printf '%s\n' "$line" | tee -a "$LOG"
    fi
  done
done

{
  echo ""
  echo "==== HOW TO READ IT ===="
  echo "  Fold ONLY this log:  python3 notes/parse_sweep_logs.py $LOG"
  echo "  (never re-parse haoyang_logs/* -- 10 curated logs are gone from disk and 189"
  echo "   records survive only inside sweep_records.json)"
  echo ""
  echo "  Build the residual grid, then read it two ways:"
  echo "    ACROSS a row (fixed M, t varies)   -> tile count varies, rows/core varies"
  echo "    DOWN  a column (fixed t, M varies) -> rows/core varies, tile count FIXED"
  echo "  The column direction is the new information: it is the only place rows/core"
  echo "  moves with t held still. Subtract each M's t=1 anchor first."
  echo ""
  echo "  If the residual tracks rows/core, the fix belongs in the coarse residency"
  echo "  term (_lx_spill_bw_derate, currently gated OFF for matmul) rather than in a"
  echo "  new per-iteration term."
} | tee -a "$LOG"
