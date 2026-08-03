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
# COARSE RE-READ LADDER -- break the (L-1) aliasing that blocks cat 6.
#
# WHAT IS ALREADY PROVEN (from IR, not from timing). In the coarse-tiled matmul
# haoyang_logs/ir/coarsemm_matmul_row_tiling_2048x2048x2048_t4.txt:
#
#     i0, i1 = index                                # ranges=[2048, 2048]
#     tmp0 = ops.load(arg0_1, r0_0 + 2048 * i0)     # A[M,K] -- index CONTAINS i0
#     tmp1 = ops.load(arg1_1, i1   + 2048 * r0_0)   # B[K,N] -- index has NO i0
#
# with loop_tiled_dims=[[0]] and DimHint(dim_names=['M'], loop_var=d0), so the tiled
# symbol is i0. A advances with the loop; B is LOOP-INVARIANT and re-entered every
# iteration. The extractor charges B ONCE (loop_factor is computed as two PER-OP
# scalars instead of per arg), so the model is FLAT in L: at M=4096 K=N=2048 cores=32
# it predicts 684.6 us at L=4, 8 AND 16 while measurement goes 677.8 -> 874.3 -> 1306.8.
#
# WHY THIS SWEEP IS NEEDED ANYWAY. The COUNT is settled; the COST is not. Three
# different effects all scale as (L-1) and are perfectly aliased on the data we have:
#
#   (a) B re-read        extra cost ~ (L-1) * K*N * dtype / BW      -- scales with B
#   (b) fixed per-iter   extra cost ~ (L-1) * c                     -- independent of B
#   (c) per-tile underfill                                          -- keys on rows/core
#
# Every one of the 77 recorded matmul_row_tiling rows has K = N = 2048, i.e. exactly ONE
# B size (8.0 MB), spread over only TWO (M,K,N,cores) cells. So B never varies and (a)
# cannot be separated from (b). Fitting a coefficient on that data would be fitting the
# aliasing. Two knobs separate all three:
#
#   N     changes B = K*N  WITHOUT touching A = M*K   -> isolates (a)
#   cores changes rows/core WITHOUT touching B         -> isolates (c)
#
# THE DISCRIMINATOR. For each cell compute the per-iteration slope
#       s = (T(L) - T(1)) / (L - 1)
# and plot s against N:
#   s grows in proportion to N   -> (a) dominates: charge the re-read at an HBM rate.
#   s flat in N                  -> (b) dominates: a fixed per-iteration cost; the
#                                   re-read is served on-chip and must NOT be charged.
#   s grows but sub-linearly     -> partial residency; the ratio s/(B/BW) measures the
#                                   fraction of each re-read that actually reaches HBM,
#                                   and THAT is the number to put in the model.
# The cores axis then says how much of any residual is really (c).
#
# NOTE the ladder is U-SHAPED -- at M=4096 tiling is 19 % FASTER at L=4 than at L=1
# before turning over. The model already reproduces the down-slope; only the up-slope is
# missing. So read the SLOPE at L >= 4, and keep L=1 only as the anchor.
#
# COST: 4 N-values x 4 L-values x 2 core counts = 32 runs. cores stays >= 8 (standing
# scope rule). reps=5 so every point is repeat-backed and cv is reported -- the L=2/L=4
# dip is small enough that single-shot points cannot resolve it.
#
#   bash docs/source/user_guide/examples/run_coarse_reread_ladder.sh
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/coarse_reread_$(date +%Y%m%d_%H%M%S).log"

M=${M:-4096}
K=${K:-2048}
REPS=${BENCH_REPS:-5}

echo "==== COARSE RE-READ LADDER  $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD)   M=$M K=$K reps=$REPS" | tee -a "$LOG"
echo "fixed M,K; N varies B=K*N; cores varies rows/core; L is the loop trip" | tee -a "$LOG"

for cores in 32 16; do
  for N in 512 1024 2048 4096; do
    bmb=$(python3 -c "print(f'{$K*$N*2/1048576:.1f}')")
    echo "" | tee -a "$LOG"
    echo "---- cores=$cores N=$N  (B = ${bmb} MB) ----" | tee -a "$LOG"
    for L in 1 4 8 16; do
      out=$(SENCORES="$cores" BENCH_OP=matmul_row_tiling \
            BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
            BENCH_TILES="$L" BENCH_REPS="$REPS" \
            timeout -k 30 "${RUN_TIMEOUT:-600}" python "$PROFILE_OPS" 2>&1)
      line=$(printf '%s\n' "$out" | grep -E '^SUMMARY' | head -1)
      if [ -z "$line" ]; then
        echo "  L=$L  FAILED" | tee -a "$LOG"
        printf '%s\n' "$out" | tail -3 | sed 's/^/     /' | tee -a "$LOG"
      else
        echo "  L=$L  $line" | tee -a "$LOG"
      fi
    done
  done
done

{
  echo ""
  echo "==== HOW TO READ IT ===="
  echo "  For each (cores, N) cell compute s = (T(L) - T(1)) / (L - 1) at L = 8 and 16."
  echo "  Then compare s ACROSS N at fixed cores:"
  echo "    s roughly doubles when N doubles -> the B re-read is real and HBM-priced."
  echo "    s barely moves when N doubles    -> it is a fixed per-iteration cost; do NOT"
  echo "                                        add re-read bytes, add a per-iteration term."
  echo "    s grows sub-linearly             -> partial residency; report s / (B/BW) as the"
  echo "                                        measured HBM fraction of each re-read."
  echo "  Compare s ACROSS cores at fixed N to attribute any residual to per-tile underfill."
  echo "  Fold with: python3 notes/parse_sweep_logs.py, then re-run"
  echo "  python3 notes/test_loop_invariant_reread.py to re-score with real B variation."
} | tee -a "$LOG"
