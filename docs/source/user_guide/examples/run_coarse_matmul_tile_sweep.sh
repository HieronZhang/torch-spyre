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
# COARSE-MATMUL TILE SWEEP -- the deciding experiment for cat-6 coarse-tiled
# matmul (matmul_row_tiling / mm_nested_m_k / matmul_k_tiling over tile count).
#
# WHY THIS SWEEP EXISTS (verdict from the clean CTFILL reps=7 slice, log
# mm_family_20260724_082545.log, recomputed against the CURRENT cost_model.py):
# the tiled-matmul residual is DEFERRED, not fittable, because a single
# monotonic per-tile-underfill term is REFUTED by the existing 7 thin points:
#
#   matmul_row_tiling, current-model err (tiles=1 = untiled gold-analog):
#     R2048 C2048: t1 +2.1%   t4 -5.2%   t8 -39.3%   (U-shaped; t4 FASTEST)
#     R8192 C2048: t1 -1.9%   t2 -14.4%  t8  -1.0%   (t2 SLOW, t8 FINE -- inverted)
#     R4096 C4096: t8 -18.6%
#   Implied compute-slowdown (meas/pred) vs per-core rows-per-tile (tile_rpc):
#     tile_rpc=64:  1.647 (R2048)  BUT 1.228 (R4096)  -> SAME rpc, 2 slowdowns
#     tile_rpc=128: 1.055 (R2048)  and 1.010 (R8192)  -> ~1
#     tile_rpc=512: 1.169 (R8192)  -> a LARGE (well-filled) tile, yet 17% slow
#   => derate is NOT a function of tile_rpc (rpc=64 gives both 1.65 and 1.23),
#      and it is ANTI-monotonic in tile_rpc within R8192 (rpc=512 slower than
#      rpc=128). So `underfill_eff(tile_rpc)` on the compute term CANNOT fit
#      this without curve-fitting, and any whole-category derate would move the
#      well-predicted untiled points (tiles=1 RMS 4.7%, the gold analog).
#
# TWO ENTANGLED, OPPOSITE-SIGN effects, both invisible to the model (io_hbm is
# CONSTANT across tiles and pt_eff is pinned to 1 for tiled matmul):
#   (A) LX-RESIDENCY SPEEDUP: as tiles grow, per-tile operand/intermediate
#       working set fits in LX and re-reads are served on-chip -> FASTER
#       (t1->t4 at R2048; t1->t8 at R8192). Not a byte effect (bytes constant).
#   (B) PER-TILE UNDERFILL / FILL-DRAIN: past some point each coarse tile is too
#       short to fill the PT pipeline, so fill/drain overhead per tile dominates
#       -> SLOWER (t4->t8 at R2048). The crossover (best tile count) is
#       shape-dependent (t4 best at R2048, t8 best at R8192) and NOT captured by
#       tile_rpc alone -- the same rpc lands on different sides of the U for
#       different (M,N,K).
# On ~7 non-monotonic points these two cannot be separated. This sweep isolates
# them so a SINGLE mechanistic term can be fit (or the U confirmed structural).
#
# DESIGN (all reps=7 noise protocol, cores=32, no forced split -> the coarse
# LOOP is the only variable):
#   1. DENSE tile ladder at FIXED shape: t = 1,2,3,4,6,8,12,16 -> resolve the U
#      (find the minimum per shape; is it one U or two regimes?).
#   2. tile_rpc HELD FIXED across shapes: scale M with tiles so tile_rpc is
#      constant (e.g. tile_rpc=64 at {M2048,t8},{M4096,t16}? -- M/(32*tiles)),
#      isolating whether the derate depends on tile_rpc or on tiles/shape.
#   3. VARY N and K independently at fixed M,tiles -> does the underfill couple
#      to the N (cols-per-core) / K (reduction depth) of the tile, not just M?
#   4. IR dump per tile count (loop_count / CoarseTileInfo / op_it_space_splits)
#      to confirm the loop structure and that io_hbm is genuinely tile-invariant.
#
# After fold (parse_sweep_logs.py) recompute err with the CURRENT model and:
#   - locate the U minimum per shape; check if `tiles_at_min ~= f(M/64)` (LX
#     capacity) -> mechanistic (A);
#   - regress the RISING leg (past the min) on tile_rpc AND on tiles: does a
#     fill/drain `+ loop_trip * c_fill(tile_geometry)` fit BOTH legs' shapes?
#   - ONLY ship if one form fits all shapes AND leaves tiles=1 (gold) within
#     the 0.6% cv noise (mac_peak/spill untouched). Else keep DEFER.
#
#   bash docs/source/user_guide/examples/run_coarse_matmul_tile_sweep.sh
# ~90 timing runs (<1 min each) + a few IR dumps. Budget-guarded by MAX_SECONDS.
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs haoyang_logs/ir
LOG="haoyang_logs/coarse_mm_tile_$(date +%Y%m%d_%H%M%S).log"
export BENCH_REPS="${BENCH_REPS:-7}"
MAX_SECONDS="${MAX_SECONDS:-9000}"
_START=$SECONDS
echo "==== COARSE-MM-TILE $(date)  git:$(git rev-parse --short HEAD 2>/dev/null)  reps:$BENCH_REPS ====" | tee "$LOG"

_budget() { [[ $((SECONDS-_START)) -ge $MAX_SECONDS ]] && { echo "## BUDGET $MAX_SECONDS s reached -- stopping." | tee -a "$LOG"; return 1; }; return 0; }

_emit() {
  local all out; all=$(cat)
  out=$(printf '%s\n' "$all" | grep -E 'op_it_space_splits|loop_count|CoarseTile|^IO |^MODEL |^SUMMARY|^TIMING ')
  if printf '%s\n' "$out" | grep -q '^SUMMARY'; then printf '%s\n' "$out" | tee -a "$LOG"
  else { echo "SUMMARY $1 FAILED"; printf '%s\n' "$all" | grep -vE '^\s*$' | tail -6 | sed 's/^/FAILDIAG /'; } | tee -a "$LOG"; fi
}

# op M(=ROWS) K(=COLS) N tiles [DUMP_IR]
runmm() {
  _budget || return 1
  local op=$1 M=$2 K=$3 N=$4 t=$5 dump=${6:-0} _t0=$SECONDS
  local ir="haoyang_logs/ir/coarsemm_${op}_${M}x${K}x${N}_t${t}.txt"
  echo "-- $op M=$M K=$K N=$N tiles=$t" | tee -a "$LOG"
  SENCORES=32 SPYRE_DUMP_IR="$dump" SPYRE_DUMP_COST=1 \
    BENCH_OP="$op" BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" BENCH_TILES="$t" \
    timeout -k 20 "${RUN_TIMEOUT:-240}" python "$PROFILE_OPS" 2>&1 \
    | { [[ "$dump" == 1 ]] && tee "$ir" || cat; } \
    | _emit "$op M=$M K=$K N=$N tiles=$t"
  echo "TIMING_RUN $op $M/$K/$N t$t $((SECONDS-_t0))s" | tee -a "$LOG"
}

PF=$(SENCORES=32 BENCH_OP=neg BENCH_ROWS=64 BENCH_COLS=64 BENCH_REPS=1 BENCH_WARMUP=1 \
  timeout -k 10 150 python "$PROFILE_OPS" 2>&1)
printf '%s\n' "$PF" | grep -q '^SUMMARY .*kernel_us=' \
  || { echo "## PREFLIGHT FAILED -- ABORTING." | tee -a "$LOG"; printf '%s\n' "$PF" | tail -6 | sed 's/^/PREFLIGHT /' | tee -a "$LOG"; exit 1; }

# ---------------------------------------------------------------------------
# (1) DENSE tile ladder at FIXED shape -> resolve the U-shape per shape.
#     Square (M=N=K=2048), tall-M (8192x2048), wide-N (2048x4096), big square
#     (4096x4096). t=16 sometimes FAILs (>32 divisibility); harness logs FAILED.
echo "## (1) dense tile ladder t={1,2,3,4,6,8,12,16} per shape (find the U min)" | tee -a "$LOG"
for shape in "2048 2048 2048" "8192 2048 2048" "2048 2048 4096" "4096 4096 4096"; do
  set -- $shape
  for t in 1 2 3 4 6 8 12 16; do runmm matmul_row_tiling "$1" "$2" "$3" "$t" || break; done
done

# ---------------------------------------------------------------------------
# (2) tile_rpc HELD FIXED (=M/(32*tiles)) across shapes -> is the derate a
#     function of tile_rpc, or of tiles/shape? Pairs at tile_rpc={64,128,256}.
#       tile_rpc=64 : (M2048,t1?no) use (M4096,t2)&(M8192,t4)&(M16384,t8)  M/(32t)=64
#       tile_rpc=128: (M4096,t1),(M8192,t2),(M16384,t4)
#       tile_rpc=256: (M8192,t1),(M16384,t2)
#     Same tile_rpc, different (M,tiles): if err is constant along a row -> the
#     derate IS tile_rpc; if it varies -> tiles/shape drives it (as the thin
#     data suggests).  N=K=2048 fixed.
echo "## (2) tile_rpc held fixed across (M,tiles): 64/128/256 rows" | tee -a "$LOG"
runmm matmul_row_tiling 4096  2048 2048 2   # rpc 64
runmm matmul_row_tiling 8192  2048 2048 4   # rpc 64
runmm matmul_row_tiling 16384 2048 2048 8   # rpc 64
runmm matmul_row_tiling 4096  2048 2048 1   # rpc 128
runmm matmul_row_tiling 8192  2048 2048 2   # rpc 128
runmm matmul_row_tiling 16384 2048 2048 4   # rpc 128
runmm matmul_row_tiling 8192  2048 2048 1   # rpc 256
runmm matmul_row_tiling 16384 2048 2048 2   # rpc 256

# ---------------------------------------------------------------------------
# (3) VARY N and K at fixed M,tiles -> does the underfill couple to N
#     (cols-per-core) or K (reduction depth) of the tile, not only M?
#     M=4096,t8 (tile_rpc=16, deep underfill) sweep N then K.
echo "## (3) N-sweep and K-sweep at fixed M=4096 tiles=8 (couple to N? to K?)" | tee -a "$LOG"
for N in 1024 2048 4096 8192; do runmm matmul_row_tiling 4096 2048 "$N" 8; done
for K in 1024 2048 4096 8192; do runmm matmul_row_tiling 4096 "$K" 2048 8; done

# ---------------------------------------------------------------------------
# (4) the OTHER two coarse-matmul ops over tiles (mm_nested tiles K inner;
#     matmul_k_tiling tiles the reduction dim -> io_hbm RISES, a different
#     sub-problem). Confirm whether the same U appears or the K-tiled io_hbm
#     rise already explains them.
echo "## (4) mm_nested_m_k & matmul_k_tiling tile ladders" | tee -a "$LOG"
for shape in "2048 2048 2048" "4096 2048 2048"; do
  set -- $shape
  for t in 1 2 4 8; do runmm mm_nested_m_k "$1" "$2" "$3" "$t" || break; done
  for t in 1 2 4 8; do runmm matmul_k_tiling "$1" "$2" "$3" "$t" || break; done
done

# ---------------------------------------------------------------------------
# (5) IR dumps: confirm loop_count/CoarseTileInfo per tile count and that
#     io_hbm is genuinely tile-invariant for matmul_row_tiling.
echo "## (5) IR dumps (loop_count / CoarseTileInfo / op_it_space_splits per tiles)" | tee -a "$LOG"
for t in 1 4 8; do runmm matmul_row_tiling 2048 2048 2048 "$t" 1; done

echo "==== COARSE-MM-TILE DONE in $((SECONDS-_START))s ($(date)) -- fold, recompute err vs CURRENT model; locate the U min per shape; test tile_rpc-only vs tiles/shape (section 2); ONLY ship a coarse-matmul term if ONE form fits BOTH legs across shapes AND leaves tiles=1 within cv noise, else keep DEFER ====" | tee -a "$LOG"
