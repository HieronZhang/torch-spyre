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
# OUTLIER CLOSE-OUT SWEEP -- every remaining measurement the cat 1-6 attack needs.
#
# WHY THESE RUNS. Scoring the CURRENT model over all normal/valid inputs (cores=32,
# non-lopsided split, non-tiny) leaves 465 points >10%, of which 164 are the add-chains
# that §3 deliberately does NOT model (a multi-op program-level effect), so 301 are in
# scope. They cluster as:
#     135  coarse-tiled (cat 6): matmul_row_tiling / mm_nested_m_k / matmul_k_tiling
#                                + bmm_k_tiling / bmm_nested_b_k
#     100  bmm family (cat 4)  : fast+mixed layouts, 3d2d projection, B=2, low-core
#      35  plain matmul (cat 3): split-shape / spill tails
#      31  broadcast+transport (cats 1-2): transpose_outer, write, bcastcol, cat0/cat1
# Each section below targets one cluster and is designed to vary ONE physical quantity so
# the residual can be modelled MECHANISTICALLY rather than curve-fitted (the standing rule).
# Everything runs under the noise protocol (BENCH_REPS back-to-back profiled reps -> the
# min/median/cv the analysis needs; single-shot points are what burned us before).
#
# SECTIONS (default: all, cheapest-and-highest-value first so an early cut-off still lands
# the most useful data):
#   TAILS    cats 1-2  transport + broadcast/write tails            ~42 runs   ~0.2 h
#   BMMFULL  cat 4     layout quads x B, 3d2d, low-core, thin       ~63 runs   ~0.7 h
#   MMSPILL  cat 3     split-shape x per-core area (spill/split)    ~30 runs   ~0.3 h
#   REDCORES cat 5     plain-reduction g(cores) confirmation        ~48 runs   ~0.2 h   [existing script]
#   GAMMA    cat 3     gamma identifiability (unsaturated)          ~11 runs   ~0.1 h   [existing script]
#   COARSERED cat 5    fused-reduction bw(cores, tiles)             ~58 runs   ~0.3 h   [existing script]
#   COARSEBMM cat 6b   bmm_k_tiling / bmm_nested_b_k tile ladders   ~16 runs   ~0.5 h
#   COARSEMM cat 6a    coarse matmul tile ladder + fixed-rpc rows   ~67 runs   ~2.1 h   [existing script]
#   ADDIR    §3        add-chain IR structure (bundle counts)        ~8 runs   ~0.1 h   [existing script]
#
# ESTIMATED TOTAL ~4.5 h. That is computed from MEASURED per-run wall-clock in the existing
# logs, not guessed: mmwd 37 s, bmm_layout 50 s, bmm_wd 24 s, matmul_row_tiling 106 s,
# mm_nested_m_k 114 s, matmul_k_tiling 127 s, bmm_k_tiling 108 s, pointwise/reduction 14-18 s,
# softmax_row_tiling 14 s. The coarse-matmul sections dominate (~100-130 s/run compiles).
#
#   bash docs/source/user_guide/examples/run_outlier_closeout.sh            # everything
#   SECTIONS="TAILS BMMFULL" bash .../run_outlier_closeout.sh               # a subset
#   MAX_SECONDS=14400 bash .../run_outlier_closeout.sh                      # 4 h budget cap
#   DRY=1 bash .../run_outlier_closeout.sh                                  # print the plan
# ============================================================================

set -u
trap 'echo "## INTERRUPTED"; exit 130' INT TERM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/closeout_$(date +%Y%m%d_%H%M%S).log"
export BENCH_REPS="${BENCH_REPS:-7}"
export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
SECTIONS="${SECTIONS:-TAILS BMMFULL MMSPILL REDCORES GAMMA COARSERED COARSEBMM COARSEMM ADDIR}"
MAX_SECONDS="${MAX_SECONDS:-0}"     # 0 = unlimited
DRY="${DRY:-0}"
_START=$SECONDS
NRUN=0
echo "==== OUTLIER CLOSE-OUT $(date)  reps=$BENCH_REPS  sections: $SECTIONS ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD 2>/dev/null)" | tee -a "$LOG"
has() { [[ " $SECTIONS " == *" $1 "* ]]; }
budget_ok() { [[ "$MAX_SECONDS" == 0 ]] && return 0; (( SECONDS-_START < MAX_SECONDS )) && return 0
  echo "## BUDGET ${MAX_SECONDS}s reached -- stopping cleanly." | tee -a "$LOG"; return 1; }
sect() { echo "## $1 -- $2 ($(date +%H:%M:%S))" | tee -a "$LOG"; SECT_T0=$SECONDS; }
esect() { echo "## $1 DONE in $((SECONDS-SECT_T0))s" | tee -a "$LOG"; }

_emit() {
  local all out; all=$(cat)
  out=$(printf '%s\n' "$all" | grep -E 'op_it_space_splits|^IO |^MODEL |^SUMMARY|^TIMING ')
  if printf '%s\n' "$out" | grep -q '^SUMMARY'; then printf '%s\n' "$out" | tee -a "$LOG"
  else { echo "SUMMARY $1 FAILED"; printf '%s\n' "$all" | grep -vE '^\s*$' | tail -5 | sed 's/^/FAILDIAG /'; } | tee -a "$LOG"; fi
}
# generic runner: run <label> <ENV=V ...> -- profile_ops with those env vars
run() {
  local label="$1"; shift
  budget_ok || return 1
  NRUN=$((NRUN+1))
  if [[ "$DRY" == "1" ]]; then echo "PLAN[$NRUN] $label :: $*" | tee -a "$LOG"; return 0; fi
  local t0=$SECONDS
  echo "-- $label" | tee -a "$LOG"
  env "$@" timeout -k 30 "${RUN_TIMEOUT:-400}" python "$PROFILE_OPS" 2>&1 | _emit "$label"
  echo "TIMING_RUN ${label%% *} $label $((SECONDS-t0))s" | tee -a "$LOG"
}

# ---- preflight (skip in dry mode) ----
if [[ "$DRY" != "1" ]]; then
  PF=$(SENCORES=32 BENCH_OP=neg BENCH_ROWS=64 BENCH_COLS=64 BENCH_REPS=1 BENCH_WARMUP=1 \
    timeout -k 10 150 python "$PROFILE_OPS" 2>&1)
  printf '%s\n' "$PF" | grep -q '^SUMMARY .*kernel_us=' \
    || { echo "## PREFLIGHT FAILED -- device busy/wedged; ABORTING." | tee -a "$LOG"
         printf '%s\n' "$PF" | tail -6 | sed 's/^/PREFLIGHT /' | tee -a "$LOG"; exit 1; }
  echo "-- preflight OK" | tee -a "$LOG"
fi
export SKIP_PREFLIGHT=1

# =================== TAILS (cats 1-2): 31 outlier points ====================
# transpose_outer is the worst (13 pts, -11% mean): it rides the DEFAULT copy model with no
# access-pattern term. Sweep R x C x the outer-scatter count M so BW_eff(R,C,M) is separable.
# write/bcastcol/cat0/cat1 tails: the large-COLS corners where the fitted laws drift.
if has TAILS; then
  sect TAILS "cats 1-2: transpose_outer BW_eff(R,C,M) + write/bcast/cat large-COLS tails"
  for M in 4 8 16; do
    for rc in "2048 8192" "8192 2048" "2048 32768" "16384 2048" "4096 8192"; do
      set -- $rc
      run "transpose_outer R=$1 C=$2 M=$M" BENCH_OP=transpose_outer BENCH_ROWS="$1" BENCH_COLS="$2" TO_MID="$M" SENCORES=32 || break 2
    done
  done
  for rc in "512 16384" "2048 16384" "8192 4096" "8192 16384"; do
    set -- $rc
    run "write R=$1 C=$2"     BENCH_OP=write     BENCH_ROWS="$1" BENCH_COLS="$2" SENCORES=32 || break
    run "bcastcol R=$1 C=$2"  BENCH_OP=bcastcol  BENCH_ROWS="$1" BENCH_COLS="$2" SENCORES=32 || break
  done
  for rc in "2048 16384" "8192 8192" "16384 2048"; do
    set -- $rc
    run "cat0 R=$1 C=$2"      BENCH_OP=cat0      BENCH_ROWS="$1" BENCH_COLS="$2" SENCORES=32 || break
    run "cat1 R=$1 C=$2"      BENCH_OP=cat1      BENCH_ROWS="$1" BENCH_COLS="$2" SENCORES=32 || break
    run "mulbcast R=$1 C=$2"  BENCH_OP=mulbcast  BENCH_ROWS="$1" BENCH_COLS="$2" SENCORES=32 || break
  done
  esect TAILS
fi

# =================== BMMFULL (cat 4): 100 outlier points ====================
# The shipped slow-rate term covers ONLY both-default layout, B>=4, cores>=8. Still open:
#  (a) FAST/MIXED layouts   -- 41 pts. Need the rate for each of the 4 layout combos.
#  (b) 3d2d projection      -- 32 pts (bmm_wd_3d2d): one rank-3 operand, its own rate.
#  (c) B=2 small-batch      -- the gated corner (~2x faster, ~108 us/GMAC).
#  (d) low-core bmm         -- implied peak 407/241/168 at c1/2/4; the cores>=8 gate hides it.
# All at MATCHED shape/MACs so the layout/batch/cores effect is isolated one at a time.
if has BMMFULL; then
  sect BMMFULL "cat 4: layout quads x B, 3d2d rate, B=2 corner, bmm vs cores"
  for B in 2 4 8; do
    for sh in "1024 2048 1024" "2048 2048 1024" "1024 2048 2048"; do
      set -- $sh
      for lay in "0,1,2 0,1,2" "1,0,2 1,0,2" "0,1,2 1,0,2" "1,0,2 0,1,2"; do
        set -- $sh $lay
        run "bmm_layout B=$B $1x$2x$3 A=$4 B=$5" BENCH_OP=bmm_layout BENCH_B="$B" \
          BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" WD_B=1 WD_M=4 WD_N=8 WD_K=1 \
          WD_LAYOUT_A="$4" WD_LAYOUT_B="$5" SENCORES=32 || break 3
      done
    done
  done
  for B in 2 4 8; do            # (b) the 3d2d projection at matched shapes
    for sh in "1024 2048 1024" "2048 2048 1024"; do
      set -- $sh
      run "bmm_wd_3d2d B=$B $1x$2x$3" BENCH_OP=bmm_wd_3d2d BENCH_B="$B" \
        BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" WD_B=1 WD_M=4 WD_N=8 WD_K=1 SENCORES=32 || break 2
    done
  done
  for c in "1 1 1" "1 2 1" "1 4 1" "2 4 1" "4 4 1" "4 8 1"; do   # (d) bmm vs cores
    set -- $c
    run "bmm_wd cores=$(( $1*$2*$3 )) 1024x2048x1024" BENCH_OP=bmm_wd BENCH_B=4 \
      BENCH_ROWS=1024 BENCH_COLS=2048 BENCH_N=1024 WD_B=1 WD_M="$1" WD_N="$2" WD_K="$3" SENCORES=32 || break
  done
  esect BMMFULL
fi

# =================== MMSPILL (cat 3): 35 outlier points =====================
# The plain-matmul residual is in the SPILL/SPLIT terms: it grows with the per-core output
# tile area (M/m)x(N/n) once it overflows on-chip capacity. Sweep the split SHAPE at fixed
# cores=32 across several aspect ratios so area and aspect are separable.
if has MMSPILL; then
  sect MMSPILL "cat 3: split-shape x per-core area (spill knee + split term)"
  for sh in "2048 2048 2048" "4096 2048 2048" "2048 2048 4096" "8192 2048 2048" "2048 4096 2048"; do
    for mn in "4 8" "8 4" "2 16" "16 2"; do
      set -- $sh $mn
      run "mmwd $1x$2x$3 split $4x$5" BENCH_OP=mmwd BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
        WD_M="$4" WD_N="$5" WD_K=1 SENCORES=32 || break 2
    done
  done
  for sh in "4096 4096 2048" "2048 4096 4096"; do   # bigger area -> deeper into the spill regime
    for mn in "4 8" "8 4" "2 16"; do
      set -- $sh $mn
      run "mmwd $1x$2x$3 split $4x$5" BENCH_OP=mmwd BENCH_ROWS="$1" BENCH_COLS="$2" BENCH_N="$3" \
        WD_M="$4" WD_N="$5" WD_K=1 SENCORES=32 || break 2
    done
  done
  esect MMSPILL
fi

# =============== existing single-purpose sweeps, chained in =================
if has REDCORES;  then sect REDCORES  "cat 5: plain-reduction g(cores) under reps"
  [[ "$DRY" == "1" ]] || bash "$SCRIPT_DIR/run_reduction_cores_sweep.sh"; esect REDCORES; fi
if has GAMMA;     then sect GAMMA     "cat 3: gamma identifiability (unsaturated shapes)"
  [[ "$DRY" == "1" ]] || bash "$SCRIPT_DIR/run_gamma_bind_sweep.sh"; esect GAMMA; fi
if has COARSERED; then sect COARSERED "cat 5: fused-reduction bw(cores, tiles)"
  [[ "$DRY" == "1" ]] || bash "$SCRIPT_DIR/run_coarse_reduction_sweep.sh"; esect COARSERED; fi

# =================== COARSEBMM (cat 6b): coarse batched matmul ==============
if has COARSEBMM; then
  sect COARSEBMM "cat 6b: bmm_k_tiling / bmm_nested_b_k tile ladders (~108 s/run)"
  for op in bmm_k_tiling bmm_nested_b_k; do
    for sh in "1024 2048 1024" "2048 2048 1024"; do
      for t in 1 2 4 8; do
        set -- $sh
        run "$op B=4 $1x$2x$3 tiles=$t" BENCH_OP="$op" BENCH_B=4 BENCH_ROWS="$1" \
          BENCH_COLS="$2" BENCH_N="$3" BENCH_TILES="$t" SENCORES=32 || break 3
      done
    done
  done
  esect COARSEBMM
fi

if has COARSEMM; then sect COARSEMM "cat 6a: coarse matmul tile ladder + fixed-rpc rows (the U-curve)"
  [[ "$DRY" == "1" ]] || bash "$SCRIPT_DIR/run_coarse_matmul_tile_sweep.sh"; esect COARSEMM; fi
if has ADDIR;    then sect ADDIR    "§3: add-chain IR structure (bundle counts)"
  [[ "$DRY" == "1" ]] || bash "$SCRIPT_DIR/run_add_chain_ir.sh"; esect ADDIR; fi

echo "==== CLOSE-OUT DONE in $((SECONDS-_START))s ($(date)) -- $NRUN runs from this script ====" | tee -a "$LOG"
echo "Fold: python3 notes/parse_sweep_logs.py haoyang_logs/closeout_*.log \\" | tee -a "$LOG"
echo "        haoyang_logs/reduction_cores_*.log haoyang_logs/gamma_bind_*.log \\" | tee -a "$LOG"
echo "        haoyang_logs/coarse_reduction_*.log haoyang_logs/coarse_mm_tile_*.log" | tee -a "$LOG"
echo "Then re-score with notes/eval_model.py and refit per category." | tee -a "$LOG"
