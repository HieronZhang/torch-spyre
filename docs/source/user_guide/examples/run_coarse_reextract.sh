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
# COARSE RE-EXTRACT + DENSE L LADDER -- one sweep, two jobs.
#
# JOB 1: RE-EXTRACT with the new PER-LEVEL `loop_factor`.
# `dump_cost_model.py` now derives each arg's transfer count from its OWN index
# expression against the per-level tiled symbols:
#
#     factor(arg) = PRODUCT over levels L of
#                     ( loop_count[L] if the index has no tiled symbol of level L else 1 )
#
# IR-verified factors (4096x2048x2048, t=4, out / A / B):
#     matmul_k_tiling    4 / 1 / 1   <- was ALREADY correct; this op is the CONTROL
#     matmul_row_tiling  1 / 1 / 4   <- B is invariant in M; the old rule said 1
#     mm_nested_m_k      4 / 1 / 2   <- old rule said 1/1/1; the OUTPUT advances at
#                                      level 0 and repeats at level 1 (1*4), while B
#                                      does the opposite (2*1). NOT the total L=8 --
#                                      charging 8 is the 4x over-count that made the
#                                      flat rule score 143 %.
# Those factors only take effect on a FRESH extraction, which is what this sweep is
# for: every recorded row scores from the `feats` captured at ITS run time, so the
# existing 2780 records still carry the old values.
#
# JOB 2: the DENSE L ladder, still outstanding.
# The re-read RATE is settled (alpha = 1.0, a full HBM pass over B per iteration,
# confirmed three ways). What is NOT settled is the opposing BENEFIT: with B's cost
# removed, `R(L) = T(L) - (L-1)*B/peak` falls and SATURATES at 0.32-0.99 of its L=1
# value, and the model's existing spill term reaches zero by L=4 while the measurement
# keeps improving through L=16. Four L values bracket that turnover but cannot resolve
# its shape; this runs L = 1,2,3,4,6,8,12,16,24,32 at two N values.
#
# WHY BOTH IN ONE SWEEP: the two fixes are ENTANGLED and each REGRESSES ALONE --
# the re-read term alone flips matmul_row_tiling -14.7 % -> +10.8 %, and removing the
# softmax-calibrated underfill cap alone moves it 21.8 % -> 22.5 %. Together they
# improve it. They must be re-fit as a pair, on data extracted with the new factors,
# so both jobs need the same run.
#
# LOG FORMAT IS LOAD-BEARING (this bit a previous script): `parse_sweep_logs.py`
# recognises only a `git: <sha>` line, `## <section>`, `-- <label>`, and a SUMMARY at
# COLUMN 0. Do not indent or prefix the SUMMARY. SPYRE_DUMP_COST=1 is REQUIRED or the
# rows carry no `feats` and cannot be scored against a changed model at all.
#
# COST: 32 runs, reps=5, cores=32 throughout. The CONTROL is first: `matmul_k_tiling`
# already had correct factors and scores 7.9 % RMS, so if it MOVES, the per-level
# change has broken something and nothing else in the run should be trusted.
#
#   bash docs/source/user_guide/examples/run_coarse_reextract.sh
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/coarse_reextract_$(date +%Y%m%d_%H%M%S).log"

M=${M:-4096}
K=${K:-2048}
REPS=${BENCH_REPS:-5}
CORES=${SENCORES:-32}

echo "==== COARSE RE-EXTRACT + DENSE L  $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD)" | tee -a "$LOG"
echo "# M=$M K=$K cores=$CORES reps=$REPS" | tee -a "$LOG"
echo "## coarse_reextract" | tee -a "$LOG"

run_one() {
  local op="$1" N="$2" t="$3"
  local out line
  out=$(SENCORES="$CORES" SPYRE_DUMP_COST=1 BENCH_OP="$op" \
        BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
        BENCH_TILES="$t" BENCH_REPS="$REPS" \
        timeout -k 30 "${RUN_TIMEOUT:-600}" python "$PROFILE_OPS" 2>&1)
  line=$(printf '%s\n' "$out" | grep -E '^SUMMARY' | head -1)
  echo "-- $op ${M}x${K}x${N} t=$t cores=$CORES" | tee -a "$LOG"
  if [ -z "$line" ]; then
    echo "#   FAILED" | tee -a "$LOG"
    printf '%s\n' "$out" | tail -3 | sed 's/^/#     /' | tee -a "$LOG"
  else
    printf '%s\n' "$out" | grep -E '^(IO |MODEL |op_it_space_splits)' | tee -a "$LOG"
    printf '%s\n' "$line" | tee -a "$LOG"
  fi
}

# ---- CONTROL FIRST: factors were already correct here; it must NOT move. ----
echo "# --- CONTROL matmul_k_tiling (expect ~7.9 % RMS, unchanged) ---" | tee -a "$LOG"
for t in 1 2 4 8; do run_one matmul_k_tiling 2048 "$t"; done

# ---- The nested case the redesign is for ----
echo "# --- mm_nested_m_k (out 1->4, B 1->2 at t=4) ---" | tee -a "$LOG"
for t in 1 2 4 8; do run_one mm_nested_m_k 2048 "$t"; done

# ---- matmul_row_tiling: matched to the existing rows, plus the DENSE ladder ----
echo "# --- matmul_row_tiling N=2048 (matches existing records) ---" | tee -a "$LOG"
for t in 1 4 8 16; do run_one matmul_row_tiling 2048 "$t"; done

for N in 1024 4096; do
  echo "# --- matmul_row_tiling N=$N DENSE L (benefit saturation) ---" | tee -a "$LOG"
  for t in 1 2 4 8 16 32; do run_one matmul_row_tiling "$N" "$t"; done   # divisors of M only: coarse_tile requires even divisibility
done

{
  echo ""
  echo "==== WHAT TO DO WITH IT ===="
  echo "  1. Fold ONLY this log:  python3 notes/parse_sweep_logs.py $LOG"
  echo "     (never re-parse haoyang_logs/* -- 10 curated logs no longer exist on disk"
  echo "      and 189 records survive only inside sweep_records.json)"
  echo "  2. CONTROL CHECK: matmul_k_tiling must stay ~7.9 % RMS. If it moved, the"
  echo "     per-level loop_factor change is wrong -- stop and diagnose."
  echo "  3. Confirm the new factors landed: the re-extracted rows should show"
  echo "     matmul_row_tiling B loop_factor = tiles, and mm_nested_m_k output = K-split"
  echo "     with B = M-split. If they are still 1, the extractor change did not take."
  echo "  4. Then re-fit the re-read term and the underfill cap AS A PAIR on the dense"
  echo "     ladder; neither may be shipped alone (each regresses on its own)."
} | tee -a "$LOG"
