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
# FINAL COARSE RE-EXTRACT -- the two ops whose recorded features are still wrong.
#
# `_loop_factor_for_index` is now correct and unit-verified 13/13. The rule is one line:
#
#     an arg REPEATS at a nesting level whenever that level's tiled symbols are ABSENT
#     from the arg's index -- for EITHER reason (the level tiles a dim this op does not
#     iterate, or it tiles a dim this arg's address does not depend on).
#     factor = PRODUCT over levels of (loop_count[L] if absent else 1)
#
# Two earlier versions were wrong, both caught by the control:
#   v1  no per-arg logic at all (two per-op scalars)  -> B under-counted on M-tiling
#   v2  guarded with `if syms`, which dropped levels whose symbols were absent
#       -> fill/combine args under-counted; ONE K-tiled bundle went 552 MB -> 216 MB
#          and the control op moved -6.3 % -> -64.2 %
#
# WHY THESE TWO OPS. Every recorded `matmul_k_tiling` and `mm_nested_m_k` row carries
# features from v1 or v2, so their byte counts are wrong and NO model term can score
# them fairly. They are currently quarantined out of the evaluation. `matmul_row_tiling`
# does NOT need this -- it is a single-op bundle with no fill/combine, and its rows were
# repaired deterministically (validated 11/11 against fresh extractions).
#
# IR-verified factors this run must produce (4096x2048x2048, t=4):
#   matmul_k_tiling   matmul op   out=4  A=1  B=1     fill/combine args = loop_trip
#   mm_nested_m_k     matmul op   out=4  A=1  B=2     combine args = K-split (=4)
# The header check below prints them so a wrong extractor is caught immediately rather
# than silently poisoning the database again.
#
# LOG FORMAT IS LOAD-BEARING: parse_sweep_logs.py needs `git: <sha>`, `## <section>`,
# `-- <label>`, and SUMMARY at COLUMN 0. SPYRE_DUMP_COST=1 is REQUIRED.
#
# COST: 2 ops x 5 tile counts x 2 shapes = 20 runs, reps=5.
#
#   bash docs/source/user_guide/examples/run_coarse_final_reextract.sh
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/coarse_final_$(date +%Y%m%d_%H%M%S).log"

K=${K:-2048}
REPS=${BENCH_REPS:-5}
CORES=${SENCORES:-32}

# Guard: refuse to run against the v2 (`if syms`) implementation.
if grep -q 'if declared == 0:' torch_spyre/_inductor/dump_cost_model.py; then
  echo "ABORT: dump_cost_model.py still has the v2 declared==0 branch." | tee "$LOG"
  echo "       Expected the simplified rule: 'if not (syms & free): factor *= trip'." | tee -a "$LOG"
  exit 1
fi

echo "==== FINAL COARSE RE-EXTRACT  $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD)" | tee -a "$LOG"
echo "# K=$K cores=$CORES reps=$REPS -- ops whose recorded feats are stale/corrupt" | tee -a "$LOG"
echo "## coarse_final" | tee -a "$LOG"

run_one() {
  local op="$1" M="$2" N="$3" t="$4"
  local out line
  out=$(SENCORES="$CORES" SPYRE_DUMP_COST=1 BENCH_OP="$op" \
        BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
        BENCH_TILES="$t" BENCH_REPS="$REPS" \
        timeout -k 30 "${RUN_TIMEOUT:-900}" python "$PROFILE_OPS" 2>&1)
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

# t must divide M evenly (coarse_tile requirement); all powers of two here.
for op in matmul_k_tiling mm_nested_m_k; do
  for M in 2048 4096; do
    echo "# --- $op M=$M ---" | tee -a "$LOG"
    for t in 1 2 4 8 16; do run_one "$op" "$M" 2048 "$t"; done
  done
done

{
  echo ""
  echo "==== CHECK BEFORE TRUSTING ANY OF IT ===="
  echo "  1. Fold ONLY this log: python3 notes/parse_sweep_logs.py $LOG"
  echo "     (never re-parse haoyang_logs/* -- 10 curated logs are gone from disk and"
  echo "      189 records survive only inside sweep_records.json)"
  echo "  2. VERIFY THE FACTORS at 4096x2048x2048 t=4:"
  echo "       matmul_k_tiling  matmul op out=4 A=1 B=1; fill/combine args = 8"
  echo "       mm_nested_m_k    matmul op out=4 A=1 B=2; combine args = 4"
  echo "     If fill/combine args are 1, the extractor fix did not take -- STOP."
  echo "  3. Then drop the quarantine in notes/eval_model.py (_BUGGY_LOOP_FACTOR_LOG)"
  echo "     for the ops this run replaces, and re-score."
} | tee -a "$LOG"
