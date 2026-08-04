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
# RE-EXTRACT FLASH ATTENTION AFTER THE RANK>=3 tile_rows_per_core FIX.
#
# WHY THIS RUN IS NEEDED. The fix is in the EXTRACTOR (dump_cost_model.py), so it
# only changes features produced from here on. Every flash record already in
# sweep_records.json carries the OLD, broken tile_rows_per_core (0.004-0.031), so
# re-scoring them offline cannot show the fix working -- the wrong number is baked
# into the stored features. Only a fresh extraction banks it.
#
# THE BUG, for the record. tile_rows_per_core took the row extent from the DEVICE
# shape as out_dims[-2]. That is the row count only at rank 2 (device rank 3). A
# rank-4 flash tensor [1,4,1024,128] has device shape [4,1024,2,1,64], so [-2] is a
# degenerate 1; a rank-3 bmm output [2,1024,1024] lays out as [1024,16,2,64], so
# [-2] is the batch. Divided by loop_trip and the core split that gives sub-unity
# "rows per core", which drove coarse_underfill_eff to ~0.007 and inflated the
# memory term 60-248x. It now reads the row extent from the LOGICAL shape, which is
# correct at every rank and verified byte-identical on all 1177 recorded rank-2 ops.
#
# EXPECTED RESULT. Simulating the fix on the 7 stored records takes flash from
# +1370..+4431 % to roughly +6..+76 % on five of seven shapes, RMS 3521 % -> 266 %.
# Two shapes stay high and are the interesting ones:
#   * the ktiles=2 variant (32 ops, K-tiled)          simulated +663 %
#   * the D=128 htiles=8 qtiles=1 run                 simulated +205 %
# If the fresh numbers land near those, the diagnosis holds and the remaining error
# is the four secondary effects in notes/cost_model_directions.md (per-tile
# matmul_macs, K/V/mask de-duplication, the rank-3 layout filters, the unreachable
# fused-reduction floor). If they do NOT, the diagnosis is wrong -- say so.
#
# ALSO WATCH: fixing rows/core wakes _lx_spill_working_set on flash for the first
# time, and _lx_spill_bw_derate applies the MATMUL cap/exponent to any bundle
# containing a matmul -- i.e. a matmul calibration applied to what is mostly a
# 21-op softmax chain. Check whether the spill derate is now firing on these runs.
#
# LOG FORMAT IS LOAD-BEARING: parse_sweep_logs.py needs `git: <sha>`, `## <section>`,
# `-- <label>`, and SUMMARY at COLUMN 0. SPYRE_DUMP_COST=1 is REQUIRED (note "2"
# would enable the new pass but DISABLE the per-op dump this parser reads).
#
# COST: 7 configs, reps=3.  ~15 min.
#
#   bash docs/source/user_guide/examples/run_flash_reextract.sh
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/flash_reextract_$(date +%Y%m%d_%H%M%S).log"
REPS="${BENCH_REPS:-3}"

# Refuse to run against the unfixed extractor -- the whole point is the new value.
if ! grep -q 'Row extent from the LOGICAL shape' torch_spyre/_inductor/dump_cost_model.py; then
  echo "ABORT: dump_cost_model.py does not carry the rank>=3 row-extent fix." | tee "$LOG"
  exit 1
fi

echo "==== FLASH RE-EXTRACT $(date) ====" | tee "$LOG"
echo "git: $(git rev-parse --short HEAD)" | tee -a "$LOG"
echo "# rank>=3 tile_rows_per_core fix; reps=$REPS" | tee -a "$LOG"
echo "## flash_reextract" | tee -a "$LOG"

run_one() {  # run_one <H> <Lq/Lk> <D> <h_tiles> <q_tiles> <k_tiles>
  local h="$1" lq="$2" d="$3" ht="$4" qt="$5" kt="$6" out line
  local label="flash H=$h Lq=$lq Lk=$lq D=$d htiles=$ht qtiles=$qt ktiles=$kt"
  out=$(SPYRE_DUMP_COST=1 TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
        BENCH_OP=flash_attn FA_H="$h" FA_LQ="$lq" FA_LK="$lq" FA_D="$d" \
        FA_H_TILES="$ht" FA_LQ_TILES="$qt" FA_LK_TILES="$kt" \
        BENCH_REPS="$REPS" \
        timeout -k 30 "${RUN_TIMEOUT:-900}" python "$PROFILE_OPS" 2>&1)
  line=$(printf '%s\n' "$out" | grep -E '^SUMMARY' | head -1)
  echo "-- $label" | tee -a "$LOG"
  if [ -z "$line" ]; then
    echo "#   FAILED" | tee -a "$LOG"
    printf '%s\n' "$out" | tail -3 | sed 's/^/#     /' | tee -a "$LOG"
  else
    printf '%s\n' "$out" | grep -E '^(IO |MODEL |op_it_space_splits)' | tee -a "$LOG"
    printf '%s\n' "$line" | tee -a "$LOG"
  fi
}

# The seven configurations already in the database, so old and new are comparable.
run_one 32 1024 128 4 2 1
run_one 32 1024 128 8 2 1
run_one 32 1024 128 8 4 1
run_one 32 2048 128 8 4 1
run_one 32 4096 128 8 4 1
run_one 32 2048 128 1 1 2      # the ktiles=2 variant
run_one 32 2048 128 8 1 1      # the D=128 htiles=8 qtiles=1 run

{
  echo ""
  echo "==== CHECK BEFORE TRUSTING ANY OF IT ===="
  echo "  1. Fold ONLY this log:  python3 notes/parse_sweep_logs.py $LOG"
  echo "     (never re-parse haoyang_logs/* -- curated logs are gone from disk and"
  echo "      189 records survive only inside sweep_records.json)"
  echo "  2. VERIFY tile_rows_per_core is now O(10-100), not O(0.01). If it is still"
  echo "     sub-unity the fix did not take -- STOP, do not fold."
  echo "  3. Re-score:  python3 notes/eval_model.py --all --op flash_attn"
  echo "     Expect roughly +6..+76 % on five shapes; ktiles=2 and the qtiles=1 run"
  echo "     are expected to stay high (~+663 % / ~+205 % simulated)."
  echo "  4. Check whether _lx_spill_bw_derate now fires on flash -- the fix wakes it,"
  echo "     with a MATMUL calibration applied to a mostly-softmax chain."
  echo "  5. Confirm gold is untouched:  python3 notes/eval_model.py --all"
  echo "     broadcast 5.7 / transport 6.1 / reduction 7.2 / pointwise 8.3 must not move."
} | tee -a "$LOG"
