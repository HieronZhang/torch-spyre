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
# FLASH-ATTENTION RUNNABILITY PROBE (cat-7) -- find which hint configs actually RUN.
#
# The old flash sweep mostly "did not finish". Two causes, handled differently here:
#   (a) LX SCRATCHPAD EXHAUSTION -> the run HANGS until the timeout. Dominant cause. Driven by
#       the per-core working set: the fused region's per-tile intermediates (scores/exp_scores,
#       each [B_t,H_t,Lq_t,Lk_t] fp16) must fit in ~512 KB/core. Shrink it with MORE tiles
#       and/or a work_div that actually gets placed.
#   (b) INVALID work_div split -> fast InductorError ("not evenly divisible"): a split must
#       divide the PER-TILE size, (dim/tiles) % split == 0.
#
# Strategy (this is the "fix"): PRE-VALIDATE every config with flash_probe.py --validate-only
# (pure arithmetic, no device, instant) and only spend a compile on the ones that can work.
# Each surviving config then runs in its OWN process under `timeout`, because a config that
# exhausts LX can hang -- including in Spyre-runtime teardown AFTER reporting its error.
#
#   bash docs/source/user_guide/examples/run_flash_probe.sh                  # default matrix
#   PROBE_TIMEOUT=180 bash .../run_flash_probe.sh                            # tighter timeout
#   DRY=1 bash .../run_flash_probe.sh          # validate only, no device, no compiles
#
# Output: one line per config (VALID/INVALID from the validator, then OK/TIMEOUT/ERROR from the
# run) + a tally, in haoyang_logs/flash_probe_<ts>.log. Start with DRY=1 to see the matrix.
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROBE="$SCRIPT_DIR/flash_probe.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/flash_probe_$(date +%Y%m%d_%H%M%S).log"
PROBE_TIMEOUT="${PROBE_TIMEOUT:-300}"   # a hung (LX-exhausted) config is killed after this
DRY="${DRY:-0}"
PY="${PYTHON:-python3}"
echo "==== FLASH PROBE $(date)  timeout=${PROBE_TIMEOUT}s  dry=${DRY} ====" | tee "$LOG"

n_valid=0; n_invalid=0; n_ok=0; n_timeout=0; n_err=0

probe() {  # probe <H> <Lq> <Lk> <h_tiles> <lq_tiles> <lk_tiles> <wd>
  local H=$1 Lq=$2 Lk=$3 ht=$4 qt=$5 kt=$6 wd=$7
  local tag="H${H}_Lq${Lq}_Lk${Lk}_h${ht}q${qt}k${kt}_wd[${wd}]"
  # ---- step 1: validate (instant, no device) ----
  local vout
  vout=$("$PY" "$PROBE" --validate-only --h "$H" --lq "$Lq" --lk "$Lk" \
          --h-tiles "$ht" --lq-tiles "$qt" --lk-tiles "$kt" --wd "$wd" 2>&1)
  local lx; lx=$(printf '%s\n' "$vout" | grep -oE 'LX working set : ~[0-9]+ KB/core at 32 cores' | head -1)
  if printf '%s\n' "$vout" | grep -q '^RESULT: INVALID'; then
    n_invalid=$((n_invalid+1))
    echo "INVALID  $tag  ${lx}" | tee -a "$LOG"
    printf '%s\n' "$vout" | grep '^  ERROR' | head -1 | sed 's/^/         /' | tee -a "$LOG"
    return 0
  fi
  n_valid=$((n_valid+1))
  local lxwarn=""
  printf '%s\n' "$vout" | grep -q 'LX OVERFLOW CERTAIN' && lxwarn="  [LX OVERFLOW CERTAIN -> will HANG]"
  printf '%s\n' "$vout" | grep -q 'LX overflow POSSIBLE' && lxwarn="  [LX overflow possible]"
  if [[ "$DRY" == "1" ]]; then
    echo "VALID    $tag  ${lx}${lxwarn}" | tee -a "$LOG"; return 0
  fi
  # ---- step 2: run it, isolated + time-boxed (a hang must not kill the sweep) ----
  local t0=$SECONDS out rc
  out=$(timeout -k 20 "$PROBE_TIMEOUT" "$PY" "$PROBE" --h "$H" --lq "$Lq" --lk "$Lk" \
          --h-tiles "$ht" --lq-tiles "$qt" --lk-tiles "$kt" --wd "$wd" 2>&1); rc=$?
  local dt=$((SECONDS-t0))
  if (( rc == 124 || rc == 137 )); then
    n_timeout=$((n_timeout+1))
    echo "TIMEOUT  $tag  after ${dt}s  ${lx}${lxwarn}" | tee -a "$LOG"
  elif printf '%s\n' "$out" | grep -q '^RESULT: OK'; then
    n_ok=$((n_ok+1))
    echo "$(printf '%s\n' "$out" | grep '^RESULT: OK' | sed 's/^RESULT: OK/OK      /')  (${dt}s)  ${lx}" | tee -a "$LOG"
  else
    n_err=$((n_err+1))
    echo "ERROR    $tag  after ${dt}s (rc=$rc)  ${lx}" | tee -a "$LOG"
    printf '%s\n' "$out" | grep -E '^RESULT:|Error|error:' | head -2 | sed 's/^/         /' | tee -a "$LOG"
  fi
}

# ---------------------------------------------------------------------------
# The matrix, ordered SMALL -> LARGE so the runnable frontier is found early.
# Per-tile scores = (H/ht)*(Lq/qt)*(Lk/kt) elems x 2 B, divided by the placed cores.
# ---------------------------------------------------------------------------
echo "## A. baseline: no work_div, increasing tile counts (isolates the LX/tiling axis)" | tee -a "$LOG"
for shape in "32 1024 1024" "32 2048 2048" "32 4096 4096"; do
  set -- $shape
  for t in "8 4 1" "8 8 2" "16 8 4" "32 16 8"; do
    set -- $shape $t
    probe "$1" "$2" "$3" "$4" "$5" "$6" "Lq:1"
  done
done

echo "## B. work_div sweep at a tiling that FITS (product <= 32, splits divide per-tile)" | tee -a "$LOG"
for wd in "H:2" "H:4" "Lq:2" "Lq:4" "H:2,Lq:2" "H:2,Lq:4" "H:4,Lq:4" "H:2,Lq:2,Lk:2" "H:4,Lq:8"; do
  probe 32 2048 2048 8 8 2 "$wd"
done

echo "## C. the OLD failing configs -- kept as regression cases (expect INVALID / LX-overflow)" | tee -a "$LOG"
probe 32 4096 4096 16 4 1 "H:4,Lq:8,Lk:8"   # per-tile H=2, split 4 -> InductorError
probe 32 4096 4096 8  4 1 "H:8,Lq:4"        # per-tile H=4, split 8 -> InductorError
probe 32 4096 4096 8  4 1 "H:4,Lq:8,Lk:8"   # product 256 -> skipped -> LX blow-up -> hang

echo "==== DONE in ${SECONDS}s ====" | tee -a "$LOG"
echo "TALLY  valid=$n_valid invalid=$n_invalid | ran: OK=$n_ok TIMEOUT=$n_timeout ERROR=$n_err" | tee -a "$LOG"
echo "Next: for every OK config, the kernel_us is on its line -- feed those into the model comparison." | tee -a "$LOG"
