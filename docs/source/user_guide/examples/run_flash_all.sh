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
# FLASH-ATTENTION DESIGN SPACE -- everything in ONE run.
#
# Explores the two ways to write the same flash-attention program, and times each:
#   DIRECTION 1  hint-driven : one monolithic program + spyre_hint(tiles=..., work_div=...)
#                              -> flash_probe.py
#   DIRECTION 2  hand-blocked: the flash loop written out in Python (online softmax over
#                              key blocks), so the live working set is set by OUR block
#                              sizes, plus the LX-allocation knobs -> flash_manual_tile.py
#
# ANCHOR: section 0 runs the config of the shipped flash_attn_example.py (H_TILES=8,
# LQ_TILES=4, LK_TILES=1, work_div H:4,Lq:8,Lk:8 at Lq=Lk=4096) -- the configuration that
# is KNOWN to run. Everything else is read relative to that number.
#
# Robustness: every config runs in its OWN process under `timeout`, and hint configs are
# pre-validated (pure arithmetic, no device) so a known-invalid work_div never costs a
# compile. Nothing here can hang the whole script.
#
#   bash docs/source/user_guide/examples/run_flash_all.sh              # the full set
#   DRY=1 bash .../run_flash_all.sh                                    # print the plan only
#   SHAPE=2048 CFG_TIMEOUT=600 bash .../run_flash_all.sh               # smaller/looser
#   SECTIONS="0 2" bash .../run_flash_all.sh                           # anchor + manual only
#
# Output: haoyang_logs/flash_all_<ts>.log, plus a final table of every variant's kernel_us.
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/flash_all_$(date +%Y%m%d_%H%M%S).log"
PY="${PYTHON:-python3}"
PROBE="$SCRIPT_DIR/flash_probe.py"
MANUAL="$SCRIPT_DIR/flash_manual_tile.py"
CFG_TIMEOUT="${CFG_TIMEOUT:-900}"     # per-config wall clock; a stuck compile cannot block us
SHAPE="${SHAPE:-4096}"                # Lq = Lk
H="${H:-32}"; D="${D:-128}"
DRY="${DRY:-0}"
SECTIONS="${SECTIONS:-0 1 2 3}"
RESULTS="$(mktemp)"                    # "label<TAB>status<TAB>kernel_us<TAB>secs"
trap 'rm -f "$RESULTS"' EXIT
has() { [[ " $SECTIONS " == *" $1 "* ]]; }

echo "==== FLASH ALL $(date) ====" | tee "$LOG"
echo "shape H=$H Lq=Lk=$SHAPE D=$D | per-config timeout ${CFG_TIMEOUT}s | sections: $SECTIONS" | tee -a "$LOG"

record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$RESULTS"; }

run_cfg() {  # run_cfg <label> <cmd...>
  local label="$1"; shift
  if [[ "$DRY" == "1" ]]; then echo "PLAN  $label :: $*" | tee -a "$LOG"; record "$label" PLAN - 0; return 0; fi
  echo "-- $label" | tee -a "$LOG"
  local t0=$SECONDS out rc
  out=$(timeout -k 30 "$CFG_TIMEOUT" "$@" 2>&1); rc=$?
  local dt=$((SECONDS-t0))
  local line; line=$(printf '%s\n' "$out" | grep -E '^RESULT:' | tail -1)
  if (( rc == 124 || rc == 137 )); then
    echo "   TIMEOUT after ${dt}s" | tee -a "$LOG"; record "$label" TIMEOUT - "$dt"
  elif printf '%s\n' "$line" | grep -q 'RESULT: OK'; then
    local us; us=$(printf '%s\n' "$line" | grep -oE 'kernel_us=[0-9.]+' | cut -d= -f2)
    echo "   OK  kernel_us=${us:-?}  (${dt}s)" | tee -a "$LOG"; record "$label" OK "${us:-?}" "$dt"
  elif printf '%s\n' "$line" | grep -qE 'RESULT: (INVALID|SPLIT-ERROR|WRONG)'; then
    echo "   ${line}" | tee -a "$LOG"
    record "$label" "$(printf '%s' "$line" | awk '{print $2}')" - "$dt"
  else
    echo "   ERROR rc=$rc (${dt}s)" | tee -a "$LOG"
    printf '%s\n' "$out" | grep -iE 'error|Traceback' | head -2 | sed 's/^/      /' | tee -a "$LOG"
    record "$label" ERROR - "$dt"
  fi
  # LX/correctness context lines are useful in the log
  printf '%s\n' "$out" | grep -E '^  (LX|blocks|correctness|hint|WARN)' | sed 's/^/      /' | tee -a "$LOG"
}

hint_cfg() {  # hint_cfg <label> <ht> <qt> <kt> <wd>
  run_cfg "$1" "$PY" "$PROBE" --h "$H" --lq "$SHAPE" --lk "$SHAPE" --d "$D" \
    --h-tiles "$2" --lq-tiles "$3" --lk-tiles "$4" --wd "$5"
}
manual_cfg() {  # manual_cfg <label> <mode> <bh> <bq> <bk> [extra...]
  local label="$1" mode="$2" bh="$3" bq="$4" bk="$5"; shift 5
  run_cfg "$label" "$PY" "$MANUAL" --mode "$mode" --h "$H" --lq "$SHAPE" --lk "$SHAPE" --d "$D" \
    --bh "$bh" --bq "$bq" --bk "$bk" "$@"
}

# ---------------------------------------------------------------- 0: the anchor
if has 0; then
  echo "## 0. ANCHOR -- the shipped flash_attn_example.py configuration (known to run)" | tee -a "$LOG"
  hint_cfg "hint/example-default(h8,q4,k1,wd H:4,Lq:8,Lk:8)" 8 4 1 "H:4,Lq:8,Lk:8"
fi

# ------------------------------------------------- 1: direction 1, the hint sweep
if has 1; then
  echo "## 1. DIRECTION 1 -- hint sweep (tile counts, then work_div at a fixed tiling)" | tee -a "$LOG"
  hint_cfg "hint/tiles h8,q4,k1  (=example)  wd none"   8  4 1 "Lq:1"
  hint_cfg "hint/tiles h8,q8,k2               wd none"   8  8 2 "Lq:1"
  hint_cfg "hint/tiles h16,q8,k4              wd none"  16  8 4 "Lq:1"
  hint_cfg "hint/tiles h32,q16,k8             wd none"  32 16 8 "Lq:1"
  for wd in "H:2" "H:4" "Lq:4" "H:2,Lq:4" "H:4,Lq:8"; do
    hint_cfg "hint/wd $wd  @h8,q8,k2" 8 8 2 "$wd"
  done
fi

# --------------------------------------- 2: direction 2, hand-blocked in Python
if has 2; then
  echo "## 2. DIRECTION 2 -- hand-blocked flash (block sizes set the live working set)" | tee -a "$LOG"
  #                label                              mode          bh   bq    bk
  manual_cfg "manual-fused bh4  bq512 bk1024" manual-fused  4  512 1024
  manual_cfg "manual-fused bh4  bq256 bk512"  manual-fused  4  256  512
  manual_cfg "manual-fused bh8  bq256 bk512"  manual-fused  8  256  512
  manual_cfg "manual-fused bh4  bq128 bk256"  manual-fused  4  128  256
  manual_cfg "manual-sep   bh4  bq256 bk512"  manual-sep    4  256  512
  manual_cfg "manual-sep   bh4  bq128 bk256"  manual-sep    4  128  256
fi

# ------------------------------------------------- 3: LX allocation, same program
if has 3; then
  echo "## 3. LX ALLOCATION -- one fixed hand-blocked program, different LX policies" | tee -a "$LOG"
  manual_cfg "LX default (frac .2 -> 1638KB)"  manual-fused 4 256 512
  manual_cfg "LX frac .05 -> 1946KB budget"    manual-fused 4 256 512 --lx-frac 0.05
  manual_cfg "LX frac .40 -> 1229KB budget"    manual-fused 4 256 512 --lx-frac 0.40
  manual_cfg "LX allow-all-ops eligible"       manual-fused 4 256 512 --lx-all
  manual_cfg "LX boundary clones (pin in/out)" manual-fused 4 256 512 --lx-boundary
  manual_cfg "LX solver=bestfit"               manual-fused 4 256 512 --lx-solver bestfit
  manual_cfg "LX OFF (all through HBM)"        manual-fused 4 256 512 --no-lx
fi

# ------------------------------------------------------------------- the summary
echo | tee -a "$LOG"
echo "=============================== SUMMARY ===============================" | tee -a "$LOG"
printf "%-46s %-9s %12s %6s\n" "variant" "status" "kernel_us" "secs" | tee -a "$LOG"
printf '%s\n' "----------------------------------------------------------------------" | tee -a "$LOG"
sort -t$'\t' -k2,2 -k3,3g "$RESULTS" | while IFS=$'\t' read -r l s us dt; do
  printf "%-46s %-9s %12s %6s\n" "$l" "$s" "$us" "$dt" | tee -a "$LOG"
done
best=$(awk -F'\t' '$2=="OK" && $3!="-" {print $3"\t"$1}' "$RESULTS" | sort -g | head -1)
[[ -n "$best" ]] && echo "FASTEST: $(printf '%s' "$best" | cut -f2)  at $(printf '%s' "$best" | cut -f1) us" | tee -a "$LOG"
echo "==== DONE in ${SECONDS}s -- $LOG ====" | tee -a "$LOG"
