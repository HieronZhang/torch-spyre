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
# Does SPYRE_MATMUL_PREFERRED_LAYOUT actually change the emitted device layout?
#
# This is the load-bearing assumption behind the whole layout A/B, and it is
# checkable WITHOUT trusting any timing. PR #3364 selects host dim_order
# [row, batch..., stick] instead of the default [batch..., row, stick]; for a
# rank-3 bmm output that is [1,0,2] rather than [0,1,2].
#
# HOW WE OBSERVE IT. `profile_ops.py` prints `MODEL FEATS <json>`, which carries
# each operand's `logical` shape and its committed device `dims`. The compiler
# default puts the batch dim at DEVICE POSITION -2, so
#       dims[-2] == logical[0]   <=>   slow default [0,1,2] order
# and the preferred order breaks that equality. This is exactly the test the
# cost model's own `_bmm_layout_pair` uses, so a flip here is also a flip in how
# the model prices the op. Verified against the recorded pre-PR data: every
# rank-3 operand of a `bmm_wd` reads DEFAULT, which is the baseline this must
# reproduce with the flag unset.
#
# We use `bmm_wd` (a plain forced-split bmm that the COMPILER lays out), not
# `bmm_layout` -- the latter pins layouts explicitly via WD_LAYOUT_A/B and would
# therefore be blind to the flag.
#
#   bash docs/source/user_guide/examples/check_preferred_layout.sh
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
PROFILE_OPS="$SCRIPT_DIR/profile_ops.py"
cd "$ROOT" || exit 1
mkdir -p haoyang_logs
LOG="haoyang_logs/preflayout_$(date +%Y%m%d_%H%M%S).log"

# The feats parser lives in its own file so the shell never has to quote Python.
PARSER="$(mktemp -t preflayout_parse_XXXXXX.py)"
trap 'rm -f "$PARSER"' EXIT
cat > "$PARSER" <<'PYEOF'
import json
import sys

raw = sys.stdin.read().strip()
if not raw:
    print("   (no MODEL FEATS line -- did the run fail?)")
    raise SystemExit
try:
    feats = json.loads(raw)
except Exception as exc:  # noqa: BLE001
    print(f"   (could not parse MODEL FEATS: {exc})")
    raise SystemExit
seen = False
for op in feats:
    if not op.get("is_matmul"):
        continue
    for arg in op.get("args") or []:
        logical = arg.get("logical") or []
        dims = arg.get("dims") or []
        if len(logical) != 3 or len(dims) < 2:
            continue
        seen = True
        is_default = dims[-2] == logical[0]
        verdict = "DEFAULT [0,1,2] (slow)" if is_default else "PREFERRED [1,0,2] (fast)"
        role = arg.get("role", "?")
        print(f"   {role:6s} logical={logical} dims={dims}  -> {verdict}")
if not seen:
    print("   (no rank-3 matmul operand found -- wrong op or shape?)")
PYEOF

B=${B:-4}; M=${M:-1024}; K=${K:-2048}; N=${N:-1024}
echo "==== PREFERRED-LAYOUT CHECK  bmm_wd B=$B ${M}x${K}x${N}  $(date) ====" | tee "$LOG"

for mode in "" "output" "on"; do
  label="${mode:-<unset/disabled>}"
  echo "" | tee -a "$LOG"
  echo "---- SPYRE_MATMUL_PREFERRED_LAYOUT=$label ----" | tee -a "$LOG"
  out=$(SPYRE_MATMUL_PREFERRED_LAYOUT="$mode" SENCORES=32 \
        BENCH_OP=bmm_wd BENCH_B="$B" BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
        WD_B=1 WD_M=4 WD_N=8 WD_K=1 BENCH_REPS="${BENCH_REPS:-3}" \
        timeout -k 30 "${RUN_TIMEOUT:-400}" python "$PROFILE_OPS" 2>&1)
  printf '%s\n' "$out" | grep -E '^SUMMARY' | tee -a "$LOG"
  printf '%s\n' "$out" | grep -E '^MODEL FEATS ' | head -1 | sed 's/^MODEL FEATS //' \
    | python3 "$PARSER" | tee -a "$LOG"
done

{
  echo ""
  echo "==== READ IT LIKE THIS ===="
  echo "  disabled : every rank-3 operand should say DEFAULT"
  echo "  output   : only the OUTPUT should flip to PREFERRED"
  echo "  on       : the two INPUTS should flip too"
  echo "  If nothing flips, the flag is not reaching the layout pass and the whole"
  echo "  A/B is meaningless -- stop and debug that before collecting any timings."
  echo "  kernel_us is printed too, but treat it as secondary: the LAYOUT is what is"
  echo "  being verified here, not the speedup."
} | tee -a "$LOG"
