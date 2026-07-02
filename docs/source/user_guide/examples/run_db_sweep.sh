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
# COST-MODEL PROFILING DATABASE -- run EVERY sweep back-to-back, then fold all
# logs into notes/sweep_records.{json,csv}. The goal is one command that rebuilds
# the measurement DB, so any model change is validated by diffing model vs the
# recorded kernel_us (never re-derived). ~170 runs, ~30-45 min (runs are now
# CPU-time-bound). A failing child does NOT abort the rest (no set -e).
#
# Chained, in the matmul isolation order (HBM -> compute -> split -> psum) plus
# the global op coverage:
#   run_hbm_ops_sweep.sh        pointwise / reduction / broadcast BW
#   run_transport_sweep.sh      transport (restickify) BW
#   run_matmul_validate_sweep.sh  matmul HBM only (SECTIONS=M1; compute/psum below)
#   run_matmul_compute_sweep.sh   matmul compute (step 2)
#   run_split_sweep.sh + run_decouple_sweep.sh  matmul split penalty (step 3)
#   run_matmul_psum_sweep.sh      matmul psum (step 4)
#   run_coarse_tiling_sweep.sh    softmax/matmul row-tiling x tile count
# Final: notes/parse_sweep_logs.py refreshes the DB from haoyang_logs/*.log.
#
# Run the WHOLE thing (do NOT set SECTIONS -- it is unset here so every child uses
# its own default and full coverage is guaranteed):
#   bash docs/source/user_guide/examples/run_db_sweep.sh
# Output: many haoyang_logs/*.log (forward all) + updated notes/sweep_records.*
# ============================================================================

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR")"
unset SECTIONS 2>/dev/null || true   # force each child's full default coverage

echo "==== run_db_sweep $(date) -- full cost-model database rebuild ===="
echo "git: $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null)"

run_one() {  # run_one <script> [SECTIONS override for this child only]
  local s="$1" sec="${2:-}"
  echo
  echo "################################################################"
  echo "## >>> $s ${sec:+(SECTIONS=$sec)}"
  echo "################################################################"
  if [[ -n "$sec" ]]; then
    SECTIONS="$sec" bash "$SCRIPT_DIR/$s" || echo "## !!! $s exited non-zero (continuing)"
  else
    bash "$SCRIPT_DIR/$s" || echo "## !!! $s exited non-zero (continuing)"
  fi
}

run_one run_hbm_ops_sweep.sh
run_one run_transport_sweep.sh
run_one run_matmul_validate_sweep.sh "M1"   # HBM only; compute/psum have own scripts
run_one run_matmul_compute_sweep.sh
run_one run_split_sweep.sh
run_one run_decouple_sweep.sh
run_one run_matmul_psum_sweep.sh
run_one run_coarse_tiling_sweep.sh

echo
echo "==== parsing all logs into notes/sweep_records.{json,csv} ===="
python "$ROOT/notes/parse_sweep_logs.py" \
  || echo "## !!! parse_sweep_logs.py failed (run it by hand)"

echo
echo "==== run_db_sweep DONE -- forward haoyang_logs/*.log + notes/sweep_records.* ===="
