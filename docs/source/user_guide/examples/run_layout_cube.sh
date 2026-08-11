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
# THE FULL 2x2x2: A layout x B layout x OUTPUT layout, for a batched multiply.
#
# WHAT THE EXISTING STUDY COVERS, AND WHAT IT DOES NOT. The 138 recorded
# `bmm_layout` runs vary the two INPUT operands only -- `_to_dev` can place an
# input with an explicit device order, but the output's layout is chosen by the
# compiler. Reading it back off those records shows C was row-outer in every
# one of them. So we can say what switching the inputs buys (2.82x mean, up to
# 3.69x) and nothing at all about the output.
#
# That gap matters because `matmul_preferred_layout` has three settings --
# off, "output" (output only) and "on" (inputs and output) -- and the study
# speaks to none of them cleanly.
#
# HOW C IS CONTROLLED. Not by the harness: the output is not a tensor we place.
# It is controlled by asking the compiler to prefer it, and then RECORDING what
# we actually got -- profile_ops.py emits `layout_c=` on the SUMMARY line,
# derived from the extracted device dims. Never assume the request was honoured;
# read the column.
#
# MEASURE ALL EIGHT CELLS IN ONE SESSION. Do not reuse the four A/B numbers
# already in the database: they were taken on an earlier toolchain, and a
# spot-check on a current build measured 261 us for a softmax the database has
# at 390 us. Mixing builds across cells of one comparison is how a compiler
# change gets mistaken for a layout effect.
#
# REQUIRES the preferred-matmul-layout feature (upstream #3364). Check with:
#   grep -c matmul_preferred_layout torch_spyre/_inductor/config.py
#
# COST: 8 cells x N shapes, a few seconds each.
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")"
ROOT="$(git rev-parse --show-toplevel)"

if ! grep -q "matmul_preferred_layout" "$ROOT/torch_spyre/_inductor/config.py"; then
  echo "This branch lacks matmul_preferred_layout (upstream #3364)."
  echo "Merge it first, e.g.:  git merge pr3364"
  exit 1
fi

OUT="${OUT:-layout_cube_$(date +%Y%m%d_%H%M%S).log}"
REPS="${REPS:-7}"
# One shape per batch size. B=4 1024x2048x1024 is the shape the report tabulates,
# so it ties the new cube back to the existing numbers.
SHAPES="${SHAPES:-4:1024:2048:1024 2:1024:2048:1024 8:1024:2048:1024}"

echo "writing to $OUT"
{
  echo "# layout cube: A x B x C, all cells measured on THIS build"
  echo "# reps=$REPS  shapes=$SHAPES"
  echo "# layout_c on each SUMMARY is OBSERVED, not requested"
} | tee "$OUT"

for shape in $SHAPES; do
  IFS=: read -r B M K N <<< "$shape"
  for LA in "0,1,2" "1,0,2"; do
    for LB in "0,1,2" "1,0,2"; do
      # "" leaves the output at the compiler default; "output" asks for the
      # preferred order on C ONLY, leaving the forced inputs alone.
      for PREF in "" "output"; do
        echo "=== B=$B ${M}x${K}x${N} A=$LA B=$LB pref='${PREF:-off}' ===" | tee -a "$OUT"
        SENCORES=32 \
        BENCH_OP=bmm_layout \
        BENCH_B="$B" BENCH_ROWS="$M" BENCH_COLS="$K" BENCH_N="$N" \
        WD_LAYOUT_A="$LA" WD_LAYOUT_B="$LB" \
        SPYRE_MATMUL_PREFERRED_LAYOUT="$PREF" \
        BENCH_REPS="$REPS" \
        BENCH_EMIT_RECORDS=1 \
        SPYRE_DUMP_COST=1 \
          python3 profile_ops.py 2>&1 | tee -a "$OUT"
      done
    done
  done
done

echo
echo "done -> $OUT"
echo
echo "Read the eight cells with:"
echo "  grep '^SUMMARY' $OUT | sed 's/.*layout_c=\([^ ]*\).*kernel_us=\([^ ]*\).*/C=\1 t=\2/'"
echo
echo "Check FIRST that layout_c actually changed between the pref='off' and"
echo "pref='output' rows. If it did not, the request was not honoured and the"
echo "two rows are the same experiment."
