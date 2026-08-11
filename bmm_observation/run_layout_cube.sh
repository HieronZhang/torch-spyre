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
# =============================================================================
# Layout cube: time a batched matmul under all eight combinations of its three
# operands' device layouts.  Reproduces the table in bmm_layout_effects.md.
#
#   A  first input,  [B, M, K]   placed explicitly by this script
#   B  second input, [B, K, N]   placed explicitly by this script
#   C  output,       [B, M, N]   requested via SPYRE_MATMUL_PREFERRED_LAYOUT
#                                and then read back from the result tensor
#
# Each operand is either row-outer (dim_order 0,1,2 -- the compiler default,
# with the row axis M slowest-varying) or batch-outer (dim_order 1,0,2, with
# the batch axis B slowest-varying).
#
# The output cannot be placed directly: its layout is chosen by the compiler.
# It is *requested* through SPYRE_MATMUL_PREFERRED_LAYOUT="output" and the
# script prints the layout the result actually came back with, because a
# request is not a guarantee.  Check that column before believing a C effect.
#
# Every cell runs in this one invocation, so the whole cube shares a single
# compiler build.  Do not splice in cells measured earlier: kernel performance
# moves as the compiler develops, and a build difference between cells is
# indistinguishable from a layout effect.
#
# Requirements: Spyre hardware, torch, torch_spyre, and the preferred-matmul-
# layout support (upstream PR #3364), which provides the C axis.
#
# Usage:
#   ./run_layout_cube.sh                     # B = 2, 4, 8 at 1024x2048x1024
#   BATCHES="4" ./run_layout_cube.sh         # one batch size
#   M=2048 K=2048 N=1024 ./run_layout_cube.sh
#   REPS=15 ./run_layout_cube.sh             # more repeats per cell
#
# Runtime: 24 cells at the default settings, a few seconds each.
# =============================================================================
set -uo pipefail

M="${M:-1024}"
K="${K:-2048}"
N="${N:-1024}"
BATCHES="${BATCHES:-2 4 8}"
REPS="${REPS:-7}"
CORES="${SENCORES:-32}"
OUT="${OUT:-layout_cube.log}"

CELL_PY="$(mktemp -t layout_cube_cell_XXXXXX.py)"
trap 'rm -f "$CELL_PY"' EXIT

# One cell = one process.  The layout preference is read from the environment
# when torch_spyre is imported, so it cannot be changed inside a running
# process; a fresh interpreter per cell is what makes the C axis controllable.
cat > "$CELL_PY" <<'PYEOF'
"""Time one bmm cell and report the output layout the compiler chose."""

import os
import statistics

import torch
from torch.profiler import ProfilerActivity, profile

import torch_spyre  # noqa: F401  (registers the "spyre" device)
from torch_spyre._C import SpyreTensorLayout

DEVICE = "spyre"
B = int(os.environ["CELL_B"])
M = int(os.environ["CELL_M"])
K = int(os.environ["CELL_K"])
N = int(os.environ["CELL_N"])
REPS = int(os.environ["CELL_REPS"])
ORDER_A = os.environ["CELL_A"]
ORDER_B = os.environ["CELL_B_ORDER"]
NAME = {"0,1,2": "row", "1,0,2": "batch"}


def place(t, order):
    """Copy a CPU tensor to the device with an explicit dim_order."""
    # The lazy device init must see a plain .to() before a layout-carrying one.
    torch.zeros(1, dtype=t.dtype).to(DEVICE)
    stl = SpyreTensorLayout(
        list(t.size()), list(t.stride()), t.dtype, [int(v) for v in order.split(",")]
    )
    return t.to(DEVICE, device_layout=stl)


def observed_layout(out):
    """Classify the layout the compiler gave the output tensor.

    The device shape of a rank-3 [B, M, N] tensor is a permutation of the batch
    and row axes plus a 64-wide stick, so whichever of B / M appears at the
    lower index is the slower-varying one.  Returns "?" when B == M, because
    the two cannot then be told apart by size, and a wrong label is worse than
    an admitted unknown.
    """
    layout = getattr(out, "device_tensor_layout", lambda: None)()
    dims = list(getattr(layout, "device_size", []) or [])
    logical = list(out.shape)
    if len(logical) != 3 or logical[0] == logical[1] or not dims:
        return "?"
    try:
        return "batch" if dims.index(logical[0]) < dims.index(logical[1]) else "row"
    except ValueError:
        return "?"


def device_us(prof):
    """Device time of the compute kernels in one profiled region, in us.

    Transfers and buffer initialisation are excluded by name: they scale with
    the host round trip, not with the kernel under test.
    """
    total = 0.0
    for ev in prof.key_averages():
        name = ev.key or ""
        dt = getattr(ev, "self_device_time_total", 0) or 0
        if dt > 0 and "Memcpy" not in name and "Memset" not in name:
            total += dt
    return total


def main():
    x = place(torch.rand(B, M, K, dtype=torch.float16), ORDER_A)
    y = place(torch.rand(B, K, N, dtype=torch.float16), ORDER_B)
    fn = torch.compile(lambda a, b: torch.bmm(a, b))

    out = fn(x, y)
    out.cpu()  # device execution is asynchronous; force completion
    layout_c = observed_layout(out)

    times = []
    for _ in range(REPS):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]
        ) as prof:
            fn(x, y).cpu()
        us = device_us(prof)
        if us > 0:
            times.append(us)

    if not times:
        print(
            "FAILED reason=no_device_time  "
            "(the profiler reported nothing; check the kineto-spyre build)"
        )
        return 1

    print(
        f"SUMMARY batch={B} m={M} k={K} n={N} "
        f"layout_a={NAME.get(ORDER_A, ORDER_A)} "
        f"layout_b={NAME.get(ORDER_B, ORDER_B)} layout_c={layout_c} "
        f"kernel_us={statistics.median(times):.1f} "
        f"min_us={min(times):.1f} max_us={max(times):.1f} reps={len(times)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PYEOF

{
  echo "# layout cube: A x B x C for torch.bmm, all cells on this build"
  echo "# M=$M K=$K N=$N batches='$BATCHES' reps=$REPS cores=$CORES"
} | tee "$OUT"

for b in $BATCHES; do
  for la in "0,1,2" "1,0,2"; do
    for lb in "0,1,2" "1,0,2"; do
      # "" leaves the output at the compiler default; "output" asks for the
      # preferred order on C only, leaving the explicitly placed inputs alone.
      for pref in "" "output"; do
        SENCORES="$CORES" \
        SPYRE_MATMUL_PREFERRED_LAYOUT="$pref" \
        CELL_B="$b" CELL_M="$M" CELL_K="$K" CELL_N="$N" CELL_REPS="$REPS" \
        CELL_A="$la" CELL_B_ORDER="$lb" \
          python3 "$CELL_PY" 2>&1 | tee -a "$OUT"
      done
    done
  done
done

echo
echo "done -> $OUT"
echo
echo "Collect the cube with:"
echo "  grep '^SUMMARY' $OUT"
echo
echo "Check first that layout_c differs between the two rows that share a"
echo "layout_a and layout_b.  If it does not, the preference was not honoured"
echo "and those two rows are the same experiment, not a C comparison."
