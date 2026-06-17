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

"""Cross-check our SPYRE_PROFILE_SYNC min-of-N against the PyTorch profiler.

The torch.profiler ``PrivateUse1`` activity (needs the kineto-spyre wheel -- see
docs/source/user_guide/profiling/pytorch_profiler.md) reports a **"Self SPYRE"**
column = the TRUE per-kernel device time. Two uses:

1. **Validate our timer.** Our SPYRE_PROFILE_SYNC min brackets the launch+sync, so it
   = device time + ~7 us host residue. Run the SAME op/size here and in bench_ops.py:
   ``our_min  ~=  Self_SPYRE(sdsc_fused_*)  +  ~7us`` confirms the measurement.
2. **Decompose the ~20 us fixed term.** The profiler surfaces a separate
   ``Memset (Device)`` event (~12.5 us on a tiny add). If it stays ~constant across
   sizes, a big chunk of our "fixed" is a real DEVICE memset, not host overhead.

NOTE: this is a TIME + memory-allocation profiler -- it has NO DRAM bandwidth / read-
vs-write / bus-utilization counters, so it CANNOT explain why read+write halves
bandwidth (that needs aiu-smi). It only gives cleaner device time.

Knobs: BENCH_OP (gelu|copy|neg|read|write), BENCH_ROWS, BENCH_COLS, BENCH_WARMUP.

Examples:
    # validate the timer at a real size (compare to bench_ops.py min for the same op)
    BENCH_OP=gelu BENCH_COLS=1024 python examples/profile_ops.py
    # is the device Memset fixed across sizes? sweep and watch its Self SPYRE us
    for n in 1024 16384; do \
      BENCH_OP=copy BENCH_COLS=$n python examples/profile_ops.py; done
"""

import os

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

DEVICE = torch.device("spyre")
OP = os.environ.get("BENCH_OP", "gelu")
ROWS = int(os.environ.get("BENCH_ROWS", "512"))
COLS = int(os.environ.get("BENCH_COLS", "16384"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "5"))

torch.manual_seed(0xAFFE)


def _rand(*shape):
    return torch.rand(*shape, dtype=torch.float16).to(DEVICE)


def make_workload():
    """Same ops as bench_ops/bench_bandwidth so the numbers line up."""
    if OP == "gelu":
        return torch.compile(F.gelu), (_rand(ROWS, COLS),)
    if OP == "copy":  # 1R+1W (scalar 1.0 is a cached broadcast -> free)
        return torch.compile(lambda x: x + 1.0), (_rand(ROWS, COLS),)
    if OP == "neg":  # genuine 1R+1W, no constant
        return torch.compile(lambda x: -x), (_rand(ROWS, COLS),)
    if OP == "read":  # read-only reduction
        return torch.compile(lambda x: x.sum(dim=-1)), (_rand(ROWS, COLS),)
    if OP == "write":  # write-only (both inputs broadcast -> cached)
        return torch.compile(lambda b, c: b + c), (_rand(1, COLS), _rand(ROWS, 1))
    raise SystemExit(f"unknown BENCH_OP={OP!r} (use gelu|copy|neg|read|write)")


def main():
    compiled, args = make_workload()
    for _ in range(WARMUP):  # compile + warm the kernel
        compiled(*args).cpu()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        compiled(*args).cpu()

    print(f"== {OP}[{ROWS}x{COLS}] -- compare 'Self SPYRE' to bench_ops.py min ==")
    print(
        prof.key_averages()
        .table(sort_by="cuda_time_total", row_limit=20)
        .replace("CUDA", "AIU")
    )


if __name__ == "__main__":
    main()
