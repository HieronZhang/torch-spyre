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

Knobs: BENCH_OP, BENCH_ROWS, BENCH_COLS, BENCH_WARMUP. BENCH_OP is any of the
bench_ops/bench_bandwidth ops: neg copy gelu relu sigmoid exp | mul add | add3 add4 |
read sumrow sumall amax mean | bcast mulbcast | write.

Examples:
    # one op (prints the table + a parseable SUMMARY line with kernel_us / memset_us)
    BENCH_OP=neg BENCH_COLS=1024 python examples/profile_ops.py
    # full golden re-sweep (rebuild the model from kernel time):
    bash examples/run_profile_sweep.sh
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


def _sum_all(*ts):
    acc = ts[0]
    for t in ts[1:]:
        acc = acc + t
    return acc


# Same ops as bench_ops/bench_bandwidth so the GOLDEN kernel times map 1:1.
_UNARY = {  # 1 read + 1 write (gelu/relu/sigmoid/exp also probe arithmetic-free)
    "neg": lambda x: -x,  # cleanest balanced 1R+1W (no constant)
    "copy": lambda x: x + 1.0,  # scalar 1.0 is a cached broadcast -> still 1R+1W
    "gelu": F.gelu,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "exp": torch.exp,
}
_BINARY = {"mul": lambda a, b: a * b, "add": lambda a, b: a + b}  # 2R + 1W
_NARY = {"add3": 3, "add4": 4}  # n inputs summed (intermediates staged in LX)
_REDUCE = {  # read-dominated; sumall reduces to a scalar -> ring combine
    "read": lambda x: x.sum(dim=-1),
    "sumrow": lambda x: x.sum(dim=-1),
    "sumall": lambda x: x.sum(),
    "amax": lambda x: x.amax(dim=-1),
    "mean": lambda x: x.mean(dim=-1),
}
_BCAST = {"bcast": lambda a, b: a + b, "mulbcast": lambda a, b: a * b}  # b = [1, COLS]


def make_workload():
    if OP in _UNARY:
        return torch.compile(_UNARY[OP]), (_rand(ROWS, COLS),)
    if OP in _BINARY:
        return torch.compile(_BINARY[OP]), (_rand(ROWS, COLS), _rand(ROWS, COLS))
    if OP in _NARY:
        xs = tuple(_rand(ROWS, COLS) for _ in range(_NARY[OP]))
        return torch.compile(_sum_all), xs
    if OP in _REDUCE:
        return torch.compile(_REDUCE[OP]), (_rand(ROWS, COLS),)
    if OP in _BCAST:
        return torch.compile(_BCAST[OP]), (_rand(ROWS, COLS), _rand(1, COLS))
    if OP == "write":  # write-only: both inputs broadcast -> cached
        return torch.compile(lambda b, c: b + c), (_rand(1, COLS), _rand(ROWS, 1))
    known = (
        list(_UNARY) + list(_BINARY) + list(_NARY) + list(_REDUCE) + list(_BCAST)
        + ["write"]
    )
    raise SystemExit(f"unknown BENCH_OP={OP!r} (use {known})")


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
    ka = prof.key_averages()
    print(ka.table(sort_by="cuda_time_total", row_limit=20).replace("CUDA", "AIU"))

    # Parseable one-liner for a size sweep -> re-fit fill + BW on the GOLDEN kernel
    # time, and track the (non-deterministic) Memset overhead separately.
    kernel = memset = other = 0.0
    for ev in ka:
        us = getattr(ev, "self_device_time_total", 0) or getattr(
            ev, "self_cuda_time_total", 0
        )
        if not us or us <= 0:
            continue
        if "sdsc_fused" in ev.key:
            kernel += us
        elif "Memset" in ev.key:
            memset += us
        else:
            other += us
    elems = ROWS * COLS
    print(
        f"SUMMARY op={OP} rows={ROWS} cols={COLS} elems={elems} "
        f"kernel_us={kernel:.3f} memset_us={memset:.3f} other_dev_us={other:.3f} "
        f"total_dev_us={kernel + memset + other:.3f}"
    )


if __name__ == "__main__":
    main()
